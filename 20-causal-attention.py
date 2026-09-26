"""Causal flash attention (causal variant of ex. 08; read 08 first).

Instead of masking everything with tl.where, split the K loop in two: the
block-column range left of the diagonal is fully valid and computed directly;
only the diagonal block needs elementwise masking. The range right of the
diagonal is empty, so no FLOPs are spent there.

Fill the mask with -1e6, not -inf: qk hasn't been reduced by m_ij yet, and
-inf - (-inf) produces NaN.

The biggest pitfall is noted at _attn_inner's return: tl.advance advances
a local variable, so after splitting into two loops, the second loop gets
the original pointers — re-computing the earlier columns. The error is only
about 0.1, and I initially chased it as a precision issue for a long time.
Inside a jit function, pointers must either be returned to the caller or
you simply shouldn't split the loop.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_inner(acc, l_i, m_i, q, K_ptr, V_ptr,
                qk_scale, lo, hi, q_block_start,
                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                APPLY_CAUSAL: tl.constexpr):
    for start_n in tl.range(lo, hi, BLOCK_N):
        k = tl.load(K_ptr)
        qk = tl.dot(q, k)     # (BLOCK_M, BLOCK_N) fp32 accumulator
        if APPLY_CAUSAL:
            # global row index vs global col index: visible only when row >= col
            rows = q_block_start + tl.arange(0, BLOCK_M)
            cols = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(rows[:, None] >= cols[None, :], qk, -1.0e6)

        m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)          # online softmax rescaling
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
        K_ptr = tl.advance(K_ptr, (0, BLOCK_N))
        V_ptr = tl.advance(V_ptr, (BLOCK_N, 0))
    # Pitfall: K_ptr/V_ptr are local variables of this function; the advanced
    # pointers must be returned to the caller, otherwise the next loop segment
    # restarts from the original offsets and re-computes the already-done columns!
    return acc, l_i, m_i, K_ptr, V_ptr


@triton.jit
def _causal_attn_fwd(
        Q, K, V, sm_scale, Out,
        stride_zh, stride_m, stride_d,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # Q/K/V layouts are identical in this example: (Z, H, N_CTX, D), sharing the (z,h) offset
    base = off_hz.to(tl.int64) * stride_zh

    q_ptr = tl.make_block_ptr(base=Q + base, shape=(N_CTX, HEAD_DIM), strides=(stride_m, stride_d),
                              offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    k_ptr = tl.make_block_ptr(base=K + base, shape=(HEAD_DIM, N_CTX), strides=(stride_d, stride_m),
                              offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    v_ptr = tl.make_block_ptr(base=V + base, shape=(N_CTX, HEAD_DIM), strides=(stride_m, stride_d),
                              offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    o_ptr = tl.make_block_ptr(base=Out + base, shape=(N_CTX, HEAD_DIM), strides=(stride_m, stride_d),
                              offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    q = tl.load(q_ptr)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.44269504    # exp → exp2 base conversion

    q_block_start = start_m * BLOCK_M
    # Segment 1: column range [0, q_block_start), fully left of the diagonal → no mask
    acc, l_i, m_i, k_ptr, v_ptr = _attn_inner(acc, l_i, m_i, q, k_ptr, v_ptr, qk_scale,
                                              0, q_block_start, q_block_start,
                                              BLOCK_M, HEAD_DIM, BLOCK_N, APPLY_CAUSAL=False)
    # Segment 2: diagonal block [q_block_start, q_block_start + BLOCK_M) → elementwise mask
    acc, l_i, m_i, k_ptr, v_ptr = _attn_inner(acc, l_i, m_i, q, k_ptr, v_ptr, qk_scale,
                                              q_block_start, q_block_start + BLOCK_M, q_block_start,
                                              BLOCK_M, HEAD_DIM, BLOCK_N, APPLY_CAUSAL=True)

    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Out.dtype.element_ty))


def causal_attention(q, k, v, sm_scale):
    Z, H, N_CTX, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _causal_attn_fwd[grid](
        q, k, v, sm_scale, o,
        q.stride(1), q.stride(2), q.stride(3),   # (z,h) dim, row dim, col dim
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        num_warps=8, num_stages=3,
    )
    return o


def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    Z, H, N_CTX, D = 2, 4, 2048, 64
    q = torch.randn((Z, H, N_CTX, D), device='cuda', dtype=torch.float16) * 0.5
    k = torch.randn((Z, H, N_CTX, D), device='cuda', dtype=torch.float16) * 0.5
    v = torch.randn((Z, H, N_CTX, D), device='cuda', dtype=torch.float16) * 0.5
    sm_scale = D ** -0.5

    o = causal_attention(q, k, v, sm_scale)
    # Reference implementation: SDPA with is_causal
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"max error = {(o.float() - ref.float()).abs().max().item():.2e}")
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-2, rtol=0)
    print("✅ causal flash attention correctness passed")
