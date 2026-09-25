"""Causal flash attention (causal variant of file 03; read that one first).

Instead of masking everything with tl.where, split the K loop in two: the
block-column range left of the diagonal is fully valid and computed directly;
only the diagonal block needs elementwise masking. The range right of the
diagonal is empty, so no FLOPs are spent there.

Fill the mask with -1e6, not -inf: qk hasn't been reduced by m_ij yet, and
-inf - (-inf) produces NaN.

A bonus of the causal split: the diagonal block's mask also covers
out-of-bounds columns when seqlen isn't a multiple of BLOCK_N — any col past
N_CTX is automatically > any valid row, so the causal predicate kills it for
free. The loads still need boundary_check so the *memory access* stays legal.

The biggest pitfall is noted at _attn_inner's return: tl.advance advances
a local variable, so after splitting into two loops, the second loop gets
the original pointers — re-computing the earlier columns. The error is only
about 0.1, and I initially chased it as a precision issue for a long time.
Inside a jit function, pointers must either be returned to the caller or
you simply shouldn't split the loop.

The other bug this file carried for a long time: one shared (stride_zh,
stride_m, stride_d) triple for Q/K/V/Out. It only works when all four have
identical layouts; file 03's write-up has the full story (short version: it
doesn't just give wrong values for mixed layouts, it writes out of bounds).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_inner(acc, l_i, m_i, q, K_ptr, V_ptr,
                qk_scale, lo, hi, q_block_start, N_CTX,
                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                APPLY_CAUSAL: tl.constexpr):
    for start_n in tl.range(lo, hi, BLOCK_N):
        k = tl.load(K_ptr, boundary_check=(1,))
        qk = tl.dot(q, k)     # (BLOCK_M, BLOCK_N) fp32 accumulator
        if APPLY_CAUSAL:
            # global row index vs global col index: visible only when row >= col.
            # cols past N_CTX also die here — causal implies col > row for them
            rows = q_block_start + tl.arange(0, BLOCK_M)
            cols = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(rows[:, None] >= cols[None, :], qk, -1.0e6)

        m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)          # online softmax rescaling
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_ptr, boundary_check=(0,))
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
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # per-tensor offsets — see file 03's docstring for the out-of-bounds-write
    # horror show the shared-offset version could produce
    q_off = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_off = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_off = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_off = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    q_ptr = tl.make_block_ptr(base=Q + q_off, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
                              offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    k_ptr = tl.make_block_ptr(base=K + k_off, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
                              offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    v_ptr = tl.make_block_ptr(base=V + v_off, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
                              offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    o_ptr = tl.make_block_ptr(base=Out + o_off, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_ok),
                              offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    q = tl.load(q_ptr, boundary_check=(0,))
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.44269504    # exp → exp2 base conversion

    q_block_start = start_m * BLOCK_M
    # Segment 1: column range [0, q_block_start), fully left of the diagonal → no mask
    acc, l_i, m_i, k_ptr, v_ptr = _attn_inner(acc, l_i, m_i, q, k_ptr, v_ptr, qk_scale,
                                              0, q_block_start, q_block_start, N_CTX,
                                              BLOCK_M, HEAD_DIM, BLOCK_N, APPLY_CAUSAL=False)
    # Segment 2: diagonal block [q_block_start, q_block_start + BLOCK_M) → elementwise mask
    acc, l_i, m_i, k_ptr, v_ptr = _attn_inner(acc, l_i, m_i, q, k_ptr, v_ptr, qk_scale,
                                              q_block_start, q_block_start + BLOCK_M, q_block_start, N_CTX,
                                              BLOCK_M, HEAD_DIM, BLOCK_N, APPLY_CAUSAL=True)

    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


def causal_attention(q, k, v, sm_scale):
    Z, H, N_CTX, D = q.shape
    o = torch.empty_like(q)
    # 128/64: the sweep (autotune-results.csv) says causal doesn't care much —
    # 128/128/w8 edges it by a few % on some runs, default wins others. keeping it
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _causal_attn_fwd[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
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


def _check_case(name, q, k, v):
    sm_scale = q.shape[-1] ** -0.5
    o = causal_attention(q, k, v, sm_scale)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-2, rtol=0)
    print(f"✅ {name}")


if __name__ == "__main__":
    torch.manual_seed(0)
    D = 64
    mk = lambda z, h, s: torch.randn((z, h, s, D), device='cuda', dtype=torch.float16) * 0.5

    _check_case("2x4x2048x64 aligned (the old case)", mk(2, 4, 2048), mk(2, 4, 2048), mk(2, 4, 2048))
    # odd seqlen: diagonal segment runs past N_CTX, causal mask covers it
    _check_case("1x2x1000x64 odd seqlen", mk(1, 2, 1000), mk(1, 2, 1000), mk(1, 2, 1000))
    # seqlen aligned to BLOCK_M but not BLOCK_N, and vice versa
    _check_case("1x1x1280x64 (1280 = 10*128, 20*64)", mk(1, 1, 1280), mk(1, 1, 1280), mk(1, 1, 1280))
    _check_case("1x1x1032x64 (1032 % 128 = 8)", mk(1, 1, 1032), mk(1, 1, 1032), mk(1, 1, 1032))
    # strided: sliced rows, and q/k/v with different layouts
    big = mk(1, 2, 2048)
    _check_case("sliced rows [::2]",
                big[:, :, ::2, :], big[:, :, ::2, :].clone(), big[:, :, ::2, :].clone())
    _check_case("q sliced, k/v contiguous",
                big[:, :, ::2, :], torch.randn_like(big[:, :, ::2, :]),
                torch.randn_like(big[:, :, ::2, :]))

    print("✅ causal flash attention correctness passed (all cases)")
