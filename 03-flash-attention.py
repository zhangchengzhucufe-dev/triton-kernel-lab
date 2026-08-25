"""Flash Attention v2 forward (a slimmed-down version of tutorial 06, no dropout / causal / fp8).

The core is online softmax: scan K in blocks, each block updates the running max, and
both the old acc and l must be rescaled by exp(m_old - m_new), divided by l at the end.
The intermediate S and P matrices never touch HBM — that's what saves memory.
The 1.44269504 multiplied into qk_scale is the log2(e) base-change factor that swaps
exp for exp2 (a faster hardware instruction).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr,
        start_m, qk_scale,
        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        offs_m, offs_n, N_CTX,
):
    # STAGE decides which segment of the causal matrix this program handles (this
    # simplified version doesn't do causal, but keeps tutorial's stage structure
    # for easier comparison with the original)
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N), fp32 accumulation

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]          # shift to <= 0 to avoid overflow
        p = tl.math.exp2(qk)                        # exp2: faster hardware instruction
        alpha = tl.math.exp2(m_i - m_ij)            # online softmax rescaling factor
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]                  # correct the previous PV partial sum
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)                     # accumulate P @ V
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
        Q, K, V, sm_scale, M, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)          # which block of Q
    off_hz = tl.program_id(1)           # (batch * num_heads) flattened to 1D
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Note: Python lambdas/closures can't be defined inside kernels; block pointers
    # must be constructed explicitly. K uses a transposed (HEAD_DIM, N_CTX) view so
    # tl.dot(q, k) works directly.
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),   # K transposed view
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504    # log2(e) base-change factor to swap exp for exp2
    q = tl.load(Q_block_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                    start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 4,
                                    offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))
    # Store logsumexp for the backward pass (this file only does forward; kept as a convention)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))


def attention(q, k, v, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        num_warps=8, num_stages=3,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    Z, H, N_CTX, HEAD_DIM = 2, 4, 1024, 64
    dtype = torch.float16
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    sm_scale = 0.5

    o_triton = attention(q, k, v, sm_scale)
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    print(f"max error = {(o_triton - o_ref).abs().max().item():.2e}")
    torch.testing.assert_close(o_triton, o_ref, atol=2e-2, rtol=0)
    print("✅ flash attention forward correctness passed")
