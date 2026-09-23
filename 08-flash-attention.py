"""Flash Attention v2 前向（tutorial 06 的精简版，无 dropout / causal / fp8）。

核心是 online softmax：分块扫 K，每块都会更新 running max，旧的
acc 和 l 都得按 exp(m_old - m_new) 重缩放一遍，最后统一除 l。
中间的 S 和 P 矩阵全程不落 HBM，这就是它省显存的原因。
qk_scale 里乘的 1.44269504 是把 exp 换成 exp2（硬件指令更快）的换底系数。
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
    # STAGE 决定本 program 处理 causal 矩阵的哪一段（本简化版不做 causal，
    # 但保留 tutorial 的 stage 结构，方便对照原版阅读）
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N)，fp32 累加

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]          # 平移到 <= 0，防溢出
        p = tl.math.exp2(qk)                        # 用 exp2：硬件指令更快
        alpha = tl.math.exp2(m_i - m_ij)            # 在线 softmax 的重缩放因子
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]                  # 修正之前的 PV 部分和
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)                     # 累加 P @ V
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
    start_m = tl.program_id(0)          # Q 的第几个 block
    off_hz = tl.program_id(1)           # (batch * num_heads) 合并成一维
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # 注意：kernel 内不能定义 Python lambda/闭包，块指针必须显式构造。
    # K 用 (HEAD_DIM, N_CTX) 的转置视图，使 tl.dot(q, k) 直接成立。
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),   # K 转置视图
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

    qk_scale = sm_scale * 1.44269504    # 把 exp 换成 exp2 的换底系数
    q = tl.load(Q_block_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                    start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 4,
                                    offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))
    # 存 logsumexp 供反向使用（本文件只做前向，保留该惯例）
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
    print(f"最大误差 = {(o_triton - o_ref).abs().max().item():.2e}")
    torch.testing.assert_close(o_triton, o_ref, atol=2e-2, rtol=0)
    print("✅ flash attention 前向正确性通过")
