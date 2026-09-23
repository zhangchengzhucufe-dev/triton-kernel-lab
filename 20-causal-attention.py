"""causal 版 flash attention（08 号的非 causal 改 causal，08 先看）。

mask 不用 tl.where 全程盖，把 K 循环拆两段：对角线左边的整块合法直接
算，对角块才逐元素 mask，右边的循环区间是空的，一个 FLOP 不花。

mask 填 -1e6 别填 -inf：qk 还没减 m_ij，-inf - (-inf) 会出 NaN。

最大的坑写在 _attn_inner 的 return 那里：tl.advance 推进的是局部变量，
拆成两段循环后第二段拿到的是原始指针，等于把前面的列重复算了一遍。
误差只有 0.1 左右，一开始当精度问题查了半天。jit 函数里的指针要么
随返回值传出去，要么干脆别拆段。
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
        qk = tl.dot(q, k)     # (BLOCK_M, BLOCK_N) fp32 累加
        if APPLY_CAUSAL:
            # 全局行号 vs 全局列号：行 >= 列 才可见
            rows = q_block_start + tl.arange(0, BLOCK_M)
            cols = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(rows[:, None] >= cols[None, :], qk, -1.0e6)

        m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)          # 在线 softmax 的重缩放
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
        K_ptr = tl.advance(K_ptr, (0, BLOCK_N))
        V_ptr = tl.advance(V_ptr, (BLOCK_N, 0))
    # 坑：K_ptr/V_ptr 是本函数的局部变量，必须把推进后的指针返回给调用者，
    # 否则下一段循环会从原始偏移重新开始，把已算过的列重复算一遍！
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
    # 本例让 Q/K/V 布局完全一致：(Z, H, N_CTX, D)，共享 (z,h) 偏移
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
    qk_scale = sm_scale * 1.44269504    # exp → exp2 换底

    q_block_start = start_m * BLOCK_M
    # 段 1：列区间 [0, q_block_start)，全部在对角线左侧 → 无 mask
    acc, l_i, m_i, k_ptr, v_ptr = _attn_inner(acc, l_i, m_i, q, k_ptr, v_ptr, qk_scale,
                                              0, q_block_start, q_block_start,
                                              BLOCK_M, HEAD_DIM, BLOCK_N, APPLY_CAUSAL=False)
    # 段 2：对角块 [q_block_start, q_block_start + BLOCK_M) → 逐元素 mask
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
        q.stride(1), q.stride(2), q.stride(3),   # (z,h) 维、行维、列维
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
    # SDPA 的 is_causal 参考实现
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"最大误差 = {(o.float() - ref.float()).abs().max().item():.2e}")
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-2, rtol=0)
    print("✅ causal flash attention 正确性通过")
