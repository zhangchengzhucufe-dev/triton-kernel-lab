"""flash-decoding：decode 阶段（q 长度=1）的注意力，KV cache 很长时用。

问题：q 只有一行，如果按 seq 维切 grid，program 数只有 batch*heads，
SM 大片空转。解法和 19 号 split-K 一个思想：把 KV 的 seq 维切开，
每个 program 算一段的部分结果，再用第二个 kernel 归并。

归并的数学还是 online softmax 那一套：每段带出 (acc, m, l) 三元组，
合并时 m 取 max，acc 和 l 都按 exp(m_i - m_max) 缩放后相加。
vLLM / FlashInfer 的 paged attention 就是这套再叠一个 block table。

中间结果 (SPLITS, Z*H, D) 很小，放 HBM 无所谓。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_split_kernel(
        Q, K, V, ACC, M_OUT, L_OUT,
        sm_scale,
        stride_kv_plane, stride_seq, stride_d,
        SEQ_LEN, CHUNK_SIZE,
        HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    split = tl.program_id(0)
    off_hz = tl.program_id(1)
    # 坑：q 的平面只有 D 这么大，K/V 的平面是 N*D，两个 stride 不能混用
    # （我第一版就用错了，误差直接 1e1 量级）
    q_base = off_hz.to(tl.int64) * HEAD_DIM
    base = off_hz.to(tl.int64) * stride_kv_plane

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_base + offs_d * stride_d).to(tl.float32)   # 只有 1 行 q

    lo = split * CHUNK_SIZE
    hi = tl.minimum(lo + CHUNK_SIZE, SEQ_LEN)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < hi
        k = tl.load(K + base + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_d,
                    mask=mask[:, None], other=0.0)
        # (BLOCK_N,) 的分数：单 q 行就是一次矩阵-向量乘
        qk = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * sm_scale
        # mask 掉的行 k 全是 0，qk 算出来是 0 而不是 -inf，
        # 不屏蔽的话这些假行会以 exp(-m_new) 的权重污染分母 l
        qk = tl.where(mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new) * 1.44269504)
        v = tl.load(V + base + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_d,
                    mask=mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    # 三元组落盘，等第二个 kernel 归并。空分片（lo >= SEQ_LEN）也要写，
    # 否则归并时读到垃圾 —— 我第一版就漏了这个
    out_of_range = lo >= SEQ_LEN
    m_safe = tl.where(out_of_range, float("-inf"), m_i)
    tl.store(ACC + split.to(tl.int64) * tl.num_programs(1) * HEAD_DIM + off_hz * HEAD_DIM + offs_d,
             tl.where(out_of_range, 0.0, acc))
    tl.store(M_OUT + split.to(tl.int64) * tl.num_programs(1) + off_hz, m_safe)
    tl.store(L_OUT + split.to(tl.int64) * tl.num_programs(1) + off_hz,
             tl.where(out_of_range, 0.0, l_i))


@triton.jit
def _decode_combine_kernel(
        ACC, M_IN, L_IN, OUT,
        num_splits,
        HEAD_DIM: tl.constexpr, BLOCK_S: tl.constexpr,
):
    off_hz = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    mask = offs_s < num_splits

    m = tl.load(M_IN + offs_s * tl.num_programs(0) + off_hz, mask=mask, other=float("-inf"))
    l = tl.load(L_IN + offs_s * tl.num_programs(0) + off_hz, mask=mask, other=0.0)
    m_max = tl.max(m, axis=0)

    # out = Σ_i acc_i·exp(m_i-m_max) / Σ_i l_i·exp(m_i-m_max)
    # （我第一版把 acc 也乘了 l，等于权重多乘了一次，结果直接错）
    e = tl.exp(m - m_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e * l, axis=0)

    offs_d = tl.arange(0, HEAD_DIM)
    acc_ptrs = ACC + offs_s[:, None] * (tl.num_programs(0) * HEAD_DIM) + off_hz * HEAD_DIM + offs_d[None, :]
    acc = tl.load(acc_ptrs, mask=mask[:, None], other=0.0)
    out = tl.sum(acc * e[:, None], axis=0) / denom
    tl.store(OUT + off_hz * HEAD_DIM + offs_d, out)


def flash_decode(q, k, v, sm_scale, num_splits=8, block_n=128):
    """q: (Z, H, 1, D)  k/v: (Z, H, N, D)，返回 (Z, H, 1, D)"""
    Z, H, _, D = q.shape
    N = k.shape[2]
    q2 = q.reshape(Z * H, D).contiguous()
    k2 = k.reshape(Z * H, N, D).contiguous()
    v2 = v.reshape(Z * H, N, D).contiguous()

    acc = torch.empty((num_splits, Z * H, D), device=q.device, dtype=torch.float32)
    m = torch.empty((num_splits, Z * H), device=q.device, dtype=torch.float32)
    l = torch.empty((num_splits, Z * H), device=q.device, dtype=torch.float32)
    chunk = triton.cdiv(N, num_splits)
    chunk = triton.next_power_of_2(max(chunk, block_n))

    _decode_split_kernel[(num_splits, Z * H)](
        q2, k2, v2, acc, m, l, sm_scale,
        k2.stride(0), k2.stride(1), k2.stride(2),
        N, chunk, HEAD_DIM=D, BLOCK_N=block_n,
    )
    out = torch.empty((Z * H, D), device=q.device, dtype=torch.float32)
    _decode_combine_kernel[(Z * H,)](acc, m, l, out, num_splits,
                                     HEAD_DIM=D, BLOCK_S=triton.next_power_of_2(num_splits))
    return out.reshape(Z, H, 1, D).to(q.dtype)


def bench(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    Z, H, N, D = 4, 32, 32768, 128     # 32K 上下文的 KV cache
    q = torch.randn((Z, H, 1, D), device='cuda', dtype=torch.float16) * 0.5
    k = torch.randn((Z, H, N, D), device='cuda', dtype=torch.float16) * 0.5
    v = torch.randn((Z, H, N, D), device='cuda', dtype=torch.float16) * 0.5

    o = flash_decode(q, k, v, D ** -0.5)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"最大误差 = {(o.float() - ref.float()).abs().max().item():.2e}")
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-2, rtol=0)
    print("✅ flash-decoding 正确性通过")

    t_split = bench(lambda: flash_decode(q, k, v, D ** -0.5))
    t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"flash-decoding {t_split:.2f} ms   SDPA {t_sdpa:.2f} ms")
