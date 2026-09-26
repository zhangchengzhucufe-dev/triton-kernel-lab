"""Flash-decoding: attention for the decode phase (q length = 1), for long KV caches.

Problem: q has only one row, so if the grid is split along the seq dim, there are
only batch*heads programs and most SMs sit idle. Same idea as ex. 19's split-K:
split KV's seq dim, each program computes a partial result for one chunk, then
a second kernel merges them.

The merge math is just online softmax again: each chunk yields an (acc, m, l)
triple; on merge, m takes the max, and acc and l are both scaled by exp(m_i - m_max)
and summed. vLLM / FlashInfer's paged attention is this scheme plus a block table.

The intermediate result (SPLITS, Z*H, D) is tiny, so keeping it in HBM is fine.
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
    # Pitfall: q's plane is only D elements, K/V's plane is N*D; the two strides
    # must not be mixed up (I got this wrong in my first version, errors hit ~1e1)
    q_base = off_hz.to(tl.int64) * HEAD_DIM
    base = off_hz.to(tl.int64) * stride_kv_plane

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_base + offs_d * stride_d).to(tl.float32)   # only 1 row of q

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
        # (BLOCK_N,) scores: a single q row makes this a matrix-vector product
        qk = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * sm_scale
        # Masked rows have k all zeros, so qk comes out 0 instead of -inf;
        # without masking, these fake rows would pollute the denominator l
        # with weight exp(-m_new)
        qk = tl.where(mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new) * 1.44269504)
        v = tl.load(V + base + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_d,
                    mask=mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    # Write the triple out for the second kernel to merge. Empty splits
    # (lo >= SEQ_LEN) must be written too, otherwise the merge reads garbage
    # — I missed this in my first version
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
    # (My first version multiplied acc by l as well, applying the weight twice;
    # the result was simply wrong)
    e = tl.exp(m - m_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e * l, axis=0)

    offs_d = tl.arange(0, HEAD_DIM)
    acc_ptrs = ACC + offs_s[:, None] * (tl.num_programs(0) * HEAD_DIM) + off_hz * HEAD_DIM + offs_d[None, :]
    acc = tl.load(acc_ptrs, mask=mask[:, None], other=0.0)
    out = tl.sum(acc * e[:, None], axis=0) / denom
    tl.store(OUT + off_hz * HEAD_DIM + offs_d, out)


def flash_decode(q, k, v, sm_scale, num_splits=8, block_n=128):
    """q: (Z, H, 1, D)  k/v: (Z, H, N, D), returns (Z, H, 1, D)"""
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
    Z, H, N, D = 4, 32, 32768, 128     # 32K-context KV cache
    q = torch.randn((Z, H, 1, D), device='cuda', dtype=torch.float16) * 0.5
    k = torch.randn((Z, H, N, D), device='cuda', dtype=torch.float16) * 0.5
    v = torch.randn((Z, H, N, D), device='cuda', dtype=torch.float16) * 0.5

    o = flash_decode(q, k, v, D ** -0.5)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"max error = {(o.float() - ref.float()).abs().max().item():.2e}")
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-2, rtol=0)
    print("✅ flash-decoding correctness passed")

    t_split = bench(lambda: flash_decode(q, k, v, D ** -0.5))
    t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"flash-decoding {t_split:.2f} ms   SDPA {t_sdpa:.2f} ms")
