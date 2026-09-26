"""Paged KV-cache decode attention — the trick that made vLLM possible.

Decode is memory-bound: for every token you generate, the whole KV cache gets
streamed through the GPU just to be multiplied by one query row. So the cache
layout matters more than FLOPs. A naive per-sequence cache needs one
contiguous allocation that keeps growing with the sequence, so you either
over-allocate or fragment badly. Paged instead chops K/V into fixed-size
pages and keeps a per-sequence block table (virtual memory, basically).
Allocation becomes "grab a free page", and waste drops to at most one partial
page per sequence.

File 21 attacked the same decode problem from a different angle (splitting
the KV dimension across SMs); this one attacks the cache layout instead.

This is a GQA example: every kv head serves H/KVH query heads. Each program
handles one (sequence, query head), walks its block table page by page with
online softmax, and never materializes the full attention row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode(
        Q, K_CACHE, V_CACHE, BLOCK_TABLE, SEQ_LENS, OUT,
        sm_scale,
        stride_qb, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_vb, stride_vt, stride_vh,
        H: tl.constexpr, KVH: tl.constexpr,
        HEAD_DIM: tl.constexpr, PAGE_SIZE: tl.constexpr, MAX_PAGES: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // (H // KVH)    # GQA: consecutive query heads share a kv head

    seq_len = tl.load(SEQ_LENS + seq)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, PAGE_SIZE)

    # one query token per sequence during decode, so q is just a (HEAD_DIM,) vector
    q = tl.load(Q + seq * stride_qb + head * stride_qh + offs_d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    for page_start in range(0, seq_len, PAGE_SIZE):
        # the block table is the whole point: where this page lives is a
        # runtime lookup, not a fixed stride
        page = tl.load(BLOCK_TABLE + seq * MAX_PAGES + page_start // PAGE_SIZE)
        # the last page is usually only half full
        valid = page_start + offs_t < seq_len
        kv_off = page * stride_kb + kv_head * stride_kh
        k = tl.load(K_CACHE + kv_off + offs_t[:, None] * stride_kt + offs_d[None, :],
                    mask=valid[:, None], other=0.0)
        v = tl.load(V_CACHE + kv_off + offs_t[:, None] * stride_vt + offs_d[None, :],
                    mask=valid[:, None], other=0.0)

        # no tl.dot here: the Q side has one row, tensor cores can't help —
        # this is a gather problem, not a matmul problem
        qk = tl.sum(q[None, :] * k.to(tl.float32), 1) * qk_scale
        qk = tl.where(valid, qk, float("-inf"))   # padded rows must not leak into the softmax
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.math.exp2(qk - m_new)
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m_i = m_new

    out = acc / l_i
    tl.store(OUT + seq * stride_qb + head * stride_qh + offs_d,
             out.to(OUT.dtype.element_ty))


def paged_decode(q, k_cache, v_cache, block_table, seq_lens, sm_scale):
    B, H, HEAD_DIM = q.shape
    KVH = k_cache.shape[2]
    PAGE_SIZE, MAX_PAGES = k_cache.shape[1], block_table.shape[1]
    out = torch.empty_like(q)
    grid = (B, H)
    _paged_decode[grid](
        q, k_cache, v_cache, block_table, seq_lens, out, sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        H=H, KVH=KVH, HEAD_DIM=HEAD_DIM, PAGE_SIZE=PAGE_SIZE, MAX_PAGES=MAX_PAGES,
        num_warps=4,
    )
    return out


def ref_decode(q, k_cache, v_cache, block_table, seq_lens, sm_scale, page_size):
    # gather the pages back into one contiguous K/V per sequence, then plain attention.
    # simple to read, and obviously not how you'd do it in production
    B, H, D = q.shape
    KVH = k_cache.shape[2]
    g = H // KVH
    out = torch.empty_like(q)
    for b in range(B):
        L = int(seq_lens[b])
        pages = block_table[b, : triton.cdiv(L, page_size)]
        k = k_cache[pages].reshape(-1, KVH, D)[:L].repeat_interleave(g, dim=1)
        v = v_cache[pages].reshape(-1, KVH, D)[:L].repeat_interleave(g, dim=1)
        att = torch.softmax(torch.einsum('hd,lhd->hl', q[b].float(), k.float()) * sm_scale, -1)
        out[b] = torch.einsum('hl,lhd->hd', att, v.float()).to(out.dtype)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, KVH, HEAD_DIM, PAGE_SIZE = 4, 8, 2, 64, 16
    MAX_SEQ, MAX_PAGES = 512, 32
    dtype = torch.float16

    seq_lens = torch.randint(1, MAX_SEQ + 1, (B,), device="cuda", dtype=torch.int32)
    num_blocks = B * MAX_PAGES + 4   # a real allocator keeps a free list; here just over-allocate
    k_cache = torch.randn(num_blocks, PAGE_SIZE, KVH, HEAD_DIM, dtype=dtype, device="cuda")
    v_cache = torch.randn(num_blocks, PAGE_SIZE, KVH, HEAD_DIM, dtype=dtype, device="cuda")
    # hand each sequence a disjoint set of pages, the way the allocator would
    block_table = torch.randperm(num_blocks, device="cuda")[:B * MAX_PAGES] \
                        .reshape(B, MAX_PAGES).to(torch.int32)
    q = torch.randn(B, H, HEAD_DIM, dtype=dtype, device="cuda")
    sm_scale = 1 / HEAD_DIM ** 0.5

    out = paged_decode(q, k_cache, v_cache, block_table, seq_lens, sm_scale)
    out_ref = ref_decode(q, k_cache, v_cache, block_table, seq_lens, sm_scale, PAGE_SIZE)
    print(f"max error = {(out - out_ref).abs().max().item():.2e}")
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=0)
    print("✅ paged attention decode correctness passed")
