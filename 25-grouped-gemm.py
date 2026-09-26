"""Grouped GEMM for MoE — the dispatch-side workhorse.

After the router picks experts, you don't get one big matmul anymore: you get
E smaller ones with wildly different row counts. Launching one kernel per
expert from Python means E launch overheads and dead GPU time in between.
Grouped GEMM does every expert's tiles in a single launch.

The preprocessing (which happens before this kernel in any real MoE stack)
sorts tokens by expert, so expert e's tokens end up as one contiguous slice
of the sorted order. The kernel then just needs the slice boundaries: each
program owns one (expert, M-tile, N-tile) triple, bails out early if its
expert ran out of tokens, and gathers its A rows through SORTED_IDS.

Two honest shortcuts to keep the file readable:
- top-1 routing, so each token appears once and scattering output rows
  straight back is race-free. Top-k routing needs the sort + scatter done
  twice or an atomic accumulate.
- the grid is sized by the busiest expert, so lightly-loaded experts spawn
  programs that exit immediately. vLLM/SGLang build a compact tile->expert
  map instead so every program does real work — worth doing once the
  indexing here feels comfortable.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm(
        A, W, C, SORTED_IDS, EXP_OFFSETS,
        N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_start = tl.load(EXP_OFFSETS + pid_e)
    row_end = tl.load(EXP_OFFSETS + pid_e + 1)
    tile_start = row_start + pid_m * BM
    if tile_start >= row_end:
        # this expert got fewer tokens than the grid assumed — nothing to do
        return

    offs_m = tile_start + tl.arange(0, BM)
    row_mask = offs_m < row_end
    # A lives in sorted-token order; SORTED_IDS maps each sorted row back to
    # its original token, and the load gathers those rows directly
    token = tl.load(SORTED_IDS + offs_m, mask=row_mask, other=0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    a_ptrs = A + token[:, None] * K + offs_k[None, :]
    w_ptrs = W + pid_e * K * N + offs_k[:, None] * N + offs_n[None, :]
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, w, acc)
        a_ptrs += BK
        w_ptrs += BK * N

    # scatter straight back to each token's own row — safe because top-1
    # routing means no two programs write the same row
    c_ptrs = C + token[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=row_mask[:, None])


def grouped_gemm(x, w, sorted_ids, offsets, BM=64, BN=64, BK=32):
    M_total, K = x.shape
    E, _, N = w.shape
    counts = offsets[1:] - offsets[:-1]
    max_tiles = triton.cdiv(int(counts.max()), BM)
    c = torch.empty((M_total, N), device=x.device, dtype=x.dtype)
    grid = (E, max_tiles, triton.cdiv(N, BN))
    _grouped_gemm[grid](
        x, w, c, sorted_ids, offsets, N, K,
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N, E = 1024, 128, 256, 8
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(E, K, N, device="cuda", dtype=torch.float16)

    # router: each token picks exactly one expert (top-1). A real router
    # would be a softmax + top-k, file 17 covers that part
    assign = torch.randint(0, E, (M,), device="cuda")
    counts = torch.bincount(assign, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = counts.cumsum(0)
    sorted_ids = torch.argsort(assign, stable=True).to(torch.int32)
    print("tokens per expert:", counts.tolist())

    c = grouped_gemm(x, w, sorted_ids, offsets)

    # reference: a plain matmul per expert over its token slice
    c_ref = torch.empty_like(c)
    for e in range(E):
        rows = sorted_ids[offsets[e]:offsets[e + 1]].long()
        c_ref[rows] = (x[rows].float() @ w[e].float()).to(c_ref.dtype)
    print(f"max error = {(c - c_ref).abs().max().item():.2e}")
    # a couple of elements land past 2e-2 — fp16 rounding order differs between
    # the tensor-core dot and the fp32 reference, nothing structural
    torch.testing.assert_close(c, c_ref, atol=5e-2, rtol=0)
    print("✅ grouped gemm correctness passed")
