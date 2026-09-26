"""split-K GEMM: when M*N is small there aren't enough tiles and SMs idle en masse;
also split the K dimension for parallelism, then reduce the partial sums. Both
reduction flavors are implemented:

- atomic: partial sums are atomic_add'ed straight into a fp32 C. Simplest, but
  the float addition order is nondeterministic, so results may differ slightly
  run to run (nondeterministic mode);
- two-stage: partial sums go to a workspace, then one kernel sums along the
  splits dimension — deterministic. cutlass's splitK/stream-K takes this path.
  I was lazy and used torch sum for stage 2.

In practice, the skinny 16x4096x16384 matrix is a bit faster than cuBLAS.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_atomic_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLITS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)          # third grid axis: K split index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per_split = tl.cdiv(K, SPLITS)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(k_start, k_end, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_end), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < k_end) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    # Each split atomically accumulates into the fp32 C directly (C must be zero_()'ed beforehand)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _splitk_two_stage_kernel(
        a_ptr, b_ptr, ws_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_wk, stride_wn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLITS: tl.constexpr,
):
    # Stage 1: same split computation as the atomic version, but partial sums go to the split's own workspace slice
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per_split = tl.cdiv(K, SPLITS)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(k_start, k_end, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_end), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < k_end) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    ws_ptrs = ws_ptr + pid_k * stride_wk + offs_m[:, None] * stride_wn + offs_n[None, :]
    tl.store(ws_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def splitk_matmul_atomic(A, B, splits=8):
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)   # atomic needs C zeroed first
    BM, BN, BK = 16, 64, 64
    _splitk_atomic_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), splits)](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLITS=splits, num_warps=4,
    )
    return C


def splitk_matmul_two_stage(A, B, splits=8):
    M, K = A.shape
    _, N = B.shape
    BM, BN, BK = 16, 64, 64
    ws = torch.empty((splits, M, N), device=A.device, dtype=torch.float32)
    _splitk_two_stage_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), splits)](
        A, B, ws, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), ws.stride(0), ws.stride(1),        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLITS=splits, num_warps=4,
    )
    return ws.sum(dim=0)      # Stage 2: torch sum here for convenience; a real implementation would use a second Triton kernel


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
    # Skinny matrix: small M, large K — split-K's home turf (e.g. LM head / decode GEMV)
    M, N, K = 16, 4096, 16384
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ref = (A.float() @ B.float())

    c1 = splitk_matmul_atomic(A, B)
    c2 = splitk_matmul_two_stage(A, B)
    torch.testing.assert_close(c1, ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c2, ref, atol=1e-1, rtol=1e-2)
    print(f"atomic version error {(c1 - ref).abs().max().item():.2e}, two-stage version error {(c2 - ref).abs().max().item():.2e}")
    print("✅ split-K GEMM correctness passed")

    t_cublas = bench(lambda: A @ B)
    t_atomic = bench(lambda: splitk_matmul_atomic(A, B))
    t_2stage = bench(lambda: splitk_matmul_two_stage(A, B))
    print(f"cuBLAS {t_cublas:.2f} ms   split-K(atomic) {t_atomic:.2f} ms   split-K(two-stage) {t_2stage:.2f} ms")
