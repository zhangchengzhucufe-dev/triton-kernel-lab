"""split-K GEMM：M*N 小的时候 tile 不够分，SM 大片空转，把 K 维也切开
并行，最后把部分和归并。两种归并都写了：

- atomic：部分和直接 atomic_add 到 fp32 的 C。最省事，但浮点加法顺序
  不定，结果每次跑可能差一点（非确定模式）；
- 两阶段：部分和写 workspace，再一个 kernel 沿 splits 维求和，结果确定。
  cutlass 的 splitK/stream-K 走这条路。阶段 2 我偷懒用 torch sum 了。

实测 16x4096x16384 的瘦矩阵比 cuBLAS 快一点。
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
    pid_k = tl.program_id(2)          # 第三档 grid：K 分片编号

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

    # 各分片直接原子累加进 fp32 的 C（C 必须预先 zero_()）
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
    # 阶段 1：与 atomic 版相同的分片计算，但把部分和写进自己的 workspace 切片
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
    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)   # atomic 要先清零
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
    return ws.sum(dim=0)      # 阶段 2：这里借 torch 求和，实际实现应是第二个 Triton kernel


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
    # 瘦矩阵：M 小、K 大 —— split-K 的主场（比如 LM 头 / 解码 GEMV）
    M, N, K = 16, 4096, 16384
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ref = (A.float() @ B.float())

    c1 = splitk_matmul_atomic(A, B)
    c2 = splitk_matmul_two_stage(A, B)
    torch.testing.assert_close(c1, ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c2, ref, atol=1e-1, rtol=1e-2)
    print(f"atomic 版误差 {(c1 - ref).abs().max().item():.2e}，两阶段版误差 {(c2 - ref).abs().max().item():.2e}")
    print("✅ split-K GEMM 正确性通过")

    t_cublas = bench(lambda: A @ B)
    t_atomic = bench(lambda: splitk_matmul_atomic(A, B))
    t_2stage = bench(lambda: splitk_matmul_two_stage(A, B))
    print(f"cuBLAS {t_cublas:.2f} ms   split-K(atomic) {t_atomic:.2f} ms   split-K(两阶段) {t_2stage:.2f} ms")
