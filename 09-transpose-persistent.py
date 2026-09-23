"""转置 + persistent kernel 两个小练习。

转置主要看 coalescing：总有一个方向是非合并访问，用 32x128 的扁块让
慢方向一次读 128B。make_block_ptr 的 boundary_check 可以替掉手写 mask。
persistent kernel 的思路是只启动 NUM_SM 个 program，每个 while 循环
领 tile，省掉调度开销，也是 stream-K 这类算法的地基。
"""

import torch
import triton
import triton.language as tl


# ---------- A. 矩阵转置 ----------

@triton.jit
def _transpose_kernel(
        input_ptr, output_ptr,
        M, N,
        stride_in_m, stride_in_n,
        stride_out_m, stride_out_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 块指针：声明"我要 M×N 矩阵里 (pid_m*BM, pid_n*BN) 开始的一块"，
    # 越界部分由 boundary_check 自动 mask，等价于手写
    #   mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    in_block = tl.make_block_ptr(
        base=input_ptr, shape=(M, N), strides=(stride_in_m, stride_in_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    out_block = tl.make_block_ptr(
        base=output_ptr, shape=(N, M), strides=(stride_out_m, stride_out_n),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M), order=(1, 0),
    )
    x = tl.load(in_block, boundary_check=(0, 1))
    tl.store(out_block, x.T, boundary_check=(0, 1))   # tl.trans(x) 同义


def triton_transpose(x):
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _transpose_kernel[grid](x, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                            BLOCK_M=32, BLOCK_N=128)
    return y


# ---------- B. 持久化 kernel 骨架 ----------

@triton.jit
def _persistent_scale_kernel(
        x_ptr, y_ptr, n_elements, scale,
        BLOCK_SIZE: tl.constexpr, NUM_SMS: tl.constexpr,
):
    # 持久化调度的标准三件套：
    #   1. tile_id 从 program_id 起步，步长为 NUM_SMS（static range 不行，必须 while）
    #   2. 每次把 tile_id 平移到该 tile 的实际偏移
    #   3. tl.multiple_of 告诉编译器对齐信息，帮助向量化
    start = tl.program_id(0) * BLOCK_SIZE
    step = NUM_SMS * BLOCK_SIZE
    for offset in tl.range(start, n_elements, step, flatten=True):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, x * scale, mask=mask)


def persistent_scale(x, scale):
    y = torch.empty_like(x)
    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    BLOCK_SIZE = 2048
    grid = (min(NUM_SMS, triton.cdiv(x.numel(), BLOCK_SIZE)),)
    _persistent_scale_kernel[grid](x, y, x.numel(), scale, BLOCK_SIZE=BLOCK_SIZE, NUM_SMS=NUM_SMS,
                                   num_warps=8)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 4096

    # A. 转置正确性 + 性能
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    y = triton_transpose(x)
    torch.testing.assert_close(y, x.T)
    print("✅ transpose 正确性通过")

    def bench(fn, iters=200):
        for _ in range(10): fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters): fn()
        end.record(); torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1e3  # us

    t_triton = bench(lambda: triton_transpose(x))
    t_torch = bench(lambda: x.T.contiguous())
    gb = x.numel() * x.element_size() * 2 / 1e9
    print(f"transpose: triton {t_triton:.1f} us ({gb/t_triton*1e6:.0f} GB/s)  "
          f"torch {t_torch:.1f} us ({gb/t_torch*1e6:.0f} GB/s)")

    # B. 持久化 kernel 正确性
    v = torch.randn(1 << 24, device='cuda', dtype=torch.float32)
    torch.testing.assert_close(persistent_scale(v, 2.0), v * 2.0)
    print("✅ persistent kernel 正确性通过")
