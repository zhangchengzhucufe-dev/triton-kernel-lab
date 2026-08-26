"""Two small exercises: transpose + persistent kernel.

Transpose is mainly about coalescing: one direction is always non-coalesced, so use
a flat 32x128 block to make the slow direction read 128B at a time. make_block_ptr's
boundary_check can replace hand-written masks. The idea behind persistent kernels is
to launch only NUM_SM programs, each grabbing tiles in a while loop — this saves
scheduling overhead and is also the foundation for stream-K style algorithms.
"""

import torch
import triton
import triton.language as tl


# ---------- A. Matrix transpose ----------

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
    # Block pointer: declares "I want the block of the M×N matrix starting at
    # (pid_m*BM, pid_n*BN)"; out-of-bounds parts are auto-masked by boundary_check,
    # equivalent to hand-writing
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
    tl.store(out_block, x.T, boundary_check=(0, 1))   # same as tl.trans(x)


def triton_transpose(x):
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _transpose_kernel[grid](x, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                            BLOCK_M=32, BLOCK_N=128)
    return y


# ---------- B. Persistent kernel skeleton ----------

@triton.jit
def _persistent_scale_kernel(
        x_ptr, y_ptr, n_elements, scale,
        BLOCK_SIZE: tl.constexpr, NUM_SMS: tl.constexpr,
):
    # The standard three pieces of persistent scheduling:
    #   1. tile_id starts at program_id with stride NUM_SMS (static range won't do; must be a while-style loop)
    #   2. each iteration maps tile_id to the actual offset of that tile
    #   3. tl.multiple_of tells the compiler the alignment info, helping vectorization
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

    # A. Transpose correctness + performance
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    y = triton_transpose(x)
    torch.testing.assert_close(y, x.T)
    print("✅ transpose correctness passed")

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

    # B. Persistent kernel correctness
    v = torch.randn(1 << 24, device='cuda', dtype=torch.float32)
    torch.testing.assert_close(persistent_scale(v, 2.0), v * 2.0)
    print("✅ persistent kernel correctness passed")
