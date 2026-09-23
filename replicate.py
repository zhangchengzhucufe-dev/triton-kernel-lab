# 自己照着 tutorial 02 复写了一遍 fused_softmax.py，不看书能写对才算掌握
import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel_softmax(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(axis=0)
    row_stride = tl.num_programs(axis=0)
    for row_idx in tl.range(row_start, n_rows, row_stride, num_stages=num_stages):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # 一行一行处理，行内偏移 = 行号 * 行长 + 列号
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        kernel_output = numerator / denominator
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(output_ptrs, kernel_output, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
SIZE_SMEM = properties["max_shared_mem"]
NUM_SMS = properties["multiprocessor_count"]


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel_softmax[(min(n_rows, NUM_SMS),)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda', dtype=torch.float32)  # 故意用非 2 的幂
    triton_output = softmax(x)
    torch_output = torch.softmax(x, dim=1)
    print(f"最大误差 = {(triton_output - torch_output).abs().max().item():.2e}")
    assert torch.allclose(triton_output, torch_output, atol=1e-6, rtol=0)
    print("✅ 复写的 softmax 正确")
