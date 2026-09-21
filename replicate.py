import torch
import triton
import triton.language as tl
from triton.runtime import driver

def naive_sofrmax(x):
    x_max = x.max(dim = 1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret

@triton.jit
def kernel_add(input_ptr, n_cols, n_rows, output_ptr, BLOCK_SIZE:tl.constexpr, num_stages:tl.constexpr):
    row_start = tl.program_id(axis=0)
    row_stride = tl.num_programs(axis=0)
    for row_idx in tl.arange(row_start, n_rows, row_stride, num_stages=num_stages):
        row_start_ptr = row_start + row_idx * n_cols
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptr = col_offsets + row_start_ptr
        mask = col_offsets < n_cols
        row = tl.load(input_ptr, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_ptr = col_offsets + row_start + row_idx * n_cols
        tl.store(output_ptr, softmax_output, mask=mask)


