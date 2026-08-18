import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask = mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_element = output.numel()
    grid = lambda meta: (triton.cdiv(n_element, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_element, BLOCK_SIZE = 1024)
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device = DEVICE)
y = torch.rand(size, device = DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')