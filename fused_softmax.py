import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx90a', 'gfx908')

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

@triton.jit
def softmax_kernel(input_ptr, output_ptr,
                   input_row_stride, output_row_stride,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.arange(row_start, n_rows, row_step, num_stages=num_stages):
        # The row we will operate on
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The range of columns we'll handle in this BLOCK
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may > n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape
    # The block size of each loop iteration is the smallest power of two greater than the number of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel, _ = triton.compile(
        softmax_kernel,
        [x, y, x.stride(0), y.stride(0), n_rows, n_cols],
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(NUM_SM,)
    )
    # Get kernel properties
    regs_per_thread = kernel.metadata.regs_per_thread
    smem_per_program = kernel.metadata.shared
    occupancy = NUM_REGS // (regs_per_thread * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // smem_per_program)
    num_programs = NUM_SM * occupancy

    # Launch kernel
    softmax_kernel[(num_programs,)](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y

# ===== Test Unit =====
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
print(f"Max difference: {torch.max(torch.abs(y_triton - y_torch))}")

# ===== Benchmark =====
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_names=['Triton', 'Torch', 'NaiveTorch'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        xlabel='N',
        plot_name='softmax-performance',
        args={'M': 4096},
    )
)
def benchmark(M, N):
    x = torch.randn(M, N, device=DEVICE)
    quantiles = [0.5, 0.2, 0.8]
    ms_triton, min_triton, max_triton = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    ms_torch, min_torch, max_torch = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    ms_naive, min_naive, max_naive = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms_triton), gbps(ms_torch), gbps(ms_naive)

benchmark.run(show_plots=True, print_data=True)
