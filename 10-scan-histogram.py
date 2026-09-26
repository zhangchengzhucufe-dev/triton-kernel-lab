"""associative_scan (prefix sum) and atomic histogram.

Prefix sums: + is associative, so it can be parallelized — Mamba's
selective scan is exactly this. The histogram is a comparison: with few
bins, global atomics all queue on the same few addresses; keeping private
copies in registers and merging once at the end is 20x faster with 8 bins.
"""

import torch
import triton
import triton.language as tl


# ---------- A. Prefix sum ----------

@triton.jit
def _combine_add(a, b):
    # combine_fn's signature is determined by the scan's "tuple width": here
    # it is a (val, idx) pair, and the return value must be a tuple of the
    # same shape. Triton uses it to build the monoid's binary operator.
    return a + b


@triton.jit
def cumsum_kernel(
        x_ptr, y_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one segment independently: compute the in-block
    # prefix sum (intra-block scan) first, then add the total of all previous
    # segments (requires inter-block dependency -> this example simplifies it
    # to single-program multi-dim processing; a cross-block version needs a
    # two-pass scan or the hierarchical variant of tl.associative_scan)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.associative_scan(x, axis=0, combine_fn=_combine_add)
    tl.store(y_ptr + offsets, y, mask=mask)


def block_cumsum(x, block_size=1024):
    """Per-segment cumsum: prefix sum within each segment. See the two-pass scan below for cross-segment."""
    y = torch.empty_like(x)
    n = x.numel()
    cumsum_kernel[(triton.cdiv(n, block_size),)](x, y, n, BLOCK_SIZE=block_size)
    return y


@triton.jit
def cumsum_two_pass_kernel(
        x_ptr, y_ptr, block_sums_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    # Standard two-pass scan structure:
    #   Pass 1 (this kernel, stage 1): local prefix sum per segment + segment total
    #   Pass 2 (host cumsums block_sums and passes it back): add offsets from all preceding segments
    # This shows stage 1; stage 2 offsets are passed in via block_sums (already prefix-summed) by the host.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.cumsum(x, axis=0)          # syntax sugar: equivalent to associative_scan(+)
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(block_sums_ptr + pid, tl.sum(x, axis=0))


def full_cumsum(x, block_size=1024):
    """Full cross-block prefix sum: the inter-block part is tiny, so the second pass runs in torch."""
    n = x.numel()
    num_blocks = triton.cdiv(n, block_size)
    y = torch.empty_like(x)
    block_sums = torch.empty((num_blocks,), device=x.device, dtype=x.dtype)
    cumsum_two_pass_kernel[(num_blocks,)](x, y, block_sums, n, BLOCK_SIZE=block_size)
    block_prefix = torch.cumsum(block_sums, dim=0) - block_sums  # total before each segment
    return y + block_prefix.repeat_interleave(block_size)[:n]


# ---------- B. Atomic histogram ----------

@triton.jit
def histogram_naive_kernel(
        x_ptr, hist_ptr, n_elements, num_bins,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Naive version: one global atomic add per element. When data concentrates
    # in a few bins (e.g. low dynamic-range gradients), all warps serialize
    # queuing on the same few addresses.
    tl.atomic_add(hist_ptr + x, 1, mask=mask)


@triton.jit
def histogram_privatized_kernel(
        x_ptr, hist_ptr, n_elements, num_bins,
        BLOCK_SIZE: tl.constexpr, NUM_PRIV: tl.constexpr,
):
    # Privatization: accumulate in each program's registers first, merge once at the end.
    # The NUM_PRIV private bins live directly in registers (shared memory also works).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-1)
    # Broadcasted comparison (BLOCK_SIZE, NUM_PRIV) -> column sum gives this block's histogram.
    # Don't write it as scalar comparisons inside a static_range loop; easy to get wrong.
    priv = tl.sum((x[:, None] == tl.arange(0, NUM_PRIV)[None, :]).to(tl.int32), axis=0)
    tl.atomic_add(hist_ptr + tl.arange(0, NUM_PRIV), priv)


def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters): fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


if __name__ == "__main__":
    torch.manual_seed(0)

    # A. Prefix sum correctness
    x = torch.randint(0, 10, (1 << 20,), device='cuda', dtype=torch.int32)
    # Per-segment version is only correct within a segment; check with single-segment data
    seg = x[:1024]
    torch.testing.assert_close(block_cumsum(seg), torch.cumsum(seg, 0).to(torch.int32), atol=0, rtol=0)
    print("✅ per-segment cumsum OK")
    torch.testing.assert_close(full_cumsum(x).to(torch.int64), torch.cumsum(x, 0), atol=0, rtol=0)
    print("✅ two-pass full cumsum OK")

    # B. Histogram: correctness + naive vs privatized performance
    data = torch.randint(0, 8, (1 << 22,), device='cuda', dtype=torch.int32)  # only 8 bins -> max contention
    num_bins = 8
    ref = torch.bincount(data, minlength=num_bins)

    h1 = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    histogram_naive_kernel[(triton.cdiv(data.numel(), 1024),)](data, h1, data.numel(), num_bins, BLOCK_SIZE=1024)
    torch.testing.assert_close(h1.to(torch.int64), ref, atol=0, rtol=0)

    h2 = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    histogram_privatized_kernel[(triton.cdiv(data.numel(), 1024),)](
        data, h2, data.numel(), num_bins, BLOCK_SIZE=1024, NUM_PRIV=num_bins)
    torch.testing.assert_close(h2.to(torch.int64), ref, atol=0, rtol=0)
    print("✅ both histogram implementations OK")

    h_naive = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    h_priv = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    grid = (triton.cdiv(data.numel(), 1024),)
    t1 = bench(lambda: histogram_naive_kernel[grid](data, h_naive, data.numel(), num_bins, BLOCK_SIZE=1024))
    t2 = bench(lambda: histogram_privatized_kernel[grid](data, h_priv, data.numel(), num_bins,
                                                         BLOCK_SIZE=1024, NUM_PRIV=num_bins))
    print(f"histogram: naive {t1:.1f} us  privatized {t2:.1f} us  (extreme contention, 8 bins)")
