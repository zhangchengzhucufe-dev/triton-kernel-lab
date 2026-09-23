"""associative_scan（前缀和）和原子直方图。

前缀和：+ 满足结合律所以能并行化，Mamba 的 selective scan 就是这个。
直方图做了个对比：bin 少的时候全局 atomic 全撞在几个地址上排队，
先在寄存器里存私有副本最后合并一次，8 bin 的场景下快 20 倍。
"""

import torch
import triton
import triton.language as tl


# ---------- A. 前缀和 ----------

@triton.jit
def _combine_add(a, b):
    # combine_fn 的签名由 scan 的"元组宽度"决定：这里是 (val, idx) 对，
    # 返回值也必须是同样形状的元组。Triton 会用它构造 monoid 的二元运算。
    return a + b


@triton.jit
def cumsum_kernel(
        x_ptr, y_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 每个 program 独立处理一段：先求本段内的前缀和（块内 scan），
    # 再加上前面所有段的总和（需要一次块间依赖 → 本例简化为单程序多维处理，
    # 跨块版本需要两遍扫描或 tl.associative_scan 的 hierarchical 变体）
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.associative_scan(x, axis=0, combine_fn=_combine_add)
    tl.store(y_ptr + offsets, y, mask=mask)


def block_cumsum(x, block_size=1024):
    """分段 cumsum：每段内部前缀和。跨段版本见下方的两遍扫描。"""
    y = torch.empty_like(x)
    n = x.numel()
    cumsum_kernel[(triton.cdiv(n, block_size),)](x, y, n, BLOCK_SIZE=block_size)
    return y


@triton.jit
def cumsum_two_pass_kernel(
        x_ptr, y_ptr, block_sums_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    # 两遍扫描（two-pass scan）的标准结构：
    #   第一遍（本 kernel 阶段 1）：每段局部前缀和 + 段总和
    #   第二遍（host 端对 block_sums 做 cumsum 后传回）：加上所有前段的偏移
    # 这里演示阶段 1；阶段 2 的偏移量由 host 传入 block_sums（已含前缀）。
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.cumsum(x, axis=0)          # 语法糖：等价 associative_scan(+)
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(block_sums_ptr + pid, tl.sum(x, axis=0))


def full_cumsum(x, block_size=1024):
    """支持跨块的完整前缀和：GPU 块间部分规模极小，用 torch 完成第二遍。"""
    n = x.numel()
    num_blocks = triton.cdiv(n, block_size)
    y = torch.empty_like(x)
    block_sums = torch.empty((num_blocks,), device=x.device, dtype=x.dtype)
    cumsum_two_pass_kernel[(num_blocks,)](x, y, block_sums, n, BLOCK_SIZE=block_size)
    block_prefix = torch.cumsum(block_sums, dim=0) - block_sums  # 每段前的总和
    return y + block_prefix.repeat_interleave(block_size)[:n]


# ---------- B. 原子直方图 ----------

@triton.jit
def histogram_naive_kernel(
        x_ptr, hist_ptr, n_elements, num_bins,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    # 朴素版：每个元素一次全局原子加。当数据集中在少数 bin 时（如低动态
    # 范围的梯度），所有 warp 会在同几个地址上排队串行化。
    tl.atomic_add(hist_ptr + x, 1, mask=mask)


@triton.jit
def histogram_privatized_kernel(
        x_ptr, hist_ptr, n_elements, num_bins,
        BLOCK_SIZE: tl.constexpr, NUM_PRIV: tl.constexpr,
):
    # privatization：先在每个 program 的寄存器里累计，最后合并一次。
    # NUM_PRIV 个私有 bin 直接驻留寄存器（也可放 shared memory）。
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-1)
    # 广播比较 (BLOCK_SIZE, NUM_PRIV) → 按列求和得到本块的直方图。
    # 注意别写成 static_range 循环里标量比较的版本，语义容易出错。
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

    # A. 前缀和正确性
    x = torch.randint(0, 10, (1 << 20,), device='cuda', dtype=torch.int32)
    # 分段版只保证段内正确，用单段数据对照
    seg = x[:1024]
    torch.testing.assert_close(block_cumsum(seg), torch.cumsum(seg, 0).to(torch.int32), atol=0, rtol=0)
    print("✅ 分段 cumsum 正确")
    torch.testing.assert_close(full_cumsum(x).to(torch.int64), torch.cumsum(x, 0), atol=0, rtol=0)
    print("✅ 两遍扫描 full cumsum 正确")

    # B. 直方图：正确性 + 朴素 vs 私有化性能
    data = torch.randint(0, 8, (1 << 22,), device='cuda', dtype=torch.int32)  # 只有 8 个 bin → 冲突拉满
    num_bins = 8
    ref = torch.bincount(data, minlength=num_bins)

    h1 = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    histogram_naive_kernel[(triton.cdiv(data.numel(), 1024),)](data, h1, data.numel(), num_bins, BLOCK_SIZE=1024)
    torch.testing.assert_close(h1.to(torch.int64), ref, atol=0, rtol=0)

    h2 = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    histogram_privatized_kernel[(triton.cdiv(data.numel(), 1024),)](
        data, h2, data.numel(), num_bins, BLOCK_SIZE=1024, NUM_PRIV=num_bins)
    torch.testing.assert_close(h2.to(torch.int64), ref, atol=0, rtol=0)
    print("✅ 直方图两种实现正确")

    h_naive = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    h_priv = torch.zeros((num_bins,), device='cuda', dtype=torch.int32)
    grid = (triton.cdiv(data.numel(), 1024),)
    t1 = bench(lambda: histogram_naive_kernel[grid](data, h_naive, data.numel(), num_bins, BLOCK_SIZE=1024))
    t2 = bench(lambda: histogram_privatized_kernel[grid](data, h_priv, data.numel(), num_bins,
                                                         BLOCK_SIZE=1024, NUM_PRIV=num_bins))
    print(f"直方图：朴素 {t1:.1f} us  私有化 {t2:.1f} us  （8 个 bin 的极端冲突场景）")
