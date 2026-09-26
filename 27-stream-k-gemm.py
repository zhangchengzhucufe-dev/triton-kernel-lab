"""Stream-K GEMM — the answer to wave quantization.

A tiled GEMM with T tiles on S SMs runs ceil(T/S) waves, and the last wave
usually has far fewer tiles than SMs, so the GPU idles while a handful of
tiles finish. Classic split-K (file 19) fixes this by splitting the K
dimension, but it splits *every* tile uniformly, which adds reduction
overhead everywhere and helps most when T is small.

Stream-K instead flattens all (tile, k-iter) work into one long list of
total_tiles * K/BK iterations, deals them out to programs in equal-length
contiguous chunks, and lets a chunk start or end in the middle of a tile.
Full tiles get stored directly; chunks that only did part of a tile write
their partial sum to a workspace, and a tiny fixup kernel adds the partials
up per tile. Deterministic — no atomics — because every partial has its own
slot and the fixup sums them in a fixed order.

The work decomposition lives on the host (it must agree with the kernel's
run-walking logic down to which workspace slot each partial lands in), and
M/N/K are assumed divisible by the block sizes — padding is plumbing.
"""

import torch
import triton
import triton.language as tl

MAX_PARTIALS_PER_TILE = 8


@triton.jit
def _streamk_gemm(
        A, B, C, P, SLOTS,
        N, K,
        total_iters, iters_per_tile, iters_per_program,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        MAX_P: tl.constexpr,
):
    pid = tl.program_id(0)
    start_iter = pid * iters_per_program
    end_iter = tl.minimum(start_iter + iters_per_program, total_iters)
    chunk_start = start_iter
    tiles_per_row = N // BN

    while start_iter < end_iter:
        tile = start_iter // iters_per_tile
        run_end = tl.minimum((tile + 1) * iters_per_tile, end_iter)
        pid_m = tile // tiles_per_row
        pid_n = tile % tiles_per_row

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        k_off = start_iter - tile * iters_per_tile
        # k_off counts BK-sized blocks, so the B block starts k_off*BK ROWS
        # into B — that's k_off*BK*N elements. missing the *N here cost me an
        # hour: everything still ran, 98% of tiles were even correct
        a_ptrs = A + offs_m[:, None] * K + k_off * BK + tl.arange(0, BK)[None, :]
        b_ptrs = B + k_off * BK * N + tl.arange(0, BK)[:, None] * N + offs_n[None, :]

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for _ in range(k_off, run_end - tile * iters_per_tile):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * N

        starts_tile = start_iter == tile * iters_per_tile
        ends_tile = run_end == (tile + 1) * iters_per_tile
        c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
        if starts_tile and ends_tile:
            # this program owned the whole tile — store straight to C
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            # partial tile: park the accumulator in this program's slot and
            # let the fixup kernel reduce. slot 2*pid is the chunk's first
            # run, 2*pid+1 its last, so a chunk never needs more than two
            slot = 2 * pid + tl.where(start_iter == chunk_start, 0, 1)
            p_offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
            tl.store(P + slot * BM * BN + p_offs, acc)
        start_iter = run_end


@triton.jit
def _streamk_fixup(
        C, P, SLOTS,
        N, tiles_per_row,
        BM: tl.constexpr, BN: tl.constexpr, MAX_P: tl.constexpr,
):
    tile = tl.program_id(0)
    # tiles owned by a single program were already stored directly — touching
    # them here would zero them out. slots[tile][0] == -1 means no partials
    if tl.load(SLOTS + tile * MAX_P) < 0:
        return
    pid_m = tile // tiles_per_row
    pid_n = tile % tiles_per_row
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    p_offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    for j in range(0, MAX_P):
        slot = tl.load(SLOTS + tile * MAX_P + j)
        # empty slots are padded with -1; clamp to slot 0 (valid memory) and
        # zero the load out with the scalar mask — cheaper than branching
        p = tl.load(P + tl.maximum(slot, 0) * BM * BN + p_offs)
        acc += p * (slot >= 0)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(C.dtype.element_ty))


def streamk_gemm(a, b, num_sms, BM=128, BN=128, BK=64):
    M, K = a.shape
    _, N = b.shape
    assert M % BM == 0 and N % BN == 0 and K % BK == 0
    total_tiles = (M // BM) * (N // BN)
    iters_per_tile = K // BK
    total_iters = total_tiles * iters_per_tile
    iters_per_program = triton.cdiv(total_iters, num_sms)

    # mirror the kernel's run-walking on the host so the fixup knows exactly
    # which workspace slots belong to which tile. same logic, python form
    slots = [[] for _ in range(total_tiles)]
    for pid in range(num_sms):
        s, e = pid * iters_per_program, min((pid + 1) * iters_per_program, total_iters)
        if s >= e:
            continue
        run_idx = 0
        while s < e:
            tile = s // iters_per_tile
            run_end = min((tile + 1) * iters_per_tile, e)
            if not (s == tile * iters_per_tile and run_end == (tile + 1) * iters_per_tile):
                slots[tile].append(2 * pid + (0 if run_idx == 0 else 1))
            s = run_end
            run_idx += 1
    assert max(len(s) for s in slots) <= MAX_PARTIALS_PER_TILE
    slot_table = torch.full((total_tiles, MAX_PARTIALS_PER_TILE), -1,
                            dtype=torch.int32, device=a.device)
    for tile, ss in enumerate(slots):
        if ss:
            slot_table[tile, :len(ss)] = torch.tensor(ss, dtype=torch.int32, device=a.device)

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # only tiles with partials need the fixup, but the exit-early cost of a
    # dense grid is a few microseconds — not worth host-side filtering
    p = torch.empty((2 * num_sms, BM, BN), device=a.device, dtype=torch.float32)
    _streamk_gemm[(num_sms,)](
        a, b, c, p, slot_table, N, K,
        total_iters, iters_per_tile, iters_per_program,
        BM=BM, BN=BN, BK=BK, MAX_P=MAX_PARTIALS_PER_TILE,
        num_warps=8, num_stages=2,
    )
    _streamk_fixup[(total_tiles,)](
        c, p, slot_table, N, N // BN,
        BM=BM, BN=BN, MAX_P=MAX_PARTIALS_PER_TILE, num_warps=4,
    )
    return c


def benchmark(fn, iters=20):
    fn()  # warmup / compile
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = streamk_gemm(a, b, num_sms)
    c_ref = torch.matmul(a, b)
    print(f"max error = {(c - c_ref).abs().max().item():.2e}")
    torch.testing.assert_close(c, c_ref, atol=2e-2, rtol=2e-2)
    print("✅ stream-K gemm correctness passed")

    # the motivation, in numbers: classic tiling gets T/S waves and the tail
    # wave runs on a fraction of the SMs
    tiles = (M // 64) * (N // 64)
    waves = tiles / num_sms
    tail = tiles % num_sms
    print(f"{tiles} tiles on {num_sms} SMs = {waves:.1f} waves "
          f"(tail wave uses only {tail} SMs); stream-K evens that out")

    ms = benchmark(lambda: streamk_gemm(a, b, num_sms))
    ms_cublas = benchmark(lambda: torch.matmul(a, b))
    tflops = lambda t: 2 * M * N * K / (t * 1e-3) / 1e12
    print(f"stream-K: {ms:.3f} ms ({tflops(ms):.1f} TFLOPS) | "
          f"cuBLAS: {ms_cublas:.3f} ms ({tflops(ms_cublas):.1f} TFLOPS)")
