"""One benchmark harness for the whole repo.

Each numbered file is runnable on its own, but scattered numbers in print
statements don't answer the questions that matter: how close does each kernel
get to the hardware's actual limits, and where does it sit on the roofline?
This script runs the headline kernels against their baselines, measures the
GPU's *achievable* peak bandwidth and FLOPs (not the spec sheet's), and dumps
perf-results.csv plus two charts (roofline.png, speedup.png).

Arithmetic intensities are computed analytically from the shapes — flash
attention's whole point is that the N x N score matrix never touches HBM, so
the intensity math has to reflect that too.
"""

import csv
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton

REPO = os.path.dirname(os.path.abspath(__file__))


def load(num):
    spec = importlib.util.spec_from_file_location(f"k{num}", os.path.join(REPO, f"{num}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bench(fn, iters=30, repeats=5):
    # laptop GPU clocks bounce around with thermals / other processes, so a
    # single timed block is noisy — repeat and keep the best. min-of-N is the
    # standard for microbenchmarks: it estimates the uncontended run
    best = float("inf")
    for _ in range(repeats):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best  # ms


# ---------------------------------------------------------------- peaks
def measure_peaks():
    # achievable HBM bandwidth. a d2d copy moves read+write; a big reduction
    # is pure read and usually gets closer to spec — take whichever is higher
    src = torch.empty(256 * 1024 * 1024, dtype=torch.float16, device="cuda")  # 512 MiB
    dst = torch.empty_like(src)
    ms_copy = bench(lambda: dst.copy_(src), iters=20)
    copy_gbs = 2 * src.numel() * 2 / (ms_copy * 1e-3) / 1e9
    ms_read = bench(lambda: src.sum(), iters=20)
    read_gbs = src.numel() * 2 / (ms_read * 1e-3) / 1e9
    bw_gbs = max(copy_gbs, read_gbs)

    # achievable fp16 tensor FLOPs: big cuBLAS matmul
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    ms = bench(lambda: a @ a, iters=10)
    tflops = 2 * 8192**3 / (ms * 1e-3) / 1e12
    return bw_gbs, tflops


# ---------------------------------------------------------------- cases
def run_cases():
    rows = []  # (case, ours, baseline, our_ms, base_ms, flops, bytes)

    k06, k08, k09, k10, k21, k28 = (load(n) for n in (
        "06-fused-matmul", "08-flash-attention", "09-transpose-persistent",
        "10-scan-histogram", "21-flash-decoding", "28-int8-gemm"))
    sdpa = torch.nn.functional.scaled_dot_product_attention

    # --- GEMM 4096^3, compute-bound
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ms, msb = bench(lambda: k06.matmul(a, b)), bench(lambda: a @ b)
    flops = 2 * M * N * K
    our_tf, base_tf = flops / (ms * 1e-3) / 1e12, flops / (msb * 1e-3) / 1e12
    print(f"gemm 4096^3        : ours {our_tf:5.1f} TF | cuBLAS {base_tf:5.1f} TF | {msb/ms:.2f}x")
    rows.append(dict(case="gemm 4096^3", ours="06 matmul", baseline="cuBLAS",
                     our_ms=ms, base_ms=msb, flops=flops,
                     bytes=2 * (M * K + K * N + M * N)))

    # --- flash attention fwd, 4k ctx — compute-bound territory
    Z, H, S, D = 2, 32, 4096, 128
    q, k, v = (torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5 for _ in range(3))
    ms, msb = bench(lambda: k08.attention(q, k, v, D**-0.5)), bench(lambda: sdpa(q, k, v))
    flops = 4 * Z * H * S * S * D
    our_tf, base_tf = flops / (ms * 1e-3) / 1e12, flops / (msb * 1e-3) / 1e12
    print(f"attn fwd 4k ctx    : ours {our_tf:5.1f} TF | SDPA {base_tf:5.1f} TF | {msb/ms:.2f}x")
    # the N^2 scores never hit HBM — only Q/K/V/O stream
    rows.append(dict(case="attn fwd 4k ctx", ours="08 flash attn", baseline="SDPA",
                     our_ms=ms, base_ms=msb, flops=flops, bytes=4 * Z * H * S * D * 2))

    # --- flash decoding, 32k cache — the memory-bound star
    Z, H, S, D = 4, 32, 32768, 128
    q = torch.randn(Z, H, 1, D, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5
    ms, msb = bench(lambda: k21.flash_decode(q, k, v, D**-0.5)), bench(lambda: sdpa(q, k, v))
    flops = 4 * Z * H * S * D
    kv_bytes = 4 * Z * H * S * D   # K + V, fp16: two tensors of BHSD elems
    our_gbs, base_gbs = kv_bytes / (ms * 1e-3) / 1e9, kv_bytes / (msb * 1e-3) / 1e9
    print(f"decode 32k cache   : ours {our_gbs:5.0f} GB/s | SDPA {base_gbs:5.0f} GB/s | {msb/ms:.2f}x")
    rows.append(dict(case="decode 32k cache", ours="21 flash-decoding", baseline="SDPA",
                     our_ms=ms, base_ms=msb, flops=flops, bytes=kv_bytes))

    # --- transpose 8192^2, pure bandwidth
    n = 8192
    x = torch.randn(n, n, device="cuda", dtype=torch.float16)
    ms, msb = bench(lambda: k09.triton_transpose(x)), bench(lambda: x.T.contiguous())
    our_gbs, base_gbs = 2 * n * n * 2 / (ms * 1e-3) / 1e9, 2 * n * n * 2 / (msb * 1e-3) / 1e9
    print(f"transpose 8192^2   : ours {our_gbs:5.0f} GB/s | torch {base_gbs:5.0f} GB/s | {msb/ms:.2f}x")
    rows.append(dict(case="transpose 8192^2", ours="09 transpose", baseline="torch .T.contiguous()",
                     our_ms=ms, base_ms=msb, flops=0, bytes=2 * n * n * 2))

    # --- histogram with 8 hot bins, atomics vs privatization
    data = torch.randint(0, 8, (1 << 22,), device="cuda", dtype=torch.int32)
    hist = torch.zeros(8, device="cuda", dtype=torch.int32)
    h1 = torch.zeros(8, device="cuda", dtype=torch.int32)

    def naive():
        hist.zero_()
        k10.histogram_naive_kernel[(triton.cdiv(data.numel(), 1024),)](
            data, hist, data.numel(), 8, BLOCK_SIZE=1024)

    def privatized():
        h1.zero_()
        k10.histogram_privatized_kernel[(triton.cdiv(data.numel(), 1024),)](
            data, h1, data.numel(), 8, BLOCK_SIZE=1024, NUM_PRIV=8)

    ms, msb = bench(privatized), bench(naive)
    print(f"histogram 8 bins   : privatized {ms:6.1f} us | naive {msb:6.1f} us | {msb/ms:.1f}x")
    rows.append(dict(case="histogram 8 bins", ours="10 privatized", baseline="10 naive atomics",
                     our_ms=ms, base_ms=msb, flops=0, bytes=data.numel() * 4))

    # --- int8 W8A8 GEMM 2048^3
    M = N = K = 2048
    a16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    a8, sa = k28.quant_symmetric(a16, dim=1)
    b8, sw = k28.quant_symmetric(b16, dim=0)
    sa, sw = sa.float(), sw.float()
    ms, msb = bench(lambda: k28.int8_gemm(a8, b8, sa, sw)), bench(lambda: a16 @ b16)
    tops = 2 * M * N * K / (ms * 1e-3) / 1e12
    base_tf = 2 * M * N * K / (msb * 1e-3) / 1e12
    print(f"int8 gemm 2048^3   : ours {tops:5.1f} TOPS | cuBLAS fp16 {base_tf:5.1f} TF | {msb/ms:.2f}x")
    rows.append(dict(case="int8 gemm 2048^3", ours="28 int8 W8A8", baseline="cuBLAS fp16",
                     our_ms=ms, base_ms=msb, flops=2 * M * N * K,
                     bytes=(M * K + K * N) * 1 + M * N * 2))

    return rows


def write_csv(rows, path):
    bw, tf = measure_peaks()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "ours", "baseline", "our_ms", "baseline_ms", "speedup",
                    "our_throughput", "baseline_throughput", "arith_intensity_flop_per_byte",
                    "pct_of_peak"])
        for r in rows:
            sp = r["base_ms"] / r["our_ms"]
            ai = r["flops"] / r["bytes"] if r["bytes"] else 0
            if ai > 30:      # compute-bound: report TFLOPS vs cuBLAS peak
                ours = r["flops"] / (r["our_ms"] * 1e-3) / 1e12
                base = r["flops"] / (r["base_ms"] * 1e-3) / 1e12
                unit, peak = "TFLOP/s", tf
            else:            # memory-bound: report GB/s vs copy peak
                ours = r["bytes"] / (r["our_ms"] * 1e-3) / 1e9
                base = r["bytes"] / (r["base_ms"] * 1e-3) / 1e9
                unit, peak = "GB/s", bw
            w.writerow([r["case"], r["ours"], r["baseline"], f"{r['our_ms']:.3f}",
                        f"{r['base_ms']:.3f}", f"{sp:.2f}", f"{ours:.1f} {unit}",
                        f"{base:.1f} {unit}", f"{ai:.1f}", f"{100 * ours / peak:.0f}%"])
    return bw, tf


def plot(rows, bw, tf):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    # roofline: only the flops-producing kernels make sense here
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [0.1 * (1.15**i) for i in range(60)]
    ys = [min(tf, bw * x) for x in xs]
    ax.plot(xs, ys, "k--", lw=1, label=f"roofline ({tf:.0f} TFLOP/s, {bw:.0f} GB/s)")
    ridge = tf / bw
    points = []
    for r in rows:
        ai = r["flops"] / r["bytes"]
        if ai < 5:
            continue  # bandwidth kernels sit flat on the roof — charted separately
        t = r["flops"] / (r["our_ms"] * 1e-3) / 1e12
        points.append((ai, t, r["case"]))
    for ai, t, case in points:
        ax.plot(ai, t, "o", ms=8)
        ax.annotate(case, (ai, t), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.axvline(ridge, color="gray", lw=0.5, ls=":")
    ax.annotate(f"ridge point ~{ridge:.0f} FLOP/B", (ridge, tf * 0.5), rotation=90,
                fontsize=8, color="gray", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4); ax.set_ylim(1, 70)
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("throughput (TFLOP/s)")
    ax.set_title(f"{torch.cuda.get_device_properties(0).name} — fp16 roofline")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "roofline.png"))

    # speedup bars, ours relative to baseline
    fig, ax = plt.subplots(figsize=(7, 4))
    cases = [r["case"].split()[0] + (" 4k" if "4k" in r["case"] else "") for r in rows]
    sp = [r["base_ms"] / r["our_ms"] for r in rows]
    colors = ["#2a9d8f" if s >= 1 else "#e76f51" for s in sp]
    ax.bar(cases, sp, color=colors)
    ax.axhline(1.0, color="k", lw=0.8)
    for i, s in enumerate(sp):
        ax.text(i, s + 0.03, f"{s:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("ours / baseline  (higher is better)")
    ax.set_title("kernels vs the library that normally does the job")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "speedup.png"))


if __name__ == "__main__":
    torch.manual_seed(0)
    bw, tf = measure_peaks()
    print(f"measured peaks: {bw:.0f} GB/s HBM, {tf:.1f} TFLOP/s fp16 "
          f"({torch.cuda.get_device_properties(0).name})\n")
    rows = run_cases()
    write_csv(rows, os.path.join(REPO, "perf-results.csv"))
    plot(rows, bw, tf)
    print("\nwrote perf-results.csv, roofline.png, speedup.png")
