"""Brute-force config sweep for the hot kernels — answers "how did you pick
num_warps/stages?" with data instead of tutorial defaults.

The kernels hardcode their block sizes / warps / stages (03: 128/64/8/3, 21:
64/64/4/2, 02: warps by heuristic, 16: 8 splits). Those numbers came from the
tutorials and stuck. This script replays the full config space per shape,
keeps every measurement in autotune-results.csv, and prints the winner table
— so the defaults in the kernels are now *documented* choices, and the CSV
shows what the alternatives were worth.

16 (flash-decoding) has no tiles to resize — its knobs are num_splits and
block_n, and in its CSV rows those live in the BLOCK_M / BLOCK_N columns.

Why not @triton.autotune everywhere: autotune re-benchmarks per shape at
first call and hides the numbers; for a study repo I'd rather have one place
that prints the whole space. (01's matmul keeps @triton.autotune — that's the
pattern worth showing in the file itself.)

Runtime is dominated by compilation (one binary per config), not measurement:
a bit over 100 configs ≈ 4-5 min on the 3060, most of it ptxas. The bench
itself is min-of-3 with 5 warmup + 10 timed iterations per repeat — laptop
clocks make single-shot timing a coin flip, same protocol as benchmark.py.
"""

import csv
import importlib.util
import itertools
import os

import torch
import triton

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "autotune-results.csv")


def load(num):
    spec = importlib.util.spec_from_file_location(f"k{num}", os.path.join(REPO, f"{num}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


k02, k03, k15, k16, k21 = (load(n) for n in
                           ("02-layer-norm", "03-flash-attention", "15-causal-attention",
                            "16-flash-decoding", "21-flash-attention-bwd"))
sdpa = torch.nn.functional.scaled_dot_product_attention


def bench(fn, iters=10, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        for _ in range(5):
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


rows = []  # dicts: kernel, shape, *config, ms, tflops_or_gbs, default_ms


def sweep_attn_fwd():
    print("== flash attention forward (03) ==")
    for Z, H, S, D in [(2, 16, 1024, 128), (2, 16, 2048, 128), (2, 16, 4096, 128),
                       (2, 16, 8192, 128), (2, 32, 4096, 64)]:
        q, k, v = (torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5
                   for _ in range(3))
        flops = 4 * Z * H * S * S * D
        shape = f"{Z}x{H}x{S}x{D}"
        bns = [32, 64, 128] if D >= 128 else [32, 64]
        space = list(itertools.product([64, 128], bns, [4, 8], [2, 3, 4]))
        best = None
        for bm, bn, w, st in space:
            if bm * bn > 128 * 128:      # register pressure floor on 6GB laptop
                continue
            o = torch.empty_like(q)
            M = torch.empty((Z, H, S), device="cuda", dtype=torch.float32)

            def run():
                k03._attn_fwd[(triton.cdiv(S, bm), Z * H)](
                    q, k, v, D ** -0.5, M, o,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    Z, H, S,
                    BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, num_warps=w, num_stages=st,
                )

            try:
                ms = bench(run)
            except triton.OutOfResources:
                continue
            tf = flops / (ms * 1e-3) / 1e12
            rows.append(dict(kernel="attn-fwd", shape=shape, BLOCK_M=bm, BLOCK_N=bn,
                             num_warps=w, num_stages=st, ms=f"{ms:.4f}",
                             throughput=f"{tf:.1f} TF"))
            if best is None or ms < best[0]:
                best = (ms, (bm, bn, w, st), tf)
        ms, (bm, bn, w, st), tf = best
        default_ms = next(r for r in rows if r["kernel"] == "attn-fwd"
                          and r["shape"] == shape
                          and (r["BLOCK_M"], r["BLOCK_N"], r["num_warps"], r["num_stages"])
                          == (128, 64, 8, 3))
        print(f"  {shape:>18}: best BM{bm}/BN{bn}/w{w}/s{st} "
              f"{ms:.3f} ms ({tf:.1f} TF), default 128/64/8/3 {float(default_ms['ms']):.3f} ms "
              f"({ms / float(default_ms['ms']):.2f}x)")


def sweep_attn_bwd():
    print("== flash attention backward (21) ==")
    for Z, H, S, D in [(2, 8, 1024, 128), (2, 8, 4096, 128)]:
        q, k, v = (torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5
                   for _ in range(3))
        o = sdpa(q, k, v)
        do = torch.randn_like(q)
        # the bwd needs the forward's lse; SDPA doesn't hand it over, so
        # recompute in torch — a test-only cost, the real op saves it (file 25)
        s = (q.float() @ k.float().transpose(-1, -2)) * D ** -0.5
        lse = torch.logsumexp(s, -1) * 1.44269504
        qf, kf, vf, of, dof = (x.reshape(Z * H, S, D) for x in (q, k, v, o, do))
        delta = (of.float() * dof.float()).sum(-1)
        dq = torch.empty_like(qf, dtype=torch.float32)
        dk = torch.empty_like(kf, dtype=torch.float32)
        dv = torch.empty_like(vf, dtype=torch.float32)
        flops = 4 * 2.5 * Z * H * S * S * D   # bwd ≈ 2.5x fwd flops
        shape = f"{Z}x{H}x{S}x{D}"
        best = None
        for bm, bn, w, st in itertools.product([64, 128], [64, 128], [4, 8], [2, 3]):
            def run():
                k21._attn_bwd_dkdv[(triton.cdiv(S, bn), Z * H)](
                    qf, kf, vf, dof, lse, delta, dk, dv, D ** -0.5, S,
                    BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, num_warps=w, num_stages=st)
                k21._attn_bwd_dq[(triton.cdiv(S, bm), Z * H)](
                    qf, kf, vf, dof, lse, delta, dq, D ** -0.5, S,
                    BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, num_warps=w, num_stages=st)

            try:
                ms = bench(run)
            except triton.OutOfResources:
                continue
            tf = flops / (ms * 1e-3) / 1e12
            rows.append(dict(kernel="attn-bwd", shape=shape, BLOCK_M=bm, BLOCK_N=bn,
                             num_warps=w, num_stages=st, ms=f"{ms:.4f}",
                             throughput=f"{tf:.1f} TF"))
            if best is None or ms < best[0]:
                best = (ms, (bm, bn, w, st), tf)
        ms, (bm, bn, w, st), tf = best
        default_ms = next(r for r in rows if r["kernel"] == "attn-bwd"
                          and r["shape"] == shape
                          and (r["BLOCK_M"], r["BLOCK_N"], r["num_warps"], r["num_stages"])
                          == (64, 64, 4, 2))
        print(f"  {shape:>18}: best BM{bm}/BN{bn}/w{w}/s{st} "
              f"{ms:.3f} ms ({tf:.1f} TF), default 64/64/4/2 {float(default_ms['ms']):.3f} ms "
              f"({ms / float(default_ms['ms']):.2f}x)")


def sweep_causal():
    print("== causal flash attention (15) ==")
    for Z, H, S, D in [(2, 16, 4096, 128)]:
        q, k, v = (torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5
                   for _ in range(3))
        flops = 2 * 2 * Z * H * S * S * D   # causal: ~half the score blocks
        shape = f"{Z}x{H}x{S}x{D}-causal"
        best = None
        for bm, bn, w, st in itertools.product([64, 128], [64, 128], [4, 8], [2, 3]):
            o = torch.empty_like(q)

            def run():
                k15._causal_attn_fwd[(triton.cdiv(S, bm), Z * H)](
                    q, k, v, D ** -0.5, o,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    Z, H, S,
                    BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, num_warps=w, num_stages=st)

            try:
                ms = bench(run)
            except triton.OutOfResources:
                continue
            tf = flops / (ms * 1e-3) / 1e12
            rows.append(dict(kernel="attn-causal", shape=shape, BLOCK_M=bm, BLOCK_N=bn,
                             num_warps=w, num_stages=st, ms=f"{ms:.4f}",
                             throughput=f"{tf:.1f} TF"))
            if best is None or ms < best[0]:
                best = (ms, (bm, bn, w, st), tf)
        ms, (bm, bn, w, st), tf = best
        default_ms = next(r for r in rows if r["kernel"] == "attn-causal"
                          and (r["BLOCK_M"], r["BLOCK_N"], r["num_warps"], r["num_stages"])
                          == (128, 64, 8, 3))
        print(f"  {shape:>18}: best BM{bm}/BN{bn}/w{w}/s{st} "
              f"{ms:.3f} ms ({tf:.1f} TF), default 128/64/8/3 {float(default_ms['ms']):.3f} ms "
              f"({ms / float(default_ms['ms']):.2f}x)")


def sweep_decode():
    print("== flash-decoding (16) ==")
    # runs FIRST in __main__ on purpose: its k/v are ~1 GB each, and if the
    # attn sweeps have already filled the caching allocator with fragments,
    # those cudaMallocs can spill into shared memory over PCIe — every config
    # then times ~160 ms at bus speed instead of ~7 ms (SDPA too, it reads
    # the same k/v; WDDM doesn't hand freed blocks back mid-process, so
    # empty_cache won't rescue it).
    Z, H, S, D = 4, 32, 32768, 128
    q = torch.randn(Z, H, 1, D, device="cuda", dtype=torch.float16) * 0.5
    k, v = (torch.randn(Z, H, S, D, device="cuda", dtype=torch.float16) * 0.5 for _ in range(2))
    flops = 4 * Z * H * S * D
    kv_bytes = 2 * Z * H * S * D * 2   # K and V stream once, fp16
    shape = f"{Z}x{H}x{S}x{D}-decode"
    ms_sdpa = bench(lambda: sdpa(q, k, v))
    best = None
    # num_splits decides how many SMs each head's cache gets; block_n is the
    # inner loop step. chunk size adapts to num_splits inside flash_decode
    for ns, bn in itertools.product([4, 8, 16, 32], [64, 128, 256]):
        def run():
            k16.flash_decode(q, k, v, D ** -0.5, num_splits=ns, block_n=bn)

        try:
            ms = bench(run)
        except triton.OutOfResources:
            continue
        gbs = kv_bytes / (ms * 1e-3) / 1e9
        rows.append(dict(kernel="flash-decode", shape=shape, BLOCK_M=ns, BLOCK_N=bn,
                         num_warps="", num_stages="", ms=f"{ms:.4f}",
                         throughput=f"{gbs:.0f} GB/s"))
        if best is None or ms < best[0]:
            best = (ms, (ns, bn), gbs)
    ms, (ns, bn), gbs = best
    default_ms = next(r for r in rows if r["kernel"] == "flash-decode"
                      and r["BLOCK_M"] == 8 and r["BLOCK_N"] == 128)
    print(f"  {shape:>22}: best splits={ns}/bn{bn} {ms:.3f} ms ({gbs:.0f} GB/s, "
          f"SDPA {ms_sdpa:.3f} ms → {ms_sdpa / ms:.2f}x), "
          f"default 8/128 {float(default_ms['ms']):.3f} ms ({ms / float(default_ms['ms']):.2f}x)")


def sweep_layernorm():
    print("== layernorm forward (02) ==")
    # BLOCK_SIZE is forced to next_pow2(N) by the one-row-per-program design;
    # the only free knobs are num_warps (and stages does nothing without a loop)
    for M, N in [(4096, 1024), (4096, 4096), (1024, 16384)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.rand(N, device="cuda", dtype=torch.float16)
        b = torch.rand(N, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        mean = torch.empty(M, device="cuda", dtype=torch.float32)
        rstd = torch.empty(M, device="cuda", dtype=torch.float32)
        bs = triton.next_power_of_2(N)
        gbs_of = lambda ms: 3 * M * N * 2 / (ms * 1e-3) / 1e9   # read x, write y (+w/b noise)
        best = None
        for nw in [1, 2, 4, 8, 16]:
            def run():
                k02._layer_norm_fwd_kernel[(M,)](
                    x, y, w, b, mean, rstd,
                    x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                    N, 1e-5, BLOCK_SIZE=bs, num_warps=nw)

            try:
                ms = bench(run)
            except triton.OutOfResources:
                continue
            rows.append(dict(kernel="layernorm", shape=f"{M}x{N}", BLOCK_M=bs, BLOCK_N="",
                             num_warps=nw, num_stages="", ms=f"{ms:.4f}",
                             throughput=f"{gbs_of(ms):.0f} GB/s"))
            if best is None or ms < best[0]:
                best = (ms, nw)
        ms, nw = best
        current_nw = k02._num_warps(bs)
        default_ms = next(r for r in rows if r["kernel"] == "layernorm"
                          and r["shape"] == f"{M}x{N}" and r["num_warps"] == current_nw)
        print(f"  {M}x{N:>6}: best warps={nw} {ms:.3f} ms ({gbs_of(ms):.0f} GB/s), "
              f"current w={current_nw} {float(default_ms['ms']):.3f} ms "
              f"({ms / float(default_ms['ms']):.2f}x)")


def write_csv():
    with open(OUT, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=[
            "kernel", "shape", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages",
            "ms", "throughput"])
        wtr.writeheader()
        wtr.writerows(rows)


if __name__ == "__main__":
    torch.manual_seed(0)
    sweep_decode()      # first — needs the cleanest VRAM state (see its comment)
    sweep_attn_fwd()
    sweep_attn_bwd()
    sweep_causal()
    sweep_layernorm()
    write_csv()
    print(f"\nwrote {OUT} ({len(rows)} configs)")
