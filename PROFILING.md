# Profiling notes — how close do these kernels actually get to the hardware?

Every kernel file prints its own numbers, but scattered TFLOPS don't answer
the real questions: *how close is this to the hardware's limit, and what
would it take to close the gap?* This file works through that per kernel,
using the numbers from `benchmark.py` (which also produces `roofline.png`,
`speedup.png`, and `perf-results.csv`).

## Method, and why the peaks are measured not quoted

Spec-sheet numbers for this GPU (RTX 3060 Laptop, 30 SMs, nominally 336 GB/s
GDDR6) are marketing numbers under ideal conditions. So `benchmark.py` measures
what's actually achievable:

- **bandwidth**: a 512 MiB device-to-device copy (counts read + write) and a
  big `sum()` reduction (pure read); whichever is higher. Measured:
  ~300–320 GB/s, i.e. ~92–95% of spec. A `copy_()` probe alone lands near
  270 GB/s and would make every later "% of peak" claim look inflated.
- **compute**: a big cuBLAS fp16 matmul (8192³) as *practical* peak:
  ~25–28 TFLOP/s. (fp32 accumulation; this GPU does more with fp16
  accumulate but cuBLAS is the honest target since it's what ships.)
- **timing**: min-of-5 repeated blocks of 30 iters. This is a laptop GPU —
  clocks swing with thermals and whatever else the machine is doing, and a
  single timed block can be off by 20%. Run-to-run variance on the same
  binary is still ±10–15%; treat all numbers here as ballpark, not law.

With ~300 GB/s and ~26 TFLOP/s the **ridge point sits around 85 FLOP/byte** —
kernels left of that are bandwidth-bound, right of it compute-bound.

## Per-kernel read of the roofline

**06 matmul, 4096³ (AI ≈ 1365 FLOP/B).** Deep in the compute region, so the
only thing that matters is MMA utilization. Measured 0.8–1.0x cuBLAS
(~20–27 TFLOP/s). The gap to peak comes from the usual suspects: no warp
specialization, epilogue traffic through registers, and autotune landing on
good-but-not-perfect tile configs. Getting the last 10–20% is exactly the
kind of grind that separates a solid kernel from a great one.

**08 flash attention forward, 4k context (AI ≈ 2048 FLOP/B).** The N² score
matrix never touches HBM — that's the entire trick — so the intensity math
counts only the four N·D streaming tensors: AI = 4N²D / 8ND = N/2. At 4k
context that's 2048 FLOP/B, hard compute-bound. It lands within noise of
SDPA (0.97–1.08x), which on a laptop GPU means the online-softmax mainloop
is squeezing the tensor cores about as hard as the vendored kernel.

**21 flash decoding, 32k cache (AI ≈ 1 FLOP/B).** Decode is the opposite
corner: one query row per head, the whole K and V cache streamed once, so
AI = 4BHSD FLOPs / 4BHSD bytes = 1.0 — three orders of magnitude left of the
ridge. Nothing but bandwidth matters. The split-KV scheme gets 270–317 GB/s,
**~90–98% of the measured copy peak**, and 1.7–1.9x over SDPA — SDPA's decode
path doesn't split the sequence across SMs, so most of the GPU idles while a
few blocks grind through 32k keys. This is the kernel in the repo I'd put in
front of an interviewer first.

**09 transpose, 8192² (AI ≈ 0).** Moves 2n² bytes, computes nothing. The
whole game is coalescing: 272–307 GB/s (~90–95% of peak) vs torch's
`.T.contiguous()` at ~97 GB/s. The 3x gap is uncoalesced 2-byte accesses in
torch's kernel for this shape; a tiled transpose with coalesced reads on
both sides is close to free performance.

**10 histogram, 8 hot bins.** Not on the roofline chart (no FLOPs to speak
of) but it's a bandwidth-adjacency story: the naive version does one global
atomic per element and collapses to ~18 GB/s — every warp in the GPU
serializing on the same 8 addresses — while the register-privatized version
hits ~280 GB/s. A 16x speedup from a data-structure change, zero FLOPs
involved. Atomics throughput is not bandwidth.

**28 int8 W8A8 GEMM, 2048³ (AI ≈ 1024 FLOP/B).** Compute-bound, and the
chart shows it *above* the fp16 roofline — because there are two rooflines:
int8 MMA rate is 2x fp16 on Ampere. Measured 38–48 TOPS vs the same-shape
cuBLAS fp16 at 15–17 TFLOP/s. The dequant scales add one FMA per output
element, which is noise at this arithmetic intensity.

## What I'd do next with a profiler budget

These numbers come from event timing + arithmetic; the next level of honesty
needs Nsight Compute, which doesn't run under WSL2 on this setup. On native
Linux I'd want: achieved SM busy vs memory busy for 06 (is the last 20% of
cuBLAS gap mainloop or epilogue?), `l1tex__data_pipe_lsu_wavefronts_mem_shared`
for the transpose tiles, and atomic throughput counters on the naive
histogram to see the serialization directly.

## Reproduce

```
python benchmark.py        # writes perf-results.csv, roofline.png, speedup.png
```
