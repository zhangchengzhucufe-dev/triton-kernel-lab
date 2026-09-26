"""Same RMSNorm, three ways: hand-rolled CUDA C++, Triton (file 22), torch.

The point of this file is the CUDA C++ column — everything else in the repo
is Triton, and kernel interviews tend to drift toward "ok, now without the
DSL". The .cu kernel (cuda/rmsnorm.cu) does one block per row with a
warp-shuffle + shared-memory two-level reduction, in two flavors: scalar
half loads and float4-vectorized loads (8 halves per transaction).

The comparison answers a question I got wrong on paper first: how much does
float4 vectorization buy over scalar half loads? I assumed "a lot" (2-byte
loads sound wasteful), but measured they come out the same — a warp's 32
consecutive halves coalesce into the same 64-byte transactions a float4
version issues. Vectorization pays off when accesses *aren't* already
coalesced; when they are, the memory system does this for you. Triton still
wins overall (its autotuner picks a better threads-per-row mapping).

First build takes a minute or two; torch caches it in cuda/build after that.
"""

import importlib.util
import os
import sys

import torch

# torch's ninja check shells out to `ninja`, which only resolves if the
# venv's bin dir is on PATH — running this file with an absolute interpreter
# path skips that. put it back before cpp_extension notices
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_rms = load(name="rmsnorm_ext", sources=[os.path.join(HERE, "cuda", "rmsnorm.cu")],
            extra_cuda_cflags=["-O3"], verbose=False,
            build_directory=os.path.join(HERE, "cuda", "build"))

spec = importlib.util.spec_from_file_location("k22", os.path.join(HERE, "22-rmsnorm.py"))
k22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k22)


def cuda_rmsnorm(x, w, eps=1e-6, force_scalar=False):
    out = torch.empty_like(x)
    _rms.rmsnorm(x, w, out, eps, force_scalar)
    return out


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 4096, 4096
    x = torch.randn(B, D, device="cuda", dtype=torch.float16)
    w = torch.randn(D, device="cuda", dtype=torch.float16)

    out = cuda_rmsnorm(x, w)
    ref = torch.nn.functional.rms_norm(x, (D,), w)
    print(f"max error = {(out - ref).abs().max().item():.2e}")
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)
    print("✅ cuda rmsnorm correctness passed")

    ms_scalar = bench(lambda: cuda_rmsnorm(x, w, force_scalar=True))
    ms_cuda = bench(lambda: cuda_rmsnorm(x, w))
    ms_triton = bench(lambda: k22.rmsnorm(x, w))
    ms_torch = bench(lambda: torch.nn.functional.rms_norm(x, (D,), w))
    gbs = lambda t: 2 * x.numel() * 2 / (t * 1e-3) / 1e9
    print(f"cuda scalar     : {ms_scalar:6.3f} ms ({gbs(ms_scalar):5.0f} GB/s)")
    print(f"cuda vectorized : {ms_cuda:6.3f} ms ({gbs(ms_cuda):5.0f} GB/s)")
    print(f"triton (file 22): {ms_triton:6.3f} ms ({gbs(ms_triton):5.0f} GB/s)")
    print(f"torch rms_norm  : {ms_torch:6.3f} ms ({gbs(ms_torch):5.0f} GB/s)")
