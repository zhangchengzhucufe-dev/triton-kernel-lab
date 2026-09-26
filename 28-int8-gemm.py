"""int8 GEMM with per-row / per-column scales — SmoothQuant-style W8A8.

Quantizing both operands to int8 halves the memory traffic and, on Ampere,
doubles the tensor-core throughput vs fp16. The catch is you need scales:
activations get one scale per row (per token), weights one per column (per
output channel), both symmetric around zero. The kernel then does a plain
int8×int8 dot with int32 accumulation — exact, no rounding in the accumulate —
and multiplies the two scales into the fp32 result at the end:

    C = (A_int @ W_int) * sa[:, None] * sw[None, :]

Because the per-row/per-col scales absorb the outliers, the accuracy loss is
a few tenths of a percent, not the garbage you'd get from one global scale.

Honest caveats, the kind an interviewer will poke at:
- quantizing the activations costs a full pass over A every call (it changes
  every forward). This file prices only the GEMM; W8A8 only pays off when
  that pass is fused into the previous op, or when you're bandwidth-bound.
- symmetric int8 uses [-127, 127] and throws away -128, which keeps the
  zero point exact at 0.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_gemm(
        A, B, SA, SW, C,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros([BM, BN], dtype=tl.int32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs)     # int8
        b = tl.load(b_ptrs)     # int8
        acc = tl.dot(a, b, acc)  # int8 inputs -> int32 accumulate, exact
        a_ptrs += BK
        b_ptrs += BK * N

    # dequant: the scales were folded out of the integer math entirely
    c = acc.to(tl.float32) * tl.load(SA + offs_m)[:, None] * tl.load(SW + offs_n)[None, :]
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c.to(C.dtype.element_ty))


def int8_gemm(a_int8, b_int8, sa, sw, BM=128, BN=128, BK=64):
    M, K = a_int8.shape
    _, N = b_int8.shape
    c = torch.empty((M, N), device=a_int8.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _int8_gemm[grid](a_int8, b_int8, sa, sw, c, M, N, K,
                     BM=BM, BN=BN, BK=BK, num_warps=8, num_stages=3)
    return c


def quant_symmetric(x, dim):
    # one scale per row (dim=1) or column (dim=0); -128 is dropped so that
    # zero maps to zero exactly
    scale = x.abs().amax(dim=dim, keepdim=True) / 127.0
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    a_int8, sa = quant_symmetric(a, dim=1)
    b_int8, sw = quant_symmetric(b, dim=0)
    c = int8_gemm(a_int8, b_int8, sa.float(), sw.float())
    c_ref = (a.float() @ b.float())

    err = (c.float() - c_ref).abs()
    rel = err.max() / c_ref.abs().max()
    print(f"max error = {err.max().item():.2e} "
          f"({rel.item() * 100:.2f}% of the output's dynamic range)")
    assert rel < 0.02, "quantization error blew past 2%"
    print("✅ int8 gemm accuracy passed")

    def benchmark(fn, iters=20):
        fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    ms_i8 = benchmark(lambda: int8_gemm(a_int8, b_int8, sa.float(), sw.float()))
    ms_fp16 = benchmark(lambda: torch.matmul(a, b))
    tflops = lambda t: 2 * M * N * K / (t * 1e-3) / 1e12
    print(f"int8:  {ms_i8:.3f} ms ({tflops(ms_i8):.1f} TOPS)")
    print(f"fp16:  {ms_fp16:.3f} ms ({tflops(ms_fp16):.1f} TFLOPS)")
