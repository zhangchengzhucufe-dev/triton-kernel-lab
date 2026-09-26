"""int8 per-row dynamic quantization (symmetric, absmax/127).

The error bound for symmetric quantization is absmax/254, which is exactly what
the test checks against. Per-tensor scale is cheapest but least accurate; per-row
is a middle ground, and block-granularity methods like SmoothQuant just shrink
the "row" here into smaller blocks.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_kernel(X, Q, SCALE, stride_x, stride_q, N_COLS,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    absmax = tl.max(tl.abs(x), axis=0)
    scale = 127.0 / tl.maximum(absmax, 1e-8)
    q = tl.extra.cuda.libdevice.round(x * scale)   # libdevice round (banker's rounding)
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(Q + row * stride_q + cols, q, mask=mask)
    tl.store(SCALE + row, 1.0 / scale)             # store the dequant scale (absmax/127)


@triton.jit
def _dequantize_kernel(Q, SCALE, X, stride_q, stride_x, N_COLS,
                       BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    q = tl.load(Q + row * stride_q + cols, mask=mask, other=0).to(tl.float32)
    scale = tl.load(SCALE + row)
    tl.store(X + row * stride_x + cols, (q * scale).to(X.dtype.element_ty), mask=mask)


def dynamic_quantize(x):
    x = x.contiguous()
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    scale = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    _quantize_kernel[(M,)](x, q, scale, x.stride(0), q.stride(0), N, BLOCK_SIZE=BLOCK)
    return q, scale


def dequantize(q, scale, dtype=torch.float16):
    M, N = q.shape
    x = torch.empty((M, N), device=q.device, dtype=dtype)
    _dequantize_kernel[(M,)](q, scale, x, q.stride(0), x.stride(0), N,
                             BLOCK_SIZE=triton.next_power_of_2(N))
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 4096
    x = torch.randn((M, N), device='cuda', dtype=torch.float16) * 3

    q, scale = dynamic_quantize(x)
    x_hat = dequantize(q, scale)

    # Compare against the PyTorch reference implementation
    ref_scale = 127.0 / x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    q_ref = (x.float() * ref_scale).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(q.to(torch.int32), q_ref.to(torch.int32), atol=1, rtol=0)
    # The error bound for symmetric quantization is per-row absmax/254; compare
    # absolute error against this bound, since relative error is meaningless for
    # elements with x≈0 (necessarily ~100% near zero)
    absmax = x.float().abs().amax(dim=1, keepdim=True)
    err = (x_hat.float() - x.float()).abs()
    bound = (absmax / 254).expand_as(err)
    print(f"max abs error = {err.max().item():.2e}, theoretical bound (absmax/254) = {bound.max().item():.2e}")
    # fp16 rounding on store adds roughly absmax/2^11 of extra error; loosen to 1.1x the bound
    assert (err <= bound * 1.1 + 1e-3).all()
    print("✅ int8 dynamic quantization correctness passed")
