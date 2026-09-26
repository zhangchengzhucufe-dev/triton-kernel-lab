"""RMSNorm, the kind used by LLaMA/Qwen (LayerNorm from ex. 07 minus the
mean subtraction and the bias).

y = x / rms(x) * w,  rms = sqrt(mean(x²) + eps)

Backward derived by hand: r = 1/sqrt(mean(x²)+eps), dr/dx_j = -r³·x_j/N,
so dx = r·(dy·w) - r³·x·sum(dy·w·x)/N. Numerically verified against
torch's nn.RMSNorm (2.4+).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(X, Y, W, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    x_hat = x / rms
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (x_hat * w).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd(X, DY, DX, DW, W, stride_x, stride_dy, stride_dx, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    r = 1.0 / rms
    dw_row = dy * x * r
    tl.atomic_add(DW + cols, dw_row, mask=mask)

    s = tl.sum(dy * w * x, axis=0) / N
    dx = r * dy * w - (r * r * r) * x * s
    tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)


class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        x = x.contiguous()
        M, N = x.shape
        y = torch.empty_like(x)
        _rmsnorm_fwd[(M,)](x, y, weight, x.stride(0), y.stride(0), N, eps,
                           BLOCK=triton.next_power_of_2(N), num_warps=8)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        M, N = x.shape
        dx = torch.empty_like(x)
        dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
        _rmsnorm_bwd[(M,)](x, dy.contiguous(), dx, dw, w,
                           x.stride(0), dy.stride(0), dx.stride(0), N, ctx.eps,
                           BLOCK=triton.next_power_of_2(N), num_warps=8)
        return dx, dw.to(w.dtype), None


rmsnorm = RMSNorm.apply

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 4096
    x = torch.randn((M, N), device='cuda', dtype=torch.float16, requires_grad=True)
    w = torch.rand(N, device='cuda', dtype=torch.float16, requires_grad=True)

    y = rmsnorm(x, w)
    ref = torch.nn.functional.rms_norm(x, (N,), w)
    print(f"forward max error = {(y - ref).abs().max().item():.2e}")
    torch.testing.assert_close(y, ref, atol=1e-2, rtol=0)

    dy = torch.randn_like(y)
    ref.backward(dy)
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    y.backward(dy)
    print(f"dX max error = {(x.grad.float() - gx.float()).abs().max().item():.2e}")
    print(f"dW max error = {(w.grad.float() - gw.float()).abs().max().item():.2e}")
    torch.testing.assert_close(x.grad.float(), gx.float(), atol=1e-2, rtol=1e-2)
    print("✅ rmsnorm forward and backward passed")
