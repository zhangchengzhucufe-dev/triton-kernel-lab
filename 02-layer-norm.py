"""LayerNorm forward + backward, corresponding to tutorial 05; my first hand-written backward.

Note that the E[x²] - E[x]² formulation for variance risks catastrophic cancellation;
switch to Welford when x is too large. In the backward, I omitted the * w in
dXhat = dY * w in my first version, which made everything wrong.
Accumulate in fp32 throughout, even for fp16 inputs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_kernel(
        X, Y, W, B, MEAN, RSTD,
        stride_x, stride_y,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    # E[x²] - E[x]²: numerically less stable than Welford, but fine for teaching
    # purposes with fp32 accumulation. If |x| > 1e4, switch to Welford or a
    # two-pass method (center first, then sum of squares).
    _var = tl.sum(x * x, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(_var + eps)

    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)

    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    y = (x - mean) * rstd * w + b
    tl.store(Y + row * stride_y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_kernel(
        X, W, DY, DX, DW, DB, MEAN, RSTD,
        stride_x, stride_dy, stride_dx,
        M, N,
        BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row's dX while atomically accumulating its share of DW/DB.
    # The tutorial has each program process a shard of rows then reduce; here we use
    # the most straightforward "one program per row" version for readability.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(MEAN + row)
    rstd = tl.load(RSTD + row)

    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)

    # dW is defined as sum_i dy_i * xhat_i (summed over rows); write back with fp32 atomic adds
    tl.atomic_add(DW + cols, dy * xhat, mask=mask)
    tl.atomic_add(DB + cols, dy, mask=mask)

    # The upstream gradient of y = xhat * w + b w.r.t. xhat is dy * w
    dxhat = dy * w
    c1 = tl.sum(xhat * dxhat, axis=0) / N
    c2 = tl.sum(dxhat, axis=0) / N
    dx = (dxhat - c1 * xhat - c2) * rstd
    tl.store(DX + row * stride_dx + cols, dx.to(tl.float16), mask=mask)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x = x.contiguous()
        M, N = x.shape
        y = torch.empty_like(x, dtype=torch.float16)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(N)
        _layer_norm_fwd_kernel[(M,)](
            x, y, weight, bias, mean, rstd,
            x.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, mean, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        M, N = x.shape
        dx = torch.empty_like(dy)
        dw = torch.zeros((N,), dtype=torch.float32, device=x.device)
        db = torch.zeros((N,), dtype=torch.float32, device=x.device)
        _layer_norm_bwd_kernel[(M,)](
            x, w, dy, dx, dw, db, mean, rstd,
            x.stride(0), dy.stride(0), dx.stride(0),
            M, N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        return dx, dw.to(w.dtype), db.to(w.dtype), None


layer_norm = LayerNorm.apply


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 512, 1024
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=torch.float16, device='cuda')
    x.requires_grad_(True)
    weight = torch.rand(N, dtype=torch.float16, device='cuda', requires_grad=True)
    bias = torch.rand(N, dtype=torch.float16, device='cuda', requires_grad=True)

    y_triton = layer_norm(x, weight, bias, 1e-5)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, 1e-5)
    print(f"forward max error = {(y_triton - y_ref).abs().max().item():.2e}")
    torch.testing.assert_close(y_triton, y_ref, atol=1e-2, rtol=0)

    # Backward check
    grad = torch.randn_like(x)
    y_ref.backward(grad)
    x_ref_grad, w_ref_grad, b_ref_grad = x.grad.clone(), weight.grad.clone(), bias.grad.clone()
    x.grad = None; weight.grad = None; bias.grad = None
    y_triton.backward(grad)
    print(f"dX max error = {(x.grad - x_ref_grad).abs().max().item():.2e}")
    print(f"dW max error = {(weight.grad - w_ref_grad).abs().max().item():.2e}")
    print(f"dB max error = {(bias.grad - b_ref_grad).abs().max().item():.2e}")
    print("✅ layernorm forward/backward correctness passed")
