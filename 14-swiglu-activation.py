"""silu(g) * u，swiglu 的激活部分，LLaMA 系 FFN 用的。

内容不复杂，主要是练逐元素激活的手写反向：
ds/dg = sigmoid(g) * (1 + g * (1 - sigmoid(g)))，推对这一个，
其他激活函数的反向都是一个套路。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_fwd_kernel(G, U, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(G + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _silu_mul_bwd_kernel(G, U, DY, DG, DU, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(G + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    ds = sig * (1 + g * (1 - sig))       # d silu / d g
    tl.store(DG + offsets, (dy * u * ds).to(DG.dtype.element_ty), mask=mask)
    tl.store(DU + offsets, (dy * g * sig).to(DU.dtype.element_ty), mask=mask)


class SiluMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, g, u):
        y = torch.empty_like(g)
        n = g.numel()
        BLOCK = 1024
        _silu_mul_fwd_kernel[(triton.cdiv(n, BLOCK),)](g, u, y, n, BLOCK=BLOCK)
        ctx.save_for_backward(g, u)
        return y

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        n = g.numel()
        BLOCK = 1024
        _silu_mul_bwd_kernel[(triton.cdiv(n, BLOCK),)](g, u, dy.contiguous(), dg, du, n, BLOCK=BLOCK)
        return dg, du


silu_mul = SiluMul.apply


if __name__ == "__main__":
    torch.manual_seed(0)
    g = torch.randn(1 << 22, device='cuda', dtype=torch.float16, requires_grad=True)
    u = torch.randn(1 << 22, device='cuda', dtype=torch.float16, requires_grad=True)

    y = silu_mul(g, u)
    y_ref = torch.nn.functional.silu(g) * u
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)
    print(f"前向最大误差 = {(y - y_ref).abs().max().item():.2e}")

    dy = torch.randn_like(y)
    y_ref.backward(dy)
    g_ref, u_ref = g.grad.clone(), u.grad.clone()
    g.grad = u.grad = None
    y.backward(dy)
    torch.testing.assert_close(g.grad.float(), g_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(u.grad.float(), u_ref.float(), atol=1e-2, rtol=1e-2)
    print(f"反向最大误差 = {(g.grad.float() - g_ref.float()).abs().max().item():.2e}")
    print("✅ silu_mul 正反传播正确")
