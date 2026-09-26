"""Register a Triton kernel as a torch custom op and wire it into torch.compile.

Writing a kernel isn't enough; in practice it needs to plug into PyTorch's
ecosystem, otherwise autograd / torch.compile / export won't cooperate. The
standard approach is torch.library.custom_op:

- custom_op registers the forward and needs a schema noting which params are Tensors;
- register_fake provides metadata (describes shape/dtype only, no real compute);
  torch.compile relies on it for tracing and never launches the kernel;
- Connect autograd to it? No — wrapping in autograd.Function and registering
  that as a custom op is convoluted. This demo shows the most common wiring:
  "forward as a custom op + manual backward": the custom op itself knows
  nothing about gradients; gradients flow through the outer Function.

Pitfall: custom_op function parameter names can't end with an underscore
like 'x_' — schema parsing breaks; returned Tensors must be freshly allocated,
never mutated in place (that fails silently).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, (x * tl.sigmoid(x)).to(Y.dtype.element_ty), mask=mask)


def _silu_impl(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    _silu_kernel[(triton.cdiv(n, 4096),)](x, y, n, BLOCK=4096)
    return y


# Registered as "myext::silu". mutates_args=[] means inputs aren't mutated in place
@torch.library.custom_op("myext::silu", mutates_args=())
def triton_silu(x: torch.Tensor) -> torch.Tensor:
    return _silu_impl(x.contiguous())


@triton_silu.register_fake
def _(x):
    # Called during compile's static tracing: only shape/dtype must be right, no kernel launch
    return torch.empty_like(x)


class SiluAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return triton_silu(x)

    @staticmethod
    def backward(ctx, dy):
        # dsilu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        # Don't misremember it as sigmoid(x)*(1+x*sigmoid(x)); sigmoid's derivative is sig*(1-sig)
        (x,) = ctx.saved_tensors
        xf = x.float()
        sig = torch.sigmoid(xf)
        return (dy.float() * sig * (1 + xf * (1 - sig))).to(x.dtype)


def silu_with_grad(x):
    return SiluAutograd.apply(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1 << 20, device='cuda', dtype=torch.float16, requires_grad=True)

    # 1. Direct call
    y = triton_silu(x.detach())
    y_ref = torch.nn.functional.silu(x.detach())
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)
    print("✅ custom op forward correct")

    # 2. autograd (the reference x2 must be a copy of the same data, otherwise
    # the gradient flows through a different input)
    x.grad = None
    silu_with_grad(x).sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    torch.nn.functional.silu(x2).sum().backward()
    torch.testing.assert_close(x.grad.float(), x2.grad.float(), atol=1e-2, rtol=1e-2)
    print("✅ autograd correct")

    # 3. Plug into torch.compile (verifies register_fake works)
    @torch.compile
    def compiled(a):
        return triton_silu(a) * 2

    out = compiled(x.detach())
    torch.testing.assert_close(out, y_ref * 2, atol=1e-2, rtol=0)
    print("✅ torch.compile trace passed")
