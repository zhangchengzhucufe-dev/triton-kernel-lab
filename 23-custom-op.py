"""把 Triton kernel 注册成 torch 自定义算子，接进 torch.compile。

光会写 kernel 不够，工程上要能塞进 PyTorch 的体系里，不然 autograd /
torch.compile / export 都不带玩。标准姿势是 torch.library.custom_op：

- custom_op 注册前向，需要带 schema 注明哪些参数是 Tensor；
- register_fake 提供元信息（只描述 shape/dtype，不真算），
  torch.compile 靠它做 trace，不真起 kernel；
- autograd 接 ConnectTheDots？不用，直接包一层 autograd.Function 再
  注册成 custom_op 有点绕，这里演示"前向 custom op + 手动 backward"
  最常见的接法：custom_op 的本体不管梯度，梯度走外面那个 Function。

踩坑：custom_op 的函数参数名不能叫 'x_' 之类带下划线结尾的，
schema 解析会挂；Tensor 返回必须新建，不能原地改输入（会静默出错）。
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


# 注册成 "myext::silu"。mutates_args=[] 表示不原地改输入
@torch.library.custom_op("myext::silu", mutates_args=())
def triton_silu(x: torch.Tensor) -> torch.Tensor:
    return _silu_impl(x.contiguous())


@triton_silu.register_fake
def _(x):
    # compile 做静态 trace 时调用：只要 shape/dtype 对就行，不跑 kernel
    return torch.empty_like(x)


class SiluAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return triton_silu(x)

    @staticmethod
    def backward(ctx, dy):
        # dsilu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        # 注意别背成 sigmoid(x)*(1+x*sigmoid(x))，sigmoid 的导数是 sig*(1-sig)
        (x,) = ctx.saved_tensors
        xf = x.float()
        sig = torch.sigmoid(xf)
        return (dy.float() * sig * (1 + xf * (1 - sig))).to(x.dtype)


def silu_with_grad(x):
    return SiluAutograd.apply(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1 << 20, device='cuda', dtype=torch.float16, requires_grad=True)

    # 1. 直接调用
    y = triton_silu(x.detach())
    y_ref = torch.nn.functional.silu(x.detach())
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)
    print("✅ custom op 前向正确")

    # 2. autograd（注意对照的 x2 要复制同一份数据，否则梯度过的是不同输入）
    x.grad = None
    silu_with_grad(x).sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    torch.nn.functional.silu(x2).sum().backward()
    torch.testing.assert_close(x.grad.float(), x2.grad.float(), atol=1e-2, rtol=1e-2)
    print("✅ autograd 正确")

    # 3. 塞进 torch.compile（验证 register_fake 起作用）
    @torch.compile
    def compiled(a):
        return triton_silu(a) * 2

    out = compiled(x.detach())
    torch.testing.assert_close(out, y_ref * 2, atol=1e-2, rtol=0)
    print("✅ torch.compile trace 通过")
