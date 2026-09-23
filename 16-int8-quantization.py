"""int8 per-row 动态量化（对称，absmax/127）。

对称量化误差上界是 absmax/254，测试里对照的就是这个界。
per-tensor 的 scale 最省但精度差，per-row 是个折中，SmoothQuant 那类
做法的 block 粒度也就是把这里的"行"换成更小的块。
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
    q = tl.extra.cuda.libdevice.round(x * scale)   # 用 libdevice 的 round（banker's rounding）
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(Q + row * stride_q + cols, q, mask=mask)
    tl.store(SCALE + row, 1.0 / scale)             # 存 dequant 用的 scale（absmax/127）


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

    # 与 PyTorch 参考实现对照
    ref_scale = 127.0 / x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    q_ref = (x.float() * ref_scale).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(q.to(torch.int32), q_ref.to(torch.int32), atol=1, rtol=0)
    # 对称量化的误差上界是每行 absmax/254；用绝对误差对照这个界，
    # 相对误差在 x≈0 的元素上没有意义（0 附近必然 ~100%）
    absmax = x.float().abs().amax(dim=1, keepdim=True)
    err = (x_hat.float() - x.float()).abs()
    bound = (absmax / 254).expand_as(err)
    print(f"最大绝对误差 = {err.max().item():.2e}，理论上界（absmax/254）= {bound.max().item():.2e}")
    # fp16 落盘的舍入会额外贡献 absmax/2^11 左右误差，放宽到界的 1.1 倍
    assert (err <= bound * 1.1 + 1e-3).all()
    print("✅ int8 动态量化正确性通过")
