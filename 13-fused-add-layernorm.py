"""add + layernorm 融合，transformer 每层跑两次的那种（参考 apex 的 FusedLayerNorm）。

比 07 号多了两个输入和一个 residual_out 输出（残差流要落盘给下一层用），
融合收益随输入个数涨，实测比 eager 快 1.5 倍。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_kernel(
        X, RES, Y, RES_OUT, W, B,
        stride_x, stride_res, stride_y, stride_res_out,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RES + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
    z = x + res

    # residual_out 原样落盘（保持输入 dtype），下一个 block 还要用它
    tl.store(RES_OUT + row * stride_res_out + cols, z.to(RES_OUT.dtype.element_ty), mask=mask)

    mean = tl.sum(z, axis=0) / N
    var = tl.sum(z * z, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = (z - mean) * rstd * w + b
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def fused_add_layernorm(x, residual, weight, bias, eps=1e-5):
    x, residual = x.contiguous(), residual.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    res_out = torch.empty_like(residual)
    _add_layernorm_kernel[(M,)](
        x, residual, y, res_out, weight, bias,
        x.stride(0), residual.stride(0), y.stride(0), res_out.stride(0),
        N, eps, BLOCK_SIZE=triton.next_power_of_2(N), num_warps=8,
    )
    return y, res_out


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 4096
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    res = torch.randn((M, N), device='cuda', dtype=torch.float16)
    w = torch.rand(N, device='cuda', dtype=torch.float16)
    b = torch.rand(N, device='cuda', dtype=torch.float16)

    y, res_out = fused_add_layernorm(x, res, w, b)
    z_ref = x + res
    y_ref = torch.nn.functional.layer_norm(z_ref, (N,), w, b, 1e-5)
    torch.testing.assert_close(res_out, z_ref)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=0)
    print("✅ fused add+layernorm 正确性通过")

    def bench(fn, iters=100):
        for _ in range(10): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1e3

    t_fused = bench(lambda: fused_add_layernorm(x, res, w, b))
    def eager():
        z = x + res
        torch.nn.functional.layer_norm(z, (N,), w, b, 1e-5)
        return z
    t_eager = bench(eager)
    print(f"eager {t_eager:.1f} us   fused {t_fused:.1f} us   加速比 {t_eager/t_fused:.1f}x")
