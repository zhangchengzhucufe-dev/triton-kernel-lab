"""fused AdamW。eager 的优化器一步要起十几个小 kernel，参数量大的时候
光 p/m/v/g 的读写就是几十 GB 的 HBM 流量，纯带宽瓶颈，融合收益最大。

注意权重衰减是解耦的（直接从 p 扣，不进 m/v），偏置修正的 1-β^t 是
标量，在 host 算好传进来，没必要每个 program 算一遍。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _adamw_kernel(
        P, M, V, GRAD,
        n_elements,
        lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    p = tl.load(P + offsets, mask=mask, other=0.0)
    g = tl.load(GRAD + offsets, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(M + offsets, mask=mask, other=0.0)
    v = tl.load(V + offsets, mask=mask, other=0.0)

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    # 偏置修正在 host 端算好传进来（1-β^t 是标量，没必要 4096 个 program 各算一遍）
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    update = m_hat / (tl.sqrt(v_hat) + eps)

    # 解耦权重衰减：直接从 p 里扣，不进 m/v，也不进梯度
    p_new = p - lr * (update + weight_decay * p)

    tl.store(P + offsets, p_new, mask=mask)
    tl.store(M + offsets, m, mask=mask)
    tl.store(V + offsets, v, mask=mask)


def adamw_step(params, grads, exp_avg, exp_avg_sq, step,
               lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    n = params.numel()
    BLOCK = 4096
    bc1 = 1.0 - betas[0] ** step
    bc2 = 1.0 - betas[1] ** step
    _adamw_kernel[(triton.cdiv(n, BLOCK),)](
        params, exp_avg, exp_avg_sq, grads, n,
        lr, betas[0], betas[1], eps, weight_decay, bc1, bc2,
        BLOCK=BLOCK, num_warps=4,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 1 << 24
    p = torch.randn(n, device='cuda', dtype=torch.float32)
    g = torch.randn(n, device='cuda', dtype=torch.float16).to(torch.float32)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)

    # Triton 版
    p_t, m_t, v_t = p.clone(), m.clone(), v.clone()
    for step in range(1, 6):
        adamw_step(p_t, g, m_t, v_t, step)

    # PyTorch AdamW 参考（同样的 weight_decay 语义）
    ref = torch.optim.AdamW([torch.nn.Parameter(p.clone())], lr=1e-3, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)
    p_ref = ref.param_groups[0]['params'][0]
    for step in range(1, 6):
        ref.param_groups[0]['params'][0].grad = g.clone()
        ref.step()

    err = (p_t - p_ref).abs().max().item()
    print(f"5 步 AdamW 后与 torch.optim.AdamW 的最大偏差 = {err:.2e}")
    assert err < 1e-5
    print("✅ fused AdamW 正确性通过")

    # 带宽视角：每步读 p/m/v/g 各 4B，写 p/m/v 各 4B，共 28B/元素
    import time
    def bench(fn, iters=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    t = bench(lambda: adamw_step(p_t, g, m_t, v_t, 5))
    print(f"每步 {t:.2f} ms → 单 kernel 消化 {n * 28 / (t * 1e-3) / 1e9:.0f} GB/s 的 HBM 流量"
          f"（融合前这些流量要分摊到十几个 kernel）")
