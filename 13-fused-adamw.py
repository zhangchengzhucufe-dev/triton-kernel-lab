"""Fused AdamW. The eager optimizer launches a dozen tiny kernels per step; with
large parameter counts, just the p/m/v/g reads and writes are tens of GB of HBM
traffic — pure bandwidth-bound, so fusion pays off the most.

Note weight decay is decoupled (subtracted straight from p, never into m/v), and
the 1-β^t bias-correction factors are scalars computed on the host and passed in —
no need to recompute them in every program.
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

    # Bias correction computed on the host and passed in (1-β^t is a scalar; no need for 4096 programs to each compute it)
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    update = m_hat / (tl.sqrt(v_hat) + eps)

    # Decoupled weight decay: subtract straight from p, never into m/v or the gradient
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

    # Triton version
    p_t, m_t, v_t = p.clone(), m.clone(), v.clone()
    for step in range(1, 6):
        adamw_step(p_t, g, m_t, v_t, step)

    # PyTorch AdamW reference (same weight_decay semantics)
    ref = torch.optim.AdamW([torch.nn.Parameter(p.clone())], lr=1e-3, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)
    p_ref = ref.param_groups[0]['params'][0]
    for step in range(1, 6):
        ref.param_groups[0]['params'][0].grad = g.clone()
        ref.step()

    err = (p_t - p_ref).abs().max().item()
    print(f"max deviation from torch.optim.AdamW after 5 AdamW steps = {err:.2e}")
    assert err < 1e-5
    print("✅ fused AdamW correctness passed")

    # Bandwidth view: each step reads p/m/v/g at 4B each and writes p/m/v at 4B each — 28B per element
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
    print(f"{t:.2f} ms per step → single kernel sustaining {n * 28 / (t * 1e-3) / 1e9:.0f} GB/s of HBM traffic"
          f" (pre-fusion, that traffic was spread across a dozen kernels)")
