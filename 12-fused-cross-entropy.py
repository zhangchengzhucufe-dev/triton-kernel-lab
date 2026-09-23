"""融合交叉熵，流式 logsumexp，前向反向都不物化 (N, V) 中间矩阵。

loss = logsumexp(logits) - logits[target]，logsumexp 可以像 flash
attention 的 online softmax 一样分块流式算。eager 写法要先存一份
log_softmax 的 fp32 中间张量，vocab 大的时候 HBM 流量很夸张，
这里 8192x32K 快了 3 倍多。反向就是 softmax - onehot。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_fwd_kernel(
        logits_ptr, targets_ptr, loss_ptr,
        stride_lm,
        N_COLS,
        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    logits_ptr += row * stride_lm
    target = tl.load(targets_ptr + row)

    # --- 第一遍：流式 logsumexp（和 flash attention 的在线 softmax 同构）---
    m = -float("inf")
    l = 0.0
    for off in tl.range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(logits_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(l)

    # --- 第二遍：取目标 logit，loss = logsumexp - x[target] ---
    target_logit = tl.load(logits_ptr + target).to(tl.float32)
    tl.store(loss_ptr + row, lse - target_logit)


@triton.jit
def cross_entropy_bwd_kernel(
        logits_ptr, targets_ptr, grad_logits_ptr, grad_loss_ptr,
        stride_lm, stride_gm,
        N_COLS,
        BLOCK_SIZE: tl.constexpr,
):
    # dL/dlogits_i = (softmax(logits)_i - onehot(target)) * dL/dloss
    row = tl.program_id(0)
    logits_ptr += row * stride_lm
    grad_logits_ptr += row * stride_gm
    target = tl.load(targets_ptr + row)
    dloss = tl.load(grad_loss_ptr + row)      # 上游梯度（标量），可做 loss 缩放

    m = -float("inf")
    l = 0.0
    for off in tl.range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + cols, mask=cols < N_COLS, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(l)

    for off in tl.range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(logits_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        softmax = tl.exp(x - lse)
        grad = (softmax - (cols == target).to(tl.float32)) * dloss
        tl.store(grad_logits_ptr + cols, grad, mask=mask)


class CrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, targets):
        N, V = logits.shape
        loss = torch.empty((N,), device=logits.device, dtype=torch.float32)
        ctx.BLOCK = 1024
        cross_entropy_fwd_kernel[(N,)](logits, targets, loss, logits.stride(0), V,
                                       BLOCK_SIZE=ctx.BLOCK)
        ctx.save_for_backward(logits, targets)
        return loss

    @staticmethod
    def backward(ctx, dloss):
        logits, targets = ctx.saved_tensors
        N, V = logits.shape
        g = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
        cross_entropy_bwd_kernel[(N,)](logits, targets, g, dloss.contiguous(),
                                       logits.stride(0), g.stride(0), V, BLOCK_SIZE=ctx.BLOCK)
        return g.to(logits.dtype), None


fused_ce = CrossEntropy.apply


def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters): fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


if __name__ == "__main__":
    torch.manual_seed(0)
    N, V = 8192, 32768          # batch 8192 × vocab 32K，模拟真实 LM 头
    logits = (torch.randn((N, V), device='cuda', dtype=torch.float32) * 3).requires_grad_(True)
    targets = torch.randint(0, V, (N,), device='cuda')

    # ---- 正确性 ----
    loss = fused_ce(logits, targets)
    ref = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    print(f"loss 最大误差 = {(loss - ref).abs().max().item():.2e}")
    torch.testing.assert_close(loss, ref, atol=1e-3, rtol=1e-3)

    grad_wrt_logits = torch.randn_like(ref) * 1e-2
    logits.grad = None
    ref.backward(grad_wrt_logits)
    ref_grad = logits.grad.clone()
    logits.grad = None
    loss.backward(grad_wrt_logits)
    print(f"grad 最大误差 = {(logits.grad.float() - ref_grad).abs().max().item():.2e}")
    torch.testing.assert_close(logits.grad.float(), ref_grad, atol=1e-3, rtol=1e-2)
    print("✅ fused cross entropy 正反传播正确")

    # ---- 性能：融合版 vs torch eager（log_softmax 物化中间张量）----
    logits2 = logits.detach().clone().requires_grad_(True)
    def eager():
        l = torch.nn.functional.cross_entropy(logits2, targets)
        l.sum().backward()
    def fused():
        l = fused_ce(logits.detach().clone().requires_grad_(True), targets)
        l.sum().backward()
    t_eager = bench(eager)
    t_fused = bench(fused)
    print(f"eager {t_eager:.1f} ms   fused {t_fused:.1f} ms   加速比 {t_eager/t_fused:.1f}x")
