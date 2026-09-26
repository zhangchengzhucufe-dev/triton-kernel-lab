"""Flash attention as a train-able op: file 08's forward + file 26's backward
glued into a torch.autograd.Function.

Files 08 and 26 each pass their own numerics checks, but "two kernels that
are individually correct" is not the same as "a kernel I can put in a model
and backprop through". The glue that makes it a real op:

- the forward's logsumexp (in log2 units, 08's convention) is exactly what
  the backward rebuilds P from, so it's saved on the ctx instead of recomputed
- backward returns grads in the input dtype — autograd requires backward
  outputs to match forward input dtypes
- constraints (N_CTX a multiple of the block sizes, no masking) are asserted
  once here rather than exploding inside a kernel

Verified end-to-end against SDPA's autograd, then timed: this whole thing
(one forward kernel + two backward kernels) vs SDPA's fused fwd+bwd.
"""

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(num):
    spec = importlib.util.spec_from_file_location(f"k{num}", os.path.join(HERE, f"{num}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


k08 = _load("08-flash-attention")
k26 = _load("26-flash-attention-bwd")
import triton  # noqa: E402  (after _load so import order reads top-down)


def _forward(q, k, v, sm_scale):
    """08's kernel, but keeping the logsumexp the backward needs. Same launch
    as k08.attention — that one allocates M and throws it away."""
    Z, H, N_CTX, HEAD_DIM = q.shape
    o = torch.empty_like(q)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)
    k08._attn_fwd[grid](
        q, k, v, sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
        num_warps=8, num_stages=3,
    )
    return o, M


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        N_CTX = q.shape[2]
        assert N_CTX % 128 == 0 and N_CTX % 64 == 0, \
            "26's backward assumes block-aligned seq len; pad before calling"
        o, lse = _forward(q, k, v, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = k26.attention_bwd(q, k, v, o, do, ctx.sm_scale, lse)
        # grads must come back in the inputs' dtype, not the fp32 the kernels
        # accumulated in
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def flash_attention(q, k, v, sm_scale):
    return FlashAttention.apply(q, k, v, sm_scale)


def bench(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    Z, H, N_CTX, HEAD_DIM = 2, 8, 1024, 64
    dtype = torch.float16
    sm_scale = HEAD_DIM ** -0.5
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")

    # end-to-end: does the whole autograd graph produce SDPA's gradients?
    q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
    q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
    do = torch.randn_like(q)

    flash_attention(q1, k1, v1, sm_scale).backward(do)
    torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, scale=sm_scale).backward(do)

    for name, mine, ref in (("dQ", q1.grad, q2.grad), ("dK", k1.grad, k2.grad),
                            ("dV", v1.grad, v2.grad)):
        err = (mine - ref).abs().max().item()
        print(f"{name} max error vs SDPA autograd = {err:.2e}")
        torch.testing.assert_close(mine, ref, atol=2e-2, rtol=0)
    print("✅ flash attention autograd integration passed")
    # note: no double-backward — the backward launches opaque kernels, so
    # grad-of-grad would silently be zeros rather than an error. same caveat
    # as every production flash-attn extension

    # fwd+bwd wall clock vs SDPA's fused path
    def sdpa_fwbw():
        o = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, scale=sm_scale)
        o.backward(do)

    def ours_fwbw():
        o = flash_attention(q1, k1, v1, sm_scale)
        o.backward(do)

    ms_ours, ms_sdpa = bench(ours_fwbw), bench(sdpa_fwbw)
    print(f"fwd+bwd: ours {ms_ours:.2f} ms | SDPA {ms_sdpa:.2f} ms | {ms_sdpa / ms_ours:.2f}x")
