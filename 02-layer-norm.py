"""LayerNorm forward + backward, corresponding to tutorial 05; my first hand-written backward.

Note that the E[x²] - E[x]² formulation for variance risks catastrophic cancellation;
switch to Welford when x is too large. In the backward, I omitted the * w in
dXhat = dY * w in my first version, which made everything wrong.
Accumulate in fp32 throughout, even for fp16 inputs.

The tutorial version only handled 2D contiguous fp16 — it called .contiguous()
on the input and hardcoded the output dtype. This version takes any dtype,
any N (masked), and batched/non-contiguous inputs. The part worth reading is
_as_2d: instead of an invisible reshape (which silently copies when it can't
view), I check the strides explicitly and only then decide whether rows can
be folded. Worst case you eat a copy; you never get silently wrong numbers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_kernel(
        X, Y, W, B, MEAN, RSTD,
        stride_xr, stride_xc, stride_yr, stride_yc,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride_xr + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    # E[x²] - E[x]²: numerically less stable than Welford, but fine for teaching
    # purposes with fp32 accumulation. If |x| > 1e4, switch to Welford or a
    # two-pass method (center first, then sum of squares).
    _var = tl.sum(x * x, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(_var + eps)

    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)

    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(Y + row * stride_yr + cols * stride_yc, y, mask=mask)


@triton.jit
def _layer_norm_bwd_kernel(
        X, W, DY, DX, DW, DB, MEAN, RSTD,
        stride_xr, stride_xc, stride_dyr, stride_dyc, stride_dxr, stride_dxc,
        N,
        BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row's dX while atomically accumulating its share of DW/DB.
    # The tutorial has each program process a shard of rows then reduce; here we use
    # the most straightforward "one program per row" version for readability.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride_xr + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_dyr + cols * stride_dyc, mask=mask, other=0.0).to(tl.float32)
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
    tl.store(DX + row * stride_dxr + cols * stride_dxc, dx, mask=mask)


def _as_2d(x):
    """Fold every leading dim into one row axis, without copying if at all possible.

    Flat row indexing is only valid when stepping from one row to the next is a
    single constant offset — i.e. the leading dims merge cleanly (torch would
    let me .view it, this check is exactly the view-compatibility condition).
    A permuted batch fails it, and for those I eat the .contiguous() copy.
    reshape() would have done this silently; I'd rather see it happen.
    """
    if x.dim() == 1:
        return x.unsqueeze(0)
    for i in range(x.dim() - 2):
        if x.stride(i) != x.shape[i + 1] * x.stride(i + 1):
            return x.contiguous().view(-1, x.shape[-1])
    return x.view(-1, x.shape[-1])


def _num_warps(block_size):
    # from autotune-sweep.py: below N≈8k the warp count hardly matters —
    # bandwidth-bound row, every config lands within a few percent. the real
    # cliff is too FEW warps on very long rows: at N=16384 one warp spills
    # ~500 fp32 slots per lane to local memory and runs 100x slow; 4 is the
    # knee. (my first read of the first sweep was "8 warps 10x slow at
    # N=4096" — that outlier never reproduced on the re-run. re-run before
    # writing a claim down.)
    return 1 if block_size <= 4096 else 4


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        assert weight.is_contiguous() and bias.is_contiguous(), \
            "weight/bias are parameters; keeping them contiguous keeps the indexing simple"
        orig_shape = x.shape
        x2 = _as_2d(x)
        M, N = x2.shape
        y = torch.empty_like(x2)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(N)
        _layer_norm_fwd_kernel[(M,)](
            x2, y, weight, bias, mean, rstd,
            x2.stride(0), x2.stride(1), y.stride(0), y.stride(1),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=_num_warps(BLOCK_SIZE),
        )
        ctx.save_for_backward(x2, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.orig_shape = orig_shape
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, mean, rstd = ctx.saved_tensors
        dy2 = _as_2d(dy)
        M, N = x2.shape
        dx = torch.empty_like(x2)
        dw = torch.zeros((N,), dtype=torch.float32, device=x2.device)
        db = torch.zeros((N,), dtype=torch.float32, device=x2.device)
        _layer_norm_bwd_kernel[(M,)](
            x2, w, dy2, dx, dw, db, mean, rstd,
            x2.stride(0), x2.stride(1), dy2.stride(0), dy2.stride(1),
            dx.stride(0), dx.stride(1),
            N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=_num_warps(ctx.BLOCK_SIZE),
        )
        # grads come back in the params' dtype — autograd would reject fp32
        # accumulators against fp16 leaves
        return dx.view(ctx.orig_shape), dw.to(w.dtype), db.to(w.dtype), None


layer_norm = LayerNorm.apply


def _check_case(name, x, weight, bias, eps=1e-5):
    """Run one shape/layout/dtype combo against F.layer_norm, fwd + bwd."""
    x = x.requires_grad_(True)
    weight = weight.requires_grad_(True)
    bias = bias.requires_grad_(True)

    y_triton = layer_norm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    torch.testing.assert_close(y_triton, y_ref, atol=2e-2, rtol=0)
    # dtype must survive the round trip — the old version silently promoted to fp16
    assert y_triton.dtype == x.dtype, f"dtype changed: {x.dtype} -> {y_triton.dtype}"

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    ref_x, ref_w, ref_b = x.grad.clone(), weight.grad.clone(), bias.grad.clone()
    x.grad = None; weight.grad = None; bias.grad = None
    y_triton.backward(grad)
    torch.testing.assert_close(x.grad, ref_x, atol=2e-2, rtol=0)
    torch.testing.assert_close(weight.grad, ref_w, atol=2e-2, rtol=0)
    torch.testing.assert_close(bias.grad, ref_b, atol=2e-2, rtol=0)
    print(f"✅ {name}")


if __name__ == "__main__":
    torch.manual_seed(0)

    # the original case: plain 2D fp16
    _check_case("2D 512x1024 fp16 (the old case)",
                torch.randn(512, 1024, dtype=torch.float16, device="cuda"),
                torch.rand(1024, dtype=torch.float16, device="cuda"),
                torch.rand(1024, dtype=torch.float16, device="cuda"))

    # N not a multiple of anything, M tiny — the mask path the old wrapper never hit
    _check_case("2D 7x1000 fp16 (odd N, mask path)",
                torch.randn(7, 1000, dtype=torch.float16, device="cuda"),
                torch.rand(1000, dtype=torch.float16, device="cuda"),
                torch.rand(1000, dtype=torch.float16, device="cuda"))

    # dtypes other than fp16 used to get silently rewritten to fp16
    _check_case("2D 128x512 fp32",
                torch.randn(128, 512, dtype=torch.float32, device="cuda"),
                torch.rand(512, dtype=torch.float32, device="cuda"),
                torch.rand(512, dtype=torch.float32, device="cuda"))
    _check_case("2D 128x512 bf16",
                torch.randn(128, 512, dtype=torch.bfloat16, device="cuda"),
                torch.rand(512, dtype=torch.bfloat16, device="cuda"),
                torch.rand(512, dtype=torch.bfloat16, device="cuda"))

    # batched: 3D folds into rows with a uniform stride — no copy needed
    _check_case("3D 2x3x128 fp16 batched",
                torch.randn(2, 3, 128, dtype=torch.float16, device="cuda"),
                torch.rand(128, dtype=torch.float16, device="cuda"),
                torch.rand(128, dtype=torch.float16, device="cuda"))

    # sliced rows: still uniformly strided, kernel must honor the stride
    big = torch.randn(8, 512, dtype=torch.float16, device="cuda")
    _check_case("2D strided rows (x[::2] view)",
                big[::2],
                torch.rand(512, dtype=torch.float16, device="cuda"),
                torch.rand(512, dtype=torch.float16, device="cuda"))

    # permuted batch: rows are NOT a fixed step apart — this one takes the
    # .contiguous() copy instead of producing garbage
    perm = torch.randn(2, 128, 64, dtype=torch.float16, device="cuda").transpose(1, 2)
    assert not perm.is_contiguous()
    _check_case("transposed (2,128,64)->(2,64,128) view",
                perm,
                torch.rand(perm.shape[-1], dtype=torch.float16, device="cuda"),
                torch.rand(perm.shape[-1], dtype=torch.float16, device="cuda"))

    print("✅ layernorm forward/backward correctness passed (all cases)")
