"""Flash Attention backward — the part the forward's "never touch HBM" trick
makes genuinely painful.

Backward wants to read P = softmax(QK^T), but the forward pass never stored
it. So both backward kernels recompute the scores from Q and K on the fly and
rebuild P from the logsumexp the forward saved:

    P = exp2(QK^T * scale * log2e - lse)        # lse is in log2 units

Per block the rest of the math is:

    dV += P^T @ dO
    dP  = dO @ V^T
    dS  = P ∘ (dP - rowsum(P ∘ dP))     # softmax Jacobian, folded into one term
    dQ += dS @ K * scale
    dK += dS^T @ Q * scale

and rowsum(P ∘ dP) is exactly rowsum(dO ∘ O), which we precompute as `delta`
in one torch line instead of a separate kernel launch.

Why two kernels: dK/dV want the Q dimension looped (each program keeps its KV
block pinned in registers), while dQ wants the KV dimension looped. Sharing
one kernel would mean accumulating dQ with atomics — slower and
non-deterministic, so everyone just eats the two launches.

The original also assumed seqlen was a multiple of both block sizes (file 25
just asserted it away). The masking turned out to have a surprise in it: an
out-of-bounds row loads q/do/lse all as 0, so pT = exp2(0 - 0) = 1 — a fake
weight of 1 in every OOB column! My gut said that has to pollute dK/dV, and I
even "fixed" it with exp2-overflow hand-waving before checking. It doesn't,
and not by accident: every place pT multiplies into dK/dV also multiplies by
the zero-filled do or q, so the whole contribution is exactly 0. The forward
isn't this lucky — its softmax denominator sums raw p over the row, so a fake
score pollutes it (file 16's bug), which is why the forward needs the -1e6
kill and the backward doesn't. I still zero the OOB entries at P explicitly:
it's free, and "invalid lanes are zero at P" is the invariant this kernel
needs the moment it grows a causal mask.

File 03/15 show the forward side of this, including the causal variant.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_dkdv(
        Q, K, V, DO, LSE, DELTA, DK, DV,
        sm_scale, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # all tensors arrive as contiguous (Z*H, N_CTX, HEAD_DIM), so one base
    # offset per head plane is all the addressing we need
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    base = off_hz.to(tl.int64) * N_CTX * HEAD_DIM

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    kv_mask = offs_n[:, None] < N_CTX
    kv_offs = offs_n[:, None] * HEAD_DIM + offs_d[None, :]

    k = tl.load(K + base + kv_offs, mask=kv_mask, other=0.0)
    v = tl.load(V + base + kv_offs, mask=kv_mask, other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N_CTX
        q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                     mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)
        delta = tl.load(DELTA + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)

        qkT = tl.dot(k, tl.trans(q)) * qk_scale     # (BLOCK_N, BLOCK_M)
        pT = tl.math.exp2(qkT - lse[None, :])       # rebuild P, transposed
        # OOB query columns carry a fake pT=1 here (q/lse load as 0 — docstring),
        # but their do/q are 0 too so nothing they multiply survives. zeroed
        # anyway: P is the one tensor whose "all invalid lanes are zero" I
        # want guaranteed by the code, not by algebra luck
        pT = tl.where(m_mask[None, :], pT, 0.0)
        dv += tl.dot(pT.to(tl.float16), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(tl.float16), q) * sm_scale

    tl.store(DK + base + kv_offs, dk, mask=kv_mask)
    tl.store(DV + base + kv_offs, dv, mask=kv_mask)


@triton.jit
def _attn_bwd_dq(
        Q, K, V, DO, LSE, DELTA, DQ,
        sm_scale, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    base = off_hz.to(tl.int64) * N_CTX * HEAD_DIM

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX

    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                 mask=m_mask[:, None], other=0.0)
    lse = tl.load(LSE + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)
    delta = tl.load(DELTA + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_CTX
        k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)       # same invariant as dkdv
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float16), k) * sm_scale

    tl.store(DQ + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq, mask=m_mask[:, None])


def attention_bwd(q, k, v, o, do, sm_scale, lse, BLOCK_M=64, BLOCK_N=64):
    Z, H, N_CTX, HEAD_DIM = q.shape
    # reshape() (not view): a strided input copies here instead of exploding —
    # acceptable for backward, the kernels below assume flat contiguous planes
    qf, kf, vf, of, dof = (x.reshape(Z * H, N_CTX, HEAD_DIM) for x in (q, k, v, o, do))

    # rowsum(dO ∘ O) — this equals rowsum(P ∘ dP), the term the softmax
    # Jacobian leaves behind. One torch line beats a custom kernel here
    delta = (dof.float() * of.float()).sum(-1)

    dq = torch.empty_like(qf, dtype=torch.float32)
    dk = torch.empty_like(kf, dtype=torch.float32)
    dv = torch.empty_like(vf, dtype=torch.float32)

    # 64/64/4/2 everywhere; the first sweep run hinted 8 warps buys ~4% at
    # ctx 4k, the re-run said the opposite — not branching on that
    _attn_bwd_dkdv[(triton.cdiv(N_CTX, BLOCK_N), Z * H)](
        qf, kf, vf, dof, lse, delta, dk, dv, sm_scale, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, num_warps=4, num_stages=2,
    )
    _attn_bwd_dq[(triton.cdiv(N_CTX, BLOCK_M), Z * H)](
        qf, kf, vf, dof, lse, delta, dq, sm_scale, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, num_warps=4, num_stages=2,
    )
    return (x.reshape(Z, H, N_CTX, HEAD_DIM) for x in (dq, dk, dv))


def ref_attention(q, k, v, sm_scale):
    att = torch.softmax(q.float() @ k.float().transpose(-1, -2) * sm_scale, -1)
    return att @ v.float()


if __name__ == "__main__":
    torch.manual_seed(0)
    dtype = torch.float16
    sm_scale = 1 / 64 ** 0.5

    # the old aligned case
    Z, H, N_CTX, HEAD_DIM = 2, 4, 1024, 64
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    do = torch.randn_like(q)

    # a real forward pass hands its logsumexp straight to backward; there is
    # no forward in this file, so recompute it in torch (log2 units, like
    # file 03 stores it). costs a full (N, N) matrix — fine for a test
    s = (q.float() @ k.float().transpose(-1, -2)) * sm_scale
    lse = torch.logsumexp(s, dim=-1) * 1.44269504
    o = ref_attention(q, k, v, sm_scale).to(dtype)

    dq, dk, dv = attention_bwd(q, k, v, o, do, sm_scale, lse)

    # reference gradients via autograd on a plain fp32 attention
    q_r = q.float().requires_grad_(True)
    k_r = k.float().requires_grad_(True)
    v_r = v.float().requires_grad_(True)
    ref_attention(q_r, k_r, v_r, sm_scale).backward(do.float())

    for name, mine, ref in (("dQ", dq, q_r.grad), ("dK", dk, k_r.grad), ("dV", dv, v_r.grad)):
        print(f"{name} max error = {(mine - ref).abs().max().item():.2e}")
        torch.testing.assert_close(mine, ref, atol=2e-2, rtol=0)
    print("✅ flash attention backward correctness passed (aligned)")

    # odd seqlen: the case the old assert forbade. N_CTX % BLOCK_M and
    # % BLOCK_N both nonzero, and different remainders, so both loop tails bite
    for n_ctx in (1000, 1032):
        q = torch.randn(1, 2, n_ctx, 64, dtype=dtype, device="cuda")
        k = torch.randn(1, 2, n_ctx, 64, dtype=dtype, device="cuda")
        v = torch.randn(1, 2, n_ctx, 64, dtype=dtype, device="cuda")
        do = torch.randn_like(q)
        s = (q.float() @ k.float().transpose(-1, -2)) * sm_scale
        lse = torch.logsumexp(s, dim=-1) * 1.44269504
        o = ref_attention(q, k, v, sm_scale).to(dtype)
        dq, dk, dv = attention_bwd(q, k, v, o, do, sm_scale, lse)
        q_r = q.float().requires_grad_(True)
        k_r = k.float().requires_grad_(True)
        v_r = v.float().requires_grad_(True)
        ref_attention(q_r, k_r, v_r, sm_scale).backward(do.float())
        for name, mine, ref in (("dQ", dq, q_r.grad), ("dK", dk, k_r.grad), ("dV", dv, v_r.grad)):
            torch.testing.assert_close(mine, ref, atol=2e-2, rtol=0)
        print(f"✅ odd seqlen {n_ctx} passed")
