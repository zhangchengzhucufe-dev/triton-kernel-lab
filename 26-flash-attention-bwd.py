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

No masking here: seq len is assumed to be a multiple of both block sizes.
File 08/20 show the forward side of this, including the causal variant.
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
    kv_offs = offs_n[:, None] * HEAD_DIM + offs_d[None, :]

    k = tl.load(K + base + kv_offs)
    v = tl.load(V + base + kv_offs)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
        do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
        lse = tl.load(LSE + off_hz * N_CTX + offs_m)
        delta = tl.load(DELTA + off_hz * N_CTX + offs_m)

        qkT = tl.dot(k, tl.trans(q)) * qk_scale     # (BLOCK_N, BLOCK_M)
        pT = tl.math.exp2(qkT - lse[None, :])       # rebuild P, transposed
        dv += tl.dot(pT.to(tl.float16), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(tl.float16), q) * sm_scale

    tl.store(DK + base + kv_offs, dk)
    tl.store(DV + base + kv_offs, dv)


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

    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    lse = tl.load(LSE + off_hz * N_CTX + offs_m)
    delta = tl.load(DELTA + off_hz * N_CTX + offs_m)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :])

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.math.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float16), k) * sm_scale

    tl.store(DQ + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq)


def attention_bwd(q, k, v, o, do, sm_scale, lse, BLOCK_M=64, BLOCK_N=64):
    Z, H, N_CTX, HEAD_DIM = q.shape
    qf, kf, vf, of, dof = (x.reshape(Z * H, N_CTX, HEAD_DIM) for x in (q, k, v, o, do))

    # rowsum(dO ∘ O) — this equals rowsum(P ∘ dP), the term the softmax
    # Jacobian leaves behind. One torch line beats a custom kernel here
    delta = (dof.float() * of.float()).sum(-1)

    dq = torch.empty_like(qf, dtype=torch.float32)
    dk = torch.empty_like(kf, dtype=torch.float32)
    dv = torch.empty_like(vf, dtype=torch.float32)

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
    Z, H, N_CTX, HEAD_DIM = 2, 4, 1024, 64
    dtype = torch.float16
    sm_scale = 1 / HEAD_DIM ** 0.5

    q = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, dtype=dtype, device="cuda")
    do = torch.randn_like(q)

    # a real forward pass hands its logsumexp straight to backward; there is
    # no forward in this file, so recompute it in torch (log2 units, like
    # file 08 stores it). costs a full (N, N) matrix — fine for a test
    s = (q.float() @ k.float().transpose(-1, -2)) * sm_scale
    lse = torch.logsumexp(s, dim=-1) * 1.44269504

    # delta needs the forward's output O; there's no forward here, so produce
    # it the plain way. In a real stack this is whatever the forward returned
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
    print("✅ flash attention backward correctness passed")
