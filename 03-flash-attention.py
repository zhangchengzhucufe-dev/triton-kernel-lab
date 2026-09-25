"""Flash Attention v2 forward (a slimmed-down version of tutorial 06, no dropout / causal / fp8).

The core is online softmax: scan K in blocks, each block updates the running max, and
both the old acc and l must be rescaled by exp(m_old - m_new), divided by l at the end.
The intermediate S and P matrices never touch HBM — that's what saves memory.
The 1.44269504 multiplied into qk_scale is the log2(e) base-change factor that swaps
exp for exp2 (a faster hardware instruction).

The tutorial version silently assumed seqlen was a multiple of
every block size — the block-pointer loads had no boundary_check, so an odd
seqlen just read past the end of K/V. Fixing that is two changes, and the
second one is the trap: boundary_check fills out-of-bounds lanes with 0, and a
0-filled k column produces a perfectly finite qk=0 score that sneaks into the
softmax denominator (file 16's bug, forward flavor). So the OOB columns get
killed explicitly with -1e6 after the dot, before the max. Arbitrary q/k/v
strides were already fine — the kernel always took all four strides per tensor.

...except they weren't, and the old self-check could never catch it: the
kernel computed ONE qvk_offset from Q's batch/head strides and reused it for
K, V and Out. Fine when everything is contiguous (the tutorial's assumption),
but feed it a sliced q and a contiguous o — exactly what empty_like hands
back — and Out gets written at Q's offsets: wrong planes, and for late heads
past the end of the output buffer entirely. Wrong values AND an out-of-bounds
write in one bug. Each tensor now gets its own offset from its own strides.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr,
        start_m, qk_scale,
        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        offs_m, offs_n, N_CTX,
):
    # STAGE decides which segment of the causal matrix this program handles (this
    # simplified version doesn't do causal, but keeps tutorial's stage structure
    # for easier comparison with the original)
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr, boundary_check=(1,))
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N), fp32 accumulation
        # boundary_check zero-fills OOB k columns, and qk=0 is a real-looking
        # score — kill it before it reaches the max/sum, not after
        qk = tl.where((start_n + offs_n)[None, :] < N_CTX, qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]          # shift to <= 0 to avoid overflow
        p = tl.math.exp2(qk)                        # exp2: faster hardware instruction
        alpha = tl.math.exp2(m_i - m_ij)            # online softmax rescaling factor
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]                  # correct the previous PV partial sum
        v = tl.load(V_block_ptr, boundary_check=(0,))
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)                     # accumulate P @ V
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
        Q, K, V, sm_scale, M, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)          # which block of Q
    off_hz = tl.program_id(1)           # (batch * num_heads) flattened to 1D
    off_z = off_hz // H
    off_h = off_hz % H
    # per-tensor offsets: the four tensors do NOT necessarily share batch/head
    # strides (sliced q + contiguous o is enough to break the shared-offset
    # version, and the failure mode is an out-of-bounds write, not just
    # wrong numbers)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # Note: Python lambdas/closures can't be defined inside kernels; block pointers
    # must be constructed explicitly. K uses a transposed (HEAD_DIM, N_CTX) view so
    # tl.dot(q, k) works directly.
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),   # K transposed view
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504    # log2(e) base-change factor to swap exp for exp2
    # rows past N_CTX come back as 0, produce garbage scores, and get thrown
    # away by the masked store at the end — the garbage never leaves the kernel
    q = tl.load(Q_block_ptr, boundary_check=(0,))
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                    start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 4,
                                    offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))
    # Store logsumexp for the backward pass (this file only does forward; kept as a convention)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX)


def attention(q, k, v, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    # 128/64/8/3 per autotune-results.csv: best at ctx 1024 and 8192, within
    # ~10% of the per-shape winner at 2048/4096 (whose configs disagree with
    # each other anyway — not chasing that). the first sweep run had a 64-wide
    # tile 1.6-2.3x faster at ctx 1024 (looked like wave quantization) and
    # this wrapper branched on seqlen for a while; every clean re-run since
    # put the big tile back on top there (0.70 vs 0.84 ms), so the branch is
    # gone.
    BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 8, 3
    N_CTX = q.shape[2]
    grid = (triton.cdiv(N_CTX, BLOCK_M), q.shape[0] * q.shape[1], 1)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def _check_case(name, q, k, v, sm_scale):
    o_triton = attention(q, k, v, sm_scale)
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    torch.testing.assert_close(o_triton.float(), o_ref.float(), atol=2e-2, rtol=0)
    print(f"✅ {name}")


if __name__ == "__main__":
    torch.manual_seed(0)
    dtype = torch.float16

    # the original case: aligned, contiguous
    Z, H, N_CTX, HEAD_DIM = 2, 4, 1024, 64
    q, k, v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(0, 0.5)
               for _ in range(3))
    _check_case("2x4x1024x64 aligned (the old case)", q, k, v, 0.5)

    # seqlen that no block size divides — the loads used to run off the end
    _check_case("1x2x1000x64 odd seqlen",
                *(torch.empty((1, 2, 1000, 64), dtype=dtype, device="cuda").normal_(0, 0.5)
                  for _ in range(3)), 0.5)

    # seqlen smaller than one block: most of the q block is out of bounds,
    # those rows come back NaN (l=0), and only the store's boundary_check
    # throws them away — worth a test, that's a lot of trust in one mask
    _check_case("1x1x32x64 tiny (smaller than one block)",
                *(torch.empty((1, 1, 32, 64), dtype=dtype, device="cuda").normal_(0, 0.5)
                  for _ in range(3)), 64 ** -0.5)

    # bigger head dim
    _check_case("2x4x1024x128",
                *(torch.empty((2, 4, 1024, 128), dtype=dtype, device="cuda").normal_(0, 0.5)
                  for _ in range(3)), 128 ** -0.5)

    # non-contiguous: sliced rows and transposed views, all four strides live
    base = torch.randn(2, 4, 2048, 64, dtype=dtype, device="cuda")
    _check_case("sliced rows q/k/v[::2]",
                *(t[:, :, ::2, :] for t in (base, base.clone(), base.clone())), 64 ** -0.5)
    # the exact shape of the qvk_offset bug: q strided, k/v contiguous — the
    # shared-offset version read K/V at Q's plane offsets and wrote O there too
    q_s = base[:, :, ::2, :]
    _check_case("q sliced, k/v contiguous (regression: per-tensor offsets)",
                q_s, torch.randn_like(q_s), torch.randn_like(q_s), 64 ** -0.5)
    perm = torch.randn(2, 64, 1024, 4, dtype=dtype, device="cuda").permute(0, 2, 3, 1)
    _check_case("permuted (D-major) views",
                *(t.contiguous() if i == 0 else t for i, t in enumerate(
                    (perm, perm.clone(), perm.clone()))), 64 ** -0.5)

    print("✅ flash attention forward correctness passed (all cases)")
