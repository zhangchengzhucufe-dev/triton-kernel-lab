"""RoPE，NeoX/HF 的半分约定：第 (i, i+d/2) 两列乘 2D 旋转。

cos/sin 直接在 kernel 里从 base 现算（写 exp2 形式快一些），没有做
cache 表。原地写回，decode 时每步都要过一遍，省一半显存流量。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
        Q, positions,
        stride_s, stride_h, stride_d,
        N_HEADS, HEAD_DIM: tl.constexpr, ROTARY_BASE: tl.constexpr,
        BLOCK_H: tl.constexpr,
):
    seq = tl.program_id(0)
    pos = tl.load(positions + seq)

    half = HEAD_DIM // 2
    # θ_i = base^(-2i/d)。写成 exp2 形式：2^(-2i/d · log2(base))，硬件更快。
    # 注意 arange 的参数必须是 constexpr，直接写 HEAD_DIM // 2（不能经由运行期变量）
    inv_freq = tl.exp2(tl.arange(0, HEAD_DIM // 2).to(tl.float32) * (-2.0 * tl.log2(ROTARY_BASE) / HEAD_DIM))
    angles = pos * inv_freq
    cos = tl.cos(angles)
    sin = tl.sin(angles)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM // 2)
    mask = (offs_h[:, None] < N_HEADS)
    ptrs_first = Q + seq * stride_s + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    ptrs_second = ptrs_first + half * stride_d

    x1 = tl.load(ptrs_first, mask=mask)
    x2 = tl.load(ptrs_second, mask=mask)
    c = cos[None, :]
    s = sin[None, :]
    tl.store(ptrs_first, x1 * c - x2 * s, mask=mask)
    tl.store(ptrs_second, x1 * s + x2 * c, mask=mask)


def apply_rope_inplace(q, positions, rotary_base=10000.0):
    """q: (seq, heads, head_dim)，原地施加 RoPE。"""
    seq, heads, head_dim = q.shape
    assert head_dim % 2 == 0
    _rope_kernel[(seq,)](
        q, positions,
        q.stride(0), q.stride(1), q.stride(2),
        heads, HEAD_DIM=head_dim, ROTARY_BASE=rotary_base,
        BLOCK_H=triton.next_power_of_2(heads), num_warps=4,
    )


def rope_reference(x, positions, base=10000.0):
    """HF 风格参考实现，用于对照。"""
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = positions.float()[:, None] * inv_freq[None, :]          # (seq, d/2)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos[:, None, :] + rotated.float() * sin[:, None, :]).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    seq, heads, head_dim = 2048, 16, 128
    q = torch.randn((seq, heads, head_dim), device='cuda', dtype=torch.float16)
    positions = torch.arange(seq, device='cuda', dtype=torch.int32)

    ref = rope_reference(q, positions)
    apply_rope_inplace(q, positions)
    print(f"最大误差 = {(q.float() - ref.float()).abs().max().item():.2e}")
    torch.testing.assert_close(q.float(), ref.float(), atol=1e-2, rtol=0)
    print("✅ RoPE 正确性通过")
