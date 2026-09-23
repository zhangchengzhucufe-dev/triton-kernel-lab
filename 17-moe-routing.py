"""MoE routing：softmax → top-k → 重归一化 → 专家计数。

k 一般只有 2~8，不用 tl.sort，迭代 k 次 argmax + 屏蔽就行。
屏蔽要用 -inf 不能置 0，否则会重复选同一个专家。
counts 用 atomic 累，是负载均衡 / load-balancing loss 的数据来源。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_route_kernel(
        logits_ptr, weights_ptr, indices_ptr, counts_ptr,
        stride_lm,
        N_EXPERTS, TOP_K: tl.constexpr,
        BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    mask = offs_e < N_EXPERTS
    logits = tl.load(logits_ptr + token * stride_lm + offs_e, mask=mask,
                     other=-float("inf")).to(tl.float32)
    # 手写 softmax（部分版本的 tl.softmax 接口不一致，手写 3 行最稳）
    probs = tl.exp(logits - tl.max(logits, axis=0))
    probs = probs / tl.sum(probs, axis=0)

    # 迭代 top-k：argmax → 记录 → 用 -inf 屏蔽。k 很小时这是最优写法。
    sum_topk = 0.0
    for k in tl.static_range(TOP_K):
        idx = tl.argmax(probs, axis=0)
        w = tl.max(probs, axis=0)
        tl.store(weights_ptr + token * TOP_K + k, w)
        tl.store(indices_ptr + token * TOP_K + k, idx)
        sum_topk += w
        probs = tl.where(offs_e == idx, -float("inf"), probs)

    # Mixtral 风格：top-k 权重重归一化
    for k in tl.static_range(TOP_K):
        w = tl.load(weights_ptr + token * TOP_K + k)
        tl.store(weights_ptr + token * TOP_K + k, w / sum_topk)

    tl.atomic_add(counts_ptr + tl.load(indices_ptr + token * TOP_K + tl.arange(0, TOP_K)), 1)


def moe_route(logits, top_k=2):
    tokens, n_experts = logits.shape
    weights = torch.empty((tokens, top_k), device=logits.device, dtype=torch.float32)
    indices = torch.empty((tokens, top_k), device=logits.device, dtype=torch.int32)
    counts = torch.zeros((n_experts,), device=logits.device, dtype=torch.int32)
    _moe_route_kernel[(tokens,)](
        logits.contiguous(), weights, indices, counts,
        logits.stride(0), n_experts, TOP_K=top_k,
        BLOCK_E=triton.next_power_of_2(n_experts),
    )
    return weights, indices, counts


if __name__ == "__main__":
    torch.manual_seed(0)
    tokens, n_experts, top_k = 8192, 64, 8

    logits = torch.randn((tokens, n_experts), device='cuda', dtype=torch.float32)
    weights, indices, counts = moe_route(logits, top_k)

    # PyTorch 参考
    probs = torch.softmax(logits, dim=-1)
    ref_w, ref_idx = torch.topk(probs, top_k, dim=-1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    ref_counts = torch.bincount(ref_idx.flatten(), minlength=n_experts)

    torch.testing.assert_close(indices, ref_idx.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(weights, ref_w, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(counts.to(torch.int64), ref_counts, atol=0, rtol=0)
    print("✅ MoE 路由正确性通过")
    print(f"负载示例（{n_experts} 个专家收到的 token 数，均值 {tokens * top_k / n_experts:.0f}）：")
    print(counts[:16].tolist(), "...")
