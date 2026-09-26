"""Modern matmul with TMA + persistent scheduling (a simplified take on tutorial 09).

TMA turns "move a tile" into a single hardware instruction — no more manual
address computation or out-of-bounds padding. Note you must register an
allocation callback via triton.set_allocator; TMA descriptors need a small
scratch buffer. See 09 for persistent scheduling details.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _alloc(size: int, alignment: int, stream):
    # TMA descriptors need a small host-allocated scratch buffer; Triton requests memory via this callback
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc)


@triton.jit
def tma_matmul_kernel(
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
):
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    for tile_id in tl.range(tl.program_id(0), total_tiles, NUM_SMS, flatten=True):
        # Column-major tile traversal (tiles in the same column share an A strip, better L2 hit rate)
        pid_n = tile_id % num_tiles_n
        pid_m = tile_id // num_tiles_n
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for off_k in tl.range(0, K, BLOCK_K):
            a = a_desc.load([off_m, off_k])       # TMA hardware copy, no manual pointers
            b = b_desc.load([off_k, off_n])
            acc = tl.dot(a, b, acc)

        c_desc.store([off_m, off_n], acc.to(tl.float16))


def tma_matmul(A, B):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)
    tma_matmul_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=3,
    )
    return C


def bench(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters): fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 4096, 4096, 4096
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)

    C = tma_matmul(A, B)
    ref = A @ B
    print(f"max abs error = {(C - ref).abs().max().item():.2e}")
    torch.testing.assert_close(C, ref, atol=1e-1, rtol=1e-2)
    print("✅ TMA persistent matmul correctness OK")

    t_tma = bench(lambda: tma_matmul(A, B))
    t_cublas = bench(lambda: A @ B)
    tflops = 2 * M * N * K / (t_tma * 1e-3) / 1e12
    print(f"TMA persistent: {t_tma:.2f} ms ({tflops:.0f} TFLOPS)   cuBLAS: {t_cublas:.2f} ms")
