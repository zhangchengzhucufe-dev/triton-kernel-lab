# triton Study Notes

Practice code worked through alongside the official tutorials; files from 06 onward are my own additions.
Environment: WSL2 + triton 3.8.0. Each file runs directly with `python xx.py` (GPU required) and automatically checks numerics against torch.

## What's written so far

- vector_add / fused_softmax / replicate.py — the first ones typed along with tutorials 01/02; benchmark results for the softmax one are in softmax-performance.png
- 06 matmul: tiling + tl.dot, with autotune and L2 swizzle. Reaches about half of cuBLAS performance at 4096³
- 07 layernorm: first hand-written backward; dW/dB accumulated with atomic_add
- 08 flash attention: forward only. Core is online softmax — each K block scanned requires rescaling the old acc by exp(m_old-m_new)
- 09 transpose + persistent kernel: transpose is 3x faster than torch's x.T.contiguous() (259 GB/s vs 84), mostly thanks to coalescing
- 10 prefix sum + histogram: associative_scan and two-pass scan; did a comparison experiment for the histogram — with 8 bins, naive atomic takes 1325us, while accumulating in registers first then merging takes only 66us, a 20x speedup
- 11 TMA matmul: replaces manual pointers with TensorDescriptor, so no need to write the K address loop by hand
- 12 fused cross-entropy: streaming logsumexp, no need to materialize the (N, V) intermediate matrix; 3.1x faster than eager at 8192x32K vocab
- 13 add+layernorm fusion, 14 silu*mul (the activation part of swiglu, backward derived by hand)
- 15 RoPE, 16 int8 quantization, 17 MoE routing
- 18 fused adamw: matched against torch.optim.AdamW for 5 steps, error 2e-6
- 19 split-K GEMM: slightly faster than cuBLAS on skinny matrices (16x4096x16384)
- 20 causal flash attention: splits the K loop into left-of-diagonal-block / diagonal-block segments; the right side doesn't need computing at all
- 21 flash-decoding: during decode (q length=1), splits along the KV seq dimension with a two-phase merge; 2x faster than SDPA with a 32K cache
- 22 rmsnorm: the LLaMA-style one, backward derived by hand, verified against torch.nn.functional.rms_norm
- 23 registering kernels as torch custom ops (torch.library.custom_op + register_fake), traceable by torch.compile
- 24 paged attention decode: K/V live in fixed-size pages with a per-sequence block table, virtual-memory style (the vLLM layout); walks the table page by page with online softmax, GQA included. Pairs with 21 — that one split the KV dim across SMs, this one fixes the cache layout
- 25 grouped GEMM for MoE: all experts in one launch. Tokens are pre-sorted by expert so each expert owns a contiguous slice; programs for experts that ran out of tokens exit early
- 26 flash attention backward: both kernels rebuild P from the logsumexp the forward saved; dK/dV and dQ run as two kernels because sharing one would need atomics on dQ. The rowsum(P∘dP) term collapses to rowsum(dO∘O), precomputed in one torch line

## Pitfalls hit along the way (all kept in code comments)

1. `tl.arange` arguments must be constexpr — can't store them in a normal variable and pass them in (bit me in file 15)
2. The nastiest one was in file 20: block pointer `tl.advance` advances a function-local variable, so after splitting the K loop into two segments, the second segment still got the original pointer — the diagonal blocks recomputed the earlier columns. Error was only around 0.1, first mistaken for a precision issue; after much digging it turned out to be double counting. Lesson: pointers shared across "logical segments" in a jit function must either be passed out or the loop shouldn't be split
3. autotune must decorate an @triton.jit kernel; wrapping it around a plain python function raises an arg_names error
4. First version of layernorm backward forgot to multiply by w, making dX completely wrong; also, a histogram written as "per-bin loop with scalar comparison" produces entirely wrong results — it must be written as broadcast comparison
5. For quantization error, don't use relative error — elements with x near 0 blow up in relative terms by definition; compare against the theoretical bound absmax/254 instead
6. File 21 took me the longest: first, q and KV have different plane strides (q is D, KV is N*D) and can't share; second, masked-out k rows produce qk=0 rather than -inf, polluting the softmax denominator; third, the merge formula must not multiply acc by l. Three bugs stacked together gave errors on the order of 1e1
7. I misremembered the silu derivative: dsilu = sig*(1 + x*(1-sig)); the sigmoid derivative is sig*(1-sig), not sig*sig
8. File 24: in the wrapper I set PAGE_SIZE from the block table's width instead of the cache's — two sizes that both look like "pages", and the kernel happily read 32-token "pages" out of 16-token blocks, garbage everywhere. Every size a kernel takes should be traced back to the exact tensor dimension it describes

## Not yet written

The stream-K variant of split-K — will add when there's time.
make_block_ptr is deprecated in 3.8; the new way is tl.make_tensor_descriptor, with an example in file 11.
