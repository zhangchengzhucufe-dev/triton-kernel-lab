// RMSNorm forward in plain CUDA — the "can you do it without the DSL" check.
//
// One block per row, 256 threads. Two flavors:
//   - scalar: each thread strides over the row with plain half loads
//   - vectorized: float4 loads (8 halves per transaction) when D % 8 == 0
// Reduction is two-level: warp shuffle down, then one shared-memory round
// across the 8 warp sums.
//
// The interesting question isn't correctness (that's easy) — it's how much
// the vectorized loads buy on a bandwidth-bound kernel. The python driver
// benchmarks both against the Triton version in file 22.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int N_THREADS = 256;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// block-wide sum of `v` across all N_THREADS threads, broadcast to every
// thread (block size must be a multiple of 32; with 256 threads that's one
// shared round after the shuffle). forgot the broadcast on the first pass
// of this kernel — only lane 0 held the real sum, every other thread
// normalized by a partial value
__device__ __forceinline__ float block_sum(float v, float* warp_sums) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    // every lane of warp 0 must join the shuffle — warp_sum with a partial
    // warp is garbage, a classic shuffle pitfall
    v = (warp == 0 && lane < N_THREADS / 32) ? warp_sums[lane] : 0.0f;
    v = warp_sum(v);
    if (threadIdx.x == 0) warp_sums[0] = v;
    __syncthreads();
    return warp_sums[0];
}

__global__ void rmsnorm_scalar(const at::Half* __restrict__ x,
                               const at::Half* __restrict__ w,
                               at::Half* __restrict__ out,
                               int D, float eps) {
    const at::Half* row = x + (long)blockIdx.x * D;
    at::Half* orow = out + (long)blockIdx.x * D;
    __shared__ float warp_sums[N_THREADS / 32];

    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        ss += float(row[i]) * float(row[i]);
    float inv_rms = rsqrtf(block_sum(ss, warp_sums) / D + eps);

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        orow[i] = at::Half(float(row[i]) * inv_rms * float(w[i]));
}

// D must be divisible by 8. Each float4 fetches 8 consecutive halves, so the
// same row needs D/8 instead of D transactions per pass.
__global__ void rmsnorm_vec8(const float4* __restrict__ x,
                             const at::Half* __restrict__ w,
                             float4* __restrict__ out,
                             int D8, float eps) {
    const float4* row = x + (long)blockIdx.x * D8;
    float4* orow = out + (long)blockIdx.x * D8;
    __shared__ float warp_sums[N_THREADS / 32];

    float ss = 0.0f;
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        float4 p = row[i];
        const __half2* h = reinterpret_cast<const __half2*>(&p);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __half22float2(h[j]);
            ss += f.x * f.x + f.y * f.y;
        }
    }
    float inv_rms = rsqrtf(block_sum(ss, warp_sums) / (D8 * 8) + eps);

    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        float4 p = row[i];
        const __half2* h = reinterpret_cast<const __half2*>(&p);
        float4 r;
        __half2* rh = reinterpret_cast<__half2*>(&r);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __half22float2(h[j]);
            rh[j] = __floats2half2_rn(f.x * inv_rms * float(w[i * 8 + 2 * j]),
                                      f.y * inv_rms * float(w[i * 8 + 2 * j + 1]));
        }
        orow[i] = r;
    }
}

void rmsnorm(torch::Tensor x, torch::Tensor w, torch::Tensor out, double eps,
             bool force_scalar) {
    TORCH_CHECK(x.is_contiguous() && x.is_cuda());
    int rows = x.numel() / x.size(-1);
    int D = x.size(-1);
    if (D % 8 == 0 && D / 8 >= N_THREADS && !force_scalar) {
        rmsnorm_vec8<<<rows, N_THREADS>>>(
            reinterpret_cast<const float4*>(x.data_ptr<at::Half>()),
            w.data_ptr<at::Half>(),
            reinterpret_cast<float4*>(out.data_ptr<at::Half>()),
            D / 8, (float)eps);
    } else {
        rmsnorm_scalar<<<rows, N_THREADS>>>(
            x.data_ptr<at::Half>(), w.data_ptr<at::Half>(),
            out.data_ptr<at::Half>(), D, (float)eps);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "rmsnorm forward (cuda)", py::arg("x"), py::arg("w"),
          py::arg("out"), py::arg("eps"), py::arg("force_scalar") = false);
}
