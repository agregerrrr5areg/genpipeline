
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Warp-Optimized Sparse 3D Convolution
 * 
 * Parallelism strategy:
 * - Each WARP (32 threads) processes ONE output voxel (b, co, ox, oy, oz).
 * - Threads within the warp parallelize the input channel (Ci) and kernel (K^3) reduction.
 * - Uses __shfl_down_sync for ultra-fast register-level reduction.
 */

__global__ void sparse_conv3d_warp_kernel(
    const float*   __restrict__ input,    // [B, Ci, D, H, W]
    const float*   __restrict__ weight,   // [Co, Ci, K, K, K]
    const float*   __restrict__ bias,     // [Co] or nullptr
    const uint8_t* __restrict__ mask,     // [B, D, H, W]
    float*         __restrict__ output,   // [B, Co, D, H, W]
    int B, int Ci, int Co, int D, int H, int W, int K,
    bool has_bias)
{
    // Warp-level indexing
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    // Total voxels per channel
    int total_voxels = D * H * W;
    int voxel_idx = warp_id % total_voxels;
    int bco = warp_id / total_voxels;

    if (bco >= B * Co) return;

    int b = bco / Co;
    int co = bco % Co;

    int oz = voxel_idx % W;
    int oy = (voxel_idx / W) % H;
    int ox = voxel_idx / (H * W);

    int pad = K / 2;
    int K3 = K * K * K;

    float local_acc = 0.0f;

    // Parallelize over kernel volume and channels
    // total_iterations = K^3 * Ci
    int total_iters = K3 * Ci;
    
    for (int i = lane_id; i < total_iters; i += 32) {
        int ki = i / Ci;
        int ci = i % Ci;

        int kz = ki % K;
        int ky = (ki / K) % K;
        int kx = ki / (K * K);

        int ix = ox + kx - pad;
        int iy = oy + ky - pad;
        int iz = oz + kz - pad;

        if (ix >= 0 && ix < D && iy >= 0 && iy < H && iz >= 0 && iz < W) {
            if (mask[b * total_voxels + ix * H * W + iy * W + iz]) {
                float v = input[b * Ci * total_voxels + ci * total_voxels + ix * H * W + iy * W + iz];
                float w = weight[co * Ci * K3 + ci * K3 + kx * K * K + ky * K + kz];
                local_acc = fmaf(v, w, local_acc);
            }
        }
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_acc += __shfl_down_sync(0xFFFFFFFF, local_acc, offset);
    }

    // Lane 0 writes the final result
    if (lane_id == 0) {
        if (has_bias) local_acc += bias[co];
        output[b * Co * total_voxels + co * total_voxels + ox * H * W + oy * W + oz] = local_acc;
    }
}

torch::Tensor sparse_conv3d_warp(
    torch::Tensor input,    // [B, Ci, D, H, W]
    torch::Tensor weight,   // [Co, Ci, K, K, K]
    torch::Tensor bias,     // [Co] or empty
    torch::Tensor mask)     // [B, D, H, W]
{
    int B  = input.size(0), Ci = input.size(1);
    int D  = input.size(2), H  = input.size(3), W = input.size(4);
    int Co = weight.size(0);
    int K  = weight.size(2);

    auto output = torch::zeros({B, Co, D, H, W}, input.options());

    int total_voxels = B * Co * D * H * W;
    int threads = 256;
    // Each 32 threads handle one voxel
    int blocks = (total_voxels * 32 + threads - 1) / threads;

    bool has_bias = (bias.numel() == Co);

    sparse_conv3d_warp_kernel<<<blocks, threads>>>(
        input.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        has_bias ? bias.contiguous().data_ptr<float>() : nullptr,
        mask.contiguous().data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        B, Ci, Co, D, H, W, K,
        has_bias
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_conv3d_warp", &sparse_conv3d_warp, "Warp-optimized sparse 3D convolution");
}
