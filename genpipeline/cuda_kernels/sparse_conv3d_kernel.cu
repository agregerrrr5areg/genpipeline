/*
 * Sparse 3D Convolution Kernel
 *
 * For sparse voxel grids (occupancy << 50 %), skipping zero-input positions
 * avoids the majority of multiply-add operations.  At 10 % occupancy with a
 * 3³ kernel, ~90 % of multiply-adds are skipped, giving large throughput gains
 * over cuDNN's dense conv at high resolutions (64³, 128³).
 *
 * Algorithm (output-driven):
 *   One CUDA thread per output element (b, c_out, ox, oy, oz).
 *   For each kernel position (kx, ky, kz):
 *     Look up occupancy mask at the corresponding input position.
 *     If occupied: accumulate weight × input over all c_in.
 *
 * Input:
 *   input   [B, C_in,  D, H, W]  float32  — dense voxel grid
 *   weight  [C_out, C_in, K, K, K]  float32
 *   mask    [B, D, H, W]  uint8  — 1 = occupied, 0 = empty
 *   bias    [C_out]  float32 (optional — pass empty tensor to skip)
 *
 * Output:
 *   output  [B, C_out, D, H, W]  float32
 *
 * Constraints:
 *   Kernel size K must be odd (1, 3, 5, 7).  Padding = K/2 (same-size output).
 *   float32 only.  Bias optional.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// ── Core kernel ───────────────────────────────────────────────────────────────

__global__ void sparse_conv3d_kernel(
    const float*   __restrict__ input,    // [B, Ci, D, H, W]
    const float*   __restrict__ weight,   // [Co, Ci, K, K, K]
    const float*   __restrict__ bias,     // [Co] or nullptr
    const uint8_t* __restrict__ mask,     // [B, D, H, W]
    float*         __restrict__ output,   // [B, Co, D, H, W]
    int B, int Ci, int Co, int D, int H, int W, int K,
    bool has_bias)
{
    // One thread per (b, co, ox, oy, oz)
    int oz  = blockIdx.x * blockDim.x + threadIdx.x;
    int oy  = blockIdx.y * blockDim.y + threadIdx.y;
    int bco = blockIdx.z;              // linearised b * Co + co

    if (oz >= W || oy >= H) return;

    int b   = bco / Co;
    int co  = bco % Co;
    if (b >= B) return;

    int pad = K / 2;
    int K2  = K * K;
    int K3  = K * K * K;

    for (int ox = 0; ox < D; ox++) {
        float acc = has_bias ? bias[co] : 0.0f;

        for (int kx = 0; kx < K; kx++) {
            int ix = ox + kx - pad;
            if (ix < 0 || ix >= D) continue;

            for (int ky = 0; ky < K; ky++) {
                int iy = oy + ky - pad;
                if (iy < 0 || iy >= H) continue;

                for (int kz = 0; kz < K; kz++) {
                    int iz = oz + kz - pad;
                    if (iz < 0 || iz >= W) continue;

                    // Skip if input position is empty
                    if (!mask[b * D * H * W + ix * H * W + iy * W + iz])
                        continue;

                    // Accumulate over all input channels
                    int w_base = co * Ci * K3 + kx * Ci * K2 + ky * Ci * K + kz * Ci;
                    // Note: weight layout [Co, Ci, K, K, K] flattened as
                    //   weight[co, ci, kx, ky, kz] =
                    //     weight[co*Ci*K3 + ci*K3 + kx*K2 + ky*K + kz]
                    // Re-arrange loop for cache-friendly weight access:

                    int in_base = b * Ci * D * H * W + ix * H * W + iy * W + iz;

                    for (int ci = 0; ci < Ci; ci++) {
                        float v = input[in_base + ci * D * H * W];
                        float w = weight[co * Ci * K3 + ci * K3 + kx * K2 + ky * K + kz];
                        acc = fmaf(v, w, acc);
                    }
                }
            }
        }

        output[b * Co * D * H * W + co * D * H * W + ox * H * W + oy * W + oz] = acc;
    }
}


// ── Helper: build occupancy mask from any-channel > threshold ─────────────────

__global__ void build_mask_kernel(
    const float*   __restrict__ input,   // [B, Ci, D, H, W]
    uint8_t*       __restrict__ mask,    // [B, D, H, W]
    float threshold,
    int B, int Ci, int D, int H, int W)
{
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int bx = blockIdx.z;   // b * D + ox

    if (iz >= W || iy >= H) return;

    int b  = bx / D;
    int ox = bx % D;
    if (b >= B) return;

    int spatial_idx = b * D * H * W + ox * H * W + iy * W + iz;
    uint8_t occ = 0;
    for (int ci = 0; ci < Ci; ci++) {
        if (input[b * Ci * D * H * W + ci * D * H * W + ox * H * W + iy * W + iz] > threshold) {
            occ = 1;
            break;
        }
    }
    mask[spatial_idx] = occ;
}


// ── C++ entry points ──────────────────────────────────────────────────────────

torch::Tensor build_occupancy_mask(
    torch::Tensor input,   // [B, Ci, D, H, W]
    float threshold = 0.0f)
{
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(input.dim() == 5, "input must be 5-D [B, Ci, D, H, W]");

    int B  = input.size(0), Ci = input.size(1);
    int D  = input.size(2), H  = input.size(3), W = input.size(4);

    auto mask = torch::zeros(
        {B, D, H, W},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device())
    );

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16, B * D);

    build_mask_kernel<<<blocks, threads>>>(
        input.contiguous().data_ptr<float>(),
        mask.data_ptr<uint8_t>(),
        threshold,
        B, Ci, D, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "build_mask_kernel failed: ", cudaGetErrorString(err));

    return mask;
}


torch::Tensor sparse_conv3d(
    torch::Tensor input,    // [B, Ci, D, H, W]
    torch::Tensor weight,   // [Co, Ci, K, K, K]
    torch::Tensor bias,     // [Co] or empty
    torch::Tensor mask)     // [B, D, H, W] uint8, from build_occupancy_mask
{
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(mask.is_cuda(),   "mask must be CUDA");
    TORCH_CHECK(input.dim() == 5,  "input must be 5-D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5-D [Co, Ci, K, K, K]");

    int B  = input.size(0), Ci = input.size(1);
    int D  = input.size(2), H  = input.size(3), W = input.size(4);
    int Co = weight.size(0);
    int K  = weight.size(2);

    TORCH_CHECK(K % 2 == 1, "kernel size K must be odd");
    TORCH_CHECK(weight.size(1) == Ci, "weight Ci must match input Ci");
    TORCH_CHECK(weight.size(3) == K && weight.size(4) == K,
                "weight must be [Co, Ci, K, K, K]");

    bool has_bias = (bias.numel() == Co);

    auto output = torch::zeros(
        {B, Co, D, H, W},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device())
    );

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16, B * Co);

    sparse_conv3d_kernel<<<blocks, threads>>>(
        input.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        has_bias ? bias.contiguous().data_ptr<float>() : nullptr,
        mask.contiguous().data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        B, Ci, Co, D, H, W, K,
        has_bias
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "sparse_conv3d_kernel failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_occupancy_mask", &build_occupancy_mask,
          "Build uint8 occupancy mask from dense voxel tensor\n"
          "Args: input [B,Ci,D,H,W] float32, threshold float\n"
          "Returns: mask [B,D,H,W] uint8");
    m.def("sparse_conv3d", &sparse_conv3d,
          "Sparse 3D convolution (skips empty voxels)\n"
          "Args: input, weight, bias, mask\n"
          "Returns: output [B,Co,D,H,W] float32");
}
