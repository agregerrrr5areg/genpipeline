
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Optimized SIMP Sensitivity Kernel
 * Uses Shared Memory Tiling and PTX inline assembly for Blackwell register efficiency.
 * 
 * Sensitivity dc_e = -p * x_e^(p-1) * u_e^T * Ke * u_e
 */

#define TILE_DIM 4
#define ELEMENTS_PER_BLOCK (TILE_DIM * TILE_DIM * TILE_DIM)

__global__ void simp_sensitivity_ptx_kernel(
    const float* __restrict__ xPhys,    // [nx * ny * nz]
    const float* __restrict__ u,        // [n_dof]
    const float* __restrict__ Ke,       // [24 * 24]
    const long*  __restrict__ edof_mat, // [n_elem * 24]
    float*       __restrict__ dc,       // [n_elem]
    float penal,
    int nx, int ny, int nz)
{
    // Shared memory for element stiffness matrix Ke (24x24)
    __shared__ float s_Ke[24 * 24];
    
    // Cooperative load of Ke into shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    if (tid < 576) {
        s_Ke[tid] = Ke[tid];
    }
    __syncthreads();

    // Map thread to voxel
    int ix = blockIdx.x * TILE_DIM + threadIdx.x;
    int iy = blockIdx.y * TILE_DIM + threadIdx.y;
    int iz = blockIdx.z * TILE_DIM + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int elem_idx = ix * ny * nz + iy * nz + iz;
    
    // 1. Calculate x_e^(p-1) using PTX for fast math
    float x_val = xPhys[elem_idx];
    if (x_val < 1e-3f) x_val = 1e-3f;
    
    float p_minus_1 = penal - 1.0f;
    float x_penal;
    
    // PTX: Use lg2 and ex2 for faster power calculation on Blackwell
    asm("{
	"
        " .reg .f32 t;
	"
        " lg2.approx.f32 t, %1;
	"
        " mul.f32 t, t, %2;
	"
        " ex2.approx.f32 %0, t;
	"
        "}" : "=f"(x_penal) : "f"(x_val), "f"(p_minus_1));

    // 2. Load element displacements u_e (24 DOFs)
    float u_e[24];
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        long dof_idx = edof_mat[elem_idx * 24 + i];
        u_e[i] = u[dof_idx];
    }

    // 3. Compute ce = u_e^T * Ke * u_e
    float ce = 0.0f;
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        float tmp = 0.0f;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            // Fused Multiply-Add
            tmp += s_Ke[i * 24 + j] * u_e[j];
        }
        ce += u_e[i] * tmp;
    }

    // 4. Final sensitivity
    dc[elem_idx] = -penal * x_penal * ce;
}

torch::Tensor simp_sensitivity_cuda(
    torch::Tensor xPhys,
    torch::Tensor u,
    torch::Tensor Ke,
    torch::Tensor edof_mat,
    float penal,
    int nx, int ny, int nz) 
{
    auto dc = torch::zeros({nx * ny * nz}, xPhys.options());

    dim3 threads(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 blocks((nx + TILE_DIM - 1) / TILE_DIM, 
                (ny + TILE_DIM - 1) / TILE_DIM, 
                (nz + TILE_DIM - 1) / TILE_DIM);

    simp_sensitivity_ptx_kernel<<<blocks, threads>>>(
        xPhys.data_ptr<float>(),
        u.data_ptr<float>(),
        Ke.data_ptr<float>(),
        edof_mat.data_ptr<long>(),
        dc.data_ptr<float>(),
        penal, nx, ny, nz
    );

    return dc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simp_sensitivity", &simp_sensitivity_cuda, "Optimized SIMP sensitivity calculation (CUDA/PTX)");
}
