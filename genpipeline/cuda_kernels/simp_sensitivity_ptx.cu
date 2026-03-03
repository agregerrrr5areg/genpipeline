#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Fused SpMV Kernel for SIMP
 * Computes y = K(x) * p directly without assembling K.
 */

#define TILE_X 8
#define TILE_Y 8
#define TILE_Z 4

__global__ void fused_spmv_kernel(
    const double* __restrict__ xPhys,    // [n_elem]
    const double* __restrict__ p,        // [n_dof]
    const double* __restrict__ Ke,       // [24 * 24]
    const long*   __restrict__ edof_mat, // [n_elem * 24]
    double*       __restrict__ y,        // [n_dof]
    double penal,
    int nx, int ny, int nz,
    int n_elem)
{
    __shared__ double s_Ke[576];
    int tid = threadIdx.x + threadIdx.y * TILE_X + threadIdx.z * TILE_X * TILE_Y;
    if (tid < 576) s_Ke[tid] = Ke[tid];
    __syncthreads();

    int ix = blockIdx.x * TILE_X + threadIdx.x;
    int iy = blockIdx.y * TILE_Y + threadIdx.y;
    int iz = blockIdx.z * TILE_Z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int elem_idx = ix * ny * nz + iy * nz + iz;
    double E_e = pow(fmax(xPhys[elem_idx], 1e-3), penal);

    // Load element p values
    double p_e[24];
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        p_e[i] = p[edof_mat[elem_idx * 24 + i]];
    }

    // Local product Ke * p_e
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        double val = 0.0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            val += s_Ke[i * 24 + j] * p_e[j];
        }
        // Atomic add to global output vector y
        atomicAdd(&y[edof_mat[elem_idx * 24 + i]], E_e * val);
    }
}

torch::Tensor fused_spmv_cuda(
    torch::Tensor xPhys,
    torch::Tensor p,
    torch::Tensor Ke,
    torch::Tensor edof_mat,
    double penal,
    int nx, int ny, int nz) 
{
    int n_dof = p.size(0);
    auto y = torch::zeros_like(p);

    dim3 threads(TILE_X, TILE_Y, TILE_Z);
    dim3 blocks((nx + TILE_X - 1) / TILE_X, 
                (ny + TILE_Y - 1) / TILE_Y, 
                (nz + TILE_Z - 1) / TILE_Z);

    fused_spmv_kernel<<<blocks, threads>>>(
        xPhys.data_ptr<double>(),
        p.data_ptr<double>(),
        Ke.data_ptr<double>(),
        edof_mat.data_ptr<long>(),
        y.data_ptr<double>(),
        penal, nx, ny, nz, nx*ny*nz
    );

    return y;
}

// ── Original Sensitivity Kernel (kept below) ──────────────────────────────────

__global__ void simp_sensitivity_vec_kernel(
    const double* __restrict__ xPhys, const double* __restrict__ u,
    const double* __restrict__ Ke, const long* __restrict__ edof_mat,
    double* __restrict__ dc, double penal, int nx, int ny, int nz, int n_dof)
{
    __shared__ double s_Ke[576];
    int tid = threadIdx.x + threadIdx.y * TILE_X + threadIdx.z * TILE_X * TILE_Y;
    if (tid < 576) s_Ke[tid] = Ke[tid];
    if (tid + 256 < 576) s_Ke[tid + 256] = Ke[tid + 256];
    if (tid + 512 < 576) s_Ke[tid + 512] = Ke[tid + 512];
    __syncthreads();
    int ix = blockIdx.x * TILE_X + threadIdx.x;
    int iy = blockIdx.y * TILE_Y + threadIdx.y;
    int iz = blockIdx.z * TILE_Z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;
    int elem_idx = ix * ny * nz + iy * nz + iz;
    double x_val = xPhys[elem_idx];
    if (x_val < 1e-3) x_val = 1e-3;
    double x_penal = pow(x_val, penal - 1.0);
    double u_e[24];
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        long d_idx = edof_mat[elem_idx * 24 + i];
        u_e[i] = (d_idx >= 0 && d_idx < n_dof) ? u[d_idx] : 0.0;
    }
    double ce = 0.0;
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        double row_sum = 0.0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            row_sum += s_Ke[i * 24 + j] * u_e[j];
        }
        ce += u_e[i] * row_sum;
    }
    dc[elem_idx] = -penal * x_penal * ce;
}

torch::Tensor simp_sensitivity_cuda(
    torch::Tensor xPhys, torch::Tensor u, torch::Tensor Ke, torch::Tensor edof_mat,
    double penal, int nx, int ny, int nz) 
{
    auto dc = torch::zeros_like(xPhys);
    int n_dof = u.size(0);
    dim3 threads(TILE_X, TILE_Y, TILE_Z);
    dim3 blocks((nx + TILE_X - 1) / TILE_X, (ny + TILE_Y - 1) / TILE_Y, (nz + TILE_Z - 1) / TILE_Z);
    simp_sensitivity_vec_kernel<<<blocks, threads>>>(
        xPhys.data_ptr<double>(), u.data_ptr<double>(), Ke.data_ptr<double>(),
        edof_mat.data_ptr<long>(), dc.data_ptr<double>(), penal, nx, ny, nz, n_dof
    );
    return dc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simp_sensitivity", &simp_sensitivity_cuda, "Vectorized SIMP sensitivity calculation");
    m.def("fused_spmv", &fused_spmv_cuda, "Fused Matrix-Free SpMV for SIMP");
}
