/*
 * dense_fem_cuda.cu - Raw CUDA kernels for fast FEM on GPU
 * Optimized for Blackwell (sm_120)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Element stiffness matrix constant (24x24)
__constant__ float Ke_global[576];

// Assemble global stiffness matrix (dense) - OPTIMIZED
__global__ void assemble_dense_K(
    const float* xPhys,
    float* K,
    int nx, int ny, int nz,
    int n_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_elem = nx * ny * nz;
    
    if (tid >= n_elem) return;
    
    int iz = tid % nz;
    int iy = (tid / nz) % ny;
    int ix = tid / (ny * nz);
    
    // SIMP penalization
    float rho = xPhys[tid];
    float E = powf(rho, 3.0f);
    
    // Node indices for this element
    int nodes[8];
    for (int i = 0; i < 8; i++) {
        int dx = (i & 1);
        int dy = (i & 2) >> 1;
        int dz = (i & 4) >> 2;
        nodes[i] = (ix + dx) * (ny + 1) * (nz + 1) + 
                   (iy + dy) * (nz + 1) + 
                   (iz + dz);
    }
    
    // Assemble 24x24 element stiffness
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int dof_i = 3 * nodes[i] + di;
                    int dof_j = 3 * nodes[j] + dj;
                    int ke_i = 3 * i + di;
                    int ke_j = 3 * j + dj;
                    atomicAdd(&K[dof_i * 3 * n_nodes + dof_j], E * Ke_global[ke_i * 24 + ke_j]);
                }
            }
        }
    }
}

// Sensitivity computation - FUSED with element extraction
__global__ void compute_sensitivity(
    const float* u,
    const float* xPhys,
    const float* Ke,
    const int* edof_mat,
    float* dc,
    int n_elem,
    float penal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elem) return;
    
    // Compute ce = u_e' * Ke * u_e
    float ce = 0.0f;
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            int dof_i = edof_mat[tid * 24 + i];
            int dof_j = edof_mat[tid * 24 + j];
            ce += u[dof_i] * Ke[i * 24 + j] * u[dof_j];
        }
    }
    
    // dc = -penal * x^(penal-1) * ce
    float x = xPhys[tid];
    dc[tid] = -penal * powf(fmaxf(x, 1e-3f), penal - 1.0f) * ce;
}

// Apply boundary conditions - Set rows/cols to identity + zero force
__global__ void apply_bc(
    float* K,
    float* f,
    const int* fixed_dofs,
    int n_fixed,
    int n_dof
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_fixed) return;
    
    int dof = fixed_dofs[tid];
    
    // Zero row and column
    for (int j = 0; j < n_dof; j++) {
        K[dof * n_dof + j] = 0.0f;
        K[j * n_dof + dof] = 0.0f;
    }
    // Set diagonal to 1
    K[dof * n_dof + dof] = 1.0f;
    // Zero force
    f[dof] = 0.0f;
}

// PyTorch bindings
torch::Tensor assemble_stiffness_dense(
    torch::Tensor xPhys, 
    torch::Tensor Ke, 
    int nx, int ny, int nz
) {
    int n_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    int n_dof = 3 * n_nodes;
    int n_elem = nx * ny * nz;
    
    auto K = torch::zeros({n_dof, n_dof}, xPhys.options());
    
    // Copy Ke to constant memory
    auto Ke_cpu = Ke.cpu();
    cudaMemcpyToSymbol(Ke_global, Ke_cpu.data_ptr<float>(), 576 * sizeof(float));
    
    int blocks = (n_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assemble_dense_K<<<blocks, BLOCK_SIZE>>>(
        xPhys.data_ptr<float>(),
        K.data_ptr<float>(),
        nx, ny, nz, n_nodes
    );
    cudaDeviceSynchronize();
    
    return K;
}

torch::Tensor compute_sensitivity_cuda(
    torch::Tensor u,
    torch::Tensor xPhys,
    torch::Tensor Ke,
    torch::Tensor edof_mat,
    float penal
) {
    int n_elem = xPhys.size(0);
    auto dc = torch::empty({n_elem}, xPhys.options());
    
    int blocks = (n_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_sensitivity<<<blocks, BLOCK_SIZE>>>(
        u.data_ptr<float>(),
        xPhys.data_ptr<float>(),
        Ke.data_ptr<float>(),
        edof_mat.data_ptr<int>(),
        dc.data_ptr<float>(),
        n_elem, penal
    );
    cudaDeviceSynchronize();
    
    return dc;
}

void apply_bc_cuda(
    torch::Tensor K,
    torch::Tensor f,
    torch::Tensor fixed_dofs
) {
    int n_fixed = fixed_dofs.size(0);
    int n_dof = K.size(0);
    
    int blocks = (n_fixed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_bc<<<blocks, BLOCK_SIZE>>>(
        K.data_ptr<float>(),
        f.data_ptr<float>(),
        fixed_dofs.data_ptr<int>(),
        n_fixed,
        n_dof
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assemble_stiffness_dense", &assemble_stiffness_dense, "Dense FEM assembly");
    m.def("compute_sensitivity_cuda", &compute_sensitivity_cuda, "GPU sensitivity");
    m.def("apply_bc_cuda", &apply_bc_cuda, "Apply BC");
}
