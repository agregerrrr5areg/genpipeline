/**
 * SIMP PCG CUDA Kernels - GPU-accelerated PCG solver
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// Kernel: Sparse Matrix-Vector Multiply (CSR)
// ============================================================================

__global__ void spmv_csr_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* x,
    float* y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_idx[i]];
        }
        
        y[row] = sum;
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

torch::Tensor spmv_csr(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x
) {
    int num_rows = row_ptr.size(0) - 1;
    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;
    
    auto y = torch::zeros_like(x);
    
    spmv_csr_kernel<<<num_blocks, block_size>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        values.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_rows
    );
    
    cudaDeviceSynchronize();
    return y;
}

// PCG Solver
torch::Tensor pcg_solve(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor diagonal,
    torch::Tensor b,
    torch::Tensor x0,
    torch::Tensor fixed_dofs,
    int max_iter,
    double tol
) {
    int n = b.size(0);
    
    auto x = x0.clone();
    auto r = b.clone();
    auto p = torch::zeros_like(b);
    auto Ap = torch::zeros_like(b);
    
    auto fixed_acc = fixed_dofs.accessor<int, 1>();
    for (int i = 0; i < fixed_dofs.size(0); i++) {
        r[fixed_acc[i]] = 0;
    }
    
    auto z = r / diagonal;
    p = z.clone();
    
    double rz = torch::dot(r, z).item<double>();
    
    for (int iter = 0; iter < max_iter; iter++) {
        Ap = spmv_csr(row_ptr, col_idx, values, p);
        
        for (int i = 0; i < fixed_dofs.size(0); i++) {
            Ap[fixed_acc[i]] = 0;
        }
        
        double denom = torch::dot(p, Ap).item<double>();
        if (fabs(denom) < 1e-30) break;
        
        double alpha = rz / denom;
        x = x + alpha * p;
        r = r - alpha * Ap;
        
        for (int i = 0; i < fixed_dofs.size(0); i++) {
            r[fixed_acc[i]] = 0;
        }
        
        if (torch::norm(r).item<double>() < tol * torch::norm(b).item<double>()) {
            break;
        }
        
        z = r / diagonal;
        double rz_new = torch::dot(r, z).item<double>();
        double beta = rz_new / rz;
        p = z + beta * p;
        rz = rz_new;
    }
    
    return x;
}

// ============================================================================
// TORCH_BINDING
// ============================================================================

TORCH_LIBRARY(genpipeline_cuda, m) {
    m.def("spmv_csr(Tensor row_ptr, Tensor col_idx, Tensor values, Tensor x) -> Tensor");
    m.def("pcg_solve(Tensor row_ptr, Tensor col_idx, Tensor values, Tensor diagonal, Tensor b, Tensor x0, Tensor fixed_dofs, int max_iter, float tol) -> Tensor");
}

TORCH_LIBRARY_IMPL(genpipeline_cuda, CUDA, m) {
    m.impl("spmv_csr", &spmv_csr);
    m.impl("pcg_solve", &pcg_solve);
}
