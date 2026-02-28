/*
 * GPU Mesh Voxelisation Kernel
 *
 * Converts a triangle mesh into a binary voxel grid using ray casting
 * (Möller–Trumbore algorithm). One CUDA thread per voxel — all 32³ = 32,768
 * voxels evaluated in parallel.
 *
 * Algorithm per voxel:
 *   1. Compute voxel centre in world space
 *   2. Cast a ray in the +Z direction
 *   3. Count how many triangles the ray intersects (above the voxel centre)
 *   4. Odd count → inside mesh → occupied voxel
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define EPSILON 1e-8f


// ── Möller–Trumbore intersection ──────────────────────────────────────────────
// Ray origin: (ox, oy, oz), direction: (0, 0, 1)  [+Z axis]
// Returns true if triangle is intersected above the ray origin (t > 0)

__device__ __forceinline__ bool ray_triangle_z(
    float ox, float oy, float oz,
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z)
{
    // Edge vectors
    float e1x = v1x - v0x,  e1y = v1y - v0y,  e1z = v1z - v0z;
    float e2x = v2x - v0x,  e2y = v2y - v0y,  e2z = v2z - v0z;

    // h = D × e2  where D = (0, 0, 1)
    // → h = (-e2y, e2x, 0)
    float hx = -e2y,  hy = e2x;   // hz = 0

    // a = e1 · h
    float a = e1x * hx + e1y * hy;   // e1z * 0 = 0
    if (fabsf(a) < EPSILON) return false;   // ray parallel to triangle

    float f = 1.0f / a;

    // s = O - v0
    float sx = ox - v0x,  sy = oy - v0y,  sz = oz - v0z;

    // u = f * (s · h)
    float u = f * (sx * hx + sy * hy);
    if (u < 0.0f || u > 1.0f) return false;

    // q = s × e1
    float qx = sy * e1z - sz * e1y;
    float qy = sz * e1x - sx * e1z;
    float qz = sx * e1y - sy * e1x;

    // v = f * (D · q)  where D = (0, 0, 1)  → D·q = qz
    float v = f * qz;
    if (v < 0.0f || u + v > 1.0f) return false;

    // t = f * (e2 · q)
    float t = f * (e2x * qx + e2y * qy + e2z * qz);
    return t > EPSILON;   // only intersections in +Z direction
}


// ── Main voxelisation kernel ──────────────────────────────────────────────────

__global__ void voxelize_kernel(
    const float* __restrict__ vertices,   // [N, 3]  float32
    const int*   __restrict__ faces,      // [M, 3]  int32
    float*       __restrict__ voxels,     // [R, R, R]  float32  (output)
    int   n_faces,
    int   resolution,
    float min_x,  float min_y,  float min_z,
    float voxel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = resolution * resolution * resolution;
    if (idx >= total) return;

    // Linear index → 3-D voxel coordinates
    int xi = idx / (resolution * resolution);
    int yi = (idx / resolution) % resolution;
    int zi = idx % resolution;

    // Voxel centre in world space
    float ox = min_x + (xi + 0.5f) * voxel_size;
    float oy = min_y + (yi + 0.5f) * voxel_size;
    float oz = min_z + (zi + 0.5f) * voxel_size;

    // Count ray–triangle intersections
    int hits = 0;
    for (int f = 0; f < n_faces; ++f) {
        int i0 = faces[f * 3 + 0];
        int i1 = faces[f * 3 + 1];
        int i2 = faces[f * 3 + 2];

        if (ray_triangle_z(
                ox, oy, oz,
                vertices[i0*3], vertices[i0*3+1], vertices[i0*3+2],
                vertices[i1*3], vertices[i1*3+1], vertices[i1*3+2],
                vertices[i2*3], vertices[i2*3+1], vertices[i2*3+2]))
        {
            ++hits;
        }
    }

    // Odd number of intersections → inside mesh
    voxels[idx] = (hits & 1) ? 1.0f : 0.0f;
}


// ── C++ entry point (called from Python) ─────────────────────────────────────

torch::Tensor voxelize_cuda(
    torch::Tensor vertices,   // [N, 3] float32 CUDA
    torch::Tensor faces,      // [M, 3] int32   CUDA
    int   resolution,
    float min_x,  float min_y,  float min_z,
    float voxel_size)
{
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(faces.is_cuda(),    "faces must be a CUDA tensor");
    TORCH_CHECK(vertices.scalar_type() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(faces.scalar_type()    == torch::kInt32,   "faces must be int32");

    int total   = resolution * resolution * resolution;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    auto voxels = torch::zeros(
        {resolution, resolution, resolution},
        torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device())
    );

    voxelize_kernel<<<blocks, threads>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        voxels.data_ptr<float>(),
        (int)faces.size(0),
        resolution,
        min_x, min_y, min_z,
        voxel_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "voxelize_kernel launch failed: ", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    return voxels;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_cuda", &voxelize_cuda,
          "GPU mesh voxelisation via ray casting (CUDA)\n"
          "Args: vertices [N,3] float32, faces [M,3] int32, resolution,\n"
          "      min_x, min_y, min_z, voxel_size");
}
