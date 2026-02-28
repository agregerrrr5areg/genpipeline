"""
GPU voxelisation — drop-in replacement for trimesh-based CPU voxelisation.

JIT-compiles voxelize_kernel.cu on first import (takes ~30s, then cached).

Usage:
    from cuda_kernels.gpu_voxelize import gpu_voxelize

    voxel_grid = gpu_voxelize(vertices, faces, resolution=32)
    # returns (32, 32, 32) float32 numpy array — same as VoxelGrid.mesh_to_voxel()
"""

import numpy as np
import torch
from pathlib import Path

_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "voxelize_kernel.cu"
    print("[gpu_voxelize] Compiling CUDA kernel (first run only, ~30s)...")
    _ext = load(
        name="voxelize_ext",
        sources=[str(src)],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("[gpu_voxelize] Kernel ready.")
    return _ext


def gpu_voxelize(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Voxelise a triangle mesh on GPU.

    Args:
        vertices:   (N, 3) float array — vertex positions
        faces:      (M, 3) int array  — triangle vertex indices
        resolution: output grid resolution (resolution³ voxels)
        device:     CUDA device string

    Returns:
        (resolution, resolution, resolution) float32 binary occupancy grid
        1.0 = inside mesh,  0.0 = outside
    """
    ext = _load_ext()

    vertices = np.asarray(vertices, dtype=np.float32)
    faces    = np.asarray(faces,    dtype=np.int32)

    # Compute mesh bounds + voxel size
    min_xyz  = vertices.min(axis=0)
    max_xyz  = vertices.max(axis=0)
    extent   = (max_xyz - min_xyz).max()
    if extent < 1e-8:
        return np.zeros((resolution, resolution, resolution), dtype=np.float32)

    voxel_size = extent / resolution
    # Small outward padding so the bounding surface is captured
    padding  = voxel_size * 0.5
    min_xyz -= padding

    verts_t = torch.from_numpy(vertices).to(device).contiguous()
    faces_t = torch.from_numpy(faces).to(device).contiguous()

    voxels = ext.voxelize_cuda(
        verts_t, faces_t,
        resolution,
        float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]),
        float(voxel_size),
    )

    return voxels.cpu().numpy()
