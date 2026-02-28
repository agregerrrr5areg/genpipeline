"""
GPU Marching Cubes — drop-in replacement for skimage.measure.marching_cubes.

JIT-compiles marching_cubes_kernel.cu on first import (~30s, then cached).

Usage:
    from cuda_kernels.gpu_marching_cubes import gpu_marching_cubes

    verts, faces = gpu_marching_cubes(voxel_grid, isovalue=0.5)
    # verts: (N*3, 3) float32 numpy array — world-space vertex positions
    # faces: (N,   3) int32  numpy array — triangle indices into verts

    # With explicit world transform:
    verts, faces = gpu_marching_cubes(
        voxel_grid, isovalue=0.5,
        origin=(x0, y0, z0),   # world position of voxel [0,0,0]
        voxel_size=0.05,       # metres per voxel
    )
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

    src = Path(__file__).parent / "marching_cubes_kernel.cu"
    print("[gpu_marching_cubes] Compiling CUDA kernel (first run only, ~30s)...")
    _ext = load(
        name="mc_ext",
        sources=[str(src)],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("[gpu_marching_cubes] Kernel ready.")
    return _ext


def gpu_marching_cubes(
    voxels: np.ndarray,
    isovalue: float = 0.5,
    origin: tuple = (0.0, 0.0, 0.0),
    voxel_size: float = 1.0,
    device: str = "cuda",
) -> tuple:
    """
    Extract an isosurface mesh from a voxel occupancy grid on GPU.

    Args:
        voxels:     (X, Y, Z) float32 array — voxel values (e.g. 0/1 binary grid)
        isovalue:   surface threshold (default 0.5 for binary grids)
        origin:     (ox, oy, oz) world-space position of voxel corner [0,0,0]
        voxel_size: edge length of one voxel in world units
        device:     CUDA device string

    Returns:
        verts: (N*3, 3) float32 numpy array — vertex positions (triangle soup)
        faces: (N,   3) int32  numpy array — face indices into verts
               Every 3 consecutive vertices in verts form one triangle,
               so faces is trivially [[0,1,2],[3,4,5],...].
               Use trimesh.util.merge_vertices() to weld shared vertices.
    """
    ext = _load_ext()

    voxels = np.asarray(voxels, dtype=np.float32)
    if voxels.ndim != 3:
        raise ValueError(f"voxels must be 3-D, got shape {voxels.shape}")

    X, Y, Z = voxels.shape
    if X < 2 or Y < 2 or Z < 2:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        )

    voxels_t = torch.from_numpy(voxels).to(device).contiguous()

    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

    verts_t, faces_t = ext.marching_cubes_cuda(
        voxels_t,
        float(isovalue),
        ox, oy, oz,
        float(voxel_size),
    )

    verts = verts_t.cpu().numpy()   # (N*3, 3) float32
    faces = faces_t.cpu().numpy()   # (N,   3) int32

    return verts, faces
