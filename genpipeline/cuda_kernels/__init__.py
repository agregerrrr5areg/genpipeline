import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.cpp_extension import load
import os

if "CUDA_HOME" not in os.environ and Path("/usr/local/cuda-12.8").exists():
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"

_EXT_CACHE = {}

def _get_ext(name, source_file):
    if name in _EXT_CACHE: return _EXT_CACHE[name]
    src = Path(__file__).parent / source_file
    print(f"[cuda_kernels] Compiling {name} (cached after first run)...")
    ext = load(name=name, sources=[str(src)], verbose=False, extra_cuda_cflags=["-O3", "--use_fast_math"])
    _EXT_CACHE[name] = ext
    return ext

def gpu_voxelize(vertices: np.ndarray, faces: np.ndarray, resolution: int = 32, device: str = "cuda") -> np.ndarray:
    ext = _get_ext("voxelize_ext", "voxelize_kernel.cu")
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    min_xyz, max_xyz = vertices.min(axis=0), vertices.max(axis=0)
    extent = (max_xyz - min_xyz).max()
    if extent < 1e-8: return np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxel_size = extent / resolution
    min_xyz -= voxel_size * 0.5
    verts_t = torch.from_numpy(vertices).to(device).contiguous()
    faces_t = torch.from_numpy(faces).to(device).contiguous()
    voxels = ext.voxelize_cuda(verts_t, faces_t, resolution, float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]), float(voxel_size))
    return voxels.cpu().numpy()

def gpu_marching_cubes(voxels: np.ndarray, isovalue: float = 0.5, origin: tuple = (0.0, 0.0, 0.0), voxel_size: float = 1.0, device: str = "cuda") -> tuple:
    ext = _get_ext("mc_ext", "marching_cubes_kernel.cu")
    voxels = np.asarray(voxels, dtype=np.float32)
    if voxels.ndim != 3: raise ValueError(f"voxels must be 3-D, got {voxels.shape}")
    if any(s < 2 for s in voxels.shape): return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    voxels_t = torch.from_numpy(voxels).to(device).contiguous()
    verts_t, faces_t = ext.marching_cubes_cuda(voxels_t, float(isovalue), float(origin[0]), float(origin[1]), float(origin[2]), float(voxel_size))
    return verts_t.cpu().numpy(), faces_t.cpu().numpy()

def fused_reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    ext = _get_ext("reparam_ext", "fused_reparam_kernel.cu")
    return ext.fused_reparameterize(mu.contiguous(), logvar.contiguous())

def build_occupancy_mask(input: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    ext = _get_ext("sparse_conv3d_ext", "sparse_conv3d_kernel.cu")
    return ext.build_occupancy_mask(input.contiguous(), threshold)

def simp_sensitivity(xPhys: torch.Tensor, u: torch.Tensor, Ke: torch.Tensor, 
                     edof_mat: torch.Tensor, penal: float, nx: int, ny: int, nz: int) -> torch.Tensor:
    ext = _get_ext("simp_sens_ext", "simp_sensitivity_ptx.cu")
    return ext.simp_sensitivity(xPhys.contiguous(), u.contiguous(), Ke.contiguous(), 
                                edof_mat.contiguous(), float(penal), int(nx), int(ny), int(nz))

def get_solid_voxels_simd(voxels: np.ndarray) -> np.ndarray:
    voxels_t = torch.from_numpy(voxels.astype(np.float32))
    # We compile with AVX-512 flags
    src = Path(__file__).parent / "mesher_simd.cpp"
    ext = load(name="mesher_simd", sources=[str(src)], extra_cflags=["-O3", "-mavx512f"])
    indices = ext.build_connectivity_simd(voxels_t)
    return indices.cpu().numpy().reshape(-1, 3)

class SparseConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True, occupancy_threshold: float = 0.0):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.threshold = in_channels, out_channels, kernel_size, occupancy_threshold
        ref = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        self.weight = nn.Parameter(ref.weight.data.clone())
        self.bias = nn.Parameter(ref.bias.data.clone()) if bias else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if not x.is_cuda:
            return torch.nn.functional.conv3d(x, self.weight, self.bias, padding=self.kernel_size // 2)
        if mask is None: mask = build_occupancy_mask(x, self.threshold)
        ext = _get_ext("sparse_conv3d_warp_ext", "sparse_conv3d_warp.cu")
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return ext.sparse_conv3d_warp(x.contiguous(), self.weight.contiguous(), bias.contiguous(), mask.contiguous())

__all__ = ["gpu_voxelize", "gpu_marching_cubes", "fused_reparameterize", "SparseConv3d", "build_occupancy_mask", "simp_sensitivity"]
