import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.cpp_extension import load
import os

# ============================================================================
# CUDA Environment Setup for Blackwell (RTX 50 series, sm_120)
# ============================================================================

_CUDA_PATHS = [
    os.environ.get("CUDA_HOME", ""),
    "/usr/local/cuda-12.8",
    "/usr/local/cuda-12.7",
    "/usr/local/cuda-12",
    "/usr/local/cuda-11.8",
]

# Force CUDA 12.8 for Blackwell (sm_120) support
# Must be done BEFORE importing torch.utils.cpp_extension
_CUDA_HOME = None
for p in _CUDA_PATHS:
    if p and Path(p).exists():
        nvcc_path = Path(p) / "bin" / "nvcc"
        if nvcc_path.exists():
            _CUDA_HOME = p
            break

# Force CUDA_HOME in environment before any CUDA operations
if _CUDA_HOME:
    os.environ["CUDA_HOME"] = _CUDA_HOME
    os.environ["PATH"] = f"{_CUDA_HOME}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_CUDA_HOME}/lib64:" + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )
    # Also set for PyTorch's cpp_extension
    import torch.utils.cpp_extension as _cpp_ext

    _cpp_ext.CUDA_HOME = _CUDA_HOME

# Find conda GCC and set up library path for proper CXXABI
_CONDA_PREFIX = os.path.dirname(os.path.dirname(sys.executable))
_CONDA_LIB = f"{_CONDA_PREFIX}/lib"
if Path(_CONDA_LIB).exists():
    os.environ["LD_LIBRARY_PATH"] = (
        _CONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

# Check for GCC with CXXABI_1.3.15
_CAN_COMPILE_CUDA = False
_GCC_VERSION = "unknown"

try:
    import subprocess

    # Find GCC
    gcc_path = os.environ.get("CC", os.environ.get("gcc", "gcc"))
    result = subprocess.run([gcc_path, "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        gcc_version = result.stdout.split("\n")[0]
        version_match = gcc_version.split()[-1] if gcc_version.split() else "unknown"
        _GCC_VERSION = version_match

        # Check major version
        major_version = (
            int(version_match.split(".")[0]) if version_match[0].isdigit() else 0
        )

        # Check if we have compatible libstdc++
        libstdcpp_paths = [
            f"{_CONDA_PREFIX}/lib/libstdc++.so.6",
            "/home/dfl/miniconda3/lib/libstdc++.so.6",
        ]
        has_cxxabi_1_3_15 = False
        for lib_path in libstdcpp_paths:
            if Path(lib_path).exists():
                try:
                    strings_result = subprocess.run(
                        ["strings", lib_path], capture_output=True, text=True
                    )
                    if "CXXABI_1.3.15" in strings_result.stdout:
                        has_cxxabi_1_3_15 = True
                        break
                except:
                    pass

        if major_version >= 11 or has_cxxabi_1_3_15:
            # Verify source files exist
            cuda_kernels_dir = Path(__file__).parent
            required_files = [
                "fused_reparam_kernel.cu",
                "simp_sensitivity_ptx.cu",
                "voxelize_kernel.cu",
            ]
            if all((cuda_kernels_dir / f).exists() for f in required_files):
                _CAN_COMPILE_CUDA = True
                print(
                    f"[cuda_kernels] CUDA {torch.version.cuda} with GCC {_GCC_VERSION} - custom kernels available"
                )
        else:
            print(
                f"[cuda_kernels] GCC {version_match} incompatible, using PyTorch fallbacks"
            )
except Exception as e:
    print(f"[cuda_kernels] CUDA compile check: {e}")

# Architecture flags for Blackwell (RTX 50 series) and backward compatibility
# sm_120 = Blackwell, sm_90 = Hopper, sm_89 = Ada Lovelace, sm_80 = Ampere
_ARCH_FLAGS = [
    "-gencode=arch=compute_120,code=sm_120",  # Blackwell (requires CUDA 12.8+)
    "-gencode=arch=compute_90,code=sm_90",  # Hopper
    "-gencode=arch=compute_89,code=sm_89",  # Ada Lovelace
    "-gencode=arch=compute_80,code=sm_80",  # Ampere
]
_EXT_CACHE = {}


def _get_ext(name, source_file, extra_cflags=None):
    """Load or build CUDA extension with Blackwell-optimized flags."""
    if not _CAN_COMPILE_CUDA:
        raise RuntimeError(
            f"CUDA kernel compilation not available. "
            f"CUDA_HOME={_CUDA_HOME} but GCC version may be incompatible."
        )

    if name in _EXT_CACHE:
        return _EXT_CACHE[name]

    src = Path(__file__).parent / source_file
    print(f"[cuda_kernels] Compiling {name} with CUDA {_CUDA_HOME}...")

    base_flags = ["-O3", "--use_fast_math", "-lineinfo"]
    cflags = base_flags + _ARCH_FLAGS
    if extra_cflags:
        cflags.extend(extra_cflags)

    ext = load(name=name, sources=[str(src)], verbose=False, extra_cuda_cflags=cflags)
    _EXT_CACHE[name] = ext
    return ext


# ============================================================================
# Fallback Implementations (Pure PyTorch - highly optimized)
# ============================================================================


def _voxelize_fallback(vertices, faces, resolution, device):
    """Pure PyTorch voxelization fallback."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Use trimesh voxelize
    voxelized = mesh.voxelized(pitch=mesh.bounding_box.extents.max() / resolution)
    voxel_grid = voxelized.matrix

    # Pad or crop to target resolution
    result = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    min_dim = min(voxel_grid.shape)
    result[:min_dim, :min_dim, :min_dim] = voxel_grid[:min_dim, :min_dim, :min_dim]

    return result


def gpu_voxelize(
    vertices: np.ndarray, faces: np.ndarray, resolution: int = 32, device: str = "cuda"
) -> np.ndarray:
    """Voxelize mesh to grid - uses CUDA if available, falls back to CPU."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    min_xyz, max_xyz = vertices.min(axis=0), vertices.max(axis=0)
    extent = (max_xyz - min_xyz).max()
    if extent < 1e-8:
        return np.zeros((resolution, resolution, resolution), dtype=np.float32)

    if _CAN_COMPILE_CUDA:
        try:
            ext = _get_ext("voxelize_ext", "voxelize_kernel.cu")
            voxel_size = extent / resolution
            min_xyz -= voxel_size * 0.5
            verts_t = torch.from_numpy(vertices).to(device).contiguous()
            faces_t = torch.from_numpy(faces).to(device).contiguous()
            voxels = ext.voxelize_cuda(
                verts_t,
                faces_t,
                resolution,
                float(min_xyz[0]),
                float(min_xyz[1]),
                float(min_xyz[2]),
                float(voxel_size),
            )
            return voxels.cpu().numpy()
        except Exception as e:
            print(f"[cuda_kernels] CUDA voxelize failed: {e}, using fallback")

    return _voxelize_fallback(vertices, faces, resolution, device)


def _marching_cubes_fallback(voxels, isovalue, origin, voxel_size, device):
    """Pure PyTorch marching cubes fallback using skimage."""
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(
        voxels, level=isovalue, spacing=(voxel_size, voxel_size, voxel_size)
    )
    verts += np.array(origin)
    return verts.astype(np.float32), faces.astype(np.int32)


def gpu_marching_cubes(
    voxels: np.ndarray,
    isovalue: float = 0.5,
    origin: tuple = (0.0, 0.0, 0.0),
    voxel_size: float = 1.0,
    device: str = "cuda",
) -> tuple:
    """Marching cubes - uses CUDA if available, falls back to CPU."""
    voxels = np.asarray(voxels, dtype=np.float32)
    if voxels.ndim != 3:
        raise ValueError(f"voxels must be 3-D, got {voxels.shape}")
    if any(s < 2 for s in voxels.shape):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    if _CAN_COMPILE_CUDA:
        try:
            ext = _get_ext("mc_ext", "marching_cubes_kernel.cu")
            voxels_t = torch.from_numpy(voxels).to(device).contiguous()
            verts_t, faces_t = ext.marching_cubes_cuda(
                voxels_t,
                float(isovalue),
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                float(voxel_size),
            )
            return verts_t.cpu().numpy(), faces_t.cpu().numpy()
        except Exception as e:
            print(f"[cuda_kernels] CUDA marching cubes failed: {e}, using fallback")

    return _marching_cubes_fallback(voxels, isovalue, origin, voxel_size, device)


def fused_reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Fused VAE reparameterization: z = mu + eps * exp(0.5*logvar)

    Uses CUDA kernel if available, falls back to optimized PyTorch.
    """
    if _CAN_COMPILE_CUDA:
        try:
            ext = _get_ext("reparam_ext", "fused_reparameter_kernel.cu")
            return ext.fused_reparameterize(mu.contiguous(), logvar.contiguous())
        except Exception as e:
            print(f"[cuda_kernels] CUDA reparameterize failed: {e}, using fallback")

    # Optimized PyTorch fallback - single fused kernel
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def build_occupancy_mask(input: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Build occupancy mask from density grid."""
    if _CAN_COMPILE_CUDA:
        try:
            ext = _get_ext("sparse_conv3d_ext", "sparse_conv3d_kernel.cu")
            return ext.build_occupancy_mask(input.contiguous(), threshold)
        except Exception as e:
            print(f"[cuda_kernels] CUDA occupancy mask failed: {e}, using fallback")

    # PyTorch fallback - use tuple dims to reduce multiple at once
    # input: (B, C, D, H, W) or (C, D, H, W)
    ndims = input.dim()
    if ndims == 5:
        return (input > threshold).any(dim=(1, 2, 3, 4))
    elif ndims == 4:
        return (input > threshold).any(dim=(1, 2, 3))
    elif ndims == 3:
        return (input > threshold).any(dim=(1, 2))
    else:
        return (input > threshold).any()


def simp_sensitivity(
    xPhys: torch.Tensor,
    u: torch.Tensor,
    Ke: torch.Tensor,
    edof_mat: torch.Tensor,
    penal: float,
    nx: int,
    ny: int,
    nz: int,
) -> torch.Tensor:
    """SIMP sensitivity calculation - uses CUDA if available."""
    if _CAN_COMPILE_CUDA:
        try:
            ext = _get_ext("simp_sens_ext", "simp_sensitivity_ptx.cu")
            return ext.simp_sensitivity(
                xPhys.contiguous(),
                u.contiguous(),
                Ke.contiguous(),
                edof_mat.contiguous(),
                float(penal),
                int(nx),
                int(ny),
                int(nz),
            )
        except Exception as e:
            print(f"[cuda_kernels] CUDA SIMP sensitivity failed: {e}, using fallback")

    # PyTorch fallback - vectorized computation
    dc = torch.zeros_like(xPhys)
    x_penal = torch.clamp(xPhys, min=1e-3) ** (penal - 1.0)

    # Compute sensitivities per element
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                elem_idx = i * ny * nz + j * nz + k
                u_e = u[edof_mat[elem_idx]]
                ce = u_e @ Ke @ u_e
                dc[i, j, k] = -penal * x_penal[i, j, k] * ce

    return dc


def fused_spmv(
    xPhys: torch.Tensor,
    p: torch.Tensor,
    Ke: torch.Tensor,
    edof_mat: torch.Tensor,
    penal: float,
    nx: int,
    ny: int,
    nz: int,
) -> torch.Tensor:
    """Fused SpMV for SIMP."""
    if _CAN_COMPILE_CUDA:
        try:
            ext = _get_ext("simp_sens_ext", "simp_sensitivity_ptx.cu")
            return ext.fused_spmv(
                xPhys.contiguous(),
                p.contiguous(),
                Ke.contiguous(),
                edof_mat.contiguous(),
                float(penal),
                int(nx),
                int(ny),
                int(nz),
            )
        except Exception as e:
            print(f"[cuda_kernels] CUDA SpMV failed: {e}, using fallback")

    # PyTorch fallback
    y = torch.zeros_like(p)
    x_penal = torch.clamp(xPhys, min=1e-3) ** penal

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                elem_idx = i * ny * nz + j * nz + k
                p_e = p[edof_mat[elem_idx]]
                Ke_e = Ke
                y[edof_mat[elem_idx]] += x_penal[i, j, k] * (Ke_e @ p_e)

    return y


def get_solid_voxels_simd(voxels: np.ndarray) -> np.ndarray:
    """Build element connectivity using SIMD."""
    voxels_t = torch.from_numpy(voxels.astype(np.float32))

    try:
        src = Path(__file__).parent / "mesher_simd.cpp"
        ext = load(
            name="mesher_simd", sources=[str(src)], extra_cflags=["-O3", "-mavx512f"]
        )
        indices = ext.build_connectivity_simd(voxels_t)
        return indices.cpu().numpy().reshape(-1, 3)
    except Exception as e:
        print(f"[cuda_kernels] SIMD mesher failed: {e}, using numpy fallback")

    # Simple numpy fallback
    solid = np.where(voxels > 0.5)
    indices = np.column_stack(solid).astype(np.int32)
    return indices


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        occupancy_threshold: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold = occupancy_threshold

        ref = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias
        )
        self.weight = nn.Parameter(ref.weight.data.clone())
        self.bias = nn.Parameter(ref.bias.data.clone()) if bias else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if not x.is_cuda:
            return nn.functional.conv3d(
                x, self.weight, self.bias, padding=self.kernel_size // 2
            )
        if mask is None:
            mask = build_occupancy_mask(x, self.threshold)

        if _CAN_COMPILE_CUDA:
            try:
                ext = _get_ext("sparse_conv3d_warp_ext", "sparse_conv3d_warp.cu")
                bias = (
                    self.bias
                    if self.bias is not None
                    else torch.empty(0, device=x.device)
                )
                return ext.sparse_conv3d_warp(
                    x.contiguous(),
                    self.weight.contiguous(),
                    bias.contiguous(),
                    mask.contiguous(),
                )
            except Exception as e:
                print(f"[cuda_kernels] CUDA sparse conv failed: {e}")

        # PyTorch fallback
        return nn.functional.conv3d(
            x, self.weight, self.bias, padding=self.kernel_size // 2
        )


__all__ = [
    "gpu_voxelize",
    "gpu_marching_cubes",
    "fused_reparameterize",
    "SparseConv3d",
    "build_occupancy_mask",
    "simp_sensitivity",
    "get_solid_voxels_simd",
]
