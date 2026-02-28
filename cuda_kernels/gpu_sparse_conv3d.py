"""
Sparse 3D Convolution — drop-in for nn.Conv3d on sparse voxel grids.

JIT-compiles sparse_conv3d_kernel.cu on first import (~10s, then cached).

Usage:
    from cuda_kernels.gpu_sparse_conv3d import SparseConv3d

    # Replace nn.Conv3d in an encoder:
    conv = SparseConv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    y = conv(x)   # x: [B, Ci, D, H, W]  — automatic mask computation

    # Or supply a pre-computed mask to amortise mask-build cost across layers:
    from cuda_kernels.gpu_sparse_conv3d import build_occupancy_mask
    mask = build_occupancy_mask(x)
    y = conv(x, mask=mask)

Performance (RTX 5080, measured):
    cuDNN (nn.Conv3d) uses Winograd/tensor-core paths and wins at 32³–64³
    regardless of occupancy.  SparseConv3d is a correct reference implementation
    and foundation for further optimisation (shared-memory tiling, warp reductions).
    May win over cuDNN at 128³+ with <5% occupancy where cuDNN memory allocation
    becomes the bottleneck.  For 32³–64³, keep nn.Conv3d.
"""

import torch
import torch.nn as nn
from pathlib import Path

_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "sparse_conv3d_kernel.cu"
    print("[gpu_sparse_conv3d] Compiling CUDA kernel (first run only, ~10s)...")
    _ext = load(
        name="sparse_conv3d_ext",
        sources=[str(src)],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("[gpu_sparse_conv3d] Kernel ready.")
    return _ext


def build_occupancy_mask(
    input: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Build a uint8 occupancy mask: 1 where any input channel > threshold.

    Args:
        input:     [B, Ci, D, H, W] float32 CUDA
        threshold: scalar (default 0.0 — any nonzero voxel is occupied)

    Returns:
        mask: [B, D, H, W] uint8 CUDA
    """
    ext = _load_ext()
    return ext.build_occupancy_mask(input.contiguous(), threshold)


class SparseConv3d(nn.Module):
    """
    Sparse 3D convolution.  API mirrors nn.Conv3d(padding_mode='zeros',
    groups=1, dilation=1).  Automatically uses same-size output (padding = K//2).

    Attributes:
        weight: [out_channels, in_channels, K, K, K]
        bias:   [out_channels] or None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        occupancy_threshold: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.threshold    = occupancy_threshold

        # Use the same initialisation as nn.Conv3d
        ref = nn.Conv3d(in_channels, out_channels, kernel_size,
                        padding=kernel_size // 2, bias=bias)
        self.weight = nn.Parameter(ref.weight.data.clone())
        self.bias   = nn.Parameter(ref.bias.data.clone()) if bias else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, Ci, D, H, W] float32 CUDA
            mask: [B, D, H, W] uint8 CUDA (optional; built automatically if None)

        Returns:
            [B, Co, D, H, W] float32 CUDA
        """
        if not x.is_cuda:
            # Fall back to plain Conv3d on CPU
            import torch.nn.functional as F
            bias = self.bias if self.bias is not None else None
            return F.conv3d(x, self.weight, bias,
                            padding=self.kernel_size // 2)

        if mask is None:
            mask = build_occupancy_mask(x, self.threshold)

        ext  = _load_ext()
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return ext.sparse_conv3d(
            x.contiguous(),
            self.weight.contiguous(),
            bias.contiguous(),
            mask.contiguous(),
        )

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"K={self.kernel_size}, threshold={self.threshold}")
