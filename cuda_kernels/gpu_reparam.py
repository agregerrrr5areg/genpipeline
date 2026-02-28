"""
Fused VAE reparameterization — drop-in for torch-based reparameterize().

JIT-compiles fused_reparam_kernel.cu on first import (~10s, then cached).

Usage:
    from cuda_kernels.gpu_reparam import fused_reparameterize

    z = fused_reparameterize(mu, logvar)
    # equivalent to: z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    # but fused into one GPU kernel — ~20% faster, no intermediate tensors

Integration in DesignVAE.reparameterize():
    def reparameterize(self, mu, logvar):
        if mu.is_cuda:
            from cuda_kernels.gpu_reparam import fused_reparameterize
            return fused_reparameterize(mu, logvar)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
"""

import torch
from pathlib import Path

_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "fused_reparam_kernel.cu"
    print("[gpu_reparam] Compiling CUDA kernel (first run only, ~10s)...")
    _ext = load(
        name="reparam_ext",
        sources=[str(src)],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("[gpu_reparam] Kernel ready.")
    return _ext


def fused_reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Fused: z = mu + randn_like(mu) * exp(0.5 * logvar)

    Args:
        mu:     [...] float32 CUDA tensor
        logvar: [...] float32 CUDA tensor, same shape as mu

    Returns:
        z: [...] float32 CUDA tensor
    """
    ext = _load_ext()
    return ext.fused_reparameterize(mu.contiguous(), logvar.contiguous())
