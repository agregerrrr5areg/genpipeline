"""
Blackwell (sm_120, RTX 50 series card) / CUDA 12.8 / torch 2.10.0 compatibility notes.

cublasDgemmStridedBatched and cublasSgemmStridedBatched are broken for
batch_size >= 2 on this hardware/driver combination. This crashes gpytorch
and botorch whenever they call batched matmul on GPU tensors.

WORKAROUND: keep BoTorch GP models on CPU. GPs in Bayesian optimisation are
small (< 1000 training points) so CPU is fast enough; the pipeline bottleneck
is the FEM simulations, not the GP.

The 3D VAE in vae_design_model.py is unaffected — it uses Conv3D, not batched
GEMM — and should stay on CUDA.

Usage in optimisation_engine.py or any BoTorch code:

    from blackwell_compat import botorch_device
    # Always 'cpu' on this machine; returns 'cuda' once the bug is fixed.

    gp = SingleTaskGP(train_X.to(botorch_device), train_Y.to(botorch_device))
"""

import torch

# Change to 'cuda' once upstream fix lands (track pytorch/pytorch #XXXXX)
# Current workaround: BoTorch models on CPU for Blackwell
botorch_device = torch.device("cpu")

# Runtime check for Blackwell hardware
BLACKWELL_CHECK = (
    "Blackwell" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False
)


# Helper function to verify device compatibility
def verify_botorch_device():
    """Verify if BoTorch should use CPU or CUDA based on hardware."""
    if not BLACKWELL_CHECK:
        return torch.device("cuda")

    # Blackwell: force CPU for BoTorch
    return torch.device("cpu")


# Update device at runtime
botorch_device = verify_botorch_device()
