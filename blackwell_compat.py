"""
Blackwell (sm_120, RTX 5080) / CUDA 12.8 / torch 2.10.0 compatibility notes.

cublasDgemmStridedBatched and cublasSgemmStridedBatched are broken for
batch_size >= 2 on this hardware/driver combination. This crashes gpytorch
and botorch whenever they call batched matmul on GPU tensors.

WORKAROUND: keep BoTorch GP models on CPU. GPs in Bayesian optimisation are
small (< 1000 training points) so CPU is fast enough; the pipeline bottleneck
is the FEM simulations, not the GP.

The 3D VAE in vae_design_model.py is unaffected — it uses Conv3D, not batched
GEMM — and should stay on CUDA.

Usage in optimization_engine.py or any BoTorch code:

    from blackwell_compat import botorch_device
    # Always 'cpu' on this machine; returns 'cuda' once the bug is fixed.

    gp = SingleTaskGP(train_X.to(botorch_device), train_Y.to(botorch_device))
"""

import torch

# Change to 'cuda' once upstream fix lands (track pytorch/pytorch #XXXXX)
botorch_device = torch.device("cpu")
