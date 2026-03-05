import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from .vae_design_model import DesignVAE
from .topology.simp_solver_gpu import SIMPSolverGPU
from .cuda_kernels import simp_sensitivity

logger = logging.getLogger(__name__)


class LatentGradientOptimizer:
    """
    End-to-End Differentiable Design Optimizer.
    Combines Backpropagation (Physics Gradients) with Markovian (Spatial) Regularization.
    """

    def __init__(self, vae: DesignVAE, device="cuda", learning_rate=0.05):
        self.vae = vae
        self.device = device
        self.lr = learning_rate
        self.vae.eval()

    def optimize(
        self,
        z_init: np.ndarray,
        sim_cfg: dict,
        n_steps: int = 20,
        markov_weight: float = 0.1,
        volfrac: float = 0.4,
    ):
        # Forward to batch implementation with size 1
        return self.optimize_batch(
            z_init[None, :], sim_cfg, n_steps, markov_weight
        ).squeeze()

    def optimize_batch(
        self,
        z_batch_init: np.ndarray,
        sim_cfg: dict,
        n_steps: int = 20,
        markov_weight: float = 0.1,
    ):
        """
        Optimizes a batch of latent vectors simultaneously using physics backprop.
        """
        batch_size = z_batch_init.shape[0]
        z = (
            torch.from_numpy(z_batch_init)
            .float()
            .to(self.device)
            .clone()
            .requires_grad_(True)
        )
        optimizer = torch.optim.Adam([z], lr=self.lr)

        # Setup static solvers - use configured voxel resolution (default 64)
        res = sim_cfg.get("voxel_resolution", 64)
        nx, ny, nz = res, res // 4, res // 4
        if sim_cfg.get("geometry_type") == "lbracket":
            nz = res

        solver = SIMPSolverGPU(
            nx=nx,
            ny=ny,
            nz=nz,
            boundary_conditions=sim_cfg.get("boundary_conditions"),
            dtype=torch.float32,
        )

        logger.info(f"SCULPTING BATCH: {batch_size} designs simultaneously...")

        for step in range(n_steps):
            optimizer.zero_grad()

            # 1. Forward Pass (Batch Decode)
            logits = self.vae.decode_logits(z)
            voxels_32 = F.interpolate(
                logits, size=(nx, ny, nz), mode="trilinear", align_corners=False
            )

            # ReLU Sparsity Gate (Rectification)
            sparsity_threshold = 0.2
            rectified = F.relu(torch.sigmoid(voxels_32) - sparsity_threshold)
            xPhys_batch = rectified / (1.0 - sparsity_threshold + 1e-6)

            # 2. Physics Batch Gradient (VJP)
            dc_batch = []
            for b in range(batch_size):
                xPhys = xPhys_batch[b].squeeze()
                with torch.no_grad():
                    f, fixed = solver._get_bcs(sim_cfg.get("force_n", 1000.0))
                    K = solver._assemble_K(xPhys)
                    u = solver._solve(K, f, fixed)

                    # Custom Vectorized Kernel Call
                    from .cuda_kernels import simp_sensitivity

                    dc = simp_sensitivity(
                        xPhys.flatten(),
                        u,
                        solver.Ke,
                        solver._edof_mat,
                        solver.penal,
                        nx,
                        ny,
                        nz,
                    ).view(1, 1, nx, ny, nz)
                    dc_batch.append(dc)

            dc_total = torch.cat(dc_batch, dim=0)

            # 3. Markov Regularization (Batch Total Variation)
            loss_markov = (
                torch.abs(
                    xPhys_batch[:, :, 1:, :, :] - xPhys_batch[:, :, :-1, :, :]
                ).mean()
                + torch.abs(
                    xPhys_batch[:, :, :, 1:, :] - xPhys_batch[:, :, :, :-1, :]
                ).mean()
                + torch.abs(
                    xPhys_batch[:, :, :, :, 1:] - xPhys_batch[:, :, :, :, :-1]
                ).mean()
            )

            # 4. End-to-End Update
            xPhys_batch.backward(gradient=dc_total + markov_weight * loss_markov)
            optimizer.step()

            if (step + 1) % 10 == 0:
                logger.info(
                    f"  Step {step + 1:2d} | Batch Gradient Norm: {z.grad.norm().item():.4f}"
                )

        return z.detach().cpu().numpy()
