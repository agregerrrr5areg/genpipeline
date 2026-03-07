import torch
import logging
from pathlib import Path
from .optimization_engine import DesignOptimizer
from .vae_design_model import DesignVAE
from .fem.voxel_fem import VoxelFEMEvaluator
from .schema import PipelineConfig

logger = logging.getLogger(__name__)


def run_optimisation(config: PipelineConfig, topo_refine: bool = False, q: int = None):
    """Orchestrate the design optimisation loop using the unified config."""
    logger.info(
        f"Starting optimisation: iterations={config.n_optimisation_iterations}, topo_refine={topo_refine}"
    )

    # Debug: verify CUDA
    logger.info(
        f"Config device: {config.device}, CUDA available: {torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize VAE
    vae = DesignVAE(input_shape=tuple(config.input_shape), latent_dim=config.latent_dim)

    # Load best checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / "vae_best.pth"
    if not checkpoint_path.exists():
        logger.error(
            f"VAE checkpoint not found at {checkpoint_path}. Run training first."
        )
        return None, None

    checkpoint = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(config.device)
    vae_device = next(vae.parameters()).device
    logger.info(f"VAE loaded to device: {vae_device}")

    # Initialize Evaluator and Optimizer — GPU FEM replaces ccx (~8s→<2s per eval)
    use_gpu_fem = torch.cuda.is_available()
    logger.info(f"FEM backend: {'GPU (vectorised)' if use_gpu_fem else 'CalculiX'}")
    evaluator = VoxelFEMEvaluator(vae_model=vae, use_gpu=use_gpu_fem)
    optimizer = DesignOptimizer(
        vae,
        evaluator,
        device=config.device,
        latent_dim=config.latent_dim,
        topo_refine=topo_refine,
    )

    import os
    effective_q = q if q is not None else os.cpu_count()
    logger.info(f"Parallel FEM workers (q): {effective_q}")

    # Run optimization
    best_z, results = optimizer.run_optimisation(
        n_iterations=config.n_optimisation_iterations, output_dir=config.output_dir,
        q=effective_q,
    )

    return best_z, results
