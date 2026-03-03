import torch
import logging
from pathlib import Path
from genpipeline.optimization_engine import DesignOptimizer
from genpipeline.vae_design_model import DesignVAE
from genpipeline.fem.voxel_fem import VoxelFEMEvaluator
from genpipeline.config import load_config
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    # Set environment variables for CUDA/Ninja
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
    os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/home/dfl/miniconda3/lib:" + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )

    # Ensure GPU is visible
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Debug: verify CUDA availability
    import torch

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available - will fall back to CPU")

    config = load_config("pipeline_config.json")

    # Initialize VAE
    vae = DesignVAE(input_shape=tuple(config.input_shape), latent_dim=config.latent_dim)

    # Load best checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / "vae_best.pth"
    if not checkpoint_path.exists():
        logger.error(
            f"VAE checkpoint not found at {checkpoint_path}. Run training first."
        )
        return

    checkpoint = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(config.device)
    vae.eval()

    # Initialize Evaluator (using VoxelFEM for speed over FreeCAD)
    evaluator = VoxelFEMEvaluator(vae_model=vae)

    # Initialize Optimizer
    optimizer = DesignOptimizer(
        vae,
        evaluator,
        device=config.device,
        latent_dim=config.latent_dim,
        topo_refine=True,  # We want the "refined" results but faster
    )

    # MANUALLY TWEAK OPTIMIZER FOR SPEED
    # 1. Reduce SIMP iterations for refinement
    # We need to monkeypatch or just run it with a smaller number of iterations
    # The current optimization_engine.py has hardcoded 20 iters in _evaluate_latent_batch

    # Let's run a small number of BO iterations
    n_iterations = 3
    logger.info(f"Running FAST optimization: {n_iterations} iterations, q=4")

    best_z, results = optimizer.run_optimisation(
        n_iterations=n_iterations, output_dir="./optimization_results/fast_run"
    )

    if best_z is not None:
        logger.info("Optimization complete!")
        out_path = Path("./optimization_results/fast_run")
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "best_z.npy", best_z)


if __name__ == "__main__":
    run()
