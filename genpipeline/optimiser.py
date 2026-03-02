import torch
import logging
from pathlib import Path
from .optimization_engine import DesignOptimizer
from .vae_design_model import DesignVAE
from .fem.voxel_fem import VoxelFEMEvaluator
from .schema import PipelineConfig

logger = logging.getLogger(__name__)

def run_optimisation(config: PipelineConfig, topo_refine: bool = False):
    """Orchestrate the design optimisation loop using the unified config."""
    logger.info(f"Starting optimisation: iterations={config.n_optimisation_iterations}, topo_refine={topo_refine}")
    
    # Initialize VAE
    vae = DesignVAE(
        input_shape=tuple(config.input_shape), 
        latent_dim=config.latent_dim
    )
    
    # Load best checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / "vae_best.pth"
    if not checkpoint_path.exists():
        logger.error(f"VAE checkpoint not found at {checkpoint_path}. Run training first.")
        return None, None
        
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(config.device)
    
    # Initialize Evaluator and Optimizer
    evaluator = VoxelFEMEvaluator(vae_model=vae)
    optimizer = DesignOptimizer(
        vae, 
        evaluator, 
        device=config.device, 
        latent_dim=config.latent_dim,
        topo_refine=topo_refine,
    )
    
    # Run optimization
    best_z, results = optimizer.run_optimisation(
        n_iterations=config.n_optimisation_iterations,
        output_dir=config.output_dir
    )
    
    return best_z, results
