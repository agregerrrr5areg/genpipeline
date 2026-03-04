import torch
import numpy as np
import logging
from pathlib import Path
from genpipeline.vae_design_model import DesignVAE
from genpipeline.optimization_engine_gradient import LatentGradientOptimizer
from genpipeline.config import load_config
from genpipeline.topology.mesh_export import density_to_stl
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_batch_discovery():
    # 1. Environment Setup
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
    os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/home/dfl/miniconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    
    config = load_config("pipeline_config.json")
    
    # 2. Load the Sharpened VAE
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=32)
    checkpoint_path = Path("checkpoints/vae_best.pth")
    if not checkpoint_path.exists():
        logger.error("VAE checkpoint not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.cuda().eval()
    
    # 3. Setup the Batch Sculptor
    sculptor = LatentGradientOptimizer(vae, device='cuda', learning_rate=0.02)
    
    sim_cfg = {
        "geometry_type": "cantilever",
        "force_n": 1500.0,
        "boundary_conditions": {"fixed_face": "x_min", "load_face": "x_max", "load_dof": 2}
    }
    
    # Generate 4 random starting designs
    batch_size = 4
    z_init_batch = np.random.randn(batch_size, 32).astype(np.float32) * 0.1
    
    # 4. Sculpt the Batch
    logger.info(f"SCULPTING BATCH of {batch_size} designs simultaneously...")
    best_z_batch = sculptor.optimize_batch(
        z_init_batch, 
        sim_cfg, 
        n_steps=20, 
        markov_weight=0.1
    )
    
    # 5. Export results
    out_dir = Path("final_designs")
    out_dir.mkdir(exist_ok=True)
    
    logger.info("Exporting batched results...")
    for b in range(batch_size):
        with torch.no_grad():
            z_t = torch.from_numpy(best_z_batch[b]).float().cuda().unsqueeze(0)
            voxels = torch.sigmoid(vae.decode_logits(z_t))
            # Anti-plant ReLU filter
            voxels = torch.where(voxels > 0.35, voxels, torch.zeros_like(voxels))
            final_density = voxels.squeeze().cpu().numpy()
            
        out_path = out_dir / f"sculpted_bracket_{b}.stl"
        density_to_stl(final_density, str(out_path), threshold=0.5)
        logger.info(f"  Saved candidate {b}: {out_path}")

if __name__ == "__main__":
    run_batch_discovery()
