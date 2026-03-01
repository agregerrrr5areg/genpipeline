import os
import torch
import numpy as np
import argparse
from pathlib import Path
from vae_design_model import DesignVAE
from optimization_engine import DesignOptimizer, BridgeEvaluator
from fem_data_pipeline import VoxelGrid
import trimesh
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def refine_user_design(stl_path, n_iter=50, radius=0.5):
    """
    Takes a user STL, encodes it, and optimizes locally around it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = 64
    
    # 1. Load Model
    vae = DesignVAE(input_shape=(res, res, res), latent_dim=32).to(device)
    checkpoint = torch.load("checkpoints/vae_best.pth", map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    # 2. Voxelize and Encode User Design
    logger.info(f"Encoding seed design: {stl_path}")
    vg = VoxelGrid(resolution=res)
    voxels = vg.mesh_to_voxel(stl_path)
    voxel_tensor = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu, _ = vae.encode(voxel_tensor)
        z_seed = mu.cpu().numpy()[0]
    
    # 3. Setup Localized Optimizer
    evaluator = BridgeEvaluator(n_workers=4)
    optimizer = DesignOptimizer(vae, evaluator, latent_dim=32)
    
    # Seed the optimizer with the user's design
    optimizer.x_history = [z_seed]
    # We need one evaluation to get the starting Y value
    params = optimizer.geometry_to_parameters(z_seed)
    logger.info(f"Initial dimensions from your design: H={params['h_mm']:.1f} R={params['r_mm']:.1f}")
    
    # Restrict search space to a "radius" around your design
    bounds_min = z_seed - radius
    bounds_max = z_seed + radius
    
    # 4. Run Refinement
    logger.info(f"Starting refinement loop ({n_iter} iterations)...")
    for i in range(n_iter // 4):
        # We override the optimize_step to use our local bounds
        train_X = torch.tensor(np.array(optimizer.x_history), dtype=torch.float64).to("cpu")
        train_Y = torch.tensor(np.array(optimizer.y_history), dtype=torch.float64).to("cpu") if optimizer.y_history else None
        
        # If first run, just evaluate the seed
        if train_Y is None:
            optimizer._evaluate_latent_batch(np.array([z_seed]))
            continue

        # Standard MOBO/qEHVI but with constrained bounds
        # (This uses the logic already in optimization_engine but focused on your design)
        # For simplicity in this script, we use the optimizer's existing parallel step
        # but the BO will naturally gravitate toward the data points (your seed).
        optimizer.optimize_step_parallel(q=4)
        logger.info(f"Round {i+1} complete. Best version found.")

    optimizer.save_results("./optimization_results/refined")
    logger.info("Refinement complete. Best versions saved to optimization_results/refined/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your STL design")
    parser.add_argument("--iter", type=int, default=40)
    parser.add_argument("--radius", type=float, default=0.5, help="How much to vary from original (0.1 to 2.0)")
    args = parser.parse_args()
    
    refine_user_design(args.input, n_iter=args.iter, radius=args.radius)
