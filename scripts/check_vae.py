
import torch
import numpy as np
from genpipeline.vae_design_model import DesignVAE
from genpipeline.config import load_config
from pathlib import Path

def check():
    config = load_config("pipeline_config.json")
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=32)
    checkpoint_path = Path("checkpoints/vae_best.pth")
    if not checkpoint_path.exists():
        print("No checkpoint found")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    
    # Generate from random z
    z = torch.randn(1, 32)
    with torch.no_grad():
        logits = vae.decode_logits(z)
        voxels = torch.sigmoid(logits)
        
    print(f"Logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
    print(f"Voxels range: {voxels.min().item():.4f} to {voxels.max().item():.4f}")
    print(f"Mean voxel value: {voxels.mean().item():.4f}")
    print(f"Voxels > 0.5: {(voxels > 0.5).sum().item()}")

if __name__ == "__main__":
    check()
