import torch
import logging
from pathlib import Path
from .vae_design_model import VAETrainer, DesignVAE
from .schema import PipelineConfig

logger = logging.getLogger(__name__)

def train_vae(config: PipelineConfig, train_loader, val_loader):
    """Orchestrate VAE training using the unified config."""
    logger.info(f"Starting VAE training: resolution={config.voxel_resolution}, latent={config.latent_dim}")
    
    model = DesignVAE(
        input_shape=tuple(config.input_shape), 
        latent_dim=config.latent_dim
    )
    
    trainer = VAETrainer(
        model, 
        train_loader, 
        val_loader, 
        device=config.device, 
        epochs=config.epochs
    )
    
    trainer.fit(epochs=config.epochs)
    
    # Ensure checkpoint directory exists
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(config.checkpoint_dir) / "vae_best.pth"
    # VAETrainer usually saves its own best, but we could add explicit export here
    
    return model
