#!/usr/bin/env python3
"""
Beta-VAE Ablation Study
Trains two VAE models: beta=1.0 (current) vs beta=0.05 (recommended)
Compares reconstruction quality and latent space properties.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
import sys

sys.path.insert(0, "/home/genpipeline")

from genpipeline.vae_design_model import DesignVAE, VAETrainer
from genpipeline.schema import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_dict(d):
    """Remove None values from dict for PyTorch collate."""
    return {k: v for k, v in d.items() if v is not None}


def custom_collate(batch):
    """Custom collate that filters None values from dicts."""
    from torch.utils.data._utils.collate import default_collate
    
    # Clean each sample
    cleaned_batch = []
    for sample in batch:
        cleaned = {}
        for k, v in sample.items():
            if isinstance(v, dict):
                cleaned[k] = clean_dict(v)
            else:
                cleaned[k] = v
        cleaned_batch.append(cleaned)
    
    return default_collate(cleaned_batch)


def train_with_beta(beta_value, dataset_path, output_dir, epochs=50):
    """Train VAE with specific beta value."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training VAE with beta={beta_value}")
    logger.info(f"{'=' * 70}\n")

    # Load dataset
    checkpoint = torch.load(dataset_path, weights_only=False)
    train_dataset = checkpoint["train_loader"].dataset
    val_dataset = checkpoint["val_loader"].dataset

    # Use custom collate to handle None values
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=custom_collate)

    # Create model
    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32)

    # Create trainer with specific beta
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda",
        lr=3e-4,
        beta=beta_value,
        epochs=epochs,
        sharpness_weight=0.5,
    )

    # Train
    trainer.fit(epochs=epochs)

    # Save model
    output_path = Path(output_dir) / f"vae_beta_{beta_value:.2f}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "beta": beta_value,
            "epochs": epochs,
            "best_val": trainer.best_val,
        },
        output_path,
    )

    logger.info(f"Model saved to {output_path}")
    logger.info(f"Best validation loss: {trainer.best_val:.4f}")

    return model, trainer.best_val


def evaluate_reconstruction(model, val_loader, device="cuda"):
    """Evaluate reconstruction quality."""
    model.eval()
    total_bce = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            geom = batch["geometry"].to(device)

            out = model(geom)
            x_rec = out[0]

            # Binary cross entropy
            bce = nn.functional.binary_cross_entropy_with_logits(
                x_rec, geom, reduction="sum"
            )

            total_bce += bce.item()
            total_samples += geom.size(0)

    avg_bce = total_bce / total_samples
    return avg_bce


if __name__ == "__main__":
    dataset_path = "/home/genpipeline/fem/data/fem_dataset.pt"
    output_dir = "/home/genpipeline/checkpoints/ablation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Beta values to test
    beta_values = [1.0, 0.05]
    results = {}

    for beta in beta_values:
        model, best_val = train_with_beta(
            beta_value=beta,
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=50,  # Shorter run for ablation
        )

        results[f"beta_{beta:.2f}"] = {
            "best_val_loss": best_val,
            "model_path": str(Path(output_dir) / f"vae_beta_{beta:.2f}.pth"),
        }

    # Save results
    results_path = Path(output_dir) / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("Ablation Study Complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Results saved to {results_path}")
    logger.info(f"\nComparison:")
    for beta_name, data in results.items():
        logger.info(f"  {beta_name}: val_loss={data['best_val_loss']:.4f}")
