#!/usr/bin/env python3
"""
eval_vae.py
===========
Post-training evaluation of the VAE:
  - Reconstruction quality on validation set (IoU, BCE)
  - Latent space coverage (PCA / t-SNE plot)
  - Sample 8 random designs, decode to STL for visual inspection

Usage:
    python eval_vae.py
    python eval_vae.py --checkpoint checkpoints/vae_best.pth --output-dir eval_results
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_model_and_data(ckpt_path: str, fem_data: str, batch_size: int, device: str):
    from vae_design_model import DesignVAE
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent_dim   = ckpt.get("latent_dim",   32)
    input_shape  = ckpt.get("input_shape",  (64, 64, 64))
    model = DesignVAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded checkpoint: latent_dim={latent_dim}  input={input_shape}")

    ds_ckpt = torch.load(Path(fem_data) / "fem_dataset.pt", weights_only=False)
    ds = ds_ckpt["dataset"]
    n_val = max(1, int(len(ds) * 0.2))
    _, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return model, val_loader, latent_dim


def evaluate_reconstruction(model, val_loader, device):
    """Compute mean IoU and BCE on the validation set."""
    total_iou = 0.0
    total_bce = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            geom = batch["geometry"].to(device)
            with autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                x_rec, mu, logvar, _, _ = model(geom)
            pred = torch.sigmoid(x_rec).float()
            target = geom.float()
            bce = F.binary_cross_entropy(pred, target).item()
            inter = (pred * target).sum(dim=(1, 2, 3, 4))
            union = (pred + target - pred * target).sum(dim=(1, 2, 3, 4)) + 1e-6
            iou = (inter / union).mean().item()
            total_iou += iou * geom.size(0)
            total_bce += bce * geom.size(0)
            n += geom.size(0)
    return {"iou": total_iou / n, "bce": total_bce / n}


def collect_latent_vectors(model, val_loader, device):
    zs, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            geom = batch["geometry"].to(device)
            with autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                mu, _ = model.encode(geom)
            zs.append(mu.float().cpu())
    return torch.cat(zs, dim=0).numpy()


def export_samples(model, latent_dim, output_dir: Path, device, n_samples=8):
    """Decode n random latent vectors to STL."""
    try:
        import trimesh
        from utils import VoxelConverter
    except ImportError:
        log.warning("trimesh or utils not available — skipping STL export")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    z = torch.randn(n_samples, latent_dim, device=device)
    with torch.no_grad():
        voxels = model.decode(z).squeeze(1).cpu().numpy()  # (N, D, H, W)
    for i, v in enumerate(voxels):
        if v.max() < 0.3:
            log.info(f"  Sample {i}: empty voxel, skipping")
            continue
        mesh_data = VoxelConverter.voxel_to_mesh(v, voxel_size=1.0 / v.shape[0])
        if mesh_data is None:
            continue
        m = trimesh.Trimesh(vertices=mesh_data["vertices"], faces=mesh_data["faces"])
        path = output_dir / f"sample_{i:02d}.stl"
        m.export(str(path))
        log.info(f"  Exported {path.name}  ({len(m.faces)} faces)")


def plot_latent_pca(zs: np.ndarray, output_dir: Path):
    try:
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("sklearn/matplotlib not available — skipping PCA plot")
        return
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(zs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(z2[:, 0], z2[:, 1], alpha=0.6, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Latent Space PCA (validation set)")
    fig.tight_layout()
    out = output_dir / "latent_pca.png"
    fig.savefig(str(out), dpi=120)
    log.info(f"PCA plot saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/vae_best.pth")
    parser.add_argument("--fem-data",    default="./fem_data")
    parser.add_argument("--output-dir",  default="./eval_results")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--n-samples",   type=int, default=8)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, val_loader, latent_dim = load_model_and_data(
        args.checkpoint, args.fem_data, args.batch_size, args.device
    )

    log.info("Evaluating reconstruction quality ...")
    metrics = evaluate_reconstruction(model, val_loader, args.device)
    log.info(f"  Val IoU:  {metrics['iou']:.4f}")
    log.info(f"  Val BCE:  {metrics['bce']:.4f}")

    log.info("Collecting latent vectors ...")
    zs = collect_latent_vectors(model, val_loader, args.device)
    log.info(f"  Latent shape: {zs.shape}  mean_norm: {np.linalg.norm(zs, axis=1).mean():.2f}")

    plot_latent_pca(zs, out_dir)

    log.info(f"Exporting {args.n_samples} random sample STLs ...")
    export_samples(model, latent_dim, out_dir / "samples", args.device, args.n_samples)

    report = {
        "checkpoint": args.checkpoint,
        "val_iou": metrics["iou"],
        "val_bce": metrics["bce"],
        "latent_mean_norm": float(np.linalg.norm(zs, axis=1).mean()),
        "latent_std": float(zs.std()),
        "n_val_samples": len(zs),
    }
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {out_dir / 'eval_report.json'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
