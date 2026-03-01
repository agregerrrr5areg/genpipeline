#!/usr/bin/env python3
"""
rebuild_dataset.py
==================
Scan fem_data/ for *_mesh.stl + *_fem_results.json pairs,
re-voxelize at 64³ (CUDA kernel), and save a fresh fem_dataset.pt.

Usage:
    python rebuild_dataset.py
    python rebuild_dataset.py --resolution 64 --fem-dir ./fem_data
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from fem_data_pipeline import VoxelGrid, FEMDataset, DesignSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_pairs(fem_dir: Path, min_stress: float = 1.0) -> list:
    """Return list of (stl_path, metrics_dict, stem) for all valid paired files."""
    pairs = []
    by_geom: dict = {}
    skipped = 0

    for json_path in sorted(fem_dir.glob("*_fem_results.json")):
        stem = json_path.stem.replace("_fem_results", "")
        stl_path = fem_dir / f"{stem}_mesh.stl"
        if not stl_path.exists():
            skipped += 1
            continue
        with open(json_path) as f:
            d = json.load(f)
        if d.get("stress_max", 0) < min_stress and d.get("compliance", 0) == 0:
            skipped += 1
            continue
        pairs.append((stl_path, d, stem))
        geom = stem[:4]
        by_geom[geom] = by_geom.get(geom, 0) + 1

    log.info(f"Found {len(pairs)} valid pairs (skipped {skipped} empty/unmatched)")
    log.info("By geometry: " + "  ".join(f"{k}={v}" for k, v in sorted(by_geom.items())))
    return pairs


def build_samples(pairs: list, resolution: int) -> list:
    vox = VoxelGrid(resolution=resolution)
    samples = []
    n = len(pairs)

    for i, (stl_path, d, stem) in enumerate(pairs):
        if i % 50 == 0:
            log.info(f"  Voxelizing {i}/{n} ...")
        grid = vox.mesh_to_voxel(str(stl_path))
        if grid.max() == 0:
            log.warning(f"  Empty voxel: {stl_path.name}")
            continue
        samples.append(DesignSample(
            geometry_path=str(stl_path),
            stress_max=float(d.get("stress_max", 0)),
            stress_mean=float(d.get("stress_mean", 0)),
            compliance=float(d.get("compliance", 0)),
            mass=float(d.get("mass", 1.0)),
            parameters=d.get("parameters", {}),
            voxel_grid=(grid * 255).astype(np.uint8),
            bbox=d.get("bbox"),
        ))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fem-dir",    default="./fem_data")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    fem_dir = Path(args.fem_dir)
    log.info(f"Scanning {fem_dir} for mesh/results pairs ...")
    pairs = load_pairs(fem_dir)

    log.info(f"Voxelizing {len(pairs)} meshes at {args.resolution}³ ...")
    samples = build_samples(pairs, args.resolution)
    log.info(f"Valid samples: {len(samples)}")

    ds = FEMDataset(samples, voxel_resolution=args.resolution, use_sdf=False)
    n_val   = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Loaders stored in checkpoint for quickstart.py --step 3 compatibility
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

    out = fem_dir / "fem_dataset.pt"
    torch.save({
        "dataset":      ds,
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "samples":      samples,
    }, out)
    log.info(f"Saved {out}")
    log.info(f"Train: {n_train}  Val: {n_val}  Total: {len(ds)}")

    # Quick shape check
    b = next(iter(DataLoader(train_ds, batch_size=2)))
    log.info(f"Batch geometry: {b['geometry'].shape}  dtype: {b['geometry'].dtype}")


if __name__ == "__main__":
    main()
