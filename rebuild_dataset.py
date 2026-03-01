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
from fem.data_pipeline import VoxelGrid, FEMDataset
from genpipeline.schema import DesignSample, FEMResult, DesignParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_pairs(fem_dir: Path, min_stress: float = 1.0) -> list:
    """Return list of (stl_path, metrics_dict, stem) for all valid paired files."""
    import pandas as pd
    pairs = []
    by_geom: dict = {}
    skipped = 0

    parquet_path = fem_dir / "results.parquet"
    if parquet_path.exists():
        log.info(f"Loading results from Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            stem = row['source_file'].replace("_fem_results.json", "") if 'source_file' in row else Path(row['geometry_path']).stem.replace("_mesh", "")
            stl_path = Path(row['geometry_path'])
            
            if not stl_path.exists():
                skipped += 1
                continue
                
            if row.get("stress_max", 0) < min_stress and row.get("compliance", 0) == 0:
                skipped += 1
                continue
                
            metrics = {
                "stress_max": row.get("stress_max", 0),
                "stress_mean": row.get("stress_mean", 0),
                "compliance": row.get("compliance", 0),
                "mass": row.get("mass", 0),
                "parameters": {k.replace("param_", ""): v for k, v in row.items() if k.startswith("param_")},
                "bbox": None # or load if available
            }
            pairs.append((stl_path, metrics, stem))
            geom = stem[:4]
            by_geom[geom] = by_geom.get(geom, 0) + 1
    else:
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
        # Parse parameters
        params_dict = d.get("parameters", {})
        # Map common names if needed or use defaults
        p = DesignParameters(
            h_mm=params_dict.get("h_mm", 10.0),
            r_mm=params_dict.get("r_mm", 2.0),
            geometry_type=params_dict.get("geometry_type", "cantilever"),
            material_name=params_dict.get("material_name", "Plastic_ABS"),
            material_cfg=params_dict.get("material_cfg")
        )

        # Parse metrics
        m = FEMResult(
            stress_max=float(d.get("stress_max", 0)),
            stress_mean=float(d.get("stress_mean", 0)),
            compliance=float(d.get("compliance", 0)),
            mass=float(d.get("mass", 1.0)),
            bbox=d.get("bbox"),
            success=True
        )

        samples.append(DesignSample(
            geometry_path=str(stl_path),
            metrics=m,
            parameters=p,
            voxel_grid=(grid * 255).astype(np.uint8)
        ))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fem-dir", type=str, default="./fem/data")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    from fem.data_pipeline import DataPipeline
    pipeline = DataPipeline(args.fem_dir, args.fem_dir)

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
