import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import trimesh
import torch
from genpipeline.fem.voxel_fem import VoxelHexMesher, VoxelFEMEvaluator
from torch.utils.data import Dataset, DataLoader
from genpipeline.schema import (
    DesignParameters,
    FEMResult,
    DesignSample as PydanticDesignSample,
)
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DesignSample = PydanticDesignSample


class FEMDataset(Dataset):
    def __init__(self, samples, voxel_resolution=64, use_sdf=False):
        self.samples = samples
        self.voxel_resolution = voxel_resolution
        self.use_sdf = use_sdf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        voxel = (
            sample.voxel_grid
            if sample.voxel_grid is not None
            else np.zeros(
                (self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)
            )
        )
        return {
            "geometry": voxel,
            "metrics": {
                "stress_max": sample.metrics.stress_max,
                "stress_mean": sample.metrics.stress_mean,
                "compliance": sample.metrics.compliance,
                "mass": sample.metrics.mass,
            },
            "parameters": sample.parameters.dict(),
        }


class DataPipeline:
    def __init__(self, designs_dir: str, output_dir: str = "./fem_data"):
        self.designs_dir = Path(designs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = VoxelFEMEvaluator(str(self.output_dir))
        self.voxelizer = VoxelHexMesher()

    def process_all_designs(self, force_reprocess=False):
        logger.info("Starting FEM data pipeline...")
        parquet_path = self.output_dir / "results.parquet"
        samples = []

        if parquet_path.exists() and not force_reprocess:
            logger.info(f"Loading results from existing Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                params = {
                    k.replace("param_", ""): v
                    for k, v in row.items()
                    if k.startswith("param_")
                }
                mesh_path = row["geometry_path"]
                if not os.path.exists(mesh_path):
                    logger.warning(f"Mesh not found at {mesh_path}, skipping")
                    continue
                try:
                    mesh = trimesh.load(mesh_path)
                    voxel = mesh.voxelized(64).matrix
                except Exception as e:
                    logger.warning(f"Failed to load mesh {mesh_path}: {e}")
                    continue
                sample = DesignSample(
                    geometry_path=mesh_path,
                    metrics=FEMResult(
                        stress_max=row.get("stress_max", 0.0),
                        stress_mean=row.get("stress_mean", 0.0),
                        compliance=row.get("compliance", 0.0),
                        mass=row.get("mass", 0.0),
                        bbox=None,
                    ),
                    parameters=DesignParameters(
                        h_mm=params.get("h_mm", 10.0),
                        r_mm=params.get("r_mm", 3.0),
                        geometry_type="cantilever",
                    ),
                    voxel_grid=voxel,
                )
                samples.append(sample)
        else:
            logger.info(f"Discovering STEP/STL files in {self.designs_dir}")
            design_files = []
            for ext in [".step", ".stp", ".stl"]:
                design_files.extend(self.designs_dir.rglob(f"*{ext}"))
            design_files = sorted(design_files)

            if not design_files:
                logger.warning(f"No STEP/STL files found in {self.designs_dir}")
                return samples

            for mesh_path in design_files:
                logger.info(f"Processing {mesh_path}")
                try:
                    mesh = trimesh.load(str(mesh_path))
                    voxel = mesh.voxelized(64).matrix
                except Exception as e:
                    logger.warning(f"Failed to load {mesh_path}: {e}")
                    continue

                import tempfile, os
                with tempfile.TemporaryDirectory() as tmp:
                    inp_path = os.path.join(tmp, 'mesh.inp')
                    VoxelHexMesher.voxels_to_inp(voxel, output_path=inp_path)
                    fem_result = VoxelHexMesher.run_ccx(inp_path, ccx_cmd='/usr/bin/ccx')

                sample = DesignSample(
                    geometry_path=str(mesh_path),
                    metrics=FEMResult(
                        stress_max=fem_result.get("stress_max", -1.0),
                        stress_mean=fem_result.get("stress_mean", -1.0),
                        compliance=fem_result.get("compliance", -1.0),
                        mass=fem_result.get("mass", -1.0),
                        bbox=None,
                    ),
                    parameters=DesignParameters(
                        h_mm=10.0,
                        r_mm=3.0,
                        geometry_type=mesh_path.suffix.lstrip("."),
                    ),
                    voxel_grid=voxel,
                )
                samples.append(sample)

        logger.info(f"Processed {len(samples)} samples")
        return samples

    def build_dataset(self, force_reprocess=False):
        samples = self.process_all_designs(force_reprocess=force_reprocess)
        return FEMDataset(samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FEM data pipeline")
    parser.add_argument("--designs-dir", required=True, help="Directory with STEP/STL files")
    parser.add_argument("--output-dir", default="./fem_data", help="Output directory")
    parser.add_argument("--force-reprocess", action="store_true")
    args = parser.parse_args()

    pipeline = DataPipeline(designs_dir=args.designs_dir, output_dir=args.output_dir)
    samples = pipeline.process_all_designs(force_reprocess=args.force_reprocess)
    logger.info(f"Done. {len(samples)} samples ready.")
