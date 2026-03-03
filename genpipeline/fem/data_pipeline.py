import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from genpipeline.schema import (
    DesignParameters,
    FEMResult,
    DesignSample as PydanticDesignSample,
)
import logging
import pandas as pd

# Fix for Windows FreeCAD access
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DesignSample is now imported from schema.py
DesignSample = PydanticDesignSample


def run_freecad_command(command: str):
    """Run FreeCAD command through subprocess"""
    try:
        from genpipeline.config import load_config

        config = load_config()
        if config.freecad_path is None:
            logger.error("FreeCAD path not configured in pipeline_config.json")
            return None
        freecad_python = config.freecad_path.replace("freecad.exe", "python.exe")

        # Fix command syntax for multi-line commands
        if "\n" in command:
            command = command.replace("\n", " ")
            command = command.replace("                    ", " ")
            command = command.replace("                ", " ")
            command = command.replace("            ", " ")
            command = command.replace("        ", " ")
            command = command.replace("    ", " ")
            command = command.replace(";", "; ")
            command = command.replace("import ", "import ")
            command = command.replace("print(", "print(")

        result = subprocess.run(
            [
                freecad_python,
                "-c",
                command,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"FreeCAD command failed: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to run FreeCAD command: {e}")
        return None
        freecad_python = config.freecad_path.replace("freecad.exe", "python.exe")

        # Fix command syntax for multi-line commands
        if "\n" in command:
            command = command.replace("\n", " ")
            command = command.replace("                    ", " ")
            command = command.replace("                ", " ")
            command = command.replace("            ", " ")
            command = command.replace("        ", " ")
            command = command.replace("    ", " ")

        result = subprocess.run(
            [
                freecad_python,
                "-c",
                command,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"FreeCAD command failed: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to run FreeCAD command: {e}")
        return None
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to run FreeCAD command: {e}")
        return None


class VoxelGrid:
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.bounds = None

    def mesh_to_voxel(self, mesh_path: str, fill_interior=True) -> np.ndarray:
        try:
            mesh = trimesh.load(mesh_path)
        except Exception as e:
            logger.error(f"Failed to load mesh {mesh_path}: {e}")
            return np.zeros(
                (self.resolution, self.resolution, self.resolution), dtype=np.float32
            )

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump()[0]
        try:
            from ..cuda_kernels import gpu_voxelize

            verts, faces = (
                np.array(mesh.vertices, dtype=np.float32),
                np.array(mesh.faces, dtype=np.int32),
            )
            return gpu_voxelize(verts, faces, self.resolution)
        except Exception:
            pass
        voxel_grid = mesh.voxelized(pitch=mesh.extents.max() / self.resolution)
        occupancy = voxel_grid.matrix.astype(np.float32)
        if occupancy.shape != (self.resolution, self.resolution, self.resolution):
            occupancy = self._resize_voxel_grid(
                occupancy, (self.resolution, self.resolution, self.resolution)
            )
        return occupancy

    def _resize_voxel_grid(self, grid, target_shape):
        from scipy.ndimage import zoom

        zoom_factors = np.array(target_shape) / np.array(grid.shape)
        return zoom(grid, zoom_factors, order=0)

    def sdf_to_voxel(self, mesh_path: str) -> np.ndarray:
        try:
            mesh = trimesh.load(mesh_path)
        except Exception as e:
            logger.error(f"Failed to load mesh {mesh_path}: {e}")
            return np.zeros(
                (self.resolution, self.resolution, self.resolution), dtype=np.float32
            )

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump()[0]

        pitch = mesh.extents.max() / self.resolution
        voxel_grid = mesh.voxelized(pitch=pitch)

        voxel_coords = np.array(list(np.ndindex(voxel_grid.matrix.shape)))
        voxel_coords_scaled = voxel_coords * pitch + mesh.bounds[0]

        sdf = np.array(
            [
                mesh.nearest.signed_distance(np.array([p]))[0]
                for p in voxel_coords_scaled
            ]
        )
        sdf = np.clip(sdf, -pitch * 5, pitch * 5)
        sdf = (sdf + pitch * 5) / (pitch * 10)

        sdf = sdf.reshape(voxel_grid.matrix.shape)

        if sdf.shape != (self.resolution, self.resolution, self.resolution):
            sdf = self._resize_voxel_grid(
                sdf, (self.resolution, self.resolution, self.resolution)
            )

        return sdf.astype(np.float32)


class FEMResultParser:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

    def save_results(self, results: list, output_filename: str = "results.parquet"):
        """Save a list of results as Parquet for efficiency"""
        if not results:
            return

        df = pd.DataFrame(results)
        output_path = self.results_dir / output_filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return output_path

    def validate_data_provenance(self, mesh_path: str) -> bool:
        """
        Enforce 'No Synthetic Data' mandate.
        Verifies that a mesh file has a corresponding physical origin (e.g., a .step
        or FreeCAD document) and isn't just a disconnected synthetic file.
        """
        path = Path(mesh_path)
        # Check for matching .step or .FCStd file in the same or parent directory
        provenance_sources = [
            path.with_suffix(".step"),
            path.with_suffix(".FCStd"),
            path.parent / (path.stem + ".step"),
            path.parent.parent / "designs" / (path.stem + ".FCStd"),
        ]

        has_physical_root = any(p.exists() for p in provenance_sources)
        if not has_physical_root:
            logger.warning(
                f"PROVENANCE REJECTED: {mesh_path} has no verifiable physical source (.step or .FCStd)."
            )
            return False
        return True

    def extract_from_freecad(
        self, freecad_doc_path: str, output_json: str = "fem_results.json"
    ):
        # Use Windows FreeCAD through WSL with proper path handling
        escaped_path = freecad_doc_path.replace("\\", "\\\\")
        command = (
            f"import FreeCAD; import FreeCADGui; doc = FreeCAD.open('{escaped_path}'); results = {{}}; \
                   for obj in doc.Objects: \
                       if obj.TypeId == 'Fem::FemAnalysis': \
                           analysis_name = obj.Name; \
                           results[analysis_name] = {{'stress_max': 0.0, 'stress_mean': 0.0, 'compliance': 0.0}}; \
                   doc.close(); \
                   import json; \
                   with open('{output_json}', 'w') as f: \
                       json.dump(results, f, indent=2, default=str); \
                   print('Extraction complete')"
        )

        output = run_freecad_command(command)
        if output is None:
            logger.error("Failed to extract results from FreeCAD")
            return {}

        with open(str(self.results_dir / output_json), "r") as f:
            results = json.load(f)

        logger.info(f"Extracted results saved to {self.results_dir / output_json}")
        return results

    def _parse_analysis(self, analysis_obj) -> FEMResult:
        """Extract key metrics from FEM analysis"""
        stress_max = 0.0
        stress_mean = 0.0
        compliance = 0.0

        for result in analysis_obj.Object:
            if hasattr(result, "StressValues"):
                stress_vals = result.StressValues
                if stress_vals:
                    stress_max = float(max(stress_vals))
                    stress_mean = float(np.mean(stress_vals))

            if hasattr(result, "DisplacementLengths"):
                displacements = result.DisplacementLengths
                if displacements:
                    compliance = float(np.sum(displacements))

        return FEMResult(
            stress_max=stress_max,
            stress_mean=stress_mean,
            compliance=compliance,
            mass=0.0,  # Mass usually extracted separately or updated later
        )

    def extract_mesh_from_results(self, freecad_doc_path: str) -> dict:
        """Extract mesh paths from FEM analysis results"""
        try:
            import FreeCAD
            import FreeCADGui
        except ImportError:
            logger.error(
                "FreeCAD not available. Install FreeCAD or run within FreeCAD Python environment."
            )
            return {}

        doc = FreeCAD.open(str(freecad_doc_path))
        mesh_paths = {}

        for obj in doc.Objects:
            if obj.TypeId == "Fem::FemAnalysis":
                analysis_name = obj.Name
                # Find associated mesh object
                for child in obj.OutList:
                    if hasattr(child, "Shape"):
                        mesh_paths[analysis_name] = str(child.Shape.exportStl())
                        break

        doc.close()
        return mesh_paths


class FEMDataset(Dataset):
    def __init__(self, samples: list, voxel_resolution=32, use_sdf=False):
        self.samples = samples
        self.voxelizer = VoxelGrid(resolution=voxel_resolution)
        self.use_sdf = use_sdf

        # Precision Optimisation: Convert voxel grids to uint8 (FP8 storage simulation)
        for s in self.samples:
            if s.voxel_grid is not None and s.voxel_grid.dtype != np.uint8:
                s.voxel_grid = (s.voxel_grid * 255).astype(np.uint8)

        self.stress_max_vals = np.array([s.metrics.stress_max for s in samples])
        self.compliance_vals = np.array([s.metrics.compliance for s in samples])

        self.stress_max_mean = self.stress_max_vals.mean()
        self.stress_max_std = self.stress_max_vals.std() + 1e-6
        self.compliance_mean = self.compliance_vals.mean()
        self.compliance_std = self.compliance_vals.std() + 1e-6

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if sample.voxel_grid is None or sample.voxel_grid.size == 0:
            if self.use_sdf:
                voxel = self.voxelizer.sdf_to_voxel(sample.geometry_path)
            else:
                voxel = self.voxelizer.mesh_to_voxel(sample.geometry_path)
            # Store as uint8 for memory efficiency
            voxel_uint8 = (voxel * 255).astype(np.uint8)
        else:
            voxel_uint8 = sample.voxel_grid

        # Re-scale to FP32 for training
        voxel_tensor = torch.from_numpy(voxel_uint8).float() / 255.0

        stress_normalized = (
            sample.metrics.stress_max - self.stress_max_mean
        ) / self.stress_max_std
        compliance_normalized = (
            sample.metrics.compliance - self.compliance_mean
        ) / self.compliance_std

        performance = torch.tensor(
            [
                float(stress_normalized),
                float(compliance_normalized),
                float(sample.metrics.mass),
            ],
            dtype=torch.float32,
        )

        params = torch.tensor(
            [float(sample.parameters.h_mm), float(sample.parameters.r_mm)],
            dtype=torch.float32,
        )

        # Scale Preservation: Extract bounding box
        if sample.metrics.bbox:
            bbox_tensor = torch.tensor(
                [
                    sample.metrics.bbox["xmin"],
                    sample.metrics.bbox["ymin"],
                    sample.metrics.bbox["zmin"],
                    sample.metrics.bbox["xmax"],
                    sample.metrics.bbox["ymax"],
                    sample.metrics.bbox["zmax"],
                ],
                dtype=torch.float32,
            )
        else:
            bbox_tensor = torch.zeros(6, dtype=torch.float32)

        return {
            "geometry": voxel_tensor.unsqueeze(0),
            "stress_max": torch.tensor(sample.metrics.stress_max, dtype=torch.float32),
            "compliance": torch.tensor(sample.metrics.compliance, dtype=torch.float32),
            "mass": torch.tensor(sample.metrics.mass, dtype=torch.float32),
            "performance": performance,
            "parameters": params,
            "bbox": bbox_tensor,
        }


class DataPipeline:
    def __init__(self, freecad_project_dir: str, output_dir: str = "./fem_data"):
        self.project_dir = Path(freecad_project_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.parser = FEMResultParser(str(self.output_dir))
        self.voxelizer = VoxelGrid(resolution=64)

    def process_all_designs(self, force_reprocess=False):
        """Main pipeline: load results from Parquet OR extract from FreeCAD"""
        logger.info("Starting FEM data pipeline...")

        parquet_path = self.output_dir / "results.parquet"
        samples = []

        if parquet_path.exists() and not force_reprocess:
            logger.info(f"Loading results from existing Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                # Extract parameters from prefixed columns
                params = {
                    k.replace("param_", ""): v
                    for k, v in row.items()
                    if k.startswith("param_")
                }

                # Check if mesh path exists
                mesh_path = row["geometry_path"]
                if not os.path.exists(mesh_path):
                    logger.warning(f"Mesh not found at {mesh_path}, skipping sample")
                    continue

                # MANDATE CHECK: Reject synthetic data
                if not self.parser.validate_data_provenance(mesh_path):
                    continue

                voxel = self.voxelizer.mesh_to_voxel(mesh_path)
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
            fcstd_files = list(self.project_dir.glob("**/*.FCStd"))
            logger.info(f"Found {len(fcstd_files)} FreeCAD files")

            for i, fcstd_file in enumerate(fcstd_files):
                logger.info(
                    f"Processing [{i + 1}/{len(fcstd_files)}] {fcstd_file.name}"
                )

                fem_results = self.parser.extract_from_freecad(str(fcstd_file))
                mesh_paths = self.parser.extract_mesh_from_results(str(fcstd_file))

                for analysis_name, metrics in fem_results.items():
                    if analysis_name in mesh_paths:
                        mesh_path = mesh_paths[analysis_name]

                        # MANDATE CHECK: Reject synthetic data
                        if not self.parser.validate_data_provenance(mesh_path):
                            continue

                        voxel = self.voxelizer.mesh_to_voxel(mesh_path)

                        sample = DesignSample(
                            geometry_path=mesh_path,
                            metrics=FEMResult(
                                stress_max=metrics.get("stress_max", 0.0),
                                stress_mean=metrics.get("stress_mean", 0.0),
                                compliance=metrics.get("compliance", 0.0),
                                mass=metrics.get("mass", 0.0),
                                bbox=metrics.get("bbox"),
                            ),
                            parameters=DesignParameters(
                                h_mm=metrics.get("parameters", {}).get("h_mm", 10.0),
                                r_mm=metrics.get("parameters", {}).get("r_mm", 3.0),
                                geometry_type="cantilever",
                            ),
                            voxel_grid=voxel,
                        )
                        samples.append(sample)

        # Consolidate results to Parquet
        if samples:
            results_list = []
            for s in samples:
                res = {
                    "geometry_path": s.geometry_path,
                    "stress_max": s.stress_max,
                    "stress_mean": s.stress_mean,
                    "compliance": s.compliance,
                    "mass": s.mass,
                }
                # Add parameters with prefix
                for k, v in s.parameters.items():
                    res[f"param_{k}"] = v
                results_list.append(res)

            self.parser.save_results(results_list)

        logger.info(f"Collected {len(samples)} samples")

        dataset = FEMDataset(samples, voxel_resolution=64, use_sdf=False)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        torch.save(
            {
                "dataset": dataset,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "samples": samples,
            },
            self.output_dir / "fem_dataset.pt",
        )

        logger.info(f"Dataset saved to {self.output_dir / 'fem_dataset.pt'}")
        logger.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
        )

        return train_loader, val_loader, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FEM data extraction pipeline")
    parser.add_argument(
        "--freecad-project",
        type=str,
        required=True,
        help="Path to FreeCAD project directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fem_data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--voxel-resolution", type=int, default=32, help="Voxel grid resolution"
    )

    args = parser.parse_args()

    pipeline = DataPipeline(args.freecad_project, args.output_dir)
    train_loader, val_loader, dataset = pipeline.process_all_designs()

    logger.info("Pipeline complete!")
    print(f"First batch shape: {next(iter(train_loader))['geometry'].shape}")
