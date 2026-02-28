import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DesignSample:
    geometry_path: str
    stress_max: float
    stress_mean: float
    compliance: float
    mass: float
    parameters: dict
    voxel_grid: np.ndarray


class VoxelGrid:
    def __init__(self, resolution=32):
        self.resolution = resolution
        self.bounds = None

    def mesh_to_voxel(self, mesh_path: str, fill_interior=True) -> np.ndarray:
        try:
            mesh = trimesh.load(mesh_path)
        except Exception as e:
            logger.error(f"Failed to load mesh {mesh_path}: {e}")
            return np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump()[0]

        # Try GPU voxelisation first (27–2890× faster than CPU trimesh)
        try:
            from cuda_kernels import gpu_voxelize
            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces,    dtype=np.int32)
            return gpu_voxelize(verts, faces, self.resolution)
        except Exception as gpu_err:
            logger.debug(f"GPU voxelise failed ({gpu_err}), falling back to CPU trimesh")

        # CPU fallback
        voxel_grid = mesh.voxelized(pitch=mesh.extents.max() / self.resolution)
        occupancy = voxel_grid.matrix.astype(np.float32)

        if occupancy.shape != (self.resolution, self.resolution, self.resolution):
            occupancy = self._resize_voxel_grid(occupancy, (self.resolution, self.resolution, self.resolution))

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
            return np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump()[0]

        pitch = mesh.extents.max() / self.resolution
        voxel_grid = mesh.voxelized(pitch=pitch)

        voxel_coords = np.array(list(np.ndindex(voxel_grid.matrix.shape)))
        voxel_coords_scaled = voxel_coords * pitch + mesh.bounds[0]

        sdf = np.array([mesh.nearest.signed_distance(np.array([p]))[0] for p in voxel_coords_scaled])
        sdf = np.clip(sdf, -pitch * 5, pitch * 5)
        sdf = (sdf + pitch * 5) / (pitch * 10)

        sdf = sdf.reshape(voxel_grid.matrix.shape)

        if sdf.shape != (self.resolution, self.resolution, self.resolution):
            sdf = self._resize_voxel_grid(sdf, (self.resolution, self.resolution, self.resolution))

        return sdf.astype(np.float32)


class FEMResultParser:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_freecad(self, freecad_doc_path: str, output_json: str = "fem_results.json"):
        try:
            import FreeCAD
            import FreeCADGui
        except ImportError:
            logger.error("FreeCAD not available. Install FreeCAD or run within FreeCAD Python environment.")
            return {}

        doc = FreeCAD.open(str(freecad_doc_path))
        results = {}

        for obj in doc.Objects:
            if obj.TypeId == "Fem::FemAnalysis":
                analysis_name = obj.Name
                results[analysis_name] = self._parse_analysis(obj)

        doc.close()

        output_path = self.results_dir / output_json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Extracted results saved to {output_path}")
        return results

    def _parse_analysis(self, analysis_obj):
        """Extract key metrics from FEM analysis"""
        metrics = {
            "stress_max": 0.0,
            "stress_mean": 0.0,
            "compliance": 0.0,
            "mass": 0.0,
            "parameters": {}
        }

        for result in analysis_obj.Object:
            if hasattr(result, 'StressValues'):
                stress_vals = result.StressValues
                if stress_vals:
                    metrics["stress_max"] = float(max(stress_vals))
                    metrics["stress_mean"] = float(np.mean(stress_vals))

            if hasattr(result, 'DisplacementLengths'):
                displacements = result.DisplacementLengths
                if displacements:
                    metrics["compliance"] = float(np.sum(displacements))

        return metrics

    def extract_mesh_from_results(self, freecad_doc_path: str, output_dir: str = None):
        """Export FEM result meshes to STL"""
        try:
            import FreeCAD
        except ImportError:
            logger.error("FreeCAD not available")
            return {}

        if output_dir is None:
            output_dir = self.results_dir / "meshes"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        doc = FreeCAD.open(str(freecad_doc_path))
        mesh_paths = {}

        for obj in doc.Objects:
            if obj.TypeId == "Fem::FemMesh":
                mesh_path = Path(output_dir) / f"{obj.Name}.stl"
                try:
                    mesh_obj = obj.FemMesh
                    mesh_obj.write(str(mesh_path))
                    mesh_paths[obj.Name] = str(mesh_path)
                    logger.info(f"Exported mesh: {mesh_path}")
                except Exception as e:
                    logger.error(f"Failed to export mesh {obj.Name}: {e}")

        doc.close()
        return mesh_paths


class FEMDataset(Dataset):
    def __init__(self, samples: list, voxel_resolution=32, use_sdf=False):
        self.samples = samples
        self.voxelizer = VoxelGrid(resolution=voxel_resolution)
        self.use_sdf = use_sdf

        self.stress_max_vals = np.array([s.stress_max for s in samples])
        self.compliance_vals = np.array([s.compliance for s in samples])

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
        else:
            voxel = sample.voxel_grid

        voxel_tensor = torch.from_numpy(voxel).float()

        stress_normalized = (sample.stress_max - self.stress_max_mean) / self.stress_max_std
        compliance_normalized = (sample.compliance - self.compliance_mean) / self.compliance_std

        performance = torch.tensor([
            float(stress_normalized),
            float(compliance_normalized),
            float(sample.mass)
        ], dtype=torch.float32)

        params = torch.tensor([
            float(sample.parameters.get("h_mm", 10.0)),
            float(sample.parameters.get("r_mm", 3.0))
        ], dtype=torch.float32)

        return {
            'geometry': voxel_tensor.unsqueeze(0),
            'stress_max': torch.tensor(sample.stress_max, dtype=torch.float32),
            'compliance': torch.tensor(sample.compliance, dtype=torch.float32),
            'mass': torch.tensor(sample.mass, dtype=torch.float32),
            'performance': performance,
            'parameters': params
        }


class DataPipeline:
    def __init__(self, freecad_project_dir: str, output_dir: str = "./fem_data"):
        self.project_dir = Path(freecad_project_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.parser = FEMResultParser(str(self.output_dir))
        self.voxelizer = VoxelGrid(resolution=32)

    def process_all_designs(self):
        """Main pipeline: extract results → voxelize → create dataset"""
        logger.info("Starting FEM data pipeline...")

        fcstd_files = list(self.project_dir.glob("**/*.FCStd"))
        logger.info(f"Found {len(fcstd_files)} FreeCAD files")

        samples = []

        for i, fcstd_file in enumerate(fcstd_files):
            logger.info(f"Processing [{i+1}/{len(fcstd_files)}] {fcstd_file.name}")

            fem_results = self.parser.extract_from_freecad(str(fcstd_file))
            mesh_paths = self.parser.extract_mesh_from_results(str(fcstd_file))

            for analysis_name, metrics in fem_results.items():
                if analysis_name in mesh_paths:
                    mesh_path = mesh_paths[analysis_name]
                    voxel = self.voxelizer.mesh_to_voxel(mesh_path)

                    sample = DesignSample(
                        geometry_path=mesh_path,
                        stress_max=metrics.get("stress_max", 0.0),
                        stress_mean=metrics.get("stress_mean", 0.0),
                        compliance=metrics.get("compliance", 0.0),
                        mass=metrics.get("mass", 0.0),
                        parameters=metrics.get("parameters", {}),
                        voxel_grid=voxel
                    )
                    samples.append(sample)

        logger.info(f"Collected {len(samples)} samples")

        dataset = FEMDataset(samples, voxel_resolution=32, use_sdf=False)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        torch.save({
            'dataset': dataset,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'samples': samples
        }, self.output_dir / "fem_dataset.pt")

        logger.info(f"Dataset saved to {self.output_dir / 'fem_dataset.pt'}")
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        return train_loader, val_loader, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FEM data extraction pipeline")
    parser.add_argument("--freecad-project", type=str, required=True, help="Path to FreeCAD project directory")
    parser.add_argument("--output-dir", type=str, default="./fem_data", help="Output directory for processed data")
    parser.add_argument("--voxel-resolution", type=int, default=32, help="Voxel grid resolution")

    args = parser.parse_args()

    pipeline = DataPipeline(args.freecad_project, args.output_dir)
    train_loader, val_loader, dataset = pipeline.process_all_designs()

    logger.info("Pipeline complete!")
    print(f"First batch shape: {next(iter(train_loader))['geometry'].shape}")
