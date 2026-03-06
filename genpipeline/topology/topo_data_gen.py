"""
topo_data_gen.py — Topology optimisation as a dataset generator.
Runs SIMP on various configurations to generate physics-based training data.

Mandate: No non-physical synthetic data.
---------------------------------------
This generator uses SIMP (Solid Isotropic Material with Penalization), which
is a physics-based method for compliance minimization. It is NOT random noise.
Model training must exclusively use physics-grounded data (FreeCAD/CalculiX or SIMP).

Pros:
- Extremely fast compared to parametric FreeCAD/CalculiX variants.
- No Windows/FreeCAD dependency — runs natively in Linux/WSL.
- Scales easily to thousands of training samples.

Cons:
- Compliance is used as a proxy for efficiency; real stress values are approximated.
- Structural distribution may differ from parametric variants (distribution shift).
"""

import os
import json
import uuid
import argparse
import logging
import numpy as np
import torch
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
from .mesh_export import density_to_stl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class TopoDataGenerator:
    """Generates topology optimisation samples using SIMP (GPU by default, CPU with --cpu flag)."""

    GEOM_BCS = {
        "cantilever": {"fixed_face": "x_min", "load_face": "x_max", "load_dof": 2},
        "tapered": {"fixed_face": "x_min", "load_face": "x_max", "load_dof": 2},
        "ribbed": {"fixed_face": "x_min", "load_face": "x_max", "load_dof": 2},
        "lbracket": {"fixed_face": "z_min", "load_face": "x_max", "load_dof": 2},
        "bridge": {
            "fixed_face": ["x_min", "x_max"],
            "load_face": "center",
            "load_dof": 2,
        },
        "arch": {"fixed_face": ["x_min", "x_max"], "load_face": "top", "load_dof": 2},
        "3dload": {
            "fixed_face": "x_min",
            "load_face": "x_max",
            "load_dof": [1, 2],
            "multi_load": True,
        },
        "offset": {"fixed_face": "x_min", "load_face": "x_max_offset", "load_dof": 2},
        "center": {"fixed_face": "x_min", "load_face": "center_z", "load_dof": 2},
        "portal": {
            "fixed_face": ["y_min", "y_max"],
            "load_face": "top_center",
            "load_dof": 2,
        },
        "spoke": {"fixed_face": "center", "load_face": "outer_ring", "load_dof": 2},
        "tower": {"fixed_face": "z_min", "load_face": "z_max", "load_dof": [1, 2]},
        "sandwich": {
            "fixed_face": ["z_min", "z_max"],
            "load_face": "center",
            "load_dof": 2,
        },
        "simply": {"fixed_face": ["x_min", "x_max"], "load_face": "top", "load_dof": 2},
        # Organic multi-load shapes
        "branch": {
            "fixed_face": "x_min",
            "load_face": "x_max",
            "load_dof": 2,
            "multi_load": True,
            "load_positions": ["top", "mid", "bottom"],
        },
        "y_tree": {
            "fixed_face": "z_min",
            "load_face": "x_max",
            "load_dof": 2,
            "multi_load": True,
            "load_positions": ["top", "mid_y", "mid_z"],
        },
        "network": {
            "fixed_face": ["x_min", "y_min"],
            "load_face": "x_max",
            "load_dof": 2,
            "multi_load": True,
            "load_pattern": "grid",
        },
    }

    GEOM_WEIGHTS = {
        "bridge": 2,
        "arch": 2,
        "3dload": 2,
        "portal": 2,
        "spoke": 2,
        "tower": 2,
        "sandwich": 2,
        "simply": 1,
        "offset": 1,
        "center": 1,
        "cantilever": 1,
        "tapered": 1,
        "ribbed": 1,
        "lbracket": 1,
    }

    GEOM_WEIGHTS = {
        "bridge": 2,
        "arch": 2,
        "3dload": 2,
        "offset": 1,
        "center": 1,
        "cantilever": 1,
        "tapered": 1,
        "ribbed": 1,
        "lbracket": 1,
        "branch": 1,
        "y_tree": 1,
        "network": 1,
    }

    def __init__(
        self,
        output_dir: str = "./fem_data",
        n_workers: int = 2,
        use_gpu: bool = True,
        n_iters: int = 30,  # Reduced from 60, early stopping will provide additional speedup
        grid_size: str = "medium",
        penal: float = 3.5,
        organic_mode: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        self.use_gpu = use_gpu
        self.n_iters = n_iters
        self.grid_size = grid_size
        self.penal = penal
        self.organic_mode = organic_mode

        grid_memory = {
            "small": "200MB",
            "medium": "800MB",
            "large": "2GB",
        }
        log.info(
            f"Grid size: {self.grid_size} (~{grid_memory.get(self.grid_size, 'unknown')} per worker)"
        )

        self._SIMP_GPU = None  # Will be set on first use
        self._SIMP_CPU = None
        if use_gpu:
            log.info(
                f"Using GPU SIMP solver (penal={penal}, organic_mode={organic_mode})"
            )
        else:
            log.info(
                f"Using CPU SIMP solver (penal={penal}, organic_mode={organic_mode})"
            )

    def _get_grid_dims(self, geom: str):
        if self.grid_size == "small":
            nx, ny, nz = 16, 8, 8
            return nx, ny, nz
        elif self.grid_size == "medium":
            if geom in ("lbracket", "bridge", "arch", "center"):
                nx = np.random.randint(20, 28)
                ny = np.random.randint(8, 12)
                nz = np.random.randint(20, 28)
            elif geom in ("3dload", "offset"):
                nx = np.random.randint(22, 32)
                ny = np.random.randint(8, 12)
                nz = np.random.randint(8, 12)
            elif geom in ("branch", "y_tree", "network"):
                nx = np.random.randint(22, 32)
                ny = np.random.randint(10, 16)
                nz = np.random.randint(10, 16)
            else:
                nx = np.random.randint(20, 32)
                ny = np.random.randint(6, 10)
                nz = np.random.randint(6, 10)
        else:
            if geom in ("lbracket", "bridge", "arch", "center"):
                nx = np.random.randint(24, 36)
                ny = np.random.randint(8, 12)
                nz = np.random.randint(24, 36)
            elif geom in ("3dload", "offset"):
                nx = np.random.randint(28, 40)
                ny = np.random.randint(8, 14)
                nz = np.random.randint(8, 14)
            elif geom in ("branch", "y_tree", "network"):
                nx = np.random.randint(24, 36)
                ny = np.random.randint(12, 18)
                nz = np.random.randint(12, 18)
            else:
                nx = np.random.randint(24, 40)
                ny = np.random.randint(6, 12)
                nz = np.random.randint(6, 12)
        return nx, ny, nz

    def _create_solver(self, nx, ny, nz, bcs):
        if self.use_gpu:
            if self._SIMP_GPU is None:
                from .simp_solver_dense_gpu import DenseGPUSolver

                self._SIMP_GPU = DenseGPUSolver
            return self._SIMP_GPU(
                nx=nx, ny=ny, nz=nz, penal=self.penal, rmin=1.5, device="cuda"
            )
        else:
            if self._SIMP_CPU is None:
                from .simp_solver import SIMPSolver

                self._SIMP_CPU = SIMPSolver
            return self._SIMP_CPU(
                nx=nx, ny=ny, nz=nz, penal=self.penal, rmin=1.5, boundary_conditions=bcs
            )

    def _select_geometry(self):
        if self.organic_mode:
            geoms = list(self.GEOM_BCS.keys())
            weights = [self.GEOM_WEIGHTS.get(g, 1) for g in geoms]
            geom = np.random.choice(geoms, p=np.array(weights) / sum(weights))
        else:
            geom = np.random.choice(list(self.GEOM_BCS.keys()))
        return geom

    def generate_single(self, i, n_samples):
        geom = self._select_geometry()
        bcs = self.GEOM_BCS[geom]

        nx, ny, nz = self._get_grid_dims(geom)

        if self.organic_mode:
            volfrac = np.random.uniform(0.12, 0.35)
            force_mag = np.random.uniform(200, 5000)
        else:
            volfrac = np.random.uniform(0.2, 0.5)
            force_mag = np.random.uniform(500, 2000)

        solver = self._create_solver(nx, ny, nz, bcs)
        density = solver.run(volfrac=volfrac, n_iters=self.n_iters, force_mag=force_mag)

        sample_id = str(uuid.uuid4())[:8]
        stem = f"{geom[:4]}_v{int(volfrac * 100)}_f{int(force_mag)}_{nx}x{ny}x{nz}_{sample_id}"
        stl_path = self.output_dir / f"{stem}_mesh.stl"
        json_path = self.output_dir / f"{stem}_fem_results.json"

        voxel_size_mm = (100.0 / nx, 20.0 / ny, 20.0 / nz)
        stl_result = density_to_stl(
            density, str(stl_path), threshold=0.5, voxel_size_mm=voxel_size_mm
        )

        if stl_result is None:
            # Empty design, raise an exception to signal failure
            raise ValueError(f"Empty design for {geom}")

        compliance = solver.last_compliance
        total_vol = 100.0 * 20.0 * 20.0
        if geom in ("lbracket", "bridge", "arch"):
            total_vol = 100.0 * 20.0 * 100.0
        mass = density.mean() * total_vol * 1.05e-6

        results = {
            "stress_max": float(compliance * 0.1),
            "compliance": float(compliance),
            "mass": float(mass),
            "parameters": {
                "geometry": geom,
                "volfrac": float(volfrac),
                "force_n": float(force_mag),
                "nx": int(nx),
                "ny": int(ny),
                "nz": int(nz),
                "penal": self.penal,
            },
        }

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"[{i + 1}/{n_samples}] Generated {geom}: {stl_path.name}")

    def generate(self, n_samples: int = 100):
        log.info(f"Generating {n_samples} samples using {self.n_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.n_workers
        ) as executor:
            futures = [
                executor.submit(self.generate_single, i, n_samples)
                for i in range(n_samples)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Sample generation failed: {e}")
                    import traceback

                    traceback.print_exc()

    def generate_scaled(self, n_samples: int = 1000, batch_size: int = 100):
        """Generate large-scale dataset with quality control and progress tracking."""
        log.info(f"Generating {n_samples} samples in batches of {batch_size}...")

        total_batches = (n_samples + batch_size - 1) // batch_size
        generated = 0

        for batch_idx in range(total_batches):
            batch_count = min(batch_size, n_samples - generated)
            log.info(
                f"Batch {batch_idx + 1}/{total_batches}: Generating {batch_count} samples..."
            )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_workers
            ) as executor:
                futures = [
                    executor.submit(self.generate_single, i, n_samples)
                    for i in range(generated, generated + batch_count)
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"Sample generation failed: {e}")
                        import traceback

                        traceback.print_exc()

            generated += batch_count
            log.info(f"Progress: {generated}/{n_samples} samples generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate topology optimization data (GPU by default)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./fem_data", help="Output directory"
    )
    # Auto-detect CPU cores (leave 2 for system)
    cpu_count = max(1, multiprocessing.cpu_count() - 2)
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count,
        help=f"Parallel workers (default: auto = CPU cores - 2)",
    )
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="Enable scaled generation with quality control",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU solver instead of GPU (slower)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=30,
        help="SIMP iterations per sample (default: 30, lower = faster)",
    )
    args = parser.parse_args()

    generator = TopoDataGenerator(
        output_dir=args.output_dir,
        n_workers=args.workers,
        use_gpu=not args.cpu,
        n_iters=args.iterations,
        grid_size=args.grid_size,
    )
    if args.scaled:
        generator.generate_scaled(n_samples=args.n_samples)
    else:
        generator.generate(n_samples=args.n_samples)
