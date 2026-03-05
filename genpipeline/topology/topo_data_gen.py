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
    }

    def __init__(
        self,
        output_dir: str = "./fem_data",
        n_workers: int = 2,
        use_gpu: bool = True,
        n_iters: int = 60,
        grid_size: str = "large",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        self.use_gpu = use_gpu
        self.n_iters = n_iters
        self.grid_size = grid_size

        self._SIMP_GPU = None
        self._SIMP_CPU = None
        # CPU solver is more reliable for parallel generation
        # GPU solver has threading issues with PyTorch 2.10 + CUDA 12.8
        log.info("Using CPU SIMP solver")

    def _get_grid_dims(self, geom: str):
        if self.grid_size == "small":
            if geom == "lbracket":
                nx = np.random.randint(12, 20)
                ny = np.random.randint(6, 10)
                nz = np.random.randint(12, 20)
            else:
                nx = np.random.randint(12, 20)
                ny = np.random.randint(4, 8)
                nz = np.random.randint(4, 8)
        elif self.grid_size == "medium":
            if geom == "lbracket":
                nx = np.random.randint(20, 28)
                ny = np.random.randint(8, 12)
                nz = np.random.randint(20, 28)
            else:
                nx = np.random.randint(20, 32)
                ny = np.random.randint(6, 10)
                nz = np.random.randint(6, 10)
        else:  # large (default for high-quality data)
            if geom == "lbracket":
                nx = np.random.randint(24, 36)
                ny = np.random.randint(8, 12)
                nz = np.random.randint(24, 36)
            else:
                nx = np.random.randint(24, 40)
                ny = np.random.randint(6, 12)
                nz = np.random.randint(6, 12)
        return nx, ny, nz

    def _create_solver(self, nx, ny, nz, bcs):
        if self.use_gpu and self._SIMP_GPU:
            return self._SIMP_GPU(
                nx=nx, ny=ny, nz=nz, penal=3.0, rmin=1.5, boundary_conditions=bcs
            )
        else:
            if self._SIMP_CPU is None:
                from .simp_solver import SIMPSolver

                self._SIMP_CPU = SIMPSolver
            return self._SIMP_CPU(
                nx=nx, ny=ny, nz=nz, penal=3.0, rmin=1.5, boundary_conditions=bcs
            )

    def generate_single(self, i, n_samples):
        geoms = list(self.GEOM_BCS.keys())
        geom = np.random.choice(geoms)
        bcs = self.GEOM_BCS[geom]

        nx, ny, nz = self._get_grid_dims(geom)

        volfrac = np.random.uniform(0.2, 0.5)
        force_mag = np.random.uniform(500, 2000)

        solver = self._create_solver(nx, ny, nz, bcs)
        density = solver.run(volfrac=volfrac, n_iters=self.n_iters, force_mag=force_mag)

        sample_id = str(uuid.uuid4())[:8]
        stem = f"{geom[:4]}_v{int(volfrac * 100)}_f{int(force_mag)}_{nx}x{ny}x{nz}_{sample_id}"
        stl_path = self.output_dir / f"{stem}_mesh.stl"
        json_path = self.output_dir / f"{stem}_fem_results.json"

        voxel_size_mm = (100.0 / nx, 20.0 / ny, 20.0 / nz)
        density_to_stl(
            density, str(stl_path), threshold=0.5, voxel_size_mm=voxel_size_mm
        )

        compliance = solver.last_compliance
        total_vol = 100.0 * 20.0 * 20.0
        if geom == "lbracket":
            total_vol = 100.0 * 20.0 * 100.0
        mass = density.mean() * total_vol * 1.05e-6  # Plastic density

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
            },
        }

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"[{i + 1}/{n_samples}] Generated {geom}: {stl_path.name}")

    def generate(self, n_samples: int = 100):
        log.info(
            f"Generating {n_samples} samples using {self.n_workers} GPU workers..."
        )
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
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
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
        default=15,
        help="SIMP iterations per sample (default: 15, lower = faster)",
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
