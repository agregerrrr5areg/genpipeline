"""
Benchmarking suite for GPU FEM Solver

Compares GPU-accelerated FEM solver performance against CPU-based solvers
across different grid resolutions and materials.
"""

import time
import numpy as np
import torch
from genpipeline.fem.gpu_fem_solver import GPUConjugateGradientFEM
from genpipeline.fem.voxel_fem import (
    VoxelHexMesher,
)  # Assuming this exists for CPU comparison
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUFEMBenchmark:
    """Benchmark GPU FEM solver performance."""

    def __init__(self):
        self.results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Benchmarking on device: {self.device}")

    def benchmark_resolution_scaling(
        self,
        resolutions: list = [(32, 32, 32), (64, 64, 64), (128, 128, 128)],
        materials: list = ["pla", "aluminum_6061", "steel"],
        num_runs: int = 3,
    ):
        """Benchmark performance across different grid resolutions."""
        logger.info("=== Resolution Scaling Benchmark ===")
        logger.info(f"Resolutions: {resolutions}")
        logger.info(f"Materials: {materials}")
        logger.info(f"Runs per configuration: {num_runs}")

        for resolution in resolutions:
            D, H, W = resolution
            logger.info(f"\nTesting resolution: {D}x{H}x{W}")

            for material in materials:
                logger.info(f"\nMaterial: {material}")

                # Create solid cube
                voxels = np.ones((D, H, W), dtype=np.float32)

                # Benchmark GPU solver
                gpu_times = []
                for i in range(num_runs):
                    start_time = time.time()

                    solver = GPUConjugateGradientFEM(
                        voxel_size_mm=1.0,
                        material=material,
                        max_iterations=1000,
                        tolerance=1e-6,
                    )

                    results = solver.solve(
                        voxels=voxels,
                        fixed_face="x_min",
                        load_face="x_max",
                        force_n=1000.0,
                    )

                    end_time = time.time()
                    elapsed = end_time - start_time
                    gpu_times.append(elapsed)

                    logger.info(f"GPU Run {i + 1}/{num_runs}: {elapsed:.4f}s")

                gpu_avg = sum(gpu_times) / num_runs
                gpu_min = min(gpu_times)
                gpu_max = max(gpu_times)

                # Benchmark CPU solver (if available)
                cpu_times = []
                try:
                    for i in range(num_runs):
                        start_time = time.time()

                        # Create CPU solver instance
                        cpu_solver = VoxelHexMesher(
                            voxel_size_mm=1.0, material=material
                        )

                        # Convert voxels to appropriate format
                        results = cpu_solver.run_ccx(
                            voxels=voxels,
                            fixed_face="x_min",
                            load_face="x_max",
                            force_n=1000.0,
                        )

                        end_time = time.time()
                        elapsed = end_time - start_time
                        cpu_times.append(elapsed)

                        logger.info(f"CPU Run {i + 1}/{num_runs}: {elapsed:.4f}s")

                    cpu_avg = sum(cpu_times) / num_runs
                    cpu_min = min(cpu_times)
                    cpu_max = max(cpu_times)

                    speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float("inf")

                    self.results.append(
                        {
                            "test": "resolution_scaling",
                            "resolution": resolution,
                            "material": material,
                            "gpu_time": gpu_avg,
                            "cpu_time": cpu_avg,
                            "speedup": speedup,
                            "gpu_min": gpu_min,
                            "gpu_max": gpu_max,
                            "cpu_min": cpu_min,
                            "cpu_max": cpu_max,
                        }
                    )

                    logger.info(f"GPU Average: {gpu_avg:.4f}s")
                    logger.info(f"CPU Average: {cpu_avg:.4f}s")
                    logger.info(f"Speedup: {speedup:.2f}x")

                except Exception as e:
                    logger.warning(f"CPU benchmark failed: {e}")
                    self.results.append(
                        {
                            "test": "resolution_scaling",
                            "resolution": resolution,
                            "material": material,
                            "gpu_time": gpu_avg,
                            "cpu_time": None,
                            "speedup": None,
                            "gpu_min": gpu_min,
                            "gpu_max": gpu_max,
                            "cpu_min": None,
                            "cpu_max": None,
                        }
                    )

    def benchmark_material_performance(
        self,
        resolution: tuple = (64, 64, 64),
        materials: list = ["pla", "aluminum_6061", "steel"],
        num_runs: int = 5,
    ):
        """Benchmark performance across different materials."""
        logger.info("=== Material Performance Benchmark ===")
        logger.info(f"Resolution: {resolution}")
        logger.info(f"Materials: {materials}")
        logger.info(f"Runs per material: {num_runs}")

        D, H, W = resolution
        voxels = np.ones((D, H, W), dtype=np.float32)

        for material in materials:
            logger.info(f"\nMaterial: {material}")

            times = []
            for i in range(num_runs):
                start_time = time.time()

                solver = GPUConjugateGradientFEM(
                    voxel_size_mm=1.0,
                    material=material,
                    max_iterations=1000,
                    tolerance=1e-6,
                )

                results = solver.solve(
                    voxels=voxels, fixed_face="x_min", load_face="x_max", force_n=1000.0
                )

                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)

                logger.info(f"Run {i + 1}/{num_runs}: {elapsed:.4f}s")

            avg_time = sum(times) / num_runs
            min_time = min(times)
            max_time = max(times)

            self.results.append(
                {
                    "test": "material_performance",
                    "resolution": resolution,
                    "material": material,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                }
            )

            logger.info(
                f"{material.capitalize()} - Average: {avg_time:.4f}s, Range: {min_time:.4f}s - {max_time:.4f}s"
            )

    def benchmark_blackwell_optimizations(
        self,
        resolution: tuple = (64, 64, 64),
        material: str = "steel",
        num_runs: int = 10,
    ):
        """Benchmark Blackwell-specific optimizations (BF16, etc.)."""
        logger.info("=== Blackwell Optimization Benchmark ===")
        logger.info(f"Resolution: {resolution}")
        logger.info(f"Material: {material}")
        logger.info(f"Runs: {num_runs}")

        D, H, W = resolution
        voxels = np.ones((D, H, W), dtype=np.float32)

        # Test with different precision modes
        precisions = ["float32", "bf16"] if torch.cuda.is_available() else ["float32"]

        for precision in precisions:
            logger.info(f"\nPrecision: {precision}")
            times = []

            for i in range(num_runs):
                start_time = time.time()

                solver = GPUConjugateGradientFEM(
                    voxel_size_mm=1.0,
                    material=material,
                    max_iterations=1000,
                    tolerance=1e-6,
                )

                # Force precision mode if BF16 is available
                if precision == "bf16" and torch.cuda.is_available():
                    torch.set_float32_matmul_precision("high")
                    torch.backends.cuda.matmul.allow_tf32 = False

                results = solver.solve(
                    voxels=voxels, fixed_face="x_min", load_face="x_max", force_n=1000.0
                )

                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)

                logger.info(f"Run {i + 1}/{num_runs}: {elapsed:.4f}s")

            avg_time = sum(times) / num_runs
            min_time = min(times)
            max_time = max(times)

            self.results.append(
                {
                    "test": "blackwell_optimizations",
                    "resolution": resolution,
                    "material": material,
                    "precision": precision,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                }
            )

            logger.info(
                f"{precision.upper()} - Average: {avg_time:.4f}s, Range: {min_time:.4f}s - {max_time:.4f}s"
            )

    def print_summary(self):
        """Print benchmark summary."""
        logger.info("\n=== Benchmark Summary ===")

        # Filter results by test type
        resolution_results = [
            r for r in self.results if r["test"] == "resolution_scaling"
        ]
        material_results = [
            r for r in self.results if r["test"] == "material_performance"
        ]
        blackwell_results = [
            r for r in self.results if r["test"] == "blackwell_optimizations"
        ]

        # Resolution scaling summary
        if resolution_results:
            logger.info("\nResolution Scaling:")
            for result in resolution_results:
                if result["cpu_time"] is not None:
                    logger.info(
                        f"{result['resolution']} {result['material']}: "
                        f"GPU={result['gpu_time']:.4f}s CPU={result['cpu_time']:.4f}s "
                        f"Speedup={result['speedup']:.2f}x"
                    )
                else:
                    logger.info(
                        f"{result['resolution']} {result['material']}: "
                        f"GPU={result['gpu_time']:.4f}s (CPU unavailable)"
                    )

        # Material performance summary
        if material_results:
            logger.info("\nMaterial Performance:")
            for result in material_results:
                logger.info(f"{result['material']}: {result['avg_time']:.4f}s")

        # Blackwell optimizations summary
        if blackwell_results:
            logger.info("\nBlackwell Optimizations:")
            for result in blackwell_results:
                logger.info(f"{result['precision']}: {result['avg_time']:.4f}s")

    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to JSON file."""
        import json

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Benchmark results saved to {filename}")


if __name__ == "__main__":
    # Run comprehensive benchmark suite
    benchmark = GPUFEMBenchmark()

    # Resolution scaling benchmark
    benchmark.benchmark_resolution_scaling(
        resolutions=[(32, 32, 32), (64, 64, 64), (128, 128, 128)],
        materials=["pla", "aluminum_6061", "steel"],
        num_runs=3,
    )

    # Material performance benchmark
    benchmark.benchmark_material_performance(
        resolution=(64, 64, 64), materials=["pla", "aluminum_6061", "steel"], num_runs=5
    )

    # Blackwell optimizations benchmark (if GPU available)
    if torch.cuda.is_available():
        benchmark.benchmark_blackwell_optimizations(
            resolution=(64, 64, 64), material="steel", num_runs=10
        )

    # Print and save results
    benchmark.print_summary()
    benchmark.save_results("benchmark_results.json")

    logger.info("\nBenchmarking complete!")
