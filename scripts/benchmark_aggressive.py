#!/usr/bin/env python3
"""
Benchmark script for aggressive SIMP sensitivity kernel.

Compares performance of:
- Standard CUDA kernel (simp_sensitivity)
- Aggressive optimized kernel (simp_sensitivity_aggressive)
- PyTorch fallback (if CUDA unavailable)

Usage:
    python scripts/benchmark_aggressive.py [--grid-size 32x8x8] [--iterations 100]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from contextlib import contextmanager


@contextmanager
def cuda_timer(name: str):
    """Context manager for CUDA timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms


def create_benchmark_data(nx: int, ny: int, nz: int, seed: int = 42) -> Tuple:
    """Create sample data for benchmarking."""
    n_elems = nx * ny * nz
    n_dof = n_elems * 24

    torch.manual_seed(seed)

    # Physical density field
    xPhys = torch.rand(n_elems, dtype=torch.float32, device="cuda") * 0.5 + 0.3

    # Displacement field
    u = torch.randn(n_dof, dtype=torch.float32, device="cuda") * 0.01

    # Element stiffness matrix (symmetric positive semi-definite)
    Ke = torch.randn(24, 24, dtype=torch.float32, device="cuda")
    Ke = Ke @ Ke.T

    # Element DOF mapping
    edof_mat = torch.randint(0, n_dof, (n_elems, 24), dtype=torch.int32, device="cuda")

    return xPhys, u, Ke, edof_mat


def benchmark_kernel(
    kernel_fn,
    xPhys,
    u,
    Ke,
    edof_mat,
    penal,
    nx,
    ny,
    nz,
    n_iterations: int,
    warmup: int = 10,
) -> List[float]:
    """Benchmark a kernel function."""
    times = []

    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)

    torch.cuda.synchronize()

    # Benchmark
    for _ in range(n_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = kernel_fn(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    return times


def run_benchmark(
    grid_sizes: List[Tuple[int, int, int]], n_iterations: int = 100, penal: float = 3.0
) -> dict:
    """Run comprehensive benchmark across multiple grid sizes."""

    from genpipeline.cuda_kernels import simp_sensitivity, simp_sensitivity_aggressive

    results = {"grid_sizes": [], "standard": {}, "aggressive": {}, "speedup": {}}

    print("\n" + "=" * 70)
    print("SIMP Sensitivity Kernel Benchmark")
    print("=" * 70)
    print(f"Iterations per test: {n_iterations}")
    print(f"Penalty factor: {penal}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70 + "\n")

    for nx, ny, nz in grid_sizes:
        n_elems = nx * ny * nz
        print(f"\nGrid: {nx}x{ny}x{z} ({n_elems:,} elements)")
        print("-" * 50)

        # Create data
        xPhys, u, Ke, edof_mat = create_benchmark_data(nx, ny, nz)

        # Benchmark standard kernel
        try:
            std_times = benchmark_kernel(
                simp_sensitivity,
                xPhys,
                u,
                Ke,
                edof_mat,
                penal,
                nx,
                ny,
                nz,
                n_iterations,
            )
            std_mean = np.mean(std_times)
            std_std = np.std(std_times)
            print(f"  Standard:  {std_mean:8.3f} ± {std_std:6.3f} ms")
        except Exception as e:
            print(f"  Standard:  FAILED - {e}")
            std_mean = float("inf")

        # Benchmark aggressive kernel
        try:
            agg_times = benchmark_kernel(
                simp_sensitivity_aggressive,
                xPhys,
                u,
                Ke,
                edof_mat,
                penal,
                nx,
                ny,
                nz,
                n_iterations,
            )
            agg_mean = np.mean(agg_times)
            agg_std = np.std(agg_times)
            print(f"  Aggressive: {agg_mean:8.3f} ± {agg_std:6.3f} ms")
        except Exception as e:
            print(f"  Aggressive: FAILED - {e}")
            agg_mean = float("inf")

        # Calculate speedup
        if std_mean > 0 and agg_mean > 0:
            speedup = std_mean / agg_mean
            print(f"  Speedup:   {speedup:8.2f}x")
        else:
            speedup = 0.0
            print(f"  Speedup:   N/A")

        # Store results
        results["grid_sizes"].append((nx, ny, nz))
        results["standard"][(nx, ny, nz)] = {
            "mean": std_mean,
            "std": std_std if "std_std" in dir() else 0,
            "times": std_times if "std_times" in dir() else [],
        }
        results["aggressive"][(nx, ny, nz)] = {
            "mean": agg_mean,
            "std": agg_std if "agg_std" in dir() else 0,
            "times": agg_times if "agg_times" in dir() else [],
        }
        results["speedup"][(nx, ny, nz)] = speedup

    return results


def print_summary(results: dict):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    speedups = [
        results["speedup"][gs]
        for gs in results["grid_sizes"]
        if results["speedup"][gs] > 0
    ]

    if speedups:
        print(f"Average speedup: {np.mean(speedups):.2f}x")
        print(f"Min speedup:     {np.min(speedups):.2f}x")
        print(f"Max speedup:     {np.max(speedups):.2f}x")
    else:
        print("No valid speedup measurements")

    print("\n" + "=" * 70)


def save_results(results: dict, output_file: str = "benchmark_results.json"):
    """Save benchmark results to JSON."""
    import json

    # Convert numpy types to native Python types
    serializable = {
        "grid_sizes": results["grid_sizes"],
        "standard": {
            f"{nx}x{ny}x{nz}": {
                "mean_ms": float(results["standard"][(nx, ny, nz)]["mean"]),
                "std_ms": float(results["standard"][(nx, ny, nz)]["std"]),
            }
            for nx, ny, nz in results["grid_sizes"]
        },
        "aggressive": {
            f"{nx}x{ny}x{nz}": {
                "mean_ms": float(results["aggressive"][(nx, ny, nz)]["mean"]),
                "std_ms": float(results["aggressive"][(nx, ny, nz)]["std"]),
            }
            for nx, ny, nz in results["grid_sizes"]
        },
        "speedup": {
            f"{nx}x{ny}x{nz}": float(results["speedup"][(nx, ny, nz)])
            for nx, ny, nz in results["grid_sizes"]
        },
    }

    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark aggressive SIMP sensitivity kernel"
    )
    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        default=["32x8x8", "64x16x16", "128x32x32"],
        help="Grid sizes to test (e.g., 32x8x8 64x16x16)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations per benchmark"
    )
    parser.add_argument("--penal", type=float, default=3.0, help="SIMP penalty factor")
    parser.add_argument(
        "--output",
        type=str,
        default="aggressive_benchmark_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Parse grid sizes
    grid_sizes = []
    for gs in args.grid_sizes:
        parts = gs.split("x")
        if len(parts) != 3:
            print(f"Invalid grid size: {gs}. Use format: NxNxN")
            continue
        grid_sizes.append(tuple(int(p) for p in parts))

    if not grid_sizes:
        print("No valid grid sizes specified")
        return 1

    if not torch.cuda.is_available():
        print("CUDA not available. Benchmark requires GPU.")
        return 1

    # Run benchmark
    results = run_benchmark(grid_sizes, args.iterations, args.penal)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
