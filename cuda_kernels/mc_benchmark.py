"""
Benchmark: GPU Marching Cubes vs skimage CPU Marching Cubes.

Run:
    source venv/bin/activate
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \\
        python cuda_kernels/mc_benchmark.py
"""

import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gpu_marching_cubes import gpu_marching_cubes


def make_test_volume(shape="sphere", resolution=32):
    """Generate a synthetic voxel occupancy volume."""
    R = resolution
    grid = np.zeros((R, R, R), dtype=np.float32)
    cx = cy = cz = R / 2.0

    if shape == "sphere":
        r = R * 0.4
        for x in range(R):
            for y in range(R):
                for z in range(R):
                    if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < r**2:
                        grid[x, y, z] = 1.0
    elif shape == "torus":
        R_major, r_minor = R * 0.32, R * 0.14
        for x in range(R):
            for y in range(R):
                for z in range(R):
                    dx, dy, dz = x - cx, y - cy, z - cz
                    q = np.sqrt(dx**2 + dy**2) - R_major
                    if q**2 + dz**2 < r_minor**2:
                        grid[x, y, z] = 1.0
    else:  # box
        lo, hi = int(R*0.2), int(R*0.8)
        grid[lo:hi, lo:hi, lo:hi] = 1.0

    return grid


def cpu_marching_cubes(volume, isovalue=0.5):
    """skimage CPU marching cubes."""
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(volume, level=isovalue)
    return verts, faces


def benchmark(shape="sphere", resolution=32, n_runs=10):
    volume = make_test_volume(shape, resolution)
    verts_gpu, faces_gpu = gpu_marching_cubes(volume, isovalue=0.5)  # warmup+compile

    print(f"\n{'='*55}")
    print(f"  Shape: {shape}  |  Resolution: {resolution}³")
    print(f"{'='*55}")

    # CPU timing
    cpu_marching_cubes(volume)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        verts_cpu, faces_cpu = cpu_marching_cubes(volume)
    cpu_ms = (time.perf_counter() - t0) / n_runs * 1000

    # GPU timing
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        verts_gpu, faces_gpu = gpu_marching_cubes(volume, isovalue=0.5)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / n_runs * 1000

    speedup = cpu_ms / gpu_ms

    print(f"  CPU (skimage):  {cpu_ms:8.2f} ms   {len(faces_cpu):,} triangles")
    print(f"  GPU (CUDA):     {gpu_ms:8.2f} ms   {len(faces_gpu):,} triangles")
    print(f"  Speedup:        {speedup:8.1f}×")

    return speedup


if __name__ == "__main__":
    print("\nGPU Marching Cubes Benchmark")
    print("Compiling CUDA kernel on first run...")

    speedups = []
    for shape in ["sphere", "torus", "box"]:
        for res in [32, 64]:
            s = benchmark(shape, resolution=res, n_runs=10)
            speedups.append(s)

    print(f"\n{'='*55}")
    print(f"  Average speedup: {np.mean(speedups):.1f}×")
    print(f"  Peak speedup:    {np.max(speedups):.1f}×")
    print(f"{'='*55}\n")
