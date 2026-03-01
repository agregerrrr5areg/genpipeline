import time
import numpy as np
import trimesh
import torch
from fem.data_pipeline import VoxelGrid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_voxelizer(stl_path, resolutions=[32, 64, 128, 256]):
    print(f"{'Resolution':<12} | {'Trimesh (ms)':<15} | {'CUDA (ms)':<12} | {'Speedup':<10}")
    print("-" * 55)
    
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump()[0]
        
    for res in resolutions:
        v = VoxelGrid(resolution=res)
        
        # Benchmark Trimesh (CPU)
        start_cpu = time.perf_counter()
        pitch = mesh.extents.max() / res
        voxel_cpu = mesh.voxelized(pitch=pitch).matrix
        cpu_time = (time.perf_counter() - start_cpu) * 1000
        
        # Benchmark CUDA
        cuda_time = 0
        try:
            from cuda_kernels import gpu_voxelize
            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            
            # Warmup
            _ = gpu_voxelize(verts, faces, res)
            torch.cuda.synchronize()
            
            start_cuda = time.perf_counter()
            for _ in range(10):
                _ = gpu_voxelize(verts, faces, res)
            torch.cuda.synchronize()
            cuda_time = ((time.perf_counter() - start_cuda) / 10) * 1000
        except Exception as e:
            logger.error(f"CUDA Voxelizer failed: {e}")
            cuda_time = float('inf')
            
        speedup = cpu_time / cuda_time if cuda_time > 0 else 0
        print(f"{res:<12} | {cpu_time:15.2f} | {cuda_time:12.2f} | {speedup:10.1f}x")

if __name__ == "__main__":
    stl_file = "fem/data/tape_v42_f1073_32x6x10_b2dae516_mesh.stl"
    benchmark_voxelizer(stl_file)
