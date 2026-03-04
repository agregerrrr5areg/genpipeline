"""
Test GPU Conjugate Gradient FEM Solver

Unit test to verify the GPU-accelerated FEM solver works correctly with sample voxel grids.
"""

import numpy as np
import torch
from genpipeline.fem.gpu_fem_solver import GPUConjugateGradientFEM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpu_fem_solver():
    """Test the GPU conjugate gradient FEM solver."""
    logger.info("=== Testing GPU Conjugate Gradient FEM Solver ===")

    # Create a simple 10x10x10 solid cube
    voxels = np.ones((10, 10, 10), dtype=np.float32)

    # Create solver instance
    solver = GPUConjugateGradientFEM(voxel_size_mm=1.0)

    logger.info("Solving FEM problem...")
    results = solver.solve(
        voxels=voxels, fixed_face="x_min", load_face="x_max", force_n=1000.0
    )

    # Verify results
    logger.info(f"Results: {results}")
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "stress_max" in results, "Results should contain stress_max"
    assert "displacement_max" in results, "Results should contain displacement_max"
    assert "compliance" in results, "Results should contain compliance"
    assert "mass" in results, "Results should contain mass"

    # Check that values are reasonable
    assert results["stress_max"] > 0, "Stress should be positive"
    assert results["displacement_max"] > 0, "Displacement should be positive"
    assert results["compliance"] > 0, "Compliance should be positive"
    assert results["mass"] > 0, "Mass should be positive"

    logger.info("PASS: GPU FEM solver test completed successfully!")
    return results


if __name__ == "__main__":
    test_gpu_fem_solver()
