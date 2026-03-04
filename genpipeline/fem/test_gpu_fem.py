"""
Test suite for GPU FEM Solver

Tests for GPUConjugateGradientFEM class with comprehensive validation of:
- Material properties and selection
- Mesh generation and element connectivity
- Stiffness matrix assembly
- Boundary condition application
- Conjugate gradient solver
- Result calculation
- GPU vs CPU performance comparisons
"""

import pytest
import numpy as np
import torch
from genpipeline.fem.gpu_fem_solver import GPUConjugateGradientFEM
from typing import Dict


@pytest.fixture
def fem_solver():
    """Create FEM solver instance with PLA material."""
    return GPUConjugateGradientFEM(material="pla")


@pytest.fixture
def small_voxel_grid():
    """Create small 3x3x3 voxel grid for testing."""
    return np.ones((3, 3, 3), dtype=np.uint8)


@pytest.fixture
def medium_voxel_grid():
    """Create medium 10x10x10 voxel grid for performance testing."""
    return np.ones((10, 10, 10), dtype=np.uint8)


@pytest.fixture
def large_voxel_grid():
    """Create large 32x32x32 voxel grid for benchmarking."""
    return np.ones((32, 32, 32), dtype=np.uint8)


@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


class TestMaterialProperties:
    """Test material property selection and validation."""

    def test_default_material(self, fem_solver):
        """Test that PLA is the default material."""
        assert fem_solver.material == "pla"
        assert fem_solver.E == 3.5e3  # 3.5 GPa
        assert fem_solver.nu == 0.36

    def test_steel_material(self):
        """Test steel material properties."""
        solver = GPUConjugateGradientFEM(material="steel")
        assert solver.material == "steel"
        assert solver.E == 210e3  # 210 GPa
        assert solver.nu == 0.30

    def test_aluminum_material(self):
        """Test aluminum material properties."""
        solver = GPUConjugateGradientFEM(material="aluminum_6061")
        assert solver.material == "aluminum_6061"
        assert solver.E == 68.9e3  # 68.9 GPa
        assert solver.nu == 0.33

    def test_invalid_material(self):
        """Test that invalid material raises ValueError."""
        with pytest.raises(ValueError):
            GPUConjugateGradientFEM(material="invalid")


class TestMeshGeneration:
    """Test mesh generation and element connectivity."""

    def test_mesh_generation(self, fem_solver, small_voxel_grid):
        """Test that mesh generation produces correct node and element counts."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        # For 3x3x3 grid: 4x4x4 = 64 nodes, 27 elements
        assert node_coords.shape[0] == 64
        assert elements.shape[0] == 27
        assert elements.shape[1] == 8  # 8 nodes per hexahedral element

    def test_element_connectivity(self, fem_solver, small_voxel_grid):
        """Test that element connectivity is correct."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        # Check first element connectivity
        first_element = elements[0]
        assert len(first_element) == 8

        # Check that node IDs are unique within element
        assert len(set(first_element.tolist())) == 8


class TestBoundaryConditions:
    """Test boundary condition identification."""

    def test_fixed_boundary_conditions(self, fem_solver, small_voxel_grid):
        """Test that fixed boundary conditions are correctly identified."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        fixed_nodes, load_nodes = fem_solver._get_boundary_conditions(
            node_coords, 3, 3, 3, "x_min", "x_max"
        )

        # For x_min face: nodes with x=0
        # Should be 4x4 = 16 nodes
        assert len(fixed_nodes) == 16

    def test_load_boundary_conditions(self, fem_solver, small_voxel_grid):
        """Test that load boundary conditions are correctly identified."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        fixed_nodes, load_nodes = fem_solver._get_boundary_conditions(
            node_coords, 3, 3, 3, "x_min", "x_max"
        )

        # For x_max face: nodes with x=3 (since voxel_size=1.0)
        # Should be 4x4 = 16 nodes
        assert len(load_nodes) == 16


class TestStiffnessMatrix:
    """Test stiffness matrix assembly and properties."""

    def test_stiffness_matrix_size(self, fem_solver, small_voxel_grid):
        """Test that stiffness matrix has correct size."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        K = fem_solver._assemble_stiffness_matrix(node_coords, elements)

        # For 64 nodes: 64 * 3 = 192 DOFs
        assert K.shape == (192, 192)

    def test_stiffness_matrix_sparsity(self, fem_solver, small_voxel_grid):
        """Test that stiffness matrix is sparse."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        K = fem_solver._assemble_stiffness_matrix(node_coords, elements)

        # Check that matrix is sparse
        assert K.is_sparse

    def test_element_stiffness_computation(self, fem_solver):
        """Test element stiffness matrix computation."""
        # Create simple element with known geometry
        node_positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Node 1
                [1.0, 0.0, 0.0],  # Node 2
                [1.0, 1.0, 0.0],  # Node 3
                [0.0, 1.0, 0.0],  # Node 4
                [0.0, 0.0, 1.0],  # Node 5
                [1.0, 0.0, 1.0],  # Node 6
                [1.0, 1.0, 1.0],  # Node 7
                [0.0, 1.0, 1.0],  # Node 8
            ],
            device=fem_solver.device,
            dtype=torch.float32,
        )

        K_el = fem_solver._compute_element_stiffness(node_positions)

        # Check matrix dimensions
        assert K_el.shape == (24, 24)

        # Check symmetry
        assert torch.allclose(K_el, K_el.t())


class TestSolver:
    """Test conjugate gradient solver functionality."""

    def test_solver_with_small_problem(self, fem_solver, small_voxel_grid):
        """Test solver with small voxel grid."""
        results = fem_solver.solve(small_voxel_grid)

        # Check that results contain expected keys
        expected_keys = ["stress_max", "displacement_max", "compliance", "mass"]
        assert all(key in results for key in expected_keys)

        # Check that values are reasonable
        assert results["stress_max"] > 0
        assert results["displacement_max"] >= 0
        assert results["compliance"] > 0
        assert results["mass"] > 0

    def test_solver_with_medium_problem(self, fem_solver, medium_voxel_grid):
        """Test solver with medium voxel grid."""
        results = fem_solver.solve(medium_voxel_grid)

        # Check that results contain expected keys
        expected_keys = ["stress_max", "displacement_max", "compliance", "mass"]
        assert all(key in results for key in expected_keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUSpecific:
    """Test GPU-specific functionality."""

    def test_gpu_device_selection(self):
        """Test that GPU is selected when available."""
        solver = GPUConjugateGradientFEM()
        assert solver.device.type == "cuda"

    def test_bf16_support(self):
        """Test that BF16 is used on GPU."""
        solver = GPUConjugateGradientFEM()
        assert solver.dtype == torch.bfloat16

    def test_gpu_tensor_creation(self, fem_solver, small_voxel_grid):
        """Test that tensors are created on GPU."""
        node_coords, elements = fem_solver._build_mesh(
            torch.from_numpy(small_voxel_grid), None
        )

        assert node_coords.device.type == "cuda"
        assert elements.device.type == "cuda"


class TestPerformance:
    """Test performance characteristics."""

    def test_solver_performance_scaling(self, fem_solver, medium_voxel_grid):
        """Test that solver scales reasonably with problem size."""
        # This is a basic performance test - actual benchmarking is in separate file
        results = fem_solver.solve(medium_voxel_grid)

        # Check that solver completes in reasonable time
        # (This is more of a sanity check than a strict performance test)
        assert results["displacement_max"] > 0

    def test_material_performance_difference(self):
        """Test that different materials give different results."""
        solver_steel = GPUConjugateGradientFEM(material="steel")
        solver_pla = GPUConjugateGradientFEM(material="pla")

        small_voxel_grid = np.ones((3, 3, 3), dtype=np.uint8)

        results_steel = solver_steel.solve(small_voxel_grid)
        results_pla = solver_pla.solve(small_voxel_grid)

        # Steel should be stiffer (less displacement) than PLA
        assert results_steel["displacement_max"] < results_pla["displacement_max"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
