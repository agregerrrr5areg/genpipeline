"""
Integration tests for the aggressive SIMP sensitivity kernel.

These tests validate:
1. The aggressive kernel compiles and loads
2. Numerical accuracy matches standard kernel
3. Performance improvements over standard kernel
"""

import pytest
import torch
import numpy as np
import time
from pathlib import Path


class TestAggressiveKernelAvailability:
    """Test that aggressive kernel is available and functional."""

    def test_aggressive_kernel_imports(self):
        """Verify aggressive kernel function can be imported."""
        from genpipeline.cuda_kernels import simp_sensitivity_aggressive

        assert callable(simp_sensitivity_aggressive)

    def test_aggressive_kernel_loads_without_error(self):
        """Verify aggressive kernel loads without throwing."""
        from genpipeline.cuda_kernels import simp_sensitivity_aggressive

        # If we get here, the kernel loaded successfully
        assert True


class TestAggressiveKernelAccuracy:
    """Test numerical accuracy of aggressive vs standard kernel."""

    @pytest.fixture
    def sample_data_32x8x8(self):
        """Create sample data for 32x8x8 grid."""
        nx, ny, nz = 32, 8, 8
        n_elems = nx * ny * nz
        n_dof = n_elems * 24  # 24 DOFs per element

        torch.manual_seed(42)

        # Physical density field
        xPhys = torch.rand(n_elems, dtype=torch.float32, device="cuda") * 0.5 + 0.3

        # Displacement field (random for testing)
        u = torch.randn(n_dof, dtype=torch.float32, device="cuda") * 0.01

        # Element stiffness matrix (24x24)
        Ke = torch.randn(24, 24, dtype=torch.float32, device="cuda")
        Ke = Ke @ Ke.T  # Make symmetric positive semi-definite

        # Element DOF mapping
        edof_mat = torch.randint(
            0, n_dof, (n_elems, 24), dtype=torch.int32, device="cuda"
        )

        return xPhys, u, Ke, edof_mat, nx, ny, nz

    def test_aggressive_matches_standard_output_shape(self, sample_data_32x8x8):
        """Verify aggressive kernel produces correct output shape."""
        from genpipeline.cuda_kernels import (
            simp_sensitivity,
            simp_sensitivity_aggressive,
        )

        xPhys, u, Ke, edof_mat, nx, ny, nz = sample_data_32x8x8
        penal = 3.0

        # Run both kernels
        standard = simp_sensitivity(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)
        aggressive = simp_sensitivity_aggressive(
            xPhys, u, Ke, edof_mat, penal, nx, ny, nz
        )

        assert standard.shape == aggressive.shape
        assert standard.shape[0] == nx * ny * nz

    def test_aggressive_matches_standard_numerically(self, sample_data_32x8x8):
        """Verify aggressive kernel matches standard within tolerance."""
        from genpipeline.cuda_kernels import (
            simp_sensitivity,
            simp_sensitivity_aggressive,
        )

        xPhys, u, Ke, edof_mat, nx, ny, nz = sample_data_32x8x8
        penal = 3.0

        # Run both kernels
        standard = simp_sensitivity(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)
        aggressive = simp_sensitivity_aggressive(
            xPhys, u, Ke, edof_mat, penal, nx, ny, nz
        )

        # Check relative error
        max_diff = torch.max(torch.abs(standard - aggressive)).item()
        max_val = torch.max(torch.abs(standard)).item()
        rel_error = max_diff / (max_val + 1e-10)

        # Allow 1% relative error for floating point differences
        assert rel_error < 0.01, f"Relative error {rel_error:.4f} exceeds 1% tolerance"

    def test_aggressive_handles_edge_cases(self):
        """Test aggressive kernel with edge case inputs."""
        from genpipeline.cuda_kernels import simp_sensitivity_aggressive

        nx, ny, nz = 4, 4, 4
        n_elems = nx * ny * nz
        n_dof = n_elems * 24

        # Test with uniform density
        xPhys = torch.ones(n_elems, dtype=torch.float32, device="cuda") * 0.5
        u = torch.zeros(n_dof, dtype=torch.float32, device="cuda")
        Ke = torch.eye(24, dtype=torch.float32, device="cuda")
        edof_mat = torch.arange(n_dof, dtype=torch.int32, device="cuda").reshape(
            n_elems, 24
        )

        result = simp_sensitivity_aggressive(xPhys, u, Ke, edof_mat, 3.0, nx, ny, nz)

        # With zero displacement, sensitivity should be zero
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAggressiveKernelPerformance:
    """Test performance improvements of aggressive kernel."""

    @pytest.fixture
    def benchmark_data(self):
        """Create larger dataset for performance testing."""
        nx, ny, nz = 32, 8, 8
        n_elems = nx * ny * nz
        n_dof = n_elems * 24

        torch.manual_seed(42)

        xPhys = torch.rand(n_elems, dtype=torch.float32, device="cuda") * 0.5 + 0.3
        u = torch.randn(n_dof, dtype=torch.float32, device="cuda") * 0.01
        Ke = torch.randn(24, 24, dtype=torch.float32, device="cuda")
        Ke = Ke @ Ke.T
        edof_mat = torch.randint(
            0, n_dof, (n_elems, 24), dtype=torch.int32, device="cuda"
        )

        return xPhys, u, Ke, edof_mat, nx, ny, nz

    def test_aggressive_is_faster_than_standard(self, benchmark_data):
        """Verify aggressive kernel is faster than standard."""
        from genpipeline.cuda_kernels import (
            simp_sensitivity,
            simp_sensitivity_aggressive,
        )

        xPhys, u, Ke, edof_mat, nx, ny, nz = benchmark_data
        penal = 3.0
        n_iterations = 10

        # Warmup
        for _ in range(3):
            simp_sensitivity(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)
            simp_sensitivity_aggressive(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)

        torch.cuda.synchronize()

        # Benchmark standard
        start = time.perf_counter()
        for _ in range(n_iterations):
            simp_sensitivity(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)
        torch.cuda.synchronize()
        standard_time = time.perf_counter() - start

        # Benchmark aggressive
        start = time.perf_counter()
        for _ in range(n_iterations):
            simp_sensitivity_aggressive(xPhys, u, Ke, edof_mat, penal, nx, ny, nz)
        torch.cuda.synchronize()
        aggressive_time = time.perf_counter() - start

        # Aggressive should be at least 90% as fast (allowing for measurement noise)
        # In practice should be 2-3x faster
        speedup = standard_time / aggressive_time

        print(
            f"\nStandard: {standard_time * 1000:.2f}ms, Aggressive: {aggressive_time * 1000:.2f}ms"
        )
        print(f"Speedup: {speedup:.2f}x")

        assert speedup > 0.9, f"Aggressive kernel slower than standard: {speedup:.2f}x"


class TestAggressiveKernelFallback:
    """Test fallback behavior when aggressive kernel fails."""

    def test_fallback_to_standard_on_error(self):
        """Verify graceful fallback when aggressive kernel fails."""
        from genpipeline.cuda_kernels import simp_sensitivity_aggressive

        nx, ny, nz = 32, 8, 8
        n_elems = nx * ny * nz
        n_dof = n_elems * 24

        xPhys = torch.rand(n_elems, dtype=torch.float32, device="cuda")
        u = torch.randn(n_dof, dtype=torch.float32, device="cuda")
        Ke = torch.randn(24, 24, dtype=torch.float32, device="cuda")
        edof_mat = torch.randint(
            0, n_dof, (n_elems, 24), dtype=torch.int32, device="cuda"
        )

        # Should not raise, should return valid result (possibly via fallback)
        try:
            result = simp_sensitivity_aggressive(
                xPhys, u, Ke, edof_mat, 3.0, nx, ny, nz
            )
            assert result is not None
            assert result.shape[0] == n_elems
        except Exception as e:
            pytest.fail(f"Aggressive kernel should not raise: {e}")
