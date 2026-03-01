
import numpy as np
import pytest
import torch
from topology.simp_solver import SIMPSolver
from topology.simp_solver_gpu import SIMPSolverGPU

def test_simp_compliance_decreases():
    """Verify that compliance generally decreases during optimization."""
    nx, ny, nz = 20, 8, 8
    solver = SIMPSolver(nx=nx, ny=ny, nz=nz) # Just init
    
    # We'll manually run a few iterations and check compliance
    volfrac = 0.4
    force_mag = 1.0
    x = np.full(nx * ny * nz, volfrac)
    xPhys = x.copy()
    
    c0 = solver._calculate_compliance(xPhys, force_mag)
    
    # Run 5 iterations
    for _ in range(5):
        dc = solver._sensitivity(xPhys, force_mag)
        dc = solver._filter_dc(dc)
        x, xPhys = solver._oc_update(x, xPhys, dc, volfrac)
    
    c5 = solver._calculate_compliance(xPhys, force_mag)
    
    assert c5 < c0, f"Compliance should decrease: {c5:.4f} vs {c0:.4f}"

def test_simp_volume_convergence():
    """Verify volume fraction converges within 2% of target."""
    nx, ny, nz = 20, 8, 8
    volfrac_target = 0.3
    solver = SIMPSolver(nx=nx, ny=ny, nz=nz)
    density = solver.run(volfrac=volfrac_target, n_iters=20)
    
    actual_volfrac = density.mean()
    assert abs(actual_volfrac - volfrac_target) < 0.02, f"Target {volfrac_target}, got {actual_volfrac}"

def test_simp_output_shape():
    """Verify output shape matches grid dimensions."""
    nx, ny, nz = 15, 7, 9
    solver = SIMPSolver(nx=nx, ny=ny, nz=nz)
    density = solver.run(n_iters=1)
    assert density.shape == (nx, ny, nz)

def test_simp_sensitivity_variance():
    """Verify sensitivity is not uniform (detects fake heuristic regression)."""
    nx, ny, nz = 20, 8, 8
    solver = SIMPSolver(nx=nx, ny=ny, nz=nz)
    volfrac = 0.4
    xPhys = np.full(nx * ny * nz, volfrac)
    
    dc = solver._sensitivity(xPhys, force_mag=1.0)
    assert np.std(dc) > 1e-6, "Sensitivity should have variance; uniform dc suggests a bug."

def test_simp_bc_consistency():
    """Verify that different BCs produce different results."""
    nx, ny, nz = 20, 8, 8
    
    # Case 1: Fixed x_min
    solver1 = SIMPSolver(nx=nx, ny=ny, nz=nz, boundary_conditions={"fixed_face": "x_min"})
    res1 = solver1.run(n_iters=5)
    
    # Case 2: Fixed y_min
    solver2 = SIMPSolver(nx=nx, ny=ny, nz=nz, boundary_conditions={"fixed_face": "y_min"})
    res2 = solver2.run(n_iters=5)
    
    # Results should be physically different
    assert not np.allclose(res1, res2), "Different boundary conditions must produce different density fields."

def test_gpu_cpu_agreement():
    """Verify that GPU and CPU SIMP results match within 1%."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nx, ny, nz = 16, 4, 4
    volfrac = 0.4
    n_iters = 1
    force_mag = 1.0
    
    # CPU Solver
    cpu_solver = SIMPSolver(nx=nx, ny=ny, nz=nz)
    cpu_res = cpu_solver.run(volfrac=volfrac, n_iters=n_iters, force_mag=force_mag)
    
    # GPU Solver
    gpu_solver = SIMPSolverGPU(nx=nx, ny=ny, nz=nz)
    gpu_res = gpu_solver.run(volfrac=volfrac, n_iters=n_iters, force_mag=force_mag)
    
    # Check shape
    assert gpu_res.shape == cpu_res.shape
    
    # Check agreement
    mean_diff = np.abs(cpu_res - gpu_res).mean()
    assert mean_diff < 0.01, f"GPU vs CPU mean difference too high: {mean_diff:.6f}"

if __name__ == "__main__":
    # Manual run if called directly
    pytest.main([__file__])
