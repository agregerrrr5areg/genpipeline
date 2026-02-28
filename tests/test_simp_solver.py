import numpy as np
from topology.simp_solver import SIMPSolver

def test_simp_output_shape():
    solver = SIMPSolver(nx=16, ny=8, nz=4)
    density = solver.run(volfrac=0.4, n_iters=5)
    assert density.shape == (16, 8, 4)

def test_simp_density_in_range():
    solver = SIMPSolver(nx=16, ny=8, nz=4)
    density = solver.run(volfrac=0.5, n_iters=5)
    assert density.min() >= 0.0
    assert density.max() <= 1.0

def test_simp_volume_fraction():
    solver = SIMPSolver(nx=16, ny=8, nz=4)
    density = solver.run(volfrac=0.4, n_iters=15)
    assert abs(density.mean() - 0.4) < 0.2
