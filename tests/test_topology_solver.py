import numpy as np
from pathlib import Path
from topology_solver.solver import TopologySolver

def test_solver_runs_and_returns_stl(tmp_path):
    ts = TopologySolver(nx=16, ny=8, nz=8, n_iters=5)
    stl_path = ts.run({"force_n": 1000}, output_dir=str(tmp_path), volfrac=0.4)
    assert Path(stl_path).exists()
    assert Path(stl_path).stat().st_size > 100

def test_solver_stores_last_density(tmp_path):
    ts = TopologySolver(nx=16, ny=8, nz=8, n_iters=5)
    ts.run({"force_n": 500}, output_dir=str(tmp_path), volfrac=0.3)
    assert ts.last_density is not None
    assert ts.last_density.shape == (16, 8, 8)

def test_solver_backend_is_simp(tmp_path):
    ts = TopologySolver(nx=8, ny=4, nz=4, n_iters=3)
    assert ts.backend in ("simp", "openlsto")
