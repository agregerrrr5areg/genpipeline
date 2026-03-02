"""solver.py — TopologySolver: tries OpenLSTO, falls back to SIMP."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np


def _try_openlsto():
    try:
        import openlsto
        return openlsto
    except ImportError:
        return None


class TopologySolver:
    """
    Level-set / SIMP topology optimisation wrapper.

    Uses OpenLSTO if built; otherwise SIMP (GPU preferred).

    Parameters
    ----------
    nx, ny, nz : voxel grid resolution
    n_iters    : solver iterations
    """

    def __init__(self, nx: int = 32, ny: int = 8, nz: int = 8, n_iters: int = 80):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_iters = n_iters
        self.last_density: np.ndarray | None = None
        self.last_compliance: float = 0.0
        self._openlsto = _try_openlsto()
        
        import torch
        if self._openlsto:
            self.backend = "openlsto"
        elif torch.cuda.is_available():
            self.backend = "simp_gpu"
        else:
            self.backend = "simp"

    def run(self, sim_cfg: dict, output_dir: str, volfrac: float = 0.4, export_stl: bool = True, x_init: np.ndarray | None = None) -> str | None:
        """
        Run topology optimisation, return path to output STL (if exported).

        Parameters
        ----------
        sim_cfg    : dict with at least {"force_n": float}
        output_dir : directory to write STL
        volfrac    : target volume fraction (0–1)
        export_stl : if False, skips mesh export and returns None
        x_init     : optional initial density (nx, ny, nz) for warm-start
        """
        from .mesh_export import density_to_stl

        t0 = time.time()
        if self.backend == "openlsto":
            density = self._run_openlsto(sim_cfg, volfrac)
        elif self.backend == "simp_gpu":
            density = self._run_simp_gpu(sim_cfg, volfrac, x_init)
        else:
            density = self._run_simp(sim_cfg, volfrac, x_init)

        self.last_density = density
        print(f"[TopologySolver] {self.backend} done in {time.time()-t0:.1f}s  "
              f"mean_density={density.mean():.3f}")

        if not export_stl:
            return None

        ts  = int(time.time())
        out = str(Path(output_dir) / f"topo_{self.backend}_{ts}_mesh.stl")
        
        # Derive spacing from nx/ny/nz (baseline 100x20x20)
        lx = sim_cfg.get("lx_mm", 100.0)
        ly = sim_cfg.get("ly_mm", 20.0)
        lz = sim_cfg.get("lz_mm", 20.0)
        vsize = (lx/self.nx, ly/self.ny, lz/self.nz)
        
        density_to_stl(density, out, voxel_size_mm=vsize)
        return out

    def _run_simp(self, sim_cfg: dict, volfrac: float, x_init: np.ndarray | None = None) -> np.ndarray:
        from .simp_solver import SIMPSolver
        bcs = sim_cfg.get("boundary_conditions")
        mask = sim_cfg.get("preserved_mask")
        s = SIMPSolver(nx=self.nx, ny=self.ny, nz=self.nz, boundary_conditions=bcs, preserved_mask=mask)
        res = s.run(volfrac=volfrac, n_iters=self.n_iters,
                    force_mag=float(sim_cfg.get("force_n", 1000)))
        # SIMPSolver (CPU) doesn't have x_init yet, let's just use it as is
        self.last_compliance = s.last_compliance
        return res

    def _run_simp_gpu(self, sim_cfg: dict, volfrac: float, x_init: np.ndarray | None = None) -> np.ndarray:
        from .simp_solver_gpu import SIMPSolverGPU
        bcs = sim_cfg.get("boundary_conditions")
        mask = sim_cfg.get("preserved_mask")
        s = SIMPSolverGPU(nx=self.nx, ny=self.ny, nz=self.nz, boundary_conditions=bcs, preserved_mask=mask)
        res = s.run(volfrac=volfrac, n_iters=self.n_iters,
                    force_mag=float(sim_cfg.get("force_n", 1000)),
                    x_init=x_init)
        self.last_compliance = s.last_compliance
        return res

    def _run_openlsto(self, sim_cfg: dict, volfrac: float) -> np.ndarray:
        result = self._openlsto.solve_cantilever(
            nx=self.nx, ny=self.ny, nz=self.nz,
            force=float(sim_cfg.get("force_n", 1000)),
            volfrac=volfrac,
            n_iters=self.n_iters,
        )
        self.last_compliance = 0.0
        return np.array(result).reshape(self.nx, self.ny, self.nz)
