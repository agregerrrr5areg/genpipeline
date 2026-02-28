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

    Uses OpenLSTO if built; otherwise SIMP fallback.

    Parameters
    ----------
    nx, ny, nz : voxel grid resolution
    n_iters    : solver iterations
    """

    def __init__(self, nx: int = 32, ny: int = 8, nz: int = 8, n_iters: int = 80):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_iters = n_iters
        self.last_density: np.ndarray | None = None
        self._openlsto = _try_openlsto()
        self.backend = "openlsto" if self._openlsto else "simp"

    def run(self, sim_cfg: dict, output_dir: str, volfrac: float = 0.4) -> str:
        """
        Run topology optimisation, return path to output STL.

        Parameters
        ----------
        sim_cfg    : dict with at least {"force_n": float}
        output_dir : directory to write STL
        volfrac    : target volume fraction (0–1)
        """
        from topology.mesh_export import density_to_stl

        t0 = time.time()
        if self._openlsto:
            density = self._run_openlsto(sim_cfg, volfrac)
        else:
            density = self._run_simp(sim_cfg, volfrac)

        self.last_density = density
        print(f"[TopologySolver] {self.backend} done in {time.time()-t0:.1f}s  "
              f"mean_density={density.mean():.3f}")

        ts  = int(time.time())
        out = str(Path(output_dir) / f"topo_{self.backend}_{ts}_mesh.stl")
        density_to_stl(density, out,
                       voxel_size_mm=(100/self.nx, 20/self.ny, 20/self.nz))
        return out

    def _run_simp(self, sim_cfg: dict, volfrac: float) -> np.ndarray:
        from topology.simp_solver import SIMPSolver
        s = SIMPSolver(nx=self.nx, ny=self.ny, nz=self.nz)
        return s.run(volfrac=volfrac, n_iters=self.n_iters,
                     force_mag=float(sim_cfg.get("force_n", 1000)))

    def _run_openlsto(self, sim_cfg: dict, volfrac: float) -> np.ndarray:
        result = self._openlsto.solve_cantilever(
            nx=self.nx, ny=self.ny, nz=self.nz,
            force=float(sim_cfg.get("force_n", 1000)),
            volfrac=volfrac,
            n_iters=self.n_iters,
        )
        return np.array(result).reshape(self.nx, self.ny, self.nz)
