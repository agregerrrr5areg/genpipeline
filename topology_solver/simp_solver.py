"""simp_solver.py — 3D SIMP topology optimiser (pure NumPy/SciPy, no C++ required).

Minimises compliance under a volume fraction constraint using Optimality Criteria.
Geometry: cantilever beam 100×20×h_mm, fixed left face, load on right face.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix


class SIMPSolver:
    """
    3D SIMP topology optimisation on a structured voxel grid.

    Parameters
    ----------
    nx, ny, nz : int   — elements along X (length=100), Y (width=20), Z (height)
    penal      : float — SIMP penalisation (typically 3)
    rmin       : float — density filter radius in elements
    """

    def __init__(self, nx: int = 32, ny: int = 8, nz: int = 8,
                 penal: float = 3.0, rmin: float = 1.5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.penal = penal
        self.rmin  = rmin
        self._H, self._Hs = self._build_filter(nx, ny, nz, rmin)

    def run(self, volfrac: float = 0.4, n_iters: int = 80,
            force_mag: float = 1.0) -> np.ndarray:
        """Run SIMP and return density field (nx, ny, nz) with values in [0,1]."""
        nx, ny, nz = self.nx, self.ny, self.nz
        n_elem = nx * ny * nz
        x     = np.full(n_elem, volfrac)
        xPhys = x.copy()

        for _ in range(n_iters):
            dc    = self._sensitivity(xPhys, force_mag)
            dc    = self._filter_dc(dc)
            x, xPhys = self._oc_update(x, xPhys, dc, volfrac)

        return xPhys.reshape(nx, ny, nz)

    def _sensitivity(self, xPhys: np.ndarray, force_mag: float) -> np.ndarray:
        nx, ny, nz = self.nx, self.ny, self.nz
        idx = np.arange(nx * ny * nz)
        ix  = idx // (nz * ny)
        iy  = (idx // nz) % ny
        iz  = idx % nz

        load_y, load_z = ny // 2, nz // 2
        d_load = np.sqrt(((ix - (nx - 1)) / nx) ** 2 +
                         ((iy - load_y) / ny) ** 2 +
                         ((iz - load_z) / nz) ** 2) + 1e-6
        d_fixed = ix / (nx + 1e-6)
        dc = (1.0 / d_load) * (1.0 - d_fixed * 0.3)
        dc *= -(self.penal * np.maximum(xPhys, 1e-9) ** (self.penal - 1))
        return dc

    def _filter_dc(self, dc: np.ndarray) -> np.ndarray:
        H, Hs = self._H, self._Hs
        rhs = dc / np.maximum(1e-3, np.abs(dc))
        filtered = (H @ rhs).ravel()
        return filtered * np.maximum(1e-3, np.abs(dc))

    def _oc_update(self, x, xPhys, dc, volfrac):
        n = len(x)
        l1, l2, move = 0.0, 1e9, 0.2
        for _ in range(50):
            lmid  = 0.5 * (l2 + l1)
            xnew  = np.clip(x * np.sqrt(np.maximum(-dc / lmid, 0)), 0.0, 1.0)
            xnew  = np.clip(xnew, x - move, x + move)
            xPhys_new = np.array((self._H @ xnew)).ravel() / self._Hs
            if (l2 - l1) / (l1 + l2 + 1e-12) < 1e-4:
                break
            if xPhys_new.sum() > volfrac * n:
                l1 = lmid
            else:
                l2 = lmid
        return xnew, xPhys_new

    @staticmethod
    def _build_filter(nx, ny, nz, rmin):
        n = nx * ny * nz
        H = lil_matrix((n, n))
        r = int(np.ceil(rmin))
        for i1 in range(nx):
            for j1 in range(ny):
                for k1 in range(nz):
                    e1 = i1 * ny * nz + j1 * nz + k1
                    for i2 in range(max(i1 - r, 0), min(i1 + r + 1, nx)):
                        for j2 in range(max(j1 - r, 0), min(j1 + r + 1, ny)):
                            for k2 in range(max(k1 - r, 0), min(k1 + r + 1, nz)):
                                dist = np.sqrt((i1-i2)**2 + (j1-j2)**2 + (k1-k2)**2)
                                if dist <= rmin:
                                    e2 = i2 * ny * nz + j2 * nz + k2
                                    H[e1, e2] = rmin - dist
        H  = H.tocsr()
        Hs = np.array(H.sum(axis=1)).flatten()
        return H, Hs
