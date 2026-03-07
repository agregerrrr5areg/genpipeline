"""simp_solver.py — 3D SIMP topology optimiser (pure NumPy/SciPy, no C++ required).

Minimises compliance under a volume fraction constraint using Optimality Criteria.
Geometry: cantilever beam 100×20×h_mm, fixed left face, load on right face.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class SIMPSolver:
    """
    3D SIMP topology optimisation on a structured voxel grid.

    Parameters
    ----------
    nx, ny, nz : int   — elements along X (length=100), Y (width=20), Z (height)
    penal      : float — SIMP penalisation (typically 3)
    rmin       : float — density filter radius in elements
    """

    def __init__(
        self,
        nx: int = 32,
        ny: int = 8,
        nz: int = 8,
        penal: float = 3.0,
        rmin: float = 1.5,
        boundary_conditions: dict | None = None,
        preserved_mask: np.ndarray | None = None,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.penal = penal
        self.rmin = rmin
        self._H, self._Hs = self._build_filter(nx, ny, nz, rmin)
        self.Ke = self._get_Ke(nu=0.3)
        self._edof_mat, self._iK, self._jK = self._build_edof_mapping()
        self.last_compliance = 0.0

        # Non-Design Domain (Locked voxels)
        if preserved_mask is not None:
            self.preserved_mask = preserved_mask.flatten().astype(bool)
        else:
            self.preserved_mask = None

        self.bcs = {
            "fixed_face": "x_min",
            "load_face": "x_max",
            "load_dof": 2,  # Z
        }
        if boundary_conditions:
            self.bcs.update(boundary_conditions)

    def run(
        self, volfrac: float = 0.4, n_iters: int = 80, force_mag: float = 1.0
    ) -> np.ndarray:
        """Run SIMP and return density field (nx, ny, nz) with values in [0,1]."""
        nx, ny, nz = self.nx, self.ny, self.nz
        n_elem = nx * ny * nz
        x = np.full(n_elem, volfrac)

        # Apply initial preservation
        if self.preserved_mask is not None:
            x[self.preserved_mask] = 1.0

        xPhys = x.copy()

        for i in range(n_iters):
            dc = self._sensitivity(xPhys, force_mag)
            dc = self._filter_dc(dc)
            x, xPhys = self._oc_update(x, xPhys, dc, volfrac)

            if (i + 1) % 10 == 0:
                print(f"  [SIMP] Iter {i + 1:3d} | Mean xPhys: {xPhys.mean():.3f}")

        self.last_compliance = self._calculate_compliance(xPhys, force_mag)
        return xPhys.reshape(nx, ny, nz)

    def _get_bcs(self, force_mag: float) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        n_dof = 3 * n_nodes

        ix, iy, iz = np.meshgrid(
            np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij"
        )

        fixed_nodes_list = []
        ff = self.bcs.get("fixed_face", "x_min")
        if isinstance(ff, list):
            for fface in ff:
                mask = self._get_face_mask(fface, ix, iy, iz, nx, ny, nz)
                fixed_nodes_list.append(
                    (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask]
                )
        else:
            mask = self._get_face_mask(ff, ix, iy, iz, nx, ny, nz)
            fixed_nodes_list.append(
                (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask]
            )

        fixed_nodes = np.unique(np.concatenate(fixed_nodes_list))
        fixed_dofs = np.concatenate(
            [3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]
        )

        f = np.zeros(n_dof)
        lf = self.bcs.get("load_face", "x_max")
        ld = self.bcs.get("load_dof", 2)
        multi_load = self.bcs.get("multi_load", False)

        if multi_load and isinstance(ld, list):
            for load_dir in ld:
                mask_l = self._get_face_mask(lf, ix, iy, iz, nx, ny, nz)
                load_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask_l]
                valid_load_nodes = load_nodes[~np.isin(load_nodes, fixed_nodes)]
                if len(valid_load_nodes) > 0:
                    for node in valid_load_nodes[::3]:
                        f[3 * node + load_dir] -= force_mag / len(valid_load_nodes[::3])
        elif lf in ("center", "center_z"):
            mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
            dist = np.sqrt((ix - mid_x) ** 2 + (iy - mid_y) ** 2 + (iz - mid_z) ** 2)
            mask_l = dist <= 2
            load_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask_l]
            valid_load_nodes = load_nodes[~np.isin(load_nodes, fixed_nodes)]
            if len(valid_load_nodes) > 0:
                for node in valid_load_nodes[::2]:
                    f[3 * node + ld] -= force_mag / len(valid_load_nodes[::2])
        elif lf == "top":
            mask_l = iz == nz
            load_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask_l]
            valid_load_nodes = load_nodes[~np.isin(load_nodes, fixed_nodes)]
            if len(valid_load_nodes) > 0:
                mid_node = valid_load_nodes[len(valid_load_nodes) // 2]
                f[3 * mid_node + ld] = -force_mag
        elif lf == "x_max_offset":
            mask_l = (ix == nx) & (iy >= ny // 3) & (iy <= 2 * ny // 3)
            load_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask_l]
            valid_load_nodes = load_nodes[~np.isin(load_nodes, fixed_nodes)]
            if len(valid_load_nodes) > 0:
                mid_node = valid_load_nodes[len(valid_load_nodes) // 2]
                f[3 * mid_node + ld] = -force_mag
        else:
            mask_l = self._get_face_mask(lf, ix, iy, iz, nx, ny, nz)
            load_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask_l]
            valid_load_nodes = load_nodes[~np.isin(load_nodes, fixed_nodes)]

            if len(valid_load_nodes) > 0:
                mid_node = valid_load_nodes[len(valid_load_nodes) // 2]
                f[3 * mid_node + ld] = -force_mag
            else:
                mid_node = load_nodes[len(load_nodes) // 2]
                f[3 * mid_node + ld] = -force_mag

        return f, fixed_dofs

    def _get_face_mask(self, face, ix, iy, iz, nx, ny, nz):
        if face == "x_min":
            return ix == 0
        elif face == "x_max":
            return ix == nx
        elif face == "y_min":
            return iy == 0
        elif face == "y_max":
            return iy == ny
        elif face == "z_min":
            return iz == 0
        elif face == "z_max":
            return iz == nz
        else:
            return ix == nx

        return f, fixed_dofs

    def _calculate_compliance(self, xPhys: np.ndarray, force_mag: float) -> float:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        n_dof = 3 * n_nodes

        sK = (
            self.Ke.ravel()[np.newaxis, :]
            * (np.maximum(xPhys, 1e-3) ** self.penal)[:, np.newaxis]
        ).ravel()
        K = coo_matrix((sK, (self._iK, self._jK)), shape=(n_dof, n_dof)).tocsc()

        f, fixed_dofs = self._get_bcs(force_mag)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        u = np.zeros(n_dof)
        u[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], f[free_dofs])
        return float(f.T @ u)

    def _sensitivity(self, xPhys: np.ndarray, force_mag: float) -> np.ndarray:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        n_dof = 3 * n_nodes

        # Assemble global stiffness matrix K
        sK = (
            self.Ke.ravel()[np.newaxis, :]
            * (np.maximum(xPhys, 1e-3) ** self.penal)[:, np.newaxis]
        ).ravel()
        K = coo_matrix((sK, (self._iK, self._jK)), shape=(n_dof, n_dof)).tocsc()

        # Load and boundary conditions
        f, fixed_dofs = self._get_bcs(force_mag)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        # Solve Ku = f
        u = np.zeros(n_dof)
        u[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], f[free_dofs])

        # Compute sensitivities dc_e = -p * x_e**(p-1) * u_e^T @ Ke @ u_e
        u_e = u[self._edof_mat]
        ce = np.sum((u_e @ self.Ke) * u_e, axis=1)
        dc = -self.penal * (np.maximum(xPhys, 1e-3) ** (self.penal - 1)) * ce
        return dc

    def _filter_dc(self, dc: np.ndarray) -> np.ndarray:
        """Density filter chain rule: dC/dx = H^T @ (dC/dxPhys / Hs)."""
        return (self._H.T @ (dc / self._Hs)).ravel()

    def _oc_update(self, x, xPhys, dc, volfrac):
        n = len(x)
        l1, l2, move = 0.0, 1e9, 0.2
        for _ in range(50):
            lmid = 0.5 * (l2 + l1)
            xnew = np.clip(x * np.sqrt(np.maximum(-dc / lmid, 1e-12)), 0.0, 1.0)
            xnew = np.clip(xnew, x - move, x + move)

            # Re-apply non-design domain constraints
            if self.preserved_mask is not None:
                xnew[self.preserved_mask] = 1.0

            xPhys_new = np.array((self._H @ xnew)).ravel() / self._Hs

            # Ensure physical density also stays 1.0 in preserved regions
            if self.preserved_mask is not None:
                xPhys_new[self.preserved_mask] = 1.0

            if (l2 - l1) / (l1 + l2 + 1e-12) < 1e-4:
                break
            if xPhys_new.mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        return xnew, xPhys_new

    def _build_edof_mapping(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        n_elem = nx * ny * nz

        # Node ordering for C3D8: (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        n_base = np.array(
            [
                0,
                (ny + 1) * (nz + 1),
                (ny + 1) * (nz + 1) + (nz + 1),
                (nz + 1),
                1,
                (ny + 1) * (nz + 1) + 1,
                (ny + 1) * (nz + 1) + (nz + 1) + 1,
                (nz + 1) + 1,
            ]
        )
        edof_base = np.zeros(24, dtype=int)
        for i in range(8):
            edof_base[3 * i : 3 * i + 3] = 3 * n_base[i] + np.arange(3)

        edof_mat = np.zeros((n_elem, 24), dtype=int)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * ny * nz + j * nz + k
                    offset = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                    edof_mat[idx, :] = edof_base + 3 * offset

        iK = np.kron(edof_mat, np.ones((24, 1), dtype=int)).flatten()
        jK = np.kron(edof_mat, np.ones((1, 24), dtype=int)).flatten()
        return edof_mat, iK, jK

    @staticmethod
    def _get_Ke(nu: float = 0.3) -> np.ndarray:
        """Standard 24x24 element stiffness matrix for C3D8 unit cube (E=1)."""
        A = np.array(
            [
                [
                    32,
                    6,
                    -8,
                    6,
                    -6,
                    4,
                    3,
                    -6,
                    -10,
                    3,
                    -3,
                    -3,
                    -4,
                    -8,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -3,
                    -7,
                    -4,
                ],
                [
                    6,
                    32,
                    6,
                    -8,
                    4,
                    -6,
                    -6,
                    3,
                    -3,
                    -10,
                    3,
                    -3,
                    -3,
                    -4,
                    -8,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -3,
                    -4,
                    -4,
                    -7,
                ],
                [
                    -8,
                    6,
                    32,
                    6,
                    3,
                    -6,
                    -6,
                    4,
                    -3,
                    3,
                    -10,
                    3,
                    -4,
                    -4,
                    -4,
                    -7,
                    -8,
                    -4,
                    -4,
                    -7,
                    -3,
                    -4,
                    -3,
                    -4,
                ],
                [
                    6,
                    -8,
                    6,
                    32,
                    -6,
                    3,
                    4,
                    -6,
                    -3,
                    -3,
                    3,
                    -10,
                    -8,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -3,
                    -4,
                ],
                [
                    -6,
                    4,
                    3,
                    -6,
                    32,
                    6,
                    -8,
                    6,
                    -3,
                    -4,
                    -4,
                    -7,
                    -10,
                    3,
                    -3,
                    -3,
                    -4,
                    -8,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -3,
                ],
                [
                    4,
                    -6,
                    -6,
                    3,
                    6,
                    32,
                    6,
                    -8,
                    -4,
                    -7,
                    -4,
                    -3,
                    3,
                    -10,
                    3,
                    -3,
                    -3,
                    -4,
                    -8,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                ],
                [
                    3,
                    -6,
                    -6,
                    4,
                    -8,
                    6,
                    32,
                    6,
                    -4,
                    -3,
                    -4,
                    -7,
                    -3,
                    3,
                    -10,
                    3,
                    -4,
                    -4,
                    -4,
                    -7,
                    -8,
                    -4,
                    -4,
                    -7,
                ],
                [
                    -6,
                    3,
                    4,
                    -6,
                    6,
                    -8,
                    6,
                    32,
                    -7,
                    -4,
                    -3,
                    -4,
                    -3,
                    -3,
                    3,
                    -10,
                    -8,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -7,
                    -4,
                ],
                [
                    -10,
                    -3,
                    -3,
                    -3,
                    -3,
                    -4,
                    -4,
                    -7,
                    32,
                    6,
                    -8,
                    6,
                    -6,
                    4,
                    3,
                    -6,
                    -4,
                    -3,
                    -4,
                    -3,
                    -7,
                    -4,
                    -4,
                    -8,
                ],
                [
                    3,
                    -10,
                    3,
                    -3,
                    -4,
                    -7,
                    -3,
                    -4,
                    6,
                    32,
                    6,
                    -8,
                    4,
                    -6,
                    -6,
                    3,
                    -3,
                    -4,
                    -3,
                    -4,
                    -4,
                    -7,
                    -8,
                    -4,
                ],
                [
                    -3,
                    3,
                    -10,
                    3,
                    -4,
                    -4,
                    -4,
                    -7,
                    -8,
                    6,
                    32,
                    6,
                    3,
                    -6,
                    -6,
                    4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -3,
                    -4,
                    -4,
                    -4,
                ],
                [
                    -3,
                    -3,
                    3,
                    -10,
                    -7,
                    -3,
                    -4,
                    -4,
                    6,
                    -8,
                    6,
                    32,
                    -6,
                    3,
                    4,
                    -6,
                    -4,
                    -7,
                    -8,
                    -4,
                    -4,
                    -3,
                    -4,
                    -3,
                ],
                [
                    -4,
                    -3,
                    -4,
                    -8,
                    -10,
                    3,
                    -3,
                    -3,
                    -6,
                    4,
                    3,
                    -6,
                    32,
                    6,
                    -8,
                    6,
                    -4,
                    -3,
                    -7,
                    -4,
                    -3,
                    -4,
                    -4,
                    -7,
                ],
                [
                    -8,
                    -4,
                    -4,
                    -4,
                    3,
                    -10,
                    3,
                    -3,
                    4,
                    -6,
                    -6,
                    3,
                    6,
                    32,
                    6,
                    -8,
                    -3,
                    -4,
                    -4,
                    -7,
                    -8,
                    -4,
                    -7,
                    -4,
                ],
                [
                    -4,
                    -8,
                    -4,
                    -3,
                    -3,
                    3,
                    -10,
                    3,
                    3,
                    -6,
                    -6,
                    4,
                    -8,
                    6,
                    32,
                    6,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -4,
                    -4,
                    -7,
                ],
                [
                    -3,
                    -4,
                    -7,
                    -4,
                    -3,
                    -3,
                    3,
                    -10,
                    -6,
                    3,
                    4,
                    -6,
                    6,
                    -8,
                    6,
                    32,
                    -7,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                ],
                [
                    -4,
                    -7,
                    -8,
                    -7,
                    -4,
                    -3,
                    -4,
                    -8,
                    -4,
                    -3,
                    -7,
                    -4,
                    -4,
                    -3,
                    -4,
                    -7,
                    32,
                    6,
                    -8,
                    6,
                    -6,
                    4,
                    3,
                    -6,
                ],
                [
                    -3,
                    -4,
                    -4,
                    -4,
                    -8,
                    -4,
                    -4,
                    -4,
                    -3,
                    -4,
                    -4,
                    -7,
                    -3,
                    -4,
                    -7,
                    -4,
                    6,
                    32,
                    6,
                    -8,
                    4,
                    -6,
                    -6,
                    3,
                ],
                [
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -8,
                    -4,
                    -3,
                    -4,
                    -3,
                    -3,
                    -8,
                    -7,
                    -4,
                    -4,
                    -3,
                    -8,
                    6,
                    32,
                    6,
                    3,
                    -6,
                    -6,
                    4,
                ],
                [
                    -3,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -4,
                    -4,
                    -4,
                    -7,
                    -3,
                    -4,
                    6,
                    -8,
                    6,
                    32,
                    -6,
                    3,
                    4,
                    -6,
                ],
                [
                    -4,
                    -3,
                    -3,
                    -3,
                    -4,
                    -7,
                    -8,
                    -7,
                    -7,
                    -4,
                    -3,
                    -4,
                    -3,
                    -8,
                    -4,
                    -7,
                    -6,
                    4,
                    3,
                    -6,
                    32,
                    6,
                    -8,
                    6,
                ],
                [
                    -3,
                    -4,
                    -4,
                    -4,
                    -7,
                    -4,
                    -4,
                    -4,
                    -4,
                    -7,
                    -4,
                    -3,
                    -4,
                    -4,
                    -4,
                    -4,
                    4,
                    -6,
                    -6,
                    3,
                    6,
                    32,
                    6,
                    -8,
                ],
                [
                    -7,
                    -4,
                    -3,
                    -3,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -8,
                    -4,
                    -4,
                    -4,
                    -7,
                    -4,
                    -3,
                    3,
                    -6,
                    -6,
                    4,
                    -8,
                    6,
                    32,
                    6,
                ],
                [
                    -4,
                    -7,
                    -4,
                    -4,
                    -3,
                    -4,
                    -7,
                    -4,
                    -8,
                    -4,
                    -4,
                    -3,
                    -7,
                    -4,
                    -7,
                    -4,
                    -6,
                    3,
                    4,
                    -6,
                    6,
                    -8,
                    6,
                    32,
                ],
            ]
        )
        B = np.array(
            [
                [
                    16,
                    0,
                    8,
                    0,
                    8,
                    4,
                    -4,
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    -4,
                    -8,
                    -4,
                    0,
                    -8,
                    -4,
                    -4,
                    0,
                    -4,
                    -8,
                    -8,
                    -4,
                ],
                [
                    0,
                    16,
                    0,
                    8,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    -8,
                    0,
                    -4,
                    -8,
                    -4,
                    -8,
                    -4,
                    -4,
                    -8,
                    0,
                    -4,
                    -8,
                    -4,
                    -4,
                    -8,
                ],
                [
                    8,
                    0,
                    16,
                    0,
                    -4,
                    0,
                    8,
                    4,
                    -4,
                    0,
                    -8,
                    0,
                    -4,
                    -4,
                    -4,
                    -8,
                    -8,
                    -4,
                    -8,
                    -4,
                    -4,
                    0,
                    -4,
                    -8,
                ],
                [
                    0,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    -8,
                    -8,
                    -4,
                    -4,
                    -8,
                    -4,
                    -8,
                    -4,
                    -8,
                    0,
                    -4,
                    -8,
                    -4,
                ],
                [
                    8,
                    4,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    0,
                    4,
                    0,
                    0,
                    4,
                    -8,
                    0,
                    4,
                    0,
                    -4,
                    -8,
                    -4,
                    0,
                    0,
                    -4,
                    -8,
                    0,
                ],
                [
                    4,
                    8,
                    0,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    0,
                    4,
                    4,
                    0,
                    0,
                    -8,
                    0,
                    4,
                    -8,
                    -4,
                    0,
                    -4,
                    4,
                    0,
                    0,
                    -8,
                ],
                [
                    -4,
                    0,
                    8,
                    4,
                    8,
                    0,
                    16,
                    0,
                    0,
                    4,
                    4,
                    0,
                    4,
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    -4,
                    -8,
                    -8,
                    -4,
                    0,
                    -4,
                ],
                [
                    0,
                    -4,
                    4,
                    8,
                    0,
                    8,
                    0,
                    16,
                    4,
                    0,
                    0,
                    4,
                    0,
                    4,
                    0,
                    -8,
                    0,
                    -4,
                    -8,
                    -4,
                    -4,
                    -8,
                    -4,
                    0,
                ],
                [
                    -8,
                    0,
                    -4,
                    0,
                    4,
                    0,
                    0,
                    4,
                    16,
                    0,
                    8,
                    0,
                    8,
                    4,
                    -4,
                    0,
                    -4,
                    0,
                    0,
                    4,
                    -8,
                    0,
                    -4,
                    0,
                ],
                [
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    4,
                    4,
                    0,
                    0,
                    16,
                    0,
                    8,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    -4,
                    4,
                    0,
                    0,
                    -8,
                    0,
                    -4,
                ],
                [
                    -4,
                    0,
                    -8,
                    0,
                    0,
                    4,
                    4,
                    0,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    0,
                    8,
                    4,
                    0,
                    4,
                    4,
                    0,
                    -4,
                    0,
                    -8,
                    0,
                ],
                [
                    0,
                    -4,
                    0,
                    -8,
                    4,
                    0,
                    0,
                    4,
                    0,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    4,
                    8,
                    4,
                    0,
                    0,
                    4,
                    0,
                    -4,
                    0,
                    -8,
                ],
                [
                    -4,
                    -8,
                    -4,
                    -8,
                    -8,
                    0,
                    4,
                    0,
                    8,
                    4,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    0,
                    8,
                    4,
                    -4,
                    0,
                    -4,
                    0,
                    -4,
                    0,
                ],
                [
                    -8,
                    -4,
                    -4,
                    -4,
                    0,
                    -8,
                    0,
                    4,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    -4,
                    0,
                    -4,
                ],
                [
                    -4,
                    -8,
                    -4,
                    -4,
                    4,
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    8,
                    4,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    0,
                    8,
                    4,
                    0,
                    4,
                    4,
                    0,
                ],
                [
                    0,
                    -4,
                    -8,
                    -8,
                    0,
                    4,
                    0,
                    -8,
                    0,
                    -4,
                    4,
                    8,
                    0,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    4,
                    8,
                    4,
                    0,
                    0,
                    4,
                ],
                [
                    -8,
                    -4,
                    -8,
                    -4,
                    -4,
                    -8,
                    -4,
                    0,
                    -4,
                    0,
                    0,
                    4,
                    8,
                    4,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    0,
                    8,
                    4,
                    -4,
                    0,
                ],
                [
                    -4,
                    -8,
                    -4,
                    -8,
                    -8,
                    -4,
                    0,
                    -4,
                    0,
                    -4,
                    4,
                    0,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    4,
                    8,
                    0,
                    -4,
                ],
                [
                    -4,
                    0,
                    -8,
                    -4,
                    -4,
                    0,
                    -4,
                    -8,
                    0,
                    4,
                    4,
                    0,
                    -4,
                    0,
                    8,
                    4,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    0,
                    8,
                    4,
                ],
                [
                    0,
                    -4,
                    -4,
                    -8,
                    0,
                    -4,
                    -8,
                    -4,
                    4,
                    0,
                    0,
                    4,
                    0,
                    -4,
                    4,
                    8,
                    0,
                    8,
                    0,
                    16,
                    0,
                    -4,
                    4,
                    8,
                ],
                [
                    -4,
                    -8,
                    -4,
                    0,
                    0,
                    4,
                    -8,
                    -4,
                    -8,
                    0,
                    -4,
                    0,
                    -4,
                    0,
                    0,
                    4,
                    8,
                    4,
                    -4,
                    0,
                    16,
                    0,
                    8,
                    0,
                ],
                [
                    -8,
                    -4,
                    0,
                    -4,
                    -4,
                    0,
                    -4,
                    -8,
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    -4,
                    4,
                    0,
                    4,
                    8,
                    0,
                    -4,
                    0,
                    16,
                    0,
                    8,
                ],
                [
                    -8,
                    -4,
                    -4,
                    -3,
                    -8,
                    0,
                    0,
                    -4,
                    -4,
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    4,
                    0,
                    -4,
                    0,
                    8,
                    4,
                    8,
                    0,
                    16,
                    0,
                ],
                [
                    -4,
                    -8,
                    -4,
                    -4,
                    0,
                    -8,
                    -4,
                    0,
                    0,
                    -4,
                    0,
                    -8,
                    0,
                    -4,
                    0,
                    4,
                    0,
                    -4,
                    4,
                    8,
                    0,
                    8,
                    0,
                    16,
                ],
            ]
        )
        return 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu) * 72.0) * (A + nu * B)

    @staticmethod
    def _build_filter(nx, ny, nz, rmin):
        n = nx * ny * nz
        from scipy.sparse import lil_matrix

        H = lil_matrix((n, n))
        r = int(np.ceil(rmin))
        for i1 in range(nx):
            for j1 in range(ny):
                for k1 in range(nz):
                    e1 = i1 * ny * nz + j1 * nz + k1
                    for i2 in range(max(i1 - r, 0), min(i1 + r + 1, nx)):
                        for j2 in range(max(j1 - r, 0), min(j1 + r + 1, ny)):
                            for k2 in range(max(k1 - r, 0), min(k1 + r + 1, nz)):
                                dist = np.sqrt(
                                    (i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2
                                )
                                if dist <= rmin:
                                    e2 = i2 * ny * nz + j2 * nz + k2
                                    H[e1, e2] = rmin - dist
        H = H.tocsr()
        Hs = np.array(H.sum(axis=1)).flatten()
        return H, Hs
