"""simp_solver_gpu.py — PyTorch-based 3D SIMP topology optimiser for GPU."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


class SIMPSolverGPU:
    """
    3D SIMP topology optimisation on a structured voxel grid using PyTorch CUDA.
    """

    _RESOURCE_CACHE = {}

    def __init__(
        self,
        nx: int = 32,
        ny: int = 8,
        nz: int = 8,
        penal: float = 3.0,
        rmin: float = 1.5,
        boundary_conditions: dict | None = None,
        device: str = "cuda",
        preserved_mask: np.ndarray | None = None,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.penal = penal
        self.rmin = rmin
        self.device = device if torch.cuda.is_available() else "cpu"

        cache_key = (nx, ny, nz, rmin, self.device)
        if cache_key not in SIMPSolverGPU._RESOURCE_CACHE:
            Ke = torch.from_numpy(self._get_Ke(nu=0.3)).double().to(self.device)
            edof_mat = self._build_edof_mapping().to(self.device)
            H, Hs = self._build_filter(nx, ny, nz, rmin)
            SIMPSolverGPU._RESOURCE_CACHE[cache_key] = {
                "Ke": Ke,
                "edof_mat": edof_mat,
                "H": H,
                "Hs": Hs,
            }

        res = SIMPSolverGPU._RESOURCE_CACHE[cache_key]
        self.Ke = res["Ke"]
        self._edof_mat = res["edof_mat"]
        self._H = res["H"]
        self._Hs = res["Hs"]

        self.last_compliance = 0.0
        if preserved_mask is not None:
            self.preserved_mask = (
                torch.from_numpy(preserved_mask.flatten()).bool().to(self.device)
            )
        else:
            self.preserved_mask = None

        self.bcs = {"fixed_face": "x_min", "load_face": "x_max", "load_dof": 2}
        if boundary_conditions:
            self.bcs.update(boundary_conditions)

    def run(
        self,
        volfrac: float = 0.4,
        n_iters: int = 80,
        force_mag: float = 1.0,
        x_init: np.ndarray | None = None,
    ) -> np.ndarray:
        n_elem = self.nx * self.ny * self.nz
        if x_init is not None:
            x = torch.from_numpy(x_init.flatten()).to(self.device).double()
            x = x * (volfrac / x.mean().clamp(min=1e-6))
            x = x.clamp(1e-3, 1.0)
        else:
            x = torch.full((n_elem,), volfrac, device=self.device, dtype=torch.float64)

        if self.preserved_mask is not None:
            x[self.preserved_mask] = 1.0
        xPhys = x.clone()

        for i in range(n_iters):
            dc = self._sensitivity(xPhys, force_mag)
            if torch.isnan(dc).any():
                print(f"  [SIMP-GPU] FATAL: NaN in sensitivity at iter {i}")
                break
            dc = self._filter_dc(dc)
            x, xPhys = self._oc_update(x, xPhys, dc, volfrac)

            if (i + 1) % 10 == 0:
                print(
                    f"  [SIMP-GPU] Iter {i + 1:3d} | Mean xPhys: {xPhys.mean().item():.3f}"
                )

        self.last_compliance = self._calculate_compliance(xPhys, force_mag)
        return xPhys.reshape(self.nx, self.ny, self.nz).cpu().numpy()

    def _get_bcs(self, force_mag: float) -> tuple[torch.Tensor, torch.Tensor]:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        n_dof = 3 * n_nodes
        ix = torch.arange(nx + 1, device=self.device)
        iy = torch.arange(ny + 1, device=self.device)
        iz = torch.arange(nz + 1, device=self.device)
        ix, iy, iz = torch.meshgrid(ix, iy, iz, indexing="ij")

        # Fixed Face
        ff = self.bcs.get("fixed_face", "x_min")
        mask = (
            (ix == 0)
            if ff == "x_min"
            else (ix == nx)
            if ff == "x_max"
            else (iy == 0)
            if ff == "y_min"
            else (iy == ny)
            if ff == "y_max"
            else (iz == 0)
            if ff == "z_min"
            else (iz == nz)
            if ff == "z_max"
            else (ix == 0)
        )
        fixed_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask]
        fixed_dofs = torch.cat(
            [3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]
        )

        # Load Face
        f = torch.zeros(n_dof, device=self.device, dtype=torch.float64)
        lf = self.bcs.get("load_face", "x_max")
        ld = self.bcs.get("load_dof", 2)
        mask_l = (
            (ix == nx)
            if lf == "x_max"
            else (ix == 0)
            if lf == "x_min"
            else (iy == ny)
            if lf == "y_max"
            else (iy == 0)
            if lf == "y_min"
            else (iz == nz)
            if lf == "z_max"
            else (iz == 0)
            if lf == "z_min"
            else (ix == nx)
        )
        load_nodes = (ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz)[mask_l]

        # Safety: avoid applying load to a fixed node
        if len(load_nodes) > 0:
            is_fixed = torch.isin(load_nodes, fixed_nodes)
            valid_nodes = load_nodes[~is_fixed]
            if len(valid_nodes) > 0:
                node = valid_nodes[len(valid_nodes) // 2]
            else:
                node = load_nodes[len(load_nodes) // 2]
            f[3 * node + ld] = -float(force_mag)

        return f, fixed_dofs

    def _assemble_K(self, xPhys: torch.Tensor) -> torch.Tensor:
        n_dof = 3 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        # Ensure E is flattened for broad-cast multiplication with Ke
        E = torch.clamp(xPhys.flatten(), min=1e-3) ** self.penal
        iK = self._edof_mat.repeat_interleave(24)
        jK = self._edof_mat.repeat(1, 24).flatten()
        sK = (self.Ke.flatten()[None, :] * E[:, None]).flatten()
        K = torch.sparse_coo_tensor(torch.stack([iK, jK]), sK, (n_dof, n_dof))
        return K.coalesce().to_sparse_csr()

    @staticmethod
    def _pcg(A_csr, b, M_inv_diag, tol=1e-7, max_iter=2000):
        x = torch.zeros_like(b)
        r = b - torch.mv(A_csr, x)
        if r.norm() < tol * b.norm():
            return x
        z = M_inv_diag * r
        p = z.clone()
        rz = torch.dot(r, z)
        for _ in range(max_iter):
            Ap = torch.mv(A_csr, p)
            denom = torch.dot(p, Ap)
            if denom < 1e-25:
                break
            alpha = rz / denom
            x = x + alpha * p
            r = r - alpha * Ap
            if r.norm() < tol * b.norm():
                break
            z = M_inv_diag * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x

    def _pcg_matrix_free(self, xPhys, b, fixed_dofs, tol=1e-7, max_iter=2000):
        """Preconditioned Conjugate Gradient without explicit matrix assembly."""
        from cuda_kernels import fused_spmv

        n_dof = b.shape[0]

        x = torch.zeros_like(b)

        # Matrix-free SpMV operator
        def K_op(p_vec):
            y = fused_spmv(
                xPhys,
                p_vec,
                self.Ke,
                self._edof_mat,
                self.penal,
                self.nx,
                self.ny,
                self.nz,
            )
            y[fixed_dofs] = 0.0  # Enforce BCs
            return y + 1e-9 * p_vec  # Diagonal regularizer

        r = b.clone()
        r[fixed_dofs] = 0.0

        # Jacobi Preconditioner Proxy
        # We assume a base stiffness for preconditioning
        diag = torch.full((n_dof,), 1e-3, device=self.device, dtype=torch.float64)
        M_inv = 1.0 / diag

        z = M_inv * r
        p = z.clone()
        rz = torch.dot(r, z)

        for _ in range(max_iter):
            Ap = K_op(p)
            denom = torch.dot(p, Ap)
            if denom < 1e-25:
                break

            alpha = rz / denom
            x += alpha * p
            r -= alpha * Ap
            if r.norm() < tol * b.norm():
                break

            z = M_inv * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x

    def _solve(self, K, f, fixed_dofs):
        # Extract diagonal via COO format (CSR .diagonal() unsupported in PyTorch)
        K_coo = K.to_sparse()
        idx = K_coo.indices()
        dmask = idx[0] == idx[1]
        diag = torch.zeros(K.shape[0], device=K.device, dtype=K.dtype)
        diag.scatter_add_(0, idx[0][dmask], K_coo.values()[dmask])
        M_inv_diag = 1.0 / diag.clamp(min=1e-10)
        f_bc = f.clone()
        f_bc[fixed_dofs] = 0.0
        return self._pcg(K, f_bc, M_inv_diag)

    def _calculate_compliance(self, xPhys: torch.Tensor, force_mag: float) -> float:
        f, fixed_dofs = self._get_bcs(force_mag)
        K = self._assemble_K(xPhys)
        u = self._solve(K, f, fixed_dofs)
        return (f @ u).item()

    def _sensitivity(self, xPhys: torch.Tensor, force_mag: float) -> torch.Tensor:
        f, fixed_dofs = self._get_bcs(force_mag)
        K = self._assemble_K(xPhys)
        u = self._solve(K, f, fixed_dofs)

        # Use new Vectorized CUDA Kernel
        from .. import cuda_kernels

        dc = cuda_kernels.simp_sensitivity(
            xPhys.flatten(),
            u,
            self.Ke,
            self._edof_mat,
            float(self.penal),
            int(self.nx),
            int(self.ny),
            int(self.nz),
        ).view(self.nx, self.ny, self.nz)
        return dc

    def _filter_dc(self, dc: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self._H.t(), (dc / self._Hs).view(-1, 1)).flatten()

    def _oc_update(self, x, xPhys, dc, volfrac):
        move, l1, l2 = 0.2, 0.0, 1e12
        for _ in range(60):
            lmid = 0.5 * (l2 + l1)
            xnew = torch.clamp(
                x * torch.sqrt(torch.clamp(-dc / lmid, min=1e-15)), 0.0, 1.0
            )
            xnew = torch.clamp(xnew, x - move, x + move)
            if self.preserved_mask is not None:
                xnew[self.preserved_mask] = 1.0
            xPhys_new = torch.sparse.mm(self._H, xnew.view(-1, 1)).flatten() / self._Hs
            if self.preserved_mask is not None:
                xPhys_new[self.preserved_mask] = 1.0
            if (l2 - l1) / (l1 + l2 + 1e-15) < 1e-5:
                break
            if xPhys_new.mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        return xnew, xPhys_new

    def _build_edof_mapping(self):
        nx, ny, nz = self.nx, self.ny, self.nz
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
        edof_mat = np.zeros((nx * ny * nz, 24), dtype=int)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    edof_mat[i * ny * nz + j * nz + k, :] = edof_base + 3 * (
                        i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                    )
        return torch.from_numpy(edof_mat).long()

    def _build_filter(self, nx, ny, nz, rmin):
        n, r = nx * ny * nz, int(np.ceil(rmin))
        rows, cols, vals = [], [], []
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
                                    rows.append(e1)
                                    cols.append(i2 * ny * nz + j2 * nz + k2)
                                    vals.append(rmin - dist)
        H = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float64), (n, n)
        )
        H = H.to(self.device).to_sparse_csr()
        Hs = torch.sparse.mm(
            H, torch.ones((n, 1), device=self.device, dtype=torch.float64)
        ).flatten()
        return H, Hs

    @staticmethod
    def _get_Ke(nu: float = 0.3) -> np.ndarray:
        from .simp_solver import SIMPSolver

        return SIMPSolver._get_Ke(nu)
