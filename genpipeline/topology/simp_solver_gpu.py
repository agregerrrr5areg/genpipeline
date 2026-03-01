
"""simp_solver_gpu.py — PyTorch-based 3D SIMP topology optimiser for GPU."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

class SIMPSolverGPU:
    """
    3D SIMP topology optimisation on a structured voxel grid using PyTorch CUDA.

    Parameters
    ----------
    nx, ny, nz : int   — elements along X, Y, Z
    penal      : float — SIMP penalisation
    rmin       : float — density filter radius
    device     : str   — 'cuda' or 'cpu'
    """

    def __init__(self, nx: int = 32, ny: int = 8, nz: int = 8,
                 penal: float = 3.0, rmin: float = 1.5,
                 boundary_conditions: dict | None = None,
                 device: str = "cuda",
                 preserved_mask: np.ndarray | None = None):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.penal = penal
        self.rmin  = rmin
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Precompute geometry-independent data
        self.Ke = torch.from_numpy(self._get_Ke(nu=0.3)).double().to(self.device)
        self._edof_mat = self._build_edof_mapping().to(self.device)
        self._H, self._Hs = self._build_filter(nx, ny, nz, rmin)
        self._H = self._H.to(self.device)
        self._Hs = self._Hs.to(self.device)
        
        self.last_compliance = 0.0
        
        # Non-Design Domain (Locked voxels)
        if preserved_mask is not None:
            self.preserved_mask = torch.from_numpy(preserved_mask.flatten()).bool().to(self.device)
        else:
            self.preserved_mask = None

        self.bcs = {
            "fixed_face": "x_min",
            "load_face": "x_max",
            "load_dof": 2 # Z
        }
        if boundary_conditions:
            self.bcs.update(boundary_conditions)

    def run(self, volfrac: float = 0.4, n_iters: int = 80,
            force_mag: float = 1.0) -> np.ndarray:
        """Run SIMP on GPU and return density field."""
        n_elem = self.nx * self.ny * self.nz
        x = torch.full((n_elem,), volfrac, device=self.device, dtype=torch.float64)
        
        if self.preserved_mask is not None:
            x[self.preserved_mask] = 1.0
            
        xPhys = x.clone()

        for i in range(n_iters):
            dc = self._sensitivity(xPhys, force_mag)
            dc = self._filter_dc(dc)
            x, xPhys = self._oc_update(x, xPhys, dc, volfrac)
            
            if (i+1) % 10 == 0:
                print(f"  [SIMP-GPU] Iter {i+1:3d} | Mean xPhys: {xPhys.mean().item():.3f}")

        self.last_compliance = self._calculate_compliance(xPhys, force_mag)
        return xPhys.reshape(self.nx, self.ny, self.nz).cpu().numpy()

    def _get_bcs(self, force_mag: float) -> tuple[torch.Tensor, torch.Tensor]:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        n_dof = 3 * n_nodes
        
        ix = torch.arange(nx+1, device=self.device)
        iy = torch.arange(ny+1, device=self.device)
        iz = torch.arange(nz+1, device=self.device)
        ix, iy, iz = torch.meshgrid(ix, iy, iz, indexing='ij')
        
        # Fixed DOFs
        ff = self.bcs.get("fixed_face", "x_min")
        if ff == "x_min": mask = (ix == 0)
        elif ff == "x_max": mask = (ix == nx)
        elif ff == "y_min": mask = (iy == 0)
        elif ff == "y_max": mask = (iy == ny)
        elif ff == "z_min": mask = (iz == 0)
        elif ff == "z_max": mask = (iz == nz)
        else: mask = (ix == 0)
        
        fixed_nodes = (ix * (ny+1)*(nz+1) + iy * (nz+1) + iz)[mask]
        fixed_dofs = torch.cat([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
        
        # Load Vector
        f = torch.zeros(n_dof, device=self.device, dtype=torch.float64)
        lf = self.bcs.get("load_face", "x_max")
        ld = self.bcs.get("load_dof", 2)
        
        if lf == "x_min": mask_l = (ix == 0)
        elif lf == "x_max": mask_l = (ix == nx)
        elif lf == "y_min": mask_l = (iy == 0)
        elif lf == "y_max": mask_l = (iy == ny)
        elif lf == "z_min": mask_l = (iz == 0)
        elif lf == "z_max": mask_l = (iz == nz)
        else: mask_l = (ix == nx)
        
        load_nodes = (ix * (ny+1)*(nz+1) + iy * (nz+1) + iz)[mask_l]
        if len(load_nodes) > 0:
            mask_fixed = torch.isin(load_nodes, fixed_nodes)
            valid_load_nodes = load_nodes[~mask_fixed]
            node = valid_load_nodes[len(valid_load_nodes)//2] if len(valid_load_nodes) > 0 else load_nodes[len(load_nodes)//2]
            f[3*node + ld] = -float(force_mag)
            
        return f, fixed_dofs

    def _assemble_K(self, xPhys: torch.Tensor) -> torch.Tensor:
        n_elem = self.nx * self.ny * self.nz
        n_nodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        n_dof = 3 * n_nodes
        
        # Penalized stiffness
        E = torch.clamp(xPhys, min=1e-3) ** self.penal
        
        # Global assembly indices
        iK = self._edof_mat.repeat_interleave(24)
        jK = self._edof_mat.repeat(1, 24).flatten()
        sK = (self.Ke.flatten()[None, :] * E[:, None]).flatten()
        
        # Add tiny epsilon to the whole diagonal for stability
        K = torch.sparse_coo_tensor(torch.stack([iK, jK]), sK, (n_dof, n_dof))
        # Summing to account for duplicate indices (overlapping element stiffness matrices)
        K = K.coalesce().to_sparse_csr()
        return K

    def _solve(self, K, f, fixed_dofs):
        n_dof = f.shape[0]
        free_dofs = torch.ones(n_dof, dtype=torch.bool, device=self.device)
        free_dofs[fixed_dofs] = False
        free_indices = torch.where(free_dofs)[0]
        
        # Extract submatrix for free DOFs
        # We add a tiny shift to the diagonal of the submatrix for extra stability
        K_free = K.to_dense()[free_indices, :][:, free_indices]
        K_free += torch.eye(K_free.shape[0], device=self.device, dtype=torch.float64) * 1e-6
        f_free = f[free_indices]
        
        u_free = torch.linalg.solve(K_free, f_free)
        
        u = torch.zeros(n_dof, device=self.device, dtype=torch.float64)
        u[free_indices] = u_free
        return u

    def _calculate_compliance(self, xPhys: torch.Tensor, force_mag: float) -> float:
        f, fixed_dofs = self._get_bcs(force_mag)
        K = self._assemble_K(xPhys)
        u = self._solve(K, f, fixed_dofs)
        return (f @ u).item()

    def _sensitivity(self, xPhys: torch.Tensor, force_mag: float) -> torch.Tensor:
        f, fixed_dofs = self._get_bcs(force_mag)
        K = self._assemble_K(xPhys)
        u = self._solve(K, f, fixed_dofs)
        
        # ce = u_e^T Ke u_e
        u_e = u[self._edof_mat]
        ce = torch.sum((u_e @ self.Ke) * u_e, dim=1)
        dc = -self.penal * (torch.clamp(xPhys, min=1e-3)**(self.penal-1)) * ce
        return dc

    def _filter_dc(self, dc: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self._H.t(), (dc / self._Hs).view(-1, 1)).flatten()

    def _oc_update(self, x, xPhys, dc, volfrac):
        move = 0.2
        l1, l2 = 0.0, 1e9
        
        # Iterative update on GPU
        for _ in range(50):
            lmid = 0.5 * (l2 + l1)
            xnew = torch.clamp(x * torch.sqrt(torch.clamp(-dc / lmid, min=1e-12)), 0.0, 1.0)
            xnew = torch.clamp(xnew, x - move, x + move)
            
            # Re-apply non-design domain constraints
            if self.preserved_mask is not None:
                xnew[self.preserved_mask] = 1.0

            xPhys_new = torch.sparse.mm(self._H, xnew.view(-1, 1)).flatten() / self._Hs
            
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
        n_base = np.array([0, (ny+1)*(nz+1), (ny+1)*(nz+1) + (nz+1), (nz+1),
                           1, (ny+1)*(nz+1) + 1, (ny+1)*(nz+1) + (nz+1) + 1, (nz+1) + 1])
        edof_base = np.zeros(24, dtype=int)
        for i in range(8):
            edof_base[3*i:3*i+3] = 3*n_base[i] + np.arange(3)
        
        edof_mat = np.zeros((n_elem, 24), dtype=int)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * ny * nz + j * nz + k
                    offset = i * (ny+1)*(nz+1) + j * (nz+1) + k
                    edof_mat[idx, :] = edof_base + 3 * offset
        return torch.from_numpy(edof_mat).long()

    def _build_filter(self, nx, ny, nz, rmin):
        n = nx * ny * nz
        r = int(np.ceil(rmin))
        
        rows, cols, vals = [], [], []
        for i1 in range(nx):
            for j1 in range(ny):
                for k1 in range(nz):
                    e1 = i1 * ny * nz + j1 * nz + k1
                    for i2 in range(max(i1-r,0), min(i1+r+1,nx)):
                        for j2 in range(max(j1-r,0), min(j1+r+1,ny)):
                            for k2 in range(max(k1-r,0), min(k1+r+1,nz)):
                                dist = np.sqrt((i1-i2)**2 + (j1-j2)**2 + (k1-k2)**2)
                                if dist <= rmin:
                                    e2 = i2 * ny * nz + j2 * nz + k2
                                    rows.append(e1)
                                    cols.append(e2)
                                    vals.append(rmin - dist)
                                    
        H = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float64), (n, n))
        H = H.to(self.device).to_sparse_csr()
        Hs = torch.sparse.mm(H, torch.ones((n, 1), device=self.device, dtype=torch.float64)).flatten()
        return H, Hs

    @staticmethod
    def _get_Ke(nu: float = 0.3) -> np.ndarray:
        # Reusing the same Ke matrix from SIMPSolver
        # (Snippet shortened for brevity, but same logic applies)
        from .simp_solver import SIMPSolver
        return SIMPSolver._get_Ke(nu)
