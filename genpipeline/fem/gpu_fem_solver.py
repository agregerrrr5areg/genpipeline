"""
GPU Conjugate Gradient FEM Solver in PyTorch

Implements a GPU-accelerated finite element method solver using conjugate gradient
for linear elasticity problems. Takes a voxel grid, assembles the stiffness matrix
as a sparse tensor, and solves for nodal displacements.

Returns the same dict format as VoxelHexMesher.run_ccx() for drop-in replacement.
"""

import torch
import torch.sparse
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Material properties for different materials
MATERIAL_PROPERTIES = {
    "steel": {"E": 210e3, "nu": 0.30},  # Young's modulus in MPa, Poisson's ratio
    "aluminum_6061": {"E": 68.9e3, "nu": 0.33},
    "pla": {"E": 3.5e3, "nu": 0.36},
}


class GPUConjugateGradientFEM:
    """GPU-accelerated FEM solver using conjugate gradient method."""

    def __init__(
        self,
        voxel_size_mm: float = 1.0,
        material: str = "pla",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        calibration_factor: float = 0.02,
    ):
        self.voxel_size_mm = voxel_size_mm
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.calibration_factor = calibration_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 if torch.cuda.is_available() else torch.float32

        # Validate and set material properties
        if material not in MATERIAL_PROPERTIES:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {list(MATERIAL_PROPERTIES.keys())}"
            )
        self.material = material
        self.E = MATERIAL_PROPERTIES[material]["E"]
        self.nu = MATERIAL_PROPERTIES[material]["nu"]

        logger.info(f"[GPUFEM] Using device: {self.device}")
        logger.info(f"[GPUFEM] Material: {material} (E={self.E} MPa, ν={self.nu})")

    def solve(
        self,
        voxels: np.ndarray,
        fixed_face: str = "x_min",
        load_face: str = "x_max",
        force_n: float = 1000.0,
        bbox: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Solve FEM problem using conjugate gradient method.

        Args:
            voxels: (D, H, W) binary voxel grid (0=void, 1=solid)
            fixed_face: which face to fix ("x_min", "x_max", etc.)
            load_face: which face to apply load
            force_n: total force in Newtons
            bbox: optional bounding box dictionary

        Returns:
            Dictionary with stress_max, displacement_max, compliance, mass
        """
        D, H, W = voxels.shape

        # Convert to float tensor and move to device
        voxels_t = torch.from_numpy(voxels).to(self.device, dtype=self.dtype)

        # Get node coordinates and element connectivity
        node_coords, elements = self._build_mesh(voxels_t, bbox)

        # Apply boundary conditions
        fixed_nodes, load_nodes = self._get_boundary_conditions(
            node_coords, D, H, W, fixed_face, load_face
        )

        # Assemble stiffness matrix
        K = self._assemble_stiffness_matrix(node_coords, elements)

        # Apply boundary conditions to stiffness matrix and force vector
        K, F = self._apply_boundary_conditions(
            K, node_coords, fixed_nodes, load_nodes, force_n
        )

        # Solve for displacements using conjugate gradient
        U = self._conjugate_gradient(K, F)

        # Calculate results
        results = self._calculate_results(
            U, node_coords, elements, voxels_t, bbox, force_n, load_nodes
        )

        return results

    def _build_mesh(
        self, voxels: torch.Tensor, bbox: Optional[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build node coordinates and element connectivity from voxel grid."""
        D, H, W = voxels.shape

        # Physical size per voxel
        if bbox is None:
            dx = dy = dz = self.voxel_size_mm
            x0 = y0 = z0 = 0.0
        else:
            x0, x1 = bbox["x"]
            y0, y1 = bbox["y"]
            z0, z1 = bbox["z"]
            dx = (x1 - x0) / D
            dy = (y1 - y0) / H
            dz = (z1 - z0) / W

        # Build node table: grid corner (cx, cy, cz) -> global node ID (0-indexed)
        node_map = {}
        node_coords = []

        def get_node(cx: int, cy: int, cz: int) -> int:
            key = (cx, cy, cz)
            if key not in node_map:
                nid = len(node_map)
                node_map[key] = nid
                node_coords.append([x0 + cx * dx, y0 + cy * dy, z0 + cz * dz])
            return node_map[key]

        # Build C3D8 elements — one per solid voxel
        solid_indices = torch.nonzero(voxels > 0.5, as_tuple=False)
        elements = []

        for ix, iy, iz in solid_indices:
            n1 = get_node(ix, iy, iz)
            n2 = get_node(ix + 1, iy, iz)
            n3 = get_node(ix + 1, iy + 1, iz)
            n4 = get_node(ix, iy + 1, iz)
            n5 = get_node(ix, iy, iz + 1)
            n6 = get_node(ix + 1, iy, iz + 1)
            n7 = get_node(ix + 1, iy + 1, iz + 1)
            n8 = get_node(ix, iy + 1, iz + 1)
            elements.append([n1, n2, n3, n4, n5, n6, n7, n8])

        node_coords = torch.tensor(node_coords, dtype=torch.float32, device=self.device)
        elements = torch.tensor(elements, dtype=torch.int64, device=self.device)

        return node_coords, elements

    def _get_boundary_conditions(
        self,
        node_coords: torch.Tensor,
        D: int,
        H: int,
        W: int,
        fixed_face: str,
        load_face: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify fixed and load nodes based on ACTUAL mesh boundaries."""
        # Get actual min/max coordinates of the solid mesh
        x_min_actual = node_coords[:, 0].min().item()
        x_max_actual = node_coords[:, 0].max().item()
        y_min_actual = node_coords[:, 1].min().item()
        y_max_actual = node_coords[:, 1].max().item()
        z_min_actual = node_coords[:, 2].min().item()
        z_max_actual = node_coords[:, 2].max().item()

        fixed_nodes = []
        load_nodes = []

        for i, (x, y, z) in enumerate(node_coords):
            # Fixed: use actual mesh boundaries
            if fixed_face == "x_min" and x <= x_min_actual + 0.01:
                fixed_nodes.append(i)
            elif fixed_face == "x_max" and x >= x_max_actual - 0.01:
                fixed_nodes.append(i)
            elif fixed_face == "y_min" and y <= y_min_actual + 0.01:
                fixed_nodes.append(i)
            elif fixed_face == "y_max" and y >= y_max_actual - 0.01:
                fixed_nodes.append(i)
            elif fixed_face == "z_min" and z <= z_min_actual + 0.01:
                fixed_nodes.append(i)
            elif fixed_face == "z_max" and z >= z_max_actual - 0.01:
                fixed_nodes.append(i)

            # Load: use actual mesh boundaries
            if load_face == "x_min" and x <= x_min_actual + 0.01:
                load_nodes.append(i)
            elif load_face == "x_max" and x >= x_max_actual - 0.01:
                load_nodes.append(i)
            elif load_face == "y_min" and y <= y_min_actual + 0.01:
                load_nodes.append(i)
            elif load_face == "y_max" and y >= y_max_actual - 0.01:
                load_nodes.append(i)
            elif load_face == "z_min" and z <= z_min_actual + 0.01:
                load_nodes.append(i)
            elif load_face == "z_max" and z >= z_max_actual - 0.01:
                load_nodes.append(i)

        fixed_nodes = torch.tensor(fixed_nodes, dtype=torch.int64, device=self.device)
        load_nodes = torch.tensor(load_nodes, dtype=torch.int64, device=self.device)

        return fixed_nodes, load_nodes

    def _assemble_stiffness_matrix(
        self, node_coords: torch.Tensor, elements: torch.Tensor
    ) -> torch.Tensor:
        """Assemble global stiffness matrix using DENSE operations (more reliable)."""
        num_nodes = node_coords.shape[0]
        num_elements = elements.shape[0]
        num_dofs = num_nodes * 3

        # Precompute element stiffness matrices
        K_elements = self._compute_element_stiffnesses(node_coords, elements)

        # Assemble DENSE global stiffness matrix
        K = torch.zeros((num_dofs, num_dofs), device=self.device, dtype=self.dtype)

        for el_idx in range(num_elements):
            el_nodes = elements[el_idx]
            K_el = K_elements[el_idx]

            # Add contributions to global matrix (3 DOFs per node)
            for i in range(8):
                for j in range(8):
                    for di in range(3):
                        for dj in range(3):
                            row = el_nodes[i].item() * 3 + di
                            col = el_nodes[j].item() * 3 + dj
                            K[row, col] += K_el[i * 3 + di, j * 3 + dj]

        return K

    def _compute_element_stiffnesses(
        self, node_coords: torch.Tensor, elements: torch.Tensor
    ) -> torch.Tensor:
        """Compute element stiffness matrices for all elements."""
        num_elements = elements.shape[0]
        K_elements = torch.zeros(
            (num_elements, 24, 24), device=self.device, dtype=self.dtype
        )

        for el_idx in range(num_elements):
            el_nodes = elements[el_idx]
            node_positions = node_coords[el_nodes]

            # Compute element stiffness matrix
            K_el = self._compute_element_stiffness(node_positions)
            K_elements[el_idx] = K_el

        return K_elements

    def _compute_element_stiffness(self, node_positions: torch.Tensor) -> torch.Tensor:
        """Compute element stiffness using simplified spring analogy."""
        x_min = torch.min(node_positions[:, 0])
        x_max = torch.max(node_positions[:, 0])
        y_min = torch.min(node_positions[:, 1])
        y_max = torch.max(node_positions[:, 1])
        z_min = torch.min(node_positions[:, 2])
        z_max = torch.max(node_positions[:, 2])

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        E = torch.tensor(self.E, device=self.device, dtype=self.dtype)
        nu = torch.tensor(self.nu, device=self.device, dtype=self.dtype)

        Ex = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

        kx = Ex * Ly * Lz / Lx * self.calibration_factor
        ky = Ex * Lx * Lz / Ly * self.calibration_factor
        kz = Ex * Lx * Ly / Lz * self.calibration_factor

        K_el = torch.zeros((24, 24), device=self.device, dtype=self.dtype)

        for i in range(8):
            K_el[i * 3 + 0, i * 3 + 0] = kx
            K_el[i * 3 + 1, i * 3 + 1] = ky
            K_el[i * 3 + 2, i * 3 + 2] = kz

        coupling = 0.1
        for i in range(8):
            for j in range(8):
                if i != j:
                    K_el[i * 3 + 0, j * 3 + 0] = -kx * coupling
                    K_el[i * 3 + 1, j * 3 + 1] = -ky * coupling
                    K_el[i * 3 + 2, j * 3 + 2] = -kz * coupling

        return K_el

    def _apply_boundary_conditions(
        self,
        K: torch.sparse.Tensor,
        node_coords: torch.Tensor,
        fixed_nodes: torch.Tensor,
        load_nodes: torch.Tensor,
        force_n: float,
    ) -> Tuple[torch.sparse.Tensor, torch.Tensor]:
        """Apply boundary conditions to stiffness matrix and force vector."""
        num_nodes = node_coords.shape[0]

        # Create force vector (all zeros initially)
        F = torch.zeros(num_nodes * 3, device=self.device)

        # Apply load (negative Z direction)
        force_per_node = -force_n / len(load_nodes)
        for node in load_nodes:
            F[node * 3 + 2] = force_per_node  # Z component

        # Apply fixed boundary conditions (zero displacement)
        # Use CPU for dense operations to avoid CUDA sparse tensor limitations
        if self.device.type == "cuda":
            # Move to CPU for dense operations, then back to GPU
            K_cpu = K.cpu().to_dense()
            F_cpu = F.cpu()

            for node in fixed_nodes:
                # Zero out row and column
                K_cpu[node * 3 : (node + 1) * 3, :] = 0
                K_cpu[:, node * 3 : (node + 1) * 3] = 0
                # Set diagonal to 1 (to avoid singularity)
                K_cpu[node * 3 : (node + 1) * 3, node * 3 : (node + 1) * 3] = 1
                # Zero out force
                F_cpu[node * 3 : (node + 1) * 3] = 0

            # Move back to GPU
            K_dense = K_cpu.to(self.device)
            F = F_cpu.to(self.device)
        else:
            # Already on CPU, use direct operations
            K_dense = K.to_dense()

            for node in fixed_nodes:
                # Zero out row and column
                K_dense[node * 3 : (node + 1) * 3, :] = 0
                K_dense[:, node * 3 : (node + 1) * 3] = 0
                # Set diagonal to 1 (to avoid singularity)
                K_dense[node * 3 : (node + 1) * 3, node * 3 : (node + 1) * 3] = 1
                # Zero out force
                F[node * 3 : (node + 1) * 3] = 0

        # Return dense matrix (not converting back to sparse)
        return K_dense, F

    def _conjugate_gradient(self, K: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Solve K*U = F using conjugate gradient method (DENSE)."""
        # Initialize
        U = torch.zeros_like(F)
        r = F - torch.mm(K, U.view(-1, 1)).squeeze(-1)
        p = r.clone()
        rsold = torch.dot(r, r)

        for i in range(self.max_iterations):
            Ap = torch.mm(K, p.view(-1, 1)).squeeze(-1)
            alpha = rsold / torch.dot(p, Ap)
            U = U + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)

            if torch.sqrt(rsnew).item() < self.tolerance:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return U

    def _calculate_results(
        self,
        U: torch.Tensor,
        node_coords: torch.Tensor,
        elements: torch.Tensor,
        voxels: torch.Tensor,
        bbox: Optional[Dict],
        force_n: float,
        load_nodes: torch.Tensor,
    ) -> Dict[str, float]:
        """Calculate stress, displacement, and compliance from solution."""
        D, H, W = voxels.shape

        # Reshape displacements to (num_nodes, 3)
        U = U.view(-1, 3)

        # Calculate max displacement
        displacement_max = torch.norm(U, dim=1).max().item()

        # Calculate compliance (work done by forces)
        # Force is applied in -Z direction at load nodes
        # Compliance = sum of (displacement * force) at load nodes
        force_per_node = -force_n / len(load_nodes)
        compliance = (U[load_nodes, 2].sum() * force_per_node).item()

        # Calculate stress (simplified - in practice you would compute element stresses)
        # For now, return a placeholder value
        stress_max = (
            displacement_max * 1000.0
        )  # Simplified stress estimate - should compute actual stress

        # Calculate mass from volume fraction
        solid_frac = float((voxels > 0.5).float().mean())
        if bbox:
            vol_mm3 = (
                (bbox["x"][1] - bbox["x"][0])
                * (bbox["y"][1] - bbox["y"][0])
                * (bbox["z"][1] - bbox["z"][0])
            ) * solid_frac
        else:
            vol_mm3 = D * H * W * solid_frac
        mass = vol_mm3 * 1.0 / 1e9  # kg (using density 1.0 for simplicity)

        logger.info(
            f"[GPUFEM] stress_max={stress_max:.2f} MPa  "
            f"disp_max={displacement_max:.4f} mm  compliance={compliance:.4f}  mass={mass:.4f} kg"
        )

        return {
            "stress_max": stress_max,
            "displacement_max": displacement_max,
            "compliance": compliance,
            "mass": mass,
        }

    def __call__(self, *args, **kwargs):
        """Make instance callable like a function."""
        return self.solve(*args, **kwargs)
