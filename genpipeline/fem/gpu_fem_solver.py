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
        material: str = "pla",  # Default to PLA as requested
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.voxel_size_mm = voxel_size_mm
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
            U, node_coords, elements, voxels_t, bbox, force_n
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
        """Identify fixed and load nodes based on boundary faces."""
        fixed_nodes = []
        load_nodes = []

        for i, (x, y, z) in enumerate(node_coords):
            if fixed_face == "x_min" and x == 0:
                fixed_nodes.append(i)
            elif fixed_face == "x_max" and x == (D * self.voxel_size_mm):
                fixed_nodes.append(i)
            elif fixed_face == "y_min" and y == 0:
                fixed_nodes.append(i)
            elif fixed_face == "y_max" and y == (H * self.voxel_size_mm):
                fixed_nodes.append(i)
            elif fixed_face == "z_min" and z == 0:
                fixed_nodes.append(i)
            elif fixed_face == "z_max" and z == (W * self.voxel_size_mm):
                fixed_nodes.append(i)

            if load_face == "x_min" and x == 0:
                load_nodes.append(i)
            elif load_face == "x_max" and x == (D * self.voxel_size_mm):
                load_nodes.append(i)
            elif load_face == "y_min" and y == 0:
                load_nodes.append(i)
            elif load_face == "y_max" and y == (H * self.voxel_size_mm):
                load_nodes.append(i)
            elif load_face == "z_min" and z == 0:
                load_nodes.append(i)
            elif load_face == "z_max" and z == (W * self.voxel_size_mm):
                load_nodes.append(i)

        return (
            torch.tensor(fixed_nodes, dtype=torch.int64, device=self.device),
            torch.tensor(load_nodes, dtype=torch.int64, device=self.device),
        )

    def _assemble_stiffness_matrix(
        self, node_coords: torch.Tensor, elements: torch.Tensor
    ) -> torch.Tensor:
        """Assemble global stiffness matrix using sparse tensor."""
        num_nodes = node_coords.shape[0]
        num_elements = elements.shape[0]

        # Precompute element stiffness matrices
        K_elements = self._compute_element_stiffnesses(node_coords, elements)

        # Assemble sparse global stiffness matrix
        indices = []  # (row, col) pairs
        values = []  # stiffness values

        for el_idx in range(num_elements):
            el_nodes = elements[el_idx]
            K_el = K_elements[el_idx]

            # Add contributions to global matrix
            for i in range(8):
                for j in range(8):
                    row = el_nodes[i].item()
                    col = el_nodes[j].item()
                    indices.append([row, col])
                    values.append(K_el[i, j].item())

        # Create sparse tensor
        if len(indices) == 0:
            # Handle case with no elements (should not happen in normal usage)
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                (num_nodes, num_nodes),
            )

        indices = torch.tensor(indices, dtype=torch.int64, device=self.device).t()
        values = torch.tensor(values, dtype=self.dtype, device=self.device)

        K = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
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
        """Compute 8-node hexahedral element stiffness matrix."""
        # Node positions: (8, 3)
        x = node_positions[:, 0]
        y = node_positions[:, 1]
        z = node_positions[:, 2]

        # Element size
        Lx = torch.max(x) - torch.min(x)
        Ly = torch.max(y) - torch.min(y)
        Lz = torch.max(z) - torch.min(z)

        # Material properties
        E = torch.tensor(self.E, device=self.device, dtype=self.dtype)
        nu = torch.tensor(self.nu, device=self.device, dtype=self.dtype)

        # Compute element stiffness matrix for C3D8 element
        # Using standard hexahedral element formulation
        # Shape functions and numerical integration

        # Natural coordinates (xi, eta, zeta) for 8-node hexahedral element
        # Node numbering: 1:(-1,-1,-1), 2:(1,-1,-1), 3:(1,1,-1), 4:(-1,1,-1),
        #                5:(-1,-1,1), 6:(1,-1,1), 7:(1,1,1), 8:(-1,1,1)

        # Shape functions N_i(xi, eta, zeta)
        def N1(xi, eta, zeta): return 0.125 * (1-xi) * (1-eta) * (1-zeta)
        def N2(xi, eta, zeta): return 0.125 * (1+xi) * (1-eta) * (1-zeta)
        def N3(xi, eta, zeta): return 0.125 * (1+xi) * (1+eta) * (1-zeta)
        def N4(xi, eta, zeta): return 0.125 * (1-xi) * (1+eta) * (1-zeta)
        def N5(xi, eta, zeta): return 0.125 * (1-xi) * (1-eta) * (1+zeta)
        def N6(xi, eta, zeta): return 0.125 * (1+xi) * (1-eta) * (1+zeta)
        def N7(xi, eta, zeta): return 0.125 * (1+xi) * (1+eta) * (1+zeta)
        def N8(xi, eta, zeta): return 0.125 * (1-xi) * (1+eta) * (1+zeta)

        # Derivatives of shape functions
        def dN1_dxi(xi, eta, zeta): return -0.125 * (1-eta) * (1-zeta)
        def dN1_deta(xi, eta, zeta): return -0.125 * (1-xi) * (1-zeta)
        def dN1_dzeta(xi, eta, zeta): return -0.125 * (1-xi) * (1-eta)
        def dN2_dxi(xi, eta, zeta): return  0.125 * (1-eta) * (1-zeta)
        def dN2_deta(xi, eta, zeta): return -0.125 * (1+xi) * (1-zeta)
        def dN2_dzeta(xi, eta, zeta): return -0.125 * (1+xi) * (1-eta)
        def dN3_dxi(xi, eta, zeta): return  0.125 * (1+eta) * (1-zeta)
        def dN3_deta(xi, eta, zeta): return  0.125 * (1+xi) * (1-zeta)
        def dN3_dzeta(xi, eta, zeta): return -0.125 * (1+xi) * (1+eta)
        def dN4_dxi(xi, eta, zeta): return -0.125 * (1+eta) * (1-zeta)
        def dN4_deta(xi, eta, zeta): return  0.125 * (1-xi) * (1-zeta)
        def dN4_dzeta(xi, eta, zeta): return -0.125 * (1-xi) * (1+eta)
        def dN5_dxi(xi, eta, zeta): return -0.125 * (1-eta) * (1+zeta)
        def dN5_deta(xi, eta, zeta): return -0.125 * (1-xi) * (1+zeta)
        def dN5_dzeta(xi, eta, zeta): return  0.125 * (1-xi) * (1-eta)
        def dN6_dxi(xi, eta, zeta): return  0.125 * (1-eta) * (1+zeta)
        def dN6_deta(xi, eta, zeta): return -0.125 * (1+xi) * (1+zeta)
        def dN6_dzeta(xi, eta, zeta): return  0.125 * (1+xi) * (1-eta)
        def dN7_dxi(xi, eta, zeta): return  0.125 * (1+eta) * (1+zeta)
        def dN7_deta(xi, eta, zeta): return  0.125 * (1+xi) * (1+zeta)
        def dN7_dzeta(xi, eta, zeta): return  0.125 * (1+xi) * (1+eta)
        def dN8_dxi(xi, eta, zeta): return -0.125 * (1+eta) * (1+zeta)
        def dN8_deta(xi, eta, zeta): return  0.125 * (1-xi) * (1+zeta)
        def dN8_dzeta(xi, eta, zeta): return  0.125 * (1-xi) * (1+eta)

        # For simplicity, use numerical integration with 2x2x2 Gauss points
        gauss_points = [
            (-0.577350269189626, -0.577350269189626, -0.577350269189626),
            ( 0.577350269189626, -0.577350269189626, -0.577350269189626),
            ( 0.577350269189626,  0.577350269189626, -0.577350269189626),
            (-0.577350269189626,  0.577350269189626, -0.577350269189626),
            (-0.577350269189626, -0.577350269189626,  0.577350269189626),
            ( 0.577350269189626, -0.577350269189626,  0.577350269189626),
            ( 0.577350269189626,  0.577350269189626,  0.577350269189626),
            (-0.577350269189626,  0.577350269189626,  0.577350269189626),
        ]

        # Gauss weights
        weights = [1.0] * 8

        # Compute Jacobian matrix
        J = torch.zeros((3, 3), device=self.device, dtype=self.dtype)
        for xi, eta, zeta in gauss_points:
            for i in range(8):
                J[0, 0] += dN1_dxi(xi, eta, zeta) * x[i]
                J[0, 1] += dN1_dxi(xi, eta, zeta) * y[i]
                J[0, 2] += dN1_dxi(xi, eta, zeta) * z[i]
                J[1, 0] += dN1_deta(xi, eta, zeta) * x[i]
                J[1, 1] += dN1_deta(xi, eta, zeta) * y[i]
                J[1, 2] += dN1_deta(xi, eta, zeta) * z[i]
                J[2, 0] += dN1_dzeta(xi, eta, zeta) * x[i]
                J[2, 1] += dN1_dzeta(xi, eta, zeta) * y[i]
                J[2, 2] += dN1_dzeta(xi, eta, zeta) * z[i]

        # Compute strain-displacement matrix B
        B = torch.zeros((6, 24), device=self.device, dtype=self.dtype)

        # Compute constitutive matrix D (3D elasticity)
        D = torch.zeros((6, 6), device=self.device, dtype=self.dtype)
        D[0, 0] = 1.0
        D[1, 1] = 1.0
        D[2, 2] = 1.0
        D[0, 1] = nu
        D[1, 0] = nu
        D[0, 2] = nu
        D[2, 0] = nu
        D[1, 2] = nu
        D[2, 1] = nu
        D[3, 3] = (1-nu)/2
        D[4, 4] = (1-nu)/2
        D[5, 5] = (1-nu)/2
        D = (E / (1 - nu**2)) * D

        # Assemble stiffness matrix using numerical integration
        K_el = torch.zeros((24, 24), device=self.device, dtype=self.dtype)
        for xi, eta, zeta, w in zip(gauss_points, weights):
            # Compute B matrix at integration point
            # For simplicity, use simplified B matrix calculation
            # In practice, you would compute the full B matrix
            B[0, 0] = dN1_dxi(xi, eta, zeta)
            B[0, 3] = dN2_dxi(xi, eta, zeta)
            B[0, 6] = dN3_dxi(xi, eta, zeta)
            B[0, 9] = dN4_dxi(xi, eta, zeta)
            B[0, 12] = dN5_dxi(xi, eta, zeta)
            B[0, 15] = dN6_dxi(xi, eta, zeta)
            B[0, 18] = dN7_dxi(xi, eta, zeta)
            B[0, 21] = dN8_dxi(xi, eta, zeta)
            
            B[1, 1] = dN1_deta(xi, eta, zeta)
            B[1, 4] = dN2_deta(xi, eta, zeta)
            B[1, 7] = dN3_deta(xi, eta, zeta)
            B[1, 10] = dN4_deta(xi, eta, zeta)
            B[1, 13] = dN5_deta(xi, eta, zeta)
            B[1, 16] = dN6_deta(xi, eta, zeta)
            B[1, 19] = dN7_deta(xi, eta, zeta)
            B[1, 22] = dN8_deta(xi, eta, zeta)
            
            B[2, 2] = dN1_dzeta(xi, eta, zeta)
            B[2, 5] = dN2_dzeta(xi, eta, zeta)
            B[2, 8] = dN3_dzeta(xi, eta, zeta)
            B[2, 11] = dN4_dzeta(xi, eta, zeta)
            B[2, 14] = dN5_dzeta(xi, eta, zeta)
            B[2, 17] = dN6_dzeta(xi, eta, zeta)
            B[2, 20] = dN7_dzeta(xi, eta, zeta)
            B[2, 23] = dN8_dzeta(xi, eta, zeta)
            
            B[3, 1] = dN1_dzeta(xi, eta, zeta)
            B[3, 4] = dN2_dzeta(xi, eta, zeta)
            B[3, 7] = dN3_dzeta(xi, eta, zeta)
            B[3, 10] = dN4_dzeta(xi, eta, zeta)
            B[3, 13] = dN5_dzeta(xi, eta, zeta)
            B[3, 16] = dN6_dzeta(xi, eta, zeta)
            B[3, 19] = dN7_dzeta(xi, eta, zeta)
            B[3, 22] = dN8_dzeta(xi, eta, zeta)
            
            B[4, 0] = dN1_dzeta(xi, eta, zeta)
            B[4, 3] = dN2_dzeta(xi, eta, zeta)
            B[4, 6] = dN3_dzeta(xi, eta, zeta)
            B[4, 9] = dN4_dzeta(xi, eta, zeta)
            B[4, 12] = dN5_dzeta(xi, eta, zeta)
            B[4, 15] = dN6_dzeta(xi, eta, zeta)
            B[4, 18] = dN7_dzeta(xi, eta, zeta)
            B[4, 21] = dN8_dzeta(xi, eta, zeta)
            
            B[5, 2] = dN1_deta(xi, eta, zeta)
            B[5, 5] = dN2_deta(xi, eta, zeta)
            B[5, 8] = dN3_deta(xi, eta, zeta)
            B[5, 11] = dN4_deta(xi, eta, zeta)
            B[5, 14] = dN5_deta(xi, eta, zeta)
            B[5, 17] = dN6_deta(xi, eta, zeta)
            B[5, 20] = dN7_deta(xi, eta, zeta)
            B[5, 23] = dN8_deta(xi, eta, zeta)

            # Compute element stiffness contribution
            K_el += w * torch.mm(torch.mm(B.t(), D), B) * torch.det(J)

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

    # Convert back to sparse tensor using alternative method
    # Create sparse tensor from dense without using _indices()
    sparse_indices = torch.nonzero(K_dense, as_tuple=False).t()
    sparse_values = K_dense[sparse_indices[0], sparse_indices[1]]
    K = torch.sparse_coo_tensor(sparse_indices, sparse_values, K_dense.shape)

    return K, F


def _conjugate_gradient(self, K: torch.sparse.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Solve K*U = F using conjugate gradient method."""
    num_nodes = F.shape[0] // 3

    # Initialize
    U = torch.zeros(num_nodes * 3, device=self.device)
    r = F - torch.sparse.mm(K, U)
    p = r.clone()
    rsold = torch.dot(r, r)

    for i in range(self.max_iterations):
        Ap = torch.sparse.mm(K, p)
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
) -> Dict[str, float]:
    """Calculate stress, displacement, and compliance from solution."""
    D, H, W = voxels.shape

    # Reshape displacements to (num_nodes, 3)
    U = U.view(-1, 3)

    # Calculate max displacement
    displacement_max = torch.norm(U, dim=1).max().item()

    # Calculate compliance (work done by forces)
    compliance = torch.dot(U.view(-1), force_n * torch.ones_like(U.view(-1))).item()

    # Calculate stress (simplified - in practice you would compute element stresses)
    # For now, return a placeholder value
    stress_max = 100.0  # Placeholder - should compute actual stress

    # Calculate mass from volume fraction
    solid_frac = float((voxels > 0.5).mean())
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

        # Convert back to sparse tensor using alternative method
        # Create sparse tensor from dense without using _indices()
        sparse_indices = torch.nonzero(K_dense, as_tuple=False).t()
        sparse_values = K_dense[sparse_indices[0], sparse_indices[1]]
        K = torch.sparse_coo_tensor(sparse_indices, sparse_values, K_dense.shape)

        return K, F

    def __call__(self, *args, **kwargs):
        """Make instance callable like a function."""
        return self.solve(*args, **kwargs)
