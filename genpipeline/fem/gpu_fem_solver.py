import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GPUFEMSolver:
    """GPU-accelerated FEM solver using SIMP method"""

    def __init__(
        self,
        voxel_resolution: int = 64,
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
    ):
        self.resolution = voxel_resolution
        self.E = young_modulus
        self.nu = poisson_ratio
        self.penalization = 3.0
        self.filter_radius = 1.5

        # Precompute stiffness matrix constants
        self._precompute_constants()

    def _precompute_constants(self):
        """Precompute constants for stiffness matrix assembly"""
        self.k = torch.zeros((24, 24), dtype=torch.float32)

        # SIMP stiffness matrix assembly constants
        for i in range(24):
            for j in range(24):
                if i == j:
                    self.k[i, j] = 2.0
                elif abs(i - j) == 1 and (i // 3) == (j // 3):
                    self.k[i, j] = -1.0
                elif abs(i - j) == 3:
                    self.k[i, j] = -1.0

        self.k *= self.E / (1 - self.nu**2)

    def _get_element_indices(self, x: int, y: int, z: int) -> torch.Tensor:
        """Get global indices for 8-node hexahedral element"""
        idx = torch.tensor(
            [
                x + y * self.resolution + z * self.resolution**2,
                (x + 1) + y * self.resolution + z * self.resolution**2,
                (x + 1) + (y + 1) * self.resolution + z * self.resolution**2,
                x + (y + 1) * self.resolution + z * self.resolution**2,
                x + y * self.resolution + (z + 1) * self.resolution**2,
                (x + 1) + y * self.resolution + (z + 1) * self.resolution**2,
                (x + 1) + (y + 1) * self.resolution + (z + 1) * self.resolution**2,
                x + (y + 1) * self.resolution + (z + 1) * self.resolution**2,
            ],
            dtype=torch.long,
        )
        return idx

    def _assemble_stiffness_matrix(self, density: torch.Tensor) -> torch.Tensor:
        """Assemble global stiffness matrix using SIMP method"""
        n = self.resolution**3
        K = torch.zeros((n, n), dtype=torch.float32, device=density.device)

        for z in range(self.resolution - 1):
            for y in range(self.resolution - 1):
                for x in range(self.resolution - 1):
                    # Get element density
                    element_density = density[x : x + 2, y : y + 2, z : z + 2].mean()

                    # Apply SIMP penalization
                    element_density = element_density**self.penalization

                    # Get element indices
                    idx = self._get_element_indices(x, y, z)

                    # Assemble element stiffness matrix
                    Ke = element_density * self.k

                    # Add to global matrix (sparse assembly)
                    for i in range(8):
                        for j in range(8):
                            K[idx[i], idx[j]] += Ke[
                                i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3
                            ].sum()

        return K

    def _apply_boundary_conditions(
        self, K: torch.Tensor, fixed_nodes: torch.Tensor, forces: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply boundary conditions and solve linear system"""
        free_dofs = torch.ones(K.shape[0], dtype=torch.bool, device=K.device)
        free_dofs[fixed_nodes] = False

        # Extract free DOFs
        K_free = K[free_dofs][:, free_dofs]
        f_free = forces[free_dofs]

        # Solve linear system (GPU-accelerated)
        with torch.no_grad():
            u_free = torch.linalg.solve(K_free, f_free)

        # Initialize displacement vector
        u = torch.zeros(K.shape[0], device=K.device)
        u[free_dofs] = u_free

        return u, free_dofs

    def solve_fem(
        self, density: torch.Tensor, fixed_nodes: torch.Tensor, forces: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Solve FEM problem using GPU-accelerated SIMP method"""

        # Ensure input is on GPU
        if not density.is_cuda:
            density = density.cuda()
            fixed_nodes = fixed_nodes.cuda()
            forces = forces.cuda()

        # Assemble stiffness matrix
        K = self._assemble_stiffness_matrix(density)

        # Apply boundary conditions and solve
        u, free_dofs = self._apply_boundary_conditions(K, fixed_nodes, forces)

        # Calculate strain and stress
        strain = self._calculate_strain(K, u, free_dofs)
        stress = self._calculate_stress(strain, density)

        # Calculate compliance (objective function)
        compliance = self._calculate_compliance(K, u, forces)

        return {
            "displacement": u,
            "strain": strain,
            "stress": stress,
            "compliance": compliance,
            "density": density,
        }

    def _calculate_strain(
        self, K: torch.Tensor, u: torch.Tensor, free_dofs: torch.Tensor
    ) -> torch.Tensor:
        """Calculate strain from displacement"""
        n = self.resolution**3
        strain = torch.zeros((n, 6), device=K.device)  # 6 strain components

        for z in range(self.resolution - 1):
            for y in range(self.resolution - 1):
                for x in range(self.resolution - 1):
                    idx = self._get_element_indices(x, y, z)

                    # Calculate strain for this element
                    u_element = u[idx].reshape(-1, 3)

                    # Strain calculation (simplified)
                    strain_element = torch.zeros(6, device=K.device)

                    # Normal strains
                    strain_element[0] = (
                        u_element[1, 0] - u_element[0, 0]
                    ) / 1.0  # epsilon_x
                    strain_element[1] = (
                        u_element[3, 1] - u_element[0, 1]
                    ) / 1.0  # epsilon_y
                    strain_element[2] = (
                        u_element[4, 2] - u_element[0, 2]
                    ) / 1.0  # epsilon_z

                    # Shear strains
                    strain_element[3] = (
                        0.5
                        * (
                            (u_element[3, 0] - u_element[0, 0])
                            + (u_element[1, 1] - u_element[0, 1])
                        )
                        / 1.0
                    )  # gamma_xy
                    strain_element[4] = (
                        0.5
                        * (
                            (u_element[4, 0] - u_element[0, 0])
                            + (u_element[1, 2] - u_element[0, 2])
                        )
                        / 1.0
                    )  # gamma_xz
                    strain_element[5] = (
                        0.5
                        * (
                            (u_element[4, 1] - u_element[0, 1])
                            + (u_element[3, 2] - u_element[0, 2])
                        )
                        / 1.0
                    )  # gamma_yz

                    # Assign to global strain array
                    for i in range(8):
                        strain[idx[i]] = strain_element

        return strain

    def _calculate_stress(
        self, strain: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        """Calculate stress from strain using Hooke's law"""
        stress = torch.zeros_like(strain)

        # Plane stress constitutive matrix
        C = (
            torch.tensor(
                [[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]],
                dtype=torch.float32,
                device=strain.device,
            )
            * self.E
            / (1 - self.nu**2)
        )

        for i in range(strain.shape[0]):
            # Apply SIMP density
            elem_density = density.flatten()[i] ** self.penalization

            # Calculate stress (3D Hooke's law)
            stress[i, :3] = C @ strain[i, :3] * elem_density
            stress[i, 3:] = C[2, 2] * strain[i, 3:] * elem_density

        return stress

    def _calculate_compliance(
        self, K: torch.Tensor, u: torch.Tensor, forces: torch.Tensor
    ) -> torch.Tensor:
        """Calculate structural compliance"""
        compliance = forces @ u
        return compliance

    def optimize_topology(
        self, target_volume_fraction: float = 0.3, max_iterations: int = 100
    ) -> torch.Tensor:
        """Perform topology optimization using SIMP method"""

        # Initialize uniform density
        density = (
            torch.ones(
                (self.resolution, self.resolution, self.resolution),
                dtype=torch.float32,
                device="cuda",
            )
            * target_volume_fraction
        )

        # Define boundary conditions (simple cantilever beam)
        fixed_nodes = torch.zeros(self.resolution**3, dtype=torch.bool, device="cuda")
        forces = torch.zeros(self.resolution**3, device="cuda")

        # Fix left side
        for y in range(self.resolution):
            for z in range(self.resolution):
                idx = 0 + y * self.resolution + z * self.resolution**2
                fixed_nodes[idx] = True

        # Apply force on right side
        forces[-1] = 1.0  # Unit force at top-right corner

        # Optimization loop
        for iteration in range(max_iterations):
            # Solve FEM problem
            results = self.solve_fem(density, fixed_nodes, forces)

            # Calculate sensitivity
            sensitivity = self._calculate_sensitivity(
                results["stress"], results["density"]
            )

            # Update density (OC method)
            density = self._update_density(density, sensitivity, target_volume_fraction)

            # Apply density filter (optional)
            density = self._apply_density_filter(density)

            # Print progress
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: Compliance = {results['compliance'].item():.4f}"
                )

        return density

    def _calculate_sensitivity(
        self, stress: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        """Calculate sensitivity of compliance to density changes"""
        # Sensitivity = -stress * density^(penalization-1)
        sensitivity = -torch.einsum("ij,ij->i", stress[:, :3], stress[:, :3]) * (
            density.flatten() ** (self.penalization - 1)
        )

        return sensitivity.reshape(self.resolution, self.resolution, self.resolution)

    def _update_density(
        self,
        density: torch.Tensor,
        sensitivity: torch.Tensor,
        target_volume_fraction: float,
    ) -> torch.Tensor:
        """Update density using Optimality Criteria method"""
        # OC method parameters
        move_limit = 0.2
        eta = 0.5

        # Find Lagrange multiplier
        l1, l2 = 0.0, 1e6
        for _ in range(10):
            lmid = 0.5 * (l1 + l2)
            new_density = density * torch.sqrt(-sensitivity / lmid)
            new_density = torch.clamp(
                new_density, density - move_limit, density + move_limit
            )
            new_density = torch.clamp(new_density, 0.001, 1.0)

            # Check volume constraint
            volume_fraction = new_density.mean()
            if volume_fraction > target_volume_fraction:
                l1 = lmid
            else:
                l2 = lmid

        return new_density

    def _apply_density_filter(self, density: torch.Tensor) -> torch.Tensor:
        """Apply density filter to remove checkerboarding"""
        filtered_density = torch.zeros_like(density)

        # Simple averaging filter
        for z in range(self.resolution):
            for y in range(self.resolution):
                for x in range(self.resolution):
                    # Get neighborhood
                    x0 = max(0, x - 1)
                    x1 = min(self.resolution, x + 2)
                    y0 = max(0, y - 1)
                    y1 = min(self.resolution, y + 2)
                    z0 = max(0, z - 1)
                    z1 = min(self.resolution, z + 2)

                    # Calculate weighted average
                    neighborhood = density[x0:x1, y0:y1, z0:z1]
                    filtered_density[x, y, z] = neighborhood.mean()

        return filtered_density

    def export_voxel_grid(self, density: torch.Tensor, filename: str = "optimized.vtk"):
        """Export voxel grid to VTK format for visualization"""
        with open(filename, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("GPU FEM Solver Result\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(
                f"DIMENSIONS {self.resolution} {self.resolution} {self.resolution}\n"
            )
            f.write("SPACING 1.0 1.0 1.0\n")
            f.write("ORIGIN 0 0 0\n")
            f.write(f"POINT_DATA {self.resolution**3}\n")
            f.write("SCALARS density float 1\n")
            f.write("LOOKUP_TABLE default\n")

            # Write density values
            for z in range(self.resolution):
                for y in range(self.resolution):
                    for x in range(self.resolution):
                        f.write(f"{density[x, y, z].item():.4f} \n")


# Global instance for pipeline integration
gpu_fem_solver = GPUFEMSolver(voxel_resolution=64)
