import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.optim import optimize_acqf
    from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    logger.warning("BoTorch not available. Install via: pip install botorch")


class FEMEvaluator:
    def __init__(self, freecad_exe_path: str, master_template: str):
        self.freecad_exe = freecad_exe_path
        self.master_template = master_template
        self.evaluation_history = []

    def evaluate(self, parameters: Dict[str, float]) -> Dict[str, float]:
        try:
            import FreeCAD
            import FreeCADGui
        except ImportError:
            logger.error("FreeCAD not available. Cannot run FEM simulation.")
            return {"stress": 0.0, "compliance": 0.0, "mass": 0.0}

        try:
            doc = FreeCAD.open(self.master_template)

            for param_name, param_value in parameters.items():
                for obj in doc.Objects:
                    if hasattr(obj, 'Name') and obj.Name == "Parameters":
                        if hasattr(obj, 'set'):
                            obj.set(f'{param_name}', param_value)

            doc.recompute()

            stress_max = self._extract_stress(doc)
            compliance = self._extract_compliance(doc)
            mass = self._extract_mass(doc)

            doc.close()

            results = {
                "stress": float(stress_max),
                "compliance": float(compliance),
                "mass": float(mass)
            }

            self.evaluation_history.append({
                "parameters": parameters,
                "results": results
            })

            return results

        except Exception as e:
            logger.error(f"FEM evaluation failed: {e}")
            return {"stress": 1e6, "compliance": 1e6, "mass": 1.0}

    def _extract_stress(self, doc) -> float:
        for obj in doc.Objects:
            if hasattr(obj, 'StressValues'):
                stress_vals = obj.StressValues
                if stress_vals:
                    return float(max(stress_vals))
        return 0.0

    def _extract_compliance(self, doc) -> float:
        for obj in doc.Objects:
            if hasattr(obj, 'DisplacementLengths'):
                displacements = obj.DisplacementLengths
                if displacements:
                    return float(np.sum(displacements))
        return 0.0

    def _extract_mass(self, doc) -> float:
        for obj in doc.Objects:
            if hasattr(obj, 'Mass'):
                return float(obj.Mass)
        return 1.0

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        logger.info(f"Evaluation history saved to {path}")


class PerformancePredictor:
    def __init__(self, vae_model, device='cuda'):
        self.vae = vae_model
        self.device = device
        self.vae.eval()

    def predict(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(self.device)
            if z_tensor.dim() == 1:
                z_tensor = z_tensor.unsqueeze(0)
            perf = self.vae.predict_performance(z_tensor)
        return perf.cpu().numpy()


class DesignOptimizer:
    def __init__(self, vae_model, fem_evaluator, device='cuda', latent_dim=16, max_iterations=100, parallel_evaluations=4, sim_cfg=None):
        self.vae = vae_model
        self.fem_evaluator = fem_evaluator
        self.device = device
        self.latent_dim = latent_dim
        self.max_iterations = max_iterations
        self.parallel_evaluations = parallel_evaluations
        self.sim_cfg = sim_cfg or {
            "w_stress": 1.0, "w_compliance": 0.1, "w_mass": 0.01,
            "max_stress_mpa": 1e9, "max_disp_mm": 1e9,
        }
        self.vae.eval()

        self.predictor = PerformancePredictor(vae_model, device)
        self.gp_model = None
        self.train_x = None
        self.train_y = None

        self.x_history = []
        self.y_history = []

    def decode_latent_to_geometry(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(self.device)
            if z_tensor.dim() == 1:
                z_tensor = z_tensor.unsqueeze(0)
            geometry = self.vae.decode(z_tensor)
        return geometry.cpu().numpy()

    def decode_to_mesh(self, z: np.ndarray):
        """Decode latent z → voxels → triangle mesh via GPU Marching Cubes.

        Returns:
            verts: (N*3, 3) float32 array of vertex positions
            faces: (N,   3) int32  array of face indices
        """
        from cuda_kernels import gpu_marching_cubes
        voxels = self.decode_latent_to_geometry(z).squeeze()  # (D, H, W)
        verts, faces = gpu_marching_cubes(voxels, isovalue=0.5)
        return verts, faces

    def geometry_to_parameters(self, geometry: np.ndarray) -> Dict[str, float]:
        voxel_grid = geometry.squeeze()

        params = {
            "thickness_mm": 2.0 + (voxel_grid.mean() * 1.0),
            "radius_mm": 5.0 + (voxel_grid.std() * 3.0),
            "feature_size_mm": 1.0 + (voxel_grid.max() * 2.0),
        }
        return params

    def objective_function(self, z: np.ndarray, real_eval=False) -> float:
        z = z.reshape(1, -1) if z.ndim == 1 else z
        cfg   = self.sim_cfg
        w_s   = cfg.get("w_stress", 1.0)
        w_c   = cfg.get("w_compliance", 0.1)
        w_m   = cfg.get("w_mass", 0.01)
        max_s = cfg.get("max_stress_mpa", 1e9)

        if real_eval:
            geometry = self.decode_latent_to_geometry(z[0])
            params = self.geometry_to_parameters(geometry)
            results = self.fem_evaluator.evaluate(params)

            stress     = results["stress"]
            compliance = results["compliance"]
            mass       = results.get("mass", 0.0)
            penalty    = 1e6 if stress > max_s else 0.0

            return w_s * stress + w_c * compliance + w_m * mass + penalty

        else:
            perf_pred       = self.predictor.predict(z)
            stress_pred     = float(perf_pred[0, 0])
            compliance_pred = float(perf_pred[0, 1])
            penalty         = 1e6 if stress_pred > max_s else 0.0

            return w_s * stress_pred + w_c * compliance_pred + penalty

    def initialize_search(self, n_init_points=5):
        logger.info(f"Initializing Bayesian optimization with {n_init_points} points...")

        for i in range(n_init_points):
            z = np.random.randn(1, self.latent_dim) * 0.5

            z = np.clip(z, -3, 3)

            obj_value = self.objective_function(z[0], real_eval=True)

            self.x_history.append(z[0])
            self.y_history.append(obj_value)

            logger.info(f"Init point {i+1}: z_norm={np.linalg.norm(z):.3f}, obj={obj_value:.4f}")

        self.train_x = torch.from_numpy(np.array(self.x_history)).float()
        self.train_y = torch.from_numpy(np.array(self.x_history)).float().unsqueeze(-1)

        if BOTORCH_AVAILABLE:
            self.gp_model = SingleTaskGP(self.train_x, self.train_y)
            mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
            fit_gpytorch_mll(mll)

    def optimize_step(self, n_candidates=20):
        logger.info("Running Bayesian optimization step...")

        if not BOTORCH_AVAILABLE:
            logger.warning("BoTorch not available. Falling back to random search.")
            return self._random_search_step(n_candidates)

        acq_func = UpperConfidenceBound(self.gp_model, beta=0.1)

        bounds = torch.tensor([[-3.0] * self.latent_dim, [3.0] * self.latent_dim], dtype=torch.float32)

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=bounds,
            q=self.parallel_evaluations,
            num_restarts=10,
            raw_samples=512,
        )

        best_candidate = candidates[0].numpy()

        logger.info(f"Candidate z_norm: {np.linalg.norm(best_candidate):.3f}")

        geometry = self.decode_latent_to_geometry(best_candidate)
        params = self.geometry_to_parameters(geometry)

        results = self.fem_evaluator.evaluate(params)
        obj_value = results["stress"] + 0.1 * results["compliance"]

        self.x_history.append(best_candidate)
        self.y_history.append(obj_value)

        logger.info(f"New point: obj={obj_value:.4f}, stress={results['stress']:.2f}, "
                   f"compliance={results['compliance']:.4f}")

        self.train_x = torch.from_numpy(np.array(self.x_history)).float()
        self.train_y = torch.from_numpy(np.array(self.y_history)).float().unsqueeze(-1)

        self.gp_model = SingleTaskGP(self.train_x, self.train_y)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

        return best_candidate, results

    def _random_search_step(self, n_candidates=20):
        best_obj = float('inf')
        best_z = None
        best_results = None

        for _ in range(n_candidates):
            z = np.random.randn(self.latent_dim) * 0.5
            z = np.clip(z, -3, 3)

            geometry = self.decode_latent_to_geometry(z)
            params = self.geometry_to_parameters(geometry)
            results = self.fem_evaluator.evaluate(params)
            obj_value = results["stress"] + 0.1 * results["compliance"]

            if obj_value < best_obj:
                best_obj = obj_value
                best_z = z
                best_results = results

        self.x_history.append(best_z)
        self.y_history.append(best_obj)

        return best_z, best_results

    def run_optimization(self, n_iterations=50):
        logger.info(f"Starting optimization for {n_iterations} iterations...")

        self.initialize_search(n_init_points=5)

        best_obj = min(self.y_history)
        best_z = self.x_history[np.argmin(self.y_history)]

        for iteration in range(n_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

            z_candidate, results = self.optimize_step()

            current_best_obj = min(self.y_history)
            if current_best_obj < best_obj:
                best_obj = current_best_obj
                best_z = self.x_history[np.argmin(self.y_history)]
                logger.info(f"New best objective: {best_obj:.4f}")

        logger.info("\n=== Optimization Complete ===")
        logger.info(f"Best objective value: {best_obj:.4f}")
        logger.info(f"Best latent vector: {best_z}")

        return best_z, best_obj

    def save_results(self, output_dir: str = "./optimization_results"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        best_idx = np.argmin(self.y_history)
        best_z   = self.x_history[best_idx]

        results = {
            "x_history": [x.tolist() for x in self.x_history],
            "y_history": [float(y) for y in self.y_history],
            "best_x": best_z.tolist(),
            "best_y": float(min(self.y_history)),
        }

        with open(Path(output_dir) / "optimization_history.json", 'w') as f:
            import json
            json.dump(results, f, indent=2)

        self.fem_evaluator.save_history(Path(output_dir) / "fem_evaluations.json")

        # Export best design as STL via GPU Marching Cubes
        try:
            import trimesh
            verts, faces = self.decode_to_mesh(best_z)
            if len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                stl_path = Path(output_dir) / "best_design.stl"
                mesh.export(str(stl_path))
                logger.info(f"Best design exported to {stl_path} ({len(faces):,} triangles)")
            else:
                logger.warning("Best design produced empty mesh — skipping STL export")
        except Exception as e:
            logger.warning(f"STL export failed: {e}")

        logger.info(f"Results saved to {output_dir}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian design optimization")
    parser.add_argument("--model-checkpoint", type=str, default="checkpoints/vae_best.pth",
                       help="Path to trained VAE checkpoint")
    parser.add_argument("--freecad-template", type=str, required=True,
                       help="Path to FreeCAD master template")
    parser.add_argument("--n-iterations", type=int, default=50,
                       help="Number of optimization iterations")
    parser.add_argument("--latent-dim", type=int, default=16,
                       help="Latent space dimension")
    parser.add_argument("--output-dir", type=str, default="./optimization_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--max-iterations", type=int, default=100,
                       help="Maximum number of iterations for optimization")
    parser.add_argument("--parallel-evaluations", type=int, default=4,
                       help="Number of parallel evaluations during optimization")

    args = parser.parse_args()

    from vae_design_model import DesignVAE

    checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
    vae = DesignVAE(input_shape=(32, 32, 32), latent_dim=args.latent_dim)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(args.device)

    fem_eval = FEMEvaluator(freecad_exe_path="/path/to/FreeCAD", 
                           master_template=args.freecad_template)

    optimizer = DesignOptimizer(vae, fem_eval, device=args.device, latent_dim=args.latent_dim,
                               max_iterations=args.max_iterations, parallel_evaluations=args.parallel_evaluations)

    best_z, best_obj = optimizer.run_optimization(n_iterations=args.n_iterations)

    optimizer.save_results(args.output_dir)

    logger.info(f"Best design found at latent vector: {best_z}")
