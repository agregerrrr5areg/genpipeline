import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import json

import freecad_bridge
from blackwell_compat import botorch_device

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


class BridgeEvaluator:
    """Evaluates designs using the FreeCAD bridge (WSL2 -> Windows)."""
    def __init__(self, freecad_path: str = None, output_dir: str = "./optimization_results/fem"):
        try:
            self.freecad_cmd = freecad_bridge.find_freecad_cmd(freecad_path)
        except Exception as e:
            logger.error(f"Could not find FreeCAD: {e}")
            self.freecad_cmd = None
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variant_win = freecad_bridge.deploy_variant_script()
        self.evaluation_history = []

    def evaluate(self, parameters: Dict[str, float]) -> Dict[str, float]:
        if not self.freecad_cmd:
            logger.error("FreeCAD command not found. Returning dummy results.")
            return {"stress": 1e6, "compliance": 1e6, "mass": 1.0}

        h_mm = float(parameters.get("h_mm", 10.0))
        r_mm = float(parameters.get("r_mm", 3.0))
        geom = parameters.get("geometry", "cantilever")

        logger.info(f"Running FEM evaluation: {geom} h={h_mm:.2f} r={r_mm:.2f}")
        
        result = freecad_bridge.run_variant(
            self.freecad_cmd, h_mm, r_mm, str(self.output_dir), 
            self.variant_win, geometry=geom
        )

        if result:
            stress = float(result.get("stress_max", 0.0))
            compliance = float(result.get("compliance", 0.0))
            
            # If stress is 0, the simulation likely failed to produce results
            if stress < 1e-6 or compliance < 1e-9:
                logger.warning(f"Simulation returned null results for {geom}. Applying penalty.")
                return {"stress": 1e6, "compliance": 1e6, "mass": 1.0}

            # Map result keys to the expected internal format
            eval_res = {
                "stress": stress,
                "compliance": compliance,
                "mass": float(result.get("mass", 1.0)),
                "bbox": result.get("bbox")
            }
            self.evaluation_history.append({
                "parameters": parameters,
                "results": eval_res
            })
            return eval_res
        else:
            logger.error("FEM simulation failed. Returning penalty.")
            return {"stress": 1e6, "compliance": 1e6, "mass": 1.0}

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
        self.best_bbox = None

    def seed_from_dataset(self, train_loader, n_seeds=10):
        """Seeds the optimizer with the best points from the existing dataset."""
        logger.info(f"Seeding optimizer with {n_seeds} points from dataset...")
        seeds = []
        
        self.vae.eval()
        with torch.no_grad():
            for batch in train_loader:
                geom = batch['geometry'].to(self.device)
                mu, _ = self.vae.encode(geom)
                
                # Get objective values for these existing points
                for i in range(len(mu)):
                    z = mu[i].cpu().numpy()
                    obj, bbox = self.objective_function(z, real_eval=True)
                    seeds.append((z, obj, bbox))
                
                if len(seeds) >= 100: break # Don't seed too many
        
        # Sort by objective and take top N
        seeds.sort(key=lambda x: x[1])
        for i in range(min(n_seeds, len(seeds))):
            z, obj, bbox = seeds[i]
            self.x_history.append(z)
            self.y_history.append(obj)
            if obj == min(self.y_history):
                self.best_bbox = bbox
            logger.info(f"Seed point {i+1}: obj={obj:.4f}")

    def decode_latent_to_geometry(self, z: np.ndarray) -> np.ndarray:
        """Decodes latent z with an Organic Density Filter."""
        from scipy.ndimage import gaussian_filter
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(self.device)
            if z_tensor.dim() == 1:
                z_tensor = z_tensor.unsqueeze(0)
            geometry = self.vae.decode(z_tensor).cpu().numpy().squeeze()
        
        # Organic Density Filter: Smooth + Project
        # This forces 'cool' organic transitions and removes pixelation
        smooth_sigma = self.sim_cfg.get("constraints", {}).get("organic_smoothness", 0.5)
        geometry = gaussian_filter(geometry, sigma=smooth_sigma)
        
        # Heaviside Projection (sharpening the edges)
        geometry = 1.0 / (1.0 + np.exp(-15.0 * (geometry - 0.5)))
        return geometry

    def decode_to_mesh(self, z: np.ndarray, bbox: dict = None):
        """Decode latent z → voxels → triangle mesh."""
        from utils import VoxelConverter
        voxels = self.decode_latent_to_geometry(z).squeeze()  # (D, H, W)
        
        mesh_data = VoxelConverter.voxel_to_mesh(voxels, bbox=bbox)
        if mesh_data:
            return mesh_data['vertices'], mesh_data['faces']
        return np.array([]), np.array([])

    def geometry_to_parameters(self, z: np.ndarray) -> Dict[str, float]:
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(self.device)
            if z_tensor.dim() == 1:
                z_tensor = z_tensor.unsqueeze(0)
            params = self.vae.predict_parameters(z_tensor).cpu().numpy()[0]
        
        return {
            "h_mm": float(params[0]),
            "r_mm": float(params[1]),
            "geometry": self.sim_cfg.get("geometry_type", "cantilever")
        }

    def objective_function(self, z: np.ndarray, real_eval=False) -> Tuple[float, dict | None]:
        from utils import GeometryMetrics
        
        z = z.reshape(1, -1) if z.ndim == 1 else z
        cfg   = self.sim_cfg
        constraints = cfg.get("constraints", {})
        
        # 1. Decode with Organic Filter
        geometry = self.decode_latent_to_geometry(z[0])
        
        # 2. Physicality Invariants (Check before simulation)
        vol_fraction = (geometry > 0.5).mean()
        n_components = GeometryMetrics.compute_connectivity(geometry)
        
        # INVARIANT: Design must be connected and non-empty
        min_vol = constraints.get("min_volume_fraction", 0.15)
        if vol_fraction < 0.01: # Absolute minimum
            return 3e6, None
            
        vol_penalty = max(0, min_vol - vol_fraction) * 1e6
        conn_penalty = (n_components - 1) * 1e4 if n_components > 1 else 0.0

        # 3. Strength & Performance
        w_s   = cfg.get("w_stress", 1.0)
        w_c   = cfg.get("w_compliance", 0.5) 
        w_m   = cfg.get("w_mass", 0.01)
        max_s = cfg.get("max_stress_mpa", 1e9)

        if real_eval:
            params = self.geometry_to_parameters(z[0])
            results = self.fem_evaluator.evaluate(params)

            stress     = results["stress"]
            compliance = results["compliance"]
            mass       = results.get("mass", 0.0)
            bbox       = results.get("bbox")
            
            stress_limit = cfg.get("yield_mpa", max_s)
            stress_penalty = 5e5 if stress > stress_limit else 0.0

            val = (w_s * stress + w_c * compliance + w_m * mass + stress_penalty + vol_penalty + conn_penalty)
            return val, bbox

        else:
            perf_pred       = self.predictor.predict(z)
            stress_pred     = float(perf_pred[0, 0])
            compliance_pred = float(perf_pred[0, 1])
            stress_penalty  = 5e5 if stress_pred > max_s else 0.0

            val = (w_s * stress_pred + w_c * compliance_pred + stress_penalty + vol_penalty + conn_penalty)
            return val, None

    def initialize_search(self, n_init_points=5):
        if n_init_points > 0:
            logger.info(f"Initializing Bayesian optimization with {n_init_points} points...")
            for i in range(n_init_points):
                z = np.random.randn(1, self.latent_dim) * 0.5
                z = np.clip(z, -3, 3)
                obj_value, bbox = self.objective_function(z[0], real_eval=True)
                self.x_history.append(z[0])
                self.y_history.append(obj_value)
                if obj_value == min(self.y_history):
                    self.best_bbox = bbox

        self.train_x = torch.from_numpy(np.array(self.x_history)).double().to(botorch_device)
        self.train_y = torch.from_numpy(np.array(self.y_history)).double().to(botorch_device).unsqueeze(-1)

        if BOTORCH_AVAILABLE:
            self.gp_model = SingleTaskGP(self.train_x, self.train_y)
            mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
            fit_gpytorch_mll(mll)

    def optimize_step(self, n_candidates=20):
        if not BOTORCH_AVAILABLE:
            return self._random_search_step(n_candidates)

        acq_func = UpperConfidenceBound(self.gp_model, beta=0.1)
        bounds = torch.tensor([[-3.0] * self.latent_dim, [3.0] * self.latent_dim], 
                               dtype=torch.float64).to(botorch_device)

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        best_candidate = candidates[0].numpy()
        obj_value, bbox = self.objective_function(best_candidate, real_eval=True)

        self.x_history.append(best_candidate)
        self.y_history.append(obj_value)
        if obj_value == min(self.y_history):
            self.best_bbox = bbox

        self.train_x = torch.from_numpy(np.array(self.x_history)).double().to(botorch_device)
        self.train_y = torch.from_numpy(np.array(self.y_history)).double().to(botorch_device).unsqueeze(-1)

        self.gp_model = SingleTaskGP(self.train_x, self.train_y)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

        return best_candidate, obj_value

    def _random_search_step(self, n_candidates=20):
        best_obj = float('inf')
        best_z = None
        for _ in range(n_candidates):
            z = np.random.randn(self.latent_dim) * 0.5
            obj_value, bbox = self.objective_function(z, real_eval=True)
            if obj_value < best_obj:
                best_obj = obj_value
                best_z = z
                self.best_bbox = bbox
        self.x_history.append(best_z)
        self.y_history.append(best_obj)
        return best_z, best_obj

    def surrogate_gpu_sweep(self, n_samples=100000, top_k=100) -> List[np.ndarray]:
        """Uses the GPU to screen designs and ensures we find valid physical starting points."""
        logger.info(f"Performing Physicality-Guided GPU Sweep...")
        cfg = self.sim_cfg
        min_vol = cfg.get("constraints", {}).get("min_volume_fraction", 0.15)
        valid_candidates = []
        attempts = 0
        while len(valid_candidates) < top_k and attempts < 20:
            attempts += 1
            with torch.no_grad():
                z_batch = torch.randn(5000, self.latent_dim).to(self.device) * 1.5
                preds = self.vae.predict_performance(z_batch)
                w_s, w_c = cfg.get("w_stress", 1.0), cfg.get("w_compliance", 0.5)
                obj_preds = w_s * preds[:, 0] + w_c * preds[:, 1]
                _, sorted_idx = torch.sort(obj_preds)
                for idx in sorted_idx:
                    z = z_batch[idx].cpu().numpy()
                    geom = self.decode_latent_to_geometry(z)
                    vol = (geom > 0.5).mean()
                    from utils import GeometryMetrics
                    if vol >= min_vol and GeometryMetrics.compute_connectivity(geom) == 1:
                        valid_candidates.append(z)
                        if len(valid_candidates) >= top_k: break
            if len(valid_candidates) < top_k:
                logger.info(f"  Sweep attempt {attempts}: found {len(valid_candidates)}/{top_k}")
        if not valid_candidates: valid_candidates = [np.zeros(self.latent_dim)]
        return valid_candidates

    def run_optimization(self, n_iterations=900):
        """Two-stage 1k Discovery."""
        logger.info(f"Starting 1,000-Point Hybrid Discovery...")
        gpu_candidates = self.surrogate_gpu_sweep(top_k=100)
        for i, z in enumerate(gpu_candidates):
            obj, bbox = self.objective_function(z, real_eval=True)
            self.x_history.append(z)
            self.y_history.append(obj)
            if obj == min(self.y_history): self.best_bbox = bbox
            if (i+1) % 10 == 0: logger.info(f"  Stage 1: {i+1}/100 (Best: {min(self.y_history):.4f})")

        logger.info(f"Entering Stage 2: BO Refinement...")
        self.initialize_search(n_init_points=0)
        best_obj = min(self.y_history)
        best_z = self.x_history[np.argmin(self.y_history)]
        for iteration in range(n_iterations):
            z_cand, obj_cand = self.optimize_step()
            if obj_cand < best_obj:
                best_obj = obj_cand
                best_z = z_cand
                logger.info(f"New global best (Iter {iteration+1}): {best_obj:.4f}")
        return best_z, best_obj

    def save_results(self, output_dir: str = "./optimization_results"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        y_np = np.array(self.y_history)
        valid_mask = y_np < 100000.0 
        best_idx = np.where(valid_mask, y_np, np.inf).argmin() if np.any(valid_mask) else np.argmin(self.y_history)
        best_z = self.x_history[best_idx]
        best_obj = float(self.y_history[best_idx])
        results = {
            "x_history": [x.tolist() for x in self.x_history],
            "y_history": [float(y) for y in self.y_history],
            "best_x": best_z.tolist(), "best_y": best_obj, "best_bbox": self.best_bbox
        }
        with open(Path(output_dir) / "optimization_history.json", 'w') as f: json.dump(results, f, indent=2)
        self.fem_evaluator.save_history(Path(output_dir) / "fem_evaluations.json")
        try:
            import trimesh
            verts, faces = self.decode_to_mesh(best_z, bbox=self.best_bbox)
            if len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                mesh.export(str(Path(output_dir) / "best_design.stl"))
        except Exception as e: logger.error(f"STL export failed: {e}")
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="./optimization_results")
    parser.add_argument("--n-iterations", type=int, default=900)
    args = parser.parse_args()
    from vae_design_model import DesignVAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    vae = DesignVAE(input_shape=(32, 32, 32), latent_dim=args.latent_dim)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    fem_eval = BridgeEvaluator(output_dir=args.output_dir)
    optimizer = DesignOptimizer(vae, fem_eval, device=device, latent_dim=args.latent_dim)
    best_z, best_obj = optimizer.run_optimization(n_iterations=args.n_iterations)
    optimizer.save_results(args.output_dir)
