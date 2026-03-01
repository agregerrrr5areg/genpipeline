import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import json
import concurrent.futures

import freecad_bridge
from blackwell_compat import botorch_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.optim import optimize_acqf
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    from botorch.utils.multi_objective.pareto import is_non_dominated
    BOTORCH_AVAILABLE = True
except ImportError as e:
    BOTORCH_AVAILABLE = False
    logger.warning(f"BoTorch not available: {e}. Multi-objective discovery will fallback to random search.")


class BridgeEvaluator:
    """Evaluates designs using the FreeCAD bridge (WSL2 -> Windows)."""
    def __init__(self, freecad_path: str = None, output_dir: str = "./optimization_results/fem", n_workers: int = 4):
        try:
            self.freecad_cmd = freecad_bridge.find_freecad_cmd(freecad_path)
        except Exception as e:
            logger.error(f"Could not find FreeCAD: {e}")
            self.freecad_cmd = None
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variant_win = freecad_bridge.deploy_variant_script()
        self.evaluation_history = []
        self.n_workers = n_workers

    def evaluate_batch(self, parameter_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Evaluates multiple designs in parallel via the FreeCAD bridge."""
        if not self.freecad_cmd:
            return [{"stress": 1e6, "compliance": 1e6, "mass": 1.0}] * len(parameter_list)

        results_out = [None] * len(parameter_list)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_idx = {}
            for idx, params in enumerate(parameter_list):
                h_mm = float(params.get("h_mm", 10.0))
                rr_mm = float(params.get("r_mm", 0.0))
                geom = params.get("geometry", "cantilever")
                
                # Realistic Dimension Guard (5mm thickness)
                if geom == "cantilever" and h_mm - (2 * rr_mm) < 4.9:
                    results_out[idx] = {"stress": 1e6, "compliance": 1e6, "mass": 1.0, "parameters": params}
                    continue

                future = executor.submit(
                    freecad_bridge.run_variant,
                    self.freecad_cmd, h_mm, rr_mm, str(self.output_dir), 
                    self.variant_win, geometry=geom, material_cfg=params.get("material_cfg")
                )
                future_to_idx[future] = idx

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    if res:
                        eval_res = {
                            "stress": float(res.get("stress_max", 1e6)),
                            "compliance": float(res.get("compliance", 1e6)),
                            "mass": float(res.get("mass", 1.0)),
                            "bbox": res.get("bbox"),
                            "parameters": parameter_list[idx]
                        }
                        results_out[idx] = eval_res
                        self.evaluation_history.append(eval_res)
                    else:
                        results_out[idx] = {"stress": 1e6, "compliance": 1e6, "mass": 1.0, "parameters": parameter_list[idx]}
                except Exception as e:
                    logger.error(f"Batch evaluation failed for index {idx}: {e}")
                    results_out[idx] = {"stress": 1e6, "compliance": 1e6, "mass": 1.0, "parameters": parameter_list[idx]}

        return results_out

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)


class DesignOptimizer:
    def __init__(self, vae_model, fem_evaluator, device='cuda', latent_dim=32, sim_cfg=None):
        self.vae = vae_model
        self.fem_evaluator = fem_evaluator
        self.device = device
        self.latent_dim = latent_dim
        self.sim_cfg = sim_cfg or {"w_stress": 1.0, "w_compliance": 0.1, "w_mass": 0.01}
        self.vae.eval()

        self.gp_model = None
        self.x_history = []
        # Multi-objective history: [[stress, mass], ...]
        self.y_history = []
        self.best_bbox = None

    def geometry_to_parameters(self, z: np.ndarray) -> Dict[str, float]:
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(self.device)
            if z_tensor.dim() == 1:
                z_tensor = z_tensor.unsqueeze(0)
            # Use specific head for latent vector input
            params = self.vae.predict_parameters(z_tensor)
            params = params.cpu().numpy()[0]
        
        return {
            "h_mm": float(params[0]),
            "r_mm": float(params[1]),
            "geometry": self.sim_cfg.get("geometry_type", "cantilever")
        }

    def optimize_step_parallel(self, q=4):
        """MOBO Step: Explore the Pareto Front of Stress vs Mass for Plastic."""
        # Focus exclusively on Plastic
        with open("materials.json", "r") as f:
            self.materials_db = json.load(f)
        self.material_names = ["Plastic_ABS"] # Hardcoded to plastic
        n_mats = len(self.material_names)

        if not BOTORCH_AVAILABLE or len(self.x_history) < 10:
            # Seed with Latent points only (material is fixed)
            z_batch = np.random.randn(q, self.latent_dim) * 2.0
            return self._evaluate_latent_batch(z_batch)

        # Fit multi-output GP on CPU (Search space is back to latent_dim only)
        train_X = torch.tensor(np.array(self.x_history), dtype=torch.float64).to(botorch_device)
        train_Y = torch.tensor(np.array(self.y_history), dtype=torch.float64).to(botorch_device)
        
        train_Y_std = (train_Y - train_Y.mean(dim=0)) / (train_Y.std(dim=0) + 1e-6)
        train_Y_std = -train_Y_std

        self.gp_model = SingleTaskGP(train_X, train_Y_std)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

        ref_point = train_Y_std.min(dim=0).values - 0.1
        acq = qExpectedHypervolumeImprovement(
            model=self.gp_model,
            ref_point=ref_point,
        )
        
        # Bounds: Latent (-3 to 3) only
        bounds = torch.tensor([[-3.0] * self.latent_dim, 
                                [3.0] * self.latent_dim], 
                               dtype=torch.float64).to(botorch_device)

        candidates, _ = optimize_acqf(
            acq, bounds=bounds, q=q, num_restarts=10, raw_samples=512,
        )

        return self._evaluate_latent_batch(candidates.cpu().numpy())

    def _evaluate_latent_batch(self, z_batch):
        # z_batch contains [z_0...z_31]
        param_list = [self.geometry_to_parameters(z) for z in z_batch]
        
        for i, p in enumerate(param_list):
            # Clamp geometry
            p["h_mm"] = max(p["h_mm"], 5.0)
            r_max = (p["h_mm"] - 5.0) / 2.0
            p["r_mm"] = max(0.0, min(p["r_mm"], r_max))
            
            # Use Plastic exclusively
            mat_name = "Plastic_ABS"
            p["material_name"] = mat_name
            p["material_cfg"] = self.materials_db[mat_name]

        results = self.fem_evaluator.evaluate_batch(param_list)
        
        for i, res in enumerate(results):
            self.x_history.append(z_batch[i])
            self.y_history.append([res["stress"], res["mass"]])
            
            # Incorporate Targets and Weights
            stress_limit = self.sim_cfg.get("max_stress_mpa", 1e9)
            penalty = 5e5 if res["stress"] > stress_limit else 0.0
            
            obj = (self.sim_cfg["w_stress"] * res["stress"] + 
                   self.sim_cfg["w_mass"] * res["mass"] + 
                   penalty)
            
            if obj == min([ (self.sim_cfg["w_stress"] * y[0] + self.sim_cfg["w_mass"] * y[1] + (5e5 if y[0] > stress_limit else 0.0)) for y in self.y_history]):
                self.best_bbox = res.get("bbox")
                logger.info(f"  New Safe Best: Stress={res['stress']:.1f}MPa (Target: {stress_limit}), Mass={res['mass']:.3f}kg")

        return z_batch, results

    def run_optimization(self, n_iterations=250, q=4):
        logger.info(f"Starting Multi-Objective Parallel Discovery (q={q})...")
        for i in range(n_iterations):
            self.optimize_step_parallel(q=q)
            logger.info(f"Round {i+1}/{n_iterations} complete. Pareto Designs Found: {is_non_dominated(-torch.tensor(self.y_history)).sum()}")
        
        # Find final Pareto Front
        y_tensor = torch.tensor(self.y_history)
        pareto_mask = is_non_dominated(-y_tensor) # negate because we minimize
        best_z_idx = np.argmin([ (self.sim_cfg["w_stress"] * y[0] + self.sim_cfg["w_mass"] * y[1]) for y in self.y_history])
        
        return self.x_history[best_z_idx], self.y_history[best_z_idx]

    def save_results(self, output_dir: str = "./optimization_results"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        y_tensor = torch.tensor(self.y_history)
        pareto_mask = is_non_dominated(-y_tensor)
        
        pareto_indices = torch.where(pareto_mask)[0].tolist()
        pareto_front = [
            {
                "stress": self.y_history[idx][0],
                "mass": self.y_history[idx][1],
                "params": self.fem_evaluator.evaluation_history[idx]["parameters"] if idx < len(self.fem_evaluator.evaluation_history) else {},
                "latent_z": self.x_history[idx].tolist()
            }
            for idx in pareto_indices
        ]

        results = {
            "pareto_front": pareto_front,
            "history_y": [list(y) for y in self.y_history],
            "best_bbox": self.best_bbox
        }
        with open(out_path / "optimization_history.json", 'w') as f:
            json.dump(results, f, indent=2)
        self.fem_evaluator.save_history(out_path / "fem_evaluations.json")
        logger.info(f"MOBO Results Saved. {len(pareto_front)} Pareto-optimal designs identified.")


if __name__ == "__main__":
    import argparse
    from vae_design_model import DesignVAE
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--q", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).to(device)
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = BridgeEvaluator(n_workers=args.q)
    optimizer = DesignOptimizer(vae, evaluator, latent_dim=32)
    
    optimizer.run_optimization(n_iterations=args.n_iter, q=args.q)
    optimizer.save_results()
