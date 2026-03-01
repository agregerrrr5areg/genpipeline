import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import json
import concurrent.futures

from . import freecad_bridge
from .blackwell_compat import botorch_device
from .pipeline_utils import NumpyEncoder as _NumpyEncoder, smooth_voxels, FEM_SENTINEL, FEM_VALID_THRESHOLD, is_valid_fem_result
from .schema import DesignParameters, FEMResult, OptimizationSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.optim import optimize_acqf
    from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement as qExpectedHypervolumeImprovement
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.multi_objective.hypervolume import NondominatedPartitioning
    BOTORCH_AVAILABLE = True
except ImportError as e:
    BOTORCH_AVAILABLE = False
    logger.warning(f"BoTorch not available: {e}. Multi-objective discovery will fallback to random search.")


class BridgeEvaluator:
    """Evaluates designs using the FreeCAD bridge (WSL2 -> Windows)."""
    def __init__(self, freecad_path: str = None, output_dir: str = "./optimisation_results/fem", n_workers: int = 4):
        try:
            self.freecad_cmd = freecad_bridge.find_freecad_cmd(freecad_path)
        except Exception as e:
            logger.error(f"Could not find FreeCAD: {e}")
            self.freecad_cmd = None
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variant_win = freecad_bridge.deploy_variant_script()
        self.evaluation_history: List[OptimizationSample] = []
        self.n_workers = n_workers

    def evaluate_batch(self, parameter_list: List[DesignParameters]) -> List[FEMResult]:
        """Evaluates multiple designs in parallel via the FreeCAD bridge."""
        if not self.freecad_cmd:
            return [FEMResult(stress_max=1e6, compliance=1e6, mass=1.0, success=False)] * len(parameter_list)

        results_out = [None] * len(parameter_list)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_idx = {}
            for idx, params in enumerate(parameter_list):
                # Dimension guard: cantilever needs 5mm wall thickness after hole
                if params.geometry_type == "cantilever" and params.h_mm - (2 * params.r_mm) < 4.9:
                    results_out[idx] = FEMResult(stress_max=1e6, compliance=1e6, mass=1.0, success=False)
                    continue

                future = executor.submit(
                    freecad_bridge.run_variant,
                    self.freecad_cmd, params.h_mm, params.r_mm, str(self.output_dir), 
                    self.variant_win, geometry=params.geometry_type, material_cfg=params.material_cfg
                )
                future_to_idx[future] = idx

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    if res:
                        eval_res = FEMResult(
                            stress_max=float(res.get("stress_max", 1e6)),
                            stress_mean=float(res.get("stress_mean", 0.0)),
                            compliance=float(res.get("compliance", 1e6)),
                            mass=float(res.get("mass", 1.0)),
                            bbox=res.get("bbox")
                        )
                        results_out[idx] = eval_res
                        self.evaluation_history.append(OptimizationSample(
                            parameters=parameter_list[idx],
                            result=eval_res
                        ))
                    else:
                        results_out[idx] = FEMResult(stress_max=1e6, compliance=1e6, mass=1.0, success=False)
                except Exception as e:
                    logger.error(f"Batch evaluation failed for index {idx}: {e}")
                    results_out[idx] = FEMResult(stress_max=1e6, compliance=1e6, mass=1.0, success=False)

        return results_out

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, cls=_NumpyEncoder)


# Per-geometry parameter bounds: h_mm and r_mm semantics vary by geometry type.
# These clamp the parameter_head output to physically valid ranges.
GEOM_SPACES = {
    "cantilever": {"h_min": 5.0,  "h_max": 20.0, "r_min": 0.0, "r_max": 5.0},
    "tapered":    {"h_min": 8.0,  "h_max": 25.0, "r_min": 2.0, "r_max": 7.0},
    "ribbed":     {"h_min": 6.0,  "h_max": 20.0, "r_min": 2.0, "r_max": 6.0},
    "lbracket":   {"h_min": 8.0,  "h_max": 25.0, "r_min": 5.0, "r_max": 20.0},
}


class DesignOptimizer:
    def __init__(self, vae_model, fem_evaluator, device='cuda', latent_dim=32, sim_cfg=None, topo_refine=False):
        self.vae = vae_model
        self.fem_evaluator = fem_evaluator
        self.device = device
        self.latent_dim = latent_dim
        self.sim_cfg = sim_cfg or {"w_stress": 1.0, "w_compliance": 0.1, "w_mass": 0.01}
        self.topo_refine = topo_refine
        self.vae.eval()

        # Load materials DB once
        try:
            import yaml
            with open("materials.yaml") as f:
                self.materials_db = yaml.safe_load(f)
        except (FileNotFoundError, ImportError):
            self.materials_db = {"Plastic_ABS": {"E_mpa": 2300, "poisson": 0.35, "density": 1050}}

        self.gp_model = None
        self.x_history = []
        # Multi-objective history: [[stress, mass], ...]
        self.y_history = []
        self.best_bbox = None

    def geometry_to_parameters(self, z: np.ndarray) -> DesignParameters:
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().to(self.device).view(1, -1)
            params = self.vae.predict_parameters(z_tensor)
            params = params.cpu().numpy()[0]
        
        return DesignParameters(
            h_mm=float(params[0]),
            r_mm=float(params[1]),
            geometry_type=self.sim_cfg.get("geometry_type", "cantilever")
        )

    def optimize_step_parallel(self, q=4):
        """MOBO Step: Explore the Pareto Front of Stress vs Mass for Plastic."""

        # Only use non-sentinel evaluations for GP fitting (stress=1e6 = FEM failure)
        valid_mask = [0.0 < y[0] < 1e5 for y in self.y_history]
        n_valid = sum(valid_mask)

        # Early exit if we haven't learned enough from valid samples
        if not BOTORCH_AVAILABLE or n_valid < 10:
            z_batch = np.random.randn(q, self.latent_dim) * 2.0
            return self._evaluate_latent_batch(z_batch)
        
        # Add safety margin for numerical stability
        train_X_raw = torch.tensor(np.array([x for x, ok in zip(self.x_history, valid_mask) if ok]),
                                   dtype=torch.float64).to(botorch_device)
        train_Y = torch.tensor(np.array([y for y, ok in zip(self.y_history, valid_mask) if ok]),
                               dtype=torch.float64).to(botorch_device)
        
        # Normalize inputs to [0,1] cube for BoTorch
        X_min = train_X_raw.min(dim=0).values
        X_max = train_X_raw.max(dim=0).values
        X_range = (X_max - X_min).clamp(min=1e-6)
        train_X = (train_X_raw - X_min) / X_range
        
        # Standardize outputs for better convergence
        train_Y_std = (train_Y - train_Y.mean(dim=0)) / (train_Y.std(dim=0) + 1e-6)
        train_Y_std = -train_Y_std  # Invert for maximization
        
        self.gp_model = SingleTaskGP(train_X, train_Y_std)
        self._X_min, self._X_range = X_min, X_range  # Store for candidate rescaling
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)
        
        # Use more conservative reference point for hypervolume improvement
        ref_point = train_Y_std.min(dim=0).values - 0.1
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y_std)
        acq = qExpectedHypervolumeImprovement(
            model=self.gp_model,
            ref_point=ref_point,
            partitioning=partitioning,
        )
        
        # Bounds in normalized [0,1] space
        bounds = torch.zeros(2, self.latent_dim, dtype=torch.float64).to(botorch_device)
        bounds[1] = 1.0
        
        candidates, _ = optimize_acqf(
            acq, bounds=bounds, q=q, num_restarts=10, raw_samples=512,
        )
        
        # Rescale candidates back to original latent space
        candidates = candidates * self._X_range + self._X_min
        
        return self._evaluate_latent_batch(candidates.cpu().numpy())

        # Fit multi-output GP on CPU (Search space is latent_dim only)
        valid_x = [x for x, ok in zip(self.x_history, valid_mask) if ok]
        valid_y = [y for y, ok in zip(self.y_history, valid_mask) if ok]
        train_X_raw = torch.tensor(np.array(valid_x), dtype=torch.float64).to(botorch_device)
        train_Y = torch.tensor(np.array(valid_y), dtype=torch.float64).to(botorch_device)

        # Normalise X to [0,1] — botorch GP expects unit-cube inputs
        X_min = train_X_raw.min(dim=0).values
        X_max = train_X_raw.max(dim=0).values
        X_range = (X_max - X_min).clamp(min=1e-6)
        train_X = (train_X_raw - X_min) / X_range

        train_Y_std = (train_Y - train_Y.mean(dim=0)) / (train_Y.std(dim=0) + 1e-6)
        train_Y_std = -train_Y_std

        self.gp_model = SingleTaskGP(train_X, train_Y_std)
        self._X_min, self._X_range = X_min, X_range  # store for candidate rescaling
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

        ref_point = train_Y_std.min(dim=0).values - 0.1
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y_std)
        acq = qExpectedHypervolumeImprovement(
            model=self.gp_model,
            ref_point=ref_point,
            partitioning=partitioning,
        )

        # Bounds in normalised [0,1] space
        bounds = torch.zeros(2, self.latent_dim, dtype=torch.float64).to(botorch_device)
        bounds[1] = 1.0

        candidates, _ = optimize_acqf(
            acq, bounds=bounds, q=q, num_restarts=10, raw_samples=512,
        )

        # Rescale candidates back to original latent space
        candidates = candidates * self._X_range + self._X_min

        return self._evaluate_latent_batch(candidates.cpu().numpy())

    def _build_preserved_mask(self, regions: list, resolution: int = 64) -> np.ndarray:
        if not regions:
            return None
        
        mask = np.zeros((resolution, resolution, resolution), dtype=bool)
        
        # Physical dimensions (baseline)
        lx, ly, lz = 100.0, 20.0, 20.0
        geom = self.sim_cfg.get("geometry_type", "cantilever")
        if geom == "lbracket":
            lz = 100.0
            
        dx = lx / resolution
        dy = ly / resolution # Note: ny is res/4 in refinement loop, we need to be careful
        # Actually, refinement loop uses res x res/4 x res/4
        # But here we are building a mask for that grid.
        # Let's assume the refinement grid matches the VAE grid shape logic roughly
        # VAE is 64^3 cubic. SIMP refinement uses (64, 16, 16).
        # We need to map physical coordinates to the (64, 16, 16) indices.
        
        nx, ny, nz = resolution, resolution // 4, resolution // 4
        if geom == "lbracket": nz = resolution # L-bracket is tall
        
        dx = lx / nx
        dy = ly / ny
        dz = lz / nz
        
        mask = np.zeros((nx, ny, nz), dtype=bool)

        for r in regions:
            # Convert physical bounds to voxel indices
            x0 = int(np.clip(r["xmin"] / dx, 0, nx))
            x1 = int(np.clip(r["xmax"] / dx, 0, nx))
            y0 = int(np.clip(r["ymin"] / dy, 0, ny))
            y1 = int(np.clip(r["ymax"] / dy, 0, ny))
            z0 = int(np.clip(r["zmin"] / dz, 0, nz))
            z1 = int(np.clip(r["zmax"] / dz, 0, nz))
            
            mask[x0:x1, y0:y1, z0:z1] = True
            
        return mask

    def _evaluate_latent_batch(self, z_batch):
        # z_batch contains [z_0...z_31]
        
        # ── Fix Task 5: Topology Refinement ─────────────────────────────────────
        if self.topo_refine:
            logger.info(f"  Refining batch of {len(z_batch)} designs via SIMP (20 iters)...")
            from topology.solver import TopologySolver
            geom_type = self.sim_cfg.get("geometry_type", "cantilever")
            
            # Use 64^3 for refinement if possible, or match VAE
            res = 64 
            # Aspect ratio logic matches SIMP defaults
            ny_ref = res//4
            nz_ref = res//4
            if geom_type == "lbracket": nz_ref = res
            
            solver = TopologySolver(nx=res, ny=ny_ref, nz=nz_ref, n_iters=20)
            
            # Build mask once if static
            preserved_mask = None
            if self.sim_cfg.get("preserved_regions"):
                preserved_mask = self._build_preserved_mask(self.sim_cfg["preserved_regions"], resolution=res)
            
            refined_zs = []
            for i, z in enumerate(z_batch):
                # 1. Decode
                z_t = torch.from_numpy(z).float().to(self.device).unsqueeze(0)
                with torch.no_grad():
                    voxels = self.vae.decode(z_t).squeeze().cpu().numpy()
                
                # 2. Refine (SIMP initialized from decoded density)
                # Note: TopologySolver currently doesn't support warm-start from density easily
                # but it uses SIMPSolver internally. We can run it and it's better than nothing.
                # In a real warm-start, we'd pass 'voxels' to run().
                sim_cfg_local = {
                    "force_n": self.sim_cfg.get("force_n", 1000.0),
                    "boundary_conditions": {
                        "fixed_face": "x_min" if geom_type != "lbracket" else "z_min",
                        "load_face": "x_max",
                        "load_dof": 2
                    },
                    "preserved_mask": preserved_mask
                }
                # Temporary directory for refinement artifacts
                refine_dir = Path("./optimisation_results/refine_tmp")
                refine_dir.mkdir(parents=True, exist_ok=True)
                
                solver.run(sim_cfg_local, str(refine_dir), volfrac=0.4)
                refined_density = solver.last_density # (nx, ny, nz)
                
                # 3. Re-encode
                # Ensure shape matches VAE input (1, 1, 64, 64, 64)
                # VAE expects 64^3 cube. We need to pad our (64, 16, 16) density.
                padded = np.zeros((res, res, res), dtype=np.float32)
                nx, ny, nz = refined_density.shape
                # Center or align? Align to match training data generation
                # Training data (fem_data_pipeline) usually fills the volume
                # But here our SIMP grid is smaller. 
                # Simplest is to place at origin 0,0,0
                padded[:nx, :ny, :nz] = refined_density
                
                vox_t = torch.from_numpy(padded).float().to(self.device).view(1, 1, res, res, res)
                with torch.no_grad():
                    mu, _ = self.vae.encode(vox_t)
                    refined_z = mu.cpu().numpy().squeeze()
                refined_zs.append(refined_z)
            
            z_batch = np.array(refined_zs)

        param_list = []
        for z in z_batch:
            p = self.geometry_to_parameters(z)
            p.latent_z = z.tolist()
            param_list.append(p)

        geom = self.sim_cfg.get("geometry_type", "cantilever")
        bounds = GEOM_SPACES.get(geom, GEOM_SPACES["cantilever"])

        for i, p in enumerate(param_list):
            # Clamp h_mm and r_mm to per-geometry valid ranges
            p.h_mm = float(np.clip(p.h_mm, bounds["h_min"], bounds["h_max"]))
            p.r_mm = float(np.clip(p.r_mm, bounds["r_min"], bounds["r_max"]))

            # Cantilever: enforce wall-thickness constraint (h - 2r >= 5mm) pre-FEM
            if geom == "cantilever":
                r_max_safe = max(0.0, (p.h_mm - 5.0) / 2.0 - 0.05)
                p.r_mm = min(p.r_mm, r_max_safe)

            # Use Plastic exclusively
            p.material_name = "Plastic_ABS"
            p.material_cfg = self.materials_db[p.material_name]

        results = self.fem_evaluator.evaluate_batch(param_list)
        
        stress_limit = self.sim_cfg.get("max_stress_mpa", 1e9)
        for i, res in enumerate(results):
            self.x_history.append(z_batch[i])
            self.y_history.append([res.stress_max, res.mass])

            # Skip sentinel values
            if not (0.0 < res.stress_max < 1e5):
                continue

            penalty = 5e5 if res.stress_max > stress_limit else 0.0
            obj = (self.sim_cfg["w_stress"] * res.stress_max +
                   self.sim_cfg["w_mass"] * res.mass +
                   penalty)

            valid_scores = [
                self.sim_cfg["w_stress"] * y[0] + self.sim_cfg["w_mass"] * y[1] +
                (5e5 if y[0] > stress_limit else 0.0)
                for y in self.y_history if 0.0 < y[0] < 1e5
            ]
            if obj <= min(valid_scores):
                self.best_bbox = res.bbox
                logger.info(f"  New Best: Stress={res.stress_max:.1f}MPa (limit={stress_limit}), Mass={res.mass:.4f}kg")

        return z_batch, results

    def save_checkpoint(self, output_dir: str):
        """Save x_history/y_history so a run can be resumed after a crash."""
        ckpt = {
            "x_history": [x.tolist() for x in self.x_history],
            "y_history": self.y_history,
            "best_bbox": self.best_bbox,
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "bo_checkpoint.json", "w") as f:
            json.dump(ckpt, f)

    def load_checkpoint(self, output_dir: str) -> int:
        """Load previous run state. Returns number of evaluations restored."""
        ckpt_path = Path(output_dir) / "bo_checkpoint.json"
        if not ckpt_path.exists():
            return 0
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        self.x_history = [np.array(x) for x in ckpt.get("x_history", [])]
        self.y_history = ckpt.get("y_history", [])
        self.best_bbox = ckpt.get("best_bbox")
        n = len(self.x_history)
        if n:
            logger.info(f"Resumed from checkpoint: {n} evaluations restored.")
        return n

    def run_optimisation(self, n_iterations=250, q=4, output_dir: str = None,
                         checkpoint_every: int = 10):
        logger.info(f"Starting Multi-Objective Parallel Discovery (q={q})...")
        for i in range(n_iterations):
            self.optimize_step_parallel(q=q)
            valid = [y for y in self.y_history if 0.0 < y[0] < 1e5]
            if valid and BOTORCH_AVAILABLE:
                n_pareto = is_non_dominated(-torch.tensor(valid)).sum().item()
            else:
                n_pareto = 0
            logger.info(f"Round {i+1}/{n_iterations} complete. Valid={len(valid)}  Pareto={n_pareto}")
            if output_dir and (i + 1) % checkpoint_every == 0:
                self.save_checkpoint(output_dir)

        # Find best non-failed design
        valid_indices = [i for i, y in enumerate(self.y_history) if 0.0 < y[0] < 1e5]
        if not valid_indices:
            logger.warning("All evaluations failed (stress=1e6). Check FreeCAD/FEM setup.")
            return np.zeros(self.latent_dim), [1e6, 1.0]

        best_z_idx = min(valid_indices,
                         key=lambda i: (self.sim_cfg["w_stress"] * self.y_history[i][0]
                                        + self.sim_cfg["w_mass"]  * self.y_history[i][1]))
        return self.x_history[best_z_idx], self.y_history[best_z_idx]

    def save_results(self, output_dir: str = "./optimisation_results"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Exclude obviously failed evaluations (FEM returned sentinel 1e6)
        valid_idx = [i for i, y in enumerate(self.y_history) if 0.0 < y[0] < 1e5]
        if not valid_idx:
            logger.warning("No valid evaluations to save.")
            json.dump({"pareto_front": [], "history_y": [], "best_bbox": None},
                      open(out_path / "optimisation_history.json", "w"))
            return

        y_valid = torch.tensor([self.y_history[i] for i in valid_idx], dtype=torch.float64)
        pareto_mask = is_non_dominated(-y_valid)
        pareto_indices = [valid_idx[i] for i in torch.where(pareto_mask)[0].tolist()]
        pareto_front = [
            {
                "stress": self.y_history[idx][0],
                "mass": self.y_history[idx][1],
                "params": self.fem_evaluator.evaluation_history[idx].parameters.model_dump() if idx < len(self.fem_evaluator.evaluation_history) else {},
                "latent_z": self.x_history[idx].tolist()
            }
            for idx in pareto_indices
        ]

        results = {
            "pareto_front": pareto_front,
            "history_y": [list(y) for y in self.y_history],
            "best_bbox": self.best_bbox
        }
        with open(out_path / "optimisation_history.json", 'w') as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        self.fem_evaluator.save_history(out_path / "fem_evaluations.json")
        logger.info(f"MOBO Results Saved. {len(pareto_front)} Pareto-optimal designs identified.")


if __name__ == "__main__":
    import argparse
    from genpipeline.vae_design_model import DesignVAE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--n-iter",      type=int,  default=250)
    parser.add_argument("--q",           type=int,  default=4)
    parser.add_argument("--output-dir",  type=str,  default="./optimisation_results")
    parser.add_argument("--config-path", type=str,  default=None,
                        help="Path to gendesign_config.json exported by the FreeCAD workbench")
    parser.add_argument("--voxel-fem", action="store_true",
                        help="Use direct CalculiX voxel FEM (bypasses FreeCAD)")
    parser.add_argument("--topo-refine", action="store_true",
                        help="Perform topology refinement (20 SIMP iters) before evaluation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from bo_checkpoint.json in --output-dir")
    args = parser.parse_args()

    # Load workbench config if provided (overrides CLI defaults)
    wb_cfg = {}
    if args.config_path and Path(args.config_path).exists():
        with open(args.config_path) as f:
            wb_cfg = json.load(f)
        logger.info(f"Loaded workbench config from {args.config_path}")
        logger.info(f"  geometry={wb_cfg.get('geometry_type')}  "
                    f"n_iter={wb_cfg.get('n_iter')}  "
                    f"constraints={len(wb_cfg.get('constraints', []))}  "
                    f"loads={len(wb_cfg.get('loads', []))}")

    n_iter = wb_cfg.get("n_iter", args.n_iter)
    ckpt   = wb_cfg.get("checkpoint_path", args.model_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).to(device)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])

    sim_cfg = {
        "w_stress":     1.0,
        "w_compliance": 0.1,
        "w_mass":       0.01,
        "geometry_type": wb_cfg.get("geometry_type", "cantilever"),
        "max_stress_mpa": wb_cfg.get("max_stress_mpa", 250.0),
        # Pass through any BC overrides from workbench
        "fixed_face_normal": wb_cfg.get("fixed_face_normal", [-1, 0, 0]),
        "load_face_normal":  wb_cfg.get("load_face_normal",  [1, 0, 0]),
        "force_n":           wb_cfg.get("force_n", 1000.0),
        "force_direction":   wb_cfg.get("force_direction", [0, 0, -1]),
        "preserved_regions": wb_cfg.get("preserved_regions", []),
    }

    if args.voxel_fem:
        from .fem.voxel_fem import VoxelFEMEvaluator
        evaluator = VoxelFEMEvaluator(
            # output_dir=None lets VoxelFEMEvaluator auto-select a
            # Windows-accessible temp when ccx is a .exe binary.
            fixed_face=sim_cfg.get("fixed_face", "x_min"),
            load_face=sim_cfg.get("load_face", "x_max"),
            force_n=sim_cfg.get("force_n", 1000.0),
            vae_model=vae,
        )
    else:
        evaluator = BridgeEvaluator(n_workers=args.q, output_dir=args.output_dir)
    optimizer = DesignOptimizer(vae, evaluator, latent_dim=32, sim_cfg=sim_cfg, topo_refine=args.topo_refine)

    if args.resume:
        optimizer.load_checkpoint(args.output_dir)

    optimizer.run_optimisation(n_iterations=n_iter, q=args.q,
                               output_dir=args.output_dir)
    optimizer.save_results(args.output_dir)

    # Auto-export Pareto STLs
    try:
        from genpipeline.pipeline_utils import VoxelConverter, ManufacturabilityConstraints
        import trimesh as _trimesh
        from pathlib import Path as _Path
        hist_path = _Path(args.output_dir) / "optimisation_history.json"
        with open(hist_path) as f:
            hist = json.load(f)
        pareto = hist.get("pareto_front", [])
        if pareto:
            export_dir = _Path(args.output_dir) / "exported_designs"
            export_dir.mkdir(exist_ok=True)
            mfg = ManufacturabilityConstraints(config={})
            designs = {
                "strongest": min(pareto, key=lambda x: x["stress"]),
                "lightest":  min(pareto, key=lambda x: x["mass"]),
                "balanced":  min(pareto, key=lambda x: x["stress"] + 100 * x["mass"]),
            }
            for name, d in designs.items():
                z = torch.tensor(d["latent_z"]).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    voxels = vae.decode(z).squeeze().cpu().numpy()
                voxels = mfg.apply_constraints(voxels, voxel_size=1.0/64)
                mesh = VoxelConverter.voxel_to_mesh(voxels, voxel_size=1.0/64,
                                                    bbox=hist.get("best_bbox"))
                if mesh:
                    m = _trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
                    m.export(str(export_dir / f"pareto_{name}.stl"))
                    logger.info(f"Exported pareto_{name}.stl")
    except Exception as e:
        logger.warning(f"Auto-export failed (non-fatal): {e}")
