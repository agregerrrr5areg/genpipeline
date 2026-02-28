"""dashboard_bo_runner.py — wraps DesignOptimizer for step-by-step BO in a thread."""
from __future__ import annotations
import numpy as np
import torch
from dashboard_state import AppState, IterResult, FEMResult

from optimization_engine import DesignOptimizer


class BORunner:
    """
    Runs Bayesian optimisation step-by-step, writing each result into AppState.

    mode="bo-only"  — uses VAE performance head as surrogate (no FreeCAD)
    mode="full-fem" — calls freecad_bridge.run_variant for real stress per step
    """

    def __init__(self, state: AppState, vae, device: str,
                 n_iters: int = 50, mode: str = "bo-only",
                 freecad_cmd: str = "", output_dir: str = "/tmp/bo_variants"):
        self.state = state
        self.vae = vae
        self.device = device
        self.n_iters = n_iters
        self.mode = mode
        self.freecad_cmd = freecad_cmd
        self.output_dir = output_dir

    def _decode_voxel(self, z: np.ndarray) -> np.ndarray | None:
        try:
            zt = torch.tensor(z, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
            with torch.no_grad():
                voxel = self.vae.decode(zt).squeeze().cpu().numpy()
            return (voxel > 0.5).astype(np.uint8)
        except Exception:
            return None

    def _fem_validate(self, z: np.ndarray) -> FEMResult | None:
        if self.mode != "full-fem" or not self.freecad_cmd:
            return None
        voxel = self._decode_voxel(z)
        if voxel is None:
            return None
        occ = float(voxel.mean())
        h_mm = float(np.clip(5.0 + occ * 150.0, 5.0, 20.0))
        r_mm = 0.0
        try:
            from freecad_bridge import run_variant
            result = run_variant(self.freecad_cmd, h_mm, r_mm, self.output_dir)
            if result:
                return FEMResult(
                    stress_max=result["stress_max"],
                    displacement_max=result["displacement_max"],
                    mass=result["mass"],
                    h_mm=h_mm,
                    r_mm=r_mm,
                )
        except Exception:
            pass
        return None

    def run(self) -> None:
        self.state.status = "running"
        try:
            optimizer = DesignOptimizer(
                vae_model=self.vae,
                fem_evaluator=None,
                device=self.device,
                latent_dim=getattr(self.vae, "latent_dim", 16),
            )
            optimizer.initialize_search(n_init_points=5)

            for step in range(self.n_iters):
                if self.state.stop_requested:
                    break

                z, _ = optimizer.optimize_step()
                obj = float(optimizer.y_history[-1])
                voxel = self._decode_voxel(z)
                fem = self._fem_validate(z)

                result = IterResult(
                    step=step + 1,
                    objective=obj,
                    z=z.tolist() if hasattr(z, "tolist") else list(z),
                    voxel=voxel,
                    fem=fem,
                )
                self.state.add_iteration(result)

        except Exception as e:
            print(f"[BORunner] error: {e}")
        finally:
            self.state.status = "done"
