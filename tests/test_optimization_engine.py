# tests/test_optimization_engine.py
"""
Tests for DesignOptimizer: GP fallback and Pareto filtering.
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_utils import FEM_SENTINEL, FEM_VALID_THRESHOLD
from optimization_engine import DesignOptimizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_vae(latent_dim=4):
    """Minimal VAE mock: decode() returns a (1, 1, D, D, D) tensor of ones."""
    import torch
    vae = MagicMock()
    vae.eval.return_value = None
    vae.latent_dim = latent_dim

    dummy = torch.ones(1, 1, 8, 8, 8)
    vae.decode.return_value = dummy

    # parameter_head mock
    def predict_parameters(z):
        return torch.tensor([[10.0, 2.0]])
    vae.predict_parameters = predict_parameters

    # .parameters() for device detection
    p = torch.nn.Parameter(torch.zeros(1))
    vae.parameters = lambda: iter([p])
    return vae


def _make_sentinel_evaluator():
    """Evaluator that always returns all-sentinel results."""
    ev = MagicMock()
    ev.evaluation_history = []

    def evaluate_batch(param_list):
        return [
            {"stress": FEM_SENTINEL, "compliance": FEM_SENTINEL,
             "mass": 1.0, "failure_reason": "no_ccx", "parameters": p}
            for p in param_list
        ]

    ev.evaluate_batch = evaluate_batch
    ev.save_history = MagicMock()
    return ev


def _make_mixed_evaluator(n_valid=3, n_sentinel=2):
    """Evaluator returning a mix of valid and sentinel results."""
    call_count = [0]
    ev = MagicMock()
    history = []

    def evaluate_batch(param_list):
        results = []
        for p in param_list:
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                r = {"stress": FEM_SENTINEL, "compliance": FEM_SENTINEL,
                     "mass": 1.0, "failure_reason": "no_ccx", "parameters": p}
            else:
                stress = float(10 + call_count[0])  # small valid stress
                r = {"stress": stress, "compliance": 0.1, "mass": 0.5, "parameters": p}
            history.append(r)
            results.append(r)
        return results

    ev.evaluate_batch = evaluate_batch
    ev.evaluation_history = history
    ev.save_history = MagicMock()
    return ev


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAllFailedEvalsFallback:
    def test_all_failed_evals_fallback_to_random(self):
        """
        When every FEM evaluation returns sentinel, optimize_step_parallel()
        must still return a batch of candidate latent vectors without raising.
        """
        latent_dim = 4
        vae = _make_mock_vae(latent_dim)
        evaluator = _make_sentinel_evaluator()
        optimizer = DesignOptimizer(
            vae_model=vae, fem_evaluator=evaluator,
            device="cpu", latent_dim=latent_dim,
        )

        # Run one optimisation step — should NOT raise even with no valid history
        z_batch, results = optimizer.optimize_step_parallel(q=2)

        assert z_batch is not None
        assert len(results) == 2
        # All stress values should equal sentinel
        for r in results:
            assert r["stress"] == FEM_SENTINEL

    def test_run_optimization_all_failed_returns_zeros(self):
        """
        run_optimization() with all-sentinel FEM results should return (zeros, sentinel)
        without crashing.
        """
        latent_dim = 4
        vae = _make_mock_vae(latent_dim)
        evaluator = _make_sentinel_evaluator()
        optimizer = DesignOptimizer(
            vae_model=vae, fem_evaluator=evaluator,
            device="cpu", latent_dim=latent_dim,
        )

        best_z, best_y = optimizer.run_optimization(n_iterations=2, q=2)

        assert best_z is not None
        assert len(best_z) == latent_dim
        assert best_y[0] == FEM_SENTINEL


class TestParetoFilterExcludesSentinel:
    def test_pareto_filter_excludes_sentinel(self, tmp_path):
        """
        save_results() should only include designs with valid (< FEM_VALID_THRESHOLD)
        stress values in the pareto_front.
        """
        import json
        import torch

        latent_dim = 4
        vae = _make_mock_vae(latent_dim)
        evaluator = _make_mixed_evaluator()
        optimizer = DesignOptimizer(
            vae_model=vae, fem_evaluator=evaluator,
            device="cpu", latent_dim=latent_dim,
        )

        # Inject history with a mix of valid and sentinel results
        rng = np.random.default_rng(42)
        for i in range(6):
            optimizer.x_history.append(rng.standard_normal(latent_dim))
            if i % 2 == 0:
                optimizer.y_history.append([FEM_SENTINEL, 1.0])   # sentinel
            else:
                optimizer.y_history.append([10.0 + i, 0.5])       # valid

        optimizer.save_results(str(tmp_path))

        hist_path = tmp_path / "optimization_history.json"
        assert hist_path.exists()
        hist = json.loads(hist_path.read_text())

        pareto = hist.get("pareto_front", [])
        # Every Pareto entry must have a valid (non-sentinel) stress
        for entry in pareto:
            assert entry["stress"] < FEM_VALID_THRESHOLD, (
                f"Sentinel stress {entry['stress']} found in Pareto front"
            )
