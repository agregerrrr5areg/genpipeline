"""
Integration tests for the core decode→FEM path.

PSEUDOCODE — what this file does end to end:
─────────────────────────────────────────────
  SETUP
    load VAE weights from checkpoints/vae_best.pth
    fix random seed → sample 4 latent vectors z[0..3]  (32-dim each)
    probe for ccx binary (FreeCAD install or PATH)
    probe for SIMD mesher extension (needs ninja + GCC)

  TestDecodeAlwaysRuns  ← runs everywhere, no external tools needed
    test_checkpoint_exists     : file exists on disk?
    test_checkpoint_keys       : saved dict has expected keys + shapes?
    test_decode_returns_correct_shape:
        FOR each z IN z[0..3]:
            z → VAE decoder (4³ → 8³ → 16³ → 32³ → 64³ conv-transpose)
        ASSERT output shape == (4, 1, 64, 64, 64)
    test_voxels_are_non_trivial:
        FOR each z IN z[0..3]:
            sigmoid(decode(z)) → voxel probabilities
            occupancy = mean(probabilities)
        ASSERT 5% ≤ occupancy ≤ 95%   (not all-empty, not all-solid)

  TestFEMWithCCX  ← skipped if no ccx; individual tests skip if no SIMD
    test_evaluate_returns_four_results:
        FOR each z IN z[0..3]:
            decode(z) → 64³ voxels
            downsample 64³ → 32³
            smooth voxels (gaussian blur + threshold)
            build C3D8 hex mesh (one element per solid voxel)
            write .inp file → run ccx.exe → parse .frd output
        ASSERT each result has keys: stress, compliance, mass

    test_at_least_one_non_sentinel:
        same 4 evaluations as above
        ASSERT at least 1 result has stress < 1e5 MPa  (not a sentinel failure)
        ON FAIL: print failure_reason for each result to show where chain broke

  SKIP LOGIC
    _CCX_OK  → class-level skipif (ccx binary absent → skip whole class)
    _SIMD_OK → per-test pytest.skip (ninja absent → skip individual test)

TestDecodeAlwaysRuns — no external dependencies, always runs.
TestFEMWithCCX       — skipped unless ccx is discoverable.
"""
import pytest
import torch
import numpy as np
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "vae_best.pth"
LATENT_DIM = 32
INPUT_SHAPE = (64, 64, 64)


def _load_vae():
    """Load DesignVAE from the best checkpoint, set eval mode."""
    from genpipeline.vae_design_model import DesignVAE
    vae = DesignVAE(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    return vae


def _four_latent_vectors():
    """Reproducible set of 4 latent vectors (seed 42)."""
    torch.manual_seed(42)
    return torch.randn(4, LATENT_DIM)


def _ccx_available() -> bool:
    from genpipeline.fem.voxel_fem import find_ccx
    return find_ccx() is not None


def _simd_available() -> bool:
    """Check whether the SIMD mesher kernel can be compiled/loaded.

    Requires ninja and a working C++ toolchain. Kept separate from
    _ccx_available() so that a ninja/compilation failure does not mask a
    genuine CalculiX availability check.
    """
    try:
        from genpipeline.cuda_kernels import get_solid_voxels_simd
        probe = np.ones((4, 4, 4), dtype=np.float32)
        get_solid_voxels_simd(probe)
    except Exception:
        return False
    return True


# Evaluated once at collection time — avoids repeated JIT compilation on every collect.
# SIMD check is kept separate: ccx absence skips the whole class; SIMD absence skips
# individual tests inside the class with a more specific reason string.
_CCX_OK: bool = _ccx_available()
_SIMD_OK: bool = _simd_available()

# ── TestDecodeAlwaysRuns ──────────────────────────────────────────────────────

class TestDecodeAlwaysRuns:

    def test_checkpoint_exists(self):
        assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"

    def test_checkpoint_keys(self):
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert ckpt.get("latent_dim") == LATENT_DIM
        assert tuple(ckpt.get("input_shape", ())) == INPUT_SHAPE

    def test_decode_returns_correct_shape(self):
        vae = _load_vae()
        zs = _four_latent_vectors()
        with torch.no_grad():
            out = vae.decode(zs)
        assert out.shape == (4, 1, 64, 64, 64), f"Unexpected shape: {out.shape}"

    def test_voxels_are_non_trivial(self):
        """Each decoded voxel must have 5%–95% occupancy (not all-zero, not saturated)."""
        vae = _load_vae()
        zs = _four_latent_vectors()
        with torch.no_grad():
            probs = torch.sigmoid(vae.decode(zs))  # (4, 1, 64, 64, 64)
        for i in range(4):
            occ = probs[i].mean().item()
            assert 0.05 <= occ <= 0.95, (
                f"z[{i}] occupancy {occ:.3f} out of expected 5%–95% range"
            )


# ── TestFEMWithCCX ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CCX_OK, reason="CalculiX (ccx) not found on this system")
class TestFEMWithCCX:

    def test_evaluate_returns_four_results(self, tmp_path):
        if not _SIMD_OK:
            pytest.skip("SIMD mesher extension unavailable (ninja + C++ toolchain required)")
        from genpipeline.fem.voxel_fem import VoxelFEMEvaluator
        vae = _load_vae()
        evaluator = VoxelFEMEvaluator(vae_model=vae, output_dir=str(tmp_path))
        zs = _four_latent_vectors().numpy()

        results = [evaluator.evaluate(z, vae) for z in zs]

        assert len(results) == 4
        for r in results:
            assert "stress" in r
            assert "compliance" in r
            assert "mass" in r

    def test_at_least_one_non_sentinel(self, tmp_path):
        if not _SIMD_OK:
            pytest.skip("SIMD mesher extension unavailable (ninja + C++ toolchain required)")
        from genpipeline.fem.voxel_fem import VoxelFEMEvaluator
        from genpipeline.pipeline_utils import is_valid_fem_result
        vae = _load_vae()
        evaluator = VoxelFEMEvaluator(vae_model=vae, output_dir=str(tmp_path))
        zs = _four_latent_vectors().numpy()

        results = [evaluator.evaluate(z, vae) for z in zs]

        non_sentinel = [r for r in results if is_valid_fem_result(r)]
        assert len(non_sentinel) >= 1, (
            f"All 4 FEM evaluations returned invalid/sentinel results. "
            f"Failure reasons: {[r.get('failure_reason') for r in results]}"
        )
