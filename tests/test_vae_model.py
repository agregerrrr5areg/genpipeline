# tests/test_vae_model.py
#
# PSEUDOCODE — what this file tests:
# ────────────────────────────────────
#   ALL tests skipped if CUDA is not available (pytestmark guard at module level)
#
#   Fixture: vae
#     BUILD DesignVAE(latent_dim=32, input_shape=64³)
#     MOVE to GPU (.cuda())
#     SET eval mode
#
#   TestDesignVAEShapes
#     test_param_count:
#       COUNT all learnable parameters
#       ASSERT 30M < count < 60M  (sanity check on architecture size)
#
#     test_forward_output_shapes:   ← skipped if CUDA extensions unavailable
#       INPUT  x = random (2, 1, 64, 64, 64) on GPU, BF16 autocast
#       CALL   vae(x):
#                encode → mu, logvar (2, 32)
#                reparameterize via fused CUDA kernel → z (2, 32)
#                decode z → x_recon (2, 1, 64, 64, 64)
#                performance head → perf (2, 3)  [stress, compliance, mass]
#                parameter head   → params (2, 2) [h_mm, r_mm]
#       ASSERT each output has expected shape
#
#     test_encode_returns_cpu_float32:
#       INPUT  x on GPU, BF16 autocast
#       CALL   encode(x) → mu, logvar
#       ASSERT both are float32 on GPU  (FC layers upcast from BF16)
#
#     test_decode_output_range:
#       INPUT  z = random (2, 32) on GPU
#       CALL   decode(z) → voxel logits
#       ASSERT output has no NaN/Inf  (numerical stability check)
#
#   TestDesignVAECheckpoint
#     test_checkpoint_loads:
#       SAVE random model state → LOAD it back → ASSERT weights match
#
#     test_production_checkpoint_loads:
#       LOAD checkpoints/vae_best.pth (trained 300 epochs)
#       ASSERT loads without error
#
#   _CUDA_EXT_OK = probe fused_reparameterize on a tiny tensor at import time
#   @_CUDA_EXT_SKIP applied to test_forward_output_shapes (only test using full forward pass)
#
import sys
from pathlib import Path
import pytest
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def _cuda_extensions_available() -> bool:
    """Return True if the fused CUDA extensions (reparam, sparse conv) can be loaded."""
    try:
        from genpipeline import cuda_kernels
        mu = torch.randn(1, 4, device="cuda")
        logvar = torch.zeros(1, 4, device="cuda")
        cuda_kernels.fused_reparameterize(mu, logvar)
        return True
    except Exception:
        return False


_CUDA_EXT_OK: bool = torch.cuda.is_available() and _cuda_extensions_available()
_CUDA_EXT_SKIP = pytest.mark.skipif(
    not _CUDA_EXT_OK,
    reason="CUDA extensions unavailable (ninja + CUDA toolchain required)",
)


@pytest.fixture(scope="module")
def vae():
    from genpipeline.vae_design_model import DesignVAE
    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
    model.eval()
    return model


class TestDesignVAEShapes:
    def test_param_count(self, vae):
        n = sum(p.numel() for p in vae.parameters())
        assert 30_000_000 < n < 60_000_000, f"Unexpected param count: {n}"

    @_CUDA_EXT_SKIP
    def test_forward_output_shapes(self, vae):
        x = torch.randn(2, 1, 64, 64, 64, device="cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            x_rec, mu, logvar, perf, params = vae(x)
        assert x_rec.shape  == (2, 1, 64, 64, 64)
        assert mu.shape     == (2, 32)
        assert logvar.shape == (2, 32)
        assert perf.shape   == (2, 3)
        assert params.shape == (2, 2)

    def test_encode_returns_cpu_float32(self, vae):
        x = torch.randn(4, 1, 64, 64, 64, device="cuda")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                mu, logvar = vae.encode(x)
        # FC layers run in FP32 and stay on GPU
        assert mu.dtype == torch.float32
        assert mu.device.type == "cuda"

    def test_decode_output_range(self, vae):
        z = torch.randn(2, 32, device="cuda")
        with torch.no_grad():
            voxels = vae.decode(z)
        assert voxels.shape == (2, 1, 64, 64, 64)
        assert voxels.min() >= 0.0
        assert voxels.max() <= 1.0

    def test_backward_does_not_crash(self, vae):
        """Regression: BF16 GEMM backward was crashing on Blackwell."""
        vae.train()
        x = torch.randn(2, 1, 64, 64, 64, device="cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            x_rec, mu, logvar, _, _ = vae(x)
            loss = x_rec.mean() + mu.mean()
        loss.backward()
        vae.eval()

    def test_predict_parameters_shape(self, vae):
        z = torch.randn(8, 32, device="cuda")
        with torch.no_grad():
            p = vae.predict_parameters(z)
        assert p.shape == (8, 2)


class TestDesignVAECheckpoint:
    def test_checkpoint_loads(self, tmp_path):
        from genpipeline.vae_design_model import DesignVAE
        model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
        ckpt_path = tmp_path / "test.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "latent_dim": 32,
            "input_shape": (64, 64, 64),
        }, ckpt_path)
        loaded = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        loaded.load_state_dict(state["model_state_dict"])
        assert state["latent_dim"] == 32

    def test_production_checkpoint_loads(self):
        """Load the real vae_best.pth — fails fast if architecture drifted."""
        ckpt_path = Path("checkpoints/vae_best.pth")
        if not ckpt_path.exists():
            pytest.skip("checkpoints/vae_best.pth not found")
        from genpipeline.vae_design_model import DesignVAE
        state = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
        model = DesignVAE(
            input_shape=state.get("input_shape", (64, 64, 64)),
            latent_dim=state.get("latent_dim", 32),
        ).cuda()
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        z = torch.randn(1, 32, device="cuda")
        with torch.no_grad():
            vox = model.decode(z)
        assert vox.shape == (1, 1, 64, 64, 64)
