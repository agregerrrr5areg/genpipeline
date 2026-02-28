"""
Synthetic end-to-end test of the FEMbyGEN pipeline.
No FreeCAD required — generates random voxel geometries with fake FEM metrics.

Stages:
  1. Generate synthetic dataset (spheres / boxes / cylinders)
  2. Train 3D VAE
  3. Bayesian optimisation loop (VAE predictor as surrogate, no real FEM)
  4. Print summary + save checkpoint

Run:
    source venv/bin/activate
    python synthetic_test.py
"""

import os, json, logging, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from blackwell_compat import botorch_device
from fem_data_pipeline import FEMDataset, DesignSample
from vae_design_model import DesignVAE, VAETrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
N_SAMPLES      = 128   # synthetic training samples
RESOLUTION     = 32    # voxel grid size
LATENT_DIM     = 16
BATCH_SIZE     = 8
EPOCHS         = 20    # fast smoke-test; raise to 100+ for real training
LR             = 1e-3
BETA_VAE       = 1.0
N_INIT_POINTS  = 8     # BO initialisation random samples
N_BO_ITERS     = 15    # BO iterations
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Synthetic geometry generators ─────────────────────────────────────────

def make_sphere(res, radius_frac=None):
    if radius_frac is None:
        radius_frac = np.random.uniform(0.2, 0.45)
    c = res / 2
    r = res * radius_frac
    xs = np.arange(res) - c
    grid = (xs[:, None, None]**2 + xs[None, :, None]**2 + xs[None, None, :]**2) <= r**2
    return grid.astype(np.float32)

def make_box(res, fill_frac=None):
    if fill_frac is None:
        fill_frac = np.random.uniform(0.3, 0.8)
    half = int(res * fill_frac / 2)
    c = res // 2
    grid = np.zeros((res, res, res), dtype=np.float32)
    grid[c-half:c+half, c-half:c+half, c-half:c+half] = 1.0
    return grid

def make_cylinder(res, radius_frac=None, height_frac=None):
    if radius_frac is None:
        radius_frac = np.random.uniform(0.2, 0.4)
    if height_frac is None:
        height_frac = np.random.uniform(0.4, 0.8)
    c = res / 2
    r = res * radius_frac
    h = int(res * height_frac)
    xs = np.arange(res) - c
    circle = (xs[:, None]**2 + xs[None, :]**2) <= r**2
    grid = np.zeros((res, res, res), dtype=np.float32)
    z0 = (res - h) // 2
    grid[:, :, z0:z0+h] = circle[:, :, None]
    return grid

SHAPE_GENERATORS = [make_sphere, make_box, make_cylinder]

def synthetic_fem_metrics(voxel):
    """Fake but physically-ish correlated FEM metrics from voxel statistics."""
    vol = voxel.mean()
    surface = ((np.diff(voxel, axis=0)**2).mean() +
               (np.diff(voxel, axis=1)**2).mean() +
               (np.diff(voxel, axis=2)**2).mean())
    # thinner / more surface → higher stress; bigger mass → lower stress
    stress    = max(0.01, 100.0 * surface / (vol + 0.01) + np.random.normal(0, 2))
    compliance = max(0.001, 5.0 * surface + np.random.normal(0, 0.1))
    mass      = max(0.01, vol * 1.5 + np.random.normal(0, 0.02))
    return float(stress), float(compliance), float(mass)


# ── 2. Build dataset ──────────────────────────────────────────────────────────

def build_synthetic_dataset(n=N_SAMPLES, res=RESOLUTION):
    logger.info(f"Generating {n} synthetic samples at {res}³ resolution...")
    samples = []
    for i in range(n):
        gen = SHAPE_GENERATORS[i % len(SHAPE_GENERATORS)]
        voxel = gen(res)
        stress, compliance, mass = synthetic_fem_metrics(voxel)
        samples.append(DesignSample(
            geometry_path="synthetic",
            stress_max=stress,
            stress_mean=stress * 0.7,
            compliance=compliance,
            mass=mass,
            parameters={"shape": gen.__name__, "idx": i},
            voxel_grid=voxel,
        ))
    return samples


# ── 3. VAE training ───────────────────────────────────────────────────────────

def train_vae(train_loader, val_loader, epochs=EPOCHS):
    model = DesignVAE(input_shape=(RESOLUTION, RESOLUTION, RESOLUTION), latent_dim=LATENT_DIM)
    trainer = VAETrainer(model, train_loader, val_loader, device=DEVICE, lr=LR, beta=BETA_VAE)

    logger.info(f"Training VAE on {DEVICE} for {epochs} epochs...")
    t0 = time.time()
    trainer.fit(epochs=epochs)
    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed:.1f}s  |  best val loss: {trainer.best_val_loss:.4f}")
    return model


# ── 4. Bayesian optimisation (no FreeCAD) ────────────────────────────────────

def run_bo(vae, n_init=N_INIT_POINTS, n_iters=N_BO_ITERS):
    """
    Pure-predictor BO: objective = VAE performance head (stress + 0.1*compliance).
    GP runs on botorch_device (CPU on Blackwell to avoid cuBLAS bug).
    """
    try:
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.optim import optimize_acqf
        from botorch.acquisition import UpperConfidenceBound
        botorch_ok = True
    except ImportError:
        botorch_ok = False
        logger.warning("BoTorch not available — falling back to random search")

    vae.eval()

    def objective(z_np):
        """Lower is better."""
        with torch.no_grad():
            z = torch.from_numpy(z_np).float().to(DEVICE)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            perf = vae.predict_performance(z).cpu().numpy()
        stress     = perf[0, 0]
        compliance = perf[0, 1]
        return float(stress + 0.1 * compliance)

    # ── random initialisation ──────────────────────────────────────────────
    x_hist, y_hist = [], []
    for _ in range(n_init):
        z = np.random.randn(LATENT_DIM).clip(-3, 3)
        x_hist.append(z)
        y_hist.append(objective(z))

    best_y = min(y_hist)
    best_z = x_hist[int(np.argmin(y_hist))].copy()
    logger.info(f"Init: best objective = {best_y:.4f}")

    if not botorch_ok:
        # random search fallback
        for i in range(n_iters):
            z = np.random.randn(LATENT_DIM).clip(-3, 3)
            y = objective(z)
            x_hist.append(z); y_hist.append(y)
            if y < best_y:
                best_y, best_z = y, z.copy()
                logger.info(f"  iter {i+1}: new best = {best_y:.4f}")
        return best_z, best_y, x_hist, y_hist

    # ── BoTorch GP loop ────────────────────────────────────────────────────
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.optim import optimize_acqf
    from botorch.acquisition import UpperConfidenceBound

    # Latent space is clipped to [-3, 3]; scale to [0, 1] for BoTorch.
    Z_LO, Z_HI = -3.0, 3.0

    def scale(z_np):
        """[-3, 3] → [0, 1]"""
        return (z_np - Z_LO) / (Z_HI - Z_LO)

    def unscale(z_np):
        """[0, 1] → [-3, 3]"""
        return z_np * (Z_HI - Z_LO) + Z_LO

    # Unit-cube bounds, float64 as BoTorch recommends
    bounds = torch.tensor([[0.0]*LATENT_DIM, [1.0]*LATENT_DIM],
                           dtype=torch.float64).to(botorch_device)

    for i in range(n_iters):
        # float64 + unit-cube scaling — suppresses both BoTorch warnings
        train_x = torch.from_numpy(scale(np.array(x_hist))).double().to(botorch_device)
        train_y = torch.from_numpy(np.array(y_hist)).double().unsqueeze(-1).to(botorch_device)

        # standardise y for GP
        y_mean, y_std = train_y.mean(), train_y.std().clamp(min=1e-6)
        train_y_norm = (train_y - y_mean) / y_std

        gp  = SingleTaskGP(train_x, train_y_norm)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq = UpperConfidenceBound(gp, beta=0.2)
        candidate, _ = optimize_acqf(
            acq, bounds=bounds, q=1, num_restarts=5, raw_samples=256,
        )
        # unscale back to [-3, 3] before evaluating the objective
        z_new = unscale(candidate.squeeze().cpu().numpy())
        y_new = objective(z_new)

        x_hist.append(z_new); y_hist.append(y_new)
        if y_new < best_y:
            best_y, best_z = y_new, z_new.copy()

        logger.info(f"  BO iter {i+1}/{n_iters}: obj={y_new:.4f}  best={best_y:.4f}")

    return best_z, best_y, x_hist, y_hist


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    logger.info(f"=== Synthetic pipeline test | device={DEVICE} | botorch_device={botorch_device} ===")

    # dataset
    samples = build_synthetic_dataset()
    dataset = FEMDataset(samples, voxel_resolution=RESOLUTION, use_sdf=False)

    split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Dataset: {len(train_ds)} train / {len(val_ds)} val")

    # VAE training
    vae = train_vae(train_loader, val_loader, epochs=EPOCHS)

    # BO
    logger.info("=== Bayesian Optimisation ===")
    best_z, best_obj, x_hist, y_hist = run_bo(vae)

    # decode best design
    vae.eval()
    with torch.no_grad():
        z_t = torch.from_numpy(best_z).float().unsqueeze(0).to(DEVICE)
        best_voxel = vae.decode(z_t).squeeze().cpu().numpy()

    # save results
    Path("optimization_results").mkdir(exist_ok=True)
    results = {
        "best_objective": float(best_obj),
        "best_z": best_z.tolist(),
        "y_history": [float(y) for y in y_hist],
        "best_voxel_occupancy": float(best_voxel.mean()),
        "best_voxel_shape": list(best_voxel.shape),
    }
    with open("optimization_results/synthetic_run.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n=== RESULTS ===")
    logger.info(f"Best objective (stress + 0.1*compliance):  {best_obj:.4f}")
    logger.info(f"Best voxel occupancy: {best_voxel.mean()*100:.1f}%")
    logger.info(f"Results saved to optimization_results/synthetic_run.json")
    logger.info(f"VAE checkpoint:       checkpoints/vae_best.pth")
    logger.info("=== Pipeline test PASSED ===")


if __name__ == "__main__":
    main()
