#! suppressed
import argparse
import logging
import torch
import numpy as np
import os
from pathlib import Path
from genpipeline.config import load_config
from genpipeline.trainer import train_vae
from genpipeline.optimiser import run_optimisation
from genpipeline.fem.data_pipeline import DataPipeline

# Limit VRAM to prevent system lag (use only 10GB of 16GB available)
if torch.cuda.is_available():
    TOTAL_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
    MAX_MEM_GB = min(10, TOTAL_MEM - 2)  # Leave 2GB for system
    torch.cuda.set_per_process_memory_fraction(MAX_MEM_GB / TOTAL_MEM)
    print(f"VRAM limited to {MAX_MEM_GB:.1f}GB of {TOTAL_MEM:.1f}GB")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GenPipeline Quickstart CLI")
    parser.add_argument(
        "--step", type=int, choices=[0, 1, 2, 3, 4, 5], help="Execute a specific step"
    )
    parser.add_argument("--all", action="store_true", help="Execute all steps (1-5)")
    parser.add_argument(
        "--config", type=str, default="pipeline_config.json", help="Path to config file"
    )
    parser.add_argument(
        "--n-samples", type=int, help="Override n_samples for data generation"
    )
    parser.add_argument("--n-iter", type=int, help="Override optimisation iterations")
    parser.add_argument(
        "--q", type=int, help="Parallel FEM workers per BO round (default: cpu_count)"
    )
    parser.add_argument("--epochs", type=int, help="Override VAE training epochs")
    parser.add_argument(
        "--topo-data",
        action="store_true",
        help="Generate topology data and rebuild dataset",
    )
    parser.add_argument(
        "--topo-refine",
        action="store_true",
        help="Perform topology refinement in optimisation loop",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # Overrides
    if args.n_iter:
        config.n_optimisation_iterations = args.n_iter
    if args.epochs:
        config.epochs = args.epochs

    if args.topo_data:
        logger.info("Generating topology optimisation samples...")
        from genpipeline.topology.topo_data_gen import TopoDataGenerator
        import subprocess

        n_samples = args.n_samples if args.n_samples else 100
        generator = TopoDataGenerator(output_dir=config.fem_data_output)
        generator.generate(n_samples=n_samples)

        logger.info("Rebuilding dataset.pt from new samples...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.resolve())
        import sys

        subprocess.run(
            [
                sys.executable,
                "scripts/rebuild_dataset.py",
                "--fem-dir",
                config.fem_data_output,
                "--resolution",
                str(config.voxel_resolution),
            ],
            check=True,
            env=env,
        )

    if args.step == 0:
        logger.info("Step 0: Legacy generation not yet moved to genpipeline package")
        # Could call legacy logic here if needed
        return

    if args.all or args.step == 2:
        logger.info("STEP 2: Run FEM Data Pipeline")
        pipeline = DataPipeline(config.freecad_project_dir, config.fem_data_output)
        pipeline.process_all_designs()

    if args.all or args.step == 3:
        logger.info("STEP 3: Train Design VAE")
        # Load dataset
        dataset_path = Path(config.fem_data_output) / "fem_dataset.pt"
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}. Run step 2 first.")
            return

        checkpoint = torch.load(dataset_path, weights_only=False)
        from torch.utils.data import DataLoader, random_split

        ds = checkpoint["dataset"]

        # Split into train/val
        n_val = max(1, int(len(ds) * 0.2))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])

        train_l = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_l = DataLoader(val_ds, batch_size=config.batch_size)

        train_vae(config, train_l, val_l)

    if args.all or args.step == 4:
        logger.info("STEP 4: Design Optimisation Loop")
        best_z, _ = run_optimisation(config, topo_refine=args.topo_refine, q=args.q)
        if best_z is not None:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(config.output_dir) / "best_z.npy", best_z)

    if args.all or args.step == 5:
        logger.info("STEP 5: Export (Placeholder - see legacy quickstart.py)")
        # Export logic involves complex trimesh/mfg logic, keep in quickstart or move to package later


if __name__ == "__main__":
    main()
