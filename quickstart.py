#!/ suppressed
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from genpipeline.config import load_config
from genpipeline.trainer import train_vae
from genpipeline.optimiser import run_optimisation
from genpipeline.fem.data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CMD CLI like claude code e.i interactive ui within terminal

def main():
    parser = argparse.ArgumentParser(description="GenPipeline Quickstart CLI")
    parser.add_argument('--step', type=int, choices=[0, 1, 2, 3, 4, 5], help="Execute a specific step")
    parser.add_argument('--all', action='store_true', help="Execute all steps (1-5)")
    parser.add_argument('--config', type=str, default='pipeline_config.json', help="Path to config file")
    parser.add_argument('--n-samples', type=int, help="Override n_samples for data generation")
    parser.add_argument('--n-iter', type=int, help="Override optimisation iterations")
    parser.add_argument('--epochs', type=int, help="Override VAE training epochs")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Overrides
    if args.n_iter: config.n_optimisation_iterations = args.n_iter
    if args.epochs: config.epochs = args.epochs

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
        from torch.utils.data import DataLoader
        train_l = DataLoader(checkpoint['train_loader'].dataset, batch_size=config.batch_size, shuffle=True)
        val_l = DataLoader(checkpoint['val_loader'].dataset, batch_size=config.batch_size)
        
        train_vae(config, train_l, val_l)

    if args.all or args.step == 4:
        logger.info("STEP 4: Design Optimisation Loop")
        best_z, _ = run_optimisation(config)
        if best_z is not None:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(config.output_dir) / 'best_z.npy', best_z)

    if args.all or args.step == 5:
        logger.info("STEP 5: Export (Placeholder - see legacy quickstart.py)")
        # Export logic involves complex trimesh/mfg logic, keep in quickstart or move to package later

if __name__ == "__main__":
    main()
