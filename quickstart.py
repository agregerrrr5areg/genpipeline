#!/usr/bin/env python3
"""
FEMbyGEN + PyTorch Generative Design Pipeline - Quick Start Example
====================================================================

This script demonstrates the complete workflow:
1. Setup FreeCAD parametric model
2. Extract FEM results
3. Train generative model
4. Optimize designs
5. Export best design

Author: Your Name
Date: 2025
"""

import argparse
import logging
from pathlib import Path
import json
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineConfig:
    def __init__(self, config_path: str = None):
        self.config = {
            'freecad_project_dir': './freecad_designs',
            'fem_data_output': './fem_data',
            'voxel_resolution': 32,
            'use_sdf': False,
            'latent_dim': 16,
            'batch_size': 8,
            'epochs': 100,
            'learning_rate': 1e-3,
            'beta_vae': 1.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_optimization_iterations': 50,
            'output_dir': './optimization_results'
        }

        if config_path and Path(config_path).exists():
            self.load(config_path)

    def load(self, config_path: str):
        with open(config_path, 'r') as f:
            loaded = json.load(f)
            self.config.update(loaded)
        logger.info(f"Loaded config from {config_path}")

    def save(self, config_path: str):
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {config_path}")

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value


def step_1_setup_freecad(config: PipelineConfig):
    """
    Step 1: Prepare FreeCAD parametric model
    ==========================================
    
    Manual steps in FreeCAD:
    1. Create parametric model using Part Design
    2. Define constraints with parameters (e.g., thickness, radius)
    3. Set up FEM analysis with materials, loads, constraints
    4. Save as master_design.FCStd
    
    Then use FEMbyGEN to generate variations:
    1. Open FEMbyGEN workbench
    2. Click "Initialize" to create Parameters spreadsheet
    3. Define parameter ranges (min/max values)
    4. Click "Generate" to create design variants
    5. Click "FEA" and run simulations
    """
    logger.info("=" * 60)
    logger.info("STEP 1: FreeCAD Setup (Manual)")
    logger.info("=" * 60)

    instructions = """
    1. In FreeCAD, create a parametric model:
       - Use Part Design workbench
       - Create sketch-based features (Pad, Pocket, etc.)
       - Constrain dimensions with parameters
    
    2. Set up FEM Analysis:
       - Switch to FEM workbench
       - Create Analysis container
       - Add loads, constraints, materials
       - Create mesh (Gmsh or Netgen)
    
    3. Create parametric variants with FEMbyGEN:
       - Install FEMbyGEN addon
       - Switch to FEMbyGEN workbench
       - Click "Initialize" button
       - Define parameter ranges in spreadsheet
       - Click "Generate" to create variations
       - Click "FEA" to run simulations
    
    4. Save results directory with all .FCStd files
    """
    print(instructions)

    project_dir = Path(config['freecad_project_dir'])
    project_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created project directory: {project_dir}")


def step_2_extract_fem_data(config: PipelineConfig):
    """
    Step 2: Extract FEM results and voxelize geometries
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Extract FEM Data")
    logger.info("=" * 60)

    try:
        from fem_data_pipeline import DataPipeline
    except ImportError:
        logger.error("fem_data_pipeline.py not found")
        return None

    pipeline = DataPipeline(
        freecad_project_dir=config['freecad_project_dir'],
        output_dir=config['fem_data_output']
    )

    try:
        train_loader, val_loader, dataset = pipeline.process_all_designs()
        logger.info(f"Processed {len(dataset)} designs")
        logger.info(f"Train batch size: {len(train_loader)}, Val batch size: {len(val_loader)}")
        return train_loader, val_loader, dataset
    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        return None


def step_3_train_vae(config: PipelineConfig, train_loader, val_loader):
    """
    Step 3: Train VAE generative model
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Train VAE Generative Model")
    logger.info("=" * 60)

    try:
        from vae_design_model import DesignVAE, VAETrainer
    except ImportError:
        logger.error("vae_design_model.py not found")
        return None

    device = config['device']
    logger.info(f"Using device: {device}")

    model = DesignVAE(
        input_shape=(config['voxel_resolution'],) * 3,
        latent_dim=config['latent_dim']
    )

    trainer = VAETrainer(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=config['learning_rate'],
        beta=config['beta_vae']
    )

    logger.info("Starting VAE training...")
    trainer.fit(epochs=config['epochs'])

    logger.info(f"Training complete. Best model saved to checkpoints/vae_best.pth")
    return model


def step_4_optimize_designs(config: PipelineConfig):
    """
    Step 4: Run Bayesian optimization to find optimal designs
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Bayesian Design Optimization")
    logger.info("=" * 60)

    try:
        from optimization_engine import DesignOptimizer, BridgeEvaluator
        from vae_design_model import DesignVAE
    except ImportError:
        logger.error("optimization_engine.py or vae_design_model.py not found")
        return None

    device = config['device']

    checkpoint = torch.load('checkpoints/vae_best.pth', map_location=device, weights_only=False)
    vae = DesignVAE(
        input_shape=(config['voxel_resolution'],) * 3,
        latent_dim=config['latent_dim']
    )
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)

    # Use BridgeEvaluator for WSL2 -> Windows communication
    fem_evaluator = BridgeEvaluator(
        freecad_path=config.config.get('freecad_path'),
        output_dir=str(Path(config['output_dir']) / 'fem')
    )

    optimizer = DesignOptimizer(
        vae,
        fem_evaluator,
        device=device,
        latent_dim=config['latent_dim'],
        sim_cfg={'geometry_type': config.config.get('geometry_type', 'cantilever')}
    )

    logger.info("Starting optimization...")
    best_z, best_obj = optimizer.run_optimization(
        n_iterations=config['n_optimization_iterations']
    )

    optimizer.save_results(config['output_dir'])

    logger.info(f"Optimization complete!")
    logger.info(f"Best design objective: {best_obj:.4f}")
    logger.info(f"Best latent vector: {best_z}")

    return best_z, best_obj


def step_5_export_design(config: PipelineConfig, best_z):
    """
    Step 5: Export optimized design
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Export Optimized Design")
    logger.info("=" * 60)

    try:
        from vae_design_model import DesignVAE
        from utils import VoxelConverter, ManufacturabilityConstraints
    except ImportError:
        logger.error("Required modules not found")
        return False

    device = config['device']

    checkpoint = torch.load('checkpoints/vae_best.pth', map_location=device, weights_only=False)
    vae = DesignVAE(
        input_shape=(config['voxel_resolution'],) * 3,
        latent_dim=config['latent_dim']
    )
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)

    with torch.no_grad():
        z_tensor = torch.from_numpy(best_z).float().unsqueeze(0).to(device)
        geometry = vae.decode(z_tensor)

    voxel_grid = geometry.squeeze().cpu().numpy()

    # Precision check: if the voxel grid is all zeros, Marching Cubes will fail.
    # This can happen if the BO finds a "void" design that it thinks has 0 mass.
    if np.max(voxel_grid) < 0.1:
        logger.error("Best design is empty (all zeros). Skipping export.")
        return False

    mfg_constraints = ManufacturabilityConstraints()
    voxel_grid = mfg_constraints.apply_constraints(voxel_grid)

    output_dir = Path(config['output_dir']) / 'exported_designs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load best_bbox if available for scale preservation
    best_bbox = None
    hist_path = Path(config['output_dir']) / "optimization_history.json"
    if hist_path.exists():
        try:
            with open(hist_path, 'r') as f:
                hist = json.load(f)
                best_bbox = hist.get("best_bbox")
        except Exception as e:
            logger.warning(f"Could not load best_bbox from history: {e}")

    mesh_data = VoxelConverter.voxel_to_mesh(
        voxel_grid,
        voxel_size=1.0 / config['voxel_resolution'],
        bbox=best_bbox
    )

    if mesh_data:
        try:
            import trimesh
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces']
            )
            output_path = output_dir / 'optimized_design.stl'
            mesh.export(str(output_path))
            logger.info(f"Exported STL: {output_path}")

            output_path_obj = output_dir / 'optimized_design.obj'
            mesh.export(str(output_path_obj))
            logger.info(f"Exported OBJ: {output_path_obj}")

            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    else:
        logger.error("Failed to generate mesh from voxels")
        return False


def run_full_pipeline(config: PipelineConfig):
    """
    Execute complete pipeline
    """
    logger.info("\n" + "=" * 60)
    logger.info("FEMbyGEN + PyTorch Generative Design Pipeline")
    logger.info("=" * 60 + "\n")

    step_1_setup_freecad(config)
    input("Press Enter after completing FreeCAD setup...")

    data_result = step_2_extract_fem_data(config)
    if data_result is None:
        logger.error("Data extraction failed")
        return False

    train_loader, val_loader, dataset = data_result

    model = step_3_train_vae(config, train_loader, val_loader)
    if model is None:
        logger.error("VAE training failed")
        return False

    opt_result = step_4_optimize_designs(config)
    if opt_result is None:
        logger.error("Optimization failed")
        return False

    best_z, best_obj = opt_result

    success = step_5_export_design(config, best_z)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {config['output_dir']}")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEMbyGEN + PyTorch Generative Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quickstart.py --step 1
  python quickstart.py --config config.json
  python quickstart.py --all
        """
    )

    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Run specific pipeline step')
    parser.add_argument('--config', type=str, default='pipeline_config.json',
                       help='Config file path')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--freecad-dir', type=str,
                       help='FreeCAD project directory')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory')
    parser.add_argument('--epochs', type=int,
                       help='Training epochs')
    parser.add_argument('--n-iter', type=int,
                       help='Optimization iterations')

    args = parser.parse_args()

    config = PipelineConfig(args.config if Path(args.config).exists() else None)

    if args.freecad_dir:
        config['freecad_project_dir'] = args.freecad_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['epochs'] = args.epochs
    if args.n_iter:
        config['n_optimization_iterations'] = args.n_iter

    config.save('pipeline_config.json')

    if args.all:
        run_full_pipeline(config)
    elif args.step == 1:
        step_1_setup_freecad(config)
    elif args.step == 2:
        data_result = step_2_extract_fem_data(config)
    elif args.step == 3:
        checkpoint = torch.load('fem_data/fem_dataset.pt', weights_only=False)
        model = step_3_train_vae(config, checkpoint['train_loader'], checkpoint['val_loader'])
    elif args.step == 4:
        best_z, best_obj = step_4_optimize_designs(config)
        np.save(Path(config['output_dir']) / 'best_z.npy', best_z)
    elif args.step == 5:
        import numpy as np
        best_z = np.load('optimization_results/best_z.npy')
        step_5_export_design(config, best_z)
    else:
        logger.info("Run with --all for complete pipeline or --step N for specific step")
        parser.print_help()
