#!/usr/bin/env python3
"""
FEMbyGEN + PyTorch Generative Design Pipeline - Quick Start Example
====================================================================
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
            'voxel_resolution': 64,
            'use_sdf': False,
            'latent_dim': 32,
            'batch_size': 32,
            'epochs': 300,
            'learning_rate': 0.0003,
            'beta_vae': 0.05,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_optimization_iterations': 1000,
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
    logger.info("=" * 60)
    logger.info("STEP 1: FreeCAD Setup (Manual)")
    logger.info("=" * 60)
    project_dir = Path(config['freecad_project_dir'])
    project_dir.mkdir(parents=True, exist_ok=True)


def step_2_extract_fem_data(config: PipelineConfig):
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
    return pipeline.process_all_designs()


def step_3_train_vae(config: PipelineConfig, train_loader, val_loader):
    logger.info("=" * 60)
    logger.info("STEP 3: Train VAE Generative Model")
    logger.info("=" * 60)
    try:
        from vae_design_model import DesignVAE, VAETrainer
    except ImportError:
        logger.error("vae_design_model.py not found")
        return None
    device = config['device']
    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=config['latent_dim'])
    trainer = VAETrainer(model, train_loader, val_loader, device=device,
                        lr=config['learning_rate'], beta=config['beta_vae'],
                        epochs=config['epochs'])
    trainer.fit(epochs=config['epochs'])
    return model


def step_4_optimize_designs(config: PipelineConfig):
    logger.info("=" * 60)
    logger.info("STEP 4: Multi-Objective Bayesian Optimization")
    logger.info("=" * 60)
    try:
        from optimization_engine import DesignOptimizer, BridgeEvaluator
        from vae_design_model import DesignVAE
    except ImportError:
        logger.error("Optimization modules not found")
        return None

    device = config['device']
    checkpoint = torch.load('checkpoints/vae_best.pth', map_location=device, weights_only=False)
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=config['latent_dim'])
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)

    q = config.config.get('optimization', {}).get('parallel_evaluations', 4)
    fem_evaluator = BridgeEvaluator(freecad_path=config.config.get('freecad_path'), output_dir=str(Path(config['output_dir']) / 'fem'), n_workers=q)
    
    optimizer = DesignOptimizer(vae, fem_evaluator, device=device, latent_dim=config['latent_dim'], 
                                sim_cfg={'geometry_type': config.config.get('geometry_type', 'cantilever'),
                                         'w_stress': config.config.get('performance_weights', {}).get('stress', 1.0),
                                         'w_mass': config.config.get('performance_weights', {}).get('mass', 0.01),
                                         'max_stress_mpa': config.config.get('performance_targets', {}).get('max_stress_mpa', 40.0)})

    n_iter = max(1, config['n_optimization_iterations'] // q)
    logger.info(f"Starting MOBO: {n_iter} rounds of {q} designs...")
    best_z, best_y = optimizer.run_optimization(n_iterations=n_iter, q=q)
    optimizer.save_results(config['output_dir'])
    
    return best_z, best_y


def step_5_export_design(config: PipelineConfig, best_z=None):
    logger.info("=" * 60)
    logger.info("STEP 5: Export Pareto-Optimal Designs")
    logger.info("=" * 60)
    try:
        from vae_design_model import DesignVAE
        from utils import VoxelConverter, ManufacturabilityConstraints
    except ImportError:
        logger.error("Modules not found")
        return False

    device = config['device']
    checkpoint = torch.load('checkpoints/vae_best.pth', map_location=device, weights_only=False)
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=config['latent_dim'])
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)

    # Load Pareto Front from history
    hist_path = Path(config['output_dir']) / "optimization_history.json"
    if not hist_path.exists():
        logger.error("No optimization history found. Run Step 4 first.")
        return False
    
    with open(hist_path, 'r') as f:
        hist = json.load(f)
    
    pareto_front = hist.get("pareto_front", [])
    if not pareto_front:
        logger.warning("No Pareto front found. Exporting single best.")
        designs_to_export = [{"name": "best_balanced", "z": best_z}] if best_z is not None else []
    else:
        # Find Strongest, Lightest, and Balanced
        strongest = min(pareto_front, key=lambda x: x['stress'])
        lightest  = min(pareto_front, key=lambda x: x['mass'])
        balanced  = min(pareto_front, key=lambda x: (x['stress'] + 100 * x['mass']))
        designs_to_export = [
            {"name": "pareto_strongest", "z": strongest['latent_z']},
            {"name": "pareto_lightest",  "z": lightest['latent_z']},
            {"name": "pareto_balanced",  "z": balanced['latent_z']}
        ]

    output_dir = Path(config['output_dir']) / 'exported_designs'
    output_dir.mkdir(parents=True, exist_ok=True)
    mfg = ManufacturabilityConstraints(config=config.config)
    res = config['voxel_resolution']

    for d in designs_to_export:
        z = torch.tensor(d['z']).float().unsqueeze(0).to(device)
        with torch.no_grad():
            voxels = vae.decode(z).squeeze().cpu().numpy()
        
        voxels = mfg.apply_constraints(voxels, voxel_size=1.0/res)
        if np.max(voxels) < 0.5: continue
        
        mesh = VoxelConverter.voxel_to_mesh(voxels, voxel_size=1.0/res, bbox=hist.get("best_bbox"))
        if mesh:
            import trimesh
            m = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
            m.export(str(output_dir / f"{d['name']}.stl"))
            logger.info(f"Exported {d['name']}.stl")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n-iter', type=int)
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override output_dir for step 5 export')
    args = parser.parse_args()

    config = PipelineConfig('pipeline_config.json')
    if args.n_iter: config['n_optimization_iterations'] = args.n_iter
    if args.results_dir: config['output_dir'] = args.results_dir

    if args.all:
        step_1_setup_freecad(config)
        data = step_2_extract_fem_data(config)
        model = step_3_train_vae(config, data[0], data[1])
        z, _ = step_4_optimize_designs(config)
        step_5_export_design(config, z)
    elif args.step == 3:
        checkpoint = torch.load(Path(config['fem_data_output']) / "fem_dataset.pt", weights_only=False)
        from torch.utils.data import DataLoader
        train_l = DataLoader(checkpoint['train_loader'].dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
        val_l = DataLoader(checkpoint['val_loader'].dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=4, persistent_workers=True)
        step_3_train_vae(config, train_l, val_l)
    elif args.step == 4:
        best_z, _ = step_4_optimize_designs(config)
        np.save('optimization_results/best_z.npy', best_z)
    elif args.step == 5:
        z = np.load('optimization_results/best_z.npy') if Path('optimization_results/best_z.npy').exists() else None
        step_5_export_design(config, z)
