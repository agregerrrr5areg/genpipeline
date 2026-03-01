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


def validate_config(cfg: dict) -> list:
    """
    Validate pipeline configuration dict.

    Returns a list of human-readable error strings.  An empty list means the
    configuration is valid.
    """
    errors = []

    required_keys = {
        'voxel_resolution': int,
        'latent_dim': int,
        'batch_size': int,
        'epochs': int,
        'learning_rate': float,
        'beta_vae': float,
        'n_optimisation_iterations': int,
    }
    for key, expected_type in required_keys.items():
        if key not in cfg:
            errors.append(f"Missing required key: '{key}'")
        elif not isinstance(cfg[key], (int, float)):
            errors.append(f"'{key}' must be a number, got {type(cfg[key]).__name__}")

    if 'beta_vae' in cfg:
        v = cfg['beta_vae']
        if not (0 < v <= 10):
            errors.append(f"'beta_vae'={v} out of range (0, 10]")

    if 'batch_size' in cfg and cfg.get('batch_size', 1) <= 0:
        errors.append(f"'batch_size' must be > 0, got {cfg['batch_size']}")

    if 'epochs' in cfg and cfg.get('epochs', 1) <= 0:
        errors.append(f"'epochs' must be > 0, got {cfg['epochs']}")

    if 'voxel_resolution' in cfg:
        res = cfg['voxel_resolution']
        if res % 16 != 0:
            errors.append(
                f"'voxel_resolution'={res} must be divisible by 16 "
                f"(VAE uses 4× stride-2 conv layers)"
            )

    return errors


class PipelineConfig:
    def __init__(self, config_path: str = None):
        self.config = {
            'freecad_project_dir': './freecad_designs',
            'fem_data_output': './fem_data',
            'voxel_resolution': 64,
            'input_shape': [64, 64, 64],
            'use_sdf': False,
            'latent_dim': 32,
            'batch_size': 32,
            'epochs': 300,
            'learning_rate': 0.0003,
            'beta_vae': 1.0,
            'pos_weight': 30.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_optimisation_iterations': 1000,
            'output_dir': './optimisation_results'
        }

        if config_path and Path(config_path).exists():
            self.load(config_path)

    def load(self, config_path: str):
        with open(config_path, 'r') as f:
            loaded = json.load(f)
            self.config.update(loaded)
        # Keep input_shape in sync with voxel_resolution
        if 'voxel_resolution' in loaded and 'input_shape' not in loaded:
            res = self.config['voxel_resolution']
            self.config['input_shape'] = [res, res, res]
        logger.info(f"Loaded config from {config_path}")
        errors = validate_config(self.config)
        for err in errors:
            logger.warning(f"Config validation: {err}")

    def save(self, config_path: str):
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {config_path}")

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value


def step_0_generate_topo_data(config: PipelineConfig, n_samples: int = 200):
    logger.info("=" * 60)
    logger.info(f"STEP 0: Generate Topology Data ({n_samples} samples)")
    logger.info("=" * 60)
    try:
        from topology.topo_data_gen import TopoDataGenerator
        import subprocess
        import sys
    except ImportError:
        logger.error("Topology generator modules not found")
        return False

    gen = TopoDataGenerator(output_dir=config['fem_data_output'])
    gen.generate(n_samples=n_samples)
    
    logger.info("Rebuilding dataset.pt...")
    cmd = [sys.executable, "rebuild_dataset.py", 
           "--fem-dir", config['fem_data_output'],
           "--resolution", str(config['voxel_resolution'])]
    subprocess.run(cmd, check=True)
    return True


def step_1_setup_freecad(config: PipelineConfig):
    logger.info("=" * 60 + "\nSTEP 1: Setup FreeCAD Environment\n" + "=" * 60)
    return True


def step_2_extract_fem_data(config: PipelineConfig):
    logger.info("=" * 60 + "\nSTEP 2: Run FEM Data Pipeline\n" + "=" * 60)
    try:
        from fem.data_pipeline import DataPipeline
    except ImportError:
        logger.error("DataPipeline module not found")
        return None
    pipeline = DataPipeline(config['freecad_project_dir'], config['fem_data_output'])
    return pipeline.process_all_designs()


def step_3_train_vae(config: PipelineConfig, train_loader, val_loader):
    logger.info("=" * 60 + "\nSTEP 3: Train Design VAE\n" + "=" * 60)
    try:
        from vae_design_model import DesignVAE, VAETrainer
    except ImportError: return None
    model = DesignVAE(input_shape=tuple(config['input_shape']), latent_dim=config['latent_dim'])
    trainer = VAETrainer(model, train_loader, val_loader, device=config['device'], epochs=config['epochs'])
    trainer.fit(epochs=config['epochs'])
    return model


def step_4_optimize_designs(config: PipelineConfig):
    logger.info("=" * 60 + "\nSTEP 4: Design Optimisation Loop\n" + "=" * 60)
    try:
        from optimisation_engine import DesignOptimizer
        from vae_design_model import DesignVAE
        from fem.voxel_fem import VoxelFEMEvaluator
    except ImportError: return None, None
    vae = DesignVAE(input_shape=tuple(config['input_shape']), latent_dim=config['latent_dim'])
    checkpoint = torch.load('checkpoints/vae_best.pth', map_location=config['device'], weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    evaluator = VoxelFEMEvaluator(resolution=config['voxel_resolution'])
    optimizer = DesignOptimizer(vae, evaluator, device=config['device'], latent_dim=config['latent_dim'])
    return optimizer.optimize(n_iterations=config['n_optimisation_iterations'])


def step_5_export_design(config: PipelineConfig, best_z=None):
    logger.info("=" * 60 + "\nSTEP 5: Export Pareto-Optimal Designs\n" + "=" * 60)
    try:
        from vae_design_model import DesignVAE
        from pipeline_utils import VoxelConverter, ManufacturabilityConstraints
    except ImportError: return False

    device = config['device']
    checkpoint = torch.load('checkpoints/vae_best.pth', map_location=device, weights_only=False)
    vae = DesignVAE(input_shape=tuple(config.config.get('input_shape', [64,64,64])), latent_dim=config['latent_dim'])
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)

    hist_path = Path(config['output_dir']) / "optimisation_history.json"
    if not hist_path.exists(): return False
    
    with open(hist_path, 'r') as f: hist = json.load(f)
    pareto_front = hist.get("pareto_front", [])
    if not pareto_front:
        designs_to_export = [{"name": "best_balanced", "z": best_z}] if best_z is not None else []
    else:
        strongest = min(pareto_front, key=lambda x: x['stress'])
        lightest  = min(pareto_front, key=lambda x: x['mass'])
        balanced  = min(pareto_front, key=lambda x: (x['stress'] + 100 * x['mass']))
        designs_to_export = [{"name": "pareto_strongest", "z": strongest['latent_z']},
                             {"name": "pareto_lightest",  "z": lightest['latent_z']},
                             {"name": "pareto_balanced",  "z": balanced['latent_z']}]

    output_dir = Path(config['output_dir']) / 'exported_designs'
    output_dir.mkdir(parents=True, exist_ok=True)
    mfg, res = ManufacturabilityConstraints(config=config.config), config['voxel_resolution']

    for d in designs_to_export:
        z = torch.tensor(d['z']).float().unsqueeze(0).to(device)
        with torch.no_grad(): voxels = vae.decode(z).squeeze().cpu().numpy()
        voxels = mfg.apply_constraints(voxels, voxel_size=1.0/res)
        if np.max(voxels) < 0.5: continue
        mesh = VoxelConverter.voxel_to_mesh(voxels, voxel_size=1.0/res, bbox=hist.get("best_bbox"))
        if mesh:
            import trimesh
            trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces']).export(str(output_dir / f"{d['name']}.stl"))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--topo-data', action='store_true', help='Bootstrap by generating topology data')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of topo samples to generate')
    parser.add_argument('--n-iter', type=int)
    parser.add_argument('--epochs', type=int, help='Override epochs for step 3')
    parser.add_argument('--batch-size', type=int, help='Override batch_size for step 3')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override output_dir for step 5 export')
    args = parser.parse_args()

    config = PipelineConfig('pipeline_config.json')
    if args.n_iter: config['n_optimisation_iterations'] = args.n_iter
    if args.epochs: config['epochs'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.results_dir: config['output_dir'] = args.results_dir

    if args.topo_data or args.step == 0:
        step_0_generate_topo_data(config, n_samples=args.n_samples)
        if args.step == 0: exit(0)

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
        np.save('optimisation_results/best_z.npy', best_z)
    elif args.step == 5:
        z = np.load('optimisation_results/best_z.npy') if Path('optimisation_results/best_z.npy').exists() else None
        step_5_export_design(config, z)
