#!/usr/bin/env python3
"""
SIMP Data Augmentation Script
Generates topology optimization samples and appends to existing FEM dataset.
"""

import torch
import numpy as np
from pathlib import Path
import logging
import sys
import subprocess

sys.path.insert(0, "/home/genpipeline")

from genpipeline.topology.topo_data_gen import TopoDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simp_augmentation(n_samples=500, output_dir="./fem_data_augmented"):
    """Generate SIMP-based training data to augment FreeCAD dataset."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SIMP Data Augmentation: Generating {n_samples} samples")
    logger.info(f"{'=' * 70}\n")

    # Create generator
    generator = TopoDataGenerator(
        output_dir=output_dir,
        n_workers=4,  # Use 4 parallel workers
    )

    # Generate samples
    generator.generate_scaled(n_samples=n_samples, batch_size=100)

    logger.info(f"\nGenerated {n_samples} SIMP samples in {output_dir}")
    logger.info("Next step: Rebuild dataset.pt with new samples")

    return output_dir


def rebuild_dataset(fem_dir, resolution=64):
    """Rebuild dataset.pt from all FEM samples."""
    logger.info(f"\nRebuilding dataset.pt from {fem_dir}...")

    env = {"PYTHONPATH": "/home/genpipeline"}
    result = subprocess.run(
        [
            sys.executable,
            "/home/genpipeline/scripts/rebuild_dataset.py",
            "--fem-dir",
            fem_dir,
            "--resolution",
            str(resolution),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Dataset rebuild failed: {result.stderr}")
        return False

    logger.info("Dataset rebuilt successfully!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Number of SIMP samples"
    )
    parser.add_argument("--output-dir", type=str, default="./fem_data_simp")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild dataset after generation"
    )
    args = parser.parse_args()

    # Generate SIMP data
    output_dir = run_simp_augmentation(args.n_samples, args.output_dir)

    # Optionally rebuild dataset
    if args.rebuild:
        rebuild_dataset(output_dir, resolution=64)
