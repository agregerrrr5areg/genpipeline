# tests/test_config_validation.py
"""Tests for validate_config() in quickstart.py."""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quickstart import validate_config, PipelineConfig


_VALID_CFG = {
    'voxel_resolution': 64,
    'latent_dim': 32,
    'batch_size': 8,
    'epochs': 10,
    'learning_rate': 3e-4,
    'beta_vae': 1.0,
    'n_optimization_iterations': 50,
}


class TestValidateConfig:
    def test_valid_config_returns_no_errors(self):
        assert validate_config(_VALID_CFG) == []

    def test_missing_required_key(self):
        cfg = {k: v for k, v in _VALID_CFG.items() if k != 'batch_size'}
        errors = validate_config(cfg)
        assert any('batch_size' in e for e in errors)

    def test_beta_vae_zero_is_invalid(self):
        cfg = {**_VALID_CFG, 'beta_vae': 0.0}
        errors = validate_config(cfg)
        assert any('beta_vae' in e for e in errors)

    def test_beta_vae_above_10_is_invalid(self):
        cfg = {**_VALID_CFG, 'beta_vae': 11.0}
        errors = validate_config(cfg)
        assert any('beta_vae' in e for e in errors)

    def test_beta_vae_valid_range(self):
        for v in (0.01, 1.0, 5.0, 10.0):
            cfg = {**_VALID_CFG, 'beta_vae': v}
            assert validate_config(cfg) == [], f"Expected valid for beta_vae={v}"

    def test_batch_size_zero_is_invalid(self):
        cfg = {**_VALID_CFG, 'batch_size': 0}
        errors = validate_config(cfg)
        assert any('batch_size' in e for e in errors)

    def test_voxel_resolution_not_divisible_by_16(self):
        cfg = {**_VALID_CFG, 'voxel_resolution': 32}   # 32 % 16 == 0, should be valid
        assert validate_config(cfg) == []

        cfg = {**_VALID_CFG, 'voxel_resolution': 40}   # 40 % 16 != 0
        errors = validate_config(cfg)
        assert any('voxel_resolution' in e for e in errors)

    def test_voxel_resolution_divisible_by_16(self):
        for res in (16, 32, 48, 64, 128):
            cfg = {**_VALID_CFG, 'voxel_resolution': res}
            assert validate_config(cfg) == [], f"Expected valid for voxel_resolution={res}"

    def test_multiple_errors_reported(self):
        cfg = {**_VALID_CFG, 'beta_vae': 0.0, 'batch_size': 0}
        errors = validate_config(cfg)
        assert len(errors) >= 2


class TestPipelineConfigDefaults:
    def test_default_beta_vae_is_1(self):
        cfg = PipelineConfig()
        assert cfg['beta_vae'] == 1.0

    def test_default_input_shape_matches_voxel_resolution(self):
        cfg = PipelineConfig()
        res = cfg['voxel_resolution']
        assert cfg.config['input_shape'] == [res, res, res]

    def test_pipeline_config_has_pos_weight(self):
        cfg = PipelineConfig()
        assert 'pos_weight' in cfg.config
        assert cfg.config['pos_weight'] == 30.0
