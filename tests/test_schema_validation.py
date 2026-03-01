import pytest
import numpy as np
import torch
from schema import DesignParameters, FEMResult, PipelineConfig
from pydantic import ValidationError

def test_design_parameters_validation():
    """Test that DesignParameters enforces types and required fields."""
    # Valid
    params = DesignParameters(h_mm=15.0, r_mm=5.0)
    assert params.h_mm == 15.0
    
    # Invalid type
    with pytest.raises(ValidationError):
        DesignParameters(h_mm="high", r_mm=5.0)

def test_fem_result_validation():
    """Test that FEMResult catches invalid simulation outputs."""
    # Valid
    res = FEMResult(stress_max=50.0, compliance=0.1, mass=1.2)
    assert res.success is True
    
    # Missing required field
    with pytest.raises(ValidationError):
        FEMResult(stress_max=50.0)

def test_pipeline_config_defaults():
    """Test that PipelineConfig populates defaults and validated types."""
    cfg = PipelineConfig()
    assert cfg.voxel_resolution == 64
    assert cfg.device in ["cuda", "cpu"]
    assert len(cfg.input_shape) == 3
    assert cfg.input_shape[0] == 64

def test_config_logic():
    """Test derived properties in PipelineConfig."""
    cfg = PipelineConfig(voxel_resolution=32)
    assert cfg.input_shape == [32, 32, 32]

if __name__ == "__main__":
    pytest.main([__file__])
