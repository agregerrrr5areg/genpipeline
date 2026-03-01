import pytest
from pathlib import Path
from genpipeline.fem.data_pipeline import FEMResultParser

def test_provenance_check_rejected(tmp_path):
    """Test that disconnected synthetic files are rejected."""
    parser = FEMResultParser(str(tmp_path))
    
    # Create a lone STL file with no .step or .FCStd context
    synthetic_mesh = tmp_path / "synthetic_bracket.stl"
    synthetic_mesh.touch()
    
    assert parser.validate_data_provenance(str(synthetic_mesh)) is False

def test_provenance_check_accepted(tmp_path):
    """Test that files with physical context are accepted."""
    parser = FEMResultParser(str(tmp_path))
    
    # Create an STL and a corresponding STEP file
    mesh_path = tmp_path / "real_bracket.stl"
    step_path = tmp_path / "real_bracket.step"
    mesh_path.touch()
    step_path.touch()
    
    assert parser.validate_data_provenance(str(mesh_path)) is True

def test_provenance_check_fcstd_accepted(tmp_path):
    """Test that FreeCAD context is accepted."""
    parser = FEMResultParser(str(tmp_path))
    
    mesh_path = tmp_path / "fcstd_bracket.stl"
    fcstd_path = tmp_path / "fcstd_bracket.FCStd"
    mesh_path.touch()
    fcstd_path.touch()
    
    assert parser.validate_data_provenance(str(mesh_path)) is True

if __name__ == "__main__":
    pytest.main([__file__])
