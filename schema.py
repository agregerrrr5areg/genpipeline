from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List, Any
import numpy as np

class DesignParameters(BaseModel):
    """Structured parameters for a design variant."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    h_mm: float = Field(..., description="Primary height/dimension in mm")
    r_mm: float = Field(..., description="Secondary radius/thickness in mm")
    geometry_type: str = Field("cantilever", description="Type of geometry (cantilever, lbracket, etc.)")
    material_name: str = Field("Plastic_ABS", description="Material identifier")
    material_cfg: Optional[Dict[str, Any]] = Field(None, description="Detailed material properties (E, poisson)")

class FEMResult(BaseModel):
    """Structured results from a Finite Element Method simulation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    stress_max: float = Field(..., description="Maximum Von Mises stress in MPa")
    stress_mean: Optional[float] = Field(None, description="Mean stress in MPa")
    compliance: float = Field(..., description="Structural compliance (inverse of stiffness)")
    mass: float = Field(..., description="Total mass of the design in kg")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box dimensions (xmin, xmax, etc.)")
    success: bool = Field(True, description="Whether the simulation completed successfully")

class OptimizationSample(BaseModel):
    """A single sample in the optimization history."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    parameters: DesignParameters
    result: FEMResult
    latent_z: Optional[List[float]] = Field(None, description="Latent space representation (z-vector)")
    timestamp: Optional[float] = Field(None, description="Unix timestamp of evaluation")

class DesignSample(BaseModel):
    """Container for a design with its associated voxel representation and metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    geometry_path: str
    metrics: FEMResult
    parameters: DesignParameters
    voxel_grid: Optional[np.ndarray] = Field(None, description="3D density grid")
