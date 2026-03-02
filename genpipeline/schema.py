from pydantic import BaseModel, Field, ConfigDict, field_validator
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
    latent_z: Optional[List[float]] = Field(None, description="Latent space representation (z-vector)")

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

class OptimisationConfig(BaseModel):
    """Configuration for the Bayesian Optimisation loop."""
    acquisition_function: str = "UCB"
    beta: float = 0.1
    use_botorch: bool = True
    num_restarts: int = 10
    raw_samples: int = 512
    max_iterations: int = 100
    parallel_evaluations: int = 4

class PerformanceWeights(BaseModel):
    """Weights for the multi-objective optimisation."""
    stress: float = 1.0
    compliance: float = 0.1
    mass: float = 0.05

class ManufacturingConstraints(BaseModel):
    """Physical constraints for manufacturability."""
    min_feature_size_mm: float = 5.0
    max_overhang_angle_deg: float = 45.0
    min_volume_fraction: float = 0.15

class PipelineConfig(BaseModel):
    """Main configuration for the entire generative design pipeline."""
    freecad_project_dir: str = "./freecad_designs"
    fem_data_output: str = "./fem/data"
    voxel_resolution: int = 64
    use_sdf: bool = False
    latent_dim: int = 32
    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 0.0003
    beta_vae: float = 1.0
    pos_weight: float = 30.0
    device: str = "cuda"
    n_optimisation_iterations: int = 1000
    output_dir: str = "./optimisation_results"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    seed: int = 42
    freecad_path: Optional[str] = "/mnt/c/Users/PC-PC/AppData/Local/Programs/FreeCAD 1.0"
    
    optimisation: OptimisationConfig = Field(default_factory=OptimisationConfig)
    performance_weights: PerformanceWeights = Field(default_factory=PerformanceWeights)
    manufacturing_constraints: ManufacturingConstraints = Field(default_factory=ManufacturingConstraints)

    @field_validator("beta_vae")
    @classmethod
    def check_beta_vae(cls, v: float) -> float:
        if not (0.0 < v <= 10.0):
            raise ValueError(f"beta_vae must be between 0.0 and 10.0, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"batch_size must be positive, got {v}")
        return v

    @field_validator("voxel_resolution")
    @classmethod
    def check_voxel_res(cls, v: int) -> int:
        if v % 16 != 0:
            raise ValueError(f"voxel_resolution must be a multiple of 16, got {v}")
        return v

    @property
    def input_shape(self) -> List[int]:
        return [self.voxel_resolution, self.voxel_resolution, self.voxel_resolution]
