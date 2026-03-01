import json
import logging
from pathlib import Path
from typing import Optional
from .schema import PipelineConfig

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = "pipeline_config.json") -> PipelineConfig:
    """Load configuration from a JSON file into the Pydantic model."""
    default_config = PipelineConfig()
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                # Map potential old naming conventions to new model
                if "n_optimization_iterations" in data:
                    data["n_optimisation_iterations"] = data.pop("n_optimization_iterations")
                if "optimization" in data:
                    data["optimisation"] = data.pop("optimization")
                
                config = PipelineConfig(**data)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}. Using defaults.")
    
    logger.info("Using default configuration.")
    return default_config

def save_config(config: PipelineConfig, config_path: str = "pipeline_config.json"):
    """Save the configuration model to a JSON file."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)
    logger.info(f"Saved configuration to {config_path}")
