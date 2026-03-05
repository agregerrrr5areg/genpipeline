# AGENTS.md

This file provides guidance to agentic coding agents working in this repository.

## Build/Test Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install PyTorch for Blackwell (RTX 50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/test_vae_model.py

# Run single test function
pytest tests/test_vae_model.py::TestDesignVAEShapes::test_param_count

# Run with CUDA (required for most tests)
CUDA_VISIBLE_DEVICES=0 pytest tests/

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=genpipeline tests/
```

### Linting and Formatting
```bash
# Check code style (if black is available)
black --check genpipeline/ tests/

# Format code
black genpipeline/ tests/

# Type checking (if mypy is available)
mypy genpipeline/
```

### Model Training/Evaluation
```bash
# Train VAE
python vae_design_model.py --dataset-path ./fem/data/fem_dataset.pt --epochs 100

# Evaluate VAE
python eval_vae.py --model-checkpoint checkpoints/vae_best.pth

# Run full pipeline
python quickstart.py --all --config pipeline_config.json
```

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import sys
from pathlib import Path
import logging

# Third-party imports second
import torch
import numpy as np
from torch import nn
from torch.optim import Adam

# Local imports last
from genpipeline.schema import DesignParameters
from genpipeline.config import load_config
```

### Type Hints and Pydantic
- Use type hints consistently throughout the codebase
- Use Pydantic models for structured data validation
- Follow the pattern in `genpipeline/schema.py`
- Use `Optional[T]` for nullable fields
- Use `List[T]` instead of `[]` for type hints

### Naming Conventions
- **Classes**: PascalCase (`DesignVAE`, `Conv3DBlock`)
- **Functions/Methods**: snake_case (`train_epoch`, `validate`)
- **Variables**: snake_case (`latent_dim`, `input_shape`)
- **Constants**: UPPER_SNAKE_CASE (`VAEOutput`, `DEFAULT_BATCH_SIZE`)
- **Private methods**: prefix with underscore (`_cuda_ext_ok`)

### Error Handling
- Use try/except blocks for GPU-specific operations
- Log warnings for non-critical issues
- Use Pydantic validation for data integrity
- Handle CUDA availability gracefully

### PyTorch Patterns
- Use `torch.no_grad()` for inference
- Use `autocast` for mixed precision training
- Use `GradScaler` for loss scaling
- Use `namedtuple` for structured outputs
- Use `nn.Module` for model components

### Logging
- Use Python's `logging` module, not print statements
- Set up logger at module level: `logger = logging.getLogger(__name__)`
- Use appropriate log levels: `logger.info()`, `logger.warning()`, `logger.error()`

### GPU Considerations
- Always check `torch.cuda.is_available()` before GPU operations
- Use `device = "cuda" if torch.cuda.is_available() else "cpu"`
- Handle Blackwell-specific issues (see CLAUDE.md)
- Use BF16 for training when possible (`autocast(dtype=torch.bfloat16)`)

### File Structure
- Keep related functionality in modules (`genpipeline/`)
- Use `__init__.py` to expose public API
- Place tests in `tests/` directory mirroring source structure
- Use `checkpoints/` for model snapshots
- Use `logs/` for TensorBoard events

### Documentation
- Use docstrings for classes and complex methods
- Follow Google or NumPy docstring style
- Include type hints in docstrings
- Document GPU-specific considerations

### Testing Patterns
- Use pytest for all tests
- Use fixtures for setup/teardown
- Mark GPU-dependent tests appropriately
- Test both happy path and error conditions
- Use parameterized tests for multiple scenarios

### Configuration
- Use JSON/YAML for configuration files
- Validate configuration with Pydantic models
- Use `pipeline_config.json` as the main config
- Provide sensible defaults in config models

## Blackwell-Specific Considerations

- RTX 50 series (Blackwell) requires CUDA 12.8 builds
- Avoid batch matmul with batch_size >= 2 on GPU
- Use CPU for BoTorch GP models
- Use `blackwell_compat.py` for device management
- Test BF16 backward compatibility on Blackwell

## Performance Guidelines

- Use fused CUDA kernels where available
- Batch operations when possible
- Use in-place operations for memory efficiency
- Profile GPU memory usage during development
- Use mixed precision training for memory savings