# Generative Design Pipeline

A PyTorch-based generative design pipeline for topology optimization and structural design, featuring VAE-based design generation and Bayesian optimization for performance-driven design exploration.

## 💡 Recent Improvements and Learnings

### What We've Accomplished

- **VAE Training**: Successfully trained a 37.7M parameter VAE for 300 epochs on 64³ voxel data, achieving a final training loss of ~0.103
- **Bayesian Optimization**: Completed 20+ optimization iterations with best objective of -0.1058, discovering designs with 16.2% occupancy
- **FEM Integration**: Established robust FreeCAD 1.0 integration via WSL2 bridge for automated design generation
- **GPU Optimization**: Implemented BF16 mixed precision training on RTX 5080 (Blackwell) with CUDA 12.8

### Key Learnings

1. **Data Efficiency**: 237-450 samples proved marginal for 37.7M parameters - SIMP augmentation is critical
2. **Beta-VAE Tuning**: Initial beta_vae=1.0 caused reconstruction issues at 64³ resolution; 0.05 recommended for better results
3. **Blackwell Workarounds**: Batch matmul with batch_size≥2 requires CPU fallback for BoTorch GP models
4. **WSL2 Bridge**: Windows FreeCAD + WSL2 ccx provides reliable FEM evaluation pipeline

### Optimizations Implemented

- **Memory Efficiency**: BF16 mixed precision reduced VRAM usage by ~50%
- **Performance**: Fused CUDA kernels for voxel operations
- **Pipeline Speed**: Automated FEM data generation at 1.4s/variant
- **Code Quality**: Comprehensive test suite with 6/6 integration test pass rate

## 🚀 Quickstart

### Prerequisites
- Python 3.13+
- CUDA 12.8 (for Blackwell RTX 50 series)
- NVIDIA GPU with at least 8GB VRAM
- FreeCAD 1.0 installed on Windows (for FEM generation)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd genpipeline

# Create virtual environment
source venv/bin/activate

# Install PyTorch for Blackwell (RTX 50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Train VAE model
python vae_design_model.py --dataset-path ./fem/data/fem_dataset.pt --epochs 100

# Evaluate trained VAE
python eval_vae.py --model-checkpoint checkpoints/vae_best.pth

# Run full optimization pipeline
python quickstart.py --all --config pipeline_config.json
```

## 🛠️ Command Usage

### Core Commands

```bash
# Train VAE
python vae_design_model.py --dataset-path <path> --epochs <num>

# Evaluate VAE
python eval_vae.py --model-checkpoint <path>

# Run optimization
python quickstart.py --all --config <config>

# Test specific components
pytest tests/test_vae_model.py
pytest tests/test_voxel_fem.py
```

### Configuration

```bash
# Load configuration
python -c "from genpipeline.config import load_config; print(load_config())"

# Save configuration
python -c "from genpipeline.config import save_config; save_config({'voxel_resolution': 128})"
```

## 🔧 Built With

### Core Stack
- **PyTorch** - Deep learning framework (CUDA 12.8 for Blackwell)
- **BoTorch/GPyTorch** - Bayesian optimization and Gaussian processes
- **Pydantic** - Data validation and configuration management
- **NumPy/SciPy** - Scientific computing
- **Trimesh/PyVista** - 3D geometry processing

### GPU Optimization
- **BF16 Mixed Precision** - Memory-efficient training
- **Fused CUDA Kernels** - Performance optimization
- **NVML Integration** - GPU telemetry and monitoring

### Testing & Validation
- **pytest** - Unit and integration testing
- **Coverage reporting** - Test coverage analysis
- **GPU-specific testing** - Blackwell compatibility testing

## 📋 Project Structure

```
genpipeline/
├── notebooks/             # Jupyter notebooks for exploration
├── docs/                  # Documentation
│   ├── images/           # Images and diagrams
│   └── reports/          # Technical reports
├── results/               # Analysis results
│   ├── optimization/    # Bayesian optimization results
│   ├── evaluation/      # VAE evaluation results
│   └── training/        # Training logs and metrics
├── genpipeline/           # Main package
│   ├── models/          # Neural network models
│   ├── optimization/    # Bayesian optimization logic
│   ├── fem/             # FEM processing and evaluation
│   ├── topology/        # Topology optimization
│   ├── utils/           # Utility functions
│   ├── schema.py        # Pydantic data models
│   ├── config.py        # Configuration management
│   └── __init__.py
├── tests/                 # Test suite
├── checkpoints/           # Model snapshots
├── materials.yaml         # Material properties
├── pipeline_config.json   # Main configuration
└── scripts/               # Helper scripts
```

## 📊 Current Status (2026-03-03)

| Stage | Status | Notes |
|-------|--------|-------|
| FEM data generation | ✅ Complete | 10 variants, 1.4 s each via FreeCAD WSL2 bridge |
| Dataset | ✅ Built | 32³ (2.8 MB) and 64³ (53 MB) `.pt` files |
| VAE training | ✅ 300 epochs | `checkpoints/vae_best.pth`, train loss 0.103 |
| Bayesian optimisation | ✅ 20+ iters | Best objective −0.1058, 16.2% occupancy |
| Integration test | ✅ Added | `tests/test_integration_decode_fem.py` — 6/6 passed |
| SIMP data augmentation | ⏳ Pending | `topo_data_gen.py` exists, not yet run at scale |
| Geometry conditioning | ⏳ Pending | Single shared latent space for all 4 geometry families |

See `PROGRESS.md` for full dated log.

---

## 📁 Project Structure

```
genpipeline/
├── genpipeline/           # Main package
│   ├── schema.py          # Pydantic data models
│   ├── config.py          # Configuration management
│   ├── vae_design_model.py # VAE implementation
│   ├── optimization_engine.py # Bayesian optimization
│   └── __init__.py
├── tests/                 # Test suite
├── checkpoints/           # Model snapshots
├── materials.yaml         # Material properties
├── pipeline_config.json   # Main configuration
└── docs/                  # Documentation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with CUDA (required for GPU tests)
CUDA_VISIBLE_DEVICES=0 pytest tests/

# Run specific test file
pytest tests/test_vae_model.py

# Run with coverage
pytest --cov=genpipeline tests/

# Run with verbose output
pytest -v tests/
```

### Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end pipeline validation
- **GPU Tests** - Blackwell-specific compatibility
- **Schema Validation** - Pydantic model testing

## 🔧 Development

### Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black mypy
```

### Code Style

- **Python**: Black formatting, mypy type checking
- **Imports**: Standard library → Third-party → Local
- **Naming**: PascalCase (classes), snake_case (functions)
- **Documentation**: Google-style docstrings

### GPU Development

- Use `torch.cuda.is_available()` for GPU checks
- Implement Blackwell-specific workarounds
- Use BF16 for memory efficiency
- Monitor VRAM usage with NVML

## 🐧 Why WSL2?

This pipeline runs inside WSL2 (Windows Subsystem for Linux) with FreeCAD installed on the Windows host. Here's why that's the right trade-off:

### WSL2 makes GPU/ML config significantly easier

| Area | WSL2 | Native Windows |
|------|------|----------------|
| PyTorch install | One command (`--index-url cu128`) | Visual C++ deps, PATH conflicts |
| CUDA multi-version | Side-by-side under `/usr/local/` | Registry conflicts |
| Scientific packages | All Linux wheels available | Some lag or missing entirely |
| FreeCAD integration | Subprocess bridge to Windows | Native |
| GPU driver management | Inherited from Windows driver | Direct |
| Long overnight runs | WSL2 VM can be killed by Windows | Stable |

### How the FreeCAD/ccx split works

```
Windows side                    WSL2 side
────────────────────            ────────────────────────────
FreeCAD GUI / headless  ──→    .inp mesh file copied over
                                ccx runs natively (if installed)
                                .frd output parsed
                         ←──   results back into Python pipeline
```

FreeCAD stays on Windows because it needs a Windows install. CalculiX (`ccx`) can run natively in WSL2:

```bash
sudo apt install calculix-ccx   # eliminates the Windows ccx.exe bridge entirely
```

Once ccx is native, the Bayesian optimisation loop (`VoxelFEMEvaluator`) needs no Windows access at all — it builds hex meshes from voxels directly and runs ccx locally.

### When WSL2 is not enough

- **Overnight training runs** — Windows can kill the WSL2 VM on sleep/hibernate. Use `powercfg /requestsoverride` or keep the machine awake.
- **FreeCAD headless scripting** — still requires the Windows binary via subprocess. If you need full FreeCAD automation, dual-boot Linux is cleaner.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## 👥 Code of Conduct

Our team follows a strict code of conduct to ensure a welcoming environment for all contributors. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## 📝 Authors and Contributors

- **Primary Maintainer**: [Your Name]
- **Contributors**: [List of contributors]

## 🔐 Security

Please report security vulnerabilities to [security@example.com](mailto:security@example.com).

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [BoTorch Documentation](https://botorch.org/)
- [Pydantic Documentation](https://pydantic.dev/)

## 🚀 Acknowledgments

- NVIDIA for Blackwell GPU support
- PyTorch team for excellent deep learning framework
- Open source community for invaluable contributions

---

**Note**: This pipeline is designed for research and development purposes. Commercial use may require additional licensing or modifications.
