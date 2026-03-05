# Contributing to Generative Design Pipeline

We welcome contributions from the community! Please read the following guidelines to ensure a smooth contribution process.

## Code of Conduct

This project follows a strict code of conduct to ensure a welcoming environment for all contributors. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Getting Started

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes**
4. **Test your changes**: `pytest tests/`
5. **Format your code**: `black genpipeline/ tests/`
6. **Type check**: `mypy genpipeline/`
7. **Submit a pull request**

## Development Guidelines

### Code Style
- Use **Black** for code formatting
- Use **mypy** for type checking
- Follow the import organization pattern in AGENTS.md
- Use Google-style docstrings

### Testing
- All new features must include tests
- GPU-specific features require Blackwell compatibility testing
- Test both happy path and error conditions
- Use pytest fixtures for setup/teardown

### Documentation
- Update README.md for new features
- Add docstrings for new classes and methods
- Document GPU-specific considerations

## Pull Request Process

1. Ensure all tests pass
2. Ensure code is properly formatted and type-checked
3. Update documentation if necessary
4. Reference any relevant issues in your PR description
5. Request review from maintainers

## Reporting Issues

If you encounter bugs or have feature requests:

1. Check existing issues first
2. Provide detailed reproduction steps
3. Include error messages and stack traces
4. Specify your hardware (especially GPU model)

## Security Issues

Please report security vulnerabilities to [security@example.com](mailto:security@example.com) directly.

## License

This project is licensed under the MIT License. By contributing, you agree that your contributions will be licensed under the same terms.