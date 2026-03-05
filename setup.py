import setuptools

# Package metadata
setuptools.setup(
    name="genpipeline",
    version="0.1.0",
    packages=setuptools.find_packages(
        exclude=["tests", "scripts", "freecad_workbench", "freecad_scripts"]
    ),
    install_requires=[
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "botorch==0.17.0",
        "gpytorch==1.15.1",
        "numpy",
        "scipy",
        "scikit-image",
        "trimesh",
        "pyvista",
        "pydantic",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "seaborn",
        "tensorboard",
        "tensorboardX",
        "ninja",
        "pybind11",
        "pytest",
        "jaxtyping",
        "nvidia-ml-py",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "genpipeline=genpipeline.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.py"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Generative Design Pipeline for Topology Optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="GenPipeline Team",
    author_email="team@genpipeline.com",
    url="https://github.com/genpipeline/genpipeline",
    license="MIT",
    zip_safe=False,
    platforms="any",
)
