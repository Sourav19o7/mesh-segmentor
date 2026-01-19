"""
Setup script for mesh-segmentor package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mesh-segmentor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automatic 3D jewelry segmentation using Point Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mesh-segmentor",
    packages=find_packages(exclude=["tests", "data", "checkpoints"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "rhino3dm>=8.4.0",
        "trimesh>=4.0.5",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "boto3>=1.33.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
            "pydantic>=2.5.0",
            "aiofiles>=23.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mesh-segmentor-train=training.train:main",
            "mesh-segmentor-api=api.main:main",
        ],
    },
)
