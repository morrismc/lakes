"""
Setup script for lake_analysis package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "CLAUDE.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "Lake distribution analysis package for CONUS"

setup(
    name="lake_analysis",
    version="0.1.0",
    description="Analysis of lake distribution patterns across the continental United States",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lake Analysis Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "geopandas>=0.10",
        "matplotlib>=3.4",
        "scipy>=1.7",
        "shapely>=1.8",
        "pyproj>=3.0",
        "rasterio>=1.2",
        "fiona>=1.8",
    ],
    extras_require={
        "bayesian": [
            "pymc>=5.0",
            "arviz>=0.12",
        ],
        "powerlaw": [
            "powerlaw>=1.5",
        ],
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="lakes geomorphology glaciation GIS",
)
