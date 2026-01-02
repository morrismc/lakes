from setuptools import setup, find_packages

setup(
    name="lake_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "matplotlib",
        "scipy",
        "pyarrow",  # for parquet support
    ],
    python_requires=">=3.8",
    description="Lake distribution analysis for CONUS",
    author="Morris Lab",
)
