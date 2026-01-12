# Installation Guide

## Quick Start

### Option 1: Install in Development Mode (Recommended)

This allows you to edit the code and see changes immediately without reinstalling:

```bash
# Navigate to the lakes directory
cd /home/user/lakes

# Install in development mode
pip install -e .

# Install with Bayesian analysis support (PyMC)
pip install -e ".[bayesian]"

# Install everything (including dev tools)
pip install -e ".[bayesian,powerlaw,dev]"
```

### Option 2: Add to Python Path (Quick Fix)

If you just want to get started quickly in a Jupyter notebook or Python script:

```python
import sys
sys.path.insert(0, '/home/user/lakes')

# Now you can import
from lake_analysis import load_conus_lake_data
```

### Option 3: Use Spyder's Python Path

In Spyder:
1. Go to **Tools** → **PYTHONPATH manager**
2. Click **Add path**
3. Add: `/home/user/lakes`
4. Click **OK**
5. Restart Spyder

## Verification

After installation, test that it works:

```python
# This should work without errors
from lake_analysis import run_size_stratified_analysis
print("✓ Installation successful!")
```

## Common Issues

### ModuleNotFoundError: No module named 'lake_analysis'
- Solution: Install the package using one of the options above

### ModuleNotFoundError: No module named 'pymc'
- Solution: Install PyMC: `pip install pymc arviz`
- Or: Install with bayesian extras: `pip install -e ".[bayesian]"`

### ImportError with numpy/pandas/geopandas
- Solution: Install dependencies: `pip install -r requirements.txt`
- Or: Install with setup.py: `pip install -e .`

## Requirements

Minimum Python version: 3.8

### Core Dependencies
- numpy >= 1.20
- pandas >= 1.3
- geopandas >= 0.10
- matplotlib >= 3.4
- scipy >= 1.7
- shapely >= 1.8
- pyproj >= 3.0
- rasterio >= 1.2
- fiona >= 1.8

### Optional Dependencies
- pymc >= 5.0 (for Bayesian analysis)
- arviz >= 0.12 (for Bayesian diagnostics)
- powerlaw >= 1.5 (for power-law analysis)

## Updating

After pulling new changes from git:

```bash
# If installed in development mode, no reinstall needed!
# Just restart your Python kernel/session

# If you added new dependencies, update with:
pip install -e ".[bayesian]"
```
