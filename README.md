# Lake Distribution Analysis

Geomorphological controls on lake distributions across the continental United States using the National Hydrography Dataset (NHD).

## Key Innovation

**Elevation-normalization of lake density** - asking not just "where are lakes?" but "where are lakes *relative to available landscape area*?"

### Core Discovery

When lake counts are normalized by available land area at each elevation, a **bimodal pattern** emerges:
- **Peak 1 (Low Elevation)**: Floodplain/coastal plain lakes (~200-400m)
- **Peak 2 (High Elevation)**: Glacial lakes (~1200-1400m)
- **Trough (Mid-elevation)**: Dissected terrain with efficient drainage networks

## Project Structure

```
lakes/
├── lake_analysis/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Paths, constants, parameters
│   ├── data_loading.py       # Load geodatabase + rasters
│   ├── normalization.py      # Compute normalized density
│   ├── visualization.py      # Plotting functions
│   ├── powerlaw_analysis.py  # MLE fitting
│   ├── main.py               # Orchestration script
│   └── outputs/              # Generated figures and tables
├── requirements.txt
└── README.md
```

## Installation

### Using Conda (Recommended for Windows)

```bash
# Create environment
conda create -n lakes python=3.10
conda activate lakes

# Install geospatial dependencies (GDAL with drivers)
conda install -c conda-forge gdal geopandas rasterio fiona pyarrow

# Install other dependencies
pip install scipy matplotlib seaborn tqdm
```

### Verify GDAL Drivers

```python
import fiona
print(fiona.supported_drivers)
# Should include: 'OpenFileGDB': 'r'
```

## Quick Start

### 1. Configure Paths

Edit `lake_analysis/config.py` to match your system:

```python
# Lake geodatabase
LAKE_GDB_PATH = r"F:\Lakes\GIS\MyProject.gdb"
LAKE_FEATURE_CLASS = "Lakes_with_all_details"

# Rasters
RASTERS = {
    'elevation': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_dem",
    'slope': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_slope",
    'relief_5km': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_rlif_5k",
    # ... etc
}
```

### 2. Verify Data Access

```python
from lake_analysis import quick_data_check
quick_data_check()
```

### 3. Run Analysis

```python
from lake_analysis import load_data, analyze_elevation, analyze_powerlaw

# Load lakes
lakes = load_data(source='gdb')

# Test H1: Bimodal elevation distribution
elev_results = analyze_elevation(lakes)

# Test H5: Power law by domain
powerlaw_results = analyze_powerlaw(lakes)
```

### 4. Or Run Full Pipeline

```bash
python -m lake_analysis.main --full
```

## Hypotheses

| ID | Hypothesis | Key Visualization |
|----|------------|-------------------|
| H1 | Elevation-normalized density is bimodal | Raw vs. normalized comparison |
| H2 | Slope threshold for lake existence | Density vs. slope curve |
| H3 | Relief controls at intermediate values | Quadratic fit to relief |
| H4 | 2D normalization reveals process domains | Elevation × slope heatmap |
| H5 | Power law exponent varies by elevation | α by elevation band |

## Data Sources

### Lake Data (NHD-derived)
- **Format**: File Geodatabase feature class
- **Key Columns**:
  - `AREASQKM` - Lake surface area (km²)
  - `Elevation_` - Elevation at centroid (m) - note trailing underscore!
  - `Slope`, `F5km_relief`, `MAT`, `precip_mm`, `AI`, `PET`

### Raster Data
- **Format**: ESRI Grid (.adf folders) or GeoTIFF
- **Variables**: Elevation, slope, relief, climate variables
- **Used for**: Computing available landscape area in each parameter bin

## Key Data Quality Filters

```python
# Applied automatically in data_loading.py
AREASQKM >= 0.0051      # Minimum reliable mapping threshold
Elevation_ >= 0         # Remove errors
Elevation_ != -9999     # NoData filter
# Similar filters for climate variables
```

## Module Reference

### data_loading.py
- `load_lake_data_from_gdb()` - Load from File Geodatabase
- `calculate_landscape_area_by_bin()` - Key normalization denominator
- `check_raster_alignment()` - Verify CRS compatibility

### normalization.py
- `compute_1d_normalized_density()` - Single variable normalization
- `compute_2d_normalized_density()` - Joint normalization (memory intensive)
- `classify_lake_domains()` - Geomorphic domain classification

### visualization.py
- `plot_raw_vs_normalized()` - Shows why normalization matters
- `plot_2d_heatmap()` - Process domain visualization
- `plot_powerlaw_rank_size()` - Power law diagnostics

### powerlaw_analysis.py
- `full_powerlaw_analysis()` - Complete MLE pipeline
- `fit_powerlaw_by_elevation_bands()` - Domain-specific fitting
- Bootstrap confidence intervals and goodness-of-fit testing

## Literature References

1. **Cael & Seekell (2016)** - Power law τ ≈ 2.14 for lakes ≥0.46 km²
2. **Clauset et al. (2009)** - MLE methodology for power laws
3. **Davis (1882, 1899)** - Lake density decreases with landscape maturity
4. **Goodchild (1988)** - Lakes on fractal surfaces

## Notes

- **Memory**: Continental-scale rasters may exceed RAM. Use chunked processing.
- **Projection**: Verify all datasets share common CRS before 2D analysis.
- **ESRI Grid**: These appear as folders in Windows Explorer but are read as single rasters.

## Contact

[Your Name/Email]

## License

[Your License]
