# CLAUDE.md - Lake Analysis Project Context

## Project Overview

This project analyzes lake distribution patterns across the contiguous United States (CONUS) to test hypotheses about lake formation and persistence. The primary scientific question is **Davis's Lake Extinction Hypothesis**: Does lake density decrease with landscape "maturity" (time since glaciation)?

## Quick Start

```python
from lake_analysis import analyze_nadi1_chronosequence

# Load lakes (from parquet for speed)
lakes = load_conus_lake_data()

# Run glacial chronosequence analysis (excludes Great Lakes)
results = analyze_nadi1_chronosequence(lakes, max_lake_area=20000)
```

## Key Scientific Concepts

### Davis's Lake Extinction Hypothesis
W.M. Davis proposed that landscapes evolve through stages, and lakes are "youthful" features that fill in over time. This predicts:
- **Wisconsin glaciation (~20 ka)**: Highest lake density (youngest landscape)
- **Illinoian glaciation (~160 ka)**: Intermediate density
- **Driftless Area (>1.5 Ma)**: Lowest density (never glaciated, oldest)

### Glacial Chronosequence
The analysis uses three "deep time end members" to fit an exponential decay model:
- D(t) = D₀ × exp(-k × t)
- Half-life (t½) = ln(2) / k ≈ 500-1000 ka (estimated)

### Lake Density Normalization
Raw lake counts are misleading because they don't account for available landscape area:
- **Normalized Density** = (# lakes in bin) / (landscape area in bin) × 1000
- Units: lakes per 1000 km²

## Project Structure

```
lakes/
├── CLAUDE.md              # This file - project context
├── lake_analysis/
│   ├── config.py          # Configuration, paths, constants
│   ├── data_loading.py    # Load lakes, rasters, shapefiles
│   ├── normalization.py   # Compute normalized lake density
│   ├── powerlaw_analysis.py  # Power law fitting (Clauset et al. 2009)
│   ├── spatial_scaling.py    # Geographic pattern analysis
│   ├── glacial_chronosequence.py  # Glacial history analysis (Davis hypothesis)
│   ├── visualization.py   # Publication-quality plots
│   └── main.py           # Entry point, orchestration
├── data/                  # Data files (not in git)
│   ├── NHD/              # National Hydrography Dataset
│   ├── rasters/          # Elevation, slope, relief, climate
│   └── glacial_boundaries/  # Glacial extent shapefiles
└── output/               # Generated figures and results
```

## Key Modules

### glacial_chronosequence.py
Main analysis for Davis's hypothesis. Key functions:
- `run_nadi1_chronosequence_analysis()`: Complete analysis pipeline
- `fit_bayesian_decay_model()`: PyMC Bayesian exponential decay
- `load_wisconsin_extent()`, `load_illinoian_extent()`, `load_driftless_area()`: Load glacial boundaries
- `assign_deglaciation_age()`: Assign ages using NADI-1 time slices

**Important parameters:**
- `min_lake_area=0.01`: Minimum lake size (km²)
- `max_lake_area=20000`: Maximum lake size (km²) - **use this to exclude Great Lakes**
- `use_bayesian=True`: Use PyMC for proper uncertainty quantification
- `compare_with_illinoian=True`: Include deep time end members

### visualization.py
- `plot_nadi1_density_decay()`: Main decay plot with Bayesian credible intervals
- `plot_bayesian_summary()`: 4-panel Bayesian analysis summary
- `plot_density_with_uncertainty()`: Density with MIN/MAX uncertainty bands

### config.py
Key constants:
- `NADI1_CONFIG`: Dalton et al. ice sheet reconstruction settings
- `GLACIAL_BOUNDARIES`: Paths to Wisconsin, Illinoian, Driftless shapefiles
- `COLS`: Column name mapping for lake attributes
- `TARGET_CRS`: EPSG:5070 (NAD83 Albers Equal Area)

## Data Sources

### Lakes
- **Source**: USGS National Hydrography Dataset (NHD)
- **Format**: File Geodatabase → Parquet (faster loading)
- **Key columns**: AREASQKM, Elevation_, Slope, F5km_relief, Latitude, Longitude

### Glacial Boundaries
- **NADI-1**: Dalton et al. (2020) ice sheet reconstructions, 1-25 ka at 0.5 ka intervals
- **Wisconsin**: Most recent glaciation (~15-25 ka)
- **Illinoian**: Older glaciation (~130-190 ka)
- **Driftless**: Never glaciated region in Wisconsin/Minnesota

### Rasters
- Elevation, slope, relief (NAD83 Albers projection)
- Climate: PET, precipitation, aridity index (WGS84)

## Critical Implementation Details

### CRS Handling
- **Target CRS**: EPSG:5070 (NAD83 Albers Equal Area) for area calculations
- **Geographic filtering**: Use WGS84 for longitude-based filtering
- Compute centroids on projected CRS, then transform to geographic for lon filtering

### Great Lakes Exclusion
The Great Lakes are inherited basins that span multiple glacial cycles and should be excluded:
```python
results = run_nadi1_chronosequence_analysis(lakes, max_lake_area=20000)
```

### Bayesian Model
Uses PyMC to fit exponential decay with proper uncertainty on:
- D₀ (initial density)
- k (decay rate)
- Half-life = ln(2) / k
- Age posteriors for each glacial stage

### Error Bar Conventions
- `density_min`/`density_max`: Named after ICE EXTENT type, not actual density bounds
- Use `np.minimum.reduce()` and `np.maximum.reduce()` to get actual bounds

## Common Issues and Solutions

### KeyError in Bayesian curves
The curve dictionary uses keys like `ci_lower_95`, not `ci_95_lower`.

### Duplicate "Loaded X ka" messages
Set `verbose=False` when calling `load_nadi1_time_slice()` from inside loops.

### Y-axis scaling issues in decay plot
Set `max_lake_area` to exclude Great Lakes, which dominate density calculations.

### CRS warnings about geographic coordinates
Compute centroids on projected CRS first, then transform to geographic.

## Typical Workflow

1. **Load data**: `lakes = load_conus_lake_data()`
2. **Filter**: Apply min/max lake area filters
3. **Run chronosequence**: `results = run_nadi1_chronosequence_analysis(lakes, max_lake_area=20000)`
4. **Visualize**: Figures auto-saved to output directory

## Git Workflow

- **Main branch**: `main` (or `master`)
- **Feature branches**: `claude/feature-name-XXXXX`
- Always commit changes before switching context
- Push with: `git push -u origin branch-name`

## References

- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661-703.
- Cael, B. B., & Seekell, D. A. (2016). The size-distribution of Earth's lakes. *Scientific Reports*, 6, 29633.
- Dalton, A. S., et al. (2020). An updated radiocarbon-based ice margin chronology for the last deglaciation of the North American Ice Sheet Complex. *Quaternary Science Reviews*, 234, 106223.
- Davis, W. M. (1899). The geographical cycle. *Geographical Journal*, 14(5), 481-504.

## Contact

Project maintained by morrismc. For questions about the glacial chronosequence analysis, see the detailed comments in `glacial_chronosequence.py`.
