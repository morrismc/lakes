# Lake Distribution Analysis
## Geomorphological Controls on Lake Distributions Across CONUS

Comprehensive analysis of lake distribution patterns across the contiguous United States using the National Hydrography Dataset (NHD), testing hypotheses about lake formation, persistence, and geomorphological controls.

---

## Key Scientific Questions

### 1. **Davis's Lake Extinction Hypothesis** ⭐
Do lakes disappear as landscapes "mature" (age since glaciation)?

**Result:** YES - Lake density follows exponential decay with landscape age
- **Half-life:** ~660 ka [95% CI: 418-1505 ka]
- **Wisconsin (20 ka):** 228 lakes per 1000 km²
- **Illinoian (160 ka):** 95 lakes per 1000 km²
- **Driftless (>1.5 Ma):** 69 lakes per 1000 km²

### 2. **Elevation Bimodality**
Are lakes concentrated at specific elevations?

**Result:** YES - Bimodal distribution after normalization
- **Peak 1 (~300m):** Floodplain/coastal plain lakes
- **Peak 2 (~1300m):** Glacial lakes
- **Trough (~700m):** Dissected terrain with efficient drainage

### 3. **Power Law Size Distribution**
Do lake sizes follow a power law?

**Result:** YES - Power law exponent α varies by elevation and glacial history
- **Global:** α ≈ 2.14 (Cael & Seekell 2016)
- **By elevation:** α ranges from ~1.8 to ~2.5

### 4. **Size-Dependent Half-Life**
Do small lakes disappear faster than large lakes?

**Result:** Inconclusive - Detection bias makes this difficult to test
- Small lakes systematically underrepresented in old landscapes
- May reflect true geological process OR mapping limitations

### 5. **Multivariate Controls** ⭐
After controlling for climate and topography, is glaciation still the PRIMARY control on lake density?

**Method:** Variance partitioning to decompose explained variance into:
- Pure glaciation effect (after controlling for climate + topography)
- Pure climate effect (after controlling for glaciation + topography)
- Pure topography effect (after controlling for glaciation + climate)
- Shared variance (collinearity between factors)

**Variables:**
- **Glaciation:** Wisconsin, Illinoian, Driftless, S. Appalachian
- **Climate:** Aridity index, precipitation
- **Topography:** Elevation, slope, relief

---

## Features

✅ **16-step comprehensive analysis pipeline**
✅ **Bayesian half-life estimation** (PyMC with uncertainty quantification)
✅ **High-resolution chronosequence** (NADI-1: 0-25 ka at 0.5 ka intervals)
✅ **Size-stratified analysis** (7 lake size classes)
✅ **Multivariate statistical analysis** (variance partitioning, PCA, multiple regression)
✅ **Power law fitting** (MLE with bootstrap confidence intervals)
✅ **Spatial scaling** (latitude, longitude, elevation patterns)
✅ **Aridity analysis** (climate effects on lake density)
✅ **80+ visualization functions** (publication-quality figures)
✅ **210+ analysis functions** across 12 core modules

---

## Quick Start

### Installation

#### Using Existing Conda/Mamba Environment (Recommended)

```bash
# Activate your existing geospatial environment
mamba activate pygis_3.9

# Install missing packages
mamba install -c conda-forge pymc arviz pyarrow
pip install tqdm
```

#### Creating New Environment

```bash
mamba create -n lakes python=3.10
mamba activate lakes

# Install geospatial stack
mamba install -c conda-forge gdal geopandas rasterio fiona pyarrow scipy matplotlib seaborn

# Install Bayesian modeling
mamba install -c conda-forge pymc arviz

# Install utilities
pip install tqdm
```

**Verify GDAL drivers:**
```python
import fiona
print(fiona.supported_drivers)  # Should include: 'OpenFileGDB': 'r'
```

---

### Configuration

Edit `lake_analysis/config.py` to match your system:

```python
# Lake data
LAKE_GDB_PATH = r"F:\Lakes\GIS\MyProject.gdb"
LAKE_FEATURE_CLASS = "Lakes_with_all_details"

# Glacial boundaries
GLACIAL_BOUNDARIES = {
    'wisconsin': {'path': r"F:\Lakes\GIS\MyProject.gdb", 'layer': 'Wisconsin_area'},
    'illinoian': {'path': r"F:\Lakes\GIS\MyProject.gdb", 'layer': 'illinoian_glacial_extent'},
    'driftless': {'path': r"F:\Lakes\GIS\MyProject.gdb", 'layer': 'definite_driftless_area_never_glaciated'},
}

# NADI-1 ice sheets (Dalton et al. 2020)
NADI1_CONFIG = {
    'base_path': r"F:\Lakes\GIS\shapefiles\Dalton_2020_NA_IceSheets\shapefiles",
    'time_range_ka': (1, 25),
    'time_step_ka': 0.5,
}

# Rasters
RASTERS = {
    'elevation': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_dem",
    'slope': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_slope",
    'relief_5km': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_rlif_5k",
    'aridity': r"F:\Lakes\GIS\rasters\Climate\aridity_index",
    # ... etc
}

# Output directory
OUTPUT_DIR = r"F:\Lakes\Analysis\outputs"
```

---

### Run Complete Analysis

```python
from lake_analysis import run_full_analysis

results = run_full_analysis(
    data_source='conus',              # 'conus' (fast), 'gdb', or 'parquet'
    include_xmin_by_elevation=True,   # Comprehensive xmin sensitivity
    include_glacial_analysis=True,    # Davis's hypothesis
    include_bayesian_halflife=True,   # Overall + size-stratified half-life
    include_spatial_scaling=True,     # Geographic patterns
    include_aridity_analysis=True,    # Climate effects
    min_lake_area=0.01,               # Minimum lake size (km²)
)

# All figures automatically saved to OUTPUT_DIR
```

**Pipeline Steps:**
1. Load data (with quality filters)
2. H1: Elevation bimodality
3. H2: Slope threshold
4. H3: Relief controls
5. H4: 2D elevation × slope domains
6. H5: Power law analysis
7. H6: Slope-relief domains
8. Relief × elevation 2D
9. Power law sensitivity
10. xmin by elevation *(optional)*
11. Domain classification
12. Glacial chronosequence *(optional)*
13. Bayesian half-life *(optional)*
14. Spatial scaling *(optional)*
15. Aridity analysis *(optional)*
16. Summary figures

**Runtime:** ~30-60 minutes on full CONUS dataset (4.9M lakes)

---

### Run Specific Analyses

#### Glacial Chronosequence (Davis's Hypothesis)

```python
from lake_analysis import analyze_glacial_chronosequence

results = analyze_glacial_chronosequence(
    lakes,
    max_lake_area=20000,  # Exclude Great Lakes
    verbose=True
)
```

#### High-Resolution Time Slices (NADI-1)

```python
from lake_analysis import analyze_nadi1_chronosequence

results = analyze_nadi1_chronosequence(
    lakes,
    max_lake_area=20000,
    use_bayesian=True,  # Fit exponential decay model
    verbose=True
)
```

#### Size-Stratified Half-Life

```python
from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    analyze_bayesian_halflife
)

lakes = load_conus_lake_data()
lakes_gdf = convert_lakes_to_gdf(lakes)
boundaries = load_all_glacial_boundaries()
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

results = analyze_bayesian_halflife(
    lakes_classified,
    run_overall=True,
    run_size_stratified=True,
    test_thresholds=True,  # Test threshold sensitivity
    save_figures=True
)
```

---

## Project Structure

```
lakes/
├── lake_analysis/              # Main package (25,563 lines)
│   ├── config.py              # Paths, constants, parameters
│   ├── data_loading.py        # Load lakes, rasters, shapefiles (22 functions)
│   ├── normalization.py       # Compute normalized density (10 functions)
│   ├── powerlaw_analysis.py   # MLE fitting, hypothesis tests (22 functions)
│   ├── glacial_chronosequence.py  # Glacial history analysis (73 functions)
│   ├── size_stratified_analysis.py  # Size-stratified half-life (9 functions)
│   ├── spatial_scaling.py     # Geographic patterns (6 functions)
│   ├── visualization.py       # Publication plots (74 functions)
│   ├── main.py               # Orchestration, entry points (39 functions)
│   └── __init__.py           # Package exports
│
├── examples/                   # Example scripts
│   ├── example_size_stratified_analysis.py
│   ├── example_halflife_sensitivity.py
│   └── run_sapp_comparison.py
│
├── Documentation               # 15+ .md files (~150 KB)
│   ├── README.md             # This file
│   ├── CLAUDE.md             # Project context for Claude Code
│   ├── COMPLETE_ANALYSIS_GUIDE.md  # All available analyses (comprehensive)
│   ├── DIAGNOSTIC_REVIEW.md  # Recent diagnostic summary
│   └── [other session notes]
│
├── Diagnostic scripts          # Test & QA/QC
│   ├── test_sapp_projection.py
│   ├── analyze_threshold_impact.py
│   └── [other diagnostics]
│
└── data/                      # Data files (not in git)
    ├── NHD/                   # National Hydrography Dataset
    ├── rasters/               # Elevation, slope, relief, climate
    └── glacial_boundaries/    # Glacial extent shapefiles
```

---

## Core Hypotheses

| ID | Hypothesis | Method | Key Visualization |
|----|------------|--------|-------------------|
| **H1** | Elevation-normalized density is bimodal | 1D normalization | Raw vs. normalized comparison |
| **H2** | Slope threshold for lake existence | 1D normalization | Density vs. slope curve |
| **H3** | Relief controls at intermediate values | 1D normalization | Quadratic fit to relief |
| **H4** | 2D normalization reveals process domains | 2D normalization | Elevation × slope heatmap |
| **H5** | Power law exponent varies by elevation | MLE fitting | α by elevation band |
| **H6** | Slope-relief domains exist | 2D normalization | Slope × relief heatmap |

---

## Advanced Analyses

### Glacial Chronosequence

**Davis's Hypothesis (1899):** Lakes are "youthful" features that fill in as landscapes mature.

**Three-Stage Analysis:**
- Wisconsin (~20 ka): Most recent glaciation
- Illinoian (~160 ka): Older glaciation
- Driftless (>1.5 Ma): Never glaciated

**NADI-1 Time Slices:**
- 50 time slices from 1-25 ka (Dalton et al. 2020)
- Bayesian exponential decay: D(t) = D₀ × exp(-k × t)
- Half-life estimation with credible intervals

### Size-Stratified Half-Life

Tests whether small lakes have shorter half-lives than large lakes.

**Size Classes:**
- Tiny: 0.05-0.1 km²
- Very Small: 0.1-0.25 km²
- Small: 0.25-0.5 km²
- Medium-Small: 0.5-1.0 km²
- Medium: 1.0-2.5 km²
- Large: 2.5-10.0 km²
- Very Large: >10.0 km²

**Methods:**
- Separate Bayesian decay models per size class
- Statistical test for size-halflife correlation
- Detection limit diagnostics

### Power Law Analysis

**Model:** P(A > a) ∝ a^(-α)

**Methods:**
- MLE estimation (Clauset et al. 2009)
- Bootstrap confidence intervals
- Goodness-of-fit testing (KS statistic)
- xmin sensitivity analysis
- Comparison to global power law (Cael & Seekell 2016)

**Domain-Specific:**
- Separate fitting by elevation band
- Glacial vs non-glacial regions
- Alpine vs continental ice

### Spatial Scaling

- Latitudinal gradient (north-south patterns)
- Longitudinal gradient (east-west patterns)
- Elevation gradient (global pattern)
- Glacial vs non-glacial comparison

### Aridity Analysis

- Aridity vs lake density
- Aridity × glacial stage interaction
- Aridity-conditional half-life

---

## Standalone Analyses (Not in Main Pipeline)

These analyses exist as standalone functions but are **NOT** automatically run in `run_full_analysis()`. See **[COMPLETE_ANALYSIS_GUIDE.md](COMPLETE_ANALYSIS_GUIDE.md)** for details.

1. **NADI-1 Chronosequence** - High-resolution (0-25 ka) time slices
2. **Dalton 18ka Analysis** - Alternative ice sheet reconstruction
3. **Western Alpine Analysis** - Alpine vs continental ice separation
4. **S. Appalachian Comparison** - Non-glacial control region
5. **Half-Life Threshold Sensitivity** - Diagnostic for min_lake_area choice
6. **Aridity-Conditional Half-Life** - Climate effects on persistence
7. **Advanced Power Law Tests** - Comprehensive hypothesis testing
8. **Multivariate Statistical Analysis** - Disentangle glaciation vs climate vs topography effects

**Access:** See `COMPLETE_ANALYSIS_GUIDE.md` for usage examples.

### Running Multivariate Analysis

```python
# Standalone script
python run_multivariate_analysis.py

# Or programmatically
from lake_analysis import run_complete_multivariate_analysis

results = run_complete_multivariate_analysis(
    lakes_classified,
    response_var='area',
    min_lake_area=0.01,
    max_lake_area=20000,
    save_figures=True
)

# Access results
corr_matrix = results['correlation_matrix']  # Spearman correlations
pca = results['pca']  # Principal components
vp = results['variance_partitioning']  # Pure and shared effects
reg = results['regression']  # Multiple regression
```

**Key Question Answered:** Is glaciation the primary control, or is it confounded with climate/topography?

**Method:** Variance partitioning decomposes R² into:
- Pure glaciation (after controlling for climate + topography)
- Pure climate (after controlling for glaciation + topography)
- Pure topography (after controlling for glaciation + climate)
- Shared variance (collinearity)

---

## Data Sources

### Lakes (NHD-derived)
- **Source:** USGS National Hydrography Dataset (NHD)
- **Format:** File Geodatabase → Parquet (for speed)
- **Count:** ~4.9M lakes in CONUS
- **Key Attributes:**
  - `AREASQKM` - Lake surface area (km²)
  - `Elevation_` - Elevation at centroid (m) *note trailing underscore!*
  - `Slope`, `F5km_relief` - Terrain metrics
  - `MAT`, `precip_mm`, `AI`, `PET` - Climate variables

### Glacial Boundaries
- **Wisconsin:** Most recent glaciation (~15-25 ka)
- **Illinoian:** Older glaciation (~130-190 ka)
- **Driftless:** Never glaciated region in Wisconsin/Minnesota
- **NADI-1:** Dalton et al. (2020) ice sheets, 1-25 ka at 0.5 ka intervals
- **Dalton 18ka:** Alternative LGM reconstruction

### Rasters
- **Elevation, slope, relief:** NAD83 Albers Equal Area projection
- **Climate:** WGS84 (aridity index, PET, precipitation)
- **Resolution:** Varies by variable (~90m-1km)

---

## Key Implementation Details

### Data Quality Filters

Applied automatically in `data_loading.py`:

```python
# Minimum reliable mapping threshold
AREASQKM >= 0.0051  # 0.0051 km² = 5,100 m² = ~0.5 hectares

# Remove errors
Elevation_ >= 0
Elevation_ != -9999  # NoData value

# Similar filters for climate variables
AI > 0              # Positive aridity index
AI != -9999         # NoData
```

### CRS Handling

- **Target CRS:** EPSG:5070 (NAD83 Albers Equal Area) for area calculations
- **Geographic filtering:** Use WGS84 for longitude-based filtering
- **Workflow:** Compute centroids on projected CRS, then transform to geographic if needed

### Great Lakes Exclusion

The Great Lakes are inherited basins spanning multiple glacial cycles and should be excluded from chronosequence analysis:

```python
results = analyze_nadi1_chronosequence(lakes, max_lake_area=20000)
```

### Bayesian Model

Uses PyMC to fit exponential decay with proper uncertainty quantification:

- **Model:** D(t) = D₀ × exp(-k × t)
- **Parameters:** D₀ (initial density), k (decay rate), t (age)
- **Half-life:** t½ = ln(2) / k
- **Age uncertainty:** Each glacial stage has age distribution (not point estimate)
- **Sampler:** NUTS with 4 chains, 1000 tune, 2000 samples per chain

---

## Common Issues & Solutions

### Issue: Memory errors with large rasters
**Solution:** Use chunked processing (enabled by default in `load_raster_chunked()`)

### Issue: Missing glacial boundaries
**Solution:** Configure paths in `config.py` → `GLACIAL_BOUNDARIES`

### Issue: Half-life varies dramatically with threshold
**Solution:** This is **detection bias**, not a bug. Small lakes are systematically missing from old landscapes. Use `test_thresholds=True` to diagnose.

### Issue: Y-axis scaling issues in decay plot
**Solution:** Set `max_lake_area=20000` to exclude Great Lakes

### Issue: S. Appalachian lakes appear in wrong location on map
**Solution:** Verify projection is correct (should be ESRI:102039 in meters, not degrees). Run `test_sapp_projection.py` to diagnose.

---

## Documentation

- **[README.md](README.md)** - This file (overview & quick start)
- **[CLAUDE.md](CLAUDE.md)** - Project context for Claude Code (workflows & implementation details)
- **[COMPLETE_ANALYSIS_GUIDE.md](COMPLETE_ANALYSIS_GUIDE.md)** - Comprehensive inventory of ALL analyses
- **[DIAGNOSTIC_REVIEW.md](DIAGNOSTIC_REVIEW.md)** - Recent diagnostic summary
- **[INSTALL.md](INSTALL.md)** - Detailed installation instructions

---

## Literature References

### Power Law Analysis
- **Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).** Power-law distributions in empirical data. *SIAM Review*, 51(4), 661-703.
- **Cael, B. B., & Seekell, D. A. (2016).** The size-distribution of Earth's lakes. *Scientific Reports*, 6, 29633.

### Glacial Chronosequence
- **Davis, W. M. (1899).** The geographical cycle. *Geographical Journal*, 14(5), 481-504.
- **Dalton, A. S., et al. (2020).** An updated radiocarbon-based ice margin chronology for the last deglaciation of the North American Ice Sheet Complex. *Quaternary Science Reviews*, 234, 106223.

### Lake Geomorphology
- **Hutchinson, G. E. (1957).** A Treatise on Limnology, Vol. 1: Geography, Physics, and Chemistry. Wiley.
- **Goodchild, M. F. (1988).** Lakes on fractal surfaces: a null hypothesis for lake-rich landscapes. *Mathematical Geology*, 20(6), 615-630.

---

## Citation

If you use this code or methodology in your research, please cite:

```
[Your Citation Here]
```

---

## Contact

Project maintained by **morrismc**

For questions:
- **Glacial chronosequence:** See detailed comments in `glacial_chronosequence.py`
- **Power law analysis:** See detailed comments in `powerlaw_analysis.py`
- **General workflow:** See `CLAUDE.md` for project context
- **All analyses:** See `COMPLETE_ANALYSIS_GUIDE.md` for comprehensive inventory

**GitHub:** [Add your GitHub URL]
**Last Updated:** 2026-01-19

---

## License

[Your License Here]
