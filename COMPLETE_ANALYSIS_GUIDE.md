# Complete Analysis Guide
## All Available Analyses in the Lake Distribution Project

**Last Updated**: 2026-01-19

This document provides a comprehensive inventory of **ALL** analyses available in this project, both integrated into the main pipeline and available as standalone functions.

---

## Table of Contents

1. [Main Analysis Pipeline](#main-analysis-pipeline) (16 steps)
2. [Standalone Analyses](#standalone-analyses) (not in main pipeline)
3. [Quick Reference by Research Question](#quick-reference-by-research-question)
4. [Module-by-Module Function List](#module-by-module-function-list)
5. [Example Workflows](#example-workflows)

---

## Main Analysis Pipeline

The primary entry point is `run_full_analysis()` in `main.py`, which runs a comprehensive 16-step pipeline:

```python
from lake_analysis import run_full_analysis

results = run_full_analysis(
    data_source='conus',              # 'conus', 'gdb', or 'parquet'
    include_xmin_by_elevation=True,   # Step 10 (optional)
    include_glacial_analysis=True,    # Step 12 (optional)
    include_bayesian_halflife=True,   # Step 13 (optional)
    include_spatial_scaling=True,     # Step 14 (optional)
    include_aridity_analysis=True,    # Step 15 (optional)
    min_lake_area=0.01,               # Minimum lake size (km²)
    prompt_for_threshold=False        # Interactive threshold selection
)
```

### Step-by-Step Breakdown

| Step | Analysis | Hypothesis | Key Output |
|------|----------|------------|------------|
| 1 | **Data Loading** | - | Loads lakes, applies quality filters |
| 2 | **Elevation Bimodality** | H1 | Tests for bimodal lake density with elevation |
| 3 | **Slope Threshold** | H2 | Identifies slope threshold for lake existence |
| 4 | **Relief Controls** | H3 | Tests relief effects on lake density |
| 5 | **2D Elevation × Slope** | H4 | Process domain classification |
| 6 | **Power Law Analysis** | H5 | Tests power law size distribution |
| 7 | **Slope-Relief Domains** | H6 | 2D slope × relief analysis |
| 8 | **Relief × Elevation 2D** | - | Extended 2D analysis |
| 9 | **Power Law Sensitivity** | - | Tests power law robustness |
| 10 | **xmin by Elevation** | - | *Optional:* Comprehensive xmin sensitivity |
| 11 | **Domain Classification** | - | Geomorphic domain assignment |
| 12 | **Glacial Chronosequence** | Davis's hypothesis | *Optional:* Wisconsin/Illinoian/Driftless analysis |
| 13 | **Bayesian Half-Life** | Lake persistence | *Optional:* Overall + size-stratified half-life |
| 14 | **Spatial Scaling** | Geographic patterns | *Optional:* Latitude/longitude/elevation |
| 15 | **Aridity Analysis** | Climate effects | *Optional:* Aridity × glacial stage |
| 16 | **Summary Figures** | - | Consolidated visualizations |

### Analysis Outputs

All figures are automatically saved to `OUTPUT_DIR` (configured in `config.py`):

```
output/
├── H1_elevation_density.png
├── H2_slope_threshold.png
├── H3_relief_controls.png
├── H4_2d_elevation_slope_heatmap.png
├── H5_powerlaw_*.png
├── H6_slope_relief_heatmap.png
├── glacial_chronosequence_summary.png
├── bayesian_overall_halflife.png
├── bayesian_glacial_extent_map.png
├── size_stratified_*.png
├── spatial_scaling_*.png
├── aridity_*.png
└── [many more diagnostic plots]
```

---

## Standalone Analyses

These substantial analyses exist but are **NOT** automatically run in `run_full_analysis()`. Call them directly for specialized research questions.

### 1. NADI-1 Time Slice Chronosequence (0-25 ka)

High-resolution deglaciation chronosequence using Dalton et al. (2020) ice sheet reconstructions at 0.5 ka intervals.

```python
from lake_analysis import analyze_nadi1_chronosequence

results = analyze_nadi1_chronosequence(
    lakes,
    max_lake_area=20000,        # Exclude Great Lakes
    min_lake_area=0.01,         # Minimum lake size
    use_bayesian=True,          # Bayesian exponential decay model
    verbose=True
)
```

**Outputs:**
- Density vs. deglaciation age (0-25 ka)
- Bayesian decay curve: D(t) = D₀ × exp(-k × t)
- Half-life estimate with 95% credible intervals
- Glaciated area timeseries

**Key Difference from Step 12:** Uses 50 time slices instead of 3 glacial stages (Wisconsin/Illinoian/Driftless)

---

### 2. Dalton 18ka Ice Sheet Reconstruction

Alternative to Wisconsin glaciation boundary using Dalton et al. (2020) 18 ka reconstruction.

```python
from lake_analysis.glacial_chronosequence import run_dalton_18ka_analysis

results = run_dalton_18ka_analysis(
    lakes,
    classify_ice_types=True,   # Separate alpine vs continental ice
    verbose=True
)
```

**Outputs:**
- Density comparison: Dalton 18ka vs Wisconsin extent
- Alpine vs continental ice classification
- Sensitivity analysis

**Use case:** Test robustness of glacial chronosequence results to boundary choice

---

### 3. Western Alpine Lakes Analysis

Regional analysis separating alpine glaciation (mountain ranges) from continental ice sheets.

```python
from lake_analysis.glacial_chronosequence import western_alpine_analysis

results = western_alpine_analysis(
    lakes,
    dalton_18ka_boundary=None,  # Auto-loads if None
    verbose=True
)
```

**Outputs:**
- Alpine vs continental lake density
- Regional breakdown (Cascades, Rockies, Sierra Nevada)
- Power law by ice type

**Use case:** Test if alpine and continental glaciation have different effects on lake persistence

---

### 4. Southern Appalachian Non-Glacial Comparison

Control region: non-glacial highlands with different hypsometry and lake formation processes.

```python
from lake_analysis.glacial_chronosequence import (
    load_southern_appalachian_lakes,
    compute_sapp_hypsometry_normalized_density
)

# Load S. Appalachian lakes
sapp_lakes = load_southern_appalachian_lakes()

# Compute hypsometry-normalized density
sapp_results = compute_sapp_hypsometry_normalized_density(
    sapp_lakes,
    elev_breaks=np.arange(0, 2001, 100),  # 100m bins
    verbose=True
)
```

**Outputs:**
- Hypsometry-normalized density (lakes per 1000 km² by elevation)
- Peak density elevation band
- Comparison with glacial regions

**Use case:** Control for elevation effects; test if lake-elevation patterns differ in non-glacial regions

**Integration:** Can be added to Bayesian half-life analysis:
```python
results = analyze_bayesian_halflife(
    lakes_classified,
    include_sapp=True,      # Add S. Appalachian comparison
    generate_map=True,
    save_figures=True
)
```

---

### 5. Half-Life Threshold Sensitivity

Diagnostic analysis showing how estimated half-life varies with minimum lake area threshold.

```python
from lake_analysis import analyze_bayesian_halflife

results = analyze_bayesian_halflife(
    lakes_classified,
    test_thresholds=True,   # Enable threshold sensitivity
    save_figures=True
)
```

**Outputs:**
- Half-life estimates for thresholds: [0.005, 0.01, 0.024, 0.05, 0.1] km²
- Detection bias diagnostic plot
- Recommended threshold identification

**Key Finding:**
- 0.005 km²: t½ = 858 ka (small lakes present in both young and old landscapes)
- 0.01 km²:  t½ = 242 ka (preferred threshold)
- 0.05 km²:  t½ = 103 ka (conservative, fewer lakes)

**Use case:** Justify choice of minimum lake area threshold; detect mapping biases

---

### 6. Aridity-Conditional Half-Life

Tests whether lake half-life varies with aridity (climate effects on lake persistence).

```python
from lake_analysis import analyze_lake_halflife

results = analyze_lake_halflife(
    lakes_with_aridity,
    glacial_stage_col='glacial_stage',
    aridity_col='AI',
    verbose=True
)
```

**Outputs:**
- Half-life by aridity class (arid, semi-arid, humid)
- Statistical test for aridity × glacial stage interaction
- Hypothesis: Arid climate accelerates lake infilling

**Use case:** Separate climate effects from glacial age effects

---

### 7. Advanced Power Law Hypothesis Testing

Comprehensive statistical tests on power law fitting beyond standard analysis.

```python
from lake_analysis.powerlaw_analysis import (
    run_all_hypothesis_tests,
    compare_xmin_methods,
    test_alpha_robustness
)

# Full hypothesis testing suite
test_results = run_all_hypothesis_tests(
    lakes,
    elevation_bands=[(0, 500), (500, 1000), (1000, 1500), (1500, 3000)]
)

# Compare xmin estimation methods
xmin_comparison = compare_xmin_methods(lakes)

# Test α robustness to xmin choice
robustness = test_alpha_robustness(lakes, xmin_range=(0.01, 10.0))
```

**Outputs:**
- Goodness-of-fit tests (KS, likelihood ratio)
- xmin method comparison (KS minimization vs likelihood)
- α stability analysis
- Sample size power calculations

**Use case:** Rigorous statistical validation of power law claims

---

## Quick Reference by Research Question

### Q1: What controls where lakes occur?

**Core Hypotheses (H1-H6):** Run `run_full_analysis()` Steps 2-7

- **H1 (Elevation bimodality):** Low-elevation floodplains + high-elevation glacial zones
- **H2 (Slope threshold):** Lakes require low slopes for retention
- **H3 (Relief controls):** Intermediate relief optimal (not too flat, not too steep)
- **H4 (Process domains):** 2D elevation × slope reveals distinct geomorphic settings
- **H6 (Slope-relief):** Similar to H4 but slope × relief space

**Key Functions:**
- `analyze_elevation(lakes)`
- `analyze_slope(lakes)`
- `analyze_relief(lakes)`
- `analyze_2d_domains(lakes)`
- `analyze_slope_relief(lakes)`

---

### Q2: Do lake size distributions follow power laws?

**Power Law Analysis (H5):** Run `run_full_analysis()` Step 6

- Fit power law: P(A > a) ∝ a^(-α)
- MLE estimation with bootstrap confidence intervals
- Goodness-of-fit testing (KS statistic)
- Comparison to Cael & Seekell (2016) global power law

**Key Functions:**
- `analyze_powerlaw(lakes)` - Main analysis
- `full_powerlaw_analysis(lakes)` - Comprehensive with all diagnostics
- `fit_powerlaw_by_elevation_bands(lakes)` - Test if α varies with elevation

**Advanced:**
- `xmin_sensitivity_analysis(lakes)` - Test robustness to threshold choice
- `run_all_hypothesis_tests(lakes)` - Complete statistical validation

---

### Q3: Does lake density decrease with landscape age (Davis's hypothesis)?

**Glacial Chronosequence:** Run `run_full_analysis()` Step 12 or standalone

**Three-stage analysis (Wisconsin/Illinoian/Driftless):**
```python
results = analyze_glacial_chronosequence(lakes)
```

**High-resolution time slices (0-25 ka with NADI-1):**
```python
results = analyze_nadi1_chronosequence(lakes, use_bayesian=True)
```

**Expected Result:**
- Wisconsin (~20 ka): ~228 lakes per 1000 km²
- Illinoian (~160 ka): ~95 lakes per 1000 km²
- Driftless (>1.5 Ma): ~69 lakes per 1000 km²
- **Half-life:** ~660 ka (estimated from exponential decay)

**Key Functions:**
- `analyze_glacial_chronosequence()` - 3-stage analysis
- `analyze_nadi1_chronosequence()` - 50 time slices
- `fit_bayesian_decay_model()` - Exponential decay: D(t) = D₀ × exp(-k × t)

---

### Q4: Do small lakes disappear faster than large lakes?

**Size-Stratified Half-Life:** Run `run_full_analysis()` Step 13 or standalone

```python
results = analyze_bayesian_halflife(
    lakes_classified,
    run_size_stratified=True,
    min_lakes_per_class=10
)
```

**Hypothesis:** Small lakes have shorter half-lives due to:
- Higher sedimentation rates (smaller catchments)
- Higher evaporative concentration (larger perimeter/volume)
- Easier for vegetation to colonize

**Outputs:**
- Half-life estimates for 7 size classes: [0.05-0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-2.5, 2.5-10, >10 km²]
- Statistical test: Is t½ correlated with lake size?
- Detection limit diagnostics (mapping biases)

**Key Functions:**
- `run_size_stratified_analysis()` - Complete pipeline
- `fit_size_stratified_halflife_models()` - Bayesian fitting per size class
- `test_halflife_size_relationship()` - Statistical correlation test

---

### Q5: Do lake patterns vary geographically?

**Spatial Scaling:** Run `run_full_analysis()` Step 14

```python
results = analyze_spatial_scaling(lakes)
```

**Tests:**
- **Latitudinal gradient:** Does lake density vary with latitude?
- **Longitudinal gradient:** East-west patterns
- **Elevation gradient:** Density vs elevation (global pattern)
- **Glacial vs non-glacial:** Regional differences

**Outputs:**
- Latitudinal/longitudinal density curves
- Statistical tests for gradients
- Glacial vs non-glacial comparison

**Key Functions:**
- `analyze_latitudinal_scaling()` - Latitude patterns
- `analyze_longitudinal_scaling()` - Longitude patterns
- `compare_glacial_vs_nonglacial_scaling()` - Regional differences

---

### Q6: How does climate (aridity) affect lake density and persistence?

**Aridity Analysis:** Run `run_full_analysis()` Step 15

```python
results = analyze_aridity(lakes_with_aridity)
```

**Tests:**
- Aridity vs lake density
- Aridity × glacial stage interaction
- Aridity-conditional half-life

**Outputs:**
- Density heatmap: aridity × glacial stage
- Half-life by aridity class
- Statistical interaction tests

**Key Functions:**
- `analyze_aridity()` - Density by aridity
- `analyze_aridity_effects()` - Detailed analysis
- `analyze_lake_halflife()` - Aridity-conditional half-life

---

## Module-by-Module Function List

### data_loading.py (22 functions)

**Lake Data:**
- `load_lake_data_from_gdb()` - Load from File Geodatabase
- `load_lake_data_from_parquet()` - Load from parquet (faster)
- `load_conus_lake_data()` - Load pre-processed CONUS dataset
- `create_conus_lake_dataset()` - Create CONUS parquet from GDB
- `export_gdb_to_parquet()` - Convert GDB to parquet

**Raster Operations:**
- `load_raster()` - Load single raster
- `load_raster_chunked()` - Memory-efficient chunked loading
- `calculate_landscape_area_by_bin()` - **KEY:** Compute available area per bin (normalization denominator)
- `sample_raster_at_points()` - Extract values at point locations
- `sample_raster_at_coords()` - Extract values at lat/lon

**Quality Control:**
- `apply_lake_quality_filters()` - Filter based on area, elevation, etc.
- `check_raster_alignment()` - CRS/projection compatibility
- `quick_data_check()` - Verify data availability

---

### normalization.py (10 functions)

**Density Computation:**
- `compute_1d_normalized_density()` - Single-variable normalization
- `compute_2d_normalized_density()` - Joint 2-variable normalization
- `compute_1d_density_with_size_classes()` - 1D with lake size stratification

**Domain Analysis:**
- `classify_lake_domains()` - Domain classification (geomorphic types)
- `compute_domain_statistics()` - Per-domain statistics
- `compute_residuals_after_covariate()` - Remove covariate effects

---

### powerlaw_analysis.py (22 functions)

**Core Fitting:**
- `estimate_alpha_mle()` - MLE power law exponent
- `estimate_xmin()` - Find optimal power law threshold
- `full_powerlaw_analysis()` - Complete pipeline with uncertainty & hypothesis tests

**Domain-Specific:**
- `fit_powerlaw_by_elevation_bands()` - Fit by elevation class
- `fit_powerlaw_by_domain()` - Fit separate power laws by domain

**Sensitivity & Diagnostics:**
- `xmin_sensitivity_analysis()` - Test sensitivity to xmin choice
- `xmin_sensitivity_by_elevation()` - xmin sensitivity across elevation bands
- `compare_xmin_methods()` - Compare different xmin estimation methods
- `test_alpha_robustness()` - α robustness to xmin choice

**Hypothesis Testing:**
- `run_all_hypothesis_tests()` - Comprehensive hypothesis testing suite
- `bootstrap_pvalue()` - Goodness-of-fit test
- `compare_distributions()` - Power law vs exponential/lognormal

---

### glacial_chronosequence.py (73 functions)

**Data Loading:**
- `load_wisconsin_extent()` - Wisconsin glaciation boundary
- `load_illinoian_extent()` - Illinoian glaciation boundary
- `load_driftless_area()` - Unglaciated area
- `load_southern_appalachian_lakes()` - Non-glacial comparison region
- `load_dalton_18ka()` - Dalton et al. ice sheet reconstruction
- `load_nadi1_time_slice()` - NADI-1 time slices (0.5 ka resolution)
- `load_all_glacial_boundaries()` - Load all boundaries with options

**Lake Classification:**
- `convert_lakes_to_gdf()` - Convert DataFrame to GeoDataFrame
- `classify_lakes_by_glacial_extent()` - Assign lakes to glacial stages
- `classify_ice_types()` - Alpine vs continental classification

**Density Computation:**
- `compute_lake_density_by_glacial_stage()` - Raw and normalized density
- `compute_density_by_deglaciation_age()` - Density by NADI-1 age bins
- `compute_sapp_hypsometry_normalized_density()` - S. Appalachian hypsometry

**Analysis Pipelines:**
- `run_glacial_chronosequence_analysis()` - Wisconsin/Illinoian/Driftless pipeline
- `run_nadi1_chronosequence_analysis()` - NADI-1 time slice pipeline
- `run_dalton_18ka_analysis()` - Dalton 18ka analysis
- `western_alpine_analysis()` - Alpine lake analysis

**Bayesian Modeling:**
- `fit_bayesian_decay_model()` - PyMC Bayesian exponential decay

**Hypothesis Testing:**
- `test_davis_hypothesis()` - Statistical test of Davis's landscape maturity hypothesis
- `compare_adjacent_stages()` - Pairwise comparisons between glacial stages

---

### size_stratified_analysis.py (9 functions)

**Pipeline:**
- `run_size_stratified_analysis()` - Complete size-stratified analysis

**Density & Half-Life:**
- `calculate_size_stratified_densities()` - Density by size class × glacial stage
- `fit_size_stratified_halflife_models()` - Bayesian decay per size class
- `fit_overall_bayesian_halflife()` - Single Bayesian decay (not size-stratified)

**Diagnostics:**
- `detection_limit_diagnostics()` - QA/QC for mapping biases
- `test_halflife_size_relationship()` - Statistical test: t½ ~ size

**Visualization:**
- `plot_size_stratified_densities()` - Size-density patterns
- `plot_bayesian_halflife_results()` - Halflife vs size
- `plot_overall_bayesian_halflife()` - Overall halflife

---

### spatial_scaling.py (6 functions)

**Geographic Patterns:**
- `analyze_latitudinal_scaling()` - Lake density vs latitude
- `analyze_longitudinal_scaling()` - Lake density vs longitude
- `compare_glacial_vs_nonglacial_scaling()` - Glacial region differences
- `analyze_elevation_size_scaling()` - Lake size vs elevation patterns

**Pipeline:**
- `run_spatial_scaling_analysis()` - Complete spatial analysis
- `create_hypothesis_summary_table()` - Consolidated results table

---

### visualization.py (74 plotting functions)

**Univariate:**
- `plot_raw_vs_normalized()` - Shows normalization impact
- `plot_1d_density()` - Single-variable density
- `plot_multiple_1d_densities()` - Compare multiple variables

**2D Analysis:**
- `plot_2d_heatmap()` - Joint density heatmap
- `plot_2d_heatmap_with_contours()` - Heatmap + contour lines
- `plot_2d_contour_with_domains()` - Domain classification overlay

**Power Law:**
- `plot_powerlaw_rank_size()` - Rank-size distribution
- `plot_powerlaw_ccdf()` - Complementary CDF
- `plot_powerlaw_by_elevation_multipanel()` - Power law across elevation bands
- `plot_xmin_sensitivity_by_elevation()` - xmin vs elevation

**Glacial Chronosequence:**
- `plot_density_by_glacial_stage()` - Density comparison across stages
- `plot_glacial_extent_map()` - Map of glacial boundaries
- `plot_nadi1_chronosequence()` - Time slice density results
- `plot_nadi1_density_decay()` - **KEY OUTPUT:** Density decay with Bayesian fit

**Bayesian:**
- `plot_bayesian_posteriors()` - Parameter posteriors
- `plot_bayesian_decay_curves()` - Decay curves with credible intervals
- `plot_bayesian_summary()` - 4-panel Bayesian summary
- `plot_halflife_threshold_sensitivity()` - Halflife sensitivity to threshold

**Spatial & Aridity:**
- `plot_latitudinal_scaling()` - Latitude patterns
- `plot_aridity_glacial_heatmap()` - Aridity × glacial stage
- `plot_sapp_hypsometry_normalized_density()` - S. Appalachian hypsometry

---

### main.py (39 functions)

**Core Hypotheses:**
- `analyze_elevation()` - H1: Bimodal elevation distribution
- `analyze_slope()` - H2: Slope threshold
- `analyze_relief()` - H3: Relief controls
- `analyze_2d_domains()` - H4: 2D elevation × slope domains
- `analyze_powerlaw()` - H5: Power law analysis
- `analyze_slope_relief()` - H6: Slope-relief domains

**Advanced Analyses:**
- `analyze_glacial_chronosequence()` - Davis's hypothesis
- `analyze_nadi1_chronosequence()` - NADI-1 time slices
- `analyze_bayesian_halflife()` - Overall + size-stratified half-life
- `analyze_spatial_scaling()` - Geographic patterns
- `analyze_aridity()` - Aridity vs glacial stage

**Main Pipeline:**
- `run_full_analysis()` - **MAIN ENTRY POINT:** Complete 16-step pipeline

**Utilities:**
- `load_data()` - Load lakes from various sources
- `quick_start()` - Quick demonstration script
- `print_analysis_summary()` - Formatted results output

---

## Example Workflows

### Example 1: Complete Standard Analysis

Run all core hypotheses and optional analyses:

```python
from lake_analysis import run_full_analysis

results = run_full_analysis(
    data_source='conus',
    include_xmin_by_elevation=True,
    include_glacial_analysis=True,
    include_bayesian_halflife=True,
    include_spatial_scaling=True,
    include_aridity_analysis=True,
    min_lake_area=0.01
)

# Results contains:
# - lakes: Raw data
# - H1_elevation, H2_slope, H3_relief, H4_2d, H5_powerlaw, H6_slope_relief
# - glacial_chronosequence
# - bayesian_halflife (overall + size_stratified)
# - spatial_scaling
# - aridity_analysis
```

---

### Example 2: Glacial Chronosequence Only

Focus on Davis's hypothesis with high-resolution time slices:

```python
from lake_analysis import (
    load_conus_lake_data,
    analyze_nadi1_chronosequence
)

# Load lakes
lakes = load_conus_lake_data()

# Run NADI-1 chronosequence (0-25 ka, 50 time slices)
results = analyze_nadi1_chronosequence(
    lakes,
    max_lake_area=20000,        # Exclude Great Lakes
    min_lake_area=0.01,         # Minimum lake size
    use_bayesian=True,          # Bayesian exponential decay
    compare_with_illinoian=True,  # Add deep time end members
    verbose=True
)

# Outputs:
# - Density vs deglaciation age plot
# - Bayesian decay curve with credible intervals
# - Half-life estimate: t½ ≈ 660 ka [95% CI: 418-1505 ka]
```

---

### Example 3: Size-Stratified Half-Life with Threshold Sensitivity

Test if small lakes disappear faster, with threshold diagnostics:

```python
from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    analyze_bayesian_halflife
)

# Load and classify
lakes = load_conus_lake_data()
lakes_gdf = convert_lakes_to_gdf(lakes)
boundaries = load_all_glacial_boundaries()
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

# Run with threshold sensitivity
results = analyze_bayesian_halflife(
    lakes_classified,
    min_lake_area=0.01,         # Primary threshold
    run_overall=True,           # Overall half-life
    run_size_stratified=True,   # Size-stratified analysis
    test_thresholds=True,       # Test multiple thresholds
    include_sapp=True,          # Include S. Appalachian comparison
    generate_map=True,
    save_figures=True
)

# Outputs:
# - bayesian_overall_halflife.png
# - size_stratified_*.png (3 figures)
# - halflife_threshold_sensitivity.png
# - bayesian_glacial_extent_map.png
```

---

### Example 4: Power Law Analysis with Advanced Diagnostics

Comprehensive power law analysis with hypothesis testing:

```python
from lake_analysis import load_conus_lake_data
from lake_analysis.powerlaw_analysis import (
    full_powerlaw_analysis,
    xmin_sensitivity_analysis,
    run_all_hypothesis_tests,
    fit_powerlaw_by_elevation_bands
)

lakes = load_conus_lake_data()

# Full power law analysis with uncertainty
powerlaw_results = full_powerlaw_analysis(
    lakes,
    xmin=0.1,               # Minimum lake size for power law
    bootstrap_iterations=1000
)

# Test if α varies with elevation
elevation_results = fit_powerlaw_by_elevation_bands(
    lakes,
    elevation_bands=[(0, 500), (500, 1000), (1000, 1500), (1500, 3000)]
)

# xmin sensitivity
sensitivity = xmin_sensitivity_analysis(
    lakes,
    xmin_range=(0.01, 10.0),
    n_points=50
)

# Full hypothesis testing suite
hypothesis_tests = run_all_hypothesis_tests(lakes)
```

---

### Example 5: Southern Appalachian Non-Glacial Comparison

Control analysis for non-glacial highlands:

```python
from lake_analysis.glacial_chronosequence import (
    load_southern_appalachian_lakes,
    compute_sapp_hypsometry_normalized_density
)
import numpy as np

# Load S. Appalachian lakes
sapp_lakes = load_southern_appalachian_lakes()

# Compute hypsometry-normalized density
# (lakes per 1000 km² at each elevation band)
sapp_results = compute_sapp_hypsometry_normalized_density(
    sapp_lakes,
    elev_breaks=np.arange(0, 2001, 100),  # 100m bins, 0-2000m
    verbose=True
)

# Compare to glacial regions
# Expected: Different elevation-density pattern due to different lake formation processes
```

---

### Example 6: Aridity-Conditional Half-Life

Test if climate affects lake persistence:

```python
from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    analyze_lake_halflife
)

# Load and classify (with aridity data)
lakes = load_conus_lake_data()
lakes_gdf = convert_lakes_to_gdf(lakes)
boundaries = load_all_glacial_boundaries()
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

# Aridity-conditional half-life
results = analyze_lake_halflife(
    lakes_classified,
    glacial_stage_col='glacial_stage',
    aridity_col='AI',
    verbose=True
)

# Hypothesis: Arid climates → faster lake infilling → shorter half-life
```

---

## Summary Statistics

**Total Functions:** ~200+ analysis functions across 10 core modules

**Main Pipeline:** 16 steps (12 required + 4 optional)

**Standalone Analyses:** 7 major analyses not in main pipeline:
1. NADI-1 chronosequence (0-25 ka)
2. Dalton 18ka ice sheet reconstruction
3. Western alpine analysis
4. S. Appalachian comparison
5. Half-life threshold sensitivity
6. Aridity-conditional half-life
7. Advanced power law hypothesis testing

**Visualization Functions:** 74 plotting functions

**Analysis Types:**
- Univariate density normalization (3 variables)
- Bivariate process domain analysis (3 combinations)
- Power law fitting (with uncertainty quantification)
- Glacial chronosequence (3-stage or 50 time slices)
- Bayesian half-life (overall + size-stratified)
- Spatial scaling (lat/lon/elevation)
- Aridity × glacial stage interaction
- Alpine vs continental ice classification

---

## Key References

- **Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).** Power-law distributions in empirical data. *SIAM Review*, 51(4), 661-703.
- **Cael, B. B., & Seekell, D. A. (2016).** The size-distribution of Earth's lakes. *Scientific Reports*, 6, 29633.
- **Dalton, A. S., et al. (2020).** An updated radiocarbon-based ice margin chronology for the last deglaciation of the North American Ice Sheet Complex. *Quaternary Science Reviews*, 234, 106223.
- **Davis, W. M. (1899).** The geographical cycle. *Geographical Journal*, 14(5), 481-504.

---

## Contact

Project maintained by morrismc. For questions:
- **Glacial chronosequence:** See comments in `glacial_chronosequence.py`
- **Power law analysis:** See comments in `powerlaw_analysis.py`
- **General workflow:** See `CLAUDE.md` for project context

**Last Updated:** 2026-01-19
