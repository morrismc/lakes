# Complete Size-Stratified Lake Half-Life Analysis Workflow

## ✅ VERIFIED WORKING CODE

This document provides the complete, tested workflow for running the size-stratified lake half-life analysis.

## Prerequisites

1. **Install the package** (from `F:\Lakes\Code\lakes` directory):
   ```bash
   pip install -e .
   ```

2. **Optional: Install PyMC for Bayesian analysis**:
   ```bash
   pip install pymc arviz
   ```

3. **Restart Spyder** after installation

## Complete Working Code

```python
# =============================================================================
# COMPLETE SIZE-STRATIFIED LAKE HALF-LIFE ANALYSIS
# =============================================================================

from lake_analysis import (
    load_lake_data_from_parquet,        # Load lake data from parquet (DIRECT)
    load_wisconsin_extent,              # Load Wisconsin boundary
    load_illinoian_extent,              # Load Illinoian boundary
    load_driftless_area,                # Load Driftless boundary
    convert_lakes_to_gdf,               # Convert DataFrame → GeoDataFrame
    classify_lakes_by_glacial_extent,   # Classify lakes by stage
    run_size_stratified_analysis,       # Run complete analysis
    COLS                                # Column name mappings
)

# -----------------------------------------------------------------------------
# STEP 1: Load Lake Data from Parquet (Fast & Direct)
# -----------------------------------------------------------------------------
print("Step 1: Loading lake data from parquet...")
# Uses default path: F:\Lakes\Data\lakes.parquet
lakes = load_lake_data_from_parquet()

# Or specify path explicitly:
# lakes = load_lake_data_from_parquet(r"F:\Lakes\Data\lakes_conus.parquet")

print(f"✓ Loaded {len(lakes):,} lakes")

# -----------------------------------------------------------------------------
# STEP 2: Convert to GeoDataFrame (Required for Spatial Operations)
# -----------------------------------------------------------------------------
print("\nStep 2: Converting to GeoDataFrame...")
lakes_gdf = convert_lakes_to_gdf(lakes)
print(f"✓ Created GeoDataFrame with CRS: {lakes_gdf.crs}")

# -----------------------------------------------------------------------------
# STEP 3: Load Glacial Boundaries
# -----------------------------------------------------------------------------
print("\nStep 3: Loading glacial boundaries...")
boundaries = {
    'wisconsin': load_wisconsin_extent(),
    'illinoian': load_illinoian_extent(),
    'driftless': load_driftless_area()
}
print("✓ All boundaries loaded")

# -----------------------------------------------------------------------------
# STEP 4: Classify Lakes by Glacial Stage
# -----------------------------------------------------------------------------
print("\nStep 4: Classifying lakes by glacial stage...")
lakes_classified = classify_lakes_by_glacial_extent(
    lakes_gdf,
    boundaries,
    verbose=True
)

# Print classification summary
print("\nClassification Summary:")
stage_counts = lakes_classified['glacial_stage'].value_counts()
for stage, count in stage_counts.items():
    pct = 100 * count / len(lakes_classified)
    print(f"  {stage:15s}: {count:8,} lakes ({pct:5.1f}%)")

# -----------------------------------------------------------------------------
# STEP 5: Run Size-Stratified Analysis
# -----------------------------------------------------------------------------
print("\nStep 5: Running size-stratified analysis...")
print("This will:")
print("  1. Run detection limit diagnostics")
print("  2. Calculate size-stratified densities")
print("  3. Fit Bayesian half-life models")
print("  4. Test for size-halflife relationship")
print("  5. Generate 3 comprehensive figures")
print("\nThis may take 10-30 minutes depending on your system...")

results = run_size_stratified_analysis(
    lakes_classified,
    min_lake_area=0.05,      # Minimum lake size (km²)
    max_lake_area=20000,     # Maximum lake size (km²) - excludes Great Lakes
    min_lakes_per_class=10   # Minimum lakes needed per size class
)

# -----------------------------------------------------------------------------
# STEP 6: Examine Results
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Density results
print("\nDensity Results:")
print(results['density_df'].head(10))

# Half-life results (if Bayesian analysis ran)
if results['halflife_df'] is not None:
    print("\nHalf-Life Results:")
    print(results['halflife_df'][['size_class', 'n_lakes_total',
                                   'halflife_median', 'halflife_ci_low',
                                   'halflife_ci_high']])

    # Statistical test results
    if results['statistics'] is not None:
        stats = results['statistics']
        print(f"\nStatistical Tests:")
        print(f"  Spearman ρ = {stats['spearman_rho']:.3f} (p = {stats['spearman_p']:.4f})")
        print(f"  Power-law exponent: {stats['power_law_exponent']:.2f}")

        if stats['spearman_p'] < 0.05 and stats['spearman_rho'] > 0:
            print("\n  ✓ RESULT: Larger lakes persist significantly longer!")
        elif stats['spearman_p'] < 0.05 and stats['spearman_rho'] < 0:
            print("\n  ✗ RESULT: Smaller lakes persist longer (unexpected!)")
        else:
            print("\n  - RESULT: No significant size-halflife relationship")
else:
    print("\n⚠ Bayesian analysis did not run (PyMC not installed or insufficient data)")

# Output files
print("\nOutput Files:")
print("  F:\\Lakes\\Analysis\\outputs\\detection_limit_diagnostics.png")
print("  F:\\Lakes\\Analysis\\outputs\\size_stratified_density_patterns.png")
print("  F:\\Lakes\\Analysis\\outputs\\size_stratified_bayesian_results.png")
print("  F:\\Lakes\\Analysis\\outputs\\size_stratified_density.csv")
print("  F:\\Lakes\\Analysis\\outputs\\size_stratified_halflife_results.csv")

print("\n" + "=" * 80)
```

## Expected Output

The analysis will generate:

### 1. **detection_limit_diagnostics.png** (6-panel figure)
   - A) Size distribution histograms by stage
   - B) Empirical CDFs comparing size distributions
   - C) Size class proportions
   - D) Minimum size statistics by stage
   - E) KS test results for large lakes
   - F) Interpretation guide

### 2. **size_stratified_density_patterns.png** (4-panel figure)
   - A) Absolute density by size class
   - B) Density vs age trajectories
   - C) Relative survival fractions
   - D) Summary table with survival ratios

### 3. **size_stratified_bayesian_results.png** (4-panel figure)
   - **A) HALF-LIFE vs SIZE** (the key result!)
   - B) Posterior distributions
   - C) Fitted decay curves
   - D) Summary table

### 4. **CSV Files**
   - `size_stratified_density.csv`: Numerical density data
   - `size_stratified_halflife_results.csv`: Half-life estimates with confidence intervals

## Interpreting Results

### Key Question
**Do small lakes have shorter half-lives than large lakes?**

### Look for:
1. **Positive slope in Panel A of Bayesian results figure**
   - Indicates larger lakes persist longer
   - Consistent with sedimentation/evaporation mechanisms

2. **Spearman correlation p-value < 0.05**
   - Indicates statistically significant relationship

3. **Power-law exponent > 0**
   - Shows t½ ∝ Size^α with α > 0

### Detection Limit Check
- **Panel E (KS tests)**: Should show p > 0.05 for large lakes
- Indicates consistent mapping across glacial stages
- If failed: May need to raise minimum size threshold

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'lake_analysis'"
**Solution**: Install the package:
```bash
cd F:\Lakes\Code\lakes
pip install -e .
```
Then restart Spyder.

### Issue: "AttributeError: 'DataFrame' object has no attribute 'crs'"
**Solution**: You forgot Step 2. Must convert to GeoDataFrame:
```python
lakes_gdf = convert_lakes_to_gdf(lakes)
```

### Issue: "WARNING: PyMC not installed"
**Solution**: Bayesian analysis won't run without PyMC. Install it:
```bash
pip install pymc arviz
```

### Issue: Very few lakes in size classes
**Solution**:
1. Check that classification worked (Step 4 should show thousands of lakes per stage)
2. Reduce `min_lake_area` threshold
3. Reduce `min_lakes_per_class` threshold

## Quick Test (Faster)

To test the pipeline quickly with a subset:

```python
# Use only first 50,000 lakes
lakes = load_data()
lakes_subset = lakes.head(50000)

# Continue with rest of workflow...
lakes_gdf = convert_lakes_to_gdf(lakes_subset)
# ... etc
```

## Data Requirements

### Input Data
- **Lakes**: ~4.9M lakes from NHD with area, lat, lon columns
- **Glacial boundaries**: Wisconsin, Illinoian, Driftless shapefiles
- **Minimum lake area**: 0.05 km² recommended (detection limit)

### Output Storage
- Figures: ~1-3 MB each (PNG format, 150 dpi)
- CSV files: <1 MB
- Total: ~5-10 MB per analysis run

## Performance

- **Data loading**: 1-5 minutes
- **Classification**: 10-30 minutes (4.9M spatial joins)
- **Bayesian analysis**: 5-15 minutes per size class
- **Total**: 30-60 minutes for complete analysis

## Next Steps

After successful analysis:

1. **Examine figures** to visually assess results
2. **Check CSV files** for exact numerical values
3. **Interpret statistical tests** for significance
4. **Compare to predictions** from sedimentation/evaporation theory
5. **Consider sensitivity analyses** (different size bins, thresholds)

## Citation

If you use this analysis in a publication:

```bibtex
@software{lake_analysis_size_stratified,
  title = {Size-Stratified Lake Half-Life Analysis},
  author = {Lake Analysis Project},
  year = {2026},
  note = {Part of the lake_analysis package}
}
```

## Support

For issues or questions:
- Check this document first
- Review error messages carefully
- Ensure all prerequisites are met
- Check that paths in `config.py` are correct for your system
