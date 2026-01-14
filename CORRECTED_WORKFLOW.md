# CORRECTED: Size-Stratified Lake Half-Life Analysis

## ✅ SIMPLIFIED WORKFLOW (Using Direct Parquet Loading)

This is the **correct** way to load your existing parquet file and run the analysis.

---

## Complete Working Code

```python
# =============================================================================
# CORRECTED SIZE-STRATIFIED ANALYSIS - Direct Parquet Loading
# =============================================================================

from lake_analysis import (
    load_lake_data_from_parquet,        # ← Direct parquet loader
    load_wisconsin_extent,
    load_illinoian_extent,
    load_driftless_area,
    convert_lakes_to_gdf,
    classify_lakes_by_glacial_extent,
    run_size_stratified_analysis
)

# -----------------------------------------------------------------------------
# STEP 1: Load Lake Data from Parquet (FAST & DIRECT)
# -----------------------------------------------------------------------------
print("Step 1: Loading lakes from parquet...")

# Option A: Use default path from config (F:\Lakes\Data\lakes.parquet)
lakes = load_lake_data_from_parquet()

# Option B: Specify path explicitly
# lakes = load_lake_data_from_parquet(r"F:\Lakes\Data\lakes_conus.parquet")

print(f"✓ Loaded {len(lakes):,} lakes")

# -----------------------------------------------------------------------------
# STEP 2: Convert DataFrame → GeoDataFrame
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

# Print summary
print("\nClassification Summary:")
stage_counts = lakes_classified['glacial_stage'].value_counts()
for stage, count in stage_counts.items():
    pct = 100 * count / len(lakes_classified)
    print(f"  {stage:15s}: {count:8,} lakes ({pct:5.1f}%)")

# -----------------------------------------------------------------------------
# STEP 5: Run Size-Stratified Analysis
# -----------------------------------------------------------------------------
print("\nStep 5: Running size-stratified analysis...")
print("This will take 30-60 minutes...")

results = run_size_stratified_analysis(
    lakes_classified,
    min_lake_area=0.05,      # 0.05 km² minimum
    max_lake_area=20000,     # Exclude Great Lakes
    min_lakes_per_class=10   # Need ≥10 lakes per size class
)

# -----------------------------------------------------------------------------
# STEP 6: View Results
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Density results
print("\nDensity Results:")
print(results['density_df'])

# Half-life results (if PyMC installed)
if results['halflife_df'] is not None:
    print("\nHalf-Life Results:")
    print(results['halflife_df'][['size_class', 'n_lakes_total',
                                   'halflife_median', 'halflife_ci_low',
                                   'halflife_ci_high']])

    # Statistical tests
    if results['statistics'] is not None:
        stats = results['statistics']
        print(f"\nStatistical Test:")
        print(f"  Spearman ρ = {stats['spearman_rho']:.3f}")
        print(f"  p-value = {stats['spearman_p']:.4f}")

        if stats['spearman_p'] < 0.05 and stats['spearman_rho'] > 0:
            print("\n  ✓ SIGNIFICANT: Larger lakes persist longer!")
        else:
            print("\n  - No significant relationship detected")

print("\nOutput files saved to: F:\\Lakes\\Analysis\\outputs\\")
```

---

## Key Differences from Previous Version

### ❌ BEFORE (Wrong)
```python
from lake_analysis import load_data  # Too complex, tries to create files

lakes = load_data()  # Triggers CONUS file creation logic
```

### ✅ AFTER (Correct)
```python
from lake_analysis import load_lake_data_from_parquet  # Direct!

lakes = load_lake_data_from_parquet()  # Just loads existing parquet
```

---

## Why This is Better

1. **Simpler**: Direct loading, no intermediate file creation
2. **Faster**: No checking for file existence or creation
3. **Clearer**: Explicitly loads from parquet
4. **More Control**: Can specify exact parquet path if needed

---

## File Paths

The function will look for parquet files at:
- **Default**: `F:\Lakes\Data\lakes.parquet`
- **CONUS only**: `F:\Lakes\Data\lakes_conus.parquet`

To use a specific file:
```python
lakes = load_lake_data_from_parquet(r"F:\Lakes\Data\lakes_conus.parquet")
```

---

## Expected Runtime

With ~4.9M lakes:
- **Step 1 (Loading)**: 10-30 seconds
- **Step 2 (Conversion)**: 1-2 minutes
- **Step 3 (Boundaries)**: 5-10 seconds
- **Step 4 (Classification)**: 15-30 minutes (spatial joins)
- **Step 5 (Analysis)**: 15-30 minutes (Bayesian sampling)
- **Total**: ~30-60 minutes

---

## Outputs

After completion, check:
```
F:\Lakes\Analysis\outputs\
├── detection_limit_diagnostics.png       (6 panels)
├── size_stratified_density_patterns.png  (4 panels)
├── size_stratified_bayesian_results.png  (4 panels - KEY FIGURE!)
├── size_stratified_density.csv
└── size_stratified_halflife_results.csv
```

---

## Quick Test (Faster)

To test with a subset first:
```python
# Load only first 10,000 lakes
lakes = load_lake_data_from_parquet()
lakes_subset = lakes.head(10000)

# Continue with rest of workflow...
lakes_gdf = convert_lakes_to_gdf(lakes_subset)
# ... etc
```

---

## Troubleshooting

### "FileNotFoundError: parquet file not found"
**Check**: Does `F:\Lakes\Data\lakes.parquet` or `F:\Lakes\Data\lakes_conus.parquet` exist?

**Solution**: Specify the correct path:
```python
lakes = load_lake_data_from_parquet(r"F:\Lakes\Data\lakes_conus.parquet")
```

### "AttributeError: 'DataFrame' object has no attribute 'crs'"
**Problem**: Forgot Step 2 (conversion to GeoDataFrame)

**Solution**: Always run `convert_lakes_to_gdf()` before classification

### "No size classes have sufficient data"
**Problem**: Not enough lakes classified in each stage

**Solution**:
- Check Step 4 output - should show thousands of lakes per stage
- Reduce `min_lakes_per_class` parameter
- Check that boundaries loaded correctly

---

## Summary

**This corrected workflow**:
✅ Loads directly from your existing parquet file
✅ Avoids unnecessary file creation logic
✅ Is simpler and more explicit
✅ Gives you full control over which parquet file to use

Copy the code above into Spyder and run!
