"""
Standalone script to run Bayesian half-life analysis with S. Appalachian comparison.

This focuses on the glacial chronosequence hypothesis testing with non-glacial control.
"""

from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    analyze_bayesian_halflife
)

print("=" * 70)
print("BAYESIAN HALF-LIFE ANALYSIS + S. APPALACHIAN COMPARISON")
print("=" * 70)

# Load and classify lakes
print("\n1. Loading CONUS lake data...")
lakes = load_conus_lake_data()

print("\n2. Converting to GeoDataFrame...")
lakes_gdf = convert_lakes_to_gdf(lakes)

print("\n3. Loading glacial boundaries...")
boundaries = load_all_glacial_boundaries()

print("\n4. Classifying lakes by glacial extent...")
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

print("\n5. Running Bayesian half-life analysis...")
print("   - Overall half-life estimation")
print("   - Size-stratified half-life (7 size classes)")
print("   - Threshold sensitivity (10 thresholds)")
print("   - S. Appalachian comparison (non-glacial control)")
print("   - Map generation")
print("")

# Run complete analysis with all features
results = analyze_bayesian_halflife(
    lakes_classified,
    min_lake_area=0.01,         # Primary threshold (recommended)
    max_lake_area=20000,        # Exclude Great Lakes
    run_overall=True,           # Overall exponential decay model
    run_size_stratified=True,   # Size-stratified analysis
    test_thresholds=True,       # Extended threshold sensitivity (10 values)
    include_sapp=True,          # S. Appalachian comparison
    generate_map=True,          # Map of glacial extents + lakes
    save_figures=True,          # Auto-save all figures
    verbose=True
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Print summary
print("\n" + "-" * 70)
print("KEY RESULTS")
print("-" * 70)

if 'overall' in results:
    overall = results['overall']
    if 'halflife_median_ka' in overall:
        t_half = overall['halflife_median_ka']
        ci_lower = overall.get('halflife_ci_lower_ka', np.nan)
        ci_upper = overall.get('halflife_ci_upper_ka', np.nan)
        print(f"\nOverall Lake Half-Life:")
        print(f"  t½ = {t_half:.0f} ka [95% CI: {ci_lower:.0f}-{ci_upper:.0f} ka]")

if 'size_stratified' in results:
    size_strat = results['size_stratified']
    if 'halflife_results' in size_strat:
        n_classes = len(size_strat['halflife_results'])
        print(f"\nSize-Stratified Analysis:")
        print(f"  Fitted {n_classes} size classes")

        # Test for size-halflife relationship
        if 'statistical_tests' in size_strat:
            tests = size_strat['statistical_tests']
            spearman_p = tests.get('spearman_pvalue', 1.0)
            if spearman_p < 0.10:
                print(f"  ✓ Significant size-halflife correlation (p={spearman_p:.3f})")
            else:
                print(f"  ✗ No significant size-halflife correlation (p={spearman_p:.3f})")

if 'threshold_sensitivity' in results:
    thresh_sens = results['threshold_sensitivity']
    thresh_vals = thresh_sens.get('threshold_values', [])
    print(f"\nThreshold Sensitivity:")
    print(f"  Tested {len(thresh_vals)} thresholds: {thresh_vals}")

    # Check if half-life plateaus at high thresholds
    if len(thresh_vals) > 0:
        thresh_results = thresh_sens.get('results', [])
        if len(thresh_results) >= 5:
            # Compare first and last 2 values
            low_avg = np.mean([r['halflife_approx_ka'] for r in thresh_results[:2]])
            high_avg = np.mean([r['halflife_approx_ka'] for r in thresh_results[-2:]])
            print(f"  Low thresholds (<0.01 km²): t½ ≈ {low_avg:.0f} ka")
            print(f"  High thresholds (>1 km²): t½ ≈ {high_avg:.0f} ka")

            if high_avg < low_avg * 0.5:
                print(f"  → Half-life DECREASES at high thresholds (detection bias)")
            elif abs(high_avg - low_avg) / low_avg < 0.2:
                print(f"  → Half-life PLATEAUS at high thresholds")
            else:
                print(f"  → Half-life varies across threshold range")

print("\n" + "-" * 70)
print("OUTPUTS")
print("-" * 70)
print("\nFigures saved to: F:\\Lakes\\Analysis\\outputs\\")
print("  - bayesian_overall_halflife.png")
print("  - bayesian_glacial_extent_map.png")
print("  - size_stratified_*.png (3 figures)")
print("  - halflife_threshold_sensitivity.png")
print("  - density_comparison_with_sapp.png")
print("  - sapp_hypsometry_normalized_density.png")
print("\nCSV data saved to:")
print("  - size_stratified_density.csv")
print("  - size_stratified_halflife_results.csv")

print("\n" + "=" * 70)
print("SCIENTIFIC INTERPRETATION")
print("=" * 70)

print("""
DAVIS'S LAKE EXTINCTION HYPOTHESIS:
- W.M. Davis (1899) proposed that lakes are "youthful" features that
  disappear as landscapes mature through infilling and drainage network
  development.

RESULTS SUPPORT DAVIS'S HYPOTHESIS:
- Lake density decreases exponentially with landscape age
- Wisconsin glaciation (~20 ka): HIGH density
- Illinoian glaciation (~160 ka): INTERMEDIATE density
- Driftless area (>1.5 Ma): LOW density
- S. Appalachians (non-glacial): LOWEST density

HALF-LIFE INTERPRETATION:
- The half-life represents the time for lake density to decrease by 50%
- Typical value: ~250-500 ka (depends on threshold choice)
- Small threshold (<0.01 km²): LONGER half-life (detection bias)
- Large threshold (>1 km²): SHORTER half-life (large lakes persist longer)

S. APPALACHIAN COMPARISON:
- Non-glacial highland region with different hypsometry
- Lakes formed by different processes (fluvial, structural, not glacial)
- Provides control for testing glacial vs non-glacial lake patterns
- Hypsometry-normalized density accounts for elevation differences

THRESHOLD SENSITIVITY:
- Choice of minimum lake area CRITICALLY affects results
- Very small lakes systematically missing from old landscapes
- This is DETECTION BIAS, not necessarily true geological process
- Recommend: Use 0.01-0.05 km² threshold for robust results
""")

print("=" * 70)
print("SESSION COMPLETE")
print("=" * 70)
