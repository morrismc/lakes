"""
Half-Life Threshold Sensitivity Analysis
=========================================

This example demonstrates how different minimum lake area thresholds
affect the estimated lake half-life.

Key Finding:
  - 0.005 km²: ~858 ka (includes more small lakes)
  - 0.010 km²: ~661 ka (original threshold)
  - 0.050 km²: ~169 ka (larger lakes only)

This reveals systematic detection bias - small lakes are preferentially
lost in older landscapes.

Usage:
    python example_halflife_sensitivity.py

Outputs:
    - halflife_threshold_sensitivity.png (4-panel diagnostic)
    - Console output showing half-life for each threshold
"""

from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    analyze_bayesian_halflife
)

print("\n" + "=" * 70)
print("HALF-LIFE THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 70)

# Step 1: Load and classify lakes
print("\n[STEP 1/3] Loading and classifying lakes...")
lakes = load_conus_lake_data()
lakes_gdf = convert_lakes_to_gdf(lakes)
boundaries = load_all_glacial_boundaries(include_dalton=False, include_sapp=False)
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries, verbose=False)
print(f"  Classified {len(lakes_classified):,} lakes")

# Step 2: Run Bayesian analysis with threshold sensitivity testing
print("\n[STEP 2/3] Running threshold sensitivity analysis...")
print("  This tests how half-life varies with min_lake_area threshold")
print("  Testing thresholds: 0.005, 0.01, 0.024, 0.05, 0.1 km²")

results = analyze_bayesian_halflife(
    lakes_classified,
    run_overall=False,           # Skip main analysis (runs in threshold test)
    run_size_stratified=False,
    test_thresholds=True,        # Enable threshold sensitivity
    save_figures=True,
    verbose=False
)

# Step 3: Display results
print("\n[STEP 3/3] Threshold Sensitivity Results")
print("=" * 70)

if 'threshold_sensitivity' in results:
    sensitivity = results['threshold_sensitivity']

    print("\nThreshold   | Lakes (W)  | Lakes (I)  | Density (W) | Density (I) | Half-Life")
    print("-" * 85)

    for r in sensitivity['results']:
        print(f"{r['threshold']:>10.3f}  | {r['wisc_count']:>10,} | {r['ill_count']:>10,} | "
              f"{r['wisc_density']:>11.1f} | {r['ill_density']:>11.1f} | {r['halflife_approx_ka']:>8.0f} ka")

    # Find optimal threshold (closest to 661 ka target)
    best = min(sensitivity['results'], key=lambda x: abs(x['halflife_approx_ka'] - 661))

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\nOptimal threshold (closest to 661 ka): {best['threshold']:.3f} km²")
    print(f"  Half-life: {best['halflife_approx_ka']:.0f} ka")
    print(f"  Wisconsin lakes: {best['wisc_count']:,}")
    print(f"  Illinoian lakes: {best['ill_count']:,}")

    print("\nInterpretation:")
    print("  - Smaller thresholds give LONGER half-lives (more small lakes preserved in young landscapes)")
    print("  - Larger thresholds give SHORTER half-lives (small lakes missing in old landscapes)")
    print("  - This reveals preferential loss of small lakes over geologic time")

    print("\nVisualization saved to: output/halflife_threshold_sensitivity.png")
else:
    print("ERROR: Threshold sensitivity results not found")

print("\n" + "=" * 70)
