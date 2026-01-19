"""
Run S. Appalachian Comparison Analysis
=======================================

This script runs the complete Bayesian half-life analysis with
S. Appalachian lakes as a non-glacial comparison region.

Usage:
    python run_sapp_comparison.py

Outputs:
    - density_comparison_with_sapp.png
    - sapp_hypsometry_normalized_density.png
    - bayesian_glacial_extent_map.png
    - bayesian_overall_halflife.png
"""

from lake_analysis import (
    load_conus_lake_data,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    analyze_bayesian_halflife
)

print("\n" + "=" * 70)
print("S. APPALACHIAN COMPARISON ANALYSIS")
print("=" * 70)

# Step 1: Load CONUS lakes
print("\n[STEP 1/3] Loading CONUS lake data...")
lakes = load_conus_lake_data()
print(f"  Loaded {len(lakes):,} lakes")

# Step 2: Load glacial boundaries and classify lakes
print("\n[STEP 2/3] Classifying lakes by glacial extent...")
boundaries = load_all_glacial_boundaries(
    include_dalton=False,
    include_sapp=False  # Don't load S. Apps here; loaded separately in analysis
)
lakes_classified = classify_lakes_by_glacial_extent(
    lakes,
    boundaries,
    verbose=True
)

# Step 3: Run Bayesian analysis WITH S. Appalachian comparison
print("\n[STEP 3/3] Running Bayesian half-life analysis with S. Apps comparison...")
results = analyze_bayesian_halflife(
    lakes_classified,
    run_overall=True,
    run_size_stratified=False,
    min_lake_area=0.01,          # Critical: use 0.01 for 661 ka half-life
    max_lake_area=20000,         # Exclude Great Lakes
    generate_map=True,           # Generate map showing all regions
    include_sapp=True,           # Include S. Appalachian comparison
    save_figures=True,
    verbose=True
)

# Display results
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

if 'density_comparison' in results:
    print("\nDensity Comparison (with S. Appalachians):")
    print(results['density_comparison'][
        ['glacial_stage', 'n_lakes', 'zone_area_km2', 'density_per_1000km2']
    ].to_string(index=False))

if results.get('overall'):
    overall = results['overall']
    halflife = overall.get('halflife_median', 'N/A')
    ci_low = overall.get('halflife_ci_low', 'N/A')
    ci_high = overall.get('halflife_ci_high', 'N/A')
    print(f"\nOverall Half-Life (glacial regions only):")
    print(f"  t½ = {halflife:.0f} ka [95% CI: {ci_low:.0f}-{ci_high:.0f} ka]")

if 'sapp_hypsometry' in results and results['sapp_hypsometry'] is not None:
    sapp = results['sapp_hypsometry']
    peak_idx = sapp['normalized_density'].idxmax()
    peak_elev_range = sapp.loc[peak_idx, 'bin_label']
    peak_density = sapp.loc[peak_idx, 'normalized_density']
    print(f"\nS. Appalachian Hypsometry-Normalized Density:")
    print(f"  Peak density: {peak_density:.2f} lakes/1000 km² at {peak_elev_range}")

print("\n" + "=" * 70)
print("Analysis complete! Check output/ directory for figures.")
print("=" * 70)
