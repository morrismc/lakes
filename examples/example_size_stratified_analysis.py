"""
Example: Size-Stratified Lake Half-Life Analysis
=================================================

This script demonstrates how to use the size-stratified analysis module to
test whether lake half-lives vary by lake size across glacial chronosequence.

Scientific Question:
    Do small lakes have shorter half-lives than large lakes?

This analysis:
1. Loads lake data and classifies lakes by glacial stage
2. Runs detection limit diagnostics to check for mapping biases
3. Calculates size-stratified lake densities
4. Fits Bayesian exponential decay models for each size class
5. Tests for statistical relationship between lake size and half-life
6. Generates comprehensive visualizations

Requirements:
    - NHD lake data loaded and processed
    - Glacial boundary shapefiles (Wisconsin, Illinoian, Driftless)
    - PyMC installed for Bayesian analysis: pip install pymc arviz

Author: Lake Analysis Project
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lake_analysis import (
    # Data loading
    load_lake_data_from_parquet,

    # Glacial classification
    load_wisconsin_extent,
    load_illinoian_extent,
    load_driftless_area,
    classify_lakes_by_glacial_extent,

    # Size-stratified analysis
    run_size_stratified_analysis,

    # Configuration
    LAKE_CONUS_PARQUET_PATH,
    OUTPUT_DIR,
    COLS,
    SIZE_STRATIFIED_BINS,
    SIZE_STRATIFIED_LANDSCAPE_AREAS,
    SIZE_STRATIFIED_AGE_ESTIMATES,
    SIZE_STRATIFIED_BAYESIAN_PARAMS,
    SIZE_STRATIFIED_MIN_LAKES,
    ensure_output_dir
)


def main():
    """
    Run complete size-stratified lake half-life analysis.
    """

    print("=" * 80)
    print("SIZE-STRATIFIED LAKE HALF-LIFE ANALYSIS")
    print("=" * 80)
    print("\nThis analysis tests whether lake persistence (half-life) varies with")
    print("lake size across the Wisconsin-Illinoian-Driftless chronosequence.")
    print("")

    # Ensure output directory exists
    output_dir = ensure_output_dir()
    print(f"Output directory: {output_dir}")

    # ========================================================================
    # STEP 1: Load lake data
    # ========================================================================

    print("\n" + "-" * 80)
    print("STEP 1: Loading lake data")
    print("-" * 80)

    try:
        lakes = load_lake_data_from_parquet(LAKE_CONUS_PARQUET_PATH)
        print(f"Loaded {len(lakes):,} lakes from CONUS dataset")
        print(f"Area range: {lakes[COLS['area']].min():.4f} - {lakes[COLS['area']].max():.1f} km²")
    except FileNotFoundError:
        print(f"ERROR: Could not find lake data at {LAKE_CONUS_PARQUET_PATH}")
        print("Please ensure the CONUS lake parquet file exists.")
        print("You can create it using the data loading utilities.")
        return

    # ========================================================================
    # STEP 2: Load glacial boundaries and classify lakes
    # ========================================================================

    print("\n" + "-" * 80)
    print("STEP 2: Loading glacial boundaries and classifying lakes")
    print("-" * 80)

    boundaries = {}

    # Load Wisconsin extent
    print("Loading Wisconsin glaciation extent...")
    try:
        boundaries['wisconsin'] = load_wisconsin_extent()
        print(f"  Loaded Wisconsin boundary")
    except Exception as e:
        print(f"  WARNING: Could not load Wisconsin boundary: {e}")

    # Load Illinoian extent
    print("Loading Illinoian glaciation extent...")
    try:
        boundaries['illinoian'] = load_illinoian_extent()
        print(f"  Loaded Illinoian boundary")
    except Exception as e:
        print(f"  WARNING: Could not load Illinoian boundary: {e}")

    # Load Driftless Area
    print("Loading Driftless Area (never glaciated)...")
    try:
        boundaries['driftless'] = load_driftless_area()
        print(f"  Loaded Driftless boundary")
    except Exception as e:
        print(f"  WARNING: Could not load Driftless boundary: {e}")

    if len(boundaries) < 3:
        print("\nERROR: Not all glacial boundaries could be loaded.")
        print("Please check your configuration and ensure the boundary shapefiles exist.")
        return

    # Classify lakes by glacial extent
    print("\nClassifying lakes by glacial stage...")
    lakes = classify_lakes_by_glacial_extent(lakes, boundaries, verbose=True)

    # Print classification summary
    print("\nGlacial stage classification:")
    stage_counts = lakes['glacial_stage'].value_counts()
    for stage, count in stage_counts.items():
        pct = 100 * count / len(lakes)
        print(f"  {stage:15s}: {count:8,} lakes ({pct:5.1f}%)")

    # ========================================================================
    # STEP 3: Filter lakes for analysis
    # ========================================================================

    print("\n" + "-" * 80)
    print("STEP 3: Filtering lakes for analysis")
    print("-" * 80)

    # Filter to relevant glacial stages
    relevant_stages = ['Wisconsin', 'Illinoian', 'Driftless']
    lakes_filtered = lakes[lakes['glacial_stage'].isin(relevant_stages)].copy()

    print(f"Filtered to {len(lakes_filtered):,} lakes in Wisconsin, Illinoian, or Driftless areas")

    # Apply minimum lake area filter
    min_lake_area = 0.05  # km²
    lakes_filtered = lakes_filtered[lakes_filtered[COLS['area']] >= min_lake_area].copy()

    print(f"After min area filter (≥{min_lake_area} km²): {len(lakes_filtered):,} lakes")

    # Optional: Apply maximum lake area filter to exclude Great Lakes
    # This is recommended to avoid their dominance in density calculations
    max_lake_area = 20000  # km²
    lakes_filtered = lakes_filtered[lakes_filtered[COLS['area']] < max_lake_area].copy()

    print(f"After max area filter (<{max_lake_area} km²): {len(lakes_filtered):,} lakes")

    # ========================================================================
    # STEP 4: Calculate landscape areas (optional - can use defaults)
    # ========================================================================

    print("\n" + "-" * 80)
    print("STEP 4: Landscape areas")
    print("-" * 80)

    # You can calculate actual landscape areas from boundaries if desired
    # For now, we'll use the default values from configuration
    landscape_areas = SIZE_STRATIFIED_LANDSCAPE_AREAS.copy()

    print("Using landscape areas:")
    for stage, area in landscape_areas.items():
        print(f"  {stage:12s}: {area:10,.0f} km²")

    # ========================================================================
    # STEP 5: Run size-stratified analysis
    # ========================================================================

    print("\n" + "-" * 80)
    print("STEP 5: Running size-stratified analysis")
    print("-" * 80)

    # Configuration
    print("\nAnalysis configuration:")
    print(f"  Size bins: {len(SIZE_STRATIFIED_BINS)} classes")
    for low, high, label in SIZE_STRATIFIED_BINS:
        print(f"    {label:15s}: {low:.2f} - {high:.2f} km²")

    print(f"\n  Minimum lakes per size class: {SIZE_STRATIFIED_MIN_LAKES}")
    print(f"  Bayesian sampling:")
    print(f"    Chains: {SIZE_STRATIFIED_BAYESIAN_PARAMS['n_chains']}")
    print(f"    Samples per chain: {SIZE_STRATIFIED_BAYESIAN_PARAMS['n_samples']}")
    print(f"    Tune samples: {SIZE_STRATIFIED_BAYESIAN_PARAMS['n_tune']}")

    # Run the analysis
    print("\nRunning analysis (this may take several minutes)...")
    print("")

    results = run_size_stratified_analysis(
        lakes_filtered,
        landscape_areas=landscape_areas,
        age_estimates=SIZE_STRATIFIED_AGE_ESTIMATES,
        size_bins=SIZE_STRATIFIED_BINS,
        bayesian_params=SIZE_STRATIFIED_BAYESIAN_PARAMS,
        area_col=COLS['area'],
        stage_col='glacial_stage',
        min_lake_area=min_lake_area,
        min_lakes_per_class=SIZE_STRATIFIED_MIN_LAKES,
        output_dir=output_dir,
        verbose=True
    )

    # ========================================================================
    # STEP 6: Summarize results
    # ========================================================================

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Density results
    print("\nDensity calculations:")
    print(f"  Total size classes × stages analyzed: {len(results['density_df'])}")

    # Half-life results
    if results['halflife_df'] is not None:
        print(f"\nHalf-life estimates:")
        print(f"  Size classes with sufficient data: {len(results['halflife_df'])}")
        print("")
        print(results['halflife_df'][['size_class', 'n_lakes_total', 'halflife_median',
                                      'halflife_ci_low', 'halflife_ci_high']].to_string(index=False))

        # Statistical tests
        if results['statistics'] is not None:
            print("\nStatistical tests:")
            stats = results['statistics']
            print(f"  Spearman ρ = {stats['spearman_rho']:.3f} (p = {stats['spearman_p']:.4f})")
            print(f"  Power-law exponent: {stats['power_law_exponent']:.2f} ± {stats['power_law_exponent_se']:.2f}")

            if stats['spearman_p'] < 0.05:
                if stats['spearman_rho'] > 0:
                    print("\n  ✓ Significant positive relationship: Larger lakes persist longer")
                else:
                    print("\n  ✗ Significant negative relationship: Smaller lakes persist longer")
            else:
                print("\n  - No significant relationship detected")
    else:
        print("\nNo size classes had sufficient data for half-life estimation")

    # Output files
    print("\n" + "-" * 80)
    print("Output files saved to:")
    print(f"  {output_dir}/")
    print("  - detection_limit_diagnostics.png")
    print("  - size_stratified_density_patterns.png")
    print("  - size_stratified_density.csv")
    if results['halflife_df'] is not None:
        print("  - size_stratified_bayesian_results.png")
        print("  - size_stratified_halflife_results.csv")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return results


# =============================================================================
# ALTERNATIVE: Step-by-step analysis (for customization)
# =============================================================================

def run_step_by_step_analysis():
    """
    Alternative approach showing how to run each analysis step individually.
    This gives you more control over parameters and intermediate results.
    """

    from lake_analysis.size_stratified_analysis import (
        detection_limit_diagnostics,
        calculate_size_stratified_densities,
        plot_size_stratified_densities,
        fit_size_stratified_halflife_models,
        plot_bayesian_halflife_results,
        test_halflife_size_relationship
    )

    print("Running step-by-step size-stratified analysis...")

    # Step 1: Load and prepare data (same as above)
    # ... (omitted for brevity)

    # Step 2: Run detection diagnostics
    fig_diag, diagnostics = detection_limit_diagnostics(
        lakes_filtered,
        area_col=COLS['area'],
        stage_col='glacial_stage',
        min_lake_area=0.05,
        verbose=True
    )

    # Step 3: Calculate densities with custom size bins
    custom_bins = [
        (0.1, 0.5, 'small'),
        (0.5, 2.0, 'medium'),
        (2.0, 10.0, 'large')
    ]

    density_df = calculate_size_stratified_densities(
        lakes_filtered,
        landscape_areas=SIZE_STRATIFIED_LANDSCAPE_AREAS,
        age_estimates=SIZE_STRATIFIED_AGE_ESTIMATES,
        area_col=COLS['area'],
        stage_col='glacial_stage',
        size_bins=custom_bins,
        verbose=True
    )

    # Step 4: Plot densities
    fig_density = plot_size_stratified_densities(density_df, verbose=True)

    # Step 5: Fit Bayesian models
    results_df, traces = fit_size_stratified_halflife_models(
        density_df,
        bayesian_params=SIZE_STRATIFIED_BAYESIAN_PARAMS,
        min_lakes=20,  # Custom threshold
        verbose=True
    )

    # Step 6: Plot Bayesian results
    if results_df is not None:
        fig_bayes = plot_bayesian_halflife_results(
            results_df, density_df, traces, verbose=True
        )

        # Step 7: Statistical tests
        stats = test_halflife_size_relationship(results_df, verbose=True)

    print("\nStep-by-step analysis complete!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run size-stratified lake half-life analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with defaults
  python example_size_stratified_analysis.py

  # Run step-by-step analysis for more control
  python example_size_stratified_analysis.py --step-by-step
        """
    )

    parser.add_argument(
        '--step-by-step',
        action='store_true',
        help='Run step-by-step analysis instead of pipeline'
    )

    args = parser.parse_args()

    if args.step_by_step:
        print("Running step-by-step analysis mode...")
        print("(Note: This requires manual editing of the script to work)")
        print("See the run_step_by_step_analysis() function for details.")
    else:
        results = main()
