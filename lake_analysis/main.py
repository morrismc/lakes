"""
Lake Distribution Analysis - Main Orchestration Script
=======================================================

This script provides the main entry point for running the lake distribution
analysis. It can be run directly or individual functions can be called
interactively in Spyder/IPython.

Usage:
    # Run full analysis
    python main.py

    # Or import and run specific analyses in Spyder:
    from main import *
    lakes = load_data()
    elev_results = analyze_elevation(lakes)

Project Hypotheses:
    H1: Elevation-normalized lake density is bimodal
    H2: Slope threshold exists for lake formation
    H3: Relief controls lake density at intermediate values
    H4: 2D normalization reveals process domains
    H5: Power law exponent varies by elevation domain
"""

import warnings
import os
import sys
from pathlib import Path

# Add module directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import project modules
from .config import (
    LAKE_GDB_PATH, LAKE_FEATURE_CLASS, RASTERS, OUTPUT_DIR,
    COLS, ELEV_BREAKS, SLOPE_BREAKS, RELIEF_BREAKS,
    POWERLAW_XMIN_THRESHOLD, ensure_output_dir, print_config_summary
)
from .data_loading import (
    load_lake_data_from_gdb, load_lake_data_from_parquet,
    calculate_landscape_area_by_bin, check_raster_alignment,
    summarize_lake_data, quick_data_check, get_raster_info
)
from .normalization import (
    compute_1d_normalized_density, compute_2d_normalized_density,
    compute_1d_density_with_size_classes, classify_lake_domains,
    compute_domain_statistics
)
from .visualization import (
    plot_raw_vs_normalized, plot_1d_density, plot_2d_heatmap,
    plot_powerlaw_rank_size, plot_domain_comparison,
    plot_bimodality_test, create_summary_figure, setup_plot_style,
    # Enhanced visualizations
    plot_powerlaw_by_elevation_multipanel, plot_powerlaw_overlay,
    plot_powerlaw_explained, plot_2d_heatmap_with_marginals,
    plot_2d_contour_with_domains, plot_lake_size_histogram_by_elevation,
    plot_cumulative_area_by_size, plot_geographic_density_map
)
from .powerlaw_analysis import (
    full_powerlaw_analysis, fit_powerlaw_by_elevation_bands,
    fit_powerlaw_by_domain
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(source='gdb'):
    """
    Load lake data from specified source.

    Parameters
    ----------
    source : str
        'gdb' for geodatabase, 'parquet' for parquet file

    Returns
    -------
    DataFrame or GeoDataFrame
    """
    print("\n" + "=" * 60)
    print("LOADING LAKE DATA")
    print("=" * 60)

    try:
        print(f"\n[STEP 1] Loading from {source}...")
        if source == 'gdb':
            lakes = load_lake_data_from_gdb()
        else:
            lakes = load_lake_data_from_parquet()

        print(f"\n[STEP 2] Generating summary statistics...")
        summarize_lake_data(lakes)

        print(f"\n[SUCCESS] Loaded {len(lakes):,} lakes successfully!")
        return lakes

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("  Please check the paths in config.py")
        raise
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        raise


# ============================================================================
# HYPOTHESIS 1: ELEVATION BIMODALITY
# ============================================================================

def analyze_elevation(lakes, raster_path=None, save_figures=True):
    """
    Test H1: Elevation-normalized lake density shows bimodal pattern.

    Steps:
    1. Calculate landscape area at each elevation bin
    2. Count lakes in each bin
    3. Compute normalized density
    4. Visualize raw vs normalized to show bimodality

    Parameters
    ----------
    lakes : DataFrame
    raster_path : str, optional
        Path to elevation raster (uses config default if None)
    save_figures : bool

    Returns
    -------
    dict
        Results including landscape_area, density DataFrame
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 1: ELEVATION-NORMALIZED BIMODALITY")
    print("=" * 60)

    try:
        if raster_path is None:
            raster_path = RASTERS.get('elevation')

        if raster_path is None or not os.path.exists(raster_path):
            print("[ERROR] Elevation raster not found.")
            print("  Please update RASTERS['elevation'] in config.py")
            return None

        # Get raster info
        print("\n[STEP 1/5] Getting raster metadata...")
        info = get_raster_info(raster_path)
        print(f"  Raster: {raster_path}")
        print(f"  Dimensions: {info['width']} x {info['height']}")
        print(f"  CRS: {info['crs']}")
        print(f"  Pixel area: {info['pixel_area_km2']:.6f} km²")

        # Calculate landscape area
        print("\n[STEP 2/5] Calculating landscape area by elevation bin...")
        print("  This may take several minutes for large rasters...")
        landscape_area = calculate_landscape_area_by_bin(raster_path, ELEV_BREAKS)
        print("  Landscape area calculation complete!")

        # Compute normalized density
        print("\n[STEP 3/5] Computing normalized lake density...")
        elev_col = COLS['elevation']
        density = compute_1d_normalized_density(
            lakes, elev_col, ELEV_BREAKS, landscape_area
        )
        print("  Density calculation complete!")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 4/5] Generating visualizations...")
            fig, axes = plot_raw_vs_normalized(
                density, 'Elevation', units='m',
                save_path=f"{OUTPUT_DIR}/H1_elevation_raw_vs_normalized.png"
            )
            plt.close(fig)

            fig, ax = plot_bimodality_test(
                density, 'Elevation', units='m',
                save_path=f"{OUTPUT_DIR}/H1_elevation_bimodality_test.png"
            )
            plt.close(fig)
            print("  Figures saved!")

        # Save data
        print("\n[STEP 5/5] Saving results to CSV...")
        density.to_csv(f"{OUTPUT_DIR}/H1_elevation_density.csv", index=False)
        landscape_area.to_csv(f"{OUTPUT_DIR}/H1_elevation_landscape_area.csv", index=False)
        print(f"  Results saved to {OUTPUT_DIR}/")

        # Identify peaks
        peaks = density.nlargest(3, 'normalized_density')
        print("\n[RESULTS] Top 3 density peaks:")
        for _, row in peaks.iterrows():
            print(f"  {row['bin_lower']:.0f}-{row['bin_upper']:.0f} m: "
                  f"{row['normalized_density']:.2f} lakes/1000 km²")

        print("\n[SUCCESS] H1 analysis complete!")

        return {
            'landscape_area': landscape_area,
            'density': density,
            'peaks': peaks,
        }

    except Exception as e:
        print(f"\n[ERROR] H1 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# HYPOTHESIS 2: SLOPE THRESHOLD
# ============================================================================

def analyze_slope(lakes, raster_path=None, save_figures=True):
    """
    Test H2: Lake density drops sharply above a critical slope threshold.

    Parameters
    ----------
    lakes : DataFrame
    raster_path : str, optional
    save_figures : bool

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: SLOPE THRESHOLD")
    print("=" * 60)

    try:
        if raster_path is None:
            raster_path = RASTERS.get('slope')

        if raster_path is None or not os.path.exists(raster_path):
            print("[ERROR] Slope raster not found.")
            print("  Please update RASTERS['slope'] in config.py")
            return None

        # Calculate landscape area
        print("\n[STEP 1/4] Calculating landscape area by slope bin...")
        print("  This may take several minutes for large rasters...")
        landscape_area = calculate_landscape_area_by_bin(raster_path, SLOPE_BREAKS)
        print("  Landscape area calculation complete!")

        # Compute normalized density
        print("\n[STEP 2/4] Computing normalized lake density...")
        slope_col = COLS['slope']
        density = compute_1d_normalized_density(
            lakes, slope_col, SLOPE_BREAKS, landscape_area
        )
        print("  Density calculation complete!")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 3/4] Generating visualizations...")
            fig, axes = plot_raw_vs_normalized(
                density, 'Slope', units='°',
                save_path=f"{OUTPUT_DIR}/H2_slope_raw_vs_normalized.png"
            )
            plt.close(fig)
            print("  Figure saved!")

        # Find threshold (steepest decline)
        print("\n[STEP 4/4] Identifying slope threshold...")
        density['density_change'] = density['normalized_density'].diff()
        threshold_idx = density['density_change'].idxmin()
        if threshold_idx is not None and not pd.isna(threshold_idx):
            threshold_slope = density.loc[threshold_idx, 'bin_lower']
            print(f"  Steepest density decline at: {threshold_slope}°")

        # Save data
        density.to_csv(f"{OUTPUT_DIR}/H2_slope_density.csv", index=False)
        print(f"  Results saved to {OUTPUT_DIR}/")

        print("\n[SUCCESS] H2 analysis complete!")

        return {
            'landscape_area': landscape_area,
            'density': density,
        }

    except Exception as e:
        print(f"\n[ERROR] H2 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# HYPOTHESIS 3: RELIEF CONTROLS
# ============================================================================

def analyze_relief(lakes, raster_path=None, save_figures=True):
    """
    Test H3: Lake density peaks at intermediate relief values.

    Parameters
    ----------
    lakes : DataFrame
    raster_path : str, optional
    save_figures : bool

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 3: RELIEF CONTROLS")
    print("=" * 60)

    try:
        if raster_path is None:
            raster_path = RASTERS.get('relief_5km')

        if raster_path is None or not os.path.exists(raster_path):
            print("[ERROR] Relief raster not found.")
            print("  Please update RASTERS['relief_5km'] in config.py")
            return None

        # Calculate landscape area
        print("\n[STEP 1/4] Calculating landscape area by relief bin...")
        print("  This may take several minutes for large rasters...")
        landscape_area = calculate_landscape_area_by_bin(raster_path, RELIEF_BREAKS)
        print("  Landscape area calculation complete!")

        # Compute normalized density
        print("\n[STEP 2/4] Computing normalized lake density...")
        relief_col = COLS.get('relief_5km', 'F5km_relief')
        density = compute_1d_normalized_density(
            lakes, relief_col, RELIEF_BREAKS, landscape_area
        )
        print("  Density calculation complete!")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 3/4] Generating visualizations...")
            fig, axes = plot_raw_vs_normalized(
                density, 'Relief (5km)', units='m',
                save_path=f"{OUTPUT_DIR}/H3_relief_raw_vs_normalized.png"
            )
            plt.close(fig)
            print("  Figure saved!")

        # Find peak (intermediate relief)
        print("\n[STEP 4/4] Identifying peak relief...")
        peak_idx = density['normalized_density'].idxmax()
        if peak_idx is not None:
            peak_relief = (density.loc[peak_idx, 'bin_lower'] +
                          density.loc[peak_idx, 'bin_upper']) / 2
            print(f"  Peak density at relief: {peak_relief:.0f} m")

        # Save data
        density.to_csv(f"{OUTPUT_DIR}/H3_relief_density.csv", index=False)
        print(f"  Results saved to {OUTPUT_DIR}/")

        print("\n[SUCCESS] H3 analysis complete!")

        return {
            'landscape_area': landscape_area,
            'density': density,
        }

    except Exception as e:
        print(f"\n[ERROR] H3 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# HYPOTHESIS 4: 2D PROCESS DOMAINS
# ============================================================================

def analyze_2d_domains(lakes, elev_raster=None, slope_raster=None,
                        save_figures=True, fine_resolution=True):
    """
    Test H4: 2D elevation × slope space reveals distinct process domains.

    Parameters
    ----------
    lakes : DataFrame
    elev_raster, slope_raster : str, optional
    save_figures : bool
    fine_resolution : bool
        If True, use finer bins for higher resolution heatmap

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 4: 2D PROCESS DOMAINS")
    print("=" * 60)

    try:
        if elev_raster is None:
            elev_raster = RASTERS.get('elevation')
        if slope_raster is None:
            slope_raster = RASTERS.get('slope')

        if not all([elev_raster, slope_raster]):
            print("[ERROR] Missing rasters for 2D analysis.")
            return None

        if not (os.path.exists(elev_raster) and os.path.exists(slope_raster)):
            print("[ERROR] Raster files not found.")
            return None

        # Check alignment
        print("\n[STEP 1/4] Checking raster alignment...")
        alignment = check_raster_alignment([elev_raster, slope_raster])
        if not alignment['aligned']:
            print("  WARNING: Rasters are not aligned!")
            print("  Consider reprojecting to common CRS before 2D analysis.")

        # Use finer bins if requested
        if fine_resolution:
            elev_breaks_2d = list(range(0, 4200, 100))  # 100m bins instead of 200m
            slope_breaks_2d = list(range(0, 46, 2))     # 2° bins instead of 3°
            print("  Using fine resolution: 100m elevation × 2° slope bins")
        else:
            elev_breaks_2d = ELEV_BREAKS
            slope_breaks_2d = SLOPE_BREAKS
            print("  Using standard resolution")

        # Compute 2D density
        elev_col = COLS['elevation']
        slope_col = COLS['slope']

        print("\n[STEP 2/4] Computing 2D normalized density...")
        print("  This may take several minutes for large rasters...")
        density_2d = compute_2d_normalized_density(
            lakes, elev_raster, slope_raster,
            elev_col, slope_col,
            elev_breaks_2d, slope_breaks_2d
        )
        print("  2D density calculation complete!")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 3/4] Generating visualizations...")

            # Standard heatmap
            fig, ax = plot_2d_heatmap(
                density_2d, elev_col, slope_col,
                var1_units='m', var2_units='°',
                save_path=f"{OUTPUT_DIR}/H4_elevation_slope_heatmap.png"
            )
            plt.close(fig)
            print("  Basic heatmap saved!")

            # Enhanced heatmap with marginal distributions
            fig, axes = plot_2d_heatmap_with_marginals(
                density_2d, elev_col, slope_col,
                var1_units='m', var2_units='°',
                save_path=f"{OUTPUT_DIR}/H4_heatmap_with_marginals.png"
            )
            plt.close(fig)
            print("  Heatmap with marginal PDFs saved!")

            # Contour plot with domain annotations
            fig, ax = plot_2d_contour_with_domains(
                density_2d, elev_col, slope_col,
                var1_units='m', var2_units='°',
                save_path=f"{OUTPUT_DIR}/H4_contour_with_domains.png"
            )
            plt.close(fig)
            print("  Contour plot with domains saved!")

        # Save data
        print("\n[STEP 4/4] Saving results...")
        density_2d.to_csv(f"{OUTPUT_DIR}/H4_2d_density.csv", index=False)
        print(f"  Results saved to {OUTPUT_DIR}/")

        print("\n[SUCCESS] H4 analysis complete!")

        return {
            'density_2d': density_2d,
        }

    except Exception as e:
        print(f"\n[ERROR] H4 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# HYPOTHESIS 5: POWER LAW BY DOMAIN
# ============================================================================

def analyze_powerlaw(lakes, by_elevation=True, save_figures=True):
    """
    Test H5: Power law exponent varies by elevation domain.

    Parameters
    ----------
    lakes : DataFrame
    by_elevation : bool
        If True, fit separate power laws for elevation bands
    save_figures : bool

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 5: POWER LAW ANALYSIS")
    print("=" * 60)

    try:
        area_col = COLS['area']
        areas = lakes[area_col].values

        # Overall power law
        print("\n[STEP 1/5] Fitting overall power law distribution...")
        overall_results = full_powerlaw_analysis(
            areas,
            xmin_threshold=POWERLAW_XMIN_THRESHOLD,
            run_bootstrap=True,
            n_bootstrap_sims=500
        )

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 2/5] Generating basic power law visualization...")
            fig, ax = plot_powerlaw_rank_size(
                areas,
                xmin=overall_results['xmin'],
                alpha=overall_results['alpha'],
                title="Lake Size Distribution (All Lakes)",
                save_path=f"{OUTPUT_DIR}/H5_powerlaw_overall.png"
            )
            plt.close(fig)
            print("  Rank-size plot saved!")

            # Educational plot explaining power law parameters
            print("\n[STEP 3/5] Generating power law explanation figure...")
            fig = plot_powerlaw_explained(
                overall_results,
                save_path=f"{OUTPUT_DIR}/H5_powerlaw_explained.png"
            )
            plt.close(fig)
            print("  Power law explanation figure saved!")

            # Cumulative area distribution
            fig, ax = plot_cumulative_area_by_size(
                lakes, area_col=area_col,
                save_path=f"{OUTPUT_DIR}/H5_cumulative_area.png"
            )
            plt.close(fig)
            print("  Cumulative area distribution saved!")

        # By elevation bands
        domain_results = None
        if by_elevation:
            print("\n[STEP 4/5] Fitting power law by elevation bands...")
            elev_bands = list(range(0, 3500, 500))  # Coarser bins for more data per bin
            domain_results = fit_powerlaw_by_elevation_bands(
                lakes, elev_bands,
                elev_column=COLS['elevation'],
                area_column=area_col
            )

            print("\n[RESULTS] Power law exponents by elevation:")
            for _, row in domain_results.iterrows():
                if pd.notna(row['alpha']):
                    print(f"  {row['domain']}: α = {row['alpha']:.3f} "
                          f"(n={row['n_tail']:.0f})")

            # Save tabular results
            domain_results.to_csv(f"{OUTPUT_DIR}/H5_powerlaw_by_elevation.csv", index=False)

            if save_figures:
                print("\n[STEP 5/5] Generating elevation-specific power law visualizations...")

                # Multi-panel: one CCDF per elevation band
                fig, axes = plot_powerlaw_by_elevation_multipanel(
                    lakes, elev_bands,
                    area_col=area_col, elev_col=COLS['elevation'],
                    save_path=f"{OUTPUT_DIR}/H5_powerlaw_multipanel.png"
                )
                plt.close(fig)
                print("  Multi-panel CCDF by elevation saved!")

                # Overlay: all elevation bands on one plot
                fig, ax = plot_powerlaw_overlay(
                    lakes, elev_bands,
                    area_col=area_col, elev_col=COLS['elevation'],
                    save_path=f"{OUTPUT_DIR}/H5_powerlaw_overlay.png"
                )
                plt.close(fig)
                print("  CCDF overlay comparison saved!")

                # Lake size histogram by elevation
                fig, ax = plot_lake_size_histogram_by_elevation(
                    lakes, elev_bands,
                    area_col=area_col, elev_col=COLS['elevation'],
                    save_path=f"{OUTPUT_DIR}/H5_size_histogram_by_elevation.png"
                )
                plt.close(fig)
                print("  Size histogram by elevation saved!")

            print(f"\n  All results saved to {OUTPUT_DIR}/")

        print("\n[SUCCESS] H5 analysis complete!")

        return {
            'overall': overall_results,
            'by_domain': domain_results,
        }

    except Exception as e:
        print(f"\n[ERROR] H5 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# DOMAIN CLASSIFICATION
# ============================================================================

def analyze_domains(lakes, save_figures=True):
    """
    Classify lakes into geomorphic process domains and compare.

    Parameters
    ----------
    lakes : DataFrame
    save_figures : bool

    Returns
    -------
    DataFrame with domain classification
    """
    print("\n" + "=" * 60)
    print("DOMAIN CLASSIFICATION")
    print("=" * 60)

    # Classify
    lakes_classified = classify_lake_domains(lakes)

    # Compute statistics
    domain_stats = compute_domain_statistics(lakes_classified)
    print("\nDomain Statistics:")
    print(domain_stats.to_string())

    # Visualize
    ensure_output_dir()

    if save_figures:
        fig, axes = plot_domain_comparison(
            lakes_classified, domain_col='domain',
            save_path=f"{OUTPUT_DIR}/domain_comparison.png"
        )
        plt.close(fig)

    # Save
    domain_stats.to_csv(f"{OUTPUT_DIR}/domain_statistics.csv", index=False)

    return lakes_classified


# ============================================================================
# FULL ANALYSIS PIPELINE
# ============================================================================

def run_full_analysis(data_source='gdb'):
    """
    Run complete analysis pipeline for all hypotheses.

    Parameters
    ----------
    data_source : str
        'gdb' or 'parquet'

    Returns
    -------
    dict
        All results
    """
    print("\n" + "=" * 70)
    print("LAKE DISTRIBUTION ANALYSIS - FULL PIPELINE")
    print("=" * 70)

    # Setup
    setup_plot_style()
    ensure_output_dir()
    print_config_summary()

    results = {}

    # Load data
    lakes = load_data(source=data_source)
    results['lakes'] = lakes

    # H1: Elevation bimodality
    results['H1_elevation'] = analyze_elevation(lakes)

    # H2: Slope threshold
    results['H2_slope'] = analyze_slope(lakes)

    # H3: Relief controls
    results['H3_relief'] = analyze_relief(lakes)

    # H4: 2D process domains (memory intensive - may skip)
    # results['H4_2d'] = analyze_2d_domains(lakes)

    # H5: Power law
    results['H5_powerlaw'] = analyze_powerlaw(lakes)

    # Domain classification
    results['domains'] = analyze_domains(lakes)

    # Create summary figure
    print("\nCreating summary figure...")
    if results.get('H1_elevation') and results.get('H2_slope'):
        fig = create_summary_figure(
            results['H1_elevation']['density'],
            results['H2_slope']['density'] if results.get('H2_slope') else None,
            results['H3_relief']['density'] if results.get('H3_relief') else None,
            save_path=f"{OUTPUT_DIR}/summary_figure.png"
        )
        plt.close(fig)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return results


# ============================================================================
# INTERACTIVE HELPERS
# ============================================================================

def generate_additional_visualizations(lakes, save_path_prefix=None):
    """
    Generate additional useful visualizations.

    Parameters
    ----------
    lakes : DataFrame
    save_path_prefix : str, optional
        Prefix for output files (uses OUTPUT_DIR if None)

    Returns
    -------
    dict
        Dictionary of figure objects
    """
    print("\n" + "=" * 60)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 60)

    ensure_output_dir()
    prefix = save_path_prefix or OUTPUT_DIR
    figures = {}

    try:
        # Geographic density map
        print("\n[1/3] Creating geographic density map...")
        fig, ax = plot_geographic_density_map(
            lakes,
            save_path=f"{prefix}/geographic_density_map.png"
        )
        if fig:
            figures['geographic_map'] = fig
            plt.close(fig)
            print("  Geographic density map saved!")

        # Cumulative area distribution
        print("\n[2/3] Creating cumulative area distribution...")
        fig, ax = plot_cumulative_area_by_size(
            lakes,
            save_path=f"{prefix}/cumulative_area_distribution.png"
        )
        figures['cumulative_area'] = fig
        plt.close(fig)
        print("  Cumulative area plot saved!")

        # Lake size histogram by elevation
        print("\n[3/3] Creating size histogram by elevation...")
        elev_bands = list(range(0, 3500, 500))
        fig, ax = plot_lake_size_histogram_by_elevation(
            lakes, elev_bands,
            save_path=f"{prefix}/size_histogram_by_elevation.png"
        )
        figures['size_histogram'] = fig
        plt.close(fig)
        print("  Size histogram by elevation saved!")

        print("\n[SUCCESS] Additional visualizations complete!")

    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()

    return figures


def quick_start():
    """
    Quick start guide for interactive use.
    """
    print("""
LAKE DISTRIBUTION ANALYSIS - Quick Start Guide
===============================================

1. Check data availability:
   >>> quick_data_check()

2. Load lake data:
   >>> lakes = load_data(source='gdb')  # or 'parquet'

3. Run individual analyses:
   >>> results = analyze_elevation(lakes)
   >>> results = analyze_slope(lakes)
   >>> results = analyze_relief(lakes)
   >>> results = analyze_powerlaw(lakes)    # Now with enhanced visualizations!
   >>> results = analyze_2d_domains(lakes)  # Now with marginal PDFs!

4. Run full pipeline:
   >>> all_results = run_full_analysis()

5. Classify domains:
   >>> lakes_classified = analyze_domains(lakes)

Tips:
- Update paths in config.py before running
- Use quick_data_check() to verify data accessibility
- Results are saved to OUTPUT_DIR specified in config.py
""")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Lake Distribution Analysis')
    parser.add_argument('--check', action='store_true',
                       help='Check data availability only')
    parser.add_argument('--source', choices=['gdb', 'parquet'], default='gdb',
                       help='Data source (default: gdb)')
    parser.add_argument('--hypothesis', type=int, choices=[1, 2, 3, 4, 5],
                       help='Run specific hypothesis test only')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis pipeline')

    args = parser.parse_args()

    if args.check:
        quick_data_check()
    elif args.full:
        run_full_analysis(data_source=args.source)
    elif args.hypothesis:
        lakes = load_data(source=args.source)
        if args.hypothesis == 1:
            analyze_elevation(lakes)
        elif args.hypothesis == 2:
            analyze_slope(lakes)
        elif args.hypothesis == 3:
            analyze_relief(lakes)
        elif args.hypothesis == 4:
            analyze_2d_domains(lakes)
        elif args.hypothesis == 5:
            analyze_powerlaw(lakes)
    else:
        quick_start()
