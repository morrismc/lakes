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
import time
from pathlib import Path
from contextlib import contextmanager
from datetime import timedelta

# Add module directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# RUNTIME TRACKING AND PROGRESS UTILITIES
# ============================================================================

class AnalysisTimer:
    """
    Track runtime for analysis steps with formatted output.

    Usage:
        timer = AnalysisTimer()
        timer.start("Loading data")
        # ... do work ...
        timer.stop()
        timer.summary()
    """

    def __init__(self):
        self.steps = []
        self.current_step = None
        self.start_time = None
        self.overall_start = None

    def start(self, step_name):
        """Start timing a new step."""
        if self.overall_start is None:
            self.overall_start = time.time()

        self.current_step = step_name
        self.start_time = time.time()

    def stop(self):
        """Stop timing current step and record."""
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        self.steps.append({
            'step': self.current_step,
            'duration': elapsed,
        })
        self.start_time = None
        self.current_step = None
        return elapsed

    def elapsed_str(self, seconds):
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return str(timedelta(seconds=int(seconds)))

    def summary(self):
        """Print summary of all step timings."""
        if not self.steps:
            print("\nNo timing data recorded.")
            return

        total = sum(s['duration'] for s in self.steps)
        overall = time.time() - self.overall_start if self.overall_start else total

        print("\n" + "=" * 60)
        print("RUNTIME SUMMARY")
        print("=" * 60)
        print(f"{'Step':<40} {'Duration':>15}")
        print("-" * 60)

        for step in self.steps:
            duration_str = self.elapsed_str(step['duration'])
            pct = (step['duration'] / total) * 100 if total > 0 else 0
            print(f"{step['step']:<40} {duration_str:>10} ({pct:>4.1f}%)")

        print("-" * 60)
        print(f"{'Total (all steps)':<40} {self.elapsed_str(total):>15}")
        print(f"{'Overall runtime':<40} {self.elapsed_str(overall):>15}")
        print("=" * 60)

        return {
            'steps': self.steps.copy(),
            'total': total,
            'overall': overall,
        }


@contextmanager
def timed_step(timer, step_name):
    """Context manager for timing analysis steps."""
    timer.start(step_name)
    try:
        yield
    finally:
        elapsed = timer.stop()
        if elapsed:
            print(f"  [DONE] {step_name} completed in {timer.elapsed_str(elapsed)}")


class ProgressBar:
    """
    Simple text-based progress bar for long-running operations.

    Usage:
        progress = ProgressBar(total=100, desc="Processing")
        for i in range(100):
            # do work
            progress.update()
        progress.close()
    """

    def __init__(self, total, desc="Progress", width=40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n=1):
        """Update progress by n steps."""
        self.current += n
        self._display()

    def _display(self):
        """Display current progress."""
        if self.total == 0:
            return

        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s" if eta < 60 else f"ETA: {eta/60:.1f}min"
        else:
            eta_str = "ETA: --"

        print(f"\r  {self.desc}: |{bar}| {self.current}/{self.total} ({pct*100:.0f}%) {eta_str}",
              end="", flush=True)

    def close(self):
        """Finish and print newline."""
        elapsed = time.time() - self.start_time
        print(f"\r  {self.desc}: Completed {self.total} items in {elapsed:.1f}s" + " " * 20)


def print_step_header(step_num, total_steps, title):
    """Print a formatted step header with progress."""
    bar_width = 30
    pct = step_num / total_steps
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\n[{bar}] Step {step_num}/{total_steps}")
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)


# Global timer instance for easy access
_global_timer = None

def get_timer():
    """Get or create the global timer instance."""
    global _global_timer
    if _global_timer is None:
        _global_timer = AnalysisTimer()
    return _global_timer

def reset_timer():
    """Reset the global timer."""
    global _global_timer
    _global_timer = AnalysisTimer()
    return _global_timer

# Import project modules - handle both package and direct execution
try:
    from .config import (
        LAKE_GDB_PATH, LAKE_FEATURE_CLASS, RASTERS, OUTPUT_DIR,
        COLS, ELEV_BREAKS, SLOPE_BREAKS, RELIEF_BREAKS,
        POWERLAW_XMIN_THRESHOLD, ensure_output_dir, print_config_summary
    )
    from .data_loading import (
        load_lake_data_from_gdb, load_lake_data_from_parquet,
        load_conus_lake_data, create_conus_lake_dataset,
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
        plot_cumulative_area_by_size, plot_geographic_density_map,
        # New visualizations
        plot_alpha_elevation_phase_diagram, plot_alpha_by_process_domain,
        plot_slope_relief_heatmap, plot_xmin_sensitivity,
        plot_significance_tests, plot_powerlaw_gof_summary,
        plot_three_panel_summary,
        # x_min sensitivity by elevation visualizations
        plot_xmin_sensitivity_by_elevation, plot_ks_curves_overlay,
        plot_optimal_xmin_vs_elevation, plot_alpha_stability_by_elevation,
        plot_xmin_elevation_summary, plot_alpha_vs_xmin_by_elevation
    )
    from .powerlaw_analysis import (
        full_powerlaw_analysis, fit_powerlaw_by_elevation_bands,
        fit_powerlaw_by_domain, fit_powerlaw_by_process_domain,
        xmin_sensitivity_analysis, compare_to_cael_seekell,
        # x_min sensitivity by elevation analysis
        xmin_sensitivity_by_elevation, compare_xmin_methods,
        test_alpha_robustness, generate_xmin_summary_table,
        # Hypothesis tests for x_min sensitivity
        run_all_hypothesis_tests, generate_hypothesis_test_report
    )
except ImportError:
    from config import (
        LAKE_GDB_PATH, LAKE_FEATURE_CLASS, RASTERS, OUTPUT_DIR,
        COLS, ELEV_BREAKS, SLOPE_BREAKS, RELIEF_BREAKS,
        POWERLAW_XMIN_THRESHOLD, ensure_output_dir, print_config_summary
    )
    from data_loading import (
        load_lake_data_from_gdb, load_lake_data_from_parquet,
        load_conus_lake_data, create_conus_lake_dataset,
        calculate_landscape_area_by_bin, check_raster_alignment,
        summarize_lake_data, quick_data_check, get_raster_info
    )
    from normalization import (
        compute_1d_normalized_density, compute_2d_normalized_density,
        compute_1d_density_with_size_classes, classify_lake_domains,
        compute_domain_statistics
    )
    from visualization import (
        plot_raw_vs_normalized, plot_1d_density, plot_2d_heatmap,
        plot_powerlaw_rank_size, plot_domain_comparison,
        plot_bimodality_test, create_summary_figure, setup_plot_style,
        # Enhanced visualizations
        plot_powerlaw_by_elevation_multipanel, plot_powerlaw_overlay,
        plot_powerlaw_explained, plot_2d_heatmap_with_marginals,
        plot_2d_contour_with_domains, plot_lake_size_histogram_by_elevation,
        plot_cumulative_area_by_size, plot_geographic_density_map,
        # New visualizations
        plot_alpha_elevation_phase_diagram, plot_alpha_by_process_domain,
        plot_slope_relief_heatmap, plot_xmin_sensitivity,
        plot_significance_tests, plot_powerlaw_gof_summary,
        plot_three_panel_summary,
        # x_min sensitivity by elevation visualizations
        plot_xmin_sensitivity_by_elevation, plot_ks_curves_overlay,
        plot_optimal_xmin_vs_elevation, plot_alpha_stability_by_elevation,
        plot_xmin_elevation_summary, plot_alpha_vs_xmin_by_elevation
    )
    from powerlaw_analysis import (
        full_powerlaw_analysis, fit_powerlaw_by_elevation_bands,
        fit_powerlaw_by_domain, fit_powerlaw_by_process_domain,
        xmin_sensitivity_analysis, compare_to_cael_seekell,
        # x_min sensitivity by elevation analysis
        xmin_sensitivity_by_elevation, compare_xmin_methods,
        test_alpha_robustness, generate_xmin_summary_table,
        # Hypothesis tests for x_min sensitivity
        run_all_hypothesis_tests, generate_hypothesis_test_report
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(source='conus'):
    """
    Load lake data from specified source.

    Parameters
    ----------
    source : str
        'conus' for CONUS-only parquet (recommended, fastest)
        'gdb' for full geodatabase
        'parquet' for full parquet file

    Returns
    -------
    DataFrame or GeoDataFrame
    """
    print("\n" + "=" * 60)
    print("LOADING LAKE DATA")
    print("=" * 60)

    try:
        print(f"\n[STEP 1] Loading from {source}...")
        if source == 'conus':
            lakes = load_conus_lake_data(create_if_missing=True)
        elif source == 'gdb':
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
# RELIEF VS ELEVATION 2D ANALYSIS
# ============================================================================

def analyze_relief_elevation(lakes, elev_raster=None, relief_raster=None,
                              save_figures=True, fine_resolution=True):
    """
    Analyze lake density in elevation × relief space.

    This reveals how lake density varies with topographic setting.

    Parameters
    ----------
    lakes : DataFrame
    elev_raster, relief_raster : str, optional
    save_figures : bool
    fine_resolution : bool

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("ELEVATION × RELIEF DOMAIN ANALYSIS")
    print("=" * 60)

    try:
        if elev_raster is None:
            elev_raster = RASTERS.get('elevation')
        if relief_raster is None:
            relief_raster = RASTERS.get('relief_5km')

        if not all([elev_raster, relief_raster]):
            print("[ERROR] Missing rasters for elevation-relief analysis.")
            return None

        if not (os.path.exists(elev_raster) and os.path.exists(relief_raster)):
            print("[ERROR] Raster files not found.")
            return None

        # Define bins
        if fine_resolution:
            elev_breaks = list(range(0, 4200, 100))   # 100m bins
            relief_breaks = list(range(0, 2100, 100))  # 100m bins
            print("  Using fine resolution: 100m elevation × 100m relief bins")
        else:
            elev_breaks = ELEV_BREAKS
            relief_breaks = RELIEF_BREAKS

        # Compute 2D density
        elev_col = COLS['elevation']
        relief_col = COLS.get('relief_5km', 'F5km_relief')

        print("\n[STEP 1/3] Computing 2D normalized density in elevation × relief space...")
        print("  This may take several minutes...")
        density_2d = compute_2d_normalized_density(
            lakes, elev_raster, relief_raster,
            elev_col, relief_col,
            elev_breaks, relief_breaks
        )
        print("  2D density calculation complete!")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 2/3] Generating visualizations...")

            # Relief-elevation heatmap with marginals
            fig, axes = plot_2d_heatmap_with_marginals(
                density_2d, elev_col, relief_col,
                var1_units='m', var2_units='m',
                title='Lake Density in Elevation × Relief Space',
                save_path=f"{OUTPUT_DIR}/elevation_relief_heatmap.png"
            )
            plt.close(fig)
            print("  Elevation-relief heatmap saved!")

        # Save data
        print("\n[STEP 3/3] Saving results...")
        density_2d.to_csv(f"{OUTPUT_DIR}/elevation_relief_density.csv", index=False)

        print("\n[SUCCESS] Elevation-relief analysis complete!")

        return {
            'density_2d': density_2d,
        }

    except Exception as e:
        print(f"\n[ERROR] Elevation-relief analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# HYPOTHESIS 6: SLOPE-RELIEF DOMAINS
# ============================================================================

def analyze_slope_relief(lakes, slope_raster=None, relief_raster=None,
                          save_figures=True, fine_resolution=True):
    """
    Analyze lake density in slope × relief space.

    This identifies "sweet spots" for lake formation - the optimal
    combination of local slope and relief.

    Parameters
    ----------
    lakes : DataFrame
    slope_raster, relief_raster : str, optional
    save_figures : bool
    fine_resolution : bool

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("SLOPE-RELIEF DOMAIN ANALYSIS")
    print("=" * 60)

    try:
        if slope_raster is None:
            slope_raster = RASTERS.get('slope')
        if relief_raster is None:
            relief_raster = RASTERS.get('relief_5km')

        if not all([slope_raster, relief_raster]):
            print("[ERROR] Missing rasters for slope-relief analysis.")
            return None

        if not (os.path.exists(slope_raster) and os.path.exists(relief_raster)):
            print("[ERROR] Raster files not found.")
            return None

        # Define bins
        if fine_resolution:
            slope_breaks = list(range(0, 46, 2))     # 2° bins
            relief_breaks = list(range(0, 2100, 100))  # 100m bins
            print("  Using fine resolution: 2° slope × 100m relief bins")
        else:
            slope_breaks = SLOPE_BREAKS
            relief_breaks = RELIEF_BREAKS

        # Compute 2D density
        slope_col = COLS['slope']
        relief_col = COLS.get('relief_5km', 'F5km_relief')

        print("\n[STEP 1/3] Computing 2D normalized density in slope × relief space...")
        print("  This may take several minutes...")
        density_2d = compute_2d_normalized_density(
            lakes, slope_raster, relief_raster,
            slope_col, relief_col,
            slope_breaks, relief_breaks
        )
        print("  2D density calculation complete!")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 2/3] Generating visualizations...")

            # Slope-relief heatmap with marginals
            fig, axes = plot_slope_relief_heatmap(
                density_2d, slope_col, relief_col,
                slope_units='°', relief_units='m',
                save_path=f"{OUTPUT_DIR}/slope_relief_heatmap.png"
            )
            plt.close(fig)
            print("  Slope-relief heatmap saved!")

        # Save data
        print("\n[STEP 3/3] Saving results...")
        density_2d.to_csv(f"{OUTPUT_DIR}/slope_relief_density.csv", index=False)

        print("\n[SUCCESS] Slope-relief analysis complete!")

        return {
            'density_2d': density_2d,
        }

    except Exception as e:
        print(f"\n[ERROR] Slope-relief analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# POWER LAW SENSITIVITY ANALYSIS
# ============================================================================

def analyze_powerlaw_sensitivity(lakes, save_figures=True):
    """
    Analyze sensitivity of power law parameters to x_min threshold.

    This is critical for understanding how conclusions depend on
    the choice of minimum lake area.

    Parameters
    ----------
    lakes : DataFrame
    save_figures : bool

    Returns
    -------
    dict
    """
    print("\n" + "=" * 60)
    print("POWER LAW SENSITIVITY ANALYSIS")
    print("=" * 60)

    try:
        area_col = COLS['area']
        areas = lakes[area_col].values
        areas = areas[areas > 0]

        # Run sensitivity analysis
        print("\n[STEP 1/4] Running x_min sensitivity analysis...")
        xmin_values = np.logspace(np.log10(0.01), np.log10(5.0), 25)
        sensitivity_results = xmin_sensitivity_analysis(
            areas, xmin_values=xmin_values, compute_uncertainty=True
        )

        # Compare to Cael & Seekell (2016)
        print("\n[STEP 2/4] Comparing to Cael & Seekell (2016) global result...")
        cs_comparison = compare_to_cael_seekell(areas, xmin=0.46)

        print(f"\n  CONUS α = {cs_comparison['alpha_conus']:.3f} ± {cs_comparison['alpha_se_conus']:.3f}")
        print(f"  Global α = {cs_comparison['alpha_global']:.3f} ± {cs_comparison['se_global']:.3f}")
        print(f"  Difference = {cs_comparison['difference']:.3f}")
        print(f"  p-value = {cs_comparison['p_value']:.4f}")
        print(f"  {cs_comparison.get('interpretation', '')}")

        # Fit by process domain
        print("\n[STEP 3/4] Fitting power law by process domain...")
        domain_results = fit_powerlaw_by_process_domain(
            lakes, fixed_xmin=0.46, compute_uncertainty=True
        )

        print("\n  Results by domain:")
        for _, row in domain_results.iterrows():
            if pd.notna(row['alpha']):
                print(f"    {row['domain']}: α = {row['alpha']:.3f} "
                      f"(95% CI: [{row.get('alpha_ci_lower', np.nan):.3f}, "
                      f"{row.get('alpha_ci_upper', np.nan):.3f}])")

        # Visualize
        ensure_output_dir()

        if save_figures:
            print("\n[STEP 4/4] Generating visualizations...")

            # Sensitivity plot
            fig, axes = plot_xmin_sensitivity(
                sensitivity_results,
                save_path=f"{OUTPUT_DIR}/powerlaw_xmin_sensitivity.png"
            )
            plt.close(fig)
            print("  x_min sensitivity plot saved!")

            # α-Elevation phase diagram
            elev_bands = list(range(0, 3500, 500))
            elev_domain_results = fit_powerlaw_by_elevation_bands(
                lakes, elev_bands,
                elev_column=COLS['elevation'],
                area_column=area_col
            )

            fig, ax = plot_alpha_elevation_phase_diagram(
                elev_domain_results,
                save_path=f"{OUTPUT_DIR}/alpha_elevation_phase_diagram.png"
            )
            plt.close(fig)
            print("  α-elevation phase diagram saved!")

            # Process domain plot
            fig, axes = plot_alpha_by_process_domain(
                domain_results,
                save_path=f"{OUTPUT_DIR}/alpha_by_process_domain.png"
            )
            plt.close(fig)
            print("  α by process domain plot saved!")

        # Save results
        sensitivity_results.to_csv(f"{OUTPUT_DIR}/powerlaw_sensitivity.csv", index=False)
        domain_results.to_csv(f"{OUTPUT_DIR}/powerlaw_by_process_domain.csv", index=False)

        print("\n[SUCCESS] Sensitivity analysis complete!")

        return {
            'sensitivity': sensitivity_results,
            'cael_seekell_comparison': cs_comparison,
            'by_process_domain': domain_results,
        }

    except Exception as e:
        print(f"\n[ERROR] Sensitivity analysis failed: {e}")
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

def run_full_analysis(data_source='conus', include_xmin_by_elevation=True):
    """
    Run complete analysis pipeline for all hypotheses.

    All plots are generated by default - no separate function call needed.
    Includes runtime tracking and progress indicators.

    Parameters
    ----------
    data_source : str
        'conus' (recommended), 'gdb', or 'parquet'
    include_xmin_by_elevation : bool
        If True, run the comprehensive x_min sensitivity by elevation analysis

    Returns
    -------
    dict
        All results
    """
    # Initialize timer
    timer = reset_timer()
    total_steps = 13 if include_xmin_by_elevation else 12

    print("\n" + "=" * 70)
    print("LAKE DISTRIBUTION ANALYSIS - FULL PIPELINE")
    print("=" * 70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total steps: {total_steps}")
    print("=" * 70)

    # Setup
    setup_plot_style()
    ensure_output_dir()
    print_config_summary()

    results = {}
    step = 0

    # Step 1: Load data
    step += 1
    print_step_header(step, total_steps, "Loading Lake Data")
    with timed_step(timer, "Load data"):
        lakes = load_data(source=data_source)
        results['lakes'] = lakes

    # Step 2: H1 - Elevation bimodality
    step += 1
    print_step_header(step, total_steps, "H1: Elevation Bimodality Analysis")
    with timed_step(timer, "H1: Elevation analysis"):
        results['H1_elevation'] = analyze_elevation(lakes)

    # Step 3: H2 - Slope threshold
    step += 1
    print_step_header(step, total_steps, "H2: Slope Threshold Analysis")
    with timed_step(timer, "H2: Slope analysis"):
        results['H2_slope'] = analyze_slope(lakes)

    # Step 4: H3 - Relief controls
    step += 1
    print_step_header(step, total_steps, "H3: Relief Controls Analysis")
    with timed_step(timer, "H3: Relief analysis"):
        results['H3_relief'] = analyze_relief(lakes)

    # Step 5: H4 - 2D process domains
    step += 1
    print_step_header(step, total_steps, "H4: 2D Elevation x Slope Domains")
    with timed_step(timer, "H4: 2D domain analysis"):
        results['H4_2d'] = analyze_2d_domains(lakes)

    # Step 6: H5 - Power law
    step += 1
    print_step_header(step, total_steps, "H5: Power Law Analysis")
    with timed_step(timer, "H5: Power law analysis"):
        results['H5_powerlaw'] = analyze_powerlaw(lakes)

    # Step 7: H6 - Slope-Relief domains
    step += 1
    print_step_header(step, total_steps, "H6: Slope-Relief Domains")
    with timed_step(timer, "H6: Slope-relief analysis"):
        results['H6_slope_relief'] = analyze_slope_relief(lakes)

    # Step 8: Relief × Elevation 2D analysis
    step += 1
    print_step_header(step, total_steps, "Relief x Elevation 2D Analysis")
    with timed_step(timer, "Relief-elevation analysis"):
        results['relief_elevation'] = analyze_relief_elevation(lakes)

    # Step 9: Power law sensitivity analysis
    step += 1
    print_step_header(step, total_steps, "Power Law Sensitivity Analysis")
    with timed_step(timer, "Power law sensitivity"):
        results['powerlaw_sensitivity'] = analyze_powerlaw_sensitivity(lakes)

    # Step 10: x_min sensitivity by elevation (if enabled)
    if include_xmin_by_elevation:
        step += 1
        print_step_header(step, total_steps, "x_min Sensitivity by Elevation")
        with timed_step(timer, "x_min by elevation analysis"):
            results['xmin_by_elevation'] = analyze_xmin_by_elevation(lakes)

    # Step 11: Domain classification
    step += 1
    print_step_header(step, total_steps, "Domain Classification")
    with timed_step(timer, "Domain classification"):
        results['domains'] = analyze_domains(lakes)

    # Step 12: Generate additional figures
    step += 1
    print_step_header(step, total_steps, "Generating Summary Figures")
    with timed_step(timer, "Summary figures"):
        # Geographic density map
        print("  Creating geographic density map...")
        try:
            fig, ax = plot_geographic_density_map(
                lakes,
                save_path=f"{OUTPUT_DIR}/geographic_density_map.png"
            )
            if fig:
                plt.close(fig)
                print("    Geographic density map saved!")
        except Exception as e:
            print(f"    Warning: Could not create geographic map: {e}")

        # 3-panel summary figure
        print("  Creating 3-panel summary figure...")
        if results.get('H1_elevation'):
            try:
                fig, axes = plot_three_panel_summary(
                    lakes,
                    results['H1_elevation']['density'],
                    results['H1_elevation']['landscape_area'],
                    save_path=f"{OUTPUT_DIR}/three_panel_summary.png"
                )
                plt.close(fig)
                print("    3-panel summary saved!")
            except Exception as e:
                print(f"    Warning: Could not create 3-panel summary: {e}")

        # Original summary figure
        print("  Creating overall summary figure...")
        if results.get('H1_elevation') and results.get('H2_slope'):
            fig = create_summary_figure(
                results['H1_elevation']['density'],
                results['H2_slope']['density'] if results.get('H2_slope') else None,
                results['H3_relief']['density'] if results.get('H3_relief') else None,
                save_path=f"{OUTPUT_DIR}/summary_figure.png"
            )
            plt.close(fig)
            print("    Summary figure saved!")

    # Final output
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Print runtime summary
    timing_results = timer.summary()
    results['timing'] = timing_results

    return results


# ============================================================================
# X_MIN SENSITIVITY BY ELEVATION ANALYSIS
# ============================================================================

def analyze_xmin_by_elevation(lakes, save_figures=True):
    """
    Run comprehensive x_min sensitivity analysis by elevation band.

    This analysis addresses key questions:
    1. Does optimal x_min differ by elevation?
    2. How sensitive is alpha to x_min choice in each band?
    3. Are alpha differences robust or threshold-dependent?

    Parameters
    ----------
    lakes : DataFrame
    save_figures : bool

    Returns
    -------
    dict
        Comprehensive results including:
        - by_elevation: per-band sensitivity results
        - method_comparison: comparison of optimal vs fixed x_min
        - robustness: assessment of alpha stability
        - summary_table: formatted summary
    """
    print("\n" + "=" * 60)
    print("X_MIN SENSITIVITY ANALYSIS BY ELEVATION")
    print("=" * 60)

    try:
        area_col = COLS['area']
        elev_col = COLS['elevation']

        # Define elevation bands (500m intervals)
        elevation_bands = [(0, 500), (500, 1000), (1000, 1500),
                          (1500, 2000), (2000, 2500), (2500, 3000)]

        # Define x_min candidates (logarithmically spaced)
        xmin_candidates = np.logspace(np.log10(0.01), np.log10(5.0), 30)

        # Fixed x_min values for comparison
        fixed_xmin_values = [0.024, 0.1, 0.46, 1.0]

        print("\n[STEP 1/7] Running sensitivity analysis for each elevation band...")
        print(f"  Elevation bands: {len(elevation_bands)}")
        print(f"  x_min candidates: {len(xmin_candidates)}")

        # Run the comprehensive analysis
        xmin_results = xmin_sensitivity_by_elevation(
            lakes,
            elevation_bands=elevation_bands,
            xmin_candidates=xmin_candidates,
            elev_col=elev_col,
            area_col=area_col,
            ks_tolerance=0.01,
            fixed_xmin_values=fixed_xmin_values,
            show_progress=True
        )

        print("\n[STEP 2/7] Comparing x_min methods...")
        method_comparison = compare_xmin_methods(xmin_results)
        xmin_results['method_comparison'] = method_comparison

        print("\n[STEP 3/7] Testing alpha robustness...")
        robustness = test_alpha_robustness(xmin_results)
        xmin_results['robustness'] = robustness

        print("\n[STEP 4/7] Generating summary table...")
        summary_table = generate_xmin_summary_table(xmin_results)
        xmin_results['summary_table'] = summary_table

        # Print summary
        if summary_table is not None and not summary_table.empty:
            print("\n  x_min Sensitivity Summary:")
            print(summary_table.to_string(index=False))

        # Save tabular results
        ensure_output_dir()
        if summary_table is not None:
            summary_table.to_csv(f"{OUTPUT_DIR}/xmin_by_elevation_summary.csv", index=False)
            print(f"\n  Summary saved to: {OUTPUT_DIR}/xmin_by_elevation_summary.csv")

        # Run hypothesis tests
        print("\n[STEP 5/7] Running hypothesis tests...")
        hypothesis_results = run_all_hypothesis_tests(xmin_results, verbose=True)
        xmin_results['hypothesis_tests'] = hypothesis_results

        # Generate hypothesis test report
        print("\n[STEP 6/7] Generating hypothesis test report...")
        report = generate_hypothesis_test_report(
            hypothesis_results,
            output_path=f"{OUTPUT_DIR}/xmin_hypothesis_test_report.txt"
        )
        xmin_results['hypothesis_report'] = report

        # Generate visualizations
        if save_figures:
            print("\n[STEP 7/7] Generating visualizations...")

            # Multi-panel KS curves
            try:
                fig, axes = plot_xmin_sensitivity_by_elevation(
                    xmin_results,
                    save_path=f"{OUTPUT_DIR}/xmin_sensitivity_by_elevation.png"
                )
                if fig:
                    plt.close(fig)
                    print("    x_min sensitivity by elevation saved!")
            except Exception as e:
                print(f"    Warning: Could not create sensitivity panels: {e}")

            # KS curves overlay
            try:
                fig, ax = plot_ks_curves_overlay(
                    xmin_results,
                    save_path=f"{OUTPUT_DIR}/ks_curves_overlay.png"
                )
                if fig:
                    plt.close(fig)
                    print("    KS curves overlay saved!")
            except Exception as e:
                print(f"    Warning: Could not create KS overlay: {e}")

            # Optimal x_min vs elevation
            try:
                fig, axes = plot_optimal_xmin_vs_elevation(
                    xmin_results,
                    save_path=f"{OUTPUT_DIR}/optimal_xmin_vs_elevation.png"
                )
                if fig:
                    plt.close(fig)
                    print("    Optimal x_min vs elevation saved!")
            except Exception as e:
                print(f"    Warning: Could not create optimal xmin plot: {e}")

            # Alpha stability
            try:
                fig, ax = plot_alpha_stability_by_elevation(
                    xmin_results,
                    save_path=f"{OUTPUT_DIR}/alpha_stability_by_elevation.png"
                )
                if fig:
                    plt.close(fig)
                    print("    Alpha stability plot saved!")
            except Exception as e:
                print(f"    Warning: Could not create stability plot: {e}")

            # Comprehensive summary
            try:
                fig, axes = plot_xmin_elevation_summary(
                    xmin_results,
                    save_path=f"{OUTPUT_DIR}/xmin_elevation_summary.png"
                )
                if fig:
                    plt.close(fig)
                    print("    Comprehensive summary figure saved!")
            except Exception as e:
                print(f"    Warning: Could not create summary figure: {e}")

            # Alpha vs x_min by elevation (multi-panel alpha sensitivity)
            try:
                fig, axes = plot_alpha_vs_xmin_by_elevation(
                    xmin_results,
                    save_path=f"{OUTPUT_DIR}/alpha_vs_xmin_by_elevation.png"
                )
                if fig:
                    plt.close(fig)
                    print("    Alpha vs x_min by elevation saved!")
            except Exception as e:
                print(f"    Warning: Could not create alpha sensitivity plot: {e}")

        print("\n[SUCCESS] x_min sensitivity by elevation analysis complete!")

        return xmin_results

    except Exception as e:
        print(f"\n[ERROR] x_min sensitivity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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
   >>> lakes = load_data()           # Default: CONUS-only (recommended)
   >>> lakes = load_data('gdb')      # Full geodatabase
   >>> lakes = load_data('parquet')  # Full parquet file

3. Run individual analyses:
   >>> results = analyze_elevation(lakes)
   >>> results = analyze_slope(lakes)
   >>> results = analyze_relief(lakes)
   >>> results = analyze_powerlaw(lakes)    # Enhanced with multiple plots
   >>> results = analyze_2d_domains(lakes)  # With marginal PDFs

4. Advanced analyses:
   >>> results = analyze_slope_relief(lakes)         # Slope x relief heatmap
   >>> results = analyze_powerlaw_sensitivity(lakes) # x_min sensitivity
   >>> results = analyze_xmin_by_elevation(lakes)    # NEW: x_min by elevation

5. Run full pipeline (with progress tracking):
   >>> all_results = run_full_analysis()

   Options:
   >>> run_full_analysis(data_source='conus')              # Default
   >>> run_full_analysis(include_xmin_by_elevation=False)  # Skip detailed x_min

6. Classify domains:
   >>> lakes_classified = analyze_domains(lakes)

Key Outputs:
- Runtime summary with step-by-step timing
- alpha-Elevation phase diagram (compares to percolation theory tau=2.05)
- Slope-Relief heatmap with "sweet spot" identification
- x_min sensitivity analysis (how threshold affects alpha)
- x_min sensitivity BY ELEVATION (does optimal x_min vary?)
- Comparison to Cael & Seekell (2016) global result

Runtime Tracking:
- Progress bars show step completion
- Step headers show overall progress
- Final summary shows per-step timing and total runtime

Tips:
- MIN_LAKE_AREA = 0.024 km^2 (NHD consistent threshold)
- Update paths in config.py before running
- Results are saved to OUTPUT_DIR specified in config.py
- Use 'conus' data source for fastest loading
""")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Lake Distribution Analysis')
    parser.add_argument('--check', action='store_true',
                       help='Check data availability only')
    parser.add_argument('--source', choices=['conus', 'gdb', 'parquet'], default='conus',
                       help='Data source (default: conus)')
    parser.add_argument('--hypothesis', type=int, choices=[1, 2, 3, 4, 5, 6],
                       help='Run specific hypothesis test only')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis pipeline')
    parser.add_argument('--no-xmin-elevation', action='store_true',
                       help='Skip x_min by elevation analysis in full pipeline')
    parser.add_argument('--xmin-elevation', action='store_true',
                       help='Run only x_min by elevation analysis')

    args = parser.parse_args()

    if args.check:
        quick_data_check()
    elif args.full:
        run_full_analysis(
            data_source=args.source,
            include_xmin_by_elevation=not args.no_xmin_elevation
        )
    elif args.xmin_elevation:
        lakes = load_data(source=args.source)
        analyze_xmin_by_elevation(lakes)
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
        elif args.hypothesis == 6:
            analyze_slope_relief(lakes)
    else:
        quick_start()
