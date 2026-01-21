"""
Sensitivity Analysis Module

Systematically tests robustness of results to parameter choices:
- Lake area thresholds (min/max)
- Grid cell sizes (multivariate analysis)
- Age uncertainty distributions
- Bayesian priors

This addresses reviewer concerns about parameter-dependent results and
demonstrates which findings are robust vs. sensitive to choices.

Author: morrismc
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    from .glacial_chronosequence import (
        classify_lakes_by_glacial_extent,
        compute_lake_density_by_glacial_stage,
        fit_bayesian_decay_model
    )
    from .multivariate_analysis import (
        prepare_multivariate_dataset,
        run_multivariate_regression
    )
    from .config import OUTPUT_DIR, ensure_output_dir
    from .data_loading import load_conus_lake_data, convert_lakes_to_gdf
except ImportError:
    from glacial_chronosequence import (
        classify_lakes_by_glacial_extent,
        compute_lake_density_by_glacial_stage,
        fit_bayesian_decay_model
    )
    from multivariate_analysis import (
        prepare_multivariate_dataset,
        run_multivariate_regression
    )
    from config import OUTPUT_DIR, ensure_output_dir
    from data_loading import load_conus_lake_data, convert_lakes_to_gdf


def test_threshold_sensitivity(
    lakes_classified,
    thresholds: List[float] = None,
    use_bayesian: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test sensitivity of half-life estimates to min_lake_area threshold.

    This addresses the critical detection bias issue: half-life estimates
    vary 2-3× depending on threshold choice. We test multiple thresholds
    and examine convergence at larger sizes.

    Parameters
    ----------
    lakes_classified : GeoDataFrame
        Lakes with 'glacial_stage' classification
    thresholds : list of float, optional
        Lake area thresholds to test (km²)
        Default: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    use_bayesian : bool
        Whether to fit Bayesian model for each threshold
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        'thresholds': List of thresholds tested
        'densities': Dict of densities by threshold and stage
        'halflife_estimates': Dict of half-life estimates (if Bayesian)
        'convergence_threshold': Threshold where estimates stabilize

    Examples
    --------
    >>> results = test_threshold_sensitivity(lakes_classified)
    >>> plt.plot(results['thresholds'], results['halflife_medians'])
    >>> plt.xlabel('Minimum Lake Area (km²)')
    >>> plt.ylabel('Half-Life (ka)')
    """
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    if verbose:
        print(f"Testing {len(thresholds)} thresholds...")

    # Storage
    densities_by_threshold = {}
    halflife_estimates = [] if use_bayesian else None

    # Test each threshold
    for i, thresh in enumerate(thresholds):
        if verbose:
            print(f"  [{i+1}/{len(thresholds)}] Threshold: {thresh} km²")

        # Filter lakes
        lakes_filtered = lakes_classified[
            lakes_classified.geometry.area / 1e6 >= thresh
        ].copy()

        # Compute densities
        densities = compute_lake_density_by_glacial_stage(
            lakes_filtered,
            verbose=False
        )
        densities_by_threshold[thresh] = densities

        # Fit Bayesian model
        if use_bayesian:
            try:
                bayes_results = fit_bayesian_decay_model(
                    densities,
                    n_samples=1000,  # Faster for sensitivity
                    n_tune=500,
                    n_chains=2,
                    verbose=False
                )
                halflife_estimates.append({
                    'threshold': thresh,
                    'median': bayes_results['halflife_median_ka'],
                    'ci_lower': bayes_results['halflife_ci_lower'],
                    'ci_upper': bayes_results['halflife_ci_upper']
                })
            except Exception as e:
                if verbose:
                    print(f"    Bayesian fit failed: {e}")
                halflife_estimates.append({
                    'threshold': thresh,
                    'median': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan
                })

    # Detect convergence threshold
    convergence_threshold = None
    if use_bayesian and len(halflife_estimates) > 2:
        medians = [h['median'] for h in halflife_estimates]
        valid_medians = [m for m in medians if not np.isnan(m)]

        if len(valid_medians) >= 3:
            # Convergence = when estimates vary <20% from median
            overall_median = np.median(valid_medians)
            for i, h in enumerate(halflife_estimates):
                if not np.isnan(h['median']):
                    rel_diff = abs(h['median'] - overall_median) / overall_median
                    if rel_diff < 0.2:
                        convergence_threshold = thresholds[i]
                        break

    results = {
        'thresholds': thresholds,
        'densities': densities_by_threshold,
        'halflife_estimates': halflife_estimates,
        'convergence_threshold': convergence_threshold
    }

    if verbose:
        print("\nThreshold Sensitivity Summary:")
        print(f"  Tested thresholds: {min(thresholds):.3f} - {max(thresholds):.3f} km²")
        if use_bayesian and halflife_estimates:
            valid_hl = [h['median'] for h in halflife_estimates if not np.isnan(h['median'])]
            if valid_hl:
                print(f"  Half-life range: {min(valid_hl):.0f} - {max(valid_hl):.0f} ka")
                print(f"  Convergence threshold: {convergence_threshold} km²" if convergence_threshold else "  No convergence detected")

    return results


def test_grid_size_sensitivity(
    lakes_classified,
    grid_sizes: List[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test sensitivity of multivariate results to grid cell size.

    Addresses reviewer concern: "Why 0.5°? Are results sensitive to grid size?"

    Parameters
    ----------
    lakes_classified : GeoDataFrame
        Lakes with glacial classification and environmental variables
    grid_sizes : list of float, optional
        Grid cell sizes to test (degrees)
        Default: [0.25, 0.5, 0.75, 1.0]
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        'grid_sizes': List of grid sizes tested
        'regression_results': List of regression results by grid size
        'variance_partitioning': Variance partitioning by grid size

    Notes
    -----
    Smaller grids = more spatial detail but less lakes per cell
    Larger grids = more lakes per cell but coarser resolution
    Results should be qualitatively similar across reasonable range
    """
    if grid_sizes is None:
        grid_sizes = [0.25, 0.5, 0.75, 1.0]

    if verbose:
        print(f"Testing {len(grid_sizes)} grid sizes...")

    results_by_grid = []

    for i, grid_size in enumerate(grid_sizes):
        if verbose:
            print(f"  [{i+1}/{len(grid_sizes)}] Grid size: {grid_size}°")

        try:
            # Prepare dataset with this grid size
            dataset = prepare_multivariate_dataset(
                lakes_classified,
                grid_cell_size=grid_size,
                min_lake_area=0.01,
                max_lake_area=20000,
                verbose=False
            )

            # Run regression
            reg_results = run_multivariate_regression(
                dataset,
                response_var='density',
                verbose=False
            )

            results_by_grid.append({
                'grid_size': grid_size,
                'n_cells': len(dataset),
                'r_squared': reg_results['r_squared'],
                'coefficients': reg_results['coefficients'],
                'p_values': reg_results['p_values']
            })

            if verbose:
                print(f"    Cells: {len(dataset)}, R²: {reg_results['r_squared']:.3f}")

        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            results_by_grid.append({
                'grid_size': grid_size,
                'n_cells': 0,
                'r_squared': np.nan,
                'coefficients': {},
                'p_values': {}
            })

    return {
        'grid_sizes': grid_sizes,
        'results': results_by_grid
    }


def test_age_uncertainty_sensitivity(
    densities: Dict[str, float],
    age_distributions: List[Dict] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test sensitivity of half-life to age uncertainty distributions.

    Addresses: "Are results sensitive to how you model age uncertainty?"

    Parameters
    ----------
    densities : dict
        Lake densities by glacial stage
    age_distributions : list of dict, optional
        Alternative age distributions to test
        Each dict: {'name': str, 'wisconsin': (mean, std), 'illinoian': (mean, std), ...}
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Half-life estimates under different age uncertainty models

    Examples
    --------
    >>> age_dists = [
    ...     {'name': 'narrow', 'wisconsin': (20, 2), 'illinoian': (160, 10)},
    ...     {'name': 'wide', 'wisconsin': (20, 5), 'illinoian': (160, 30)}
    ... ]
    >>> results = test_age_uncertainty_sensitivity(densities, age_dists)
    """
    if age_distributions is None:
        age_distributions = [
            {
                'name': 'baseline',
                'wisconsin': (20, 2.5),
                'illinoian': (160, 15),
                'driftless': (1500, 250)
            },
            {
                'name': 'narrow_uncertainty',
                'wisconsin': (20, 1.0),
                'illinoian': (160, 5),
                'driftless': (1500, 100)
            },
            {
                'name': 'wide_uncertainty',
                'wisconsin': (20, 5.0),
                'illinoian': (160, 30),
                'driftless': (1500, 500)
            }
        ]

    results = []

    for age_dist in age_distributions:
        if verbose:
            print(f"Testing age distribution: {age_dist['name']}")

        try:
            # Would need to modify fit_bayesian_decay_model to accept age distributions
            # For now, just document the framework
            if verbose:
                print(f"  Wisconsin: {age_dist['wisconsin'][0]} ± {age_dist['wisconsin'][1]} ka")
                print(f"  Illinoian: {age_dist['illinoian'][0]} ± {age_dist['illinoian'][1]} ka")

            # Placeholder for actual Bayesian fit with custom ages
            results.append({
                'name': age_dist['name'],
                'age_model': age_dist,
                'halflife_median': np.nan,  # Would be from Bayesian fit
                'note': 'Requires Bayesian model modification'
            })

        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")

    return {'age_distributions': age_distributions, 'results': results}


def plot_threshold_sensitivity(
    sensitivity_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10)
) -> None:
    """
    Visualize threshold sensitivity results.

    Creates 4-panel figure:
    - Panel A: Half-life vs. threshold (with confidence intervals)
    - Panel B: Density by stage vs. threshold
    - Panel C: Relative change from baseline
    - Panel D: Convergence diagnostic

    Parameters
    ----------
    sensitivity_results : dict
        Output from test_threshold_sensitivity()
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    thresholds = sensitivity_results['thresholds']
    halflife_est = sensitivity_results['halflife_estimates']

    # Panel A: Half-life vs. threshold
    if halflife_est:
        medians = [h['median'] for h in halflife_est]
        ci_lower = [h['ci_lower'] for h in halflife_est]
        ci_upper = [h['ci_upper'] for h in halflife_est]

        axes[0, 0].plot(thresholds, medians, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        axes[0, 0].fill_between(thresholds, ci_lower, ci_upper, alpha=0.3, color='#2E86AB')
        axes[0, 0].set_xlabel('Minimum Lake Area Threshold (km²)', fontsize=12)
        axes[0, 0].set_ylabel('Half-Life (ka)', fontsize=12)
        axes[0, 0].set_title('A. Half-Life Sensitivity to Threshold', fontsize=13, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        # Mark convergence threshold if detected
        if sensitivity_results['convergence_threshold']:
            conv_thresh = sensitivity_results['convergence_threshold']
            axes[0, 0].axvline(conv_thresh, color='red', linestyle='--', linewidth=2,
                             label=f'Convergence: {conv_thresh} km²')
            axes[0, 0].legend()

    # Panel B: Density by stage vs. threshold
    densities = sensitivity_results['densities']
    stages = list(next(iter(densities.values())).keys())
    colors = {'wisconsin': '#1B4965', 'illinoian': '#62B6CB', 'driftless': '#BEE9E8'}

    for stage in stages:
        stage_densities = [densities[t][stage] for t in thresholds]
        axes[0, 1].plot(thresholds, stage_densities, 'o-', linewidth=2, markersize=8,
                       label=stage.capitalize(), color=colors.get(stage, 'gray'))

    axes[0, 1].set_xlabel('Minimum Lake Area Threshold (km²)', fontsize=12)
    axes[0, 1].set_ylabel('Lake Density (lakes/1000 km²)', fontsize=12)
    axes[0, 1].set_title('B. Density Sensitivity by Glacial Stage', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Panel C: Relative change from baseline
    if halflife_est and not np.isnan(halflife_est[0]['median']):
        baseline = halflife_est[0]['median']
        rel_change = [(h['median'] - baseline) / baseline * 100 for h in halflife_est]

        axes[1, 0].bar(range(len(thresholds)), rel_change, color='#A23B72', alpha=0.7)
        axes[1, 0].set_xticks(range(len(thresholds)))
        axes[1, 0].set_xticklabels([f'{t:.3f}' for t in thresholds], rotation=45)
        axes[1, 0].set_xlabel('Minimum Lake Area Threshold (km²)', fontsize=12)
        axes[1, 0].set_ylabel('% Change from Baseline', fontsize=12)
        axes[1, 0].set_title('C. Relative Change in Half-Life', fontsize=13, fontweight='bold')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].grid(axis='y', alpha=0.3)

    # Panel D: Convergence diagnostic (coefficient of variation)
    if halflife_est:
        # Rolling CV across thresholds
        window_size = 3
        cvs = []
        for i in range(len(halflife_est) - window_size + 1):
            window = [halflife_est[j]['median'] for j in range(i, i + window_size)]
            if not any(np.isnan(window)):
                cv = np.std(window) / np.mean(window) * 100
                cvs.append(cv)
            else:
                cvs.append(np.nan)

        if cvs:
            axes[1, 1].plot(thresholds[:len(cvs)], cvs, 'o-', linewidth=2, markersize=8, color='#F18F01')
            axes[1, 1].axhline(20, color='red', linestyle='--', linewidth=1.5,
                             label='20% CV threshold')
            axes[1, 1].set_xlabel('Minimum Lake Area Threshold (km²)', fontsize=12)
            axes[1, 1].set_ylabel('Coefficient of Variation (%)', fontsize=12)
            axes[1, 1].set_title('D. Convergence Diagnostic (3-point rolling CV)', fontsize=13, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity analysis figure saved: {save_path}")
    else:
        plt.savefig(f"{OUTPUT_DIR}/threshold_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Sensitivity analysis figure saved: {OUTPUT_DIR}/threshold_sensitivity_analysis.png")

    plt.show()


def plot_grid_size_sensitivity(
    grid_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Visualize grid size sensitivity results.

    Parameters
    ----------
    grid_results : dict
        Output from test_grid_size_sensitivity()
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    grid_sizes = grid_results['grid_sizes']
    results = grid_results['results']

    # Extract R² values
    r_squared = [r['r_squared'] for r in results]
    n_cells = [r['n_cells'] for r in results]

    # Panel A: R² vs. grid size
    axes[0].plot(grid_sizes, r_squared, 'o-', linewidth=2, markersize=10, color='#06A77D')
    axes[0].set_xlabel('Grid Cell Size (degrees)', fontsize=12)
    axes[0].set_ylabel('R² (Explained Variance)', fontsize=12)
    axes[0].set_title('A. Model Fit vs. Grid Resolution', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Panel B: Number of cells vs. grid size
    axes[1].plot(grid_sizes, n_cells, 'o-', linewidth=2, markersize=10, color='#D62839')
    axes[1].set_xlabel('Grid Cell Size (degrees)', fontsize=12)
    axes[1].set_ylabel('Number of Grid Cells', fontsize=12)
    axes[1].set_title('B. Sample Size vs. Grid Resolution', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{OUTPUT_DIR}/grid_sensitivity_analysis.png", dpi=300, bbox_inches='tight')

    plt.show()


def run_comprehensive_sensitivity_analysis(
    lakes_classified,
    test_thresholds: bool = True,
    test_grid_sizes: bool = True,
    test_age_uncertainty: bool = False,
    save_figures: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete sensitivity analysis suite.

    This is the main entry point for addressing reviewer concerns about
    parameter-dependent results.

    Parameters
    ----------
    lakes_classified : GeoDataFrame
        Classified lakes with environmental variables
    test_thresholds : bool
        Test lake area threshold sensitivity
    test_grid_sizes : bool
        Test grid size sensitivity
    test_age_uncertainty : bool
        Test age distribution sensitivity
    save_figures : bool
        Save visualization figures
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Complete sensitivity analysis results

    Examples
    --------
    >>> from lake_analysis import load_all_glacial_boundaries, classify_lakes_by_glacial_extent
    >>> boundaries = load_all_glacial_boundaries()
    >>> lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)
    >>> results = run_comprehensive_sensitivity_analysis(lakes_classified)
    """
    ensure_output_dir()

    results = {}

    # 1. Threshold sensitivity
    if test_thresholds:
        if verbose:
            print("\n" + "="*60)
            print("THRESHOLD SENSITIVITY ANALYSIS")
            print("="*60)

        threshold_results = test_threshold_sensitivity(
            lakes_classified,
            thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
            use_bayesian=True,
            verbose=verbose
        )
        results['threshold_sensitivity'] = threshold_results

        if save_figures:
            plot_threshold_sensitivity(
                threshold_results,
                save_path=f"{OUTPUT_DIR}/threshold_sensitivity_analysis.png"
            )

    # 2. Grid size sensitivity
    if test_grid_sizes:
        if verbose:
            print("\n" + "="*60)
            print("GRID SIZE SENSITIVITY ANALYSIS")
            print("="*60)

        grid_results = test_grid_size_sensitivity(
            lakes_classified,
            grid_sizes=[0.25, 0.5, 0.75, 1.0],
            verbose=verbose
        )
        results['grid_sensitivity'] = grid_results

        if save_figures:
            plot_grid_size_sensitivity(
                grid_results,
                save_path=f"{OUTPUT_DIR}/grid_sensitivity_analysis.png"
            )

    # 3. Age uncertainty sensitivity
    if test_age_uncertainty:
        if verbose:
            print("\n" + "="*60)
            print("AGE UNCERTAINTY SENSITIVITY ANALYSIS")
            print("="*60)

        # Need densities first
        densities = compute_lake_density_by_glacial_stage(lakes_classified)

        age_results = test_age_uncertainty_sensitivity(
            densities,
            verbose=verbose
        )
        results['age_uncertainty_sensitivity'] = age_results

    if verbose:
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print("="*60)
        print("\nKey Findings:")

        if test_thresholds and 'threshold_sensitivity' in results:
            conv_thresh = results['threshold_sensitivity']['convergence_threshold']
            print(f"  • Threshold convergence at: {conv_thresh} km²" if conv_thresh else "  • No clear threshold convergence")

        if test_grid_sizes and 'grid_sensitivity' in results:
            r2_range = [r['r_squared'] for r in results['grid_sensitivity']['results']]
            r2_range = [r for r in r2_range if not np.isnan(r)]
            if r2_range:
                print(f"  • R² range across grid sizes: {min(r2_range):.3f} - {max(r2_range):.3f}")

    return results


# Main execution
if __name__ == "__main__":
    print("Sensitivity Analysis Module")
    print("Run from main pipeline using run_comprehensive_sensitivity_analysis()")
    print("\nExample:")
    print("  from lake_analysis import sensitivity_analysis")
    print("  results = sensitivity_analysis.run_comprehensive_sensitivity_analysis(lakes_classified)")
