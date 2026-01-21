"""
Detection Bias Modeling and Correction

This module addresses the critical methodological concern: small lakes are
systematically under-detected in older landscapes, creating artificial
half-life variation with threshold choice.

**Scientific Problem:**
- Mapping technology biases toward larger, more persistent features
- Older landscapes: depressions filled in → smaller remaining lakes harder to map
- Younger landscapes: more lakes of all sizes → better detection
- Creates confound: Is decay real or just detection artifact?

**Solution Approach:**
1. Model detection probability: P(detected | size, age)
2. Estimate detection limits empirically from size distributions
3. Correct density estimates for under-detection
4. Show pattern persists after correction

Author: morrismc
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    from .config import OUTPUT_DIR, ensure_output_dir
    from .glacial_chronosequence import compute_lake_density_by_glacial_stage
except ImportError:
    from config import OUTPUT_DIR, ensure_output_dir
    from glacial_chronosequence import compute_lake_density_by_glacial_stage


def estimate_detection_probability_logistic(
    lake_areas: np.ndarray,
    detection_labels: np.ndarray
) -> Dict[str, float]:
    """
    Estimate detection probability using logistic regression.

    Model: P(detected | area) = 1 / (1 + exp(-(β₀ + β₁ × log(area))))

    Parameters
    ----------
    lake_areas : array
        Lake areas (km²)
    detection_labels : array
        Binary labels: 1 = detected, 0 = not detected
        (in practice, we use observed vs. expected from power law)

    Returns
    -------
    dict
        'beta_0': Intercept parameter
        'beta_1': Slope parameter (positive = larger lakes more detectable)
        'area_50': Area at which P(detection) = 0.5
        'area_90': Area at which P(detection) = 0.9

    Examples
    --------
    >>> # Estimate from comparison with power law expectation
    >>> params = estimate_detection_probability_logistic(areas, detected)
    >>> area_50 = params['area_50']  # 50% detection threshold
    """
    # Logistic regression
    from scipy.special import expit

    def negative_log_likelihood(params, areas, labels):
        """Negative log-likelihood for logistic model"""
        beta_0, beta_1 = params
        log_areas = np.log(areas)
        p = expit(beta_0 + beta_1 * log_areas)  # Sigmoid function

        # Clip probabilities to avoid log(0)
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # Bernoulli log-likelihood
        ll = np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        return -ll

    # Initial guess
    initial_params = [0, 1]

    # Optimize
    result = optimize.minimize(
        negative_log_likelihood,
        initial_params,
        args=(lake_areas, detection_labels),
        method='BFGS'
    )

    beta_0, beta_1 = result.x

    # Calculate threshold areas
    # P = 0.5 when β₀ + β₁ × log(area) = 0
    area_50 = np.exp(-beta_0 / beta_1) if beta_1 != 0 else np.nan

    # P = 0.9 when β₀ + β₁ × log(area) = log(9) ≈ 2.197
    area_90 = np.exp((np.log(9) - beta_0) / beta_1) if beta_1 != 0 else np.nan

    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'area_50': area_50,
        'area_90': area_90,
        'success': result.success
    }


def detect_size_distribution_truncation(
    lakes_by_stage: Dict[str, pd.DataFrame],
    size_bins: np.ndarray = None,
    method: str = 'ks_test'
) -> Dict[str, Any]:
    """
    Detect systematic truncation in size distributions across glacial stages.

    Tests whether size distributions differ more than expected, suggesting
    differential detection limits.

    Parameters
    ----------
    lakes_by_stage : dict
        Dictionary mapping stage names to lake GeoDataFrames
    size_bins : array, optional
        Bin edges for size distribution comparison
        Default: log-spaced from 0.001 to 100 km²
    method : str
        Detection method: 'ks_test', 'quantile_comparison', or 'power_law_deviation'

    Returns
    -------
    dict
        'detection_limits': Estimated detection limit by stage (km²)
        'ks_statistics': KS test statistics comparing stages
        'truncation_detected': Boolean by stage

    Notes
    -----
    If Wisconsin has more small lakes than Illinoian (after normalization),
    suggests differential detection, not true ecological pattern.
    """
    if size_bins is None:
        size_bins = np.logspace(-3, 2, 50)  # 0.001 to 100 km²

    results = {
        'detection_limits': {},
        'ks_statistics': {},
        'truncation_detected': {}
    }

    # Get area column
    area_col = 'AREASQKM' if 'AREASQKM' in list(lakes_by_stage.values())[0].columns else 'area'

    if method == 'ks_test':
        # Compare size distributions pairwise
        stages = list(lakes_by_stage.keys())

        for i, stage1 in enumerate(stages):
            for stage2 in stages[i+1:]:
                areas1 = lakes_by_stage[stage1][area_col].values
                areas2 = lakes_by_stage[stage2][area_col].values

                # Two-sample KS test
                ks_stat, p_value = stats.ks_2samp(areas1, areas2)

                results['ks_statistics'][f'{stage1}_vs_{stage2}'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.01
                }

    elif method == 'quantile_comparison':
        # Compare lower quantiles (where truncation appears)
        quantiles = [0.01, 0.05, 0.1, 0.25]

        for stage, lakes in lakes_by_stage.items():
            areas = lakes[area_col].values
            stage_quantiles = np.quantile(areas, quantiles)

            # Detection limit = area below which distribution is truncated
            # Heuristic: if 1st percentile > expected from power law
            results['detection_limits'][stage] = stage_quantiles[0]

    elif method == 'power_law_deviation':
        # Fit power law to each stage, compare xmin values
        # Higher xmin in older stages suggests truncation

        for stage, lakes in lakes_by_stage.items():
            areas = lakes[area_col].values

            # Simple MLE power law fit
            if len(areas) > 100:
                # Estimate xmin as 10th percentile (conservative)
                xmin = np.quantile(areas, 0.1)
                areas_above_xmin = areas[areas >= xmin]

                if len(areas_above_xmin) > 50:
                    # MLE alpha
                    n = len(areas_above_xmin)
                    alpha = 1 + n / np.sum(np.log(areas_above_xmin / xmin))

                    results['detection_limits'][stage] = xmin
                    results['truncation_detected'][stage] = xmin > 0.01  # Arbitrary threshold

    return results


def estimate_detection_curve_by_stage(
    lakes_by_stage: Dict[str, pd.DataFrame],
    reference_stage: str = 'wisconsin',
    size_bins: np.ndarray = None
) -> Dict[str, Dict]:
    """
    Estimate detection curves for each glacial stage relative to reference.

    Assumption: Wisconsin (youngest) has best detection across all sizes.
    Older stages have progressively worse detection at small sizes.

    Parameters
    ----------
    lakes_by_stage : dict
        Lakes grouped by glacial stage
    reference_stage : str
        Stage assumed to have best detection (usually 'wisconsin')
    size_bins : array, optional
        Size bins for analysis

    Returns
    -------
    dict
        Detection curves by stage: {stage: {sizes, probabilities}}

    Method
    ------
    1. Compute size-frequency distributions for each stage
    2. Normalize by total lake count
    3. Ratio of observed/reference = detection probability
    4. Fit logistic curve to ratios
    """
    if size_bins is None:
        size_bins = np.logspace(-3, 2, 30)

    # Get area column
    area_col = 'AREASQKM' if 'AREASQKM' in list(lakes_by_stage.values())[0].columns else 'area'

    # Compute size distributions
    size_distributions = {}
    for stage, lakes in lakes_by_stage.items():
        areas = lakes[area_col].values
        counts, _ = np.histogram(areas, bins=size_bins)

        # Normalize to frequency (sum to 1)
        freq = counts / counts.sum() if counts.sum() > 0 else counts

        size_distributions[stage] = {
            'bin_centers': (size_bins[:-1] + size_bins[1:]) / 2,
            'frequencies': freq,
            'counts': counts
        }

    # Estimate detection probabilities relative to reference
    detection_curves = {}

    if reference_stage in size_distributions:
        ref_freq = size_distributions[reference_stage]['frequencies']

        for stage in lakes_by_stage.keys():
            if stage == reference_stage:
                # Reference stage has perfect detection (by definition)
                detection_curves[stage] = {
                    'sizes': size_distributions[stage]['bin_centers'],
                    'probabilities': np.ones_like(size_distributions[stage]['bin_centers']),
                    'relative_to_reference': True
                }
            else:
                stage_freq = size_distributions[stage]['frequencies']

                # Detection probability = observed / expected
                # P(detect | size) = freq_stage / freq_reference
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob = np.divide(stage_freq, ref_freq,
                                   out=np.zeros_like(stage_freq),
                                   where=ref_freq > 0)

                # Clip to [0, 1] (some bins may have higher freq due to noise)
                prob = np.clip(prob, 0, 1)

                detection_curves[stage] = {
                    'sizes': size_distributions[stage]['bin_centers'],
                    'probabilities': prob,
                    'relative_to_reference': True
                }

    return detection_curves


def correct_density_for_detection_bias(
    densities: Dict[str, float],
    detection_curves: Dict[str, Dict],
    method: str = 'mean_probability'
) -> Dict[str, float]:
    """
    Correct lake density estimates for detection bias.

    Parameters
    ----------
    densities : dict
        Raw lake densities by stage (lakes/1000 km²)
    detection_curves : dict
        Detection probability curves by stage
    method : str
        Correction method:
        - 'mean_probability': density_corrected = density_raw / mean(P(detect))
        - 'extrapolation': Extrapolate to zero detection limit

    Returns
    -------
    dict
        Bias-corrected densities

    Examples
    --------
    >>> corrected = correct_density_for_detection_bias(densities, detection_curves)
    >>> print(f"Wisconsin: {densities['wisconsin']:.1f} → {corrected['wisconsin']:.1f}")
    """
    corrected_densities = {}

    for stage, raw_density in densities.items():
        if stage in detection_curves:
            prob = detection_curves[stage]['probabilities']

            if method == 'mean_probability':
                # Average detection probability across sizes
                mean_prob = np.mean(prob[prob > 0])  # Exclude zeros

                if mean_prob > 0:
                    corrected_densities[stage] = raw_density / mean_prob
                else:
                    corrected_densities[stage] = raw_density  # Can't correct

            elif method == 'extrapolation':
                # Fit curve and extrapolate to P = 1.0
                # More sophisticated but requires assumptions
                corrected_densities[stage] = raw_density  # Placeholder

        else:
            # No correction available
            corrected_densities[stage] = raw_density

    return corrected_densities


def plot_detection_bias_diagnostic(
    lakes_by_stage: Dict[str, pd.DataFrame],
    detection_curves: Dict[str, Dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 10)
) -> None:
    """
    Create comprehensive detection bias diagnostic figure.

    6-panel figure:
    A. Size distributions by stage (log-log)
    B. Cumulative distributions (detect truncation)
    C. Detection probability curves
    D. Quantile-quantile plot (Wisconsin vs. Illinoian)
    E. Density vs. detection threshold
    F. Corrected vs. uncorrected densities

    Parameters
    ----------
    lakes_by_stage : dict
        Lakes grouped by glacial stage
    detection_curves : dict, optional
        Pre-computed detection curves
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    area_col = 'AREASQKM' if 'AREASQKM' in list(lakes_by_stage.values())[0].columns else 'area'
    colors = {'wisconsin': '#1B4965', 'illinoian': '#62B6CB', 'driftless': '#BEE9E8',
              'unclassified': '#CCCCCC'}

    # Panel A: Size distributions (log-log)
    size_bins = np.logspace(-3, 2, 40)

    for stage, lakes in lakes_by_stage.items():
        if stage in ['wisconsin', 'illinoian', 'driftless']:  # Skip unclassified
            areas = lakes[area_col].values
            counts, bin_edges = np.histogram(areas, bins=size_bins)

            # Normalize by bin width (PDF estimate)
            bin_widths = np.diff(bin_edges)
            density = counts / bin_widths / counts.sum()

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            axes[0, 0].loglog(bin_centers, density, 'o-', label=stage.capitalize(),
                            color=colors[stage], alpha=0.7, markersize=4)

    axes[0, 0].set_xlabel('Lake Area (km²)', fontsize=11)
    axes[0, 0].set_ylabel('Probability Density', fontsize=11)
    axes[0, 0].set_title('A. Size Distributions by Glacial Stage', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Panel B: Cumulative distributions
    for stage, lakes in lakes_by_stage.items():
        if stage in ['wisconsin', 'illinoian', 'driftless']:
            areas = np.sort(lakes[area_col].values)
            cdf = np.arange(1, len(areas) + 1) / len(areas)

            axes[0, 1].semilogx(areas, cdf, '-', label=stage.capitalize(),
                              color=colors[stage], linewidth=2, alpha=0.7)

    axes[0, 1].set_xlabel('Lake Area (km²)', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative Probability', fontsize=11)
    axes[0, 1].set_title('B. Cumulative Size Distributions', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Panel C: Detection probability curves (if provided)
    if detection_curves:
        for stage, curve in detection_curves.items():
            if stage in ['wisconsin', 'illinoian', 'driftless']:
                axes[0, 2].semilogx(curve['sizes'], curve['probabilities'], 'o-',
                                  label=stage.capitalize(), color=colors[stage],
                                  linewidth=2, markersize=5)

        axes[0, 2].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% detection')
        axes[0, 2].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='50% detection')
        axes[0, 2].set_xlabel('Lake Area (km²)', fontsize=11)
        axes[0, 2].set_ylabel('Detection Probability', fontsize=11)
        axes[0, 2].set_title('C. Estimated Detection Probability', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        axes[0, 2].set_ylim([0, 1.1])

    # Panel D: Q-Q plot (Wisconsin vs. Illinoian)
    if 'wisconsin' in lakes_by_stage and 'illinoian' in lakes_by_stage:
        areas_wisc = lakes_by_stage['wisconsin'][area_col].values
        areas_ill = lakes_by_stage['illinoian'][area_col].values

        # Sample to same size for Q-Q
        n = min(len(areas_wisc), len(areas_ill), 10000)
        q_wisc = np.quantile(areas_wisc, np.linspace(0, 1, n))
        q_ill = np.quantile(areas_ill, np.linspace(0, 1, n))

        axes[1, 0].loglog(q_wisc, q_ill, 'o', alpha=0.3, markersize=3, color='#2E86AB')
        axes[1, 0].loglog([1e-3, 1e2], [1e-3, 1e2], 'k--', linewidth=1.5, label='1:1 line')

        axes[1, 0].set_xlabel('Wisconsin Quantiles (km²)', fontsize=11)
        axes[1, 0].set_ylabel('Illinoian Quantiles (km²)', fontsize=11)
        axes[1, 0].set_title('D. Q-Q Plot: Wisconsin vs. Illinoian', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    # Panel E: Density vs. detection threshold
    thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]
    densities_by_thresh = {stage: [] for stage in ['wisconsin', 'illinoian', 'driftless']}

    for thresh in thresholds:
        for stage in ['wisconsin', 'illinoian', 'driftless']:
            if stage in lakes_by_stage:
                # Count lakes above threshold
                areas = lakes_by_stage[stage][area_col].values
                n_above = np.sum(areas >= thresh)

                # Normalize by area (placeholder - would need actual landscape area)
                density = n_above  # Relative density
                densities_by_thresh[stage].append(density)

    for stage in ['wisconsin', 'illinoian', 'driftless']:
        axes[1, 1].plot(thresholds, densities_by_thresh[stage], 'o-',
                       label=stage.capitalize(), color=colors[stage],
                       linewidth=2, markersize=7)

    axes[1, 1].set_xlabel('Minimum Lake Area Threshold (km²)', fontsize=11)
    axes[1, 1].set_ylabel('Lake Count (relative)', fontsize=11)
    axes[1, 1].set_title('E. Lake Count Sensitivity to Threshold', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xscale('log')

    # Panel F: Summary statistics
    summary_text = "Detection Bias Summary:\n\n"

    for stage in ['wisconsin', 'illinoian', 'driftless']:
        if stage in lakes_by_stage:
            areas = lakes_by_stage[stage][area_col].values
            summary_text += f"{stage.capitalize()}:\n"
            summary_text += f"  N = {len(areas):,}\n"
            summary_text += f"  Min = {areas.min():.4f} km²\n"
            summary_text += f"  10th %ile = {np.quantile(areas, 0.1):.4f} km²\n"
            summary_text += f"  Median = {np.median(areas):.4f} km²\n\n"

    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('F. Summary Statistics', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detection bias diagnostic saved: {save_path}")
    else:
        plt.savefig(f"{OUTPUT_DIR}/detection_bias_diagnostic.png", dpi=300, bbox_inches='tight')
        print(f"Detection bias diagnostic saved: {OUTPUT_DIR}/detection_bias_diagnostic.png")

    plt.show()


def run_detection_bias_analysis(
    lakes_classified,
    save_figures: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete detection bias analysis pipeline.

    Parameters
    ----------
    lakes_classified : GeoDataFrame
        Lakes with glacial_stage classification
    save_figures : bool
        Save diagnostic figures
    verbose : bool
        Print progress

    Returns
    -------
    dict
        'truncation_tests': Results from size distribution tests
        'detection_curves': Estimated detection probability curves
        'corrected_densities': Bias-corrected density estimates
        'diagnostic_figure': Path to diagnostic figure (if saved)

    Examples
    --------
    >>> results = run_detection_bias_analysis(lakes_classified)
    >>> corrected = results['corrected_densities']
    >>> print(f"Wisconsin density (corrected): {corrected['wisconsin']:.1f}")
    """
    ensure_output_dir()

    if verbose:
        print("="*60)
        print("DETECTION BIAS ANALYSIS")
        print("="*60)

    # Group lakes by stage
    lakes_by_stage = {}
    for stage in lakes_classified['glacial_stage'].unique():
        lakes_by_stage[stage] = lakes_classified[
            lakes_classified['glacial_stage'] == stage
        ].copy()

        if verbose:
            print(f"\n{stage.capitalize()}: {len(lakes_by_stage[stage]):,} lakes")

    # 1. Test for truncation
    if verbose:
        print("\n1. Testing for size distribution truncation...")

    truncation_results = detect_size_distribution_truncation(
        lakes_by_stage,
        method='ks_test'
    )

    if verbose and 'ks_statistics' in truncation_results:
        print("\nKolmogorov-Smirnov Tests:")
        for comparison, result in truncation_results['ks_statistics'].items():
            sig = "***" if result['significant'] else ""
            print(f"  {comparison}: KS = {result['ks_statistic']:.3f}, p = {result['p_value']:.3e} {sig}")

    # 2. Estimate detection curves
    if verbose:
        print("\n2. Estimating detection probability curves...")

    detection_curves = estimate_detection_curve_by_stage(
        lakes_by_stage,
        reference_stage='wisconsin'
    )

    # 3. Compute raw and corrected densities
    if verbose:
        print("\n3. Computing bias-corrected densities...")

    raw_densities = compute_lake_density_by_glacial_stage(
        lakes_classified,
        verbose=False
    )

    corrected_densities = correct_density_for_detection_bias(
        raw_densities,
        detection_curves,
        method='mean_probability'
    )

    if verbose:
        print("\nDensity Comparison (lakes/1000 km²):")
        print(f"{'Stage':<15} {'Raw':<12} {'Corrected':<12} {'Change':<10}")
        print("-" * 50)
        for stage in ['wisconsin', 'illinoian', 'driftless']:
            if stage in raw_densities:
                raw = raw_densities[stage]
                corr = corrected_densities[stage]
                change = (corr - raw) / raw * 100
                print(f"{stage.capitalize():<15} {raw:<12.1f} {corr:<12.1f} {change:>+.1f}%")

    # 4. Create diagnostic figure
    diagnostic_path = None
    if save_figures:
        if verbose:
            print("\n4. Creating diagnostic figure...")

        diagnostic_path = f"{OUTPUT_DIR}/detection_bias_diagnostic.png"
        plot_detection_bias_diagnostic(
            lakes_by_stage,
            detection_curves=detection_curves,
            save_path=diagnostic_path
        )

    results = {
        'truncation_tests': truncation_results,
        'detection_curves': detection_curves,
        'raw_densities': raw_densities,
        'corrected_densities': corrected_densities,
        'diagnostic_figure': diagnostic_path
    }

    if verbose:
        print("\n" + "="*60)
        print("DETECTION BIAS ANALYSIS COMPLETE")
        print("="*60)

    return results


# Main execution
if __name__ == "__main__":
    print("Detection Bias Module")
    print("Run from main pipeline using run_detection_bias_analysis()")
    print("\nExample:")
    print("  from lake_analysis import detection_bias")
    print("  results = detection_bias.run_detection_bias_analysis(lakes_classified)")
