"""
Size-Stratified Lake Half-Life Analysis Module
================================================

This module extends the glacial chronosequence analysis to investigate whether
lake density decay rates vary by lake size. The key scientific question is:

    Do small lakes have shorter half-lives than large lakes?

This could reflect differences in:
- Sedimentation rates (scaling with lake volume/area ratio)
- Erosional infilling (depends on catchment/lake size ratio)
- Evaporative concentration (surface area/volume scaling)

The analysis uses Bayesian exponential decay models fitted separately for each
lake size class to estimate half-lives while accounting for:
- Age uncertainty in glacial stages
- Poisson uncertainty in lake counts
- Detection limit issues in older landscapes

Key Features:
-------------
- Detection limit diagnostics (are small lakes under-detected in older terrains?)
- Size-stratified density calculations by glacial stage
- Bayesian half-life estimation for each size class
- Statistical tests for half-life vs size relationship
- Comprehensive visualizations

Dependencies:
-------------
- numpy, pandas, matplotlib, scipy
- pymc (optional but recommended for Bayesian analysis)
- arviz (for posterior diagnostics)

Author: Lake Analysis Project
Integration: Adapted from standalone template for integration with existing
             glacial chronosequence analysis framework
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import PyMC - will gracefully degrade if not available
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("WARNING: PyMC not installed. Bayesian analysis will be unavailable.")
    print("Install with: pip install pymc arviz")

# Handle imports for both package and direct execution
try:
    from .config import OUTPUT_DIR, COLS, ensure_output_dir, PLOT_PARAMS, PLOT_STYLE
    from .glacial_chronosequence import (
        load_wisconsin_extent,
        load_illinoian_extent,
        load_driftless_area,
        classify_lakes_by_glacial_extent
    )
except ImportError:
    from config import OUTPUT_DIR, COLS, ensure_output_dir, PLOT_PARAMS, PLOT_STYLE
    from glacial_chronosequence import (
        load_wisconsin_extent,
        load_illinoian_extent,
        load_driftless_area,
        classify_lakes_by_glacial_extent
    )

# Set matplotlib style
plt.style.use(PLOT_STYLE)
for key, val in PLOT_PARAMS.items():
    plt.rcParams[key] = val

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default size bins for analysis (min, max in km², label)
DEFAULT_SIZE_BINS = [
    (0.05, 0.1, 'tiny'),
    (0.1, 0.25, 'very_small'),
    (0.25, 0.5, 'small'),
    (0.5, 1.0, 'medium_small'),
    (1.0, 2.5, 'medium'),
    (2.5, 10.0, 'large'),
    (10.0, np.inf, 'very_large')
]

# Default landscape areas (km²) - will be calculated dynamically if possible
# These are approximate values based on glacial boundary analyses
DEFAULT_LANDSCAPE_AREAS = {
    'Wisconsin': 1225000,
    'Illinoian': 145000,
    'Driftless': 25500
}

# Default age estimates for each glacial stage (ka = thousands of years)
DEFAULT_AGE_ESTIMATES = {
    'Wisconsin': {'mean': 20, 'std': 5},
    'Illinoian': {'mean': 160, 'std': 30},
    'Driftless': {'mean': 1500, 'std': 500}
}

# Bayesian sampling parameters
DEFAULT_BAYESIAN_PARAMS = {
    'n_samples': 2000,
    'n_tune': 1000,
    'n_chains': 4,
    'target_accept': 0.95
}

# Color scheme for glacial stages
STAGE_COLORS = {
    'Wisconsin': '#3498db',     # Blue
    'Illinoian': '#e74c3c',     # Red
    'Driftless': '#2ecc71',     # Green
    'unclassified': '#95a5a6'   # Gray
}


# =============================================================================
# DETECTION LIMIT DIAGNOSTICS
# =============================================================================

def detection_limit_diagnostics(df, area_col='AREASQKM', stage_col='glacial_stage',
                                min_lake_area=0.05, output_dir=None, verbose=True):
    """
    Comprehensive detection limit analysis.

    Key question: Are small lakes systematically under-represented in older
    surfaces due to mapping issues, or does this reflect real lake loss?

    Parameters
    ----------
    df : DataFrame
        Lake data with area and glacial stage columns
    area_col : str
        Name of lake area column (in km²)
    stage_col : str
        Name of glacial stage classification column
    min_lake_area : float
        Minimum lake area already applied to data (km²)
    output_dir : str, optional
        Directory to save figure. If None, uses OUTPUT_DIR from config.
    verbose : bool
        Print diagnostic messages

    Returns
    -------
    fig : matplotlib.Figure
        6-panel diagnostic figure
    diagnostics : dict
        Dictionary containing diagnostic statistics
    """

    if output_dir is None:
        output_dir = ensure_output_dir()

    stages = ['Wisconsin', 'Illinoian', 'Driftless']
    colors = STAGE_COLORS

    # Filter to relevant stages
    df_filtered = df[df[stage_col].isin(stages)].copy()

    if verbose:
        print("\n" + "=" * 70)
        print("DETECTION LIMIT DIAGNOSTICS")
        print("=" * 70)
        for stage in stages:
            n = len(df_filtered[df_filtered[stage_col] == stage])
            print(f"  {stage}: {n:,} lakes")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Detection Limit Diagnostics', fontsize=14, fontweight='bold')

    # Panel A: Log-histogram of lake sizes
    ax = axes[0, 0]
    for stage in stages:
        subset = df_filtered[df_filtered[stage_col] == stage][area_col]
        if len(subset) > 0:
            ax.hist(np.log10(subset), bins=50, alpha=0.5,
                   label=f'{stage} (n={len(subset):,})',
                   color=colors.get(stage, 'gray'),
                   density=True)

    ax.axvline(np.log10(min_lake_area), color='red', linestyle='--',
               linewidth=2, label=f"Min cutoff ({min_lake_area} km²)")
    ax.set_xlabel('log₁₀(Lake Area, km²)')
    ax.set_ylabel('Density')
    ax.set_title('A) Size Distribution\n(Parallel left edges = consistent detection)')
    ax.legend(fontsize=8)

    # Panel B: Empirical CDF
    ax = axes[0, 1]
    for stage in stages:
        subset = df_filtered[df_filtered[stage_col] == stage][area_col].sort_values()
        if len(subset) > 0:
            cdf = np.arange(1, len(subset) + 1) / len(subset)
            ax.plot(subset, cdf, label=stage, color=colors.get(stage, 'gray'), linewidth=2)

    ax.set_xscale('log')
    ax.axvline(0.1, color='orange', linestyle='--', alpha=0.7, label='0.1 km²')
    ax.axvline(0.5, color='purple', linestyle='--', alpha=0.7, label='0.5 km²')
    ax.set_xlabel('Lake Area (km²)')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('B) Empirical CDF\n(Divergence at small sizes = potential issue)')
    ax.legend(fontsize=8)

    # Panel C: Size class proportions
    ax = axes[0, 2]
    size_bins_edges = [0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 100]
    bin_labels = ['0.05-0.1', '0.1-0.25', '0.25-0.5', '0.5-1.0', '1.0-5.0', '>5.0']

    proportions = {}
    for stage in stages:
        subset = df_filtered[df_filtered[stage_col] == stage][area_col]
        if len(subset) > 0:
            counts, _ = np.histogram(subset, bins=size_bins_edges)
            proportions[stage] = counts / counts.sum()

    x = np.arange(len(bin_labels))
    width = 0.8 / len(stages)
    for i, stage in enumerate(stages):
        if stage in proportions:
            ax.bar(x + i*width - 0.4 + width/2, proportions[stage], width,
                  label=stage, color=colors.get(stage, 'gray'), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Size Class (km²)')
    ax.set_ylabel('Proportion of Lakes')
    ax.set_title('C) Size Class Proportions\n(Real decay: depleted small lakes in older stages)')
    ax.legend(fontsize=8)

    # Panel D: Minimum sizes by stage
    ax = axes[1, 0]
    stats_data = []
    for stage in stages:
        subset = df_filtered[df_filtered[stage_col] == stage][area_col]
        if len(subset) > 0:
            stats_data.append({
                'stage': stage,
                'min': subset.min(),
                'p5': np.percentile(subset, 5),
                'p10': np.percentile(subset, 10),
                'median': np.median(subset)
            })

    stats_df = pd.DataFrame(stats_data)
    x = np.arange(len(stats_df))

    ax.bar(x - 0.3, stats_df['min'], 0.2, label='Minimum', color='navy')
    ax.bar(x - 0.1, stats_df['p5'], 0.2, label='5th %ile', color='steelblue')
    ax.bar(x + 0.1, stats_df['p10'], 0.2, label='10th %ile', color='lightblue')
    ax.bar(x + 0.3, stats_df['median'], 0.2, label='Median', color='skyblue')

    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['stage'])
    ax.set_ylabel('Lake Area (km²)')
    ax.set_title('D) Size Statistics by Stage\n(Similar minimums = consistent detection)')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # Panel E: KS test results for large lakes
    ax = axes[1, 1]
    ax.axis('off')

    ks_results = []
    thresholds = [0.25, 0.5, 1.0, 2.5]

    for thresh in thresholds:
        for i, stage1 in enumerate(stages):
            for stage2 in stages[i+1:]:
                subset1 = df_filtered[(df_filtered[stage_col] == stage1) &
                                     (df_filtered[area_col] > thresh)][area_col]
                subset2 = df_filtered[(df_filtered[stage_col] == stage2) &
                                     (df_filtered[area_col] > thresh)][area_col]

                if len(subset1) > 10 and len(subset2) > 10:
                    stat, pval = stats.ks_2samp(subset1, subset2)
                    ks_results.append({
                        'threshold': f'>{thresh}',
                        'comparison': f'{stage1[:4]} vs {stage2[:4]}',
                        'n1': len(subset1),
                        'n2': len(subset2),
                        'KS': f'{stat:.3f}',
                        'p': f'{pval:.3f}',
                        'same?': '✓' if pval > 0.05 else '✗'
                    })

    if ks_results:
        ks_df = pd.DataFrame(ks_results)
        table = ax.table(
            cellText=ks_df.values,
            colLabels=ks_df.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.4)
    ax.set_title('E) KS Tests: Are Large Lake Distributions Similar?\n(✓ = consistent detection above threshold)',
                 fontsize=10, pad=20)

    # Panel F: Interpretation guide
    ax = axes[1, 2]
    ax.axis('off')

    text = """
    INTERPRETATION GUIDE
    ════════════════════════════════════════

    IF detection is consistent across stages:
    • Minimum lake sizes should be similar
    • CDFs should be parallel at small sizes
    • KS tests for large lakes: p > 0.05

    IF small lakes under-detected in Driftless:
    • Minimum size larger in Driftless
    • CDF shifted right for small sizes
    • Consider raising analysis threshold

    RECOMMENDATION:
    • Find threshold where KS test passes
    • Use that as conservative x_min for
      size-stratified analysis
    • Report sensitivity to threshold choice
    """
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'detection_limit_diagnostics.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nSaved: {fig_path}")

    # Compile diagnostics
    diagnostics = {
        'size_stats': stats_df,
        'ks_tests': pd.DataFrame(ks_results) if ks_results else None,
        'proportions': proportions
    }

    return fig, diagnostics


# =============================================================================
# SIZE-STRATIFIED DENSITY CALCULATIONS
# =============================================================================

def calculate_size_stratified_densities(df, landscape_areas, age_estimates,
                                       area_col='AREASQKM', stage_col='glacial_stage',
                                       size_bins=None, verbose=True):
    """
    Calculate lake density by size class and glacial stage.

    Parameters
    ----------
    df : DataFrame
        Lake data with area and glacial stage columns
    landscape_areas : dict
        Dictionary mapping stage names to land area (km²)
    age_estimates : dict
        Dictionary mapping stage names to age info (mean, std in ka)
    area_col : str
        Name of lake area column (in km²)
    stage_col : str
        Name of glacial stage classification column
    size_bins : list of tuples, optional
        List of (min, max, label) for each size class
        If None, uses DEFAULT_SIZE_BINS
    verbose : bool
        Print progress messages

    Returns
    -------
    DataFrame
        Density table with columns:
        - stage, size_class, size_min, size_max
        - n_lakes, landscape_area_km2
        - density_per_1000km2, density_se
        - age_mean_ka, age_std_ka
    """

    if size_bins is None:
        size_bins = DEFAULT_SIZE_BINS

    stages = list(landscape_areas.keys())

    if verbose:
        print("\n" + "=" * 70)
        print("SIZE-STRATIFIED DENSITY CALCULATION")
        print("=" * 70)

    results = []

    for stage in stages:
        stage_df = df[df[stage_col] == stage]
        land_area = landscape_areas.get(stage, 1)
        ages = age_estimates.get(stage, {'mean': 100, 'std': 50})

        if verbose:
            print(f"\n{stage}:")
            print(f"  Total lakes: {len(stage_df):,}")
            print(f"  Landscape area: {land_area:,.0f} km²")

        for low, high, size_class in size_bins:
            mask = (stage_df[area_col] >= low) & (stage_df[area_col] < high)
            n_lakes = mask.sum()

            # Density per 1000 km²
            density = (n_lakes / land_area) * 1000

            # Poisson SE on count, propagated to density
            density_se = (np.sqrt(max(n_lakes, 1)) / land_area) * 1000

            if verbose and n_lakes > 0:
                print(f"    {size_class:15s} [{low:.2f}-{high:.2f} km²]: "
                      f"n={n_lakes:6,}, density={density:7.2f} ± {density_se:.2f}")

            results.append({
                'stage': stage,
                'size_class': size_class,
                'size_min': low,
                'size_max': high,
                'n_lakes': n_lakes,
                'landscape_area_km2': land_area,
                'density_per_1000km2': density,
                'density_se': density_se,
                'age_mean_ka': ages['mean'],
                'age_std_ka': ages['std']
            })

    density_df = pd.DataFrame(results)

    if verbose:
        print(f"\nTotal size classes × stages: {len(density_df)}")

    return density_df


def plot_size_stratified_densities(density_df, output_dir=None, verbose=True):
    """
    Visualize density patterns by size class.

    Parameters
    ----------
    density_df : DataFrame
        Output from calculate_size_stratified_densities()
    output_dir : str, optional
        Directory to save figure. If None, uses OUTPUT_DIR from config.
    verbose : bool
        Print progress messages

    Returns
    -------
    fig : matplotlib.Figure
        4-panel visualization
    """

    if output_dir is None:
        output_dir = ensure_output_dir()

    stages = density_df['stage'].unique()
    colors = STAGE_COLORS

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Lake Density by Size Class and Glacial Stage',
                 fontsize=14, fontweight='bold')

    size_classes = density_df['size_class'].unique()

    # Panel A: Density bar chart
    ax = axes[0, 0]
    x = np.arange(len(size_classes))
    width = 0.8 / len(stages)

    for i, stage in enumerate(stages):
        stage_data = density_df[density_df['stage'] == stage].set_index('size_class')
        densities = [stage_data.loc[sc, 'density_per_1000km2'] if sc in stage_data.index else 0
                    for sc in size_classes]
        errors = [stage_data.loc[sc, 'density_se'] if sc in stage_data.index else 0
                 for sc in size_classes]
        ax.bar(x + i*width - 0.4 + width/2, densities, width,
               label=stage, color=colors.get(stage, 'gray'),
               yerr=errors, capsize=2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(size_classes, rotation=45, ha='right')
    ax.set_ylabel('Lakes per 1,000 km²')
    ax.set_title('A) Absolute Density by Size Class')
    ax.legend()
    ax.set_yscale('log')

    # Panel B: Decay trajectories
    ax = axes[0, 1]

    for size_class in size_classes:
        class_data = density_df[density_df['size_class'] == size_class]
        if len(class_data) == len(stages):
            ages = class_data['age_mean_ka'].values
            densities = class_data['density_per_1000km2'].values
            if all(densities > 0):
                ax.plot(ages, densities, 'o-', label=size_class, linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Landscape Age (ka)')
    ax.set_ylabel('Lakes per 1,000 km²')
    ax.set_title('B) Density vs Age by Size Class')
    ax.legend(fontsize=8, ncol=2)

    # Panel C: Relative survival (normalized to Wisconsin)
    ax = axes[1, 0]

    wisc_densities = density_df[density_df['stage'] == 'Wisconsin'].set_index('size_class')['density_per_1000km2']

    for size_class in size_classes:
        class_data = density_df[density_df['size_class'] == size_class].copy()
        wisc_d = wisc_densities.get(size_class, 0)

        if wisc_d > 0:
            class_data['relative_density'] = class_data['density_per_1000km2'] / wisc_d
            ax.plot(class_data['age_mean_ka'], class_data['relative_density'],
                   'o-', label=size_class, linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% survival')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='25% survival')
    ax.set_xlabel('Landscape Age (ka)')
    ax.set_ylabel('Relative Density (vs Wisconsin)')
    ax.set_title('C) Survival Fraction by Size Class')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.3)

    # Panel D: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    # Pivot for display
    pivot = density_df.pivot(index='size_class', columns='stage', values='density_per_1000km2')
    pivot = pivot.reindex(columns=stages)

    # Add survival columns
    if 'Wisconsin' in pivot.columns:
        for stage in [s for s in stages if s != 'Wisconsin']:
            if stage in pivot.columns:
                pivot[f'Surv→{stage[:4]}'] = (pivot[stage] / pivot['Wisconsin'] * 100).round(1).astype(str) + '%'

    # Format density columns
    for stage in stages:
        if stage in pivot.columns:
            pivot[stage] = pivot[stage].round(1)

    table_data = pivot.reset_index().values
    col_labels = ['Size Class'] + list(pivot.columns)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.6)
    ax.set_title('D) Density (per 1000 km²) and Survival Ratios', pad=20)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'size_stratified_density_patterns.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nSaved: {fig_path}")

    return fig


# =============================================================================
# BAYESIAN HALF-LIFE ESTIMATION
# =============================================================================

def fit_size_stratified_halflife_models(density_df, bayesian_params=None,
                                        min_lakes=10, verbose=True):
    """
    Fit Bayesian exponential decay models for each size class.

    Model: D(t) = D₀ × exp(-k × t)
    Half-life: t½ = ln(2) / k

    Parameters
    ----------
    density_df : DataFrame
        Output from calculate_size_stratified_densities()
    bayesian_params : dict, optional
        Sampling parameters (n_samples, n_tune, n_chains, target_accept)
        If None, uses DEFAULT_BAYESIAN_PARAMS
    min_lakes : int
        Minimum total lakes needed for a size class to be analyzed
    verbose : bool
        Print progress messages

    Returns
    -------
    results_df : DataFrame or None
        DataFrame with half-life estimates for each size class, or None if
        PyMC is not available or no size classes have enough data
    traces : dict
        Dictionary mapping size_class to PyMC trace objects
    """

    if not PYMC_AVAILABLE:
        print("PyMC not available - skipping Bayesian analysis")
        return None, {}

    if bayesian_params is None:
        bayesian_params = DEFAULT_BAYESIAN_PARAMS

    stages = density_df['stage'].unique()
    size_classes = density_df['size_class'].unique()

    results = []
    traces = {}

    if verbose:
        print("\n" + "=" * 70)
        print("BAYESIAN HALF-LIFE ESTIMATION BY SIZE CLASS")
        print("=" * 70)

    for size_class in size_classes:
        class_data = density_df[density_df['size_class'] == size_class].copy()

        # Check sample size
        total_lakes = class_data['n_lakes'].sum()
        if total_lakes < min_lakes:
            if verbose:
                print(f"\n{size_class}: SKIPPED (only {total_lakes} lakes, need ≥{min_lakes})")
            continue

        # Check for decay pattern
        stage_list = class_data['stage'].values
        if 'Wisconsin' not in stage_list or stages[-1] not in stage_list:
            if verbose:
                print(f"\n{size_class}: SKIPPED (missing required stages)")
            continue

        wisc_d = class_data[class_data['stage'] == 'Wisconsin']['density_per_1000km2'].values[0]
        last_d = class_data[class_data['stage'] == stages[-1]]['density_per_1000km2'].values[0]

        if last_d >= wisc_d:
            if verbose:
                print(f"\n{size_class}: SKIPPED (no decay: Wisc={wisc_d:.2f}, {stages[-1]}={last_d:.2f})")
            continue

        if verbose:
            print(f"\n{size_class}:")
            for stage in stages:
                row = class_data[class_data['stage'] == stage]
                if len(row) > 0:
                    n = row['n_lakes'].values[0]
                    d = row['density_per_1000km2'].values[0]
                    print(f"  {stage:12s}: n={n:7,}, density={d:7.2f}")

        # Prepare data
        ages = class_data['age_mean_ka'].values
        age_errors = class_data['age_std_ka'].values
        densities = class_data['density_per_1000km2'].values
        density_errors = np.maximum(class_data['density_se'].values, 0.1)

        # Build PyMC model
        with pm.Model() as model:
            # Prior on initial density - weakly informative
            D0 = pm.HalfNormal('D0', sigma=wisc_d * 2)

            # Prior on half-life (in ka) - broad log-normal
            # Centered around ~500 ka with wide uncertainty
            halflife = pm.LogNormal('halflife', mu=np.log(500), sigma=1.5)
            k = pm.Deterministic('k', np.log(2) / halflife)

            # Account for age uncertainty with latent true ages
            true_ages = pm.Normal('true_ages', mu=ages, sigma=age_errors, shape=len(ages))

            # Expected density under exponential decay
            mu = D0 * pm.math.exp(-k * true_ages)

            # Observation noise
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)
            total_sigma = pm.math.sqrt(density_errors**2 + sigma_obs**2)

            # Likelihood
            D_obs = pm.Normal('D_obs', mu=mu, sigma=total_sigma, observed=densities)

            # Sample
            trace = pm.sample(
                bayesian_params['n_samples'],
                tune=bayesian_params['n_tune'],
                chains=bayesian_params['n_chains'],
                target_accept=bayesian_params['target_accept'],
                return_inferencedata=True,
                progressbar=False,  # Suppress progress bar for cleaner output
                random_seed=42
            )

        # Extract posteriors
        halflife_samples = trace.posterior['halflife'].values.flatten()
        D0_samples = trace.posterior['D0'].values.flatten()

        halflife_median = np.median(halflife_samples)
        halflife_mean = np.mean(halflife_samples)
        halflife_ci = np.percentile(halflife_samples, [2.5, 97.5])

        D0_mean = np.mean(D0_samples)
        D0_ci = np.percentile(D0_samples, [2.5, 97.5])

        # Check convergence
        rhat_vals = az.rhat(trace)
        max_rhat = max(rhat_vals['halflife'].values.max(), rhat_vals['D0'].values.max())

        if verbose:
            print(f"  → Half-life: {halflife_median:.0f} ka (95% CI: {halflife_ci[0]:.0f}–{halflife_ci[1]:.0f})")
            print(f"  → D₀: {D0_mean:.1f} (95% CI: {D0_ci[0]:.1f}–{D0_ci[1]:.1f})")
            print(f"  → Convergence (R-hat): {max_rhat:.3f} {'✓' if max_rhat < 1.05 else '⚠'}")

        results.append({
            'size_class': size_class,
            'size_min': class_data['size_min'].iloc[0],
            'size_max': class_data['size_max'].iloc[0],
            'n_lakes_total': total_lakes,
            'halflife_mean': halflife_mean,
            'halflife_median': halflife_median,
            'halflife_ci_low': halflife_ci[0],
            'halflife_ci_high': halflife_ci[1],
            'D0_mean': D0_mean,
            'D0_ci_low': D0_ci[0],
            'D0_ci_high': D0_ci[1],
            'max_rhat': max_rhat
        })

        traces[size_class] = trace

    results_df = pd.DataFrame(results) if results else None

    if verbose and results_df is not None:
        print(f"\nSuccessfully fitted {len(results_df)} size classes")

    return results_df, traces


def plot_bayesian_halflife_results(results_df, density_df, traces, output_dir=None, verbose=True):
    """
    Comprehensive visualization of Bayesian half-life results.

    Parameters
    ----------
    results_df : DataFrame
        Output from fit_size_stratified_halflife_models()
    density_df : DataFrame
        Original density data
    traces : dict
        Dictionary of PyMC traces for each size class
    output_dir : str, optional
        Directory to save figure. If None, uses OUTPUT_DIR from config.
    verbose : bool
        Print progress messages

    Returns
    -------
    fig : matplotlib.Figure or None
        4-panel visualization, or None if no results
    """

    if results_df is None or len(results_df) == 0:
        print("No Bayesian results to plot")
        return None

    if output_dir is None:
        output_dir = ensure_output_dir()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Bayesian Half-Life Analysis by Lake Size', fontsize=14, fontweight='bold')

    # Calculate geometric mean size for each class
    results_df = results_df.copy()
    results_df['size_geom'] = np.sqrt(
        results_df['size_min'] * results_df['size_max'].replace(np.inf, 100)
    )

    # Panel A: Half-life vs Size (THE KEY PLOT)
    ax1 = fig.add_subplot(2, 2, 1)

    ax1.errorbar(
        results_df['size_geom'],
        results_df['halflife_median'],
        yerr=[results_df['halflife_median'] - results_df['halflife_ci_low'],
              results_df['halflife_ci_high'] - results_df['halflife_median']],
        fmt='o', markersize=12, capsize=6, capthick=2, linewidth=2,
        color='darkblue', ecolor='steelblue'
    )

    # Annotate
    for _, row in results_df.iterrows():
        ax1.annotate(
            row['size_class'],
            (row['size_geom'], row['halflife_median']),
            textcoords="offset points", xytext=(8, 0), fontsize=9
        )

    # Fit trend line if enough points
    if len(results_df) >= 3:
        log_size = np.log10(results_df['size_geom'])
        log_half = np.log10(results_df['halflife_median'])
        slope, intercept, r, p, se = stats.linregress(log_size, log_half)

        x_fit = np.logspace(np.log10(results_df['size_geom'].min()),
                           np.log10(results_df['size_geom'].max()), 50)
        y_fit = 10**(slope * np.log10(x_fit) + intercept)
        ax1.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                label=f't½ ∝ Size^{slope:.2f} (R²={r**2:.2f}, p={p:.3f})')
        ax1.legend(loc='lower right')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Lake Size (km², geometric mean of bin)', fontsize=11)
    ax1.set_ylabel('Half-life (ka)', fontsize=11)
    ax1.set_title('A) Lake Half-life vs Size\n(Key test: positive slope = smaller lakes decay faster)')
    ax1.grid(True, alpha=0.3)

    # Panel B: Posterior distributions
    ax2 = fig.add_subplot(2, 2, 2)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(traces)))

    for (size_class, trace), color in zip(traces.items(), colors):
        samples = trace.posterior['halflife'].values.flatten()
        ax2.hist(samples, bins=50, alpha=0.4, color=color, density=True, label=size_class)
        ax2.axvline(np.median(samples), color=color, linestyle='--', linewidth=2)

    ax2.set_xlabel('Half-life (ka)')
    ax2.set_ylabel('Posterior Density')
    ax2.set_title('B) Posterior Distributions')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel C: Fitted decay curves
    ax3 = fig.add_subplot(2, 2, 3)

    ages_plot = np.linspace(1, 3000, 200)

    for (size_class, trace), color in zip(traces.items(), colors):
        D0_samples = trace.posterior['D0'].values.flatten()
        k_samples = trace.posterior['k'].values.flatten()

        # Median fit
        D0_med = np.median(D0_samples)
        k_med = np.median(k_samples)
        ax3.plot(ages_plot, D0_med * np.exp(-k_med * ages_plot),
                color=color, linewidth=2, label=size_class)

        # Uncertainty band (30 draws)
        for _ in range(30):
            idx = np.random.randint(len(D0_samples))
            ax3.plot(ages_plot, D0_samples[idx] * np.exp(-k_samples[idx] * ages_plot),
                    color=color, alpha=0.03, linewidth=0.5)

        # Data points
        class_data = density_df[density_df['size_class'] == size_class]
        ax3.scatter(class_data['age_mean_ka'], class_data['density_per_1000km2'],
                   color=color, s=100, edgecolor='black', zorder=5)

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Landscape Age (ka)')
    ax3.set_ylabel('Lake Density (per 1000 km²)')
    ax3.set_title('C) Fitted Decay Curves')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Panel D: Summary table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['size_class'],
            f"{row['size_min']:.2f}–{row['size_max']:.1f}" if row['size_max'] < 100 else f">{row['size_min']:.0f}",
            f"{row['n_lakes_total']:,}",
            f"{row['halflife_median']:.0f}",
            f"[{row['halflife_ci_low']:.0f}, {row['halflife_ci_high']:.0f}]",
            f"{'✓' if row['max_rhat'] < 1.05 else '⚠'}"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Size Class', 'Range (km²)', 'n Lakes', 't½ (ka)', '95% CI', 'Conv.'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    ax4.set_title('D) Summary of Estimates', pad=20)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'size_stratified_bayesian_results.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nSaved: {fig_path}")

    return fig


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def test_halflife_size_relationship(results_df, verbose=True):
    """
    Test for significant relationship between lake size and half-life.

    Parameters
    ----------
    results_df : DataFrame
        Output from fit_size_stratified_halflife_models()
    verbose : bool
        Print test results

    Returns
    -------
    dict
        Dictionary containing test statistics
    """

    if results_df is None or len(results_df) < 3:
        if verbose:
            print("Need at least 3 size classes for statistical testing")
        return None

    # Calculate geometric mean sizes
    sizes = np.sqrt(results_df['size_min'] * results_df['size_max'].replace(np.inf, 100))
    halflives = results_df['halflife_median']

    # Spearman correlation (rank-based, robust)
    rho, p_spearman = stats.spearmanr(sizes, halflives)

    # Pearson correlation on log-log (tests power-law relationship)
    r, p_pearson = stats.pearsonr(np.log10(sizes), np.log10(halflives))

    # Linear regression on log-log
    slope, intercept, _, p_reg, stderr = stats.linregress(np.log10(sizes), np.log10(halflives))

    results = {
        'spearman_rho': rho,
        'spearman_p': p_spearman,
        'pearson_r': r,
        'pearson_p': p_pearson,
        'power_law_exponent': slope,
        'power_law_exponent_se': stderr,
        'power_law_p': p_reg,
        'n_size_classes': len(results_df)
    }

    if verbose:
        print("\n" + "=" * 70)
        print("STATISTICAL TEST: HALF-LIFE ~ SIZE RELATIONSHIP")
        print("=" * 70)
        print(f"\nSpearman ρ = {rho:.3f} (p = {p_spearman:.4f})")
        print(f"Pearson r (log-log) = {r:.3f} (p = {p_pearson:.4f})")
        print(f"Power-law scaling: t½ ∝ Size^{slope:.2f} ± {stderr:.2f}")

        if p_spearman < 0.05:
            if rho > 0:
                print("\n→ RESULT: Significant positive relationship (p < 0.05)")
                print("  Larger lakes appear to persist longer")
            else:
                print("\n→ RESULT: Significant negative relationship (p < 0.05)")
                print("  Smaller lakes appear to persist longer (unexpected!)")
        elif p_spearman < 0.1:
            print("\n→ RESULT: Suggestive relationship (p < 0.10)")
        else:
            print("\n→ RESULT: No significant relationship detected (p ≥ 0.10)")
            print("  May need more data or different size binning")

    return results


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_size_stratified_analysis(lake_gdf, landscape_areas=None, age_estimates=None,
                                 size_bins=None, bayesian_params=None,
                                 area_col='AREASQKM', stage_col='glacial_stage',
                                 min_lake_area=0.05, min_lakes_per_class=10,
                                 output_dir=None, verbose=True):
    """
    Complete size-stratified lake half-life analysis pipeline.

    This is the main entry point for the analysis. It:
    1. Runs detection limit diagnostics
    2. Calculates size-stratified densities
    3. Fits Bayesian half-life models for each size class
    4. Tests for size-halflife relationship
    5. Generates all visualizations

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial stage classifications
    landscape_areas : dict, optional
        Dictionary mapping stage names to land area (km²)
        If None, uses DEFAULT_LANDSCAPE_AREAS
    age_estimates : dict, optional
        Dictionary mapping stage names to age info (mean, std in ka)
        If None, uses DEFAULT_AGE_ESTIMATES
    size_bins : list of tuples, optional
        List of (min, max, label) for each size class
        If None, uses DEFAULT_SIZE_BINS
    bayesian_params : dict, optional
        Sampling parameters for PyMC
        If None, uses DEFAULT_BAYESIAN_PARAMS
    area_col : str
        Name of lake area column (in km²)
    stage_col : str
        Name of glacial stage classification column
    min_lake_area : float
        Minimum lake area already applied to data (km²)
    min_lakes_per_class : int
        Minimum total lakes needed for a size class to be analyzed
    output_dir : str, optional
        Directory to save outputs. If None, uses OUTPUT_DIR from config.
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Dictionary containing all results:
        - density_df: Size-stratified density table
        - halflife_df: Bayesian half-life estimates
        - traces: PyMC trace objects
        - statistics: Statistical test results
        - diagnostics: Detection limit diagnostics
    """

    if output_dir is None:
        output_dir = ensure_output_dir()

    if landscape_areas is None:
        landscape_areas = DEFAULT_LANDSCAPE_AREAS

    if age_estimates is None:
        age_estimates = DEFAULT_AGE_ESTIMATES

    if verbose:
        print("\n" + "=" * 70)
        print("SIZE-STRATIFIED LAKE HALF-LIFE ANALYSIS")
        print("=" * 70)
        print(f"\nTotal lakes: {len(lake_gdf):,}")
        print(f"Minimum lake area: {min_lake_area} km²")
        print(f"Minimum lakes per size class: {min_lakes_per_class}")

    # Convert to DataFrame if GeoDataFrame
    if hasattr(lake_gdf, 'geometry'):
        df = pd.DataFrame(lake_gdf.drop(columns='geometry'))
    else:
        df = lake_gdf

    # Step 1: Detection diagnostics
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 1: Detection Limit Diagnostics")
        print("-" * 70)

    fig_diag, diagnostics = detection_limit_diagnostics(
        df, area_col=area_col, stage_col=stage_col,
        min_lake_area=min_lake_area, output_dir=output_dir, verbose=verbose
    )

    # Step 2: Calculate densities
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 2: Size-Stratified Density Calculation")
        print("-" * 70)

    density_df = calculate_size_stratified_densities(
        df, landscape_areas, age_estimates,
        area_col=area_col, stage_col=stage_col,
        size_bins=size_bins, verbose=verbose
    )

    # Save density table
    density_path = os.path.join(output_dir, 'size_stratified_density.csv')
    density_df.to_csv(density_path, index=False)
    if verbose:
        print(f"\nSaved density table: {density_path}")

    # Plot densities
    fig_density = plot_size_stratified_densities(density_df, output_dir=output_dir, verbose=verbose)

    # Step 3: Bayesian analysis
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 3: Bayesian Half-Life Estimation")
        print("-" * 70)

    results_df, traces = fit_size_stratified_halflife_models(
        density_df, bayesian_params=bayesian_params,
        min_lakes=min_lakes_per_class, verbose=verbose
    )

    if results_df is not None and len(results_df) > 0:
        # Save results
        results_path = os.path.join(output_dir, 'size_stratified_halflife_results.csv')
        results_df.to_csv(results_path, index=False)
        if verbose:
            print(f"\nSaved half-life results: {results_path}")

        # Plot results
        fig_bayes = plot_bayesian_halflife_results(
            results_df, density_df, traces, output_dir=output_dir, verbose=verbose
        )

        # Step 4: Statistical testing
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 4: Statistical Testing")
            print("-" * 70)

        statistics = test_halflife_size_relationship(results_df, verbose=verbose)
    else:
        if verbose:
            print("\nNo size classes had sufficient data for Bayesian analysis")
        fig_bayes = None
        statistics = None

    # Compile all results
    results = {
        'density_df': density_df,
        'halflife_df': results_df,
        'traces': traces,
        'statistics': statistics,
        'diagnostics': diagnostics
    }

    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nOutputs saved to: {output_dir}")

    return results


# =============================================================================
# OVERALL BAYESIAN HALF-LIFE ANALYSIS (Not Size-Stratified)
# =============================================================================

def fit_overall_bayesian_halflife(
    density_by_stage,
    age_estimates=None,
    bayesian_params=None,
    verbose=True
):
    """
    Fit Bayesian exponential decay model to overall lake density across glacial stages.

    This estimates the OVERALL half-life for ALL lakes (not stratified by size).
    Model: D(t) = D₀ × exp(-k × t)
    Half-life: t½ = ln(2) / k

    Parameters
    ----------
    density_by_stage : dict or DataFrame
        Lake density data for each glacial stage.
        If dict: {'Wisconsin': 228.2, 'Illinoian': 202.8, 'Driftless': 69.4}
        If DataFrame: Must have columns 'stage', 'density_per_1000km2', 'n_lakes'
    age_estimates : dict, optional
        Age estimates for each stage: {'Wisconsin': {'mean': 20, 'std': 5}, ...}
        If None, uses DEFAULT_AGE_ESTIMATES
    bayesian_params : dict, optional
        PyMC sampling parameters. If None, uses DEFAULT_BAYESIAN_PARAMS
    verbose : bool
        Print progress messages

    Returns
    -------
    dict or None
        Results dict with:
        - halflife_mean, halflife_median, halflife_ci_low, halflife_ci_high (ka)
        - D0: Initial density posterior
        - k: Decay rate posterior
        - trace: PyMC InferenceData
        - summary: DataFrame with posterior summaries
        Returns None if PyMC not available or insufficient data
    """

    if not PYMC_AVAILABLE:
        print("PyMC not available - skipping overall Bayesian half-life analysis")
        return None

    if bayesian_params is None:
        bayesian_params = DEFAULT_BAYESIAN_PARAMS

    if age_estimates is None:
        age_estimates = DEFAULT_AGE_ESTIMATES

    if verbose:
        print("\n" + "=" * 70)
        print("OVERALL BAYESIAN HALF-LIFE ANALYSIS")
        print("=" * 70)

    # Parse input data
    if isinstance(density_by_stage, dict):
        stages = list(density_by_stage.keys())
        densities = np.array([density_by_stage[s] for s in stages])
        # Assume equal weight if n_lakes not provided
        n_lakes = np.ones(len(stages)) * 1000
    else:
        # DataFrame input
        stages = density_by_stage['glacial_stage'].values
        densities = density_by_stage['density_per_1000km2'].values
        n_lakes = density_by_stage.get('n_lakes', np.ones(len(stages)) * 1000).values

    # Get ages (use NaN for stages without age estimates)
    ages_mean = np.array([
        age_estimates.get(s, {}).get('mean', np.nan) for s in stages
    ])
    ages_std = np.array([
        age_estimates.get(s, {}).get('std', np.nan) for s in stages
    ])

    # Remove stages with NaN density or NaN age (e.g., 'unclassified')
    valid_mask = ~np.isnan(densities) & (densities > 0) & ~np.isnan(ages_mean)
    if valid_mask.sum() < 2:
        if verbose:
            print("  Insufficient valid stages for analysis (need ≥2)")
        return None

    stages = stages[valid_mask]
    densities = densities[valid_mask]
    n_lakes = n_lakes[valid_mask]
    ages_mean = ages_mean[valid_mask]
    ages_std = ages_std[valid_mask]

    if verbose:
        print(f"\nFitting exponential decay to {len(stages)} stages:")
        for i, stage in enumerate(stages):
            print(f"  {stage:12s}: density={densities[i]:7.2f}, "
                  f"age={ages_mean[i]:6.0f} ± {ages_std[i]:4.0f} ka, "
                  f"n={n_lakes[i]:,.0f}")

    # Density uncertainty from Poisson counting statistics
    density_se = np.sqrt(n_lakes) / (n_lakes / densities) / 1000  # Propagated SE
    density_se = np.maximum(density_se, densities * 0.05)  # At least 5% error

    # Build PyMC model
    with pm.Model() as model:
        # Prior on initial density (Wisconsin baseline)
        D0 = pm.HalfNormal('D0', sigma=densities.max() * 2)

        # Prior on half-life (in ka) - broad log-normal
        # Centered around 500 ka with wide uncertainty
        halflife = pm.LogNormal('halflife', mu=np.log(500), sigma=1.5)
        k = pm.Deterministic('k', np.log(2) / halflife)

        # Account for age uncertainty
        true_ages = pm.Normal('true_ages', mu=ages_mean, sigma=ages_std, shape=len(ages_mean))

        # Expected density under exponential decay
        mu = D0 * pm.math.exp(-k * true_ages)

        # Observation noise
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=10)
        total_sigma = pm.math.sqrt(density_se**2 + sigma_obs**2)

        # Likelihood
        D_obs = pm.Normal('D_obs', mu=mu, sigma=total_sigma, observed=densities)

        # Sample
        if verbose:
            print("\nSampling posterior...")

        trace = pm.sample(
            bayesian_params['n_samples'],
            tune=bayesian_params['n_tune'],
            chains=bayesian_params['n_chains'],
            target_accept=bayesian_params.get('target_accept', 0.95),
            return_inferencedata=True,
            progressbar=verbose,
            random_seed=42
        )

    # Extract posteriors
    halflife_samples = trace.posterior['halflife'].values.flatten()
    D0_samples = trace.posterior['D0'].values.flatten()
    k_samples = trace.posterior['k'].values.flatten()

    halflife_median = np.median(halflife_samples)
    halflife_mean = np.mean(halflife_samples)
    halflife_ci = np.percentile(halflife_samples, [2.5, 97.5])

    D0_mean = np.mean(D0_samples)
    D0_ci = np.percentile(D0_samples, [2.5, 97.5])

    k_mean = np.mean(k_samples)
    k_ci = np.percentile(k_samples, [2.5, 97.5])

    # Check convergence
    rhat_vals = az.rhat(trace)
    max_rhat = max(rhat_vals['halflife'].values.max(), rhat_vals['D0'].values.max())

    if verbose:
        print(f"\nResults:")
        print(f"  Half-life: {halflife_median:.0f} ka (95% CI: [{halflife_ci[0]:.0f}, {halflife_ci[1]:.0f}])")
        print(f"  D₀: {D0_mean:.1f} lakes/1000km² (95% CI: [{D0_ci[0]:.1f}, {D0_ci[1]:.1f}])")
        print(f"  k: {k_mean:.6f} per ka (95% CI: [{k_ci[0]:.6f}, {k_ci[1]:.6f}])")
        print(f"  Convergence (R-hat): {max_rhat:.3f} {'✓' if max_rhat < 1.05 else '⚠'}")

    # Get summary
    summary = az.summary(trace, var_names=['D0', 'k', 'halflife'])

    results = {
        'halflife_mean': halflife_mean,
        'halflife_median': halflife_median,
        'halflife_ci_low': halflife_ci[0],
        'halflife_ci_high': halflife_ci[1],
        'D0': {
            'mean': D0_mean,
            'ci_low': D0_ci[0],
            'ci_high': D0_ci[1],
            'samples': D0_samples
        },
        'k': {
            'mean': k_mean,
            'ci_low': k_ci[0],
            'ci_high': k_ci[1],
            'samples': k_samples
        },
        'halflife_samples': halflife_samples,
        'trace': trace,
        'summary': summary,
        'max_rhat': max_rhat,
        'stages': stages,
        'densities': densities,
        'ages': ages_mean
    }

    return results


def plot_overall_bayesian_halflife(results, output_dir=None, verbose=True):
    """
    Visualize overall Bayesian half-life results.

    Parameters
    ----------
    results : dict
        Output from fit_overall_bayesian_halflife()
    output_dir : str, optional
        Directory to save figure. If None, uses OUTPUT_DIR from config.
    verbose : bool
        Print progress messages

    Returns
    -------
    fig : matplotlib.Figure
        2-panel figure (decay curve + posteriors)
    """

    if results is None:
        print("No results to plot")
        return None

    if output_dir is None:
        from .config import OUTPUT_DIR, ensure_output_dir
        output_dir = ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Overall Bayesian Half-Life Analysis', fontsize=14, fontweight='bold')

    # Panel A: Decay curve with uncertainty
    ax = axes[0]

    ages_plot = np.linspace(1, max(results['ages']) * 1.5, 200)

    # Plot posterior samples (uncertainty band)
    D0_samples = results['D0']['samples']
    k_samples = results['k']['samples']

    for _ in range(50):
        idx = np.random.randint(len(D0_samples))
        ax.plot(ages_plot, D0_samples[idx] * np.exp(-k_samples[idx] * ages_plot),
                color='steelblue', alpha=0.05, linewidth=0.5)

    # Plot median fit
    D0_med = results['D0']['mean']
    k_med = results['k']['mean']
    ax.plot(ages_plot, D0_med * np.exp(-k_med * ages_plot),
            color='darkblue', linewidth=2, label='Median fit')

    # Plot data points
    ax.scatter(results['ages'], results['densities'],
              s=100, color='red', edgecolor='black', zorder=5,
              label='Observed')

    # Add half-life annotation
    halflife = results['halflife_median']
    halflife_ci = (results['halflife_ci_low'], results['halflife_ci_high'])
    ax.axvline(halflife, color='green', linestyle='--', linewidth=2,
              label=f't½ = {halflife:.0f} ka [{halflife_ci[0]:.0f}, {halflife_ci[1]:.0f}]')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Landscape Age (ka)', fontsize=11)
    ax.set_ylabel('Lake Density (per 1000 km²)', fontsize=11)
    ax.set_title('A) Exponential Decay Fit')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel B: Posterior distributions
    ax = axes[1]

    # Halflife posterior
    halflife_samples = results['halflife_samples']
    ax.hist(halflife_samples, bins=50, alpha=0.6, color='green',
           density=True, label='Half-life')
    ax.axvline(np.median(halflife_samples), color='darkgreen',
              linestyle='--', linewidth=2)

    ax.set_xlabel('Half-life (ka)', fontsize=11)
    ax.set_ylabel('Posterior Density', fontsize=11)
    ax.set_title('B) Half-Life Posterior Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig_path = os.path.join(output_dir, 'bayesian_overall_halflife.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nSaved: {fig_path}")

    return fig


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("This module is designed to be imported and used with existing lake data.")
    print("\nExample usage:")
    print("""
    from lake_analysis import load_conus_lake_data
    from lake_analysis.glacial_chronosequence import (
        load_wisconsin_extent, load_illinoian_extent, load_driftless_area,
        classify_lakes_by_glacial_extent
    )
    from lake_analysis.size_stratified_analysis import run_size_stratified_analysis

    # Load and classify lakes
    lakes = load_conus_lake_data()
    boundaries = {
        'wisconsin': load_wisconsin_extent(),
        'illinoian': load_illinoian_extent(),
        'driftless': load_driftless_area()
    }
    lakes = classify_lakes_by_glacial_extent(lakes, boundaries)

    # Run size-stratified analysis
    results = run_size_stratified_analysis(
        lakes,
        min_lake_area=0.05,
        min_lakes_per_class=10
    )
    """)
