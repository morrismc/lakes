# =============================================================================
# COMPLETE BAYESIAN HALF-LIFE INTEGRATION
# =============================================================================
# This file contains all the code needed to integrate Bayesian half-life
# analysis into the main workflow. Copy sections to appropriate files.
# =============================================================================

# =============================================================================
# PART 1: ADD TO lake_analysis/size_stratified_analysis.py
# =============================================================================

# Add this after the existing fit_size_stratified_halflife_models() function:

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
        stages = density_by_stage['stage'].values
        densities = density_by_stage['density_per_1000km2'].values
        n_lakes = density_by_stage.get('n_lakes', np.ones(len(stages)) * 1000).values

    # Get ages
    ages_mean = np.array([age_estimates[s]['mean'] for s in stages])
    ages_std = np.array([age_estimates[s]['std'] for s in stages])

    # Remove stages with NaN density
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
# PART 2: ADD TO lake_analysis/main.py
# =============================================================================

# Add this after analyze_aridity_effects():

def analyze_bayesian_halflife(
    lakes,
    run_overall=True,
    run_size_stratified=True,
    min_lake_area=0.05,
    max_lake_area=20000,
    min_lakes_per_class=10,
    save_figures=True,
    verbose=True
):
    """
    Run Bayesian half-life analysis (overall and/or size-stratified).

    This function can be run standalone or as part of run_full_analysis().
    It estimates lake half-lives across glacial chronosequence using Bayesian
    exponential decay models.

    Two modes:
    1. Overall: Fit single half-life to all lakes in each glacial stage
    2. Size-stratified: Fit separate half-life for each lake size class

    Parameters
    ----------
    lakes : DataFrame or GeoDataFrame
        Lake data. Must have 'glacial_stage' column (from classify_lakes_by_glacial_extent)
    run_overall : bool
        If True, run overall Bayesian half-life analysis
    run_size_stratified : bool
        If True, run size-stratified half-life analysis
    min_lake_area : float
        Minimum lake area (km²) for analysis
    max_lake_area : float
        Maximum lake area (km²) to exclude Great Lakes
    min_lakes_per_class : int
        Minimum lakes needed per size class for Bayesian analysis
    save_figures : bool
        Generate and save visualizations
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results with 'overall' and/or 'size_stratified' keys containing analysis results

    Examples
    --------
    # Run both analyses
    >>> results = analyze_bayesian_halflife(lakes)

    # Run only overall half-life
    >>> results = analyze_bayesian_halflife(lakes, run_size_stratified=False)

    # Run only size-stratified
    >>> results = analyze_bayesian_halflife(lakes, run_overall=False)
    """

    print("\n" + "=" * 70)
    print("BAYESIAN HALF-LIFE ANALYSIS")
    print("=" * 70)

    if not run_overall and not run_size_stratified:
        print("ERROR: Must run at least one analysis type (overall or size-stratified)")
        return None

    # Check for glacial_stage column
    if 'glacial_stage' not in lakes.columns:
        print("ERROR: 'glacial_stage' column not found in lakes dataframe")
        print("Run glacial chronosequence analysis first to classify lakes by stage")
        return None

    results = {}

    # Import required functions
    try:
        from .size_stratified_analysis import (
            fit_overall_bayesian_halflife,
            plot_overall_bayesian_halflife,
            run_size_stratified_analysis
        )
        from .glacial_chronosequence import compute_lake_density_by_glacial_stage
    except ImportError:
        from size_stratified_analysis import (
            fit_overall_bayesian_halflife,
            plot_overall_bayesian_halflife,
            run_size_stratified_analysis
        )
        from glacial_chronosequence import compute_lake_density_by_glacial_stage

    # ---------------------------------------------------------------------
    # OVERALL HALF-LIFE ANALYSIS
    # ---------------------------------------------------------------------
    if run_overall:
        print("\n" + "-" * 70)
        print("OVERALL BAYESIAN HALF-LIFE (All Lakes Per Stage)")
        print("-" * 70)

        try:
            # Compute density by stage
            if verbose:
                print("\nComputing lake density by glacial stage...")

            density_by_stage = compute_lake_density_by_glacial_stage(
                lakes,
                zone_areas=None,  # Will use default areas
                verbose=verbose
            )

            if density_by_stage is None or len(density_by_stage) < 2:
                print("  Insufficient data for overall half-life analysis")
                results['overall'] = None
            else:
                # Fit Bayesian model
                overall_results = fit_overall_bayesian_halflife(
                    density_by_stage,
                    verbose=verbose
                )

                results['overall'] = overall_results

                # Generate visualization
                if save_figures and overall_results is not None:
                    fig = plot_overall_bayesian_halflife(
                        overall_results,
                        verbose=verbose
                    )
                    if fig:
                        plt.close(fig)

        except Exception as e:
            print(f"\nERROR in overall half-life analysis: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            results['overall'] = {'error': str(e)}

    # ---------------------------------------------------------------------
    # SIZE-STRATIFIED HALF-LIFE ANALYSIS
    # ---------------------------------------------------------------------
    if run_size_stratified:
        print("\n" + "-" * 70)
        print("SIZE-STRATIFIED BAYESIAN HALF-LIFE")
        print("-" * 70)

        try:
            # Run size-stratified analysis
            size_results = run_size_stratified_analysis(
                lakes,
                min_lake_area=min_lake_area,
                max_lake_area=max_lake_area,
                min_lakes_per_class=min_lakes_per_class,
                verbose=verbose
            )

            results['size_stratified'] = size_results

        except Exception as e:
            print(f"\nERROR in size-stratified analysis: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            results['size_stratified'] = {'error': str(e)}

    # ---------------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BAYESIAN HALF-LIFE ANALYSIS COMPLETE")
    print("=" * 70)

    if results.get('overall') and results['overall'] is not None and 'error' not in results['overall']:
        overall = results['overall']
        print(f"\nOverall Half-Life:")
        print(f"  t½ = {overall['halflife_median']:.0f} ka")
        print(f"  95% CI: [{overall['halflife_ci_low']:.0f}, {overall['halflife_ci_high']:.0f}] ka")

    if results.get('size_stratified') and results['size_stratified'].get('halflife_df') is not None:
        print(f"\nSize-Stratified Half-Lives:")
        halflife_df = results['size_stratified']['halflife_df']
        print(f"  Analyzed {len(halflife_df)} size classes")

        if results['size_stratified'].get('statistics'):
            stats = results['size_stratified']['statistics']
            if stats.get('spearman_p', 1) < 0.05:
                if stats['spearman_rho'] > 0:
                    print(f"  ✓ Larger lakes persist longer (ρ={stats['spearman_rho']:.3f}, p={stats['spearman_p']:.4f})")
                else:
                    print(f"  ✗ Smaller lakes persist longer (ρ={stats['spearman_rho']:.3f}, p={stats['spearman_p']:.4f})")
            else:
                print(f"  - No significant size-halflife relationship (p={stats['spearman_p']:.4f})")

    return results


# =============================================================================
# PART 3: MODIFY run_full_analysis() in lake_analysis/main.py
# =============================================================================

# In the function signature, add:
# include_bayesian_halflife=True,

# In the total_steps calculation, add:
# if include_bayesian_halflife:
#     total_steps += 1

# After the glacial chronosequence step (around line 2598), add:

"""
    # Step N: Bayesian half-life analysis (if enabled)
    if include_bayesian_halflife and results.get('glacial_chronosequence'):
        step += 1
        print_step_header(step, total_steps, "Bayesian Half-Life Analysis")
        with timed_step(timer, "Bayesian half-life analysis"):
            # Check if glacial analysis succeeded
            glacial = results['glacial_chronosequence']
            if glacial and 'lake_gdf' in glacial:
                results['bayesian_halflife'] = analyze_bayesian_halflife(
                    glacial['lake_gdf'],
                    run_overall=True,
                    run_size_stratified=True,
                    verbose=True
                )
            else:
                print("  Skipping: Glacial chronosequence analysis required")
                results['bayesian_halflife'] = {'error': 'glacial_analysis_required'}
"""

# In print_analysis_summary(), add this section after glacial chronosequence:

"""
    # 4. BAYESIAN HALF-LIFE RESULTS
    bayesian = results.get('bayesian_halflife', {})
    if bayesian and 'error' not in bayesian:
        print("\n┌" + "─" * 78 + "┐")
        print("│ 4. BAYESIAN HALF-LIFE ANALYSIS" + " " * 47 + "│")
        print("└" + "─" * 78 + "┘")

        if bayesian.get('overall'):
            overall = bayesian['overall']
            halflife = overall.get('halflife_median')
            if halflife:
                ci_low = overall.get('halflife_ci_low')
                ci_high = overall.get('halflife_ci_high')
                print(f"\n  Overall Half-Life:")
                print(f"    t½ = {halflife:.0f} ka (95% CI: [{ci_low:.0f}, {ci_high:.0f}])")
                print(f"    (Time for lake density to decrease by 50%)")

        if bayesian.get('size_stratified') and bayesian['size_stratified'].get('halflife_df') is not None:
            halflife_df = bayesian['size_stratified']['halflife_df']
            print(f"\n  Size-Stratified Half-Lives:")
            print(f"    Analyzed {len(halflife_df)} size classes")

            if bayesian['size_stratified'].get('statistics'):
                stats = bayesian['size_stratified']['statistics']
                print(f"    Spearman ρ = {stats.get('spearman_rho', 0):.3f}")
                print(f"    p-value = {stats.get('spearman_p', 1):.4f}")
"""

# =============================================================================
# PART 4: ADD TO lake_analysis/config.py
# =============================================================================

# Add after SIZE_STRATIFIED_STAGE_COLORS:

"""
# ============================================================================
# BAYESIAN HALF-LIFE ANALYSIS DEFAULTS
# ============================================================================

# Default parameters for Bayesian half-life analysis
BAYESIAN_HALFLIFE_DEFAULTS = {
    'run_overall': True,           # Run overall half-life analysis
    'run_size_stratified': True,   # Run size-stratified analysis
    'min_lake_area': 0.05,         # Minimum lake size (km²)
    'max_lake_area': 20000,        # Maximum lake size (km²) - excludes Great Lakes
    'min_lakes_per_class': 10,     # Min lakes per size class for Bayesian fit
    'n_samples': 2000,             # PyMC samples per chain
    'n_tune': 1000,                # PyMC tuning samples
    'n_chains': 4,                 # PyMC MCMC chains
    'target_accept': 0.95          # PyMC target acceptance rate
}

# Glacial stages for half-life analysis (supports future expansion)
GLACIAL_STAGES_CONFIG = {
    'Wisconsin': {
        'age_mean_ka': 20,
        'age_std_ka': 5,
        'boundary_key': 'wisconsin',
        'required': True,
        'description': 'Most recent glaciation (~15-25 ka)'
    },
    'Illinoian': {
        'age_mean_ka': 160,
        'age_std_ka': 30,
        'boundary_key': 'illinoian',
        'required': True,
        'description': 'Older glaciation (~130-190 ka)'
    },
    'Pre-Illinoian': {
        'age_mean_ka': 500,
        'age_std_ka': 100,
        'boundary_key': 'pre_illinoian',
        'required': False,  # Not yet available - placeholder for future
        'description': 'Pre-Illinoian glaciation (>~500 ka)'
    },
    'Driftless': {
        'age_mean_ka': 1500,
        'age_std_ka': 500,
        'boundary_key': 'driftless',
        'required': True,
        'description': 'Never glaciated (>1.5 Ma)'
    }
}
"""

# =============================================================================
# PART 5: UPDATE lake_analysis/__init__.py
# =============================================================================

# Add to exports (after size_stratified_analysis imports):

"""
# Bayesian half-life analysis (overall)
from .size_stratified_analysis import (
    fit_overall_bayesian_halflife,
    plot_overall_bayesian_halflife
)

# Bayesian half-life analysis (standalone function)
from .main import analyze_bayesian_halflife

# Bayesian configuration
from .config import (
    BAYESIAN_HALFLIFE_DEFAULTS,
    GLACIAL_STAGES_CONFIG
)
"""

# =============================================================================
# END OF IMPLEMENTATION
# =============================================================================
