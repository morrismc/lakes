"""
Power Law Analysis Module for Lake Distribution Analysis
==========================================================

This module implements Maximum Likelihood Estimation (MLE) for fitting
power law distributions to lake size data, following the methodology
of Clauset et al. (2009).

Key Features:
- MLE estimation of power law exponent (alpha)
- Automatic x_min selection using KS statistic
- Bootstrap confidence intervals
- Goodness-of-fit testing
- Comparison of power law vs alternative distributions

Reference:
    Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
    Power-law distributions in empirical data. SIAM Review, 51(4), 661-703.

Global Reference:
    Cael, B. B., & Seekell, D. A. (2016). The size-distribution of
    Earth's lakes. Scientific Reports, 6, 29633.
    - Found τ ≈ 2.14 for lakes ≥ 0.46 km²
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, brentq
from collections import Counter
import warnings

from .config import COLS, POWERLAW_XMIN_THRESHOLD, MIN_LAKES_FOR_POWERLAW, RANDOM_SEED


# ============================================================================
# CORE MLE ESTIMATION
# ============================================================================

def estimate_alpha_mle(data, xmin):
    """
    Estimate power law exponent using Maximum Likelihood.

    For data x_i >= x_min, the MLE estimator is:
        α = 1 + n / Σ ln(x_i / x_min)

    Parameters
    ----------
    data : array-like
        Observed values (lake areas)
    xmin : float
        Lower cutoff for power law

    Returns
    -------
    float
        MLE estimate of alpha
    """
    data = np.asarray(data)
    tail = data[data >= xmin]

    if len(tail) == 0:
        return np.nan

    n = len(tail)
    alpha = 1 + n / np.sum(np.log(tail / xmin))

    return alpha


def estimate_alpha_uncertainty(data, xmin, alpha, n_bootstrap=1000):
    """
    Estimate uncertainty in alpha using bootstrap resampling.

    Parameters
    ----------
    data : array-like
        Full dataset
    xmin : float
        Lower cutoff
    alpha : float
        Point estimate of alpha
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    dict
        Contains: se (standard error), ci_lower, ci_upper (95% CI)
    """
    np.random.seed(RANDOM_SEED)

    data = np.asarray(data)
    tail = data[data >= xmin]
    n = len(tail)

    if n < 10:
        return {'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

    bootstrap_alphas = []
    for _ in range(n_bootstrap):
        # Resample from tail
        boot_sample = np.random.choice(tail, size=n, replace=True)
        boot_alpha = estimate_alpha_mle(boot_sample, xmin)
        bootstrap_alphas.append(boot_alpha)

    bootstrap_alphas = np.array(bootstrap_alphas)
    bootstrap_alphas = bootstrap_alphas[np.isfinite(bootstrap_alphas)]

    return {
        'se': np.std(bootstrap_alphas),
        'ci_lower': np.percentile(bootstrap_alphas, 2.5),
        'ci_upper': np.percentile(bootstrap_alphas, 97.5),
    }


# ============================================================================
# X_MIN SELECTION
# ============================================================================

def compute_ks_statistic(data, xmin, alpha):
    """
    Compute Kolmogorov-Smirnov statistic comparing data to theoretical power law.

    Parameters
    ----------
    data : array-like
        Observed values
    xmin : float
        Lower cutoff
    alpha : float
        Power law exponent

    Returns
    -------
    float
        KS statistic (max absolute difference between empirical and theoretical CDFs)
    """
    data = np.asarray(data)
    tail = data[data >= xmin]

    if len(tail) == 0:
        return np.inf

    # Sort data
    sorted_tail = np.sort(tail)

    # Empirical CDF
    n = len(sorted_tail)
    empirical_cdf = np.arange(1, n + 1) / n

    # Theoretical CDF for power law: F(x) = 1 - (x/xmin)^(1-alpha)
    theoretical_cdf = 1 - (sorted_tail / xmin) ** (1 - alpha)

    # KS statistic
    ks = np.max(np.abs(empirical_cdf - theoretical_cdf))

    return ks


def estimate_xmin(data, xmin_candidates=None, method='ks'):
    """
    Estimate optimal x_min using Clauset et al. (2009) methodology.

    The optimal x_min minimizes the KS statistic between the data and
    the fitted power law.

    Parameters
    ----------
    data : array-like
        Observed values
    xmin_candidates : array-like, optional
        Candidate values for x_min. If None, uses unique values in data.
    method : str
        Method for selection: 'ks' (default) or 'likelihood_ratio'

    Returns
    -------
    dict
        Contains: xmin, alpha, ks_statistic, n_tail
    """
    data = np.asarray(data)
    data = data[data > 0]  # Remove zeros

    if xmin_candidates is None:
        # Use unique values in upper portion of data
        unique_vals = np.unique(data)
        # Only consider candidates that leave at least MIN_LAKES_FOR_POWERLAW points
        min_idx = max(0, len(data) - MIN_LAKES_FOR_POWERLAW)
        sorted_data = np.sort(data)
        min_xmin = sorted_data[min_idx] if min_idx < len(sorted_data) else sorted_data[-1]
        xmin_candidates = unique_vals[unique_vals >= min_xmin * 0.1]  # Include some below threshold

        # Limit number of candidates for efficiency
        if len(xmin_candidates) > 100:
            xmin_candidates = np.linspace(
                xmin_candidates.min(),
                xmin_candidates.max(),
                100
            )

    best_ks = np.inf
    best_xmin = None
    best_alpha = None

    for xmin in xmin_candidates:
        tail = data[data >= xmin]
        if len(tail) < 50:  # Require minimum data points
            continue

        alpha = estimate_alpha_mle(data, xmin)
        if np.isnan(alpha) or alpha <= 1:
            continue

        ks = compute_ks_statistic(data, xmin, alpha)

        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin
            best_alpha = alpha

    if best_xmin is None:
        return {
            'xmin': np.nan,
            'alpha': np.nan,
            'ks_statistic': np.nan,
            'n_tail': 0,
        }

    return {
        'xmin': best_xmin,
        'alpha': best_alpha,
        'ks_statistic': best_ks,
        'n_tail': np.sum(data >= best_xmin),
    }


# ============================================================================
# GOODNESS OF FIT TESTING
# ============================================================================

def bootstrap_pvalue(data, xmin, alpha, n_simulations=500, verbose=True):
    """
    Compute bootstrap p-value for power law fit.

    Method:
    1. Generate synthetic datasets from fitted power law
    2. Fit power law to each synthetic dataset
    3. Compute KS statistic for each
    4. p-value = fraction of synthetic KS >= observed KS

    Parameters
    ----------
    data : array-like
    xmin : float
    alpha : float
    n_simulations : int
    verbose : bool

    Returns
    -------
    dict
        Contains: p_value, observed_ks, synthetic_ks_values
    """
    np.random.seed(RANDOM_SEED)

    data = np.asarray(data)
    n_total = len(data)
    n_tail = np.sum(data >= xmin)
    n_below = n_total - n_tail

    # Observed KS
    observed_ks = compute_ks_statistic(data, xmin, alpha)

    synthetic_ks_values = []

    if verbose:
        print(f"Running bootstrap goodness-of-fit test ({n_simulations} simulations)...")

    for i in range(n_simulations):
        # Generate synthetic data
        # Below xmin: sample from empirical distribution
        below_xmin = data[data < xmin]
        if len(below_xmin) > 0 and n_below > 0:
            synth_below = np.random.choice(below_xmin, size=n_below, replace=True)
        else:
            synth_below = np.array([])

        # Above xmin: sample from power law
        # Inverse CDF method: x = xmin * u^(1/(1-alpha))
        u = np.random.uniform(0, 1, n_tail)
        synth_above = xmin * (1 - u) ** (1 / (1 - alpha))

        # Combine
        synth_data = np.concatenate([synth_below, synth_above])

        # Fit power law to synthetic data
        synth_result = estimate_xmin(synth_data)
        synth_ks = synth_result['ks_statistic']

        if np.isfinite(synth_ks):
            synthetic_ks_values.append(synth_ks)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations")

    synthetic_ks_values = np.array(synthetic_ks_values)

    # p-value: fraction of synthetic KS >= observed
    p_value = np.mean(synthetic_ks_values >= observed_ks)

    return {
        'p_value': p_value,
        'observed_ks': observed_ks,
        'synthetic_ks_mean': np.mean(synthetic_ks_values),
        'n_simulations': len(synthetic_ks_values),
    }


# ============================================================================
# ALTERNATIVE DISTRIBUTION COMPARISON
# ============================================================================

def compare_distributions(data, xmin):
    """
    Compare power law fit to alternative distributions.

    Alternatives considered:
    - Exponential
    - Lognormal
    - Truncated power law

    Uses likelihood ratio test.

    Parameters
    ----------
    data : array-like
    xmin : float

    Returns
    -------
    dict
        Comparison results for each alternative
    """
    data = np.asarray(data)
    tail = data[data >= xmin]
    n = len(tail)

    if n < 50:
        return {'error': 'Insufficient data for comparison'}

    results = {}

    # Fit power law
    alpha_pl = estimate_alpha_mle(tail, xmin)
    ll_pl = powerlaw_loglikelihood(tail, xmin, alpha_pl)

    # Fit exponential: f(x) = λ exp(-λ(x - xmin))
    lambda_exp = 1 / (np.mean(tail) - xmin) if np.mean(tail) > xmin else 1
    ll_exp = exponential_loglikelihood(tail, xmin, lambda_exp)

    # Likelihood ratio
    lr_exp = 2 * (ll_pl - ll_exp)
    # p-value using chi-squared (1 df)
    p_exp = 1 - stats.chi2.cdf(abs(lr_exp), 1)

    results['exponential'] = {
        'lambda': lambda_exp,
        'loglik': ll_exp,
        'likelihood_ratio': lr_exp,
        'p_value': p_exp,
        'favors': 'power_law' if lr_exp > 0 else 'exponential',
    }

    # Fit lognormal
    try:
        log_tail = np.log(tail)
        mu_ln = np.mean(log_tail)
        sigma_ln = np.std(log_tail)
        ll_ln = lognormal_loglikelihood(tail, mu_ln, sigma_ln)

        lr_ln = 2 * (ll_pl - ll_ln)
        p_ln = 1 - stats.chi2.cdf(abs(lr_ln), 1)

        results['lognormal'] = {
            'mu': mu_ln,
            'sigma': sigma_ln,
            'loglik': ll_ln,
            'likelihood_ratio': lr_ln,
            'p_value': p_ln,
            'favors': 'power_law' if lr_ln > 0 else 'lognormal',
        }
    except Exception as e:
        results['lognormal'] = {'error': str(e)}

    results['power_law'] = {
        'alpha': alpha_pl,
        'loglik': ll_pl,
    }

    return results


def powerlaw_loglikelihood(data, xmin, alpha):
    """Log-likelihood for power law distribution."""
    n = len(data)
    return n * np.log(alpha - 1) - n * np.log(xmin) - alpha * np.sum(np.log(data / xmin))


def exponential_loglikelihood(data, xmin, lam):
    """Log-likelihood for exponential distribution (shifted by xmin)."""
    return len(data) * np.log(lam) - lam * np.sum(data - xmin)


def lognormal_loglikelihood(data, mu, sigma):
    """Log-likelihood for lognormal distribution."""
    n = len(data)
    return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((np.log(data) - mu)**2) / (2 * sigma**2) - np.sum(np.log(data))


# ============================================================================
# DOMAIN-SPECIFIC ANALYSIS
# ============================================================================

def fit_powerlaw_by_domain(lake_df, domain_column,
                           area_column=None,
                           min_lakes=MIN_LAKES_FOR_POWERLAW,
                           fixed_xmin=None,
                           compute_uncertainty=True):
    """
    Fit power law to lake sizes within each domain.

    Parameters
    ----------
    lake_df : DataFrame
    domain_column : str
        Column defining domains (e.g., elevation bins, geographic regions)
    area_column : str
        Column with lake areas (default from config)
    min_lakes : int
        Minimum lakes required for fitting
    fixed_xmin : float, optional
        If provided, use this xmin for all domains (useful for comparison)
    compute_uncertainty : bool
        If True, compute bootstrap confidence intervals

    Returns
    -------
    DataFrame
        Power law parameters for each domain
    """
    if area_column is None:
        area_column = COLS['area']

    results = []

    for domain, group in lake_df.groupby(domain_column):
        areas = group[area_column].values
        areas = areas[areas > 0]

        result = {'domain': domain, 'n_total': len(areas)}

        if len(areas) < min_lakes:
            result['alpha'] = np.nan
            result['xmin'] = np.nan
            result['n_tail'] = 0
            result['ks_statistic'] = np.nan
            results.append(result)
            continue

        # Fit power law
        if fixed_xmin is not None:
            xmin = fixed_xmin
            alpha = estimate_alpha_mle(areas, xmin)
            ks = compute_ks_statistic(areas, xmin, alpha)
            n_tail = np.sum(areas >= xmin)
        else:
            fit = estimate_xmin(areas)
            xmin = fit['xmin']
            alpha = fit['alpha']
            ks = fit['ks_statistic']
            n_tail = fit['n_tail']

        result['xmin'] = xmin
        result['alpha'] = alpha
        result['ks_statistic'] = ks
        result['n_tail'] = n_tail

        # Confidence intervals
        if compute_uncertainty and np.isfinite(alpha):
            uncertainty = estimate_alpha_uncertainty(areas, xmin, alpha, n_bootstrap=500)
            result['alpha_se'] = uncertainty['se']
            result['alpha_ci_lower'] = uncertainty['ci_lower']
            result['alpha_ci_upper'] = uncertainty['ci_upper']

        results.append(result)

    return pd.DataFrame(results)


def fit_powerlaw_by_elevation_bands(lake_df, elev_breaks,
                                     elev_column=None,
                                     area_column=None):
    """
    Convenience function to fit power law in elevation bands.

    Parameters
    ----------
    lake_df : DataFrame
    elev_breaks : list
        Elevation bin edges (e.g., [0, 500, 1000, 1500, 2000, ...])

    Returns
    -------
    DataFrame
        Power law parameters for each elevation band
    """
    if elev_column is None:
        elev_column = COLS['elevation']
    if area_column is None:
        area_column = COLS['area']

    # Create elevation bands
    lake_df = lake_df.copy()
    lake_df['elev_band'] = pd.cut(
        lake_df[elev_column],
        bins=elev_breaks,
        include_lowest=True
    )

    return fit_powerlaw_by_domain(lake_df, 'elev_band', area_column)


# ============================================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================================

def full_powerlaw_analysis(lake_areas, xmin_threshold=POWERLAW_XMIN_THRESHOLD,
                           run_bootstrap=True, n_bootstrap_sims=500,
                           compare_alternatives=True):
    """
    Run complete power law analysis pipeline.

    Parameters
    ----------
    lake_areas : array-like
        Lake areas in km²
    xmin_threshold : float
        Minimum x_min to consider (from literature)
    run_bootstrap : bool
        Run bootstrap p-value test (slow but recommended)
    n_bootstrap_sims : int
        Number of bootstrap simulations
    compare_alternatives : bool
        Compare to alternative distributions

    Returns
    -------
    dict
        Comprehensive results including fit parameters, uncertainty, p-values
    """
    print("=" * 60)
    print("POWER LAW ANALYSIS")
    print("=" * 60)

    areas = np.asarray(lake_areas)
    areas = areas[areas > 0]
    print(f"\nData: {len(areas):,} lakes")
    print(f"Area range: {areas.min():.4f} to {areas.max():.2f} km²")

    # Step 1: Estimate xmin and alpha
    print("\n[1] Estimating x_min and α...")

    # Constrain xmin search to reasonable range
    xmin_candidates = np.unique(areas[areas >= xmin_threshold * 0.5])
    if len(xmin_candidates) > 100:
        xmin_candidates = np.linspace(xmin_candidates.min(), xmin_candidates.max(), 100)

    fit = estimate_xmin(areas, xmin_candidates)
    print(f"    x_min = {fit['xmin']:.4f} km²")
    print(f"    α = {fit['alpha']:.3f}")
    print(f"    n (tail) = {fit['n_tail']:,}")
    print(f"    KS statistic = {fit['ks_statistic']:.4f}")

    result = {
        'n_total': len(areas),
        'xmin': fit['xmin'],
        'alpha': fit['alpha'],
        'n_tail': fit['n_tail'],
        'ks_statistic': fit['ks_statistic'],
    }

    # Step 2: Bootstrap uncertainty
    if not np.isnan(fit['alpha']):
        print("\n[2] Estimating uncertainty (bootstrap)...")
        uncertainty = estimate_alpha_uncertainty(areas, fit['xmin'], fit['alpha'])
        print(f"    α = {fit['alpha']:.3f} ± {uncertainty['se']:.3f}")
        print(f"    95% CI: [{uncertainty['ci_lower']:.3f}, {uncertainty['ci_upper']:.3f}]")
        result.update(uncertainty)

    # Step 3: Goodness of fit
    if run_bootstrap:
        print("\n[3] Bootstrap goodness-of-fit test...")
        gof = bootstrap_pvalue(areas, fit['xmin'], fit['alpha'], n_bootstrap_sims)
        print(f"    p-value = {gof['p_value']:.3f}")
        if gof['p_value'] >= 0.1:
            print("    → Power law is plausible (p ≥ 0.1)")
        else:
            print("    → Power law may not be appropriate (p < 0.1)")
        result['p_value'] = gof['p_value']

    # Step 4: Compare to alternatives
    if compare_alternatives:
        print("\n[4] Comparing to alternative distributions...")
        comparison = compare_distributions(areas, fit['xmin'])

        for dist_name, dist_result in comparison.items():
            if dist_name == 'power_law':
                continue
            if 'error' in dist_result:
                print(f"    {dist_name}: Error - {dist_result['error']}")
            else:
                print(f"    vs {dist_name}: LR = {dist_result['likelihood_ratio']:.2f}, "
                      f"p = {dist_result['p_value']:.3f} → favors {dist_result['favors']}")

        result['comparison'] = comparison

    # Reference comparison
    print("\n[5] Comparison to literature:")
    print(f"    Cael & Seekell (2016): τ = 2.14 for lakes ≥ 0.46 km²")
    print(f"    This dataset: α = {fit['alpha']:.3f} for lakes ≥ {fit['xmin']:.4f} km²")

    if not np.isnan(fit['alpha']):
        diff = abs(fit['alpha'] - 2.14)
        if diff < 0.1:
            print(f"    → Very close to global estimate (Δ = {diff:.3f})")
        elif diff < 0.3:
            print(f"    → Moderately different from global estimate (Δ = {diff:.3f})")
        else:
            print(f"    → Substantially different from global estimate (Δ = {diff:.3f})")

    print("\n" + "=" * 60)

    return result


# ============================================================================
# SUMMARY AND REPORTING
# ============================================================================

def generate_powerlaw_report(results, output_path=None):
    """
    Generate formatted text report of power law analysis.

    Parameters
    ----------
    results : dict
        Output from full_powerlaw_analysis()
    output_path : str, optional
        If provided, save report to file
    """
    lines = [
        "POWER LAW ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Sample size: {results['n_total']:,} lakes",
        "",
        "FITTED PARAMETERS:",
        f"  x_min = {results['xmin']:.4f} km²",
        f"  α = {results['alpha']:.3f}",
        f"  95% CI: [{results.get('ci_lower', 'N/A'):.3f}, {results.get('ci_upper', 'N/A'):.3f}]" if results.get('ci_lower') else "",
        f"  n (tail) = {results['n_tail']:,}",
        "",
        "GOODNESS OF FIT:",
        f"  KS statistic = {results['ks_statistic']:.4f}",
        f"  Bootstrap p-value = {results.get('p_value', 'Not computed'):.3f}" if results.get('p_value') else "",
        "",
        "INTERPRETATION:",
    ]

    # Add interpretation
    if results.get('p_value', 1) >= 0.1:
        lines.append("  Power law distribution is plausible for this dataset.")
    else:
        lines.append("  Power law may not be the best fit for this dataset.")

    if 'comparison' in results:
        lines.append("")
        lines.append("COMPARISON TO ALTERNATIVES:")
        for dist_name, dist_result in results['comparison'].items():
            if dist_name != 'power_law' and 'favors' in dist_result:
                lines.append(f"  vs {dist_name}: favors {dist_result['favors']}")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    print("Power Law Analysis module loaded.")
    print("\nKey functions:")
    print("  - full_powerlaw_analysis(areas)")
    print("  - fit_powerlaw_by_domain(df, domain_col)")
    print("  - estimate_xmin(data)")
    print("  - compare_distributions(data, xmin)")
