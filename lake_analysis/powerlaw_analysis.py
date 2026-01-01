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

# Handle imports for both package and direct execution
try:
    from .config import COLS, POWERLAW_XMIN_THRESHOLD, MIN_LAKES_FOR_POWERLAW, RANDOM_SEED
except ImportError:
    from config import COLS, POWERLAW_XMIN_THRESHOLD, MIN_LAKES_FOR_POWERLAW, RANDOM_SEED


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
# BAYESIAN POWER LAW ESTIMATION (for small samples)
# ============================================================================

def bayesian_powerlaw_estimate(data, xmin, prior_mean=2.1, prior_sd=0.3,
                                n_samples=5000, verbose=False):
    """
    Bayesian estimation of power law exponent using conjugate prior.

    For small samples (n < 100), MLE estimates can be unreliable. This
    function uses a weakly informative prior based on theoretical expectations
    (percolation theory τ ≈ 2.05, global estimate τ ≈ 2.14) to provide
    more stable estimates with proper uncertainty quantification.

    The likelihood for power law data is:
        L(α | x) ∝ ∏ (α-1) * xmin^(α-1) * x_i^(-α)

    With a Normal prior on α:
        π(α) ~ N(prior_mean, prior_sd²)

    Parameters
    ----------
    data : array-like
        Observed lake areas
    xmin : float
        Lower cutoff for power law
    prior_mean : float
        Prior mean for α (default 2.1 based on theory)
    prior_sd : float
        Prior standard deviation (default 0.3 for weak informativeness)
    n_samples : int
        Number of MCMC samples (uses simple Metropolis-Hastings)
    verbose : bool
        Print sampling progress

    Returns
    -------
    dict
        Contains:
        - alpha_posterior_mean : Posterior mean of α
        - alpha_posterior_median : Posterior median of α
        - alpha_credible_interval : 95% credible interval (tuple)
        - alpha_samples : Posterior samples
        - effective_sample_size : Approximate ESS
        - prior_influence : Ratio of prior to likelihood influence
        - method : 'bayesian'
        - n_tail : Number of observations used
        - warning : Any warnings about sample size
    """
    data = np.asarray(data)
    tail = data[data >= xmin]
    n = len(tail)

    result = {
        'method': 'bayesian',
        'n_tail': n,
        'xmin': xmin,
        'prior_mean': prior_mean,
        'prior_sd': prior_sd,
        'warning': None
    }

    if n < 5:
        result['warning'] = f"Very small sample (n={n}). Results dominated by prior."
        result['alpha_posterior_mean'] = prior_mean
        result['alpha_posterior_median'] = prior_mean
        result['alpha_credible_interval'] = (prior_mean - 2*prior_sd, prior_mean + 2*prior_sd)
        result['alpha_samples'] = np.array([prior_mean])
        result['effective_sample_size'] = 0
        result['prior_influence'] = 1.0
        return result

    # Log-likelihood function for power law
    log_data_sum = np.sum(np.log(tail / xmin))

    def log_likelihood(alpha):
        if alpha <= 1:
            return -np.inf
        return n * np.log(alpha - 1) - alpha * log_data_sum

    def log_prior(alpha):
        return -0.5 * ((alpha - prior_mean) / prior_sd) ** 2

    def log_posterior(alpha):
        return log_likelihood(alpha) + log_prior(alpha)

    # MLE as starting point
    alpha_mle = 1 + n / log_data_sum if log_data_sum > 0 else prior_mean

    # Simple Metropolis-Hastings sampling
    np.random.seed(RANDOM_SEED)
    samples = []
    current_alpha = max(alpha_mle, 1.1)  # Ensure valid starting point
    proposal_sd = 0.1  # Proposal standard deviation

    n_accept = 0
    for i in range(n_samples + 1000):  # Include burn-in
        # Propose new alpha
        proposed_alpha = current_alpha + np.random.normal(0, proposal_sd)

        if proposed_alpha > 1:  # Valid alpha must be > 1
            # Accept/reject
            log_ratio = log_posterior(proposed_alpha) - log_posterior(current_alpha)
            if np.log(np.random.random()) < log_ratio:
                current_alpha = proposed_alpha
                if i >= 1000:  # After burn-in
                    n_accept += 1

        if i >= 1000:  # After burn-in
            samples.append(current_alpha)

    samples = np.array(samples)

    # Compute posterior statistics
    result['alpha_posterior_mean'] = np.mean(samples)
    result['alpha_posterior_median'] = np.median(samples)
    result['alpha_credible_interval'] = (np.percentile(samples, 2.5),
                                          np.percentile(samples, 97.5))
    result['alpha_samples'] = samples
    result['acceptance_rate'] = n_accept / n_samples

    # Estimate effective sample size (simple approximation)
    # Using lag-1 autocorrelation
    if len(samples) > 10:
        lag1_corr = np.corrcoef(samples[:-1], samples[1:])[0, 1]
        if np.isfinite(lag1_corr) and lag1_corr < 1:
            result['effective_sample_size'] = int(len(samples) * (1 - lag1_corr) / (1 + lag1_corr))
        else:
            result['effective_sample_size'] = len(samples)
    else:
        result['effective_sample_size'] = len(samples)

    # Estimate prior influence (shrinkage toward prior)
    if alpha_mle > 1:
        posterior_mean = result['alpha_posterior_mean']
        # How much did the posterior move from MLE toward prior?
        shrinkage = abs(posterior_mean - alpha_mle) / abs(prior_mean - alpha_mle) if prior_mean != alpha_mle else 0
        result['prior_influence'] = min(shrinkage, 1.0)
    else:
        result['prior_influence'] = 1.0

    # Add warning for small samples
    if n < 30:
        result['warning'] = f"Small sample (n={n}). Posterior influenced by prior (shrinkage={result['prior_influence']:.1%})."
    elif n < 100:
        result['warning'] = f"Moderate sample (n={n}). Some prior influence ({result['prior_influence']:.1%} shrinkage)."

    if verbose:
        print(f"  Bayesian power law: α = {result['alpha_posterior_mean']:.3f} "
              f"[{result['alpha_credible_interval'][0]:.3f}, {result['alpha_credible_interval'][1]:.3f}]")
        print(f"  n = {n}, acceptance rate = {result['acceptance_rate']:.1%}")
        if result['warning']:
            print(f"  Warning: {result['warning']}")

    return result


def adaptive_powerlaw_estimate(data, xmin, min_n_for_mle=100, verbose=False):
    """
    Adaptively choose between MLE and Bayesian estimation based on sample size.

    For large samples (n >= min_n_for_mle): Use standard MLE with bootstrap CI
    For small samples (n < min_n_for_mle): Use Bayesian with informative prior

    Parameters
    ----------
    data : array-like
        Observed lake areas
    xmin : float
        Lower cutoff for power law
    min_n_for_mle : int
        Minimum sample size for reliable MLE (default 100)
    verbose : bool
        Print method selection and results

    Returns
    -------
    dict
        Results with 'method' key indicating which approach was used
    """
    data = np.asarray(data)
    tail = data[data >= xmin]
    n = len(tail)

    if verbose:
        print(f"  Sample size: n = {n} (x_min = {xmin:.4f})")

    if n >= min_n_for_mle:
        # Use MLE for large samples
        if verbose:
            print(f"  Using MLE (n >= {min_n_for_mle})")

        alpha = estimate_alpha_mle(data, xmin)
        uncertainty = estimate_alpha_uncertainty(data, xmin, alpha, n_bootstrap=1000)

        return {
            'method': 'MLE',
            'alpha': alpha,
            'alpha_se': uncertainty['se'],
            'alpha_ci': (uncertainty['ci_lower'], uncertainty['ci_upper']),
            'n_tail': n,
            'xmin': xmin,
            'warning': None
        }
    else:
        # Use Bayesian for small samples
        if verbose:
            print(f"  Using Bayesian estimation (n < {min_n_for_mle})")

        bayes_result = bayesian_powerlaw_estimate(data, xmin, verbose=verbose)

        return {
            'method': 'Bayesian',
            'alpha': bayes_result['alpha_posterior_mean'],
            'alpha_median': bayes_result['alpha_posterior_median'],
            'alpha_se': np.std(bayes_result['alpha_samples']),
            'alpha_ci': bayes_result['alpha_credible_interval'],
            'n_tail': n,
            'xmin': xmin,
            'prior_influence': bayes_result['prior_influence'],
            'warning': bayes_result['warning']
        }


def compute_sample_size_power(n, alpha_true=2.1, alpha_diff=0.1, xmin=0.1):
    """
    Estimate statistical power to detect departure from theoretical alpha.

    Useful for understanding whether small samples can meaningfully test
    hypotheses about power law exponents.

    Parameters
    ----------
    n : int
        Sample size in power law tail
    alpha_true : float
        True alpha value
    alpha_diff : float
        Minimum detectable difference
    xmin : float
        Lower cutoff

    Returns
    -------
    dict
        Power analysis results
    """
    # MLE standard error approximation: SE(α) ≈ (α-1) / √n
    se_alpha = (alpha_true - 1) / np.sqrt(n)

    # Power to detect alpha_diff at α = 0.05
    z_crit = 1.96
    z_power = (alpha_diff / se_alpha) - z_crit

    # Convert to power (probability)
    from scipy.stats import norm
    power = norm.cdf(z_power)

    # Minimum detectable effect at 80% power
    mde_80 = (z_crit + 0.84) * se_alpha

    return {
        'n': n,
        'se_alpha': se_alpha,
        'power_to_detect': power,
        'target_difference': alpha_diff,
        'mde_80_power': mde_80,
        'ci_width_95': 2 * z_crit * se_alpha,
        'recommendation': 'adequate' if power >= 0.8 else 'underpowered' if power >= 0.5 else 'severely_underpowered'
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
        'synthetic_ks_values': synthetic_ks_values,  # Full array for visualization
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
        result['gof_results'] = gof  # Store full GOF results for visualization

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


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def xmin_sensitivity_analysis(lake_areas, xmin_values=None, compute_uncertainty=True,
                               n_bootstrap=500):
    """
    Analyze sensitivity of power law parameters to x_min threshold.

    This is critical for understanding how the choice of minimum area
    affects conclusions about the power law distribution.

    Parameters
    ----------
    lake_areas : array-like
        Lake areas in km²
    xmin_values : array-like, optional
        Values of x_min to test. If None, uses logarithmically spaced values.
    compute_uncertainty : bool
        If True, compute bootstrap confidence intervals for each x_min
    n_bootstrap : int
        Number of bootstrap samples for uncertainty estimation

    Returns
    -------
    DataFrame
        Results for each x_min value including alpha, n_tail, ks_statistic
    """
    np.random.seed(RANDOM_SEED)

    areas = np.asarray(lake_areas)
    areas = areas[areas > 0]

    if xmin_values is None:
        # Use logarithmically spaced values from minimum to reasonable upper bound
        min_area = max(0.01, np.percentile(areas, 1))
        max_area = np.percentile(areas, 50)  # Don't go too high
        xmin_values = np.logspace(np.log10(min_area), np.log10(max_area), 20)

    results = []

    print(f"Running sensitivity analysis for {len(xmin_values)} x_min values...")

    for i, xmin in enumerate(xmin_values):
        if (i + 1) % 5 == 0:
            print(f"  Processing {i + 1}/{len(xmin_values)}...")

        tail = areas[areas >= xmin]
        n_tail = len(tail)

        if n_tail < MIN_LAKES_FOR_POWERLAW:
            results.append({
                'xmin': xmin,
                'alpha': np.nan,
                'alpha_se': np.nan,
                'alpha_ci_lower': np.nan,
                'alpha_ci_upper': np.nan,
                'n_tail': n_tail,
                'ks_statistic': np.nan,
                'pct_data': n_tail / len(areas) * 100,
            })
            continue

        # Estimate alpha
        alpha = estimate_alpha_mle(areas, xmin)

        # KS statistic
        ks = compute_ks_statistic(areas, xmin, alpha)

        result = {
            'xmin': xmin,
            'alpha': alpha,
            'n_tail': n_tail,
            'ks_statistic': ks,
            'pct_data': n_tail / len(areas) * 100,
        }

        # Bootstrap uncertainty
        if compute_uncertainty and np.isfinite(alpha):
            uncertainty = estimate_alpha_uncertainty(areas, xmin, alpha, n_bootstrap)
            result['alpha_se'] = uncertainty['se']
            result['alpha_ci_lower'] = uncertainty['ci_lower']
            result['alpha_ci_upper'] = uncertainty['ci_upper']

        results.append(result)

    print("  Sensitivity analysis complete!")

    return pd.DataFrame(results)


def compare_to_cael_seekell(lake_areas, xmin=0.46):
    """
    Compare CONUS lake size distribution to Cael & Seekell (2016) global result.

    Cael & Seekell found τ = 2.14 ± 0.01 for global lakes ≥ 0.46 km².

    Parameters
    ----------
    lake_areas : array-like
        Lake areas in km²
    xmin : float
        Minimum area threshold (default 0.46 km² from C&S)

    Returns
    -------
    dict
        Comparison results including z-test for difference from global value
    """
    areas = np.asarray(lake_areas)
    areas = areas[areas > 0]

    # Fit to CONUS data
    alpha_conus = estimate_alpha_mle(areas, xmin)
    n_tail = np.sum(areas >= xmin)

    # Bootstrap uncertainty
    uncertainty = estimate_alpha_uncertainty(areas, xmin, alpha_conus)

    # Global values from Cael & Seekell (2016)
    alpha_global = 2.14
    se_global = 0.01

    # Z-test for difference
    if uncertainty['se'] > 0:
        z_stat = (alpha_conus - alpha_global) / np.sqrt(uncertainty['se']**2 + se_global**2)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = np.nan
        p_value = np.nan

    result = {
        'alpha_conus': alpha_conus,
        'alpha_se_conus': uncertainty['se'],
        'alpha_ci_lower': uncertainty['ci_lower'],
        'alpha_ci_upper': uncertainty['ci_upper'],
        'n_tail_conus': n_tail,
        'alpha_global': alpha_global,
        'se_global': se_global,
        'difference': alpha_conus - alpha_global,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant_at_005': p_value < 0.05 if pd.notna(p_value) else False,
    }

    # Interpretation
    if pd.notna(p_value):
        if p_value < 0.05:
            if alpha_conus > alpha_global:
                result['interpretation'] = ("CONUS has significantly FEWER large lakes "
                                            "than global average (steeper distribution)")
            else:
                result['interpretation'] = ("CONUS has significantly MORE large lakes "
                                            "than global average (shallower distribution)")
        else:
            result['interpretation'] = "CONUS distribution is consistent with global average"

    return result


def fit_powerlaw_by_process_domain(lake_df, elev_col=None, area_col=None,
                                    fixed_xmin=None, compute_uncertainty=True):
    """
    Fit power law to lakes classified by geomorphic process domain.

    Uses ELEVATION_DOMAINS from config to classify lakes.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with elevation and area columns
    elev_col, area_col : str
        Column names (defaults from config)
    fixed_xmin : float, optional
        If provided, use same x_min for all domains (enables fair comparison)
    compute_uncertainty : bool
        If True, compute bootstrap confidence intervals

    Returns
    -------
    DataFrame
        Power law parameters for each process domain
    """
    if elev_col is None:
        elev_col = COLS['elevation']
    if area_col is None:
        area_col = COLS['area']

    # Import ELEVATION_DOMAINS here to avoid circular import
    try:
        from .config import ELEVATION_DOMAINS
    except ImportError:
        from config import ELEVATION_DOMAINS

    # Classify lakes into domains
    lake_df = lake_df.copy()

    def classify_domain(elev):
        for domain_name, (low, high) in ELEVATION_DOMAINS.items():
            if low <= elev < high:
                return domain_name
        return 'unclassified'

    lake_df['process_domain'] = lake_df[elev_col].apply(classify_domain)

    # Fit power law for each domain
    return fit_powerlaw_by_domain(
        lake_df, 'process_domain',
        area_column=area_col,
        fixed_xmin=fixed_xmin,
        compute_uncertainty=compute_uncertainty
    )


if __name__ == "__main__":
    print("Power Law Analysis module loaded.")
    print("\nKey functions:")
    print("  - full_powerlaw_analysis(areas)")
    print("  - fit_powerlaw_by_domain(df, domain_col)")
    print("  - fit_powerlaw_by_process_domain(df)      # NEW")
    print("  - xmin_sensitivity_analysis(areas)        # NEW")
    print("  - compare_to_cael_seekell(areas)          # NEW")
    print("  - xmin_sensitivity_by_elevation(df)       # NEW - Full elevation analysis")


# ============================================================================
# COMPREHENSIVE X_MIN SENSITIVITY BY ELEVATION
# ============================================================================

def compute_powerlaw_metrics(lake_areas, xmin):
    """
    For a given dataset and x_min, compute comprehensive power law metrics.

    Parameters
    ----------
    lake_areas : array-like
        Lake areas in km²
    xmin : float
        Lower cutoff for power law

    Returns
    -------
    dict
        Contains: alpha, alpha_se, ks_statistic, n_tail, n_total, pct_tail
    """
    areas = np.asarray(lake_areas)
    areas = areas[areas > 0]
    n_total = len(areas)
    tail = areas[areas >= xmin]
    n_tail = len(tail)

    if n_tail < 10:
        return {
            'alpha': np.nan,
            'alpha_se': np.nan,
            'ks_statistic': np.nan,
            'n_tail': n_tail,
            'n_total': n_total,
            'pct_tail': n_tail / n_total * 100 if n_total > 0 else 0,
        }

    # MLE alpha
    alpha = estimate_alpha_mle(areas, xmin)

    # Standard error (analytical approximation)
    alpha_se = (alpha - 1) / np.sqrt(n_tail)

    # KS statistic
    ks = compute_ks_statistic(areas, xmin, alpha)

    return {
        'alpha': alpha,
        'alpha_se': alpha_se,
        'ks_statistic': ks,
        'n_tail': n_tail,
        'n_total': n_total,
        'pct_tail': n_tail / n_total * 100,
    }


def find_optimal_xmin_detailed(lake_areas, xmin_candidates, ks_tolerance=0.01):
    """
    Sweep through x_min candidates, compute KS for each.
    Return optimal x_min and the "plateau range" where KS is near minimum.

    Parameters
    ----------
    lake_areas : array-like
        Lake areas
    xmin_candidates : array-like
        x_min values to test
    ks_tolerance : float
        Consider x_min values within this of minimum KS as acceptable

    Returns
    -------
    dict
        optimal_xmin, optimal_alpha, optimal_ks, acceptable_range, full_results
    """
    areas = np.asarray(lake_areas)
    areas = areas[areas > 0]

    results = []
    for xmin in xmin_candidates:
        metrics = compute_powerlaw_metrics(areas, xmin)
        metrics['xmin'] = xmin
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Find optimal (minimum KS)
    valid = results_df.dropna(subset=['ks_statistic'])
    if len(valid) == 0:
        return {
            'optimal_xmin': np.nan,
            'optimal_alpha': np.nan,
            'optimal_ks': np.nan,
            'acceptable_range': (np.nan, np.nan),
            'full_results': results_df,
        }

    min_ks_idx = valid['ks_statistic'].idxmin()
    optimal_xmin = valid.loc[min_ks_idx, 'xmin']
    optimal_alpha = valid.loc[min_ks_idx, 'alpha']
    optimal_ks = valid.loc[min_ks_idx, 'ks_statistic']

    # Find acceptable range (within tolerance of minimum)
    acceptable_mask = valid['ks_statistic'] <= (optimal_ks + ks_tolerance)
    acceptable = valid[acceptable_mask]
    if len(acceptable) > 0:
        acceptable_range = (acceptable['xmin'].min(), acceptable['xmin'].max())
    else:
        acceptable_range = (optimal_xmin, optimal_xmin)

    return {
        'optimal_xmin': optimal_xmin,
        'optimal_alpha': optimal_alpha,
        'optimal_ks': optimal_ks,
        'acceptable_range': acceptable_range,
        'full_results': results_df,
    }


def analyze_elevation_band_xmin(lakes_df, elev_min, elev_max, xmin_candidates,
                                 elev_col=None, area_col=None, ks_tolerance=0.01,
                                 fixed_xmin_values=None):
    """
    Comprehensive x_min sensitivity analysis for a single elevation band.

    Parameters
    ----------
    lakes_df : DataFrame
        Lake data
    elev_min, elev_max : float
        Elevation range
    xmin_candidates : array-like
        x_min values to test
    elev_col, area_col : str
        Column names
    ks_tolerance : float
        Tolerance for acceptable x_min range
    fixed_xmin_values : list, optional
        Additional fixed x_min values to compute alpha at (e.g., [0.5, 1.0, 2.0])

    Returns
    -------
    dict
        Comprehensive results for this elevation band
    """
    if elev_col is None:
        elev_col = COLS['elevation']
    if area_col is None:
        area_col = COLS['area']

    if fixed_xmin_values is None:
        fixed_xmin_values = [0.1, 0.5, 1.0, 2.0]

    # Filter to elevation band
    mask = (lakes_df[elev_col] >= elev_min) & (lakes_df[elev_col] < elev_max)
    band_lakes = lakes_df.loc[mask, area_col].values
    band_lakes = band_lakes[band_lakes > 0]

    n_total = len(band_lakes)

    if n_total < MIN_LAKES_FOR_POWERLAW:
        return {
            'elev_min': elev_min,
            'elev_max': elev_max,
            'elev_mid': (elev_min + elev_max) / 2,
            'n_total': n_total,
            'optimal_xmin': np.nan,
            'optimal_alpha': np.nan,
            'optimal_ks': np.nan,
            'acceptable_range': (np.nan, np.nan),
            'alpha_at_fixed_xmin': {},
            'full_results': pd.DataFrame(),
            'status': 'insufficient_data',
        }

    # Run detailed x_min analysis
    xmin_analysis = find_optimal_xmin_detailed(band_lakes, xmin_candidates, ks_tolerance)

    # Compute alpha at fixed x_min values for comparison
    alpha_at_fixed = {}
    for fixed_xmin in fixed_xmin_values:
        metrics = compute_powerlaw_metrics(band_lakes, fixed_xmin)
        alpha_at_fixed[fixed_xmin] = {
            'alpha': metrics['alpha'],
            'alpha_se': metrics['alpha_se'],
            'n_tail': metrics['n_tail'],
            'ks_statistic': metrics['ks_statistic'],
        }

    # Test stability within acceptable range
    full_results = xmin_analysis['full_results']
    acc_range = xmin_analysis['acceptable_range']

    if pd.notna(acc_range[0]) and pd.notna(acc_range[1]):
        in_range = full_results[
            (full_results['xmin'] >= acc_range[0]) &
            (full_results['xmin'] <= acc_range[1])
        ]
        if len(in_range) > 0:
            alpha_range = (in_range['alpha'].min(), in_range['alpha'].max())
            alpha_stability = alpha_range[1] - alpha_range[0]
        else:
            alpha_range = (np.nan, np.nan)
            alpha_stability = np.nan
    else:
        alpha_range = (np.nan, np.nan)
        alpha_stability = np.nan

    return {
        'elev_min': elev_min,
        'elev_max': elev_max,
        'elev_mid': (elev_min + elev_max) / 2,
        'n_total': n_total,
        'optimal_xmin': xmin_analysis['optimal_xmin'],
        'optimal_alpha': xmin_analysis['optimal_alpha'],
        'optimal_ks': xmin_analysis['optimal_ks'],
        'acceptable_range': acc_range,
        'alpha_range_in_acceptable': alpha_range,
        'alpha_stability': alpha_stability,
        'alpha_at_fixed_xmin': alpha_at_fixed,
        'full_results': full_results,
        'status': 'success',
    }


def xmin_sensitivity_by_elevation(lakes_df, elevation_bands=None, xmin_candidates=None,
                                   elev_col=None, area_col=None, ks_tolerance=0.01,
                                   fixed_xmin_values=None, show_progress=True):
    """
    Run comprehensive x_min sensitivity analysis across elevation bands.

    This analysis addresses key questions:
    1. Does optimal x_min differ by elevation?
    2. How sensitive is alpha to x_min choice in each band?
    3. Are alpha differences robust or threshold-dependent?

    Parameters
    ----------
    lakes_df : DataFrame
        Lake data with elevation and area columns
    elevation_bands : list of tuples, optional
        List of (min, max) elevation ranges. Defaults to standard bands.
    xmin_candidates : array-like, optional
        x_min values to test. Defaults to log-spaced 0.01 to 10 km².
    elev_col, area_col : str
        Column names (defaults from config)
    ks_tolerance : float
        Tolerance for acceptable x_min range identification
    fixed_xmin_values : list, optional
        Additional fixed x_min values to compare
    show_progress : bool
        If True, show progress messages

    Returns
    -------
    dict
        Results keyed by elevation band with comprehensive metrics
    """
    if elev_col is None:
        elev_col = COLS['elevation']
    if area_col is None:
        area_col = COLS['area']

    if elevation_bands is None:
        elevation_bands = [
            (0, 500), (500, 1000), (1000, 1500),
            (1500, 2000), (2000, 2500), (2500, 3000)
        ]

    if xmin_candidates is None:
        xmin_candidates = np.logspace(-2, 1, 30)  # 0.01 to 10 km²

    if fixed_xmin_values is None:
        fixed_xmin_values = [0.1, 0.5, 1.0, 2.0]

    results = {}
    n_bands = len(elevation_bands)

    if show_progress:
        print("\n" + "=" * 60)
        print("X_MIN SENSITIVITY ANALYSIS BY ELEVATION")
        print("=" * 60)
        print(f"Analyzing {n_bands} elevation bands...")
        print(f"Testing {len(xmin_candidates)} x_min values per band")

    for i, (elev_min, elev_max) in enumerate(elevation_bands):
        band_key = f"{elev_min}-{elev_max}m"

        if show_progress:
            print(f"\n[{i+1}/{n_bands}] Processing {band_key}...")

        band_result = analyze_elevation_band_xmin(
            lakes_df, elev_min, elev_max, xmin_candidates,
            elev_col=elev_col, area_col=area_col,
            ks_tolerance=ks_tolerance,
            fixed_xmin_values=fixed_xmin_values
        )

        results[band_key] = band_result

        if show_progress and band_result['status'] == 'success':
            print(f"    n_total: {band_result['n_total']:,}")
            print(f"    optimal x_min: {band_result['optimal_xmin']:.3f} km²")
            print(f"    optimal alpha: {band_result['optimal_alpha']:.3f}")
            print(f"    acceptable range: [{band_result['acceptable_range'][0]:.3f}, "
                  f"{band_result['acceptable_range'][1]:.3f}] km²")
            if pd.notna(band_result['alpha_stability']):
                print(f"    alpha stability: ±{band_result['alpha_stability']/2:.3f}")

    if show_progress:
        print("\n" + "=" * 60)
        print("X_MIN SENSITIVITY ANALYSIS COMPLETE")
        print("=" * 60)

    return results


def compare_xmin_methods(xmin_results):
    """
    Compare alpha estimates using different x_min selection methods.

    Methods compared:
    1. Band-specific optimal x_min
    2. Fixed x_min = 0.5 km² (smaller threshold)
    3. Fixed x_min = 1.0 km² (round number)
    4. Fixed x_min = 2.0 km² (larger threshold)

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()

    Returns
    -------
    DataFrame
        Comparison of alpha estimates across methods
    """
    comparison = []

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            continue

        row = {
            'elevation_band': band_key,
            'n_total': results['n_total'],
            'alpha_optimal': results['optimal_alpha'],
            'xmin_optimal': results['optimal_xmin'],
            'ks_optimal': results['optimal_ks'],
        }

        # Add fixed x_min results
        for xmin_fixed, metrics in results['alpha_at_fixed_xmin'].items():
            row[f'alpha_xmin_{xmin_fixed}'] = metrics['alpha']
            row[f'n_tail_xmin_{xmin_fixed}'] = metrics['n_tail']

        # Compute differences from optimal
        for xmin_fixed in results['alpha_at_fixed_xmin'].keys():
            if pd.notna(row.get(f'alpha_xmin_{xmin_fixed}')):
                row[f'diff_xmin_{xmin_fixed}'] = (
                    row[f'alpha_xmin_{xmin_fixed}'] - results['optimal_alpha']
                )

        comparison.append(row)

    return pd.DataFrame(comparison)


def test_alpha_robustness(xmin_results):
    """
    Test robustness of alpha estimates across elevation bands.

    For each band, reports:
    - Alpha at optimal x_min
    - Alpha range within acceptable x_min range
    - Whether the band shows "unusual" alpha that might be threshold-dependent

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()

    Returns
    -------
    DataFrame
        Robustness metrics for each elevation band
    """
    robustness = []

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            continue

        row = {
            'elevation_band': band_key,
            'elev_mid': results['elev_mid'],
            'alpha_optimal': results['optimal_alpha'],
            'alpha_range_low': results['alpha_range_in_acceptable'][0],
            'alpha_range_high': results['alpha_range_in_acceptable'][1],
            'alpha_stability': results['alpha_stability'],
            'xmin_optimal': results['optimal_xmin'],
            'xmin_range_low': results['acceptable_range'][0],
            'xmin_range_high': results['acceptable_range'][1],
        }

        # Classify robustness
        if pd.notna(results['alpha_stability']):
            if results['alpha_stability'] < 0.05:
                row['robustness'] = 'high'
            elif results['alpha_stability'] < 0.1:
                row['robustness'] = 'moderate'
            else:
                row['robustness'] = 'low'
        else:
            row['robustness'] = 'unknown'

        robustness.append(row)

    return pd.DataFrame(robustness)


def generate_xmin_summary_table(xmin_results, include_fixed=True):
    """
    Generate publication-ready summary table.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    include_fixed : bool
        If True, include alpha at fixed x_min values

    Returns
    -------
    DataFrame
        Summary table suitable for publication
    """
    rows = []

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            row = {
                'Elevation Band': band_key,
                'n_total': results['n_total'],
                'Status': 'Insufficient data',
            }
            rows.append(row)
            continue

        row = {
            'Elevation Band': band_key,
            'n_total': results['n_total'],
            'Optimal x_min (km²)': f"{results['optimal_xmin']:.3f}",
            'x_min Range (km²)': f"[{results['acceptable_range'][0]:.2f}, {results['acceptable_range'][1]:.2f}]",
            'α (optimal)': f"{results['optimal_alpha']:.3f}",
            'KS Statistic': f"{results['optimal_ks']:.4f}",
            'α Stability': f"±{results['alpha_stability']/2:.3f}" if pd.notna(results['alpha_stability']) else 'N/A',
        }

        if include_fixed:
            for xmin_fixed, metrics in results['alpha_at_fixed_xmin'].items():
                if pd.notna(metrics['alpha']):
                    row[f'α (x_min={xmin_fixed})'] = f"{metrics['alpha']:.3f}"
                else:
                    row[f'α (x_min={xmin_fixed})'] = 'N/A'

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# HYPOTHESIS TESTS FOR X_MIN SENSITIVITY BY ELEVATION
# ============================================================================

def test_xmin_variation_by_elevation(xmin_results):
    """
    Hypothesis Test 1: Does optimal x_min vary by elevation band?

    Tests whether glacially-scoured terrain (high elevation) has a systematically
    different transition scale than floodplain environments (low elevation).

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()

    Returns
    -------
    dict
        Test results including:
        - xmin_by_band: optimal x_min for each band
        - xmin_range: (min, max) across all bands
        - xmin_cv: coefficient of variation
        - significant_variation: bool, True if CV > 0.3
        - elevation_correlation: correlation between elevation and x_min
        - interpretation: text describing findings
    """
    xmin_values = []
    elev_mids = []
    bands = []

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            continue
        if pd.notna(results['optimal_xmin']):
            xmin_values.append(results['optimal_xmin'])
            elev_mids.append(results['elev_mid'])
            bands.append(band_key)

    if len(xmin_values) < 2:
        return {
            'xmin_by_band': {},
            'xmin_range': (np.nan, np.nan),
            'xmin_cv': np.nan,
            'significant_variation': False,
            'elevation_correlation': np.nan,
            'p_value': np.nan,
            'interpretation': 'Insufficient data for comparison',
        }

    xmin_arr = np.array(xmin_values)
    elev_arr = np.array(elev_mids)

    # Compute variation metrics
    xmin_mean = np.mean(xmin_arr)
    xmin_std = np.std(xmin_arr)
    xmin_cv = xmin_std / xmin_mean if xmin_mean > 0 else np.nan

    # Correlation with elevation
    if len(xmin_arr) >= 3:
        corr, p_value = stats.spearmanr(elev_arr, xmin_arr)
    else:
        corr, p_value = np.nan, np.nan

    # Build band-level results
    xmin_by_band = {band: xmin for band, xmin in zip(bands, xmin_values)}

    # Interpretation
    if xmin_cv > 0.5:
        variation_level = "high"
    elif xmin_cv > 0.3:
        variation_level = "moderate"
    else:
        variation_level = "low"

    interpretation_parts = [f"Coefficient of variation in x_min is {variation_level} ({xmin_cv:.2f})"]

    if pd.notna(corr):
        if p_value < 0.05:
            if corr > 0:
                interpretation_parts.append(
                    f"x_min increases significantly with elevation (ρ={corr:.2f}, p={p_value:.3f})")
            else:
                interpretation_parts.append(
                    f"x_min decreases significantly with elevation (ρ={corr:.2f}, p={p_value:.3f})")
        else:
            interpretation_parts.append(
                f"No significant correlation with elevation (ρ={corr:.2f}, p={p_value:.3f})")

    return {
        'xmin_by_band': xmin_by_band,
        'xmin_range': (np.min(xmin_arr), np.max(xmin_arr)),
        'xmin_mean': xmin_mean,
        'xmin_std': xmin_std,
        'xmin_cv': xmin_cv,
        'significant_variation': xmin_cv > 0.3,
        'elevation_correlation': corr,
        'p_value': p_value,
        'interpretation': '; '.join(interpretation_parts),
    }


def analyze_ks_behavior(xmin_results):
    """
    Hypothesis Test 2: How does KS statistic behave in each band?

    A flat KS curve suggests the power law fit is robust across x_min choices.
    A sharply defined minimum suggests a clear physical threshold.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()

    Returns
    -------
    dict
        For each band:
        - ks_curve_shape: 'flat', 'moderate', or 'sharp'
        - ks_min: minimum KS value
        - ks_range_at_threshold: range of KS values within 50% of optimal x_min
        - flatness_score: lower = flatter curve (more robust)
    """
    band_results = {}

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            band_results[band_key] = {
                'ks_curve_shape': 'unknown',
                'ks_min': np.nan,
                'flatness_score': np.nan,
                'interpretation': 'Insufficient data',
            }
            continue

        full_results = results['full_results']
        if len(full_results) == 0:
            continue

        # Get KS values
        ks_values = full_results['ks_statistic'].dropna()
        xmin_values = full_results.loc[ks_values.index, 'xmin']

        if len(ks_values) < 3:
            band_results[band_key] = {
                'ks_curve_shape': 'unknown',
                'ks_min': np.nan,
                'flatness_score': np.nan,
                'interpretation': 'Insufficient x_min points',
            }
            continue

        ks_min = ks_values.min()
        ks_max = ks_values.max()
        ks_range = ks_max - ks_min

        # Compute flatness: std(KS) / mean(KS) in acceptable range
        optimal_xmin = results['optimal_xmin']
        near_optimal_mask = (xmin_values >= optimal_xmin * 0.5) & (xmin_values <= optimal_xmin * 2.0)
        ks_near_optimal = ks_values[near_optimal_mask]

        if len(ks_near_optimal) > 0:
            flatness_score = ks_near_optimal.std() / ks_near_optimal.mean() if ks_near_optimal.mean() > 0 else np.nan
        else:
            flatness_score = np.nan

        # Classify shape
        if pd.notna(flatness_score):
            if flatness_score < 0.1:
                shape = 'flat'
                interpretation = 'KS curve is flat - power law fit is robust across x_min choices'
            elif flatness_score < 0.25:
                shape = 'moderate'
                interpretation = 'KS curve shows moderate variation - power law fit is reasonably stable'
            else:
                shape = 'sharp'
                interpretation = 'KS curve has a sharp minimum - suggests a clear physical threshold'
        else:
            shape = 'unknown'
            interpretation = 'Could not determine KS curve shape'

        band_results[band_key] = {
            'ks_curve_shape': shape,
            'ks_min': ks_min,
            'ks_max': ks_max,
            'ks_range': ks_range,
            'flatness_score': flatness_score,
            'n_points_analyzed': len(ks_near_optimal),
            'interpretation': interpretation,
        }

    return band_results


def analyze_alpha_trajectory(xmin_results, high_xmin_threshold=2.0):
    """
    Hypothesis Test 3: What's the α trajectory as x_min increases in each band?

    If all bands converge toward similar α at high x_min, elevation differences
    might be driven by what's happening near the threshold. If they remain
    separated, the signal is more robust.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    high_xmin_threshold : float
        x_min value considered "high" for convergence analysis

    Returns
    -------
    dict
        - alpha_at_low_xmin: alpha values at x_min near 0.1 km²
        - alpha_at_high_xmin: alpha values at x_min near threshold
        - convergence_metric: spread at high x_min vs low x_min
        - converges: bool, True if bands converge at high x_min
        - interpretation: text describing pattern
    """
    alpha_low = {}
    alpha_high = {}
    alpha_optimal = {}

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            continue

        full_results = results['full_results']
        if len(full_results) == 0:
            continue

        alpha_optimal[band_key] = results['optimal_alpha']

        # Get alpha at low x_min (near 0.1)
        low_xmin_mask = full_results['xmin'] <= 0.2
        if low_xmin_mask.any():
            low_alphas = full_results.loc[low_xmin_mask, 'alpha'].dropna()
            if len(low_alphas) > 0:
                alpha_low[band_key] = low_alphas.iloc[0]

        # Get alpha at high x_min
        high_xmin_mask = full_results['xmin'] >= high_xmin_threshold
        if high_xmin_mask.any():
            high_alphas = full_results.loc[high_xmin_mask, 'alpha'].dropna()
            if len(high_alphas) > 0:
                alpha_high[band_key] = high_alphas.iloc[-1]

    if len(alpha_low) < 2 or len(alpha_high) < 2:
        return {
            'alpha_at_low_xmin': alpha_low,
            'alpha_at_high_xmin': alpha_high,
            'alpha_at_optimal': alpha_optimal,
            'spread_at_low': np.nan,
            'spread_at_high': np.nan,
            'convergence_ratio': np.nan,
            'converges': None,
            'interpretation': 'Insufficient data for convergence analysis',
        }

    # Compute spread (range) at low and high x_min
    low_values = np.array(list(alpha_low.values()))
    high_values = np.array(list(alpha_high.values()))

    spread_low = np.max(low_values) - np.min(low_values)
    spread_high = np.max(high_values) - np.min(high_values)

    # Convergence ratio: if < 1, bands converge at high x_min
    convergence_ratio = spread_high / spread_low if spread_low > 0 else np.nan

    # Determine if convergence occurs
    if pd.notna(convergence_ratio):
        if convergence_ratio < 0.5:
            converges = True
            interpretation = (f"Elevation bands CONVERGE at high x_min (spread reduces from "
                            f"{spread_low:.3f} to {spread_high:.3f}). Alpha differences may be "
                            "threshold-dependent rather than reflecting true elevation effects.")
        elif convergence_ratio > 1.5:
            converges = False
            interpretation = (f"Elevation bands DIVERGE at high x_min (spread increases from "
                            f"{spread_low:.3f} to {spread_high:.3f}). Alpha differences are "
                            "robust and amplify with stricter thresholds.")
        else:
            converges = None
            interpretation = (f"Elevation bands maintain similar spread across x_min values "
                            f"(low: {spread_low:.3f}, high: {spread_high:.3f}). Alpha differences "
                            "are stable but not strongly convergent or divergent.")
    else:
        converges = None
        interpretation = "Could not determine convergence pattern"

    return {
        'alpha_at_low_xmin': alpha_low,
        'alpha_at_high_xmin': alpha_high,
        'alpha_at_optimal': alpha_optimal,
        'spread_at_low': spread_low,
        'spread_at_high': spread_high,
        'convergence_ratio': convergence_ratio,
        'converges': converges,
        'interpretation': interpretation,
    }


def test_ntail_stability(xmin_results, min_ntail_fraction=0.05):
    """
    Hypothesis Test 4: Sample size stability in tail by band.

    Verifies that n_tail doesn't collapse as x_min shifts, which would
    undermine the statistical reliability of alpha estimates.

    For the 1000-1500m band with n=535 in tail and α ≈ 1.73,
    standard error is (0.73)/√535 ≈ 0.032.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    min_ntail_fraction : float
        Minimum acceptable fraction of n_total in tail

    Returns
    -------
    dict
        For each band:
        - n_tail_at_optimal: sample size at optimal x_min
        - n_tail_range: (min, max) across acceptable x_min range
        - se_at_optimal: standard error at optimal x_min
        - se_range: (min, max) SE across acceptable x_min range
        - stable: bool, True if n_tail stays above threshold
        - interpretation: text describing stability
    """
    band_results = {}

    # Skip non-elevation-band keys
    skip_keys = {'method_comparison', 'robustness', 'summary_table',
                 'hypothesis_tests', 'hypothesis_report'}

    for band_key, results in xmin_results.items():
        # Skip metadata keys
        if band_key in skip_keys:
            continue
        # Skip if not a dict or doesn't have expected structure
        if not isinstance(results, dict) or 'status' not in results:
            continue
        if results['status'] != 'success':
            band_results[band_key] = {
                'n_tail_at_optimal': 0,
                'stable': False,
                'interpretation': 'Insufficient data',
            }
            continue

        n_total = results['n_total']
        full_results = results['full_results']

        if len(full_results) == 0:
            continue

        # Get n_tail at optimal x_min
        optimal_xmin = results['optimal_xmin']
        optimal_alpha = results['optimal_alpha']
        optimal_idx = (full_results['xmin'] - optimal_xmin).abs().idxmin()
        n_tail_optimal = full_results.loc[optimal_idx, 'n_tail']

        # Standard error at optimal
        if pd.notna(optimal_alpha) and n_tail_optimal > 0:
            se_optimal = (optimal_alpha - 1) / np.sqrt(n_tail_optimal)
        else:
            se_optimal = np.nan

        # Get n_tail range within acceptable x_min range
        acc_low, acc_high = results['acceptable_range']
        if pd.notna(acc_low) and pd.notna(acc_high):
            acc_mask = (full_results['xmin'] >= acc_low) & (full_results['xmin'] <= acc_high)
            acc_data = full_results[acc_mask]

            if len(acc_data) > 0:
                n_tail_range = (acc_data['n_tail'].min(), acc_data['n_tail'].max())

                # Compute SE range
                valid_alpha = acc_data['alpha'].dropna()
                valid_ntail = acc_data.loc[valid_alpha.index, 'n_tail']
                if len(valid_alpha) > 0 and (valid_ntail > 0).all():
                    se_values = (valid_alpha - 1) / np.sqrt(valid_ntail)
                    se_range = (se_values.min(), se_values.max())
                else:
                    se_range = (np.nan, np.nan)
            else:
                n_tail_range = (np.nan, np.nan)
                se_range = (np.nan, np.nan)
        else:
            n_tail_range = (np.nan, np.nan)
            se_range = (np.nan, np.nan)

        # Check stability: n_tail should not fall below threshold fraction
        min_acceptable = int(n_total * min_ntail_fraction)
        if pd.notna(n_tail_range[0]):
            stable = n_tail_range[0] >= min_acceptable
        else:
            stable = n_tail_optimal >= min_acceptable

        # Interpretation
        if stable:
            if pd.notna(se_optimal):
                interpretation = (f"Sample size stable (n_tail={n_tail_optimal:,} at optimal x_min). "
                                f"SE = {se_optimal:.4f}, providing reliable alpha estimates.")
            else:
                interpretation = f"Sample size stable (n_tail={n_tail_optimal:,} at optimal x_min)."
        else:
            interpretation = (f"WARNING: Sample size may be unstable. n_tail={n_tail_optimal:,} "
                            f"at optimal x_min, but drops to {n_tail_range[0]:,} at upper x_min range. "
                            "Consider using a lower x_min for more reliable estimates.")

        band_results[band_key] = {
            'n_total': n_total,
            'n_tail_at_optimal': n_tail_optimal,
            'n_tail_range': n_tail_range,
            'pct_tail_at_optimal': n_tail_optimal / n_total * 100 if n_total > 0 else 0,
            'se_at_optimal': se_optimal,
            'se_range': se_range,
            'min_acceptable_ntail': min_acceptable,
            'stable': stable,
            'interpretation': interpretation,
        }

    return band_results


def run_all_hypothesis_tests(xmin_results, verbose=True):
    """
    Run all four hypothesis tests on x_min sensitivity results.

    Tests performed:
    1. Does optimal x_min vary by elevation band?
    2. How does KS statistic behave in each band?
    3. What's the α trajectory as x_min increases?
    4. Sample size stability in tail by band

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    verbose : bool
        If True, print detailed results

    Returns
    -------
    dict
        Comprehensive hypothesis test results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS TESTS FOR X_MIN SENSITIVITY BY ELEVATION")
        print("=" * 70)

    # Test 1: x_min variation
    if verbose:
        print("\n" + "-" * 70)
        print("TEST 1: Does optimal x_min vary by elevation band?")
        print("-" * 70)

    test1 = test_xmin_variation_by_elevation(xmin_results)

    if verbose:
        print(f"  x_min range: [{test1['xmin_range'][0]:.3f}, {test1['xmin_range'][1]:.3f}] km²")
        print(f"  x_min mean: {test1['xmin_mean']:.3f} km² (std: {test1['xmin_std']:.3f})")
        print(f"  Coefficient of variation: {test1['xmin_cv']:.3f}")
        print(f"  Elevation correlation: ρ = {test1['elevation_correlation']:.3f} (p = {test1['p_value']:.4f})")
        print(f"\n  → {test1['interpretation']}")

    # Test 2: KS behavior
    if verbose:
        print("\n" + "-" * 70)
        print("TEST 2: How does KS statistic behave in each band?")
        print("-" * 70)

    test2 = analyze_ks_behavior(xmin_results)

    if verbose:
        for band_key, band_result in test2.items():
            if band_result['ks_curve_shape'] != 'unknown':
                print(f"  {band_key}: {band_result['ks_curve_shape']} "
                      f"(flatness={band_result['flatness_score']:.3f}, "
                      f"KS_min={band_result['ks_min']:.4f})")

    # Test 3: Alpha trajectory
    if verbose:
        print("\n" + "-" * 70)
        print("TEST 3: What's the α trajectory as x_min increases?")
        print("-" * 70)

    test3 = analyze_alpha_trajectory(xmin_results)

    if verbose:
        print(f"  Spread at low x_min: {test3['spread_at_low']:.3f}")
        print(f"  Spread at high x_min: {test3['spread_at_high']:.3f}")
        print(f"  Convergence ratio: {test3['convergence_ratio']:.3f}")
        print(f"\n  → {test3['interpretation']}")

    # Test 4: n_tail stability
    if verbose:
        print("\n" + "-" * 70)
        print("TEST 4: Sample size stability in tail by band")
        print("-" * 70)

    test4 = test_ntail_stability(xmin_results)

    if verbose:
        for band_key, band_result in test4.items():
            if 'n_tail_at_optimal' in band_result and band_result['n_tail_at_optimal'] > 0:
                status = "✓ STABLE" if band_result['stable'] else "⚠ UNSTABLE"
                print(f"  {band_key}: n_tail={band_result['n_tail_at_optimal']:,}, "
                      f"SE={band_result['se_at_optimal']:.4f} {status}")

    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS TESTS COMPLETE")
        print("=" * 70)

    return {
        'xmin_variation': test1,
        'ks_behavior': test2,
        'alpha_trajectory': test3,
        'ntail_stability': test4,
    }


def generate_hypothesis_test_report(hypothesis_results, output_path=None):
    """
    Generate a formatted text report of hypothesis test results.

    Parameters
    ----------
    hypothesis_results : dict
        Output from run_all_hypothesis_tests()
    output_path : str, optional
        If provided, save report to file

    Returns
    -------
    str
        Formatted report text
    """
    lines = [
        "=" * 70,
        "HYPOTHESIS TEST REPORT: X_MIN SENSITIVITY BY ELEVATION",
        "=" * 70,
        "",
        "SUMMARY",
        "-" * 70,
        "",
    ]

    # Test 1 summary
    test1 = hypothesis_results['xmin_variation']
    lines.extend([
        "1. OPTIMAL X_MIN VARIATION BY ELEVATION",
        f"   Range: [{test1['xmin_range'][0]:.3f}, {test1['xmin_range'][1]:.3f}] km²",
        f"   CV: {test1['xmin_cv']:.3f} ({'significant' if test1['significant_variation'] else 'not significant'})",
        f"   Correlation with elevation: ρ = {test1['elevation_correlation']:.3f}",
        f"   Interpretation: {test1['interpretation']}",
        "",
    ])

    # Test 2 summary
    test2 = hypothesis_results['ks_behavior']
    shape_counts = {'flat': 0, 'moderate': 0, 'sharp': 0}
    for band_result in test2.values():
        shape = band_result.get('ks_curve_shape', 'unknown')
        if shape in shape_counts:
            shape_counts[shape] += 1

    lines.extend([
        "2. KS STATISTIC BEHAVIOR",
        f"   Flat curves: {shape_counts['flat']} bands (robust fit)",
        f"   Moderate curves: {shape_counts['moderate']} bands",
        f"   Sharp curves: {shape_counts['sharp']} bands (clear threshold)",
        "",
    ])

    # Test 3 summary
    test3 = hypothesis_results['alpha_trajectory']
    lines.extend([
        "3. ALPHA TRAJECTORY WITH INCREASING X_MIN",
        f"   Spread at low x_min: {test3['spread_at_low']:.3f}",
        f"   Spread at high x_min: {test3['spread_at_high']:.3f}",
        f"   Convergence ratio: {test3['convergence_ratio']:.3f}",
        f"   Interpretation: {test3['interpretation']}",
        "",
    ])

    # Test 4 summary
    test4 = hypothesis_results['ntail_stability']
    n_stable = sum(1 for r in test4.values() if r.get('stable', False))
    n_total = sum(1 for r in test4.values() if 'stable' in r)
    lines.extend([
        "4. SAMPLE SIZE STABILITY",
        f"   Stable bands: {n_stable}/{n_total}",
        "",
        "   Per-band details:",
    ])

    for band_key, band_result in test4.items():
        if 'n_tail_at_optimal' in band_result and band_result['n_tail_at_optimal'] > 0:
            lines.append(f"     {band_key}: n={band_result['n_tail_at_optimal']:,}, "
                        f"SE={band_result['se_at_optimal']:.4f}")

    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

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
    print("  - fit_powerlaw_by_process_domain(df)")
    print("  - xmin_sensitivity_analysis(areas)")
    print("  - compare_to_cael_seekell(areas)")
    print("  - xmin_sensitivity_by_elevation(df)    # Comprehensive elevation analysis")
    print("  - compare_xmin_methods(results)")
    print("  - test_alpha_robustness(results)")
    print("  - generate_xmin_summary_table(results)")
    print("\n  Hypothesis Tests:")
    print("  - test_xmin_variation_by_elevation(results)")
    print("  - analyze_ks_behavior(results)")
    print("  - analyze_alpha_trajectory(results)")
    print("  - test_ntail_stability(results)")
    print("  - run_all_hypothesis_tests(results)    # Run all 4 tests")
