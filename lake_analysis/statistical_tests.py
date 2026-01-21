"""
Statistical Testing Module with Multiple Testing Correction

This module provides statistical testing functions with proper multiple testing
correction to avoid inflated Type I error rates.

**Scientific Problem:**
When performing multiple hypothesis tests, the probability of at least one
false positive increases. Without correction, conclusions may be spurious.

**Solution:**
Apply False Discovery Rate (FDR) or Family-Wise Error Rate (FWER) correction
to p-values from multiple comparisons.

**Methods:**
- Bonferroni: FWER control (conservative)
- Holm-Bonferroni: Sequential FWER (less conservative)
- Benjamini-Hochberg: FDR control (recommended for exploratory analysis)

Author: morrismc
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from statsmodels.stats.multitest import multipletests


def test_davis_hypothesis_with_correction(
    density_df: pd.DataFrame,
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test Davis's hypothesis with multiple testing correction.

    Performs correlation and regression tests to examine relationship between
    glacial stage age and lake density, with proper correction for multiple
    comparisons.

    Parameters
    ----------
    density_df : DataFrame
        Output from compute_lake_density_by_glacial_stage()
        Must have columns: 'age_ka', 'density_per_1000km2'
    correction_method : str
        Multiple testing correction method:
        - 'bonferroni': Bonferroni correction (FWER)
        - 'holm': Holm-Bonferroni (sequential FWER)
        - 'fdr_bh': Benjamini-Hochberg FDR (recommended)
        - 'fdr_by': Benjamini-Yekutieli FDR (more conservative)
    alpha : float
        Significance level (default: 0.05)
    verbose : bool
        Print results

    Returns
    -------
    dict
        Test results including corrected p-values

    Examples
    --------
    >>> results = test_davis_hypothesis_with_correction(density_df)
    >>> print(f"Pearson p-value (corrected): {results['pearson_p_corrected']:.4f}")
    >>> print(f"Significant after correction: {results['significant']}")

    Notes
    -----
    Davis's hypothesis (1899) predicts negative correlation between landscape
    age and lake density. We test this with:
    1. Pearson correlation (parametric)
    2. Spearman correlation (non-parametric)
    3. Linear regression

    All p-values are corrected for multiple testing (3 tests).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DAVIS'S HYPOTHESIS TEST (WITH MULTIPLE TESTING CORRECTION)")
        print("=" * 70)
        print(f"Correction method: {correction_method}")
        print(f"Significance level: {alpha}")
        print("\nH0: Lake density does not vary with glacial stage age")
        print("Ha: Lake density decreases with increasing landscape age\n")

    # Filter to valid data
    valid = density_df.dropna(subset=['age_ka', 'density_per_1000km2'])

    if len(valid) < 3:
        return {
            'error': 'Insufficient data',
            'n_stages': len(valid),
            'significant': False
        }

    ages = valid['age_ka'].values
    densities = valid['density_per_1000km2'].values

    # Run tests
    pearson_r, pearson_p = stats.pearsonr(ages, densities)
    spearman_r, spearman_p = stats.spearmanr(ages, densities)
    slope, intercept, r_value, reg_p, std_err = stats.linregress(ages, densities)

    # Collect uncorrected p-values
    p_values = [pearson_p, spearman_p, reg_p]
    test_names = ['Pearson correlation', 'Spearman correlation', 'Linear regression']

    # Apply multiple testing correction
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values,
        alpha=alpha,
        method=correction_method
    )

    results = {
        'n_stages': len(valid),
        'ages': ages.tolist(),
        'densities': densities.tolist(),

        # Pearson correlation
        'pearson_r': pearson_r,
        'pearson_p_uncorrected': pearson_p,
        'pearson_p_corrected': p_corrected[0],
        'pearson_significant': reject[0],

        # Spearman correlation
        'spearman_r': spearman_r,
        'spearman_p_uncorrected': spearman_p,
        'spearman_p_corrected': p_corrected[1],
        'spearman_significant': reject[1],

        # Regression
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'reg_p_uncorrected': reg_p,
        'reg_p_corrected': p_corrected[2],
        'reg_significant': reject[2],
        'std_err': std_err,

        # Overall result
        'supports_davis': any(reject) and (pearson_r < 0),
        'correction_method': correction_method,
        'alpha': alpha
    }

    if verbose:
        print(f"Sample size: {len(valid)} glacial stages\n")
        print("Test Results:")
        print("-" * 70)
        print(f"{'Test':<25} {'Statistic':<12} {'p (raw)':<12} {'p (adj)':<12} {'Sig?':<6}")
        print("-" * 70)

        print(f"{'Pearson correlation':<25} {f'r={pearson_r:.3f}':<12} "
              f"{pearson_p:<12.4f} {p_corrected[0]:<12.4f} {'***' if reject[0] else 'ns':<6}")
        print(f"{'Spearman correlation':<25} {f'rho={spearman_r:.3f}':<12} "
              f"{spearman_p:<12.4f} {p_corrected[1]:<12.4f} {'***' if reject[1] else 'ns':<6}")
        print(f"{'Linear regression':<25} {f'R²={r_value**2:.3f}':<12} "
              f"{reg_p:<12.4f} {p_corrected[2]:<12.4f} {'***' if reject[2] else 'ns':<6}")
        print("-" * 70)

        print(f"\nRegression equation: Density = {slope:.4f} * Age + {intercept:.2f}")
        print(f"Standard error: {std_err:.4f}")

        print(f"\n{'CONCLUSION:'}")
        if results['supports_davis']:
            print(f"  ✓ SUPPORTS Davis's hypothesis after correction")
            print(f"    Negative correlation significant at α={alpha}")
            print(f"    Lake density decreases {abs(slope)*1000:.2f} lakes/1000km² per 1000 years")
        elif pearson_r < 0 and not any(reject):
            print(f"  ~ TREND supports Davis but NOT significant after correction")
            print(f"    p-values after {correction_method} correction exceed α={alpha}")
        else:
            print(f"  ✗ DOES NOT support Davis's hypothesis")

    return results


def pairwise_comparisons_with_correction(
    data_by_group: Dict[str, np.ndarray],
    test: str = 'mann-whitney',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform pairwise comparisons between multiple groups with correction.

    Parameters
    ----------
    data_by_group : dict
        Dictionary mapping group names to data arrays
    test : str
        Statistical test: 'mann-whitney', 't-test', 'ks'
    correction_method : str
        Multiple testing correction method
    alpha : float
        Significance level
    verbose : bool
        Print results

    Returns
    -------
    dict
        Pairwise comparison results with corrected p-values

    Examples
    --------
    >>> data = {
    ...     'wisconsin': wisconsin_densities,
    ...     'illinoian': illinoian_densities,
    ...     'driftless': driftless_densities
    ... }
    >>> results = pairwise_comparisons_with_correction(data)
    """
    groups = list(data_by_group.keys())
    n_groups = len(groups)
    n_comparisons = n_groups * (n_groups - 1) // 2

    comparisons = []
    p_values = []
    statistics = []

    # Perform all pairwise comparisons
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            group1 = groups[i]
            group2 = groups[j]
            data1 = data_by_group[group1]
            data2 = data_by_group[group2]

            # Choose test
            if test == 'mann-whitney':
                stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            elif test == 't-test':
                stat, p = stats.ttest_ind(data1, data2)
            elif test == 'ks':
                stat, p = stats.ks_2samp(data1, data2)
            else:
                raise ValueError(f"Unknown test: {test}")

            comparisons.append((group1, group2))
            p_values.append(p)
            statistics.append(stat)

    # Apply correction
    reject, p_corrected, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method=correction_method
    )

    results = {
        'n_comparisons': n_comparisons,
        'test': test,
        'correction_method': correction_method,
        'comparisons': []
    }

    for i, (group1, group2) in enumerate(comparisons):
        results['comparisons'].append({
            'group1': group1,
            'group2': group2,
            'statistic': statistics[i],
            'p_uncorrected': p_values[i],
            'p_corrected': p_corrected[i],
            'significant': reject[i],
            'n1': len(data_by_group[group1]),
            'n2': len(data_by_group[group2])
        })

    if verbose:
        print("\n" + "=" * 70)
        print(f"PAIRWISE COMPARISONS ({test.upper()})")
        print(f"Correction: {correction_method}, α={alpha}")
        print("=" * 70)
        print(f"\n{n_comparisons} comparisons among {n_groups} groups:\n")
        print(f"{'Comparison':<30} {'n1':<6} {'n2':<6} {'Statistic':<12} {'p (raw)':<12} {'p (adj)':<12} {'Sig?':<6}")
        print("-" * 90)

        for comp in results['comparisons']:
            print(f"{comp['group1']+' vs '+comp['group2']:<30} "
                  f"{comp['n1']:<6} {comp['n2']:<6} "
                  f"{comp['statistic']:<12.2f} "
                  f"{comp['p_uncorrected']:<12.4f} "
                  f"{comp['p_corrected']:<12.4f} "
                  f"{'***' if comp['significant'] else 'ns':<6}")

        n_significant = sum(reject)
        print("-" * 90)
        print(f"\n{n_significant}/{n_comparisons} comparisons significant after correction")

    return results


def regression_with_multiple_testing(
    X: np.ndarray,
    y: np.ndarray,
    predictor_names: List[str],
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Multiple regression with corrected p-values for coefficients.

    Parameters
    ----------
    X : ndarray
        Predictor matrix (n_samples, n_predictors)
    y : ndarray
        Response variable (n_samples,)
    predictor_names : list of str
        Names of predictors
    correction_method : str
        Multiple testing correction method
    alpha : float
        Significance level
    verbose : bool
        Print results

    Returns
    -------
    dict
        Regression results with corrected p-values

    Notes
    -----
    When interpreting multiple regression coefficients, each coefficient
    test is a hypothesis test. With p predictors, we perform p tests,
    requiring multiple testing correction.
    """
    from sklearn.linear_model import LinearRegression
    from scipy.stats import t as t_dist

    # Fit model
    model = LinearRegression().fit(X, y)

    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred

    n = len(y)
    p = X.shape[1]
    dof = n - p - 1

    # Standard errors
    mse = np.sum(residuals ** 2) / dof
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    std_errors = np.sqrt(np.diag(cov_matrix))

    # t-statistics and p-values
    coeffs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coeffs / std_errors
    p_values_uncorrected = 2 * (1 - t_dist.cdf(np.abs(t_stats), dof))

    # Apply correction (skip intercept)
    p_values_corrected_predictors = multipletests(
        p_values_uncorrected[1:],
        alpha=alpha,
        method=correction_method
    )[1]

    results = {
        'r_squared': model.score(X, y),
        'n_samples': n,
        'n_predictors': p,
        'intercept': model.intercept_,
        'intercept_se': std_errors[0],
        'intercept_p': p_values_uncorrected[0],
        'coefficients': [],
        'correction_method': correction_method,
        'alpha': alpha
    }

    for i, name in enumerate(predictor_names):
        results['coefficients'].append({
            'name': name,
            'coef': model.coef_[i],
            'std_err': std_errors[i + 1],
            't_stat': t_stats[i + 1],
            'p_uncorrected': p_values_uncorrected[i + 1],
            'p_corrected': p_values_corrected_predictors[i],
            'significant': p_values_corrected_predictors[i] < alpha
        })

    if verbose:
        print("\n" + "=" * 80)
        print("MULTIPLE REGRESSION WITH CORRECTED P-VALUES")
        print(f"Correction: {correction_method}, α={alpha}")
        print("=" * 80)
        print(f"\nR² = {results['r_squared']:.3f}")
        print(f"n = {n}, p = {p}, df = {dof}\n")
        print(f"{'Predictor':<20} {'Coef':<12} {'SE':<12} {'t':<10} {'p (raw)':<12} {'p (adj)':<12} {'Sig?':<6}")
        print("-" * 90)
        print(f"{'(Intercept)':<20} {results['intercept']:<12.4f} "
              f"{results['intercept_se']:<12.4f} {results['intercept']/<results['intercept_se']:<10.2f} "
              f"{results['intercept_p']:<12.4f} {'--':<12} {'--':<6}")

        for coef_result in results['coefficients']:
            print(f"{coef_result['name']:<20} {coef_result['coef']:<12.4f} "
                  f"{coef_result['std_err']:<12.4f} {coef_result['t_stat']:<10.2f} "
                  f"{coef_result['p_uncorrected']:<12.4f} {coef_result['p_corrected']:<12.4f} "
                  f"{'***' if coef_result['significant'] else 'ns':<6}")

        print("-" * 90)
        n_significant = sum(c['significant'] for c in results['coefficients'])
        print(f"\n{n_significant}/{p} predictors significant after correction")

    return results


# Convenience function
def apply_multiple_testing_correction(
    p_values: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to a list of p-values.

    Parameters
    ----------
    p_values : list of float
        Uncorrected p-values
    method : str
        Correction method (see multipletests documentation)
    alpha : float
        Significance level

    Returns
    -------
    reject : ndarray
        Boolean array indicating which tests reject null
    p_corrected : ndarray
        Corrected p-values

    Examples
    --------
    >>> p_values = [0.01, 0.04, 0.03, 0.08, 0.09]
    >>> reject, p_adj = apply_multiple_testing_correction(p_values)
    >>> print(f"Significant tests: {np.sum(reject)}/{len(p_values)}")
    """
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return reject, p_corrected


# Main execution
if __name__ == "__main__":
    print("Statistical Testing Module with Multiple Testing Correction")
    print("Use these functions instead of raw scipy.stats tests when performing")
    print("multiple comparisons to avoid inflated Type I error rates.")
