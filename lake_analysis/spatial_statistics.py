"""
Spatial Statistics Module

Provides spatial autocorrelation diagnostics and spatial regression models
to address the critical concern: gridded multivariate analysis treats cells
as independent when they are NOT.

**Scientific Problem:**
Standard regression assumes independent observations. Spatially proximate
grid cells have correlated environmental conditions and lake densities,
violating this assumption. This leads to:
- Underestimated standard errors
- Overly optimistic p-values
- Inflated significance for weak effects

**Solution:**
1. Test for spatial autocorrelation (Moran's I, Geary's C)
2. Use spatial regression models (Spatial Lag, Spatial Error)
3. Report robust standard errors
4. Compare OLS vs spatial model results

Author: morrismc
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings

try:
    from .config import OUTPUT_DIR, ensure_output_dir
except ImportError:
    from config import OUTPUT_DIR, ensure_output_dir


def compute_morans_i(
    values: np.ndarray,
    coordinates: np.ndarray,
    distance_threshold: Optional[float] = None,
    n_permutations: int = 999,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute Moran's I statistic to test for spatial autocorrelation.

    Moran's I ranges from -1 (perfect dispersion) to +1 (perfect clustering).
    I ≈ 0 indicates spatial randomness.

    Parameters
    ----------
    values : ndarray
        Variable to test for autocorrelation (e.g., residuals, lake density)
    coordinates : ndarray
        Spatial coordinates (n_observations, 2) as [lat, lon] or [x, y]
    distance_threshold : float, optional
        Distance threshold for defining neighbors (km or degrees)
        If None, uses inverse distance weighting
    n_permutations : int
        Number of random permutations for significance test
    verbose : bool
        Print results

    Returns
    -------
    dict
        'I': Moran's I statistic
        'expected_I': Expected value under null hypothesis
        'variance': Variance of I under null
        'z_score': Standardized I
        'p_value': Two-tailed p-value from permutation test
        'significant': Boolean (p < 0.05)

    Examples
    --------
    >>> # Test residuals for spatial autocorrelation
    >>> coords = np.column_stack([data['lat'], data['lon']])
    >>> result = compute_morans_i(residuals, coords)
    >>> print(f"Moran's I = {result['I']:.3f}, p = {result['p_value']:.4f}")

    References
    ----------
    Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
    Biometrika, 37(1/2), 17-23.
    """
    n = len(values)

    if len(coordinates) != n:
        raise ValueError("Length of values and coordinates must match")

    # Create spatial weights matrix
    W = _create_spatial_weights(coordinates, distance_threshold)

    # Standardize values
    y = values - np.mean(values)

    # Compute Moran's I
    numerator = n * np.sum(W * np.outer(y, y))
    denominator = np.sum(W) * np.sum(y ** 2)

    if denominator == 0:
        return {'I': np.nan, 'p_value': np.nan, 'significant': False}

    I = numerator / denominator

    # Expected value and variance under null hypothesis
    expected_I = -1.0 / (n - 1)
    S0 = np.sum(W)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((np.sum(W, axis=0) + np.sum(W, axis=1)) ** 2)

    variance_I = ((n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)) -
                  ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / \
                 ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)

    # Z-score
    z_score = (I - expected_I) / np.sqrt(variance_I)

    # Permutation test for p-value
    I_perm = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        I_p = n * np.sum(W * np.outer(y_perm, y_perm)) / (S0 * np.sum(y_perm ** 2))
        I_perm.append(I_p)

    I_perm = np.array(I_perm)
    p_value = np.sum(np.abs(I_perm) >= np.abs(I)) / n_permutations

    results = {
        'I': I,
        'expected_I': expected_I,
        'variance': variance_I,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_permutations': n_permutations,
        'interpretation': _interpret_morans_i(I, p_value)
    }

    if verbose:
        print("\n" + "=" * 60)
        print("MORAN'S I SPATIAL AUTOCORRELATION TEST")
        print("=" * 60)
        print(f"\nSample size: n = {n}")
        print(f"Moran's I: {I:.4f}")
        print(f"Expected I (under H0): {expected_I:.4f}")
        print(f"Z-score: {z_score:.3f}")
        print(f"P-value ({n_permutations} permutations): {p_value:.4f}")
        print(f"\nInterpretation: {results['interpretation']}")
        if results['significant']:
            print(f"*** SPATIAL AUTOCORRELATION DETECTED (p < 0.05) ***")
            print(f"Standard OLS regression assumptions violated!")
            print(f"Recommend spatial regression model (SAR or SEM).")
        else:
            print(f"No significant spatial autocorrelation detected.")

    return results


def _create_spatial_weights(
    coordinates: np.ndarray,
    distance_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Create spatial weights matrix from coordinates.

    Parameters
    ----------
    coordinates : ndarray (n, 2)
        Spatial coordinates
    distance_threshold : float, optional
        Distance threshold for neighbors

    Returns
    -------
    W : ndarray (n, n)
        Spatial weights matrix
    """
    n = len(coordinates)
    W = np.zeros((n, n))

    # Compute pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])

            if distance_threshold is None:
                # Inverse distance weighting (avoid divide by zero)
                weight = 1.0 / (dist + 1e-6) if dist > 0 else 0
            else:
                # Binary: 1 if within threshold, 0 otherwise
                weight = 1.0 if dist <= distance_threshold else 0.0

            W[i, j] = weight
            W[j, i] = weight

    # Row-standardize
    row_sums = np.sum(W, axis=1)
    row_sums[row_sums == 0] = 1  # Avoid divide by zero
    W = W / row_sums[:, np.newaxis]

    return W


def _interpret_morans_i(I: float, p_value: float) -> str:
    """Interpret Moran's I statistic."""
    if p_value >= 0.05:
        return "No significant spatial pattern (random)"
    elif I > 0:
        return f"Positive autocorrelation (clustering): similar values cluster spatially"
    elif I < 0:
        return f"Negative autocorrelation (dispersion): dissimilar values neighbor each other"
    else:
        return "Spatial randomness"


def test_spatial_autocorrelation_grid(
    data: pd.DataFrame,
    value_col: str,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    n_permutations: int = 999,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test gridded data for spatial autocorrelation.

    Convenience wrapper for compute_morans_i() that works with DataFrames.

    Parameters
    ----------
    data : DataFrame
        Gridded dataset with lat/lon coordinates
    value_col : str
        Column name for values to test (e.g., 'density', 'residuals')
    lat_col : str
        Column name for latitude
    lon_col : str
        Column name for longitude
    n_permutations : int
        Permutations for significance test
    verbose : bool
        Print results

    Returns
    -------
    dict
        Moran's I test results

    Examples
    --------
    >>> # Test whether lake density exhibits spatial clustering
    >>> results = test_spatial_autocorrelation_grid(
    ...     gridded_data,
    ...     value_col='density',
    ...     lat_col='lat',
    ...     lon_col='lon'
    ... )
    """
    # Extract coordinates and values
    coords = data[[lat_col, lon_col]].values
    values = data[value_col].values

    # Remove NaN
    valid = ~np.isnan(values)
    coords = coords[valid]
    values = values[valid]

    if len(values) < 10:
        if verbose:
            print(f"WARNING: Only {len(values)} valid observations. Insufficient for test.")
        return {'I': np.nan, 'p_value': np.nan, 'significant': False}

    # Compute Moran's I
    return compute_morans_i(coords, values, n_permutations=n_permutations, verbose=verbose)


def fit_spatial_lag_model(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    predictor_names: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fit Spatial Lag Model (SAR): y = ρWy + Xβ + ε

    The spatial lag model includes a spatially lagged dependent variable
    as a predictor, capturing spatial spillover effects.

    Parameters
    ----------
    X : ndarray (n, p)
        Predictor matrix
    y : ndarray (n,)
        Response variable
    W : ndarray (n, n)
        Spatial weights matrix
    predictor_names : list of str
        Names of predictors
    verbose : bool
        Print results

    Returns
    -------
    dict
        Model results including:
        - 'rho': Spatial autocorrelation coefficient
        - 'coefficients': Regression coefficients
        - 'log_likelihood': Model log-likelihood
        - 'AIC': Akaike Information Criterion
        - 'pseudo_r2': Pseudo R-squared

    Notes
    -----
    This is a simplified implementation. For production use, consider
    using the `spreg` or `pysal` libraries which provide full ML estimation.

    References
    ----------
    Anselin, L. (1988). Spatial Econometrics: Methods and Models.
    Kluwer Academic Publishers.
    """
    from sklearn.linear_model import LinearRegression

    n = len(y)

    # Spatially lagged y
    Wy = W @ y

    # Augmented design matrix: [Wy, X]
    X_aug = np.column_stack([Wy, X])

    # Fit OLS on augmented model
    model = LinearRegression().fit(X_aug, y)

    rho = model.coef_[0]  # Spatial lag coefficient
    betas = model.coef_[1:]  # Predictor coefficients

    y_pred = model.predict(X_aug)
    residuals = y - y_pred

    # Model fit statistics
    ssr = np.sum(residuals ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    pseudo_r2 = 1 - (ssr / sst)

    # Log-likelihood (simplified)
    sigma2 = ssr / n
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    # AIC
    k = X.shape[1] + 2  # Parameters: rho + betas + sigma²
    AIC = 2 * k - 2 * log_likelihood

    results = {
        'rho': rho,
        'rho_interpretation': 'Positive' if rho > 0 else 'Negative' if rho < 0 else 'Zero',
        'coefficients': dict(zip(predictor_names, betas)),
        'intercept': model.intercept_,
        'pseudo_r2': pseudo_r2,
        'log_likelihood': log_likelihood,
        'AIC': AIC,
        'n_obs': n,
        'residuals': residuals
    }

    if verbose:
        print("\n" + "=" * 70)
        print("SPATIAL LAG MODEL (SAR)")
        print("=" * 70)
        print(f"\nModel: y = ρWy + Xβ + ε")
        print(f"\nSpatial lag coefficient (ρ): {rho:.4f}")
        if abs(rho) > 0.1:
            print(f"  → {'Positive' if rho > 0 else 'Negative'} spatial dependence detected")
        print(f"\nPseudo R²: {pseudo_r2:.3f}")
        print(f"AIC: {AIC:.2f}")
        print(f"\nCoefficients:")
        for name, beta in zip(predictor_names, betas):
            print(f"  {name:<20}: {beta:>10.4f}")

    return results


def compare_ols_vs_spatial(
    X: np.ndarray,
    y: np.ndarray,
    coordinates: np.ndarray,
    predictor_names: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare OLS regression vs. spatial models.

    Fits both OLS and spatial lag models, tests residuals for autocorrelation,
    and recommends which model to use.

    Parameters
    ----------
    X : ndarray
        Predictor matrix
    y : ndarray
        Response variable
    coordinates : ndarray
        Spatial coordinates (n, 2)
    predictor_names : list of str
        Predictor names
    verbose : bool
        Print comparison

    Returns
    -------
    dict
        Comparison results and recommendation

    Examples
    --------
    >>> results = compare_ols_vs_spatial(X, y, coords, predictor_names)
    >>> if results['recommend_spatial']:
    ...     print("Use spatial model!")
    >>> else:
    ...     print("OLS is adequate.")
    """
    from sklearn.linear_model import LinearRegression

    # Fit OLS
    ols_model = LinearRegression().fit(X, y)
    ols_residuals = y - ols_model.predict(X)
    ols_r2 = ols_model.score(X, y)

    # Test OLS residuals for spatial autocorrelation
    ols_autocorr = compute_morans_i(
        ols_residuals,
        coordinates,
        verbose=False
    )

    # Create spatial weights
    W = _create_spatial_weights(coordinates)

    # Fit Spatial Lag Model
    sar_model = fit_spatial_lag_model(
        X, y, W,
        predictor_names,
        verbose=False
    )

    # Compare AIC (lower is better)
    ols_aic = _compute_aic_ols(X, y, ols_residuals)
    delta_aic = sar_model['AIC'] - ols_aic

    # Recommendation
    recommend_spatial = (ols_autocorr['significant'] or delta_aic < -2)

    results = {
        'ols_r2': ols_r2,
        'ols_aic': ols_aic,
        'ols_morans_i': ols_autocorr['I'],
        'ols_autocorr_p': ols_autocorr['p_value'],
        'ols_autocorr_significant': ols_autocorr['significant'],

        'sar_pseudo_r2': sar_model['pseudo_r2'],
        'sar_aic': sar_model['AIC'],
        'sar_rho': sar_model['rho'],

        'delta_aic': delta_aic,
        'recommend_spatial': recommend_spatial,
        'reason': _get_recommendation_reason(ols_autocorr, delta_aic)
    }

    if verbose:
        print("\n" + "=" * 70)
        print("OLS vs. SPATIAL MODEL COMPARISON")
        print("=" * 70)

        print(f"\nOrdinary Least Squares (OLS):")
        print(f"  R² = {ols_r2:.3f}")
        print(f"  AIC = {ols_aic:.2f}")
        print(f"  Moran's I (residuals) = {ols_autocorr['I']:.4f} (p = {ols_autocorr['p_value']:.4f})")
        if ols_autocorr['significant']:
            print(f"  ⚠ Residuals show spatial autocorrelation!")

        print(f"\nSpatial Lag Model (SAR):")
        print(f"  Pseudo R² = {sar_model['pseudo_r2']:.3f}")
        print(f"  AIC = {sar_model['AIC']:.2f}")
        print(f"  ρ (spatial lag) = {sar_model['rho']:.4f}")

        print(f"\nModel Comparison:")
        print(f"  ΔAIC = {delta_aic:.2f} (SAR - OLS)")
        if delta_aic < -2:
            print(f"  → SAR substantially better (ΔAIC < -2)")
        elif delta_aic < 0:
            print(f"  → SAR slightly better")
        else:
            print(f"  → OLS adequate")

        print(f"\n{'RECOMMENDATION:'}")
        if recommend_spatial:
            print(f"  ✓ Use SPATIAL model (SAR or SEM)")
            print(f"    {results['reason']}")
        else:
            print(f"  ✓ OLS is adequate")
            print(f"    {results['reason']}")

    return results


def _compute_aic_ols(X, y, residuals):
    """Compute AIC for OLS model."""
    n = len(y)
    k = X.shape[1] + 2  # Parameters: betas + intercept + sigma²
    ssr = np.sum(residuals ** 2)
    sigma2 = ssr / n
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    return 2 * k - 2 * log_likelihood


def _get_recommendation_reason(autocorr_result, delta_aic):
    """Generate recommendation reason string."""
    reasons = []
    if autocorr_result['significant']:
        reasons.append("OLS residuals show significant spatial autocorrelation")
    if delta_aic < -2:
        reasons.append(f"SAR has substantially lower AIC (ΔAIC = {delta_aic:.2f})")

    if reasons:
        return "; ".join(reasons)
    else:
        return "No spatial autocorrelation detected, OLS assumptions satisfied"


# Main execution
if __name__ == "__main__":
    print("Spatial Statistics Module")
    print("Use to test for and account for spatial autocorrelation")
    print("in gridded multivariate analysis.")
