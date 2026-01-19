"""
Multivariate statistical analysis to disentangle controls on lake density.

This module addresses the question: "Is glaciation the PRIMARY control on lake
density, or are climate and topography equally/more important?"

Methods:
- Correlation matrix and network
- Principal Component Analysis (PCA)
- Multiple linear regression
- Variance partitioning (glaciation vs climate vs topography)
- Partial correlation (controlling for confounders)
- Relative importance analysis

Scientific Question:
After controlling for elevation, slope, relief, aridity, and precipitation,
does glaciation STILL significantly affect lake density?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings


def prepare_multivariate_dataset(lakes_gdf, response_var='area', min_lake_area=0.01,
                                max_lake_area=20000, include_glacial=True,
                                min_samples=100, verbose=True):
    """
    Prepare dataset for multivariate analysis with all relevant variables.

    Parameters
    ----------
    lakes_gdf : GeoDataFrame
        Lakes with attributes and glacial classification
    response_var : str
        Response variable name (e.g., 'area', 'log_area')
    min_lake_area : float
        Minimum lake area in km² (exclude small lakes)
    max_lake_area : float
        Maximum lake area in km² (exclude Great Lakes)
    include_glacial : bool
        Include glacial stage as categorical variable
    min_samples : int
        Minimum samples required per glacial stage
    verbose : bool
        Print progress

    Returns
    -------
    tuple
        (DataFrame with clean data, dict with column mapping)
    """
    try:
        from .config import COLS
    except ImportError:
        from config import COLS

    if verbose:
        print("\n" + "=" * 70)
        print("PREPARING MULTIVARIATE DATASET")
        print("=" * 70)

    # Required columns
    vars_continuous = {
        'elevation': COLS.get('elevation', 'Elevation_'),
        'slope': COLS.get('slope', 'Slope'),
        'relief': COLS.get('relief', 'F5km_relief'),
        'aridity': COLS.get('aridity', 'AI'),
        'precipitation': COLS.get('precipitation', 'precip_mm'),
        'area': COLS.get('area', 'AREASQKM'),
    }

    # Check which variables are available
    available_vars = {}
    for var_name, col_name in vars_continuous.items():
        if col_name in lakes_gdf.columns:
            available_vars[var_name] = col_name
        elif verbose:
            print(f"  WARNING: {var_name} column '{col_name}' not found")

    if len(available_vars) < 3:
        raise ValueError(f"Need at least 3 variables for multivariate analysis, found {len(available_vars)}")

    # Filter lakes by area first (on original GeoDataFrame)
    area_col = available_vars.get('area', 'AREASQKM')
    if area_col in lakes_gdf.columns:
        n_before_filter = len(lakes_gdf)
        lakes_filtered = lakes_gdf[
            (lakes_gdf[area_col] >= min_lake_area) &
            (lakes_gdf[area_col] <= max_lake_area)
        ].copy()
        n_after_filter = len(lakes_filtered)
        if verbose:
            print(f"\nLake area filtering:")
            print(f"  Range: {min_lake_area} - {max_lake_area} km²")
            print(f"  Before: {n_before_filter:,} lakes")
            print(f"  After: {n_after_filter:,} lakes")
            print(f"  Removed: {n_before_filter - n_after_filter:,} lakes")
    else:
        lakes_filtered = lakes_gdf.copy()

    # Extract variables from filtered data
    data = pd.DataFrame()
    for var_name, col_name in available_vars.items():
        data[var_name] = lakes_filtered[col_name].values

    # Add glacial stage if available and requested
    if include_glacial and 'glacial_stage' in lakes_filtered.columns:
        data['glacial_stage'] = lakes_filtered['glacial_stage'].values

        # Filter to well-represented stages
        stage_counts = data['glacial_stage'].value_counts()
        valid_stages = stage_counts[stage_counts >= min_samples].index
        data = data[data['glacial_stage'].isin(valid_stages)].copy()

        if verbose:
            print(f"\nGlacial stages included:")
            for stage in valid_stages:
                n = (data['glacial_stage'] == stage).sum()
                print(f"  {stage}: {n:,} lakes")

    # Remove rows with missing data
    n_before = len(data)
    data = data.dropna()
    n_after = len(data)

    if verbose:
        print(f"\nData preparation:")
        print(f"  Before cleaning: {n_before:,} lakes")
        print(f"  After cleaning: {n_after:,} lakes")
        print(f"  Removed: {n_before - n_after:,} lakes ({100*(n_before-n_after)/n_before:.1f}%)")
        print(f"\nVariables included: {list(available_vars.keys())}")

    # Return data and column mapping
    return data, available_vars


def compute_correlation_matrix(data, method='spearman', verbose=True):
    """
    Compute correlation matrix between all continuous variables.

    Parameters
    ----------
    data : DataFrame
        Dataset with continuous variables
    method : str
        'pearson' or 'spearman'
    verbose : bool
        Print results

    Returns
    -------
    DataFrame
        Correlation matrix
    """
    # Select only continuous variables (exclude glacial_stage if present)
    continuous_cols = [col for col in data.columns if col != 'glacial_stage']

    if method == 'spearman':
        corr_matrix = data[continuous_cols].corr(method='spearman')
    else:
        corr_matrix = data[continuous_cols].corr(method='pearson')

    if verbose:
        print(f"\n{method.capitalize()} Correlation Matrix:")
        print(corr_matrix.round(3))

    return corr_matrix


def compute_partial_correlation(data, var1, var2, control_vars, verbose=True):
    """
    Compute partial correlation between var1 and var2, controlling for control_vars.

    This answers: "What is the correlation between var1 and var2 AFTER removing
    the effects of control_vars?"

    Parameters
    ----------
    data : DataFrame
        Dataset
    var1, var2 : str
        Variables to correlate
    control_vars : list of str
        Variables to control for
    verbose : bool
        Print results

    Returns
    -------
    dict
        Partial correlation coefficient and p-value
    """
    from scipy.stats import linregress

    # Remove control variables from both var1 and var2
    # Fit: var1 ~ control_vars
    X_control = data[control_vars].values
    y1 = data[var1].values
    y2 = data[var2].values

    # Residuals after removing control variable effects
    from sklearn.linear_model import LinearRegression
    model1 = LinearRegression().fit(X_control, y1)
    model2 = LinearRegression().fit(X_control, y2)

    resid1 = y1 - model1.predict(X_control)
    resid2 = y2 - model2.predict(X_control)

    # Correlate residuals
    r, p = stats.spearmanr(resid1, resid2)

    if verbose:
        control_str = ', '.join(control_vars)
        print(f"\nPartial correlation: {var1} ~ {var2} | controlling for [{control_str}]")
        print(f"  r = {r:.3f}, p = {p:.4f}")

    return {'r': r, 'p': p, 'var1': var1, 'var2': var2, 'controls': control_vars}


def run_pca_analysis(data, n_components=None, verbose=True):
    """
    Principal Component Analysis to identify dominant gradients.

    Parameters
    ----------
    data : DataFrame
        Dataset with continuous variables
    n_components : int, optional
        Number of components to compute (default: all)
    verbose : bool
        Print results

    Returns
    -------
    dict
        PCA results including scores, loadings, explained variance
    """
    # Select continuous variables only
    continuous_cols = [col for col in data.columns if col != 'glacial_stage']
    X = data[continuous_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    if n_components is None:
        n_components = len(continuous_cols)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # Loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    if verbose:
        print("\n" + "=" * 70)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 70)
        print(f"\nExplained variance:")
        for i in range(n_components):
            var = pca.explained_variance_ratio_[i] * 100
            cum_var = np.sum(pca.explained_variance_ratio_[:i+1]) * 100
            print(f"  PC{i+1}: {var:.1f}% (cumulative: {cum_var:.1f}%)")

        print(f"\nLoadings (PC1 and PC2):")
        loading_df = pd.DataFrame(
            loadings[:, :2],
            index=continuous_cols,
            columns=['PC1', 'PC2']
        )
        print(loading_df.round(3))

    results = {
        'pca': pca,
        'scores': scores,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_ratio_,
        'variable_names': continuous_cols,
        'scaler': scaler
    }

    # Add glacial stage if available
    if 'glacial_stage' in data.columns:
        results['glacial_stage'] = data['glacial_stage'].values

    return results


def variance_partitioning(data, response_var='area', verbose=True):
    """
    Variance partitioning to determine relative importance of variable groups.

    Decomposes variance in lake density/area into:
    - Pure glaciation effect
    - Pure climate effect (aridity + precipitation)
    - Pure topography effect (elevation + slope + relief)
    - Shared effects

    Parameters
    ----------
    data : DataFrame
        Dataset with all variables including glacial_stage
    response_var : str
        Response variable (typically 'area' for lake density proxy)
    verbose : bool
        Print results

    Returns
    -------
    dict
        Variance partition results (R² for each component)
    """
    from sklearn.preprocessing import LabelEncoder

    if 'glacial_stage' not in data.columns:
        print("WARNING: glacial_stage not found, cannot partition variance")
        return None

    # Define variable groups
    glacial_vars = ['glacial_stage']
    climate_vars = [v for v in ['aridity', 'precipitation'] if v in data.columns]
    topo_vars = [v for v in ['elevation', 'slope', 'relief'] if v in data.columns]

    if len(climate_vars) == 0 or len(topo_vars) == 0:
        print("WARNING: Need both climate and topography variables")
        return None

    # Encode glacial stage
    le = LabelEncoder()
    data_encoded = data.copy()
    data_encoded['glacial_stage_encoded'] = le.fit_transform(data['glacial_stage'])

    # Response variable
    y = data_encoded[response_var].values

    # Fit models
    def fit_and_score(predictors):
        X = data_encoded[predictors].values
        model = LinearRegression().fit(X, y)
        return model.score(X, y)

    # Individual R²
    r2_glacial = fit_and_score(['glacial_stage_encoded'])
    r2_climate = fit_and_score(climate_vars)
    r2_topo = fit_and_score(topo_vars)

    # Combined R²
    r2_glacial_climate = fit_and_score(['glacial_stage_encoded'] + climate_vars)
    r2_glacial_topo = fit_and_score(['glacial_stage_encoded'] + topo_vars)
    r2_climate_topo = fit_and_score(climate_vars + topo_vars)
    r2_all = fit_and_score(['glacial_stage_encoded'] + climate_vars + topo_vars)

    # Variance partitioning (following Legendre & Legendre method)
    pure_glacial = r2_all - r2_climate_topo
    pure_climate = r2_all - r2_glacial_topo
    pure_topo = r2_all - r2_glacial_climate

    shared_all = r2_all - pure_glacial - pure_climate - pure_topo

    if verbose:
        print("\n" + "=" * 70)
        print("VARIANCE PARTITIONING")
        print("=" * 70)
        print(f"\nResponse variable: {response_var}")
        print(f"\nTotal variance explained (R²): {r2_all:.3f}")
        print(f"\nPure effects:")
        print(f"  Glaciation (after controlling for climate + topo): {pure_glacial:.3f} ({100*pure_glacial/r2_all:.1f}%)")
        print(f"  Climate (after controlling for glaciation + topo): {pure_climate:.3f} ({100*pure_climate/r2_all:.1f}%)")
        print(f"  Topography (after controlling for glaciation + climate): {pure_topo:.3f} ({100*pure_topo/r2_all:.1f}%)")
        print(f"  Shared variance: {shared_all:.3f} ({100*shared_all/r2_all:.1f}%)")

    return {
        'r2_total': r2_all,
        'pure_glacial': pure_glacial,
        'pure_climate': pure_climate,
        'pure_topo': pure_topo,
        'shared': shared_all,
        'r2_glacial': r2_glacial,
        'r2_climate': r2_climate,
        'r2_topo': r2_topo,
    }


def run_multivariate_regression(data, response_var='area', verbose=True):
    """
    Multiple linear regression to quantify relative importance of predictors.

    Parameters
    ----------
    data : DataFrame
        Dataset with all variables
    response_var : str
        Response variable
    verbose : bool
        Print results

    Returns
    -------
    dict
        Regression results including coefficients, R², p-values
    """
    from sklearn.preprocessing import LabelEncoder

    # Prepare predictors
    continuous_vars = [v for v in ['elevation', 'slope', 'relief', 'aridity', 'precipitation']
                      if v in data.columns]

    data_model = data.copy()

    # Encode glacial stage if present
    if 'glacial_stage' in data.columns:
        le = LabelEncoder()
        data_model['glacial_stage_encoded'] = le.fit_transform(data['glacial_stage'])
        predictors = continuous_vars + ['glacial_stage_encoded']
    else:
        predictors = continuous_vars

    # Fit model
    X = data_model[predictors].values
    y = data_model[response_var].values

    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression().fit(X_scaled, y)

    # Results
    r2 = model.score(X_scaled, y)
    coefficients = model.coef_

    # Statistical significance (approximate using t-test)
    n = len(y)
    k = len(predictors)
    residuals = y - model.predict(X_scaled)
    mse = np.sum(residuals**2) / (n - k - 1)
    var_coef = mse * np.diag(np.linalg.inv(X_scaled.T @ X_scaled))
    se_coef = np.sqrt(var_coef)
    t_stats = coefficients / se_coef
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

    if verbose:
        print("\n" + "=" * 70)
        print("MULTIPLE LINEAR REGRESSION")
        print("=" * 70)
        print(f"\nResponse variable: {response_var}")
        print(f"R² = {r2:.3f}")
        print(f"\nStandardized coefficients:")
        for i, pred in enumerate(predictors):
            sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
            print(f"  {pred:25s}: β = {coefficients[i]:7.3f}, p = {p_values[i]:.4f} {sig}")

    results = {
        'model': model,
        'r2': r2,
        'coefficients': coefficients,
        'p_values': p_values,
        'predictors': predictors,
        'scaler': scaler
    }

    return results


def run_complete_multivariate_analysis(lakes_gdf, response_var='area',
                                      min_lake_area=0.01, max_lake_area=20000,
                                      save_figures=True, output_dir=None, verbose=True):
    """
    Complete multivariate analysis pipeline.

    This comprehensive analysis answers: "After controlling for elevation, slope,
    relief, aridity, and precipitation, does glaciation STILL matter?"

    Parameters
    ----------
    lakes_gdf : GeoDataFrame
        Lakes with attributes and glacial classification
    response_var : str
        Response variable name (default: 'area')
    min_lake_area : float
        Minimum lake area in km² (default: 0.01)
    max_lake_area : float
        Maximum lake area in km² (default: 20000, excludes Great Lakes)
    save_figures : bool
        Generate and save visualization figures
    output_dir : str, optional
        Output directory for figures
    verbose : bool
        Print progress

    Returns
    -------
    dict
        All multivariate analysis results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("COMPLETE MULTIVARIATE ANALYSIS")
        print("Disentangling glaciation, climate, and topography effects")
        print("=" * 70)

    results = {}

    # 1. Prepare dataset
    data, column_mapping = prepare_multivariate_dataset(
        lakes_gdf,
        response_var=response_var,
        min_lake_area=min_lake_area,
        max_lake_area=max_lake_area,
        include_glacial=True,
        verbose=verbose
    )
    results['data'] = data
    results['column_mapping'] = column_mapping

    # 2. Correlation matrix
    corr_matrix = compute_correlation_matrix(data, method='spearman', verbose=verbose)
    results['correlation_matrix'] = corr_matrix

    # 3. PCA
    pca_results = run_pca_analysis(data, n_components=3, verbose=verbose)
    results['pca'] = pca_results

    # 4. Variance partitioning
    vp_results = variance_partitioning(data, response_var=response_var, verbose=verbose)
    results['variance_partitioning'] = vp_results

    # 5. Multiple regression
    reg_results = run_multivariate_regression(data, response_var=response_var, verbose=verbose)
    results['regression'] = reg_results

    # 6. Partial correlations
    if verbose:
        print("\n" + "=" * 70)
        print("PARTIAL CORRELATIONS")
        print("(Controlling for confounding variables)")
        print("=" * 70)

    partial_corrs = []

    # Key partial correlations to test
    continuous_vars = [v for v in ['elevation', 'slope', 'relief', 'aridity', 'precipitation']
                      if v in data.columns]

    # Example: aridity ~ area, controlling for elevation + slope
    if 'aridity' in continuous_vars and 'elevation' in continuous_vars:
        control_vars = [v for v in ['elevation', 'slope', 'relief'] if v in continuous_vars and v != 'aridity']
        if len(control_vars) > 0:
            pc = compute_partial_correlation(data, 'aridity', 'area', control_vars, verbose=verbose)
            partial_corrs.append(pc)

    results['partial_correlations'] = partial_corrs

    # 7. Generate figures
    if save_figures:
        try:
            from .visualization import plot_multivariate_summary
            from .config import OUTPUT_DIR
        except ImportError:
            from visualization import plot_multivariate_summary
            from config import OUTPUT_DIR

        if output_dir is None:
            output_dir = OUTPUT_DIR

        # This function will be created in visualization.py
        print(f"\nGenerating multivariate analysis figures...")
        # plot_multivariate_summary(results, save_path=output_dir)

    if verbose:
        print("\n" + "=" * 70)
        print("SCIENTIFIC INTERPRETATION")
        print("=" * 70)

        if vp_results:
            pure_glac = vp_results['pure_glacial']
            pure_clim = vp_results['pure_climate']
            pure_topo = vp_results['pure_topo']
            total = vp_results['r2_total']

            print(f"\nKEY FINDING:")
            if pure_glac / total > 0.3:
                print(f"  ✓ Glaciation is the PRIMARY control ({100*pure_glac/total:.0f}% of explained variance)")
            elif pure_glac / total > 0.15:
                print(f"  → Glaciation is IMPORTANT but not dominant ({100*pure_glac/total:.0f}% of explained variance)")
            else:
                print(f"  ✗ Glaciation is MINOR compared to climate/topography ({100*pure_glac/total:.0f}% of explained variance)")

            print(f"\n  After controlling for climate and topography:")
            print(f"    Glaciation explains {100*pure_glac:.1f}% of lake area variance")
            print(f"  After controlling for glaciation and topography:")
            print(f"    Climate explains {100*pure_clim:.1f}% of lake area variance")
            print(f"  After controlling for glaciation and climate:")
            print(f"    Topography explains {100*pure_topo:.1f}% of lake area variance")

    return results


# Export key functions
__all__ = [
    'prepare_multivariate_dataset',
    'compute_correlation_matrix',
    'compute_partial_correlation',
    'run_pca_analysis',
    'variance_partitioning',
    'run_multivariate_regression',
    'run_complete_multivariate_analysis',
]
