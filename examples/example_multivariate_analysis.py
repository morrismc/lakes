"""
Example script demonstrating multivariate analysis functions.

This shows how to use individual components of the multivariate analysis
to address specific research questions about controls on lake density.
"""

import numpy as np
import pandas as pd
from lake_analysis import (
    # Data loading
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    # Multivariate analysis functions
    prepare_multivariate_dataset,
    compute_correlation_matrix,
    compute_partial_correlation,
    run_pca_analysis,
    variance_partitioning,
    run_multivariate_regression,
    # Visualization functions
    plot_correlation_matrix,
    plot_pca_biplot,
    plot_variance_partitioning,
    plot_variable_importance
)

# ==============================================================================
# EXAMPLE 1: Load and prepare data
# ==============================================================================

print("=" * 80)
print("EXAMPLE 1: DATA PREPARATION")
print("=" * 80)

# Load lakes
lakes = load_conus_lake_data()
lakes_gdf = convert_lakes_to_gdf(lakes)

# Classify by glacial extent
boundaries = load_all_glacial_boundaries()
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

# Prepare multivariate dataset
data_clean, column_mapping = prepare_multivariate_dataset(
    lakes_classified,
    response_var='area',
    min_lake_area=0.01,
    max_lake_area=20000,  # Exclude Great Lakes
    verbose=True
)

print(f"\nCleaned dataset: {len(data_clean):,} lakes")
print(f"Variables: {list(data_clean.columns)}")

# ==============================================================================
# EXAMPLE 2: Correlation analysis
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: CORRELATION MATRIX")
print("=" * 80)

# Compute Spearman correlation matrix
corr_matrix = compute_correlation_matrix(data_clean, method='spearman')

print("\nCorrelation matrix:")
print(corr_matrix.round(3))

# Plot correlation heatmap
fig, ax = plot_correlation_matrix(
    corr_matrix,
    figsize=(10, 8),
    save_path='output/example_correlation_matrix.png'
)

# Find strongest correlations with glacial_stage
if 'glacial_stage_encoded' in corr_matrix.index:
    glacial_corrs = corr_matrix.loc['glacial_stage_encoded']
    print("\nCorrelations with glaciation:")
    for var, corr in glacial_corrs.items():
        if var != 'glacial_stage_encoded':
            print(f"  {var:15s}: ρ = {corr:6.3f}")

# ==============================================================================
# EXAMPLE 3: Partial correlation
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: PARTIAL CORRELATION")
print("=" * 80)

# Example: Correlation between glaciation and lake area,
# controlling for elevation and aridity

if all(v in data_clean.columns for v in ['glacial_stage_encoded', 'area', 'elevation', 'aridity']):
    partial_corr, p_value = compute_partial_correlation(
        data_clean,
        var1='glacial_stage_encoded',
        var2='area',
        control_vars=['elevation', 'aridity']
    )

    print(f"\nPartial correlation between glaciation and lake area,")
    print(f"controlling for elevation and aridity:")
    print(f"  ρ_partial = {partial_corr:.3f}, p = {p_value:.4f}")

    # Compare with zero-order correlation
    zero_order = corr_matrix.loc['glacial_stage_encoded', 'area']
    print(f"\nFor comparison, zero-order correlation:")
    print(f"  ρ = {zero_order:.3f}")

    if abs(partial_corr) < abs(zero_order):
        print(f"\n→ Controlling for elevation and aridity REDUCES the correlation")
        print(f"  This suggests elevation/aridity partially explain the relationship")
    else:
        print(f"\n→ Controlling for elevation and aridity does NOT reduce the correlation")
        print(f"  This suggests a direct glaciation effect independent of these variables")

# ==============================================================================
# EXAMPLE 4: Principal Component Analysis
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: PRINCIPAL COMPONENT ANALYSIS")
print("=" * 80)

# Run PCA
pca_results = run_pca_analysis(
    data_clean,
    response_var='area',
    n_components=3,
    verbose=True
)

print(f"\nExplained variance:")
for i, var in enumerate(pca_results['explained_variance']):
    print(f"  PC{i+1}: {100*var:.1f}%")

print(f"\nCumulative variance:")
cumvar = np.cumsum(pca_results['explained_variance']) * 100
for i, cv in enumerate(cumvar):
    print(f"  PC1-{i+1}: {cv:.1f}%")

# Show loadings on PC1
print(f"\nPC1 loadings (dominant environmental gradient):")
loadings = pca_results['loadings'][:, 0]
var_names = pca_results['variable_names']
for var, loading in zip(var_names, loadings):
    print(f"  {var:15s}: {loading:6.3f}")

# Plot PCA biplot
if 'glacial_stage' in data_clean.columns:
    pca_results['glacial_stage'] = data_clean['glacial_stage'].values

fig, axes = plot_pca_biplot(
    pca_results,
    figsize=(12, 10),
    save_path='output/example_pca_biplot.png'
)

# ==============================================================================
# EXAMPLE 5: Variance partitioning
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: VARIANCE PARTITIONING")
print("=" * 80)

# Partition variance to separate pure effects
vp_results = variance_partitioning(
    data_clean,
    response_var='area',
    verbose=True
)

if vp_results is not None:
    print(f"\nVariance partitioning results:")
    print(f"  Total R² = {vp_results['r2_total']:.3f}")
    print(f"\n  Pure glaciation:  {vp_results['pure_glacial']:.3f}")
    print(f"  Pure climate:     {vp_results['pure_climate']:.3f}")
    print(f"  Pure topography:  {vp_results['pure_topo']:.3f}")
    print(f"  Shared variance:  {vp_results['shared']:.3f}")

    # Convert to percentages of explained variance
    r2_tot = vp_results['r2_total']
    if r2_tot > 0:
        print(f"\n  As % of explained variance:")
        print(f"    Pure glaciation:  {100 * vp_results['pure_glacial'] / r2_tot:5.1f}%")
        print(f"    Pure climate:     {100 * vp_results['pure_climate'] / r2_tot:5.1f}%")
        print(f"    Pure topography:  {100 * vp_results['pure_topo'] / r2_tot:5.1f}%")
        print(f"    Shared:           {100 * vp_results['shared'] / r2_tot:5.1f}%")

    # Plot variance partitioning
    fig, ax = plot_variance_partitioning(
        vp_results,
        figsize=(10, 8),
        save_path='output/example_variance_partitioning.png'
    )

# ==============================================================================
# EXAMPLE 6: Multiple regression
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: MULTIPLE REGRESSION")
print("=" * 80)

# Fit multiple regression model
reg_results = run_multivariate_regression(
    data_clean,
    response_var='area',
    verbose=True
)

print(f"\nMultiple regression results:")
print(f"  R² = {reg_results['r2']:.3f}")
print(f"  Adjusted R² = {reg_results['r2_adj']:.3f}")
print(f"  F-statistic = {reg_results['f_statistic']:.2f}, p = {reg_results['f_pvalue']:.4e}")

print(f"\n  Standardized coefficients (β):")
predictors = reg_results['predictors']
coefs = reg_results['coefficients']
p_vals = reg_results['p_values']

for pred, coef, p in zip(predictors, coefs, p_vals):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {pred:20s}: β = {coef:6.3f} {sig:3s} (p = {p:.4f})")

# Plot variable importance
fig, ax = plot_variable_importance(
    reg_results,
    figsize=(10, 6),
    save_path='output/example_variable_importance.png'
)

# ==============================================================================
# EXAMPLE 7: Answering specific research questions
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 7: RESEARCH QUESTIONS")
print("=" * 80)

print("""
QUESTION 1: Is glaciation the primary control on lake density?

ANSWER: Check variance partitioning results above.
- If pure_glacial > pure_climate and pure_glacial > pure_topo:
  YES, glaciation is the primary control
- If shared variance is large: Glaciation effects are confounded with
  climate/topography (e.g., glaciated areas are also higher elevation)

QUESTION 2: What is the dominant environmental gradient?

ANSWER: Check PC1 loadings from PCA.
- Positive loadings: Variables that increase together along PC1
- Negative loadings: Variables that decrease along PC1
- If PC1 has high glacial_stage loading: Glaciation defines the primary axis

QUESTION 3: After controlling for everything, what still matters?

ANSWER: Check multiple regression standardized coefficients (β).
- These coefficients represent the effect of each variable AFTER controlling
  for all others
- Compare |β| values: larger magnitude = stronger independent effect
- Check p-values for statistical significance

QUESTION 4: Is the glaciation-lake relationship direct or mediated?

ANSWER: Compare zero-order correlation vs partial correlation.
- If partial correlation (controlling for climate/topo) is much smaller:
  The relationship is MEDIATED by climate/topography
- If partial correlation remains strong:
  The relationship is DIRECT, independent of other variables
""")

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nFigures saved to: output/example_*.png")
print("\nFor a complete analysis pipeline, see: run_multivariate_analysis.py")
