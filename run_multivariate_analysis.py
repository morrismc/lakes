"""
Standalone script to run multivariate analysis of lake density controls.

This addresses the key scientific question: After controlling for climate and
topography, is glaciation still the PRIMARY control on lake density?

Key analyses:
- Correlation matrices (Spearman rank correlation)
- Principal Component Analysis (PCA) to identify dominant environmental gradients
- Variance partitioning to decompose R² into pure and shared effects
- Multiple regression with standardized coefficients
"""

import numpy as np
from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    run_complete_multivariate_analysis
)

print("=" * 80)
print("MULTIVARIATE ANALYSIS: DISENTANGLING CONTROLS ON LAKE DENSITY")
print("=" * 80)
print()
print("SCIENTIFIC QUESTION:")
print("  After controlling for elevation, slope, relief, aridity, and precipitation,")
print("  does glaciation STILL significantly affect lake density?")
print()
print("METHOD:")
print("  Variance partitioning (Legendre & Legendre approach) to decompose R² into:")
print("  - Pure glaciation effect")
print("  - Pure climate effect (aridity + precipitation)")
print("  - Pure topography effect (elevation + slope + relief)")
print("  - Shared variance")
print("=" * 80)

# Load and classify lakes
print("\n1. Loading CONUS lake data...")
lakes = load_conus_lake_data()
print(f"   Loaded {len(lakes):,} lakes")

print("\n2. Converting to GeoDataFrame...")
lakes_gdf = convert_lakes_to_gdf(lakes)

print("\n3. Loading glacial boundaries...")
boundaries = load_all_glacial_boundaries()
print(f"   Loaded boundaries: {list(boundaries.keys())}")

print("\n4. Classifying lakes by glacial extent...")
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

# Check available variables
print("\n5. Checking available variables...")
available_vars = []
for var in ['aridity', 'precipitation', 'elevation', 'slope', 'relief']:
    if var in lakes_classified.columns:
        available_vars.append(var)
        print(f"   ✓ {var}")
    else:
        print(f"   ✗ {var} (not available)")

if len(available_vars) == 0:
    print("\n⚠️  WARNING: No environmental variables found!")
    print("   Multivariate analysis requires climate and topography data.")
    print("   Please ensure lakes have been sampled with raster data.")
    print("\nExiting...")
    exit(1)

# Run multivariate analysis
print("\n6. Running complete multivariate analysis...")
print("   This includes:")
print("   - Data preparation and cleaning")
print("   - Correlation matrix (Spearman)")
print("   - Principal Component Analysis (PCA)")
print("   - Variance partitioning")
print("   - Multiple linear regression")
print()

try:
    results = run_complete_multivariate_analysis(
        lakes_classified,
        response_var='area',   # Use lake area as response
        min_lake_area=0.005,   # Filter small lakes (match min threshold)
        max_lake_area=20000,   # Exclude Great Lakes
        save_figures=True,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Print key results
    print("\n" + "-" * 80)
    print("KEY RESULTS")
    print("-" * 80)

    # Correlation matrix
    if 'correlation_matrix' in results:
        print("\n1. CORRELATION MATRIX")
        corr = results['correlation_matrix']
        print(f"   Variables analyzed: {list(corr.columns)}")

        # Find strongest correlations with glacial_stage_encoded
        if 'glacial_stage_encoded' in corr.index:
            glacial_corrs = corr.loc['glacial_stage_encoded'].drop('glacial_stage_encoded')
            glacial_corrs_sorted = glacial_corrs.abs().sort_values(ascending=False)
            print(f"\n   Strongest correlations with glaciation:")
            for var in glacial_corrs_sorted.index[:3]:
                print(f"     {var}: ρ = {glacial_corrs[var]:.3f}")

    # PCA
    if 'pca' in results:
        pca = results['pca']
        print("\n2. PRINCIPAL COMPONENT ANALYSIS")
        print(f"   PC1 explains {100*pca['explained_variance'][0]:.1f}% of variance")
        print(f"   PC2 explains {100*pca['explained_variance'][1]:.1f}% of variance")
        print(f"   First 3 PCs explain {100*np.sum(pca['explained_variance'][:3]):.1f}% of variance")

        # Show top loadings on PC1
        loadings = pca['loadings'][:, 0]
        var_names = pca['variable_names']
        top_idx = np.argsort(np.abs(loadings))[-3:][::-1]
        print(f"\n   Top 3 variables on PC1 (primary gradient):")
        for idx in top_idx:
            print(f"     {var_names[idx]}: {loadings[idx]:.3f}")

    # Variance partitioning
    if 'variance_partitioning' in results and results['variance_partitioning'] is not None:
        vp = results['variance_partitioning']
        print("\n3. VARIANCE PARTITIONING")
        print(f"   Total R² = {vp['r2_total']:.3f}")

        r2_tot = vp['r2_total']
        if r2_tot > 0:
            pct_glacial = 100 * vp['pure_glacial'] / r2_tot
            pct_climate = 100 * vp['pure_climate'] / r2_tot
            pct_topo = 100 * vp['pure_topo'] / r2_tot
            pct_shared = 100 * vp['shared'] / r2_tot

            print(f"\n   Pure glaciation effect:   {pct_glacial:5.1f}% of explained variance")
            print(f"   Pure climate effect:      {pct_climate:5.1f}% of explained variance")
            print(f"   Pure topography effect:   {pct_topo:5.1f}% of explained variance")
            print(f"   Shared variance:          {pct_shared:5.1f}% of explained variance")

            # Interpretation
            print(f"\n   INTERPRETATION:")
            if pct_glacial > pct_climate and pct_glacial > pct_topo:
                print(f"   → Glaciation is the PRIMARY control on lake density")
                print(f"     (explains {pct_glacial:.1f}% after controlling for climate + topography)")
            elif pct_climate > pct_glacial and pct_climate > pct_topo:
                print(f"   → Climate is the PRIMARY control on lake density")
                print(f"     (explains {pct_climate:.1f}% after controlling for glaciation + topography)")
            elif pct_topo > pct_glacial and pct_topo > pct_climate:
                print(f"   → Topography is the PRIMARY control on lake density")
                print(f"     (explains {pct_topo:.1f}% after controlling for glaciation + climate)")
            else:
                print(f"   → No single factor dominates; effects are balanced")

            if pct_shared > 50:
                print(f"\n   ⚠️  Large shared variance ({pct_shared:.1f}%) indicates strong")
                print(f"       collinearity between glaciation, climate, and topography")

    # Multiple regression
    if 'regression' in results:
        reg = results['regression']
        print("\n4. MULTIPLE REGRESSION")
        print(f"   R² = {reg['r2']:.3f}")
        print(f"   Adjusted R² = {reg['r2_adj']:.3f}")

        # Show standardized coefficients (sorted by absolute value)
        predictors = reg['predictors']
        coefs = reg['coefficients']
        p_vals = reg['p_values']

        sort_idx = np.argsort(np.abs(coefs))[::-1]

        print(f"\n   Standardized coefficients (β):")
        for idx in sort_idx:
            sig = "***" if p_vals[idx] < 0.001 else "**" if p_vals[idx] < 0.01 else "*" if p_vals[idx] < 0.05 else "ns"
            print(f"     {predictors[idx]:20s}: β = {coefs[idx]:6.3f} {sig:3s} (p = {p_vals[idx]:.4f})")

    # Output files
    print("\n" + "-" * 80)
    print("OUTPUTS")
    print("-" * 80)
    print("\nFigures saved to: F:\\Lakes\\Analysis\\outputs\\")
    print("  - multivariate_correlation_matrix.png")
    print("  - multivariate_pca_biplot.png")
    print("  - multivariate_variance_partitioning.png")
    print("  - multivariate_variable_importance.png")

    print("\n" + "=" * 80)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 80)

    print("""
CO-AUTHOR QUESTIONS ADDRESSED:

Q: "Is glaciation just a proxy for climate or topography?"
A: Use variance partitioning results above to determine if glaciation has
   a PURE effect independent of climate and topography.

Q: "What is the dominant environmental gradient controlling lake density?"
A: Check PCA results - PC1 loadings show which combination of variables
   creates the strongest pattern in the data.

Q: "After controlling for all other variables, what matters most?"
A: Multiple regression standardized coefficients (β) are directly comparable.
   Larger |β| = stronger independent effect.

LIMITATIONS:
- This analysis assumes LINEAR relationships between variables
- Variance partitioning can produce negative values if variables are highly collinear
- Causality cannot be inferred from correlations alone
- Spatial autocorrelation not accounted for

NEXT STEPS:
- Consider spatial regression models (SAR/CAR) if autocorrelation detected
- Test for non-linear effects with GAMs or polynomial regression
- Bootstrap confidence intervals for variance partitioning estimates
""")

    print("=" * 80)
    print("SESSION COMPLETE")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR during analysis:")
    print(f"   {type(e).__name__}: {e}")
    print(f"\nPlease check that:")
    print(f"  1. Lakes have all required environmental variables")
    print(f"  2. Sufficient sample size (>100 lakes recommended)")
    print(f"  3. Variables have sufficient variation")
    import traceback
    traceback.print_exc()
