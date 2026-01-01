"""
Spatial Scaling Analysis Module
================================

Analyzes geographic patterns in lake distributions including:
- Latitudinal gradients in lake density and size
- Longitudinal patterns (east-west gradients)
- Elevation-dependent scaling
- Glacial vs non-glacial size distributions
- Regional breakdowns

Scientific Questions:
1. Does lake size scale with latitude (larger lakes at higher latitudes)?
2. Are there east-west gradients in lake properties?
3. How does glaciation affect lake size distributions?
4. What are the spatial autocorrelation patterns in lake density?
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Handle imports for both package and direct execution
try:
    from .config import COLS, OUTPUT_DIR, ensure_output_dir
except ImportError:
    from config import COLS, OUTPUT_DIR, ensure_output_dir


# ============================================================================
# LATITUDINAL SCALING
# ============================================================================

def analyze_latitudinal_scaling(lake_df, lat_col=None, area_col=None,
                                  lat_bins=None, verbose=True):
    """
    Analyze how lake properties vary with latitude.

    Tests whether lake size, density, or power law exponent changes
    systematically from south to north.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with latitude and area columns
    lat_col : str, optional
        Latitude column name
    area_col : str, optional
        Lake area column name
    lat_bins : array-like, optional
        Latitude bin edges (default: 5° bins from 25-50°N)
    verbose : bool
        Print results

    Returns
    -------
    dict
        Contains:
        - binned_stats: DataFrame with statistics by latitude band
        - correlation: Size-latitude correlation
        - regression: Linear regression results
        - hypothesis_test: Test for latitudinal gradient
    """
    if lat_col is None:
        lat_col = COLS.get('lat', 'Latitude')
    if area_col is None:
        area_col = COLS.get('area', 'AreaSqKm')

    if lat_col not in lake_df.columns:
        return {'error': f'Latitude column {lat_col} not found'}

    if lat_bins is None:
        lat_bins = np.arange(25, 52, 3)  # 3° bands from 25°N to 51°N

    if verbose:
        print("\n" + "=" * 60)
        print("LATITUDINAL SCALING ANALYSIS")
        print("=" * 60)

    results = {
        'lat_col': lat_col,
        'area_col': area_col,
        'lat_bins': lat_bins
    }

    # Bin lakes by latitude
    lake_df = lake_df.copy()
    lake_df['lat_bin'] = pd.cut(lake_df[lat_col], bins=lat_bins)

    # Compute statistics by latitude band
    binned_stats = []
    for i in range(len(lat_bins) - 1):
        lat_low, lat_high = lat_bins[i], lat_bins[i+1]
        lat_mid = (lat_low + lat_high) / 2

        mask = (lake_df[lat_col] >= lat_low) & (lake_df[lat_col] < lat_high)
        band_lakes = lake_df[mask]

        n_lakes = len(band_lakes)
        if n_lakes > 0:
            areas = band_lakes[area_col].dropna().values
            areas = areas[areas > 0]

            # Power law exponent (MLE)
            if len(areas) >= 30:
                xmin = 0.01  # Lower threshold
                tail = areas[areas >= xmin]
                if len(tail) > 10:
                    alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
                    alpha_se = (alpha - 1) / np.sqrt(len(tail))
                else:
                    alpha, alpha_se = np.nan, np.nan
            else:
                alpha, alpha_se = np.nan, np.nan

            binned_stats.append({
                'lat_band': f'{lat_low:.0f}-{lat_high:.0f}°N',
                'lat_mid': lat_mid,
                'n_lakes': n_lakes,
                'mean_area_km2': np.mean(areas) if len(areas) > 0 else np.nan,
                'median_area_km2': np.median(areas) if len(areas) > 0 else np.nan,
                'total_area_km2': np.sum(areas) if len(areas) > 0 else 0,
                'max_area_km2': np.max(areas) if len(areas) > 0 else np.nan,
                'alpha': alpha,
                'alpha_se': alpha_se,
            })

    binned_df = pd.DataFrame(binned_stats)
    results['binned_stats'] = binned_df

    # Overall correlation: lake size vs latitude
    valid_mask = lake_df[area_col].notna() & (lake_df[area_col] > 0)
    areas = lake_df.loc[valid_mask, area_col].values
    lats = lake_df.loc[valid_mask, lat_col].values

    # Use log(area) for correlation (more appropriate for power law)
    log_areas = np.log10(areas)
    corr, p_val = stats.pearsonr(lats, log_areas)

    results['correlation'] = {
        'r': corr,
        'p_value': p_val,
        'interpretation': 'larger lakes at higher latitudes' if corr > 0 else 'smaller lakes at higher latitudes'
    }

    # Linear regression: log(area) ~ latitude
    slope, intercept, r, p, se = stats.linregress(lats, log_areas)
    results['regression'] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r**2,
        'p_value': p,
        'slope_se': se
    }

    # Test: Does alpha vary with latitude?
    valid_alpha = binned_df.dropna(subset=['alpha'])
    if len(valid_alpha) >= 3:
        alpha_lat_corr, alpha_lat_p = stats.pearsonr(
            valid_alpha['lat_mid'], valid_alpha['alpha']
        )
        results['alpha_latitude_test'] = {
            'r': alpha_lat_corr,
            'p_value': alpha_lat_p,
            'significant': alpha_lat_p < 0.05,
            'interpretation': 'α varies with latitude' if alpha_lat_p < 0.05 else 'α constant across latitudes'
        }

    if verbose:
        print(f"\nAnalyzed {len(lake_df):,} lakes across {len(lat_bins)-1} latitude bands")
        print(f"\nLog(Area) ~ Latitude correlation:")
        print(f"  r = {corr:.4f}, p = {p_val:.4e}")
        print(f"  → {results['correlation']['interpretation']}")
        print(f"\nRegression: log₁₀(Area) = {slope:.4f} × Lat + {intercept:.2f}")
        print(f"  R² = {r**2:.4f}")

        if 'alpha_latitude_test' in results:
            print(f"\nPower law exponent (α) vs latitude:")
            print(f"  r = {results['alpha_latitude_test']['r']:.4f}")
            print(f"  → {results['alpha_latitude_test']['interpretation']}")

    return results


def analyze_longitudinal_scaling(lake_df, lon_col=None, area_col=None,
                                   lon_bins=None, verbose=True):
    """
    Analyze how lake properties vary with longitude (east-west gradient).

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with longitude and area columns
    lon_col : str, optional
        Longitude column name
    area_col : str, optional
        Lake area column name
    lon_bins : array-like, optional
        Longitude bin edges (default: 5° bins)
    verbose : bool
        Print results

    Returns
    -------
    dict
        Contains binned statistics, correlation, and regression results
    """
    if lon_col is None:
        lon_col = COLS.get('lon', 'Longitude')
    if area_col is None:
        area_col = COLS.get('area', 'AreaSqKm')

    if lon_col not in lake_df.columns:
        return {'error': f'Longitude column {lon_col} not found'}

    if lon_bins is None:
        # US spans roughly -125 to -65
        lon_bins = np.arange(-130, -60, 5)

    if verbose:
        print("\n" + "=" * 60)
        print("LONGITUDINAL SCALING ANALYSIS")
        print("=" * 60)

    results = {'lon_col': lon_col, 'area_col': area_col}

    # Bin lakes by longitude
    lake_df = lake_df.copy()

    # Compute statistics by longitude band
    binned_stats = []
    for i in range(len(lon_bins) - 1):
        lon_low, lon_high = lon_bins[i], lon_bins[i+1]
        lon_mid = (lon_low + lon_high) / 2

        mask = (lake_df[lon_col] >= lon_low) & (lake_df[lon_col] < lon_high)
        band_lakes = lake_df[mask]

        n_lakes = len(band_lakes)
        if n_lakes > 0:
            areas = band_lakes[area_col].dropna().values
            areas = areas[areas > 0]

            binned_stats.append({
                'lon_band': f'{lon_low:.0f} to {lon_high:.0f}°',
                'lon_mid': lon_mid,
                'n_lakes': n_lakes,
                'mean_area_km2': np.mean(areas) if len(areas) > 0 else np.nan,
                'median_area_km2': np.median(areas) if len(areas) > 0 else np.nan,
                'total_area_km2': np.sum(areas) if len(areas) > 0 else 0,
            })

    binned_df = pd.DataFrame(binned_stats)
    results['binned_stats'] = binned_df

    # Correlation
    valid_mask = lake_df[area_col].notna() & (lake_df[area_col] > 0)
    areas = lake_df.loc[valid_mask, area_col].values
    lons = lake_df.loc[valid_mask, lon_col].values

    log_areas = np.log10(areas)
    corr, p_val = stats.pearsonr(lons, log_areas)

    results['correlation'] = {
        'r': corr,
        'p_value': p_val,
        'interpretation': 'larger lakes in the east' if corr > 0 else 'larger lakes in the west'
    }

    if verbose:
        print(f"\nAnalyzed {len(lake_df):,} lakes across {len(lon_bins)-1} longitude bands")
        print(f"\nLog(Area) ~ Longitude correlation:")
        print(f"  r = {corr:.4f}, p = {p_val:.4e}")
        print(f"  → {results['correlation']['interpretation']}")

    return results


# ============================================================================
# GLACIAL VS NON-GLACIAL COMPARISON
# ============================================================================

def compare_glacial_vs_nonglacial_scaling(lake_gdf, area_col=None,
                                           xmin_threshold=0.01, verbose=True):
    """
    Compare lake size distributions between glaciated and non-glaciated regions.

    Tests whether:
    1. Mean/median lake sizes differ
    2. Power law exponents differ
    3. Size distributions have different shapes

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with 'glacial_stage' column
    area_col : str, optional
        Lake area column
    xmin_threshold : float
        Minimum area for power law fitting
    verbose : bool
        Print results

    Returns
    -------
    dict
        Comprehensive comparison results including statistical tests
    """
    if area_col is None:
        area_col = COLS.get('area', 'AreaSqKm')

    if 'glacial_stage' not in lake_gdf.columns:
        return {'error': 'glacial_stage column not found'}

    if verbose:
        print("\n" + "=" * 60)
        print("GLACIAL VS NON-GLACIAL SIZE SCALING COMPARISON")
        print("=" * 60)

    # Separate glaciated and non-glaciated lakes
    glaciated_mask = ~lake_gdf['glacial_stage'].str.lower().str.contains(
        'driftless|never|unglaciated', na=True
    )

    glaciated = lake_gdf[glaciated_mask]
    non_glaciated = lake_gdf[~glaciated_mask]

    results = {
        'n_glaciated': len(glaciated),
        'n_non_glaciated': len(non_glaciated),
    }

    if verbose:
        print(f"\nGlaciated lakes: {len(glaciated):,}")
        print(f"Non-glaciated lakes: {len(non_glaciated):,}")

    # Get areas
    glac_areas = glaciated[area_col].dropna().values
    glac_areas = glac_areas[glac_areas > 0]

    nonglac_areas = non_glaciated[area_col].dropna().values
    nonglac_areas = nonglac_areas[nonglac_areas > 0]

    # Basic statistics
    results['glaciated_stats'] = {
        'n': len(glac_areas),
        'mean': np.mean(glac_areas),
        'median': np.median(glac_areas),
        'std': np.std(glac_areas),
        'total_area': np.sum(glac_areas),
        'max': np.max(glac_areas)
    }

    results['non_glaciated_stats'] = {
        'n': len(nonglac_areas),
        'mean': np.mean(nonglac_areas),
        'median': np.median(nonglac_areas),
        'std': np.std(nonglac_areas),
        'total_area': np.sum(nonglac_areas),
        'max': np.max(nonglac_areas)
    }

    # Statistical tests
    # 1. Mann-Whitney U test (non-parametric)
    if len(glac_areas) > 10 and len(nonglac_areas) > 10:
        u_stat, mw_p = stats.mannwhitneyu(glac_areas, nonglac_areas, alternative='two-sided')
        results['mann_whitney_test'] = {
            'U_statistic': u_stat,
            'p_value': mw_p,
            'significant': mw_p < 0.05,
            'interpretation': 'Size distributions differ significantly' if mw_p < 0.05 else 'No significant difference'
        }

        if verbose:
            print(f"\nMann-Whitney U test:")
            print(f"  U = {u_stat:.0f}, p = {mw_p:.4e}")
            print(f"  → {results['mann_whitney_test']['interpretation']}")

    # 2. Kolmogorov-Smirnov test (distribution shape)
    if len(glac_areas) > 10 and len(nonglac_areas) > 10:
        ks_stat, ks_p = stats.ks_2samp(glac_areas, nonglac_areas)
        results['ks_test'] = {
            'D_statistic': ks_stat,
            'p_value': ks_p,
            'significant': ks_p < 0.05,
            'interpretation': 'Distribution shapes differ' if ks_p < 0.05 else 'Similar distribution shapes'
        }

        if verbose:
            print(f"\nKolmogorov-Smirnov test:")
            print(f"  D = {ks_stat:.4f}, p = {ks_p:.4e}")
            print(f"  → {results['ks_test']['interpretation']}")

    # 3. Power law exponent comparison
    def fit_alpha(areas, xmin):
        tail = areas[areas >= xmin]
        if len(tail) < 30:
            return np.nan, np.nan
        alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
        se = (alpha - 1) / np.sqrt(len(tail))
        return alpha, se

    glac_alpha, glac_se = fit_alpha(glac_areas, xmin_threshold)
    nonglac_alpha, nonglac_se = fit_alpha(nonglac_areas, xmin_threshold)

    results['power_law_comparison'] = {
        'glaciated_alpha': glac_alpha,
        'glaciated_se': glac_se,
        'non_glaciated_alpha': nonglac_alpha,
        'non_glaciated_se': nonglac_se,
    }

    # Z-test for alpha difference
    if not np.isnan(glac_alpha) and not np.isnan(nonglac_alpha):
        alpha_diff = glac_alpha - nonglac_alpha
        pooled_se = np.sqrt(glac_se**2 + nonglac_se**2)
        z_stat = alpha_diff / pooled_se
        z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results['power_law_comparison']['alpha_difference'] = alpha_diff
        results['power_law_comparison']['z_statistic'] = z_stat
        results['power_law_comparison']['p_value'] = z_p
        results['power_law_comparison']['significant'] = z_p < 0.05

        if verbose:
            print(f"\nPower law exponent comparison (x_min = {xmin_threshold} km²):")
            print(f"  Glaciated: α = {glac_alpha:.3f} ± {glac_se:.3f}")
            print(f"  Non-glaciated: α = {nonglac_alpha:.3f} ± {nonglac_se:.3f}")
            print(f"  Difference: Δα = {alpha_diff:.3f}")
            print(f"  Z-test: z = {z_stat:.2f}, p = {z_p:.4f}")
            if z_p < 0.05:
                if alpha_diff > 0:
                    print(f"  → Glaciated regions have STEEPER size distribution (more small lakes)")
                else:
                    print(f"  → Non-glaciated regions have STEEPER size distribution")
            else:
                print(f"  → No significant difference in power law exponents")

    # Effect size: ratio of mean lake sizes
    if len(glac_areas) > 0 and len(nonglac_areas) > 0:
        size_ratio = np.mean(glac_areas) / np.mean(nonglac_areas)
        median_ratio = np.median(glac_areas) / np.median(nonglac_areas)

        results['effect_size'] = {
            'mean_ratio': size_ratio,
            'median_ratio': median_ratio,
            'interpretation': f"Glaciated lakes are {size_ratio:.1f}x larger on average"
        }

        if verbose:
            print(f"\nEffect size:")
            print(f"  Mean lake size ratio (glaciated/non-glaciated): {size_ratio:.2f}x")
            print(f"  Median lake size ratio: {median_ratio:.2f}x")

    return results


# ============================================================================
# ELEVATION-DEPENDENT SCALING
# ============================================================================

def analyze_elevation_size_scaling(lake_df, elev_col=None, area_col=None,
                                     elev_bins=None, verbose=True):
    """
    Analyze how lake size distributions change with elevation.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data
    elev_col : str, optional
        Elevation column name
    area_col : str, optional
        Lake area column
    elev_bins : array-like, optional
        Elevation bin edges
    verbose : bool
        Print results

    Returns
    -------
    dict
        Elevation-dependent size scaling results
    """
    if elev_col is None:
        elev_col = COLS.get('elev', 'Elevation')
    if area_col is None:
        area_col = COLS.get('area', 'AreaSqKm')

    if elev_col not in lake_df.columns:
        return {'error': f'Elevation column {elev_col} not found'}

    if elev_bins is None:
        elev_bins = np.arange(0, 4500, 500)

    if verbose:
        print("\n" + "=" * 60)
        print("ELEVATION-DEPENDENT SIZE SCALING")
        print("=" * 60)

    results = {'elev_col': elev_col, 'elev_bins': elev_bins}

    # Statistics by elevation band
    binned_stats = []
    for i in range(len(elev_bins) - 1):
        elev_low, elev_high = elev_bins[i], elev_bins[i+1]
        elev_mid = (elev_low + elev_high) / 2

        mask = (lake_df[elev_col] >= elev_low) & (lake_df[elev_col] < elev_high)
        band_lakes = lake_df[mask]

        n_lakes = len(band_lakes)
        if n_lakes > 0:
            areas = band_lakes[area_col].dropna().values
            areas = areas[areas > 0]

            # Power law fit
            xmin = 0.01
            tail = areas[areas >= xmin]
            if len(tail) >= 30:
                alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
                alpha_se = (alpha - 1) / np.sqrt(len(tail))
            else:
                alpha, alpha_se = np.nan, np.nan

            binned_stats.append({
                'elev_band': f'{elev_low:.0f}-{elev_high:.0f}m',
                'elev_mid': elev_mid,
                'n_lakes': n_lakes,
                'mean_area_km2': np.mean(areas) if len(areas) > 0 else np.nan,
                'median_area_km2': np.median(areas) if len(areas) > 0 else np.nan,
                'log_mean_area': np.mean(np.log10(areas)) if len(areas) > 0 else np.nan,
                'alpha': alpha,
                'alpha_se': alpha_se,
            })

    binned_df = pd.DataFrame(binned_stats)
    results['binned_stats'] = binned_df

    # Test for elevation trend in alpha
    valid = binned_df.dropna(subset=['alpha', 'elev_mid'])
    if len(valid) >= 3:
        corr, p_val = stats.pearsonr(valid['elev_mid'], valid['alpha'])
        results['alpha_elevation_trend'] = {
            'r': corr,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'direction': 'increasing' if corr > 0 else 'decreasing',
        }

        if verbose:
            print(f"\nα vs Elevation trend:")
            print(f"  r = {corr:.3f}, p = {p_val:.4f}")
            if p_val < 0.05:
                print(f"  → α {results['alpha_elevation_trend']['direction']} with elevation")

    # Correlation: lake size vs elevation
    valid_mask = lake_df[area_col].notna() & (lake_df[area_col] > 0)
    areas = lake_df.loc[valid_mask, area_col].values
    elevs = lake_df.loc[valid_mask, elev_col].values

    log_areas = np.log10(areas)
    corr, p_val = stats.pearsonr(elevs, log_areas)
    results['size_elevation_correlation'] = {
        'r': corr,
        'p_value': p_val,
        'interpretation': 'larger lakes at higher elevations' if corr > 0 else 'smaller lakes at higher elevations'
    }

    if verbose:
        print(f"\nLog(Area) ~ Elevation:")
        print(f"  r = {corr:.4f}, p = {p_val:.4e}")
        print(f"  → {results['size_elevation_correlation']['interpretation']}")

    return results


# ============================================================================
# COMPREHENSIVE SPATIAL ANALYSIS
# ============================================================================

def run_spatial_scaling_analysis(lake_df, lake_gdf=None, verbose=True):
    """
    Run comprehensive spatial scaling analysis.

    Combines latitudinal, longitudinal, elevation, and glacial analyses.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with coordinates
    lake_gdf : GeoDataFrame, optional
        Lake data with glacial classification
    verbose : bool
        Print results

    Returns
    -------
    dict
        All spatial scaling results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SPATIAL SCALING ANALYSIS")
        print("=" * 70)

    results = {}

    # 1. Latitudinal scaling
    if verbose:
        print("\n[1/4] Analyzing latitudinal patterns...")
    results['latitudinal'] = analyze_latitudinal_scaling(lake_df, verbose=verbose)

    # 2. Longitudinal scaling
    if verbose:
        print("\n[2/4] Analyzing longitudinal patterns...")
    results['longitudinal'] = analyze_longitudinal_scaling(lake_df, verbose=verbose)

    # 3. Elevation scaling
    if verbose:
        print("\n[3/4] Analyzing elevation-dependent scaling...")
    results['elevation'] = analyze_elevation_size_scaling(lake_df, verbose=verbose)

    # 4. Glacial vs non-glacial (if available)
    if lake_gdf is not None and 'glacial_stage' in lake_gdf.columns:
        if verbose:
            print("\n[4/4] Comparing glacial vs non-glacial scaling...")
        results['glacial_comparison'] = compare_glacial_vs_nonglacial_scaling(
            lake_gdf, verbose=verbose
        )
    else:
        if verbose:
            print("\n[4/4] Skipping glacial comparison (no classification available)")
        results['glacial_comparison'] = None

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SPATIAL SCALING SUMMARY")
        print("=" * 70)

        # Collect key findings
        findings = []

        if 'latitudinal' in results and 'correlation' in results['latitudinal']:
            lat_corr = results['latitudinal']['correlation']
            if lat_corr['p_value'] < 0.05:
                findings.append(f"• Latitude: {lat_corr['interpretation']} (r={lat_corr['r']:.3f})")

        if 'longitudinal' in results and 'correlation' in results['longitudinal']:
            lon_corr = results['longitudinal']['correlation']
            if lon_corr['p_value'] < 0.05:
                findings.append(f"• Longitude: {lon_corr['interpretation']} (r={lon_corr['r']:.3f})")

        if 'elevation' in results and 'size_elevation_correlation' in results['elevation']:
            elev_corr = results['elevation']['size_elevation_correlation']
            if elev_corr['p_value'] < 0.05:
                findings.append(f"• Elevation: {elev_corr['interpretation']} (r={elev_corr['r']:.3f})")

        if results.get('glacial_comparison') and 'effect_size' in results['glacial_comparison']:
            glac_effect = results['glacial_comparison']['effect_size']
            findings.append(f"• Glaciation: {glac_effect['interpretation']}")

        if findings:
            print("\nKey findings:")
            for f in findings:
                print(f"  {f}")
        else:
            print("\nNo significant spatial patterns detected.")

    return results


# ============================================================================
# HYPOTHESIS TESTING SUMMARY
# ============================================================================

def create_hypothesis_summary_table(results):
    """
    Create a summary table of all hypothesis tests from spatial analysis.

    Parameters
    ----------
    results : dict
        Output from run_spatial_scaling_analysis()

    Returns
    -------
    DataFrame
        Summary table with hypothesis, test statistic, p-value, and conclusion
    """
    rows = []

    # Latitudinal
    if 'latitudinal' in results and 'correlation' in results['latitudinal']:
        lat = results['latitudinal']
        rows.append({
            'Hypothesis': 'Lake size increases with latitude',
            'Test': 'Pearson correlation',
            'Statistic': f"r = {lat['correlation']['r']:.3f}",
            'p-value': lat['correlation']['p_value'],
            'Significant': lat['correlation']['p_value'] < 0.05,
            'Conclusion': lat['correlation']['interpretation']
        })

        if 'alpha_latitude_test' in lat:
            alpha_test = lat['alpha_latitude_test']
            rows.append({
                'Hypothesis': 'Power law α varies with latitude',
                'Test': 'Pearson correlation',
                'Statistic': f"r = {alpha_test['r']:.3f}",
                'p-value': alpha_test['p_value'],
                'Significant': alpha_test['significant'],
                'Conclusion': alpha_test['interpretation']
            })

    # Longitudinal
    if 'longitudinal' in results and 'correlation' in results['longitudinal']:
        lon = results['longitudinal']
        rows.append({
            'Hypothesis': 'Lake size varies with longitude',
            'Test': 'Pearson correlation',
            'Statistic': f"r = {lon['correlation']['r']:.3f}",
            'p-value': lon['correlation']['p_value'],
            'Significant': lon['correlation']['p_value'] < 0.05,
            'Conclusion': lon['correlation']['interpretation']
        })

    # Elevation
    if 'elevation' in results and 'size_elevation_correlation' in results['elevation']:
        elev = results['elevation']
        rows.append({
            'Hypothesis': 'Lake size varies with elevation',
            'Test': 'Pearson correlation',
            'Statistic': f"r = {elev['size_elevation_correlation']['r']:.3f}",
            'p-value': elev['size_elevation_correlation']['p_value'],
            'Significant': elev['size_elevation_correlation']['p_value'] < 0.05,
            'Conclusion': elev['size_elevation_correlation']['interpretation']
        })

    # Glacial comparison
    if results.get('glacial_comparison'):
        glac = results['glacial_comparison']

        if 'mann_whitney_test' in glac:
            mw = glac['mann_whitney_test']
            rows.append({
                'Hypothesis': 'Glacial/non-glacial size distributions differ',
                'Test': 'Mann-Whitney U',
                'Statistic': f"U = {mw['U_statistic']:.0f}",
                'p-value': mw['p_value'],
                'Significant': mw['significant'],
                'Conclusion': mw['interpretation']
            })

        if 'power_law_comparison' in glac and 'p_value' in glac['power_law_comparison']:
            pl = glac['power_law_comparison']
            rows.append({
                'Hypothesis': 'Power law α differs (glacial vs non-glacial)',
                'Test': 'Z-test',
                'Statistic': f"z = {pl['z_statistic']:.2f}",
                'p-value': pl['p_value'],
                'Significant': pl['significant'],
                'Conclusion': 'Exponents differ' if pl['significant'] else 'No difference'
            })

    return pd.DataFrame(rows)
