"""
Normalization Module for Lake Distribution Analysis
=====================================================

This module computes normalized lake density - the key innovation of this analysis.

Core Concept:
    Raw lake counts are misleading because they don't account for how much
    landscape exists at each elevation/slope/etc. A mountain range might have
    fewer lakes simply because there's less total area at high elevations.

    Normalized Density = (# lakes in bin) / (landscape area in bin)

This module provides:
- 1D normalization (e.g., density vs elevation)
- 2D normalization (e.g., density in elevation × slope space)
- Residual analysis after controlling for covariates
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    COLS, ELEV_BREAKS, SLOPE_BREAKS, RELIEF_BREAKS,
    RASTER_TILE_SIZE, TARGET_CRS, RASTER_METADATA
)
from .data_loading import (
    calculate_landscape_area_by_bin,
    load_raster_chunked,
    get_raster_info,
    get_raster_nodata
)


# ============================================================================
# 1D NORMALIZED DENSITY
# ============================================================================

def compute_1d_normalized_density(lake_df, value_column, breaks,
                                   landscape_area_df,
                                   density_per=1000):
    """
    Compute normalized lake density across a single variable.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with column specified by value_column
    value_column : str
        Column name to bin (e.g., 'Elevation_', 'Slope')
    breaks : list
        Bin edges
    landscape_area_df : DataFrame
        Output from calculate_landscape_area_by_bin() with columns:
        bin_lower, bin_upper, area_km2
    density_per : float
        Report density per this many km² (default 1000 = lakes per 1000 km²)

    Returns
    -------
    DataFrame
        Columns: bin_lower, bin_upper, bin_label, n_lakes, area_km2,
                 raw_density, normalized_density, cumulative_lakes, cumulative_area
    """
    print(f"Computing normalized density for: {value_column}")

    # Validate column exists
    if value_column not in lake_df.columns:
        raise ValueError(f"Column '{value_column}' not found. Available: {list(lake_df.columns)}")

    # Bin the lakes
    lake_df = lake_df.copy()
    lake_df['_bin_idx'] = pd.cut(
        lake_df[value_column],
        bins=breaks,
        labels=False,
        include_lowest=True,
        right=True
    )

    # Count lakes per bin
    lake_counts = lake_df.groupby('_bin_idx').size().reset_index(name='n_lakes')
    lake_counts['_bin_idx'] = lake_counts['_bin_idx'].astype(int)

    # Prepare result DataFrame
    result = landscape_area_df.copy()
    result['_bin_idx'] = range(len(result))

    # Merge lake counts
    result = result.merge(lake_counts, on='_bin_idx', how='left')
    result['n_lakes'] = result['n_lakes'].fillna(0).astype(int)

    # Calculate densities
    # Raw density (lakes per km²)
    result['raw_density'] = result['n_lakes'] / result['area_km2']
    result['raw_density'] = result['raw_density'].replace([np.inf, -np.inf], np.nan)

    # Normalized density (lakes per density_per km²)
    result['normalized_density'] = result['raw_density'] * density_per

    # Cumulative statistics (useful for understanding distributions)
    result['cumulative_lakes'] = result['n_lakes'].cumsum()
    result['cumulative_area'] = result['area_km2'].cumsum()
    result['cumulative_density'] = result['cumulative_lakes'] / result['cumulative_area'] * density_per

    # Clean up
    result = result.drop(columns=['_bin_idx'])

    # Summary statistics
    total_lakes = result['n_lakes'].sum()
    total_area = result['area_km2'].sum()
    overall_density = total_lakes / total_area * density_per

    print(f"  Total lakes: {total_lakes:,}")
    print(f"  Total landscape area: {total_area:,.0f} km²")
    print(f"  Overall density: {overall_density:.2f} lakes per {density_per:,} km²")

    # Find peak bins
    max_density_idx = result['normalized_density'].idxmax()
    if not pd.isna(max_density_idx):
        peak_bin = result.loc[max_density_idx]
        print(f"  Peak density at: {peak_bin['bin_lower']:.0f}-{peak_bin['bin_upper']:.0f}")
        print(f"  Peak density value: {peak_bin['normalized_density']:.2f}")

    return result


def compute_1d_density_with_size_classes(lake_df, value_column, breaks,
                                          landscape_area_df,
                                          size_breaks=[0, 0.01, 0.1, 1, 10, 100, np.inf]):
    """
    Compute normalized density separately for different lake size classes.

    Useful for testing whether small vs large lakes have different distributions.

    Parameters
    ----------
    size_breaks : list
        Lake area thresholds for size classes (km²)

    Returns
    -------
    DataFrame
        Same as compute_1d_normalized_density but with additional 'size_class' column
    """
    area_col = COLS['area']
    results = []

    # Create size class labels
    size_labels = []
    for i in range(len(size_breaks) - 1):
        lower = size_breaks[i]
        upper = size_breaks[i + 1]
        if upper == np.inf:
            label = f"≥{lower}"
        else:
            label = f"{lower}-{upper}"
        size_labels.append(label)

    # Compute density for each size class
    for i, label in enumerate(size_labels):
        lower = size_breaks[i]
        upper = size_breaks[i + 1]

        # Filter lakes by size
        mask = (lake_df[area_col] >= lower) & (lake_df[area_col] < upper)
        subset = lake_df[mask]

        if len(subset) > 0:
            density_df = compute_1d_normalized_density(
                subset, value_column, breaks, landscape_area_df
            )
            density_df['size_class'] = label
            density_df['size_lower'] = lower
            density_df['size_upper'] = upper
            results.append(density_df)

    return pd.concat(results, ignore_index=True)


# ============================================================================
# 2D NORMALIZED DENSITY
# ============================================================================

def compute_2d_normalized_density(lake_df, raster1_path, raster2_path,
                                   var1_col, var2_col,
                                   var1_breaks, var2_breaks,
                                   tile_size=RASTER_TILE_SIZE):
    """
    Compute normalized lake density in 2D parameter space.

    This creates a heatmap showing where lakes concentrate when accounting
    for available landscape in both dimensions simultaneously.

    Example: Elevation × Slope space reveals:
    - Glacial domain: high elevation, moderate slope
    - Floodplain domain: low elevation, low slope
    - "Dead zone": intermediate elevation, steep slopes

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with columns var1_col and var2_col
    raster1_path, raster2_path : str
        Paths to rasters for landscape area calculation
    var1_col, var2_col : str
        Column names in lake_df
    var1_breaks, var2_breaks : list
        Bin edges for each variable

    Returns
    -------
    DataFrame
        Columns for both variables' bin bounds, landscape_pixels, n_lakes,
        normalized_density

    Notes
    -----
    MEMORY WARNING: This can be very intensive for large rasters!
    Both rasters must have the same dimensions and alignment.
    """
    print(f"\nComputing 2D normalized density: {var1_col} × {var2_col}")

    # Get raster info
    info1 = get_raster_info(raster1_path)
    info2 = get_raster_info(raster2_path)

    # Check alignment
    if (info1['width'] != info2['width']) or (info1['height'] != info2['height']):
        raise ValueError(
            f"Rasters must have same dimensions!\n"
            f"  {var1_col}: {info1['width']}x{info1['height']}\n"
            f"  {var2_col}: {info2['width']}x{info2['height']}"
        )

    pixel_area_km2 = info1['pixel_area_km2']
    print(f"  Pixel area: {pixel_area_km2:.6f} km²")

    # Initialize 2D counts for landscape
    # Shape: (n_bins_var1+1, n_bins_var2+1) to catch out-of-range values
    landscape_counts = np.zeros((len(var1_breaks) + 1, len(var2_breaks) + 1), dtype=np.int64)

    # Process rasters in parallel chunks
    print(f"  Processing rasters in tiles...")
    import rasterio

    # Get NoData values from RASTER_METADATA (more reliable than raster file metadata)
    nodata1 = get_raster_nodata(raster1_path)
    nodata2 = get_raster_nodata(raster2_path)
    print(f"  NoData values: var1={nodata1}, var2={nodata2}")

    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        # Fall back to file metadata if not in RASTER_METADATA
        if nodata1 is None:
            nodata1 = src1.nodata
        if nodata2 is None:
            nodata2 = src2.nodata

        n_tiles = 0
        total_valid_pixels = 0
        for row_off in range(0, src1.height, tile_size):
            for col_off in range(0, src1.width, tile_size):
                # Calculate window
                win_height = min(tile_size, src1.height - row_off)
                win_width = min(tile_size, src1.width - col_off)
                window = rasterio.windows.Window(col_off, row_off, win_width, win_height)

                # Read both rasters
                data1 = src1.read(1, window=window).astype(float)
                data2 = src2.read(1, window=window).astype(float)

                # Handle nodata - use custom values from RASTER_METADATA
                if nodata1 is not None:
                    # Handle special case for float min (slope raster)
                    if nodata1 < -1e30:
                        data1[data1 < -1e30] = np.nan
                    else:
                        data1[data1 == nodata1] = np.nan
                if nodata2 is not None:
                    if nodata2 < -1e30:
                        data2[data2 < -1e30] = np.nan
                    else:
                        data2[data2 == nodata2] = np.nan

                # Flatten and find valid pixels
                v1 = data1.flatten()
                v2 = data2.flatten()
                valid = ~(np.isnan(v1) | np.isnan(v2))
                v1 = v1[valid]
                v2 = v2[valid]
                total_valid_pixels += len(v1)

                # Digitize
                bins1 = np.digitize(v1, var1_breaks)
                bins2 = np.digitize(v2, var2_breaks)

                # Count in 2D
                for b1, b2 in zip(bins1, bins2):
                    landscape_counts[b1, b2] += 1

                n_tiles += 1

                # Progress indicator every 100 tiles
                if n_tiles % 100 == 0:
                    print(f"    Processed {n_tiles} tiles, {total_valid_pixels:,} valid pixels...")

        print(f"  Processed {n_tiles} tiles total")
        print(f"  Total valid pixels: {total_valid_pixels:,}")

    # Bin the lakes in 2D
    print(f"  Binning lakes...")
    lake_bins1 = np.digitize(lake_df[var1_col].values, var1_breaks)
    lake_bins2 = np.digitize(lake_df[var2_col].values, var2_breaks)

    lake_counts = np.zeros_like(landscape_counts)
    for b1, b2 in zip(lake_bins1, lake_bins2):
        lake_counts[b1, b2] += 1

    # Build result DataFrame
    print(f"  Building result DataFrame...")
    results = []
    for i in range(1, len(var1_breaks)):
        for j in range(1, len(var2_breaks)):
            n_landscape = landscape_counts[i, j]
            n_lakes = lake_counts[i, j]

            # Normalized density (lakes per 1000 km²)
            if n_landscape > 0:
                area_km2 = n_landscape * pixel_area_km2
                density = (n_lakes / area_km2) * 1000
            else:
                area_km2 = 0
                density = np.nan

            results.append({
                f'{var1_col}_bin_lower': var1_breaks[i-1],
                f'{var1_col}_bin_upper': var1_breaks[i],
                f'{var1_col}_mid': (var1_breaks[i-1] + var1_breaks[i]) / 2,
                f'{var2_col}_bin_lower': var2_breaks[j-1],
                f'{var2_col}_bin_upper': var2_breaks[j],
                f'{var2_col}_mid': (var2_breaks[j-1] + var2_breaks[j]) / 2,
                'landscape_pixels': n_landscape,
                'area_km2': area_km2,
                'n_lakes': n_lakes,
                'normalized_density': density,
            })

    result_df = pd.DataFrame(results)

    # Summary
    valid_cells = result_df['normalized_density'].notna().sum()
    total_cells = len(result_df)
    print(f"  Valid cells: {valid_cells}/{total_cells}")
    print(f"  Max density: {result_df['normalized_density'].max():.2f}")

    return result_df


def compute_2d_density_from_lake_attributes(lake_df, var1_col, var2_col,
                                             var1_breaks, var2_breaks,
                                             landscape_area_2d=None):
    """
    Simplified 2D density when you have both variables in the lake DataFrame
    and a pre-computed landscape area table.

    Faster than compute_2d_normalized_density when landscape areas are pre-computed.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with columns for both variables
    var1_col, var2_col : str
        Column names
    var1_breaks, var2_breaks : list
        Bin edges
    landscape_area_2d : DataFrame, optional
        Pre-computed landscape area in each 2D bin. If None, returns raw counts only.

    Returns
    -------
    DataFrame
        2D density table
    """
    # Bin lakes
    lake_df = lake_df.copy()
    lake_df['_bin1'] = pd.cut(lake_df[var1_col], bins=var1_breaks, labels=False, include_lowest=True)
    lake_df['_bin2'] = pd.cut(lake_df[var2_col], bins=var2_breaks, labels=False, include_lowest=True)

    # Count lakes per 2D bin
    counts = lake_df.groupby(['_bin1', '_bin2']).size().reset_index(name='n_lakes')

    # Expand to full grid
    results = []
    for i in range(len(var1_breaks) - 1):
        for j in range(len(var2_breaks) - 1):
            mask = (counts['_bin1'] == i) & (counts['_bin2'] == j)
            n_lakes = counts.loc[mask, 'n_lakes'].sum() if mask.any() else 0

            results.append({
                f'{var1_col}_bin_lower': var1_breaks[i],
                f'{var1_col}_bin_upper': var1_breaks[i+1],
                f'{var1_col}_mid': (var1_breaks[i] + var1_breaks[i+1]) / 2,
                f'{var2_col}_bin_lower': var2_breaks[j],
                f'{var2_col}_bin_upper': var2_breaks[j+1],
                f'{var2_col}_mid': (var2_breaks[j] + var2_breaks[j+1]) / 2,
                'n_lakes': n_lakes,
            })

    result_df = pd.DataFrame(results)

    # Merge with landscape areas if provided
    if landscape_area_2d is not None:
        # Merge and compute normalized density
        result_df = result_df.merge(
            landscape_area_2d[[f'{var1_col}_bin_lower', f'{var2_col}_bin_lower', 'area_km2']],
            on=[f'{var1_col}_bin_lower', f'{var2_col}_bin_lower'],
            how='left'
        )
        result_df['normalized_density'] = (result_df['n_lakes'] / result_df['area_km2']) * 1000
        result_df['normalized_density'] = result_df['normalized_density'].replace([np.inf, -np.inf], np.nan)

    return result_df


# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

def compute_residuals_after_covariate(lake_df, primary_var, covariate_var,
                                       primary_breaks, covariate_breaks,
                                       landscape_primary, landscape_2d):
    """
    Compute residual lake density after controlling for a covariate.

    Example: After controlling for elevation, how does slope affect lake density?

    Method:
    1. Compute expected density in each covariate bin
    2. For each primary variable bin, compute observed vs expected
    3. Residual = observed - expected (or ratio)

    Parameters
    ----------
    lake_df : DataFrame
    primary_var : str
        Variable of interest (e.g., 'Slope')
    covariate_var : str
        Variable to control for (e.g., 'Elevation_')
    primary_breaks, covariate_breaks : list
    landscape_primary : DataFrame
        1D landscape area for primary variable
    landscape_2d : DataFrame
        2D landscape area

    Returns
    -------
    DataFrame
        Residual density for each primary variable bin
    """
    # Compute 2D density
    density_2d = compute_2d_density_from_lake_attributes(
        lake_df, covariate_var, primary_var,
        covariate_breaks, primary_breaks,
        landscape_2d
    )

    # Compute marginal (expected) density for covariate
    marginal = compute_1d_normalized_density(
        lake_df, covariate_var, covariate_breaks, landscape_primary
    )

    # For each primary bin, compute weighted expected vs observed
    results = []
    for i in range(len(primary_breaks) - 1):
        primary_mask = (
            (density_2d[f'{primary_var}_bin_lower'] == primary_breaks[i])
        )
        subset = density_2d[primary_mask]

        if len(subset) == 0:
            continue

        # Observed: actual density in this primary bin across all covariate bins
        observed_lakes = subset['n_lakes'].sum()
        observed_area = subset['area_km2'].sum()
        observed_density = (observed_lakes / observed_area * 1000) if observed_area > 0 else np.nan

        # Expected: weighted by covariate distribution
        # This is more complex - simplified version uses overall mean
        expected_density = lake_df.groupby(
            pd.cut(lake_df[primary_var], bins=primary_breaks, labels=False)
        ).size().sum() / landscape_primary['area_km2'].sum() * 1000

        results.append({
            f'{primary_var}_bin_lower': primary_breaks[i],
            f'{primary_var}_bin_upper': primary_breaks[i+1],
            'observed_density': observed_density,
            'expected_density': expected_density,
            'residual': observed_density - expected_density if not np.isnan(observed_density) else np.nan,
            'residual_ratio': observed_density / expected_density if expected_density > 0 else np.nan,
        })

    return pd.DataFrame(results)


# ============================================================================
# DOMAIN CLASSIFICATION
# ============================================================================

def classify_lake_domains(lake_df, elev_col=None, slope_col=None):
    """
    Classify lakes into geomorphic process domains.

    Domains based on elevation and slope:
    - Glacial: high elevation (>1500m), moderate slopes
    - Alpine: very high elevation (>2500m)
    - Floodplain: low elevation (<300m), low slopes (<5°)
    - Hillslope: steep slopes (>15°)
    - Plateau: mid-elevation, low slopes

    Parameters
    ----------
    lake_df : DataFrame
    elev_col : str, optional
        Elevation column name (default from config)
    slope_col : str, optional
        Slope column name (default from config)

    Returns
    -------
    DataFrame
        Original data with added 'domain' column
    """
    if elev_col is None:
        elev_col = COLS['elevation']
    if slope_col is None:
        slope_col = COLS['slope']

    df = lake_df.copy()

    # Initialize domain as 'other'
    df['domain'] = 'other'

    # Floodplain: low elevation, gentle slopes
    mask_floodplain = (df[elev_col] < 300) & (df[slope_col] < 5)
    df.loc[mask_floodplain, 'domain'] = 'floodplain'

    # Glacial: high elevation
    mask_glacial = (df[elev_col] >= 1500) & (df[elev_col] < 2500)
    df.loc[mask_glacial, 'domain'] = 'glacial'

    # Alpine: very high elevation
    mask_alpine = df[elev_col] >= 2500
    df.loc[mask_alpine, 'domain'] = 'alpine'

    # Hillslope: steep regardless of elevation (overrides others)
    mask_hillslope = df[slope_col] >= 15
    df.loc[mask_hillslope, 'domain'] = 'hillslope'

    # Plateau: mid-elevation, gentle slopes
    mask_plateau = (
        (df[elev_col] >= 300) &
        (df[elev_col] < 1500) &
        (df[slope_col] < 5) &
        (df['domain'] == 'other')
    )
    df.loc[mask_plateau, 'domain'] = 'plateau'

    # Summary
    print("\nDomain Classification Summary:")
    print(df['domain'].value_counts())

    return df


# ============================================================================
# SUMMARY STATISTICS BY DOMAIN
# ============================================================================

def compute_domain_statistics(lake_df, domain_col='domain'):
    """
    Compute summary statistics for each domain.

    Returns
    -------
    DataFrame
        Statistics for each domain
    """
    area_col = COLS['area']
    elev_col = COLS['elevation']

    stats = lake_df.groupby(domain_col).agg({
        area_col: ['count', 'mean', 'median', 'sum', 'std'],
        elev_col: ['mean', 'median', 'min', 'max'],
    })

    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()

    return stats


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_full_1d_analysis(lake_df, raster_path, column_name, breaks,
                          output_prefix=None):
    """
    Run complete 1D normalized density analysis for one variable.

    Steps:
    1. Calculate landscape area by bin
    2. Compute normalized density
    3. Optionally save results

    Returns
    -------
    tuple
        (landscape_area_df, normalized_density_df)
    """
    # Calculate landscape area
    landscape_area = calculate_landscape_area_by_bin(raster_path, breaks)

    # Compute normalized density
    density = compute_1d_normalized_density(lake_df, column_name, breaks, landscape_area)

    # Save if prefix provided
    if output_prefix:
        from config import OUTPUT_DIR, ensure_output_dir
        ensure_output_dir()

        landscape_area.to_csv(f"{OUTPUT_DIR}/{output_prefix}_landscape_area.csv", index=False)
        density.to_csv(f"{OUTPUT_DIR}/{output_prefix}_normalized_density.csv", index=False)
        print(f"  Results saved to {OUTPUT_DIR}/")

    return landscape_area, density


if __name__ == "__main__":
    # Example usage
    print("Normalization module loaded.")
    print("Key functions:")
    print("  - compute_1d_normalized_density()")
    print("  - compute_2d_normalized_density()")
    print("  - classify_lake_domains()")
