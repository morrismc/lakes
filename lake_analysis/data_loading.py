"""
Data Loading Module for Lake Distribution Analysis
====================================================

This module handles loading data from various sources:
- File Geodatabase (.gdb) for lake features
- ESRI Grid rasters (.adf format)
- GeoTIFF rasters

Key Features:
- Automatic CRS detection and reprojection
- NoData value handling
- Memory-efficient chunked raster processing
- Data quality filtering

Dependencies:
- geopandas (requires GDAL with OpenFileGDB driver)
- rasterio
- pandas
- numpy
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .config import (
        LAKE_GDB_PATH, LAKE_FEATURE_CLASS, LAKE_PARQUET_PATH,
        RASTERS, RASTER_METADATA, COLS, TARGET_CRS,
        NODATA_VALUE, MIN_LAKE_AREA, RASTER_TILE_SIZE
    )
except ImportError:
    from config import (
        LAKE_GDB_PATH, LAKE_FEATURE_CLASS, LAKE_PARQUET_PATH,
        RASTERS, RASTER_METADATA, COLS, TARGET_CRS,
        NODATA_VALUE, MIN_LAKE_AREA, RASTER_TILE_SIZE
    )


def get_raster_nodata(raster_name_or_path):
    """
    Get the NoData value for a raster, checking RASTER_METADATA first.

    Different rasters have different NoData values:
    - elevation: 32767
    - slope: -3.4028235e+38 (float min)
    - relief_5km: -32768
    - climate variables: -32768 or -9999

    Parameters
    ----------
    raster_name_or_path : str
        Either a key from RASTERS dict (e.g., 'elevation') or a full path

    Returns
    -------
    float or None
        NoData value from metadata, or None if not found
    """
    # Check if it's a key in RASTERS
    for name, path in RASTERS.items():
        if raster_name_or_path == name or raster_name_or_path == path:
            if name in RASTER_METADATA:
                return RASTER_METADATA[name].get('nodata')
    return None


def estimate_raster_memory(raster_path):
    """
    Estimate memory required to load a raster into RAM.

    Parameters
    ----------
    raster_path : str

    Returns
    -------
    dict
        Memory estimates and recommendations
    """
    import rasterio
    with rasterio.open(raster_path) as src:
        # Calculate size in bytes
        dtype_sizes = {
            'int8': 1, 'uint8': 1,
            'int16': 2, 'uint16': 2,
            'int32': 4, 'uint32': 4,
            'float32': 4, 'float64': 8,
        }
        dtype_str = str(src.dtypes[0])
        bytes_per_pixel = dtype_sizes.get(dtype_str, 4)

        total_pixels = src.width * src.height
        raw_bytes = total_pixels * bytes_per_pixel
        float_bytes = total_pixels * 8  # If converted to float64

        return {
            'dimensions': (src.height, src.width),
            'total_pixels': total_pixels,
            'dtype': dtype_str,
            'raw_size_gb': raw_bytes / 1e9,
            'float64_size_gb': float_bytes / 1e9,
            'recommend_chunked': raw_bytes > 1e9,  # >1GB
        }


# ============================================================================
# LAKE DATA LOADING
# ============================================================================

def load_lake_data_from_gdb(gdb_path=LAKE_GDB_PATH,
                            layer_name=LAKE_FEATURE_CLASS,
                            apply_filters=True):
    """
    Load lake data from a File Geodatabase.

    Parameters
    ----------
    gdb_path : str
        Path to the .gdb folder
    layer_name : str
        Name of the feature class to load
    apply_filters : bool
        If True, apply data quality filters

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned lake data with geometry

    Notes
    -----
    Requires GDAL with OpenFileGDB driver. Install via:
        conda install -c conda-forge gdal geopandas

    Column names are case-sensitive! 'Elevation_' has trailing underscore.
    """
    print(f"Loading lakes from geodatabase...")
    print(f"  Path: {gdb_path}")
    print(f"  Layer: {layer_name}")

    # Check if path exists
    if not os.path.exists(gdb_path):
        raise FileNotFoundError(f"Geodatabase not found: {gdb_path}")

    # List available layers (useful for debugging)
    try:
        import fiona
        layers = fiona.listlayers(gdb_path)
        print(f"  Available layers: {len(layers)}")
        if layer_name not in layers:
            print(f"  WARNING: '{layer_name}' not found. Available layers:")
            for l in layers[:10]:  # Show first 10
                print(f"    - {l}")
            if len(layers) > 10:
                print(f"    ... and {len(layers) - 10} more")
    except Exception as e:
        print(f"  Could not list layers: {e}")

    # Load the geodataframe
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    print(f"  Raw records loaded: {len(gdf):,}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")

    if apply_filters:
        gdf = apply_lake_quality_filters(gdf)

    return gdf


def load_lake_data_from_parquet(parquet_path=LAKE_PARQUET_PATH,
                                 apply_filters=True):
    """
    Load lake data from a parquet file (faster for repeated analyses).

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file
    apply_filters : bool
        If True, apply data quality filters

    Returns
    -------
    pandas.DataFrame
        Cleaned lake data (no geometry)
    """
    print(f"Loading lakes from parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  Raw records: {len(df):,}")

    if apply_filters:
        df = apply_lake_quality_filters(df)

    return df


def apply_lake_quality_filters(df, verbose=True):
    """
    Apply data quality filters to lake DataFrame.

    Filters:
    - Remove lakes with area = 0 or below minimum threshold
    - Remove records with NoData (-9999) in key columns
    - Remove negative elevations (errors)

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Lake data
    verbose : bool
        Print filtering statistics

    Returns
    -------
    DataFrame or GeoDataFrame
        Filtered data
    """
    initial_count = len(df)

    # Build filter mask
    mask = pd.Series(True, index=df.index)

    # Area filters
    area_col = COLS['area']
    if area_col in df.columns:
        mask &= (df[area_col] != 0)
        mask &= (df[area_col] >= MIN_LAKE_AREA)
        if verbose:
            print(f"  After area filter (>= {MIN_LAKE_AREA} km²): {mask.sum():,}")

    # Elevation filters
    elev_col = COLS['elevation']
    if elev_col in df.columns:
        mask &= (df[elev_col] != NODATA_VALUE)
        mask &= (df[elev_col] >= 0)
        if verbose:
            print(f"  After elevation filter (>= 0, != -9999): {mask.sum():,}")

    # Climate variable filters
    climate_cols = ['MAT', 'precip', 'PET', 'aridity']
    for col_key in climate_cols:
        col_name = COLS.get(col_key)
        if col_name and col_name in df.columns:
            mask &= (df[col_name] != NODATA_VALUE)

    if verbose:
        print(f"  After climate filters (!= -9999): {mask.sum():,}")

    # Slope filter
    slope_col = COLS.get('slope')
    if slope_col and slope_col in df.columns:
        mask &= (df[slope_col] != NODATA_VALUE)

    df_clean = df[mask].copy()

    if verbose:
        removed = initial_count - len(df_clean)
        pct_removed = (removed / initial_count) * 100
        print(f"  Final count: {len(df_clean):,} ({pct_removed:.1f}% removed)")

    return df_clean


def export_gdb_to_parquet(gdb_path=LAKE_GDB_PATH,
                          layer_name=LAKE_FEATURE_CLASS,
                          output_path=LAKE_PARQUET_PATH,
                          include_geometry=False):
    """
    Export geodatabase feature class to parquet for faster subsequent loading.

    Parameters
    ----------
    include_geometry : bool
        If True, include WKT geometry column (increases file size significantly)
    """
    print("Exporting geodatabase to parquet...")

    gdf = load_lake_data_from_gdb(gdb_path, layer_name, apply_filters=False)

    if include_geometry:
        gdf['geometry_wkt'] = gdf.geometry.to_wkt()

    # Convert to regular DataFrame (drop geometry)
    df = pd.DataFrame(gdf.drop(columns=['geometry'] if 'geometry' in gdf.columns else []))

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


# ============================================================================
# RASTER DATA LOADING
# ============================================================================

def get_raster_info(raster_path):
    """
    Get information about a raster file without loading data.

    Works with GeoTIFF, ESRI Grid (.adf), and other GDAL-supported formats.

    Parameters
    ----------
    raster_path : str
        Path to raster file or ESRI Grid folder

    Returns
    -------
    dict
        Raster metadata including CRS, dimensions, resolution, nodata value
    """
    with rasterio.open(raster_path) as src:
        # Calculate pixel area
        transform = src.transform
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])

        # Determine units
        crs = src.crs
        if crs and crs.is_geographic:
            # Approximate area at center of raster
            # For geographic CRS, pixel size is in degrees
            center_lat = (src.bounds.bottom + src.bounds.top) / 2
            # At equator: 1 degree ≈ 111 km
            lat_factor = np.cos(np.radians(center_lat))
            pixel_area_km2 = (pixel_width * 111) * (pixel_height * 111 * lat_factor)
            units = 'degrees'
        else:
            # Projected CRS, assume units are meters
            pixel_area_km2 = (pixel_width * pixel_height) / 1e6
            units = 'meters' if crs else 'unknown'

        info = {
            'path': raster_path,
            'width': src.width,
            'height': src.height,
            'crs': str(src.crs),
            'crs_units': units,
            'bounds': src.bounds,
            'transform': transform,
            'pixel_width': pixel_width,
            'pixel_height': pixel_height,
            'pixel_area_km2': pixel_area_km2,
            'nodata': src.nodata,
            'dtype': str(src.dtypes[0]),
            'count': src.count,  # Number of bands
        }

        return info


def load_raster(raster_path, band=1, masked=True):
    """
    Load a raster file into memory.

    WARNING: For large rasters, this may exceed available RAM!
    Use load_raster_chunked() for continental-scale data.

    Parameters
    ----------
    raster_path : str
        Path to raster file (GeoTIFF, ESRI Grid folder, etc.)
    band : int
        Band number to read (1-indexed)
    masked : bool
        If True, return masked array with NoData as NaN

    Returns
    -------
    tuple
        (data_array, transform, crs, nodata_value)
    """
    print(f"Loading raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(band)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        print(f"  Shape: {data.shape}")
        print(f"  CRS: {crs}")
        print(f"  NoData value: {nodata}")

        if masked and nodata is not None:
            data = data.astype(float)
            data[data == nodata] = np.nan
            valid_count = np.sum(~np.isnan(data))
            print(f"  Valid pixels: {valid_count:,} ({100*valid_count/data.size:.1f}%)")

        return data, transform, crs, nodata


def load_raster_chunked(raster_path, tile_size=RASTER_TILE_SIZE, custom_nodata=None):
    """
    Generator that yields raster data in tiles for memory-efficient processing.

    Parameters
    ----------
    raster_path : str
        Path to raster file
    tile_size : int
        Size of tiles in pixels (default from config)
    custom_nodata : float, optional
        Override NoData value. If None, uses value from RASTER_METADATA
        or the raster file metadata.

    Yields
    ------
    tuple
        (tile_data, window, row_offset, col_offset)

    Notes
    -----
    IMPORTANT: Different rasters have different NoData values!
    - elevation (srtm_ea): 32767
    - slope (srtm_slope): -3.4028235e+38
    - relief (srtm_rlif_5k): -32768
    """
    # Get nodata value from metadata if not provided
    if custom_nodata is None:
        custom_nodata = get_raster_nodata(raster_path)

    with rasterio.open(raster_path) as src:
        # Use custom nodata, then file metadata, then None
        nodata = custom_nodata if custom_nodata is not None else src.nodata

        for row_off in range(0, src.height, tile_size):
            for col_off in range(0, src.width, tile_size):
                # Calculate window size (handle edge cases)
                win_height = min(tile_size, src.height - row_off)
                win_width = min(tile_size, src.width - col_off)

                window = Window(col_off, row_off, win_width, win_height)
                data = src.read(1, window=window)

                # Replace nodata with NaN
                data = data.astype(float)
                if nodata is not None:
                    # Handle special case for float min (slope raster)
                    if nodata < -1e30:
                        data[data < -1e30] = np.nan
                    else:
                        data[data == nodata] = np.nan

                yield data, window, row_off, col_off


def check_raster_alignment(raster_paths):
    """
    Check if multiple rasters are aligned (same CRS, extent, resolution).

    Parameters
    ----------
    raster_paths : list
        List of paths to raster files

    Returns
    -------
    dict
        Alignment report with CRS comparison and recommendations
    """
    print("Checking raster alignment...")

    infos = {}
    for path in raster_paths:
        name = Path(path).stem
        try:
            infos[name] = get_raster_info(path)
            print(f"  ✓ {name}: {infos[name]['crs']}")
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")

    if len(infos) < 2:
        return {'aligned': True, 'message': 'Not enough rasters to compare'}

    # Compare CRS
    crs_list = [info['crs'] for info in infos.values()]
    crs_match = len(set(crs_list)) == 1

    # Compare resolution
    resolutions = [(info['pixel_width'], info['pixel_height']) for info in infos.values()]
    res_match = len(set(resolutions)) == 1

    # Compare bounds
    bounds_list = [info['bounds'] for info in infos.values()]

    report = {
        'aligned': crs_match and res_match,
        'crs_match': crs_match,
        'resolution_match': res_match,
        'unique_crs': list(set(crs_list)),
        'unique_resolutions': list(set(resolutions)),
        'infos': infos,
    }

    if not crs_match:
        print(f"\n  WARNING: CRS mismatch detected!")
        print(f"  Unique CRS values: {report['unique_crs']}")
        print(f"  Recommendation: Reproject to {TARGET_CRS}")

    if not res_match:
        print(f"\n  WARNING: Resolution mismatch detected!")
        print(f"  Unique resolutions: {report['unique_resolutions']}")

    return report


def reproject_raster_to_target(input_path, output_path, target_crs=TARGET_CRS,
                                resampling=Resampling.bilinear):
    """
    Reproject a raster to the target CRS.

    Parameters
    ----------
    input_path : str
        Path to input raster
    output_path : str
        Path for output raster (GeoTIFF)
    target_crs : str
        Target CRS (e.g., 'EPSG:5070')
    resampling : rasterio.Resampling
        Resampling method

    Returns
    -------
    str
        Path to reprojected raster
    """
    print(f"Reprojecting {Path(input_path).stem} to {target_crs}...")

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'compress': 'lzw',
        })

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    print(f"  Saved to: {output_path}")
    return output_path


# ============================================================================
# LANDSCAPE AREA CALCULATIONS
# ============================================================================

def calculate_landscape_area_by_bin(raster_path, breaks,
                                     use_chunked=True,
                                     tile_size=RASTER_TILE_SIZE,
                                     custom_nodata=None):
    """
    Calculate total landscape area in each bin of a raster variable.

    This is the KEY function for computing the normalization denominator.
    It tells us how much landscape area exists at each elevation/slope/etc.

    Parameters
    ----------
    raster_path : str
        Path to raster file
    breaks : list
        Bin edges (e.g., [0, 200, 400, 600, ...])
    use_chunked : bool
        If True, process in tiles (memory efficient). STRONGLY recommended
        for rasters > 1GB (elevation, slope, relief).
    tile_size : int
        Tile size for chunked processing
    custom_nodata : float, optional
        Override NoData value. If None, uses RASTER_METADATA.

    Returns
    -------
    pandas.DataFrame
        Columns: bin_lower, bin_upper, bin_label, pixel_count, area_km2

    Notes
    -----
    Memory Warning: Your topographic rasters are 8GB uncompressed!
    - elevation (srtm_ea): 8.05 GB
    - slope (srtm_slope): 8.05 GB
    - relief (srtm_rlif_5k): 3.96 GB
    Always use use_chunked=True (the default) for these rasters.
    """
    raster_name = Path(raster_path).stem
    print(f"Calculating landscape area by bin...")
    print(f"  Raster: {raster_name}")
    print(f"  Number of bins: {len(breaks) - 1}")

    # Estimate memory and warn if large
    mem_info = estimate_raster_memory(raster_path)
    if mem_info['recommend_chunked']:
        print(f"  WARNING: Large raster ({mem_info['raw_size_gb']:.2f} GB)")
        print(f"  Using chunked processing (tile_size={tile_size})")
        if not use_chunked:
            print(f"  CAUTION: use_chunked=False may cause memory issues!")

    # Get raster info for pixel area calculation
    info = get_raster_info(raster_path)
    pixel_area_km2 = info['pixel_area_km2']
    print(f"  Pixel area: {pixel_area_km2:.6f} km²")
    print(f"  CRS: {info['crs']}")

    # Get nodata value
    if custom_nodata is None:
        custom_nodata = get_raster_nodata(raster_path)
    if custom_nodata is not None:
        print(f"  NoData value: {custom_nodata}")

    # Initialize bin counts
    bin_counts = np.zeros(len(breaks) + 1, dtype=np.int64)

    if use_chunked:
        # Memory-efficient chunked processing
        total_pixels = 0
        n_tiles = 0
        for tile_data, window, _, _ in load_raster_chunked(raster_path, tile_size, custom_nodata):
            # Flatten and remove NaN
            values = tile_data.flatten()
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            total_pixels += len(values)
            n_tiles += 1

            # Digitize into bins
            bin_indices = np.digitize(values, breaks)

            # Count per bin
            unique, counts = np.unique(bin_indices, return_counts=True)
            for idx, count in zip(unique, counts):
                bin_counts[idx] += count

            # Progress indicator every 100 tiles
            if n_tiles % 100 == 0:
                print(f"    Processed {n_tiles} tiles, {total_pixels:,} valid pixels...")

        print(f"  Total tiles: {n_tiles}")
        print(f"  Total valid pixels processed: {total_pixels:,}")
    else:
        # Load entire raster (memory intensive)
        data, _, _, _ = load_raster(raster_path)
        values = data.flatten()
        values = values[~np.isnan(values)]

        bin_indices = np.digitize(values, breaks)
        unique, counts = np.unique(bin_indices, return_counts=True)
        for idx, count in zip(unique, counts):
            bin_counts[idx] += count

    # Build result DataFrame
    results = []
    for i in range(1, len(breaks)):
        bin_label = f"({breaks[i-1]}, {breaks[i]}]"
        count = bin_counts[i]
        results.append({
            'bin_lower': breaks[i-1],
            'bin_upper': breaks[i],
            'bin_label': bin_label,
            'pixel_count': count,
            'area_km2': count * pixel_area_km2,
        })

    result_df = pd.DataFrame(results)

    # Summary stats
    total_area = result_df['area_km2'].sum()
    print(f"  Total landscape area: {total_area:,.0f} km²")

    return result_df


# ============================================================================
# DATA SUMMARY FUNCTIONS
# ============================================================================

def summarize_lake_data(df):
    """
    Print comprehensive summary of lake dataset.

    Parameters
    ----------
    df : DataFrame
        Lake data
    """
    print("\n" + "=" * 60)
    print("LAKE DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal lakes: {len(df):,}")

    # Area statistics
    area_col = COLS['area']
    if area_col in df.columns:
        areas = df[area_col]
        print(f"\nLake Area ({area_col}):")
        print(f"  Min:    {areas.min():.6f} km²")
        print(f"  Median: {areas.median():.4f} km²")
        print(f"  Mean:   {areas.mean():.4f} km²")
        print(f"  Max:    {areas.max():.2f} km²")
        print(f"  Total:  {areas.sum():,.0f} km²")

        # Size classes
        print(f"\n  Size distribution:")
        print(f"    < 0.01 km²:   {(areas < 0.01).sum():,} ({100*(areas < 0.01).mean():.1f}%)")
        print(f"    0.01-0.1 km²: {((areas >= 0.01) & (areas < 0.1)).sum():,}")
        print(f"    0.1-1 km²:    {((areas >= 0.1) & (areas < 1)).sum():,}")
        print(f"    1-10 km²:     {((areas >= 1) & (areas < 10)).sum():,}")
        print(f"    10-100 km²:   {((areas >= 10) & (areas < 100)).sum():,}")
        print(f"    > 100 km²:    {(areas >= 100).sum():,}")

    # Elevation statistics
    elev_col = COLS['elevation']
    if elev_col in df.columns:
        elev = df[elev_col]
        print(f"\nElevation ({elev_col}):")
        print(f"  Min:    {elev.min():.0f} m")
        print(f"  Median: {elev.median():.0f} m")
        print(f"  Mean:   {elev.mean():.0f} m")
        print(f"  Max:    {elev.max():.0f} m")

    # Geographic extent
    lat_col = COLS.get('lat')
    lon_col = COLS.get('lon')
    if lat_col in df.columns and lon_col in df.columns:
        print(f"\nGeographic Extent:")
        print(f"  Latitude:  {df[lat_col].min():.2f}° to {df[lat_col].max():.2f}°")
        print(f"  Longitude: {df[lon_col].min():.2f}° to {df[lon_col].max():.2f}°")

    print("\n" + "=" * 60)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_all_raster_infos():
    """Load info for all configured rasters."""
    infos = {}
    for name, path in RASTERS.items():
        try:
            infos[name] = get_raster_info(path)
        except Exception as e:
            infos[name] = {'error': str(e)}
    return infos


def quick_data_check():
    """
    Perform quick check of all data sources.
    Useful for initial validation that everything is accessible.
    """
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY CHECK")
    print("=" * 60)

    # Check geodatabase
    print("\n[1] Lake Geodatabase:")
    if os.path.exists(LAKE_GDB_PATH):
        print(f"  ✓ Found: {LAKE_GDB_PATH}")
        try:
            import fiona
            layers = fiona.listlayers(LAKE_GDB_PATH)
            if LAKE_FEATURE_CLASS in layers:
                print(f"  ✓ Layer '{LAKE_FEATURE_CLASS}' exists")
            else:
                print(f"  ✗ Layer '{LAKE_FEATURE_CLASS}' NOT FOUND")
                print(f"    Available: {layers[:5]}...")
        except Exception as e:
            print(f"  ✗ Cannot read layers: {e}")
    else:
        print(f"  ✗ NOT FOUND: {LAKE_GDB_PATH}")

    # Check rasters
    print("\n[2] Rasters:")
    for name, path in RASTERS.items():
        if os.path.exists(path):
            try:
                info = get_raster_info(path)
                print(f"  ✓ {name}: {info['width']}x{info['height']}, {info['crs']}")
            except Exception as e:
                print(f"  ? {name}: exists but error reading - {e}")
        else:
            print(f"  ✗ {name}: NOT FOUND")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run quick data check when module is executed directly
    quick_data_check()
