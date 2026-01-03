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
        LAKE_CONUS_PARQUET_PATH, RASTERS, RASTER_METADATA, COLS, TARGET_CRS,
        NODATA_VALUE, MIN_LAKE_AREA, RASTER_TILE_SIZE
    )
except ImportError:
    from config import (
        LAKE_GDB_PATH, LAKE_FEATURE_CLASS, LAKE_PARQUET_PATH,
        LAKE_CONUS_PARQUET_PATH, RASTERS, RASTER_METADATA, COLS, TARGET_CRS,
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


def apply_lake_quality_filters(df, verbose=True, min_lake_area=None):
    """
    Apply data quality filters to lake DataFrame.

    Filters:
    - Remove lakes with area = 0
    - Optionally remove lakes below minimum area threshold
    - Remove records with NoData (-9999) in key columns
    - Remove negative elevations (errors)

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Lake data
    verbose : bool
        Print filtering statistics
    min_lake_area : float, optional
        Minimum lake area in km². If None, no area threshold is applied
        (only zero-area lakes are removed). This allows filtering at
        analysis time rather than data loading time.

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
        mask &= (df[area_col] > 0)  # Remove negative/invalid areas
        if verbose:
            print(f"  After removing zero/invalid areas: {mask.sum():,}")

        # Only apply min area filter if explicitly specified
        if min_lake_area is not None:
            mask &= (df[area_col] >= min_lake_area)
            if verbose:
                print(f"  After area threshold (>= {min_lake_area} km²): {mask.sum():,}")

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


def sample_raster_at_points(raster_path, points_gdf, value_column='raster_value',
                            src_crs=None, batch_size=50000, verbose=True):
    """
    Sample raster values at point locations.

    This function efficiently extracts raster values at the locations of
    point geometries, handling CRS transformations and NoData values.

    Parameters
    ----------
    raster_path : str
        Path to raster file (GeoTIFF, ESRI Grid, etc.)
    points_gdf : GeoDataFrame
        GeoDataFrame with point geometries
    value_column : str
        Name of the column to store sampled values (default: 'raster_value')
    src_crs : str or CRS, optional
        CRS of the input points. If None, uses points_gdf.crs
    batch_size : int
        Number of points to process at once (memory optimization)
    verbose : bool
        Print progress information

    Returns
    -------
    GeoDataFrame
        Input GeoDataFrame with new column containing sampled values.
        NoData locations are set to NaN.

    Notes
    -----
    - If points are outside the raster extent, values will be NaN
    - Points in NoData regions of the raster will have NaN values
    - CRS transformation is handled automatically
    """
    from pyproj import Transformer

    if verbose:
        print(f"\nSampling raster values at {len(points_gdf):,} point locations...")
        print(f"  Raster: {Path(raster_path).name}")

    # Open raster to get metadata
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
        raster_bounds = src.bounds
        nodata = src.nodata
        raster_data = src.read(1)

        if verbose:
            print(f"  Raster CRS: {raster_crs}")
            print(f"  Raster shape: {raster_data.shape}")
            print(f"  NoData value: {nodata}")

        # Determine source CRS
        if src_crs is None:
            src_crs = points_gdf.crs

        # Check if we need to transform coordinates
        need_transform = src_crs is not None and not CRS.from_user_input(src_crs).equals(raster_crs)

        if need_transform:
            if verbose:
                print(f"  Transforming points from {src_crs} to {raster_crs}")
            transformer = Transformer.from_crs(src_crs, raster_crs, always_xy=True)

        # Extract coordinates
        coords = np.array([(geom.x, geom.y) for geom in points_gdf.geometry])

        if need_transform:
            x_coords, y_coords = transformer.transform(coords[:, 0], coords[:, 1])
        else:
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]

        # Sample values
        values = np.full(len(points_gdf), np.nan)

        # Process in batches for memory efficiency
        n_batches = (len(points_gdf) + batch_size - 1) // batch_size
        valid_count = 0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(points_gdf))

            batch_x = x_coords[start_idx:end_idx]
            batch_y = y_coords[start_idx:end_idx]

            # Convert coordinates to row/col indices
            # rasterio transform: ~transform * (col, row) = (x, y)
            # Inverse: (col, row) = ~transform.inverse * (x, y)
            cols, rows = ~raster_transform * (batch_x, batch_y)
            rows = rows.astype(int)
            cols = cols.astype(int)

            # Check bounds
            valid_mask = (
                (rows >= 0) & (rows < raster_data.shape[0]) &
                (cols >= 0) & (cols < raster_data.shape[1])
            )

            # Sample valid points
            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]
            sampled = raster_data[valid_rows, valid_cols]

            # Handle NoData
            if nodata is not None:
                sampled = sampled.astype(float)
                sampled[sampled == nodata] = np.nan

            # Store values
            batch_values = np.full(end_idx - start_idx, np.nan)
            batch_values[valid_mask] = sampled
            values[start_idx:end_idx] = batch_values

            valid_count += np.sum(~np.isnan(batch_values))

        if verbose:
            pct_valid = 100 * valid_count / len(points_gdf)
            print(f"  Valid samples: {valid_count:,} ({pct_valid:.1f}%)")

    # Add values to GeoDataFrame
    result_gdf = points_gdf.copy()
    result_gdf[value_column] = values

    return result_gdf


def sample_raster_at_coords(raster_path, lons, lats, points_crs='EPSG:4326',
                            batch_size=50000, verbose=True):
    """
    Sample raster values at coordinate arrays (simpler interface).

    Parameters
    ----------
    raster_path : str
        Path to raster file
    lons : array-like
        Longitude values
    lats : array-like
        Latitude values
    points_crs : str
        CRS of the input coordinates (default: EPSG:4326 for lat/lon)
    batch_size : int
        Number of points to process at once
    verbose : bool
        Print progress information

    Returns
    -------
    numpy.ndarray
        Sampled raster values (NaN where invalid/NoData)
    """
    from pyproj import Transformer

    lons = np.asarray(lons)
    lats = np.asarray(lats)

    if verbose:
        print(f"\nSampling raster at {len(lons):,} coordinates...")
        print(f"  Raster: {Path(raster_path).name}")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
        nodata = src.nodata
        raster_data = src.read(1)

        if verbose:
            print(f"  Raster CRS: {raster_crs}")
            print(f"  NoData value: {nodata}")

        # Transform coordinates to raster CRS
        need_transform = not CRS.from_user_input(points_crs).equals(raster_crs)

        if need_transform:
            if verbose:
                print(f"  Transforming from {points_crs} to {raster_crs}")
            transformer = Transformer.from_crs(points_crs, raster_crs, always_xy=True)
            x_coords, y_coords = transformer.transform(lons, lats)
        else:
            x_coords, y_coords = lons, lats

        # Sample values
        values = np.full(len(lons), np.nan)
        valid_count = 0

        n_batches = (len(lons) + batch_size - 1) // batch_size
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(lons))

            batch_x = x_coords[start_idx:end_idx]
            batch_y = y_coords[start_idx:end_idx]

            cols, rows = ~raster_transform * (batch_x, batch_y)
            rows = rows.astype(int)
            cols = cols.astype(int)

            valid_mask = (
                (rows >= 0) & (rows < raster_data.shape[0]) &
                (cols >= 0) & (cols < raster_data.shape[1])
            )

            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]
            sampled = raster_data[valid_rows, valid_cols]

            if nodata is not None:
                sampled = sampled.astype(float)
                sampled[sampled == nodata] = np.nan

            batch_values = np.full(end_idx - start_idx, np.nan)
            batch_values[valid_mask] = sampled
            values[start_idx:end_idx] = batch_values

            valid_count += np.sum(~np.isnan(batch_values))

        if verbose:
            pct_valid = 100 * valid_count / len(lons)
            print(f"  Valid samples: {valid_count:,} ({pct_valid:.1f}%)")

    return values


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


# ============================================================================
# CONUS CLIPPING AND PROJECTION UTILITIES
# ============================================================================

def load_conus_boundary(shapefile_path=None, target_crs=None):
    """
    Load the CONUS boundary shapefile and optionally reproject.

    The Census Bureau shapefile includes all US territories, so this
    filters to just the contiguous 48 states.

    Parameters
    ----------
    shapefile_path : str, optional
        Path to shapefile (defaults to config SHAPEFILES['conus_boundary'])
    target_crs : str, optional
        Target CRS for reprojection (e.g., 'EPSG:5070' for Albers)

    Returns
    -------
    GeoDataFrame
        CONUS boundary polygon
    """
    try:
        from .config import SHAPEFILES
    except ImportError:
        from config import SHAPEFILES

    if shapefile_path is None:
        shapefile_path = SHAPEFILES.get('conus_boundary')

    if shapefile_path is None or not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"CONUS boundary shapefile not found: {shapefile_path}")

    print(f"Loading CONUS boundary from: {shapefile_path}")
    boundary = gpd.read_file(shapefile_path)

    # The cb_2024_us_nation shapefile is just the nation boundary
    # If it has multiple features (Alaska, Hawaii, territories), filter for CONUS
    # For nation-level file, typically just one feature - may need to clip by extent

    # CONUS approximate bounding box (NAD83 geographic)
    conus_bounds = {
        'min_lon': -125.0,
        'max_lon': -66.0,
        'min_lat': 24.0,
        'max_lat': 50.0,
    }

    # If the boundary includes Alaska/Hawaii/territories, clip to CONUS extent
    # This is a rough clip - for precise work, use a proper CONUS polygon
    if len(boundary) == 1:
        # Single polygon - clip to CONUS bounding box
        from shapely.geometry import box
        conus_box = box(conus_bounds['min_lon'], conus_bounds['min_lat'],
                        conus_bounds['max_lon'], conus_bounds['max_lat'])
        boundary['geometry'] = boundary.geometry.intersection(conus_box)

    print(f"  Original CRS: {boundary.crs}")

    # Reproject if target CRS specified
    if target_crs is not None:
        boundary = boundary.to_crs(target_crs)
        print(f"  Reprojected to: {boundary.crs}")

    return boundary


def clip_lakes_to_conus(lakes_gdf, conus_boundary=None):
    """
    Clip lake dataset to CONUS boundary.

    Parameters
    ----------
    lakes_gdf : GeoDataFrame
        Lake data with geometry
    conus_boundary : GeoDataFrame, optional
        CONUS boundary polygon (loads from config if None)

    Returns
    -------
    GeoDataFrame
        Lakes within CONUS only
    """
    if conus_boundary is None:
        conus_boundary = load_conus_boundary()

    # Ensure same CRS
    if lakes_gdf.crs != conus_boundary.crs:
        print(f"  Reprojecting boundary from {conus_boundary.crs} to {lakes_gdf.crs}")
        conus_boundary = conus_boundary.to_crs(lakes_gdf.crs)

    print(f"  Clipping {len(lakes_gdf):,} lakes to CONUS boundary...")
    lakes_clipped = gpd.clip(lakes_gdf, conus_boundary)
    print(f"  Result: {len(lakes_clipped):,} lakes within CONUS")

    return lakes_clipped


# CONUS bounding box (NAD83 geographic coordinates)
CONUS_BOUNDS = {
    'min_lon': -125.0,
    'max_lon': -66.5,
    'min_lat': 24.5,
    'max_lat': 49.5,
}


def create_conus_lake_dataset(source='gdb', output_path=None, use_bbox=True):
    """
    Create a CONUS-only lake dataset and save to parquet.

    This should be run ONCE to create the clipped dataset, which is then
    used for all subsequent analyses.

    Parameters
    ----------
    source : str
        'gdb' to load from geodatabase, 'parquet' to load from existing parquet
    output_path : str, optional
        Output path for CONUS parquet (defaults to LAKE_CONUS_PARQUET_PATH)
    use_bbox : bool
        If True, use simple bounding box filter (fast)
        If False, use actual CONUS shapefile for precise clipping (slower)

    Returns
    -------
    DataFrame
        CONUS-only lake dataset
    """
    print("\n" + "=" * 60)
    print("CREATING CONUS LAKE DATASET")
    print("=" * 60)

    if output_path is None:
        output_path = LAKE_CONUS_PARQUET_PATH

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    print(f"\n[STEP 1/3] Loading full lake dataset from {source}...")
    if source == 'gdb':
        lakes = load_lake_data_from_gdb()
    else:
        lakes = load_lake_data_from_parquet()

    n_original = len(lakes)
    print(f"  Loaded {n_original:,} lakes")

    # Filter to CONUS
    print(f"\n[STEP 2/3] Filtering to CONUS extent...")

    lat_col = COLS.get('lat', 'Latitude')
    lon_col = COLS.get('lon', 'Longitude')

    if use_bbox:
        # Simple bounding box filter (fast)
        print(f"  Using bounding box: lon=[{CONUS_BOUNDS['min_lon']}, {CONUS_BOUNDS['max_lon']}], "
              f"lat=[{CONUS_BOUNDS['min_lat']}, {CONUS_BOUNDS['max_lat']}]")

        mask = (
            (lakes[lon_col] >= CONUS_BOUNDS['min_lon']) &
            (lakes[lon_col] <= CONUS_BOUNDS['max_lon']) &
            (lakes[lat_col] >= CONUS_BOUNDS['min_lat']) &
            (lakes[lat_col] <= CONUS_BOUNDS['max_lat'])
        )
        lakes_conus = lakes[mask].copy()
    else:
        # Use actual shapefile (slower but more precise)
        print("  Using CONUS shapefile for precise clipping...")
        if hasattr(lakes, 'geometry'):
            lakes_conus = clip_lakes_to_conus(lakes)
        else:
            # If not a GeoDataFrame, fall back to bbox
            print("  Warning: No geometry column, falling back to bounding box")
            mask = (
                (lakes[lon_col] >= CONUS_BOUNDS['min_lon']) &
                (lakes[lon_col] <= CONUS_BOUNDS['max_lon']) &
                (lakes[lat_col] >= CONUS_BOUNDS['min_lat']) &
                (lakes[lat_col] <= CONUS_BOUNDS['max_lat'])
            )
            lakes_conus = lakes[mask].copy()

    n_conus = len(lakes_conus)
    n_removed = n_original - n_conus
    print(f"  CONUS lakes: {n_conus:,} ({n_removed:,} removed)")

    # Summary of what was removed
    if n_removed > 0:
        outside = lakes[~lakes.index.isin(lakes_conus.index)]
        if lat_col in outside.columns:
            ak_count = len(outside[outside[lat_col] > 50])
            hi_count = len(outside[(outside[lat_col] < 24) & (outside[lon_col] < -150)])
            other = n_removed - ak_count - hi_count
            print(f"  Removed: ~{ak_count:,} Alaska, ~{hi_count:,} Hawaii, ~{other:,} other/territories")

    # Save to parquet
    print(f"\n[STEP 3/3] Saving to {output_path}...")

    # Convert to regular DataFrame if GeoDataFrame (parquet doesn't need geometry)
    if hasattr(lakes_conus, 'geometry'):
        # Keep geometry as WKT for potential future use
        lakes_conus = pd.DataFrame(lakes_conus)
        if 'geometry' in lakes_conus.columns:
            lakes_conus['geometry_wkt'] = lakes_conus['geometry'].astype(str)
            lakes_conus = lakes_conus.drop(columns=['geometry'])

    lakes_conus.to_parquet(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved! File size: {file_size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print(f"SUCCESS: CONUS lake dataset created with {n_conus:,} lakes")
    print(f"Path: {output_path}")
    print("=" * 60)

    return lakes_conus


def load_conus_lake_data(create_if_missing=True):
    """
    Load the CONUS-only lake dataset.

    If the dataset doesn't exist and create_if_missing=True, it will be created.

    Parameters
    ----------
    create_if_missing : bool
        If True, create the CONUS dataset if it doesn't exist

    Returns
    -------
    DataFrame
        CONUS lake data
    """
    if os.path.exists(LAKE_CONUS_PARQUET_PATH):
        print(f"Loading CONUS lake data from: {LAKE_CONUS_PARQUET_PATH}")
        lakes = pd.read_parquet(LAKE_CONUS_PARQUET_PATH)
        print(f"  Loaded {len(lakes):,} CONUS lakes")
        return lakes
    elif create_if_missing:
        print("CONUS lake dataset not found. Creating it now...")
        return create_conus_lake_dataset()
    else:
        raise FileNotFoundError(
            f"CONUS lake dataset not found: {LAKE_CONUS_PARQUET_PATH}\n"
            "Run create_conus_lake_dataset() first to create it."
        )


def recreate_conus_parquet(force=False):
    """
    Recreate the CONUS lake parquet file from the original geodatabase.

    Use this after changing filter settings or to get an unfiltered dataset.
    The new parquet will include ALL lakes (no area threshold filtering).
    Area filtering can then be applied at analysis time via min_lake_area parameter.

    Parameters
    ----------
    force : bool
        If True, recreate even if file exists. If False, asks for confirmation.

    Returns
    -------
    DataFrame
        The new CONUS lake dataset

    Example
    -------
    >>> from lake_analysis.data_loading import recreate_conus_parquet
    >>> lakes = recreate_conus_parquet(force=True)
    >>> print(f"New dataset has {len(lakes):,} lakes")
    >>> print(f"Minimum area: {lakes['AREASQKM'].min():.6f} km²")
    """
    import os

    if os.path.exists(LAKE_CONUS_PARQUET_PATH):
        if not force:
            print(f"\nExisting CONUS parquet found: {LAKE_CONUS_PARQUET_PATH}")
            current_lakes = pd.read_parquet(LAKE_CONUS_PARQUET_PATH)
            area_col = COLS.get('area', 'AREASQKM')
            if area_col in current_lakes.columns:
                print(f"  Current lake count: {len(current_lakes):,}")
                print(f"  Current min area: {current_lakes[area_col].min():.6f} km²")

            response = input("\nRecreate with unfiltered data? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return None

        # Delete existing file
        os.remove(LAKE_CONUS_PARQUET_PATH)
        print(f"\nRemoved old parquet file.")

    # Create new dataset (will use updated filter function with no area threshold)
    print("\nCreating new CONUS parquet from geodatabase...")
    print("This will include ALL lakes (no area threshold).")
    lakes = create_conus_lake_dataset(source='gdb')

    # Report results
    area_col = COLS.get('area', 'AREASQKM')
    if area_col in lakes.columns:
        print(f"\nNew dataset summary:")
        print(f"  Total lakes: {len(lakes):,}")
        print(f"  Min area: {lakes[area_col].min():.6f} km²")
        print(f"  Max area: {lakes[area_col].max():.1f} km²")

    print(f"\nTo filter at analysis time, use:")
    print(f"  run_full_analysis(min_lake_area=0.01)  # or any threshold")

    return lakes


def get_target_crs_info():
    """
    Return information about the target CRS for analysis.

    The rasters use NAD_1983_Albers (EPSG:5070 or similar), which is an
    equal-area projection ideal for area-based analyses.

    Returns
    -------
    dict
        CRS information and recommendations
    """
    return {
        'recommended_crs': 'EPSG:5070',  # NAD83 / Conus Albers
        'crs_name': 'NAD83 / Conus Albers',
        'units': 'meters',
        'properties': 'Equal-area projection, preserves area measurements',
        'notes': [
            'Your topographic rasters (elevation, slope, relief) use NAD_1983_Albers',
            'This is essentially EPSG:5070 or a custom Albers definition',
            'For consistency, reproject all data to match the raster CRS',
            'The CONUS shapefile (NAD83/EPSG:4269) needs reprojection for clipping rasters',
        ],
        'workflow': [
            '1. Load CONUS boundary (NAD83 geographic)',
            '2. Reproject CONUS to match raster CRS (Albers)',
            '3. Clip rasters using reprojected CONUS boundary',
            '4. Clip/filter lakes using same boundary',
            '5. All data now in consistent Albers CRS',
        ],
    }


def check_crs_consistency(print_report=True):
    """
    Check CRS consistency across all data sources.

    Returns
    -------
    dict
        CRS information for each data source
    """
    try:
        from .config import RASTERS, SHAPEFILES, LAKE_GDB_PATH, LAKE_FEATURE_CLASS
    except ImportError:
        from config import RASTERS, SHAPEFILES, LAKE_GDB_PATH, LAKE_FEATURE_CLASS

    report = {'rasters': {}, 'shapefiles': {}, 'lakes': None}

    # Check rasters
    for name, path in RASTERS.items():
        if path and os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    report['rasters'][name] = str(src.crs)
            except Exception as e:
                report['rasters'][name] = f"Error: {e}"

    # Check shapefiles
    for name, path in SHAPEFILES.items():
        if path and os.path.exists(path):
            try:
                gdf = gpd.read_file(path, rows=1)
                report['shapefiles'][name] = str(gdf.crs)
            except Exception as e:
                report['shapefiles'][name] = f"Error: {e}"

    # Check lakes
    if os.path.exists(LAKE_GDB_PATH):
        try:
            gdf = gpd.read_file(LAKE_GDB_PATH, layer=LAKE_FEATURE_CLASS, rows=1)
            report['lakes'] = str(gdf.crs)
        except Exception as e:
            report['lakes'] = f"Error: {e}"

    if print_report:
        print("\n" + "=" * 60)
        print("CRS CONSISTENCY CHECK")
        print("=" * 60)

        print("\nRasters:")
        for name, crs in report['rasters'].items():
            print(f"  {name}: {crs}")

        print("\nShapefiles:")
        for name, crs in report['shapefiles'].items():
            print(f"  {name}: {crs}")

        print(f"\nLakes: {report['lakes']}")

        # Check consistency
        raster_crs = set(report['rasters'].values())
        if len(raster_crs) > 1:
            print("\n⚠ WARNING: Rasters have different CRS!")
        else:
            print(f"\n✓ All rasters use: {list(raster_crs)[0]}")

        print("=" * 60)

    return report


if __name__ == "__main__":
    # Run quick data check when module is executed directly
    quick_data_check()
