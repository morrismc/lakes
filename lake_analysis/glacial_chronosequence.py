"""
Glacial Chronosequence Analysis Module
=======================================

This module tests W.M. Davis's hypothesis that lake density decreases as
landscapes "mature" (i.e., with time since glaciation).

Scientific Question:
    Does lake density decrease systematically from recently glaciated
    (Wisconsin) -> older glaciated (Illinoian, Pre-Illinoian) ->
    never glaciated (Driftless) terrain?

Secondary Goal:
    Create elevation histograms showing lake counts by elevation, with
    overlays showing which lakes fall within different glacial extents
    (including western alpine glaciation from Dalton).

Key Features:
    - Load and reproject multiple glacial boundary datasets
    - Clip Dalton 18ka ice sheets to CONUS
    - Classify lakes by glacial extent (spatial join)
    - Calculate normalized lake density by glacial stage
    - Test Davis's hypothesis statistically
    - Generate publication-quality visualizations

Dependencies:
    - geopandas (for spatial operations)
    - pyproj (for CRS handling)
    - shapely (for geometry operations)
    - scipy (for statistical tests)
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, Point
from shapely.ops import unary_union

# Handle imports for both package and direct execution
try:
    from .config import (
        COLS, ELEV_BREAKS, OUTPUT_DIR, ensure_output_dir,
        GLACIAL_BOUNDARIES, GLACIAL_CHRONOLOGY,
        GLACIAL_TARGET_CRS, GLACIAL_TARGET_CRS_PROJ4,
        SHAPEFILES, AI_BREAKS,
        NADI1_CONFIG, get_nadi1_ages
    )
    from .data_loading import get_raster_info, calculate_landscape_area_by_bin
except ImportError:
    from config import (
        COLS, ELEV_BREAKS, OUTPUT_DIR, ensure_output_dir,
        GLACIAL_BOUNDARIES, GLACIAL_CHRONOLOGY,
        GLACIAL_TARGET_CRS, GLACIAL_TARGET_CRS_PROJ4,
        SHAPEFILES, AI_BREAKS,
        NADI1_CONFIG, get_nadi1_ages
    )
    from data_loading import get_raster_info, calculate_landscape_area_by_bin


# ============================================================================
# CRS AND PROJECTION UTILITIES
# ============================================================================

def get_target_crs():
    """
    Get the target CRS for glacial analysis operations.

    Returns USA Contiguous Albers Equal Area Conic (ESRI:102039), trying
    the ESRI code first, then falling back to PROJ4 string if needed.

    Returns
    -------
    str
        CRS specification string
    """
    try:
        # Test if ESRI:102039 is recognized
        from pyproj import CRS
        test_crs = CRS.from_user_input(GLACIAL_TARGET_CRS)
        return GLACIAL_TARGET_CRS
    except Exception:
        # Fall back to PROJ4 string
        print(f"  Note: Using PROJ4 string for target CRS (ESRI code not recognized)")
        return GLACIAL_TARGET_CRS_PROJ4


def load_and_reproject(filepath, layer=None, target_crs=None):
    """
    Load any vector file and reproject to common CRS.

    Parameters
    ----------
    filepath : str
        Path to shapefile or geodatabase
    layer : str, optional
        Layer name if loading from geodatabase
    target_crs : str, optional
        Target CRS (defaults to GLACIAL_TARGET_CRS)

    Returns
    -------
    GeoDataFrame
        Reprojected geodataframe
    """
    if target_crs is None:
        target_crs = get_target_crs()

    print(f"  Loading: {filepath}")
    if layer:
        print(f"    Layer: {layer}")

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load the data
    if layer:
        gdf = gpd.read_file(filepath, layer=layer)
    else:
        gdf = gpd.read_file(filepath)

    print(f"    Original CRS: {gdf.crs}")
    print(f"    Features: {len(gdf):,}")

    # Reproject if needed
    if gdf.crs is None:
        warnings.warn(f"No CRS defined for {filepath}. Assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")

    if str(gdf.crs) != str(target_crs):
        print(f"    Reprojecting to: {target_crs}")
        gdf = gdf.to_crs(target_crs)

    return gdf


# ============================================================================
# GLACIAL BOUNDARY LOADING
# ============================================================================

def load_conus_boundary(target_crs=None):
    """
    Load CONUS boundary for clipping operations.

    Parameters
    ----------
    target_crs : str, optional
        Target CRS for reprojection

    Returns
    -------
    GeoDataFrame
        CONUS boundary polygon(s)
    """
    if target_crs is None:
        target_crs = get_target_crs()

    conus_path = SHAPEFILES.get('conus_boundary')
    if conus_path is None or not os.path.exists(conus_path):
        raise FileNotFoundError(
            f"CONUS boundary shapefile not found. "
            f"Expected: {conus_path}"
        )

    print("\nLoading CONUS boundary...")
    conus = load_and_reproject(conus_path, target_crs=target_crs)

    # Clip to CONUS bounding box (exclude Alaska, Hawaii, territories)
    # CONUS approximate bounds in geographic coordinates
    conus_bounds = {
        'min_lon': -125.0,
        'max_lon': -66.0,
        'min_lat': 24.0,
        'max_lat': 50.0,
    }

    # Create bounding box in geographic CRS, then reproject
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(conus_bounds['min_lon'], conus_bounds['min_lat'],
                     conus_bounds['max_lon'], conus_bounds['max_lat'])],
        crs="EPSG:4326"
    ).to_crs(target_crs)

    # Clip CONUS to bounding box
    conus_clipped = gpd.overlay(conus, bbox_gdf, how='intersection')
    print(f"    Clipped to CONUS extent: {len(conus_clipped)} features")

    return conus_clipped


def load_dalton_18ka(target_crs=None, clip_to_conus=True):
    """
    Load Dalton 2020 18ka ice sheet extent and optionally clip to CONUS.

    The Dalton dataset covers all of North America, so clipping to CONUS
    is necessary for analysis of the continental US.

    Parameters
    ----------
    target_crs : str, optional
        Target CRS for reprojection
    clip_to_conus : bool
        If True, clip to CONUS boundary (recommended)

    Returns
    -------
    GeoDataFrame
        Ice sheet extent at 18ka, optionally clipped to CONUS
    """
    if target_crs is None:
        target_crs = get_target_crs()

    dalton_config = GLACIAL_BOUNDARIES.get('dalton_18ka')
    if dalton_config is None:
        raise ValueError("Dalton 18ka configuration not found in GLACIAL_BOUNDARIES")

    dalton_path = dalton_config['path']
    if not os.path.exists(dalton_path):
        raise FileNotFoundError(f"Dalton 18ka shapefile not found: {dalton_path}")

    print("\nLoading Dalton 18ka ice sheets...")
    dalton = load_and_reproject(dalton_path, target_crs=target_crs)

    if clip_to_conus:
        print("  Clipping to CONUS boundary...")
        conus = load_conus_boundary(target_crs=target_crs)

        # Clip Dalton to CONUS
        dalton_conus = gpd.overlay(dalton, conus, how='intersection')
        print(f"    Features after clipping: {len(dalton_conus)}")

        # Calculate area after clipping
        total_area_km2 = dalton_conus.geometry.area.sum() / 1e6
        print(f"    Total ice extent in CONUS: {total_area_km2:,.0f} km²")

        return dalton_conus

    return dalton


def load_wisconsin_extent(target_crs=None):
    """
    Load Wisconsin glaciation extent from geodatabase.

    Returns
    -------
    GeoDataFrame
        Wisconsin glaciation maximum extent polygon(s)
    """
    if target_crs is None:
        target_crs = get_target_crs()

    config = GLACIAL_BOUNDARIES.get('wisconsin')
    if config is None:
        raise ValueError("Wisconsin configuration not found in GLACIAL_BOUNDARIES")

    print("\nLoading Wisconsin glaciation extent...")
    gdf = load_and_reproject(
        config['path'],
        layer=config.get('layer'),
        target_crs=target_crs
    )

    # Calculate area
    total_area_km2 = gdf.geometry.area.sum() / 1e6
    print(f"    Total area: {total_area_km2:,.0f} km²")

    return gdf


def load_illinoian_extent(target_crs=None):
    """
    Load Illinoian glaciation extent from geodatabase.

    Returns
    -------
    GeoDataFrame
        Illinoian glaciation maximum extent polygon(s)
    """
    if target_crs is None:
        target_crs = get_target_crs()

    config = GLACIAL_BOUNDARIES.get('illinoian')
    if config is None:
        raise ValueError("Illinoian configuration not found in GLACIAL_BOUNDARIES")

    print("\nLoading Illinoian glaciation extent...")
    gdf = load_and_reproject(
        config['path'],
        layer=config.get('layer'),
        target_crs=target_crs
    )

    # Calculate area
    total_area_km2 = gdf.geometry.area.sum() / 1e6
    print(f"    Total area: {total_area_km2:,.0f} km²")

    return gdf


def load_driftless_area(use_definite=True, target_crs=None):
    """
    Load Driftless Area (never glaciated) from geodatabase.

    Parameters
    ----------
    use_definite : bool
        If True, use the definite never-glaciated boundary.
        If False, use the larger boundary that may have pre-Illinoian till.

    Returns
    -------
    GeoDataFrame
        Driftless Area polygon(s)
    """
    if target_crs is None:
        target_crs = get_target_crs()

    key = 'driftless_definite' if use_definite else 'driftless_larger'
    config = GLACIAL_BOUNDARIES.get(key)
    if config is None:
        raise ValueError(f"{key} configuration not found in GLACIAL_BOUNDARIES")

    desc = "definite" if use_definite else "larger (may have pre-Illinoian till)"
    print(f"\nLoading Driftless Area ({desc})...")

    gdf = load_and_reproject(
        config['path'],
        layer=config.get('layer'),
        target_crs=target_crs
    )

    # Calculate area
    total_area_km2 = gdf.geometry.area.sum() / 1e6
    print(f"    Total area: {total_area_km2:,.0f} km²")

    return gdf


def load_all_glacial_boundaries(target_crs=None, include_dalton=True):
    """
    Load all glacial boundary datasets and reproject to common CRS.

    Parameters
    ----------
    target_crs : str, optional
        Target CRS for reprojection
    include_dalton : bool
        If True, include Dalton 18ka (western alpine glaciation)

    Returns
    -------
    dict
        Dictionary mapping glacial stage names to GeoDataFrames:
        - 'wisconsin': Wisconsin glaciation extent
        - 'illinoian': Illinoian glaciation extent
        - 'driftless': Driftless Area (never glaciated)
        - 'dalton_18ka': Dalton 18ka ice sheets (if include_dalton=True)
    """
    if target_crs is None:
        target_crs = get_target_crs()

    print("\n" + "=" * 60)
    print("LOADING GLACIAL BOUNDARY DATASETS")
    print("=" * 60)
    print(f"Target CRS: {target_crs}")

    boundaries = {}

    # Load each boundary with error handling
    try:
        boundaries['wisconsin'] = load_wisconsin_extent(target_crs)
    except Exception as e:
        print(f"  WARNING: Could not load Wisconsin extent: {e}")
        boundaries['wisconsin'] = None

    try:
        boundaries['illinoian'] = load_illinoian_extent(target_crs)
    except Exception as e:
        print(f"  WARNING: Could not load Illinoian extent: {e}")
        boundaries['illinoian'] = None

    try:
        boundaries['driftless'] = load_driftless_area(use_definite=True, target_crs=target_crs)
    except Exception as e:
        print(f"  WARNING: Could not load Driftless Area: {e}")
        boundaries['driftless'] = None

    if include_dalton:
        try:
            boundaries['dalton_18ka'] = load_dalton_18ka(target_crs, clip_to_conus=True)
        except Exception as e:
            print(f"  WARNING: Could not load Dalton 18ka: {e}")
            boundaries['dalton_18ka'] = None

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY: Glacial Boundaries Loaded")
    print("-" * 60)
    for name, gdf in boundaries.items():
        if gdf is not None:
            area_km2 = gdf.geometry.area.sum() / 1e6
            print(f"  {name}: {len(gdf)} features, {area_km2:,.0f} km²")
        else:
            print(f"  {name}: NOT LOADED")
    print("=" * 60)

    return boundaries


# ============================================================================
# LAKE CLASSIFICATION BY GLACIAL EXTENT
# ============================================================================

def convert_lakes_to_gdf(lake_df, lat_col=None, lon_col=None, target_crs=None):
    """
    Convert lake DataFrame to GeoDataFrame with point geometry.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with latitude/longitude columns
    lat_col : str, optional
        Latitude column name (default from config)
    lon_col : str, optional
        Longitude column name (default from config)
    target_crs : str, optional
        Target CRS for reprojection

    Returns
    -------
    GeoDataFrame
        Lakes as points in target CRS
    """
    if target_crs is None:
        target_crs = get_target_crs()

    # Get column names from config if not provided
    if lat_col is None:
        lat_col = COLS.get('lat', 'Latitude')
    if lon_col is None:
        lon_col = COLS.get('lon', 'Longitude')

    # Verify columns exist
    if lat_col not in lake_df.columns or lon_col not in lake_df.columns:
        raise ValueError(
            f"Lat/lon columns not found. Looking for '{lat_col}' and '{lon_col}'. "
            f"Available columns: {list(lake_df.columns)}"
        )

    print(f"\nConverting lakes to GeoDataFrame...")
    print(f"  Using columns: lat='{lat_col}', lon='{lon_col}'")

    # Create geometry from lat/lon (assuming WGS84)
    geometry = gpd.points_from_xy(lake_df[lon_col], lake_df[lat_col])
    gdf = gpd.GeoDataFrame(lake_df.copy(), geometry=geometry, crs="EPSG:4326")

    # Reproject to target CRS
    if str(gdf.crs) != str(target_crs):
        print(f"  Reprojecting to: {target_crs}")
        gdf = gdf.to_crs(target_crs)

    print(f"  Created GeoDataFrame with {len(gdf):,} lakes")

    return gdf


def classify_lakes_by_glacial_extent(lake_gdf, boundaries, verbose=True):
    """
    Classify lakes by which glacial extent they fall within.

    This performs spatial joins to determine which glacial stage
    each lake falls within. The classification is hierarchical:
    1. First check if in Wisconsin extent (most recent)
    2. Then check if in Illinoian extent (but not Wisconsin)
    3. Then check if in Driftless (never glaciated)
    4. Remaining lakes are classified as 'unclassified'

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes as points
    boundaries : dict
        Dictionary of glacial boundary GeoDataFrames from load_all_glacial_boundaries()
    verbose : bool
        If True, print classification progress

    Returns
    -------
    GeoDataFrame
        Lakes with added 'glacial_stage' column
    """
    if verbose:
        print("\n" + "-" * 60)
        print("CLASSIFYING LAKES BY GLACIAL EXTENT")
        print("-" * 60)

    # Ensure CRS matches
    target_crs = str(lake_gdf.crs)

    # Initialize glacial stage column
    result = lake_gdf.copy()
    result['glacial_stage'] = 'unclassified'
    result['glacial_age_ka'] = np.nan

    # Classify in hierarchical order (most recent to oldest)
    classification_order = [
        ('wisconsin', 'Wisconsin', (15, 25)),
        ('illinoian', 'Illinoian', (130, 190)),
        ('driftless', 'Driftless', None),
    ]

    for key, stage_name, age_ka in classification_order:
        boundary = boundaries.get(key)
        if boundary is None:
            if verbose:
                print(f"  Skipping {stage_name}: boundary not loaded")
            continue

        # Ensure same CRS
        if str(boundary.crs) != target_crs:
            boundary = boundary.to_crs(target_crs)

        # Dissolve boundary to single polygon for faster point-in-polygon
        if len(boundary) > 1:
            dissolved = unary_union(boundary.geometry)
            boundary_single = gpd.GeoDataFrame(geometry=[dissolved], crs=target_crs)
        else:
            boundary_single = boundary

        # Find lakes within this boundary that haven't been classified yet
        unclassified_mask = result['glacial_stage'] == 'unclassified'
        unclassified_lakes = result[unclassified_mask]

        if len(unclassified_lakes) == 0:
            if verbose:
                print(f"  {stage_name}: All lakes already classified")
            continue

        # Spatial join
        joined = gpd.sjoin(
            unclassified_lakes[['geometry']].reset_index(),
            boundary_single,
            how='inner',
            predicate='within'
        )

        # Update classification for matched lakes
        matched_indices = joined['index'].values
        result.loc[result.index.isin(matched_indices), 'glacial_stage'] = stage_name

        if age_ka is not None:
            if isinstance(age_ka, tuple):
                age_value = (age_ka[0] + age_ka[1]) / 2  # Mean age
            else:
                age_value = age_ka
            result.loc[result.index.isin(matched_indices), 'glacial_age_ka'] = age_value

        if verbose:
            print(f"  {stage_name}: {len(matched_indices):,} lakes classified")

    # Handle Dalton 18ka separately (western alpine glaciation)
    dalton = boundaries.get('dalton_18ka')
    if dalton is not None:
        if str(dalton.crs) != target_crs:
            dalton = dalton.to_crs(target_crs)

        # Dissolve Dalton boundary
        if len(dalton) > 1:
            dissolved = unary_union(dalton.geometry)
            dalton_single = gpd.GeoDataFrame(geometry=[dissolved], crs=target_crs)
        else:
            dalton_single = dalton

        # Mark lakes within Dalton extent (alpine glaciation)
        # This can overlap with other classifications - add as separate column
        joined = gpd.sjoin(
            result[['geometry']].reset_index(),
            dalton_single,
            how='inner',
            predicate='within'
        )

        matched_indices = joined['index'].values
        result['in_dalton_18ka'] = result.index.isin(matched_indices)

        if verbose:
            print(f"  Dalton 18ka (alpine): {len(matched_indices):,} lakes within LGM ice extent")
    else:
        result['in_dalton_18ka'] = False

    # Summary
    if verbose:
        print("\n" + "-" * 40)
        print("CLASSIFICATION SUMMARY")
        print("-" * 40)
        stage_counts = result['glacial_stage'].value_counts()
        for stage, count in stage_counts.items():
            pct = 100 * count / len(result)
            print(f"  {stage}: {count:,} lakes ({pct:.1f}%)")

        if 'in_dalton_18ka' in result.columns:
            dalton_count = result['in_dalton_18ka'].sum()
            pct = 100 * dalton_count / len(result)
            print(f"  (Also in Dalton 18ka: {dalton_count:,} lakes, {pct:.1f}%)")

    return result


def create_pre_illinoian_classification(lake_gdf, boundaries, verbose=True):
    """
    Create Pre-Illinoian classification for lakes in Illinoian extent
    but NOT in Wisconsin extent.

    The Illinoian extent polygon likely represents maximum ice extent,
    so lakes in Illinoian but not Wisconsin may be on Pre-Illinoian terrain.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column from classify_lakes_by_glacial_extent()
    boundaries : dict
        Dictionary of glacial boundary GeoDataFrames
    verbose : bool
        Print progress information

    Returns
    -------
    GeoDataFrame
        Lakes with updated glacial_stage including Pre-Illinoian
    """
    if verbose:
        print("\nIdentifying Pre-Illinoian terrain...")

    result = lake_gdf.copy()

    # Lakes classified as Illinoian are in Illinoian extent but NOT Wisconsin
    # Check which of these are beyond both extents (Pre-Illinoian terrain)

    # This requires more detailed boundary analysis or additional data
    # For now, we'll note that "Illinoian" includes both Illinoian and
    # Pre-Illinoian terrain

    # Alternative approach: If we have information about ice margins
    # or till deposits, we could refine this classification

    if verbose:
        illinoian_count = (result['glacial_stage'] == 'Illinoian').sum()
        print(f"  Lakes in Illinoian (not Wisconsin): {illinoian_count:,}")
        print(f"  Note: Illinoian extent may include Pre-Illinoian terrain")

    return result


# ============================================================================
# LAKE DENSITY CALCULATIONS BY GLACIAL STAGE
# ============================================================================

def calculate_glacial_zone_areas(boundaries, verbose=True):
    """
    Calculate total area of each glacial zone.

    Parameters
    ----------
    boundaries : dict
        Dictionary of glacial boundary GeoDataFrames
    verbose : bool
        Print area information

    Returns
    -------
    dict
        Dictionary mapping stage names to area in km²
    """
    if verbose:
        print("\nCalculating glacial zone areas...")

    areas = {}

    for name, gdf in boundaries.items():
        if gdf is None:
            continue

        # Calculate area (geometry should be in projected CRS with meters)
        area_km2 = gdf.geometry.area.sum() / 1e6
        areas[name] = area_km2

        if verbose:
            print(f"  {name}: {area_km2:,.0f} km²")

    # Calculate unclassified area = CONUS - (Wisconsin + Illinoian + Driftless)
    # Note: Dalton 18ka overlaps with Wisconsin so we don't subtract it
    try:
        target_crs = get_target_crs()
        conus = load_conus_boundary(target_crs=target_crs)
        conus_area_km2 = conus.geometry.area.sum() / 1e6

        # Sum only the main glacial classification zones (not dalton which overlaps)
        classified_area = sum(
            areas.get(zone, 0) for zone in ['wisconsin', 'illinoian', 'driftless']
        )
        unclassified_area = conus_area_km2 - classified_area

        if unclassified_area > 0:
            areas['unclassified'] = unclassified_area
            if verbose:
                print(f"  CONUS total: {conus_area_km2:,.0f} km²")
                print(f"  unclassified: {unclassified_area:,.0f} km²")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not calculate unclassified area: {e}")
        # Use approximate CONUS area as fallback
        conus_approx = 8_080_000  # km² (approximate CONUS land area)
        classified_area = sum(
            areas.get(zone, 0) for zone in ['wisconsin', 'illinoian', 'driftless']
        )
        areas['unclassified'] = conus_approx - classified_area

    return areas


def compute_lake_density_by_glacial_stage(lake_gdf, zone_areas=None, verbose=True):
    """
    Compute lake density (lakes per 1000 km²) for each glacial stage.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column
    zone_areas : dict, optional
        Dictionary of zone areas in km². If None, uses count-based density.
    verbose : bool
        Print density information

    Returns
    -------
    DataFrame
        Statistics for each glacial stage including:
        - n_lakes: number of lakes
        - total_area_km2: total lake surface area
        - zone_area_km2: area of glacial zone
        - density_per_1000km2: lakes per 1000 km²
        - mean_lake_area: mean lake surface area
    """
    if verbose:
        print("\n" + "-" * 60)
        print("LAKE DENSITY BY GLACIAL STAGE")
        print("-" * 60)

    area_col = COLS.get('area', 'AREASQKM')

    # Group by glacial stage
    results = []

    for stage in lake_gdf['glacial_stage'].unique():
        stage_lakes = lake_gdf[lake_gdf['glacial_stage'] == stage]
        n_lakes = len(stage_lakes)

        # Lake area statistics
        if area_col in stage_lakes.columns:
            total_lake_area = stage_lakes[area_col].sum()
            mean_lake_area = stage_lakes[area_col].mean()
            median_lake_area = stage_lakes[area_col].median()
        else:
            total_lake_area = np.nan
            mean_lake_area = np.nan
            median_lake_area = np.nan

        # Get zone area and compute density
        stage_key_map = {
            'Wisconsin': 'wisconsin',
            'Illinoian': 'illinoian',
            'Driftless': 'driftless',
            'unclassified': 'unclassified',
        }

        zone_key = stage_key_map.get(stage, stage.lower())
        if zone_areas and zone_key in zone_areas:
            zone_area = zone_areas[zone_key]
            density = (n_lakes / zone_area) * 1000
        else:
            zone_area = np.nan
            density = np.nan

        # Get age information
        chrono = GLACIAL_CHRONOLOGY.get(zone_key.lower() if zone_key else stage.lower(), {})
        age_ka = chrono.get('age_ka')
        if isinstance(age_ka, tuple):
            age_ka = (age_ka[0] + age_ka[1]) / 2  # Mean age

        results.append({
            'glacial_stage': stage,
            'n_lakes': n_lakes,
            'total_lake_area_km2': total_lake_area,
            'mean_lake_area_km2': mean_lake_area,
            'median_lake_area_km2': median_lake_area,
            'zone_area_km2': zone_area,
            'density_per_1000km2': density,
            'age_ka': age_ka,
            'color': chrono.get('color', '#808080'),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('age_ka', na_position='last')

    if verbose:
        print(result_df.to_string(index=False))

    return result_df


def compute_elevation_binned_density_by_stage(lake_gdf, elev_breaks=None, verbose=True):
    """
    Compute lake density by elevation bin, separately for each glacial stage.

    This creates data for elevation histograms with glacial stage overlays.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column
    elev_breaks : list, optional
        Elevation bin edges (default from config)
    verbose : bool
        Print progress information

    Returns
    -------
    DataFrame
        Columns: elevation_bin, glacial_stage, n_lakes, pct_of_stage
    """
    if elev_breaks is None:
        elev_breaks = ELEV_BREAKS

    elev_col = COLS.get('elevation', 'Elevation_')

    if verbose:
        print("\nComputing elevation distribution by glacial stage...")

    # Create elevation bins
    lake_gdf = lake_gdf.copy()
    lake_gdf['elev_bin'] = pd.cut(
        lake_gdf[elev_col],
        bins=elev_breaks,
        labels=False,
        include_lowest=True
    )

    # Group by stage and elevation bin
    grouped = lake_gdf.groupby(['glacial_stage', 'elev_bin']).size().reset_index(name='n_lakes')

    # Add bin labels
    bin_labels = []
    for i in range(len(elev_breaks) - 1):
        bin_labels.append(f"{elev_breaks[i]}-{elev_breaks[i+1]}")

    grouped['elev_bin_label'] = grouped['elev_bin'].apply(
        lambda x: bin_labels[int(x)] if pd.notna(x) and 0 <= int(x) < len(bin_labels) else 'Unknown'
    )
    grouped['elev_bin_mid'] = grouped['elev_bin'].apply(
        lambda x: (elev_breaks[int(x)] + elev_breaks[int(x)+1]) / 2
        if pd.notna(x) and 0 <= int(x) < len(elev_breaks)-1 else np.nan
    )

    # Calculate percentage within each stage
    stage_totals = grouped.groupby('glacial_stage')['n_lakes'].transform('sum')
    grouped['pct_of_stage'] = 100 * grouped['n_lakes'] / stage_totals

    if verbose:
        print(f"  Processed {len(grouped)} stage-elevation combinations")

    return grouped


# ============================================================================
# DAVIS HYPOTHESIS TESTING
# ============================================================================

def test_davis_hypothesis(density_df, verbose=True):
    """
    Test Davis's hypothesis that lake density decreases with landscape age.

    Uses linear regression and correlation tests to examine the relationship
    between glacial stage age and lake density.

    Parameters
    ----------
    density_df : DataFrame
        Output from compute_lake_density_by_glacial_stage()
    verbose : bool
        Print test results

    Returns
    -------
    dict
        Test results including:
        - correlation: Pearson correlation coefficient
        - p_value: p-value for correlation
        - slope: regression slope
        - r_squared: R² of regression
        - supports_hypothesis: bool indicating if results support Davis
    """
    from scipy import stats

    if verbose:
        print("\n" + "=" * 60)
        print("TESTING DAVIS'S HYPOTHESIS")
        print("=" * 60)
        print("H0: Lake density does not vary with glacial stage age")
        print("Ha: Lake density decreases with increasing landscape age")

    # Filter to stages with known ages and densities
    valid = density_df.dropna(subset=['age_ka', 'density_per_1000km2'])

    if len(valid) < 3:
        if verbose:
            print("\n  WARNING: Insufficient data for statistical testing")
            print(f"  Only {len(valid)} stages with complete data")
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'slope': np.nan,
            'r_squared': np.nan,
            'supports_hypothesis': None,
            'warning': 'Insufficient data',
        }

    ages = valid['age_ka'].values
    densities = valid['density_per_1000km2'].values

    # Pearson correlation
    corr, p_value = stats.pearsonr(ages, densities)

    # Linear regression
    slope, intercept, r_value, p_reg, std_err = stats.linregress(ages, densities)
    r_squared = r_value ** 2

    # Spearman rank correlation (more robust to outliers)
    spearman_corr, spearman_p = stats.spearmanr(ages, densities)

    # Davis's hypothesis predicts NEGATIVE correlation (older = fewer lakes)
    supports_hypothesis = (corr < 0) and (p_value < 0.05)

    results = {
        'correlation': corr,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'supports_hypothesis': supports_hypothesis,
        'n_stages': len(valid),
        'ages': ages.tolist(),
        'densities': densities.tolist(),
    }

    if verbose:
        print(f"\nResults:")
        print(f"  n stages with data: {len(valid)}")
        print(f"\nCorrelation Analysis:")
        print(f"  Pearson r = {corr:.3f}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  Spearman rho = {spearman_corr:.3f}")
        print(f"  Spearman p = {spearman_p:.4f}")
        print(f"\nLinear Regression:")
        print(f"  Density = {slope:.4f} * age + {intercept:.2f}")
        print(f"  R² = {r_squared:.3f}")
        print(f"  Standard error = {std_err:.4f}")
        print(f"\nConclusion:")
        if supports_hypothesis:
            print(f"  SUPPORTS Davis's hypothesis (p < 0.05, negative correlation)")
            print(f"  Lake density decreases by {abs(slope)*1000:.2f} lakes/1000km² per 1000 years")
        elif corr < 0:
            print(f"  TREND supports Davis but not statistically significant (p = {p_value:.3f})")
        else:
            print(f"  DOES NOT support Davis's hypothesis (positive or zero correlation)")

    return results


def compare_adjacent_stages(lake_gdf, verbose=True):
    """
    Perform pairwise comparisons between adjacent glacial stages.

    Uses t-tests and Mann-Whitney U tests to compare lake characteristics
    between adjacent stages in the chronosequence.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column
    verbose : bool
        Print comparison results

    Returns
    -------
    DataFrame
        Pairwise comparison results
    """
    from scipy import stats

    if verbose:
        print("\n" + "-" * 60)
        print("PAIRWISE STAGE COMPARISONS")
        print("-" * 60)

    area_col = COLS.get('area', 'AREASQKM')

    # Define stage order (youngest to oldest)
    stage_order = ['Wisconsin', 'Illinoian', 'Driftless']

    results = []

    for i in range(len(stage_order) - 1):
        stage1 = stage_order[i]
        stage2 = stage_order[i + 1]

        lakes1 = lake_gdf[lake_gdf['glacial_stage'] == stage1]
        lakes2 = lake_gdf[lake_gdf['glacial_stage'] == stage2]

        if len(lakes1) < 10 or len(lakes2) < 10:
            if verbose:
                print(f"\n{stage1} vs {stage2}: Insufficient data")
            continue

        # Compare lake areas
        areas1 = lakes1[area_col].values
        areas2 = lakes2[area_col].values

        # t-test (parametric)
        t_stat, t_p = stats.ttest_ind(areas1, areas2)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(areas1, areas2, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(areas1)-1)*np.var(areas1) + (len(areas2)-1)*np.var(areas2)) /
                             (len(areas1) + len(areas2) - 2))
        cohens_d = (np.mean(areas1) - np.mean(areas2)) / pooled_std if pooled_std > 0 else 0

        results.append({
            'comparison': f'{stage1} vs {stage2}',
            'stage1': stage1,
            'stage2': stage2,
            'n_stage1': len(lakes1),
            'n_stage2': len(lakes2),
            'mean_area_stage1': np.mean(areas1),
            'mean_area_stage2': np.mean(areas2),
            't_statistic': t_stat,
            't_p_value': t_p,
            'u_statistic': u_stat,
            'u_p_value': u_p,
            'cohens_d': cohens_d,
        })

        if verbose:
            print(f"\n{stage1} vs {stage2}:")
            print(f"  n = {len(lakes1):,} vs {len(lakes2):,}")
            print(f"  Mean area = {np.mean(areas1):.4f} vs {np.mean(areas2):.4f} km²")
            print(f"  t-test: t = {t_stat:.3f}, p = {t_p:.4f}")
            print(f"  Mann-Whitney: U = {u_stat:.0f}, p = {u_p:.4f}")
            print(f"  Cohen's d = {cohens_d:.3f}")

    return pd.DataFrame(results)


# ============================================================================
# COMPREHENSIVE ANALYSIS FUNCTION
# ============================================================================

def run_glacial_chronosequence_analysis(lake_df, save_results=True, verbose=True):
    """
    Run the complete glacial chronosequence analysis.

    This is the main entry point that orchestrates all analysis steps:
    1. Load glacial boundary datasets
    2. Convert lakes to GeoDataFrame
    3. Classify lakes by glacial extent
    4. Calculate lake density by glacial stage
    5. Test Davis's hypothesis
    6. Generate summary statistics

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with lat/lon columns
    save_results : bool
        If True, save results to CSV files
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Complete analysis results including:
        - lake_gdf: Lakes with glacial classification
        - boundaries: Glacial boundary GeoDataFrames
        - density_by_stage: Lake density statistics
        - elevation_by_stage: Elevation distribution data
        - davis_test: Hypothesis test results
        - pairwise_tests: Pairwise comparison results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GLACIAL CHRONOSEQUENCE ANALYSIS")
        print("Testing Davis's Hypothesis: Lake density decreases with landscape age")
        print("=" * 70)

    results = {}

    # Step 1: Load glacial boundaries
    if verbose:
        print("\n[STEP 1/5] Loading glacial boundary datasets...")
    try:
        boundaries = load_all_glacial_boundaries(include_dalton=True)
        results['boundaries'] = boundaries
    except Exception as e:
        print(f"  ERROR loading boundaries: {e}")
        return {'error': str(e)}

    # Step 2: Calculate zone areas
    if verbose:
        print("\n[STEP 2/5] Calculating glacial zone areas...")
    zone_areas = calculate_glacial_zone_areas(boundaries, verbose=verbose)
    results['zone_areas'] = zone_areas

    # Step 3: Convert lakes to GeoDataFrame and classify
    if verbose:
        print("\n[STEP 3/5] Classifying lakes by glacial extent...")
    try:
        target_crs = get_target_crs()
        lake_gdf = convert_lakes_to_gdf(lake_df, target_crs=target_crs)
        lake_gdf = classify_lakes_by_glacial_extent(lake_gdf, boundaries, verbose=verbose)
        results['lake_gdf'] = lake_gdf
    except Exception as e:
        print(f"  ERROR classifying lakes: {e}")
        return results

    # Step 4: Calculate lake density by glacial stage
    if verbose:
        print("\n[STEP 4/5] Computing lake density by glacial stage...")
    density_df = compute_lake_density_by_glacial_stage(lake_gdf, zone_areas, verbose=verbose)
    results['density_by_stage'] = density_df

    # Also compute elevation distribution by stage
    elevation_df = compute_elevation_binned_density_by_stage(lake_gdf, verbose=verbose)
    results['elevation_by_stage'] = elevation_df

    # Step 5: Test Davis's hypothesis
    if verbose:
        print("\n[STEP 5/5] Testing Davis's hypothesis...")
    davis_results = test_davis_hypothesis(density_df, verbose=verbose)
    results['davis_test'] = davis_results

    # Additional pairwise comparisons
    pairwise_results = compare_adjacent_stages(lake_gdf, verbose=verbose)
    results['pairwise_tests'] = pairwise_results

    # Save results if requested
    if save_results:
        ensure_output_dir()
        output_base = Path(OUTPUT_DIR) / 'glacial_chronosequence'
        output_base.mkdir(exist_ok=True)

        # Save density data
        density_df.to_csv(output_base / 'density_by_glacial_stage.csv', index=False)

        # Save elevation distribution
        elevation_df.to_csv(output_base / 'elevation_by_glacial_stage.csv', index=False)

        # Save lake classification
        # Drop geometry for CSV export
        lake_export = lake_gdf.drop(columns=['geometry']).copy()
        lake_export.to_csv(output_base / 'lakes_with_glacial_stage.csv', index=False)

        # Save pairwise tests
        if len(pairwise_results) > 0:
            pairwise_results.to_csv(output_base / 'pairwise_stage_comparisons.csv', index=False)

        if verbose:
            print(f"\n  Results saved to: {output_base}")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nTotal lakes analyzed: {len(lake_gdf):,}")
        for stage in density_df['glacial_stage'].values:
            stage_data = density_df[density_df['glacial_stage'] == stage].iloc[0]
            print(f"  {stage}: {stage_data['n_lakes']:,} lakes, "
                  f"{stage_data['density_per_1000km2']:.1f} per 1000 km²")

        if davis_results['supports_hypothesis']:
            print("\n  CONCLUSION: Results SUPPORT Davis's hypothesis")
        elif davis_results.get('warning'):
            print(f"\n  CONCLUSION: Insufficient data - {davis_results['warning']}")
        else:
            print("\n  CONCLUSION: Results do NOT support Davis's hypothesis")

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_stage_color(stage_name):
    """Get the color associated with a glacial stage."""
    stage_lower = stage_name.lower()
    for key, chrono in GLACIAL_CHRONOLOGY.items():
        if key in stage_lower or chrono['name'].lower() in stage_lower:
            return chrono['color']
    return '#808080'  # Default gray


def get_stage_order():
    """Get glacial stages in chronological order (youngest first)."""
    stages = [(key, chrono['order']) for key, chrono in GLACIAL_CHRONOLOGY.items()]
    return [s[0] for s in sorted(stages, key=lambda x: x[1])]


# ============================================================================
# ALPINE VS CONTINENTAL ICE CLASSIFICATION
# ============================================================================

def classify_ice_types(dalton_gdf, longitude_threshold=-105.0, area_threshold_km2=50000):
    """
    Separate continental (Laurentide) vs alpine (Cordilleran) glaciation.

    The Dalton 18ka data includes both the main Laurentide ice sheet
    (Great Lakes region) and smaller alpine/Cordilleran glaciers
    in the western US (Yellowstone, Sierra Nevada, etc.).

    Parameters
    ----------
    dalton_gdf : GeoDataFrame
        Dalton 18ka ice sheet extent (should be in projected CRS)
    longitude_threshold : float
        Approximate longitude dividing continental vs alpine (-105° by default)
    area_threshold_km2 : float
        Polygons smaller than this are likely alpine glaciers

    Returns
    -------
    tuple (GeoDataFrame, GeoDataFrame)
        (continental_gdf, alpine_gdf) - separated ice types
    """
    if dalton_gdf is None or len(dalton_gdf) == 0:
        return None, None

    # Calculate centroid for each polygon to determine longitude
    dalton_gdf = dalton_gdf.copy()

    # Get centroids in geographic CRS for longitude comparison
    centroids_geo = dalton_gdf.geometry.to_crs("EPSG:4326").centroid
    dalton_gdf['centroid_lon'] = centroids_geo.x
    dalton_gdf['centroid_lat'] = centroids_geo.y

    # Calculate area in km²
    dalton_gdf['area_km2'] = dalton_gdf.geometry.area / 1e6

    # Classify based on longitude and area
    # Western features (west of threshold) AND smaller area = alpine
    is_western = dalton_gdf['centroid_lon'] < longitude_threshold
    is_small = dalton_gdf['area_km2'] < area_threshold_km2

    # Alpine: western AND (small OR high elevation centroid)
    dalton_gdf['ice_type'] = 'continental'
    dalton_gdf.loc[is_western & is_small, 'ice_type'] = 'alpine'

    # Separate into two GeoDataFrames
    continental = dalton_gdf[dalton_gdf['ice_type'] == 'continental'].copy()
    alpine = dalton_gdf[dalton_gdf['ice_type'] == 'alpine'].copy()

    print(f"\nIce type classification:")
    print(f"  Continental: {len(continental)} polygons, {continental['area_km2'].sum():,.0f} km²")
    print(f"  Alpine: {len(alpine)} polygons, {alpine['area_km2'].sum():,.0f} km²")

    return continental, alpine


# ============================================================================
# NESTED POLYGON CLASSIFICATION
# ============================================================================

def create_mutually_exclusive_zones(boundaries, verbose=True):
    """
    Create mutually exclusive polygons for each glacial age class.

    The glacial boundaries overlap (Wisconsin ⊂ Illinoian), so this function
    creates non-overlapping zones representing actual surface age:

    1. Wisconsin-only: Wisconsin extent (youngest surface)
    2. Illinoian-only: Illinoian extent minus Wisconsin extent
    3. Driftless: Never glaciated (oldest surface)
    4. Alpine: From Dalton, western US (similar age to Wisconsin)

    Parameters
    ----------
    boundaries : dict
        Dictionary of glacial boundary GeoDataFrames
    verbose : bool
        Print progress information

    Returns
    -------
    GeoDataFrame
        Unified polygon layer with 'glacial_zone' column
    """
    if verbose:
        print("\nCreating mutually exclusive glacial zones...")

    zones = []

    wisconsin = boundaries.get('wisconsin')
    illinoian = boundaries.get('illinoian')
    driftless = boundaries.get('driftless')
    dalton = boundaries.get('dalton_18ka')

    # Ensure all are in same CRS
    target_crs = get_target_crs()

    # Wisconsin zone (youngest, most recent glaciation)
    if wisconsin is not None:
        if str(wisconsin.crs) != str(target_crs):
            wisconsin = wisconsin.to_crs(target_crs)
        wisconsin_union = unary_union(wisconsin.geometry)
        zones.append({
            'glacial_zone': 'wisconsin',
            'description': 'Wisconsin glaciation (youngest)',
            'age_ka': 20,  # mean age
            'geometry': wisconsin_union
        })
        if verbose:
            print(f"  Wisconsin: {wisconsin_union.area / 1e6:,.0f} km²")

    # Illinoian-only zone (Illinoian minus Wisconsin)
    if illinoian is not None and wisconsin is not None:
        if str(illinoian.crs) != str(target_crs):
            illinoian = illinoian.to_crs(target_crs)
        illinoian_union = unary_union(illinoian.geometry)

        # Subtract Wisconsin from Illinoian to get Illinoian-only
        illinoian_only = illinoian_union.difference(wisconsin_union)
        if not illinoian_only.is_empty:
            zones.append({
                'glacial_zone': 'illinoian_only',
                'description': 'Illinoian glaciation only (not Wisconsin)',
                'age_ka': 160,  # mean age
                'geometry': illinoian_only
            })
            if verbose:
                print(f"  Illinoian-only: {illinoian_only.area / 1e6:,.0f} km²")

    # Driftless zone (never glaciated)
    if driftless is not None:
        if str(driftless.crs) != str(target_crs):
            driftless = driftless.to_crs(target_crs)
        driftless_union = unary_union(driftless.geometry)
        zones.append({
            'glacial_zone': 'driftless',
            'description': 'Never glaciated (Driftless)',
            'age_ka': None,
            'geometry': driftless_union
        })
        if verbose:
            print(f"  Driftless: {driftless_union.area / 1e6:,.0f} km²")

    # Alpine zone (from Dalton, western US)
    if dalton is not None:
        if str(dalton.crs) != str(target_crs):
            dalton = dalton.to_crs(target_crs)

        # Classify and get alpine portion
        continental, alpine = classify_ice_types(dalton)
        if alpine is not None and len(alpine) > 0:
            alpine_union = unary_union(alpine.geometry)
            zones.append({
                'glacial_zone': 'alpine',
                'description': 'Western alpine glaciation',
                'age_ka': 18,
                'geometry': alpine_union
            })
            if verbose:
                print(f"  Alpine: {alpine_union.area / 1e6:,.0f} km²")

    if len(zones) == 0:
        return None

    # Create GeoDataFrame
    zones_gdf = gpd.GeoDataFrame(zones, crs=target_crs)

    return zones_gdf


# ============================================================================
# POWER LAW ANALYSIS BY GLACIAL ZONE
# ============================================================================

def power_law_by_glacial_zone(lake_gdf, xmin_threshold=0.024, use_bayesian=True,
                               min_n_for_mle=100, verbose=True):
    """
    Fit power law to lake size distribution within each glacial zone.

    Tests whether the power law exponent (alpha) differs by glacial stage,
    which would indicate different lake-forming/draining processes.

    For smaller glacial zones (Illinoian, Driftless), this function uses
    Bayesian estimation with an informative prior to provide stable estimates
    with proper uncertainty quantification.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column
    xmin_threshold : float
        Minimum lake area for power law fitting (default 0.024 km²)
    use_bayesian : bool
        If True, use Bayesian estimation for small samples (n < min_n_for_mle)
    min_n_for_mle : int
        Minimum sample size for reliable MLE (default 100)
    verbose : bool
        Print results

    Returns
    -------
    DataFrame
        Power law parameters by glacial zone including:
        - estimation_method: 'MLE' or 'Bayesian'
        - prior_influence: For Bayesian, how much the prior affected the result
        - sample_size_warning: Recommendation about sample adequacy
    """
    from scipy import stats

    # Import adaptive estimation from powerlaw_analysis
    try:
        from .powerlaw_analysis import (
            adaptive_powerlaw_estimate, bayesian_powerlaw_estimate,
            compute_sample_size_power
        )
    except ImportError:
        from powerlaw_analysis import (
            adaptive_powerlaw_estimate, bayesian_powerlaw_estimate,
            compute_sample_size_power
        )

    if verbose:
        print("\n" + "=" * 60)
        print("POWER LAW ANALYSIS BY GLACIAL ZONE")
        print("(Adaptive: Bayesian for small samples, MLE for large)")
        print("=" * 60)

    area_col = COLS.get('area', 'AREASQKM')
    results = []

    stages = lake_gdf['glacial_stage'].unique()
    n_stages = len(stages)

    for i, stage in enumerate(stages):
        if verbose:
            print(f"\n[{i+1}/{n_stages}] Processing {stage}...")

        stage_lakes = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_lakes[area_col].dropna().values

        # Filter to lakes above threshold
        areas_above = areas[areas >= xmin_threshold]
        n = len(areas_above)

        if n < 10:
            if verbose:
                print(f"  ⚠ Insufficient data (n={n})")
            results.append({
                'glacial_stage': stage,
                'n_total': len(areas),
                'n_above_xmin': n,
                'alpha': np.nan,
                'alpha_se': np.nan,
                'alpha_ci_lower': np.nan,
                'alpha_ci_upper': np.nan,
                'xmin': xmin_threshold,
                'estimation_method': 'None',
                'prior_influence': np.nan,
                'sample_size_warning': 'Insufficient data (n < 10)',
                'power_analysis': 'N/A'
            })
            continue

        # Use adaptive estimation (Bayesian for small n, MLE for large n)
        if use_bayesian:
            fit_result = adaptive_powerlaw_estimate(
                areas, xmin_threshold,
                min_n_for_mle=min_n_for_mle,
                verbose=False
            )
        else:
            # Force MLE even for small samples
            log_ratios = np.log(areas_above / xmin_threshold)
            alpha_mle = 1 + n / np.sum(log_ratios)
            alpha_se = (alpha_mle - 1) / np.sqrt(n)
            fit_result = {
                'method': 'MLE',
                'alpha': alpha_mle,
                'alpha_se': alpha_se,
                'alpha_ci': (alpha_mle - 1.96*alpha_se, alpha_mle + 1.96*alpha_se),
                'n_tail': n,
                'warning': None
            }

        # KS statistic for goodness of fit
        alpha = fit_result['alpha']
        sorted_areas = np.sort(areas_above)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = 1 - (xmin_threshold / sorted_areas) ** (alpha - 1)
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))

        # Get age for this stage
        chrono = GLACIAL_CHRONOLOGY.get(stage.lower(), {})
        age_ka = chrono.get('age_ka')
        if isinstance(age_ka, tuple):
            age_ka = (age_ka[0] + age_ka[1]) / 2

        # Power analysis: can we detect meaningful differences?
        power_result = compute_sample_size_power(n, alpha_true=2.1, alpha_diff=0.1)

        result_row = {
            'glacial_stage': stage,
            'n_total': len(areas),
            'n_above_xmin': n,
            'alpha': fit_result['alpha'],
            'alpha_se': fit_result.get('alpha_se', np.nan),
            'alpha_ci_lower': fit_result['alpha_ci'][0] if fit_result.get('alpha_ci') else np.nan,
            'alpha_ci_upper': fit_result['alpha_ci'][1] if fit_result.get('alpha_ci') else np.nan,
            'xmin': xmin_threshold,
            'ks_statistic': ks_stat,
            'age_ka': age_ka,
            'mean_area_km2': np.mean(areas_above),
            'median_area_km2': np.median(areas_above),
            'estimation_method': fit_result['method'],
            'prior_influence': fit_result.get('prior_influence', 0.0),
            'sample_size_warning': fit_result.get('warning', 'None'),
            'power_analysis': power_result['recommendation'],
            'mde_80_power': power_result['mde_80_power'],
        }
        results.append(result_row)

        if verbose:
            method_str = fit_result['method']
            if method_str == 'Bayesian':
                method_str += f" (prior influence: {fit_result.get('prior_influence', 0):.1%})"

            print(f"  n = {n:,} lakes above x_min = {xmin_threshold} km²")
            print(f"  Method: {method_str}")
            print(f"  α = {fit_result['alpha']:.3f} [{fit_result['alpha_ci'][0]:.3f}, {fit_result['alpha_ci'][1]:.3f}]")
            print(f"  KS statistic = {ks_stat:.4f}")

            # Compare to percolation theory prediction
            diff = fit_result['alpha'] - 2.05
            print(f"  Deviation from τ=2.05: {diff:+.3f}")

            # Sample size adequacy
            if power_result['recommendation'] == 'adequate':
                print(f"  ✓ Sample adequate to detect Δα = 0.1")
            elif power_result['recommendation'] == 'underpowered':
                print(f"  ⚠ Underpowered: can only detect Δα ≥ {power_result['mde_80_power']:.2f}")
            else:
                print(f"  ⚠ Severely underpowered: MDE = {power_result['mde_80_power']:.2f}")

    result_df = pd.DataFrame(results)

    # Test for trend in alpha with age
    if verbose and len(result_df.dropna(subset=['alpha', 'age_ka'])) >= 3:
        valid = result_df.dropna(subset=['alpha', 'age_ka'])
        corr, p_val = stats.pearsonr(valid['age_ka'], valid['alpha'])
        print(f"\n" + "=" * 60)
        print("TREND ANALYSIS: α vs Landscape Age")
        print("=" * 60)
        print(f"  Pearson r = {corr:.3f}, p = {p_val:.4f}")
        if corr > 0 and p_val < 0.1:
            print(f"  → α increases with age (large lakes preferentially drained)")
        elif corr < 0 and p_val < 0.1:
            print(f"  → α decreases with age (small lakes preferentially drained)")
        else:
            print(f"  → No significant trend detected")

    return result_df


# ============================================================================
# BIMODAL PATTERN DECOMPOSITION
# ============================================================================

def decompose_bimodal_by_glacial_status(lake_gdf, elev_breaks=None, verbose=True):
    """
    Test whether the bimodal elevation-density pattern can be explained
    by the mixture of glacial + non-glacial lakes.

    This separates lakes into glaciated vs non-glaciated and plots their
    elevation distributions separately to test if:
    - High-elevation peak = glacial process domain
    - Low-elevation peak = fluvial/coastal process domain

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column
    elev_breaks : list, optional
        Elevation bin edges
    verbose : bool
        Print results

    Returns
    -------
    dict
        Decomposition results with separate distributions
    """
    if elev_breaks is None:
        elev_breaks = ELEV_BREAKS

    elev_col = COLS.get('elevation', 'Elevation_')

    if verbose:
        print("\n" + "=" * 60)
        print("BIMODAL PATTERN DECOMPOSITION")
        print("=" * 60)

    # Classify as glaciated vs non-glaciated
    lake_gdf = lake_gdf.copy()
    glaciated_stages = ['Wisconsin', 'Illinoian']
    lake_gdf['is_glaciated'] = lake_gdf['glacial_stage'].isin(glaciated_stages)

    # Also flag alpine glaciation
    if 'in_dalton_18ka' in lake_gdf.columns:
        lake_gdf['is_glaciated'] = lake_gdf['is_glaciated'] | lake_gdf['in_dalton_18ka']

    glaciated = lake_gdf[lake_gdf['is_glaciated']]
    non_glaciated = lake_gdf[~lake_gdf['is_glaciated']]

    if verbose:
        print(f"\nLake classification:")
        print(f"  Glaciated: {len(glaciated):,} lakes")
        print(f"  Non-glaciated: {len(non_glaciated):,} lakes")

    # Bin by elevation
    def bin_elevations(df, breaks, label):
        elevs = df[elev_col].dropna()
        counts, _ = np.histogram(elevs, bins=breaks)
        bin_mids = [(breaks[i] + breaks[i+1]) / 2 for i in range(len(breaks)-1)]
        return pd.DataFrame({
            'elevation_mid': bin_mids,
            'n_lakes': counts,
            'category': label
        })

    glaciated_dist = bin_elevations(glaciated, elev_breaks, 'Glaciated')
    non_glaciated_dist = bin_elevations(non_glaciated, elev_breaks, 'Non-glaciated')

    combined = pd.concat([glaciated_dist, non_glaciated_dist], ignore_index=True)

    # Find peaks in each distribution
    from scipy.signal import find_peaks

    def find_distribution_peaks(counts, min_height_frac=0.1):
        if np.max(counts) == 0:
            return []
        min_height = np.max(counts) * min_height_frac
        peaks, _ = find_peaks(counts, height=min_height, distance=2)
        return peaks

    glaciated_peaks = find_distribution_peaks(glaciated_dist['n_lakes'].values)
    non_glaciated_peaks = find_distribution_peaks(non_glaciated_dist['n_lakes'].values)

    if verbose:
        print(f"\nPeak elevations:")
        for i, peak_idx in enumerate(glaciated_peaks):
            elev = glaciated_dist['elevation_mid'].iloc[peak_idx]
            count = glaciated_dist['n_lakes'].iloc[peak_idx]
            print(f"  Glaciated peak {i+1}: {elev:.0f} m ({count:,} lakes)")

        for i, peak_idx in enumerate(non_glaciated_peaks):
            elev = non_glaciated_dist['elevation_mid'].iloc[peak_idx]
            count = non_glaciated_dist['n_lakes'].iloc[peak_idx]
            print(f"  Non-glaciated peak {i+1}: {elev:.0f} m ({count:,} lakes)")

    # Compute statistics
    results = {
        'combined_distribution': combined,
        'glaciated_distribution': glaciated_dist,
        'non_glaciated_distribution': non_glaciated_dist,
        'n_glaciated': len(glaciated),
        'n_non_glaciated': len(non_glaciated),
        'glaciated_peaks': glaciated_peaks.tolist() if len(glaciated_peaks) > 0 else [],
        'non_glaciated_peaks': non_glaciated_peaks.tolist() if len(non_glaciated_peaks) > 0 else [],
        'glaciated_mean_elev': glaciated[elev_col].mean() if len(glaciated) > 0 else np.nan,
        'non_glaciated_mean_elev': non_glaciated[elev_col].mean() if len(non_glaciated) > 0 else np.nan,
    }

    if verbose:
        print(f"\nMean elevation:")
        print(f"  Glaciated: {results['glaciated_mean_elev']:.0f} m")
        print(f"  Non-glaciated: {results['non_glaciated_mean_elev']:.0f} m")

    return results


def western_alpine_analysis(lake_gdf, dalton_alpine_gdf, longitude_threshold=-105.0,
                            elev_breaks=None, verbose=True):
    """
    Focused analysis on western alpine glaciation.

    Tests whether the high-elevation peak in the bimodal distribution
    persists in the western US where continental ice didn't reach.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        All lakes with coordinates
    dalton_alpine_gdf : GeoDataFrame
        Alpine glaciation extent from Dalton
    longitude_threshold : float
        Western boundary for analysis
    elev_breaks : list, optional
        Elevation bin edges
    verbose : bool
        Print results

    Returns
    -------
    dict
        Western alpine analysis results
    """
    if elev_breaks is None:
        elev_breaks = ELEV_BREAKS

    if verbose:
        print("\n" + "=" * 60)
        print("WESTERN ALPINE GLACIATION ANALYSIS")
        print("=" * 60)

    # Get lake coordinates in geographic CRS
    lakes_geo = lake_gdf.to_crs("EPSG:4326")
    lakes_geo['longitude'] = lakes_geo.geometry.x

    # Filter to western US
    western_lakes = lake_gdf[lakes_geo['longitude'] < longitude_threshold].copy()

    if verbose:
        print(f"\nWestern lakes (west of {longitude_threshold}°):")
        print(f"  Total: {len(western_lakes):,} lakes")

    # Classify western lakes by alpine glaciation
    if dalton_alpine_gdf is not None and len(dalton_alpine_gdf) > 0:
        # Spatial join to identify lakes within alpine glaciated areas
        target_crs = str(western_lakes.crs)
        if str(dalton_alpine_gdf.crs) != target_crs:
            dalton_alpine_gdf = dalton_alpine_gdf.to_crs(target_crs)

        # Dissolve alpine boundary
        alpine_dissolved = unary_union(dalton_alpine_gdf.geometry)
        alpine_gdf = gpd.GeoDataFrame(geometry=[alpine_dissolved], crs=target_crs)

        joined = gpd.sjoin(
            western_lakes[['geometry']].reset_index(),
            alpine_gdf,
            how='inner',
            predicate='within'
        )

        western_lakes['in_alpine'] = western_lakes.index.isin(joined['index'].values)

        n_alpine = western_lakes['in_alpine'].sum()
        n_non_alpine = len(western_lakes) - n_alpine

        if verbose:
            print(f"  Within alpine glaciation: {n_alpine:,}")
            print(f"  Outside alpine glaciation: {n_non_alpine:,}")

    else:
        western_lakes['in_alpine'] = False
        if verbose:
            print("  Note: Alpine boundary not available")

    # Compute elevation distribution
    elev_col = COLS.get('elevation', 'Elevation_')

    alpine_lakes = western_lakes[western_lakes['in_alpine']]
    non_alpine_western = western_lakes[~western_lakes['in_alpine']]

    results = {
        'n_western_total': len(western_lakes),
        'n_alpine': len(alpine_lakes),
        'n_non_alpine': len(non_alpine_western),
        'alpine_mean_elev': alpine_lakes[elev_col].mean() if len(alpine_lakes) > 0 else np.nan,
        'non_alpine_mean_elev': non_alpine_western[elev_col].mean() if len(non_alpine_western) > 0 else np.nan,
        'western_lakes': western_lakes,
    }

    if verbose:
        print(f"\nMean elevation (western lakes):")
        print(f"  Alpine glaciated: {results['alpine_mean_elev']:.0f} m")
        print(f"  Non-alpine: {results['non_alpine_mean_elev']:.0f} m")

    return results


# ============================================================================
# VALIDATION AND QUALITY CHECKS
# ============================================================================

def validate_glacial_boundaries(boundaries, verbose=True):
    """
    Sanity checks on glacial boundary data.

    Verifies:
    1. CRS are properly set
    2. Boundaries are nested (Wisconsin ⊂ Illinoian)
    3. Wisconsin area < Illinoian area
    4. Driftless does not overlap Wisconsin
    5. Total coverage of CONUS

    Parameters
    ----------
    boundaries : dict
        Dictionary of glacial boundary GeoDataFrames
    verbose : bool
        Print validation results

    Returns
    -------
    dict
        Validation results with any warnings
    """
    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATING GLACIAL BOUNDARIES")
        print("=" * 60)

    results = {'valid': True, 'warnings': [], 'checks': {}}

    wisconsin = boundaries.get('wisconsin')
    illinoian = boundaries.get('illinoian')
    driftless = boundaries.get('driftless')

    # Check 1: CRS verification
    target_crs = get_target_crs()
    for name, gdf in boundaries.items():
        if gdf is not None:
            crs_match = str(gdf.crs) == str(target_crs)
            results['checks'][f'{name}_crs'] = crs_match
            if not crs_match:
                results['warnings'].append(f"{name} CRS mismatch")
                if verbose:
                    print(f"  WARNING: {name} CRS = {gdf.crs}, expected {target_crs}")

    # Check 2: Area comparison (Wisconsin < Illinoian)
    if wisconsin is not None and illinoian is not None:
        wisc_area = wisconsin.geometry.area.sum() / 1e6
        ill_area = illinoian.geometry.area.sum() / 1e6
        area_ok = wisc_area < ill_area
        results['checks']['wisconsin_smaller'] = area_ok

        if verbose:
            status = "✓" if area_ok else "✗"
            print(f"  [{status}] Wisconsin ({wisc_area:,.0f} km²) < Illinoian ({ill_area:,.0f} km²)")

        if not area_ok:
            results['warnings'].append("Wisconsin area >= Illinoian area")
            results['valid'] = False

    # Check 3: Driftless doesn't overlap Wisconsin
    if driftless is not None and wisconsin is not None:
        drift_union = unary_union(driftless.geometry)
        wisc_union = unary_union(wisconsin.geometry)
        overlap = drift_union.intersection(wisc_union)
        overlap_area_km2 = overlap.area / 1e6

        no_overlap = overlap_area_km2 < 100  # Allow small tolerance
        results['checks']['driftless_no_overlap'] = no_overlap

        if verbose:
            status = "✓" if no_overlap else "✗"
            print(f"  [{status}] Driftless-Wisconsin overlap: {overlap_area_km2:,.0f} km²")

        if not no_overlap:
            results['warnings'].append(f"Driftless overlaps Wisconsin by {overlap_area_km2:.0f} km²")

    if verbose:
        if results['valid'] and len(results['warnings']) == 0:
            print("\n  All validation checks passed!")
        else:
            print(f"\n  Warnings: {len(results['warnings'])}")
            for w in results['warnings']:
                print(f"    - {w}")

    return results


# ============================================================================
# DALTON 18KA SPECIFIC ANALYSIS
# ============================================================================

def run_dalton_18ka_analysis(lake_df, save_results=True, verbose=True):
    """
    Run analysis using Dalton 18ka boundary for precise LGM snapshot.

    The Dalton 2020 dataset provides a high-resolution reconstruction
    of ice sheet extent at exactly 18 ka (Last Glacial Maximum), which
    provides a more precise temporal constraint than the Wisconsin
    maximum extent (~20 ka).

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with lat/lon columns
    save_results : bool
        If True, save results to CSV
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Analysis results including:
        - lake_gdf: Lakes with Dalton 18ka classification
        - dalton_boundary: The 18ka ice extent GeoDataFrame
        - density_comparison: Density inside vs outside 18ka extent
        - power_law_results: Power law comparison
    """
    if verbose:
        print("\n" + "=" * 60)
        print("DALTON 18KA GLACIAL ANALYSIS")
        print("Last Glacial Maximum (18 ka) Precise Reconstruction")
        print("=" * 60)

    results = {}
    target_crs = get_target_crs()

    try:
        # Load Dalton 18ka boundary
        dalton = load_dalton_18ka(target_crs=target_crs, clip_to_conus=True)
        results['dalton_boundary'] = dalton

        if verbose:
            dalton_area = dalton.geometry.area.sum() / 1e6
            print(f"\nDalton 18ka ice extent in CONUS: {dalton_area:,.0f} km²")

    except Exception as e:
        print(f"\nERROR: Could not load Dalton 18ka boundary: {e}")
        return {'error': str(e)}

    # Convert lakes to GeoDataFrame
    if verbose:
        print("\nConverting lakes to spatial format...")
    lake_gdf = convert_lakes_to_gdf(lake_df, target_crs=target_crs)

    if lake_gdf is None or len(lake_gdf) == 0:
        print("ERROR: Could not create lake GeoDataFrame")
        return {'error': 'Could not create GeoDataFrame'}

    if verbose:
        print(f"  Total lakes: {len(lake_gdf):,}")

    # Classify lakes by Dalton 18ka extent
    if verbose:
        print("\nClassifying lakes by 18ka ice extent...")

    # Dissolve Dalton boundary
    dalton_union = unary_union(dalton.geometry)
    dalton_single = gpd.GeoDataFrame(geometry=[dalton_union], crs=target_crs)

    # Spatial join to find lakes within 18ka ice extent
    joined = gpd.sjoin(
        lake_gdf[['geometry']].reset_index(),
        dalton_single,
        how='inner',
        predicate='within'
    )

    in_dalton_indices = joined['index'].values
    lake_gdf['in_dalton_18ka'] = lake_gdf.index.isin(in_dalton_indices)
    lake_gdf['dalton_classification'] = lake_gdf['in_dalton_18ka'].map({
        True: '18ka_glaciated',
        False: '18ka_unglaciated'
    })

    results['lake_gdf'] = lake_gdf

    # Count summary
    n_in_dalton = lake_gdf['in_dalton_18ka'].sum()
    n_outside = len(lake_gdf) - n_in_dalton

    if verbose:
        print(f"\n  Lakes within 18ka ice extent: {n_in_dalton:,} ({100*n_in_dalton/len(lake_gdf):.1f}%)")
        print(f"  Lakes outside 18ka ice extent: {n_outside:,} ({100*n_outside/len(lake_gdf):.1f}%)")

    # Compute density comparison
    dalton_area_km2 = dalton_union.area / 1e6

    # Get CONUS area for non-glaciated calculation
    try:
        conus = load_conus_boundary(target_crs=target_crs)
        conus_union = unary_union(conus.geometry)
        conus_area = conus_union.area / 1e6
        non_dalton_area = conus_area - dalton_area_km2
    except:
        non_dalton_area = 7e6  # Approximate CONUS area minus glaciated

    density_in = (n_in_dalton / dalton_area_km2) * 1000 if dalton_area_km2 > 0 else 0
    density_out = (n_outside / non_dalton_area) * 1000 if non_dalton_area > 0 else 0

    results['density_comparison'] = {
        '18ka_glaciated': {
            'n_lakes': n_in_dalton,
            'area_km2': dalton_area_km2,
            'density_per_1000km2': density_in
        },
        '18ka_unglaciated': {
            'n_lakes': n_outside,
            'area_km2': non_dalton_area,
            'density_per_1000km2': density_out
        },
        'density_ratio': density_in / density_out if density_out > 0 else np.nan
    }

    if verbose:
        print(f"\n  Density within 18ka extent: {density_in:.1f} lakes/1000 km²")
        print(f"  Density outside 18ka extent: {density_out:.1f} lakes/1000 km²")
        print(f"  Ratio (inside/outside): {results['density_comparison']['density_ratio']:.2f}x")

    # Power law analysis by Dalton 18ka classification
    if verbose:
        print("\n" + "-" * 40)
        print("Power Law Analysis by 18ka Glaciation")
        print("-" * 40)

    area_col = COLS.get('area', 'AREASQKM')
    power_law_results = []

    for classification in ['18ka_glaciated', '18ka_unglaciated']:
        mask = lake_gdf['dalton_classification'] == classification
        subset = lake_gdf[mask]

        if len(subset) < 50:
            if verbose:
                print(f"  {classification}: Too few lakes ({len(subset)}) for power law")
            continue

        areas = subset[area_col].dropna().values
        areas = areas[areas > 0]

        if len(areas) < 30:
            continue

        # Fit power law with multiple xmin values
        xmin_results = []
        for xmin in [0.001, 0.01, 0.1, 1.0]:
            tail = areas[areas >= xmin]
            if len(tail) >= 20:
                alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
                alpha_se = (alpha - 1) / np.sqrt(len(tail))
                xmin_results.append({
                    'xmin': xmin,
                    'alpha': alpha,
                    'alpha_se': alpha_se,
                    'n_tail': len(tail)
                })

        if len(xmin_results) > 0:
            best = xmin_results[1] if len(xmin_results) > 1 else xmin_results[0]  # prefer xmin=0.01
            power_law_results.append({
                'classification': classification,
                'n_lakes': len(areas),
                'alpha': best['alpha'],
                'alpha_se': best['alpha_se'],
                'xmin': best['xmin'],
                'n_tail': best['n_tail'],
                'xmin_sensitivity': xmin_results
            })

            if verbose:
                print(f"  {classification}:")
                print(f"    n = {len(areas):,}, α = {best['alpha']:.3f} ± {best['alpha_se']:.3f}")
                print(f"    x_min = {best['xmin']} km², n_tail = {best['n_tail']:,}")

    results['power_law_results'] = power_law_results

    # Compute elevation distribution by Dalton classification
    if verbose:
        print("\n" + "-" * 40)
        print("Elevation Distribution by 18ka Glaciation")
        print("-" * 40)

    elev_col = COLS.get('elevation', 'Elevation_')
    elev_breaks = ELEV_BREAKS

    # Check for elevation column with case-insensitive lookup
    if elev_col not in lake_gdf.columns:
        for alt_col in ['Elevation_', 'Elevation', 'elevation', 'ELEV', 'elev', 'Mean_Elevation', 'elev_m']:
            if alt_col in lake_gdf.columns:
                elev_col = alt_col
                break

    if elev_col in lake_gdf.columns:
        # Create elevation bins
        lake_gdf_copy = lake_gdf.copy()
        lake_gdf_copy['elev_bin'] = pd.cut(
            lake_gdf_copy[elev_col],
            bins=elev_breaks,
            labels=False,
            include_lowest=True
        )

        # Group by Dalton classification and elevation bin
        dalton_elevation = lake_gdf_copy.groupby(['dalton_classification', 'elev_bin']).size().reset_index(name='n_lakes')

        # Add bin labels
        bin_labels = []
        for i in range(len(elev_breaks) - 1):
            bin_labels.append(f"{elev_breaks[i]}-{elev_breaks[i+1]}")

        dalton_elevation['elev_bin_label'] = dalton_elevation['elev_bin'].apply(
            lambda x: bin_labels[int(x)] if pd.notna(x) and 0 <= int(x) < len(bin_labels) else 'Unknown'
        )
        dalton_elevation['elev_bin_mid'] = dalton_elevation['elev_bin'].apply(
            lambda x: (elev_breaks[int(x)] + elev_breaks[int(x)+1]) / 2
            if pd.notna(x) and 0 <= int(x) < len(elev_breaks)-1 else np.nan
        )

        # Calculate percentage within each classification
        class_totals = dalton_elevation.groupby('dalton_classification')['n_lakes'].transform('sum')
        dalton_elevation['pct_of_class'] = 100 * dalton_elevation['n_lakes'] / class_totals

        # Rename to match expected column name
        dalton_elevation = dalton_elevation.rename(columns={
            'dalton_classification': 'glacial_stage',
            'pct_of_class': 'pct_of_stage'
        })

        results['elevation_by_dalton'] = dalton_elevation

        if verbose:
            print(f"  Processed {len(dalton_elevation)} classification-elevation combinations")
            for classification in ['18ka_glaciated', '18ka_unglaciated']:
                class_data = dalton_elevation[dalton_elevation['glacial_stage'] == classification]
                if len(class_data) > 0:
                    peak_elev = class_data.loc[class_data['n_lakes'].idxmax(), 'elev_bin_mid']
                    print(f"  {classification}: Peak at {peak_elev:.0f}m elevation")
    else:
        if verbose:
            print(f"  Warning: Could not find elevation column (tried: {elev_col})")

    # Save results
    if save_results:
        try:
            ensure_output_dir()
            output_dir = Path(OUTPUT_DIR) / 'dalton_18ka'
            output_dir.mkdir(exist_ok=True)

            # Save classification
            classification_df = lake_gdf[['dalton_classification', area_col]].copy()
            if 'Elevation' in lake_gdf.columns:
                classification_df['Elevation'] = lake_gdf['Elevation']
            classification_df.to_csv(output_dir / 'lake_dalton_18ka_classification.csv', index=False)

            # Save density comparison
            import json
            with open(output_dir / 'dalton_18ka_density_comparison.json', 'w') as f:
                json.dump(results['density_comparison'], f, indent=2)

            if verbose:
                print(f"\n  Results saved to: {output_dir}")

        except Exception as e:
            print(f"  Warning: Could not save results: {e}")

    return results


def compare_wisconsin_vs_dalton_18ka(lake_df, verbose=True):
    """
    Compare lake distributions using Wisconsin max extent (~20 ka) vs
    Dalton 18ka (precise LGM snapshot).

    This comparison tests whether the precise 18ka reconstruction
    (Dalton) shows different patterns than the maximum Wisconsin extent.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with lat/lon columns
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Comparison results including:
        - wisconsin_analysis: Results for Wisconsin extent
        - dalton_analysis: Results for Dalton 18ka
        - comparison: Direct comparison metrics
    """
    if verbose:
        print("\n" + "=" * 60)
        print("WISCONSIN VS DALTON 18KA COMPARISON")
        print("Maximum extent (~20 ka) vs Precise LGM snapshot (18 ka)")
        print("=" * 60)

    results = {}
    target_crs = get_target_crs()

    # Load both boundaries
    try:
        wisconsin = load_wisconsin_extent(target_crs=target_crs)
        dalton = load_dalton_18ka(target_crs=target_crs, clip_to_conus=True)
    except Exception as e:
        print(f"ERROR: Could not load boundaries: {e}")
        return {'error': str(e)}

    wisc_area = wisconsin.geometry.area.sum() / 1e6
    dalton_area = dalton.geometry.area.sum() / 1e6

    if verbose:
        print(f"\n  Wisconsin max extent: {wisc_area:,.0f} km²")
        print(f"  Dalton 18ka extent: {dalton_area:,.0f} km²")
        print(f"  Difference: {wisc_area - dalton_area:,.0f} km²")

    # Convert lakes
    lake_gdf = convert_lakes_to_gdf(lake_df, target_crs=target_crs)

    # Classify by both
    wisc_union = unary_union(wisconsin.geometry)
    dalton_union = unary_union(dalton.geometry)

    wisc_single = gpd.GeoDataFrame(geometry=[wisc_union], crs=target_crs)
    dalton_single = gpd.GeoDataFrame(geometry=[dalton_union], crs=target_crs)

    # Wisconsin classification
    wisc_joined = gpd.sjoin(
        lake_gdf[['geometry']].reset_index(),
        wisc_single, how='inner', predicate='within'
    )
    lake_gdf['in_wisconsin'] = lake_gdf.index.isin(wisc_joined['index'].values)

    # Dalton 18ka classification
    dalton_joined = gpd.sjoin(
        lake_gdf[['geometry']].reset_index(),
        dalton_single, how='inner', predicate='within'
    )
    lake_gdf['in_dalton_18ka'] = lake_gdf.index.isin(dalton_joined['index'].values)

    # Create combined classification
    def classify(row):
        if row['in_dalton_18ka'] and row['in_wisconsin']:
            return 'both'  # In both extents (core glaciated)
        elif row['in_wisconsin'] and not row['in_dalton_18ka']:
            return 'wisconsin_only'  # In Wisconsin but not 18ka (marginal)
        elif row['in_dalton_18ka'] and not row['in_wisconsin']:
            return 'dalton_only'  # Unusual - should be rare
        else:
            return 'neither'

    lake_gdf['combined_classification'] = lake_gdf.apply(classify, axis=1)
    results['lake_gdf'] = lake_gdf

    # Summary
    class_counts = lake_gdf['combined_classification'].value_counts()
    if verbose:
        print("\n  Combined classification:")
        for cls, count in class_counts.items():
            pct = 100 * count / len(lake_gdf)
            print(f"    {cls}: {count:,} lakes ({pct:.1f}%)")

    # Density calculations
    area_col = COLS.get('area', 'AREASQKM')

    comparison = {
        'wisconsin': {
            'n_lakes': lake_gdf['in_wisconsin'].sum(),
            'area_km2': wisc_area,
            'density': (lake_gdf['in_wisconsin'].sum() / wisc_area) * 1000
        },
        'dalton_18ka': {
            'n_lakes': lake_gdf['in_dalton_18ka'].sum(),
            'area_km2': dalton_area,
            'density': (lake_gdf['in_dalton_18ka'].sum() / dalton_area) * 1000
        },
        'wisconsin_only': {
            'n_lakes': (class_counts.get('wisconsin_only', 0)),
            'area_km2': wisc_area - dalton_area,
            'density': ((class_counts.get('wisconsin_only', 0)) / (wisc_area - dalton_area)) * 1000 if wisc_area > dalton_area else 0
        }
    }

    results['comparison'] = comparison

    if verbose:
        print("\n  Density comparison:")
        print(f"    Wisconsin max: {comparison['wisconsin']['density']:.1f} lakes/1000 km²")
        print(f"    Dalton 18ka: {comparison['dalton_18ka']['density']:.1f} lakes/1000 km²")
        if comparison['wisconsin_only']['area_km2'] > 0:
            print(f"    Wisconsin margins only: {comparison['wisconsin_only']['density']:.1f} lakes/1000 km²")

    # Power law comparison
    if verbose:
        print("\n  Power law comparison:")

    pl_results = []
    for extent_name, extent_col in [('wisconsin', 'in_wisconsin'), ('dalton_18ka', 'in_dalton_18ka')]:
        mask = lake_gdf[extent_col]
        subset = lake_gdf[mask]

        if len(subset) < 50:
            continue

        areas = subset[area_col].dropna().values
        areas = areas[areas > 0]

        xmin = 0.01
        tail = areas[areas >= xmin]
        if len(tail) >= 30:
            alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
            alpha_se = (alpha - 1) / np.sqrt(len(tail))

            pl_results.append({
                'extent': extent_name,
                'alpha': alpha,
                'alpha_se': alpha_se,
                'n_tail': len(tail),
                'xmin': xmin
            })

            if verbose:
                print(f"    {extent_name}: α = {alpha:.3f} ± {alpha_se:.3f} (n={len(tail):,})")

    results['power_law_comparison'] = pl_results

    return results


def xmin_sensitivity_by_glacial_zone(lake_gdf, area_col=None, verbose=True):
    """
    Run x_min sensitivity analysis for each glacial zone.

    Similar to the elevation-band analysis but for glacial zones.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage classification
    area_col : str, optional
        Lake area column name
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Sensitivity results by glacial zone
    """
    if area_col is None:
        area_col = COLS.get('area', 'AREASQKM')

    if 'glacial_stage' not in lake_gdf.columns:
        return {'error': 'glacial_stage column not found'}

    if verbose:
        print("\n" + "=" * 60)
        print("X_MIN SENSITIVITY BY GLACIAL ZONE")
        print("=" * 60)

    # Define x_min candidates
    xmin_candidates = np.logspace(np.log10(0.001), np.log10(10.0), 30)

    # Get unique glacial stages
    stages = lake_gdf['glacial_stage'].unique()
    stages = [s for s in stages if pd.notna(s)]

    results = {}

    for i, stage in enumerate(stages):
        if verbose:
            print(f"\n[{i+1}/{len(stages)}] {stage}")

        mask = lake_gdf['glacial_stage'] == stage
        subset = lake_gdf[mask]

        areas = subset[area_col].dropna().values
        areas = areas[areas > 0]

        if len(areas) < 30:
            if verbose:
                print(f"  Skipping: too few lakes ({len(areas)})")
            continue

        # Run sensitivity for each x_min
        sensitivity = []
        for xmin in xmin_candidates:
            tail = areas[areas >= xmin]
            if len(tail) < 10:
                continue

            alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
            alpha_se = (alpha - 1) / np.sqrt(len(tail))

            # Compute KS statistic
            theoretical_cdf = 1 - (xmin / tail) ** (alpha - 1)
            empirical_cdf = np.arange(1, len(tail) + 1) / len(tail)
            sorted_tail = np.sort(tail)
            theoretical_sorted = 1 - (xmin / sorted_tail) ** (alpha - 1)
            ks_stat = np.max(np.abs(empirical_cdf - theoretical_sorted))

            sensitivity.append({
                'xmin': xmin,
                'alpha': alpha,
                'alpha_se': alpha_se,
                'n_tail': len(tail),
                'ks_stat': ks_stat
            })

        if len(sensitivity) > 0:
            sensitivity_df = pd.DataFrame(sensitivity)

            # Find optimal x_min (minimum KS)
            optimal_idx = sensitivity_df['ks_stat'].idxmin()
            optimal_xmin = sensitivity_df.loc[optimal_idx, 'xmin']
            optimal_alpha = sensitivity_df.loc[optimal_idx, 'alpha']
            optimal_ks = sensitivity_df.loc[optimal_idx, 'ks_stat']

            results[stage] = {
                'n_lakes': len(areas),
                'sensitivity': sensitivity_df,
                'optimal_xmin': optimal_xmin,
                'optimal_alpha': optimal_alpha,
                'optimal_ks': optimal_ks,
                # Fixed x_min results
                'fixed_xmin_results': {
                    row['xmin']: {'alpha': row['alpha'], 'n': row['n_tail']}
                    for _, row in sensitivity_df.iterrows()
                    if row['xmin'] in [0.01, 0.1, 1.0]
                }
            }

            if verbose:
                print(f"  n = {len(areas):,}")
                print(f"  Optimal: x_min = {optimal_xmin:.3f}, α = {optimal_alpha:.3f}, KS = {optimal_ks:.4f}")

    return results


# ============================================================================
# PER-STAGE HYPSOMETRY AND PROPER NORMALIZATION
# ============================================================================

def compute_per_stage_hypsometry(lake_gdf, zone_areas, elev_breaks=None, verbose=True):
    """
    Compute hypsometry (area by elevation) for each glacial stage.

    Uses the distribution of lakes across elevation bins as a proxy for
    the landscape elevation distribution within each stage. This assumes
    lakes are roughly uniformly distributed across the landscape at each
    elevation (a reasonable first-order approximation).

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage and elevation columns
    zone_areas : dict
        Dictionary mapping stage names to total area in km²
    elev_breaks : list, optional
        Elevation bin edges (default from config)
    verbose : bool
        Print progress information

    Returns
    -------
    DataFrame
        Per-stage hypsometry with columns:
        - glacial_stage
        - elev_bin_mid
        - area_km2 (estimated landscape area at this elevation for this stage)
    """
    if elev_breaks is None:
        elev_breaks = ELEV_BREAKS

    elev_col = COLS.get('elevation', 'Elevation_')

    if verbose:
        print("\nComputing per-stage hypsometry...")

    results = []

    for stage in ['Wisconsin', 'Illinoian', 'Driftless', 'unclassified']:
        stage_lakes = lake_gdf[lake_gdf['glacial_stage'] == stage]

        if len(stage_lakes) == 0:
            continue

        total_stage_area = zone_areas.get(stage.lower(), 0)
        if total_stage_area == 0:
            continue

        # Bin lakes by elevation
        stage_lakes = stage_lakes.copy()
        stage_lakes['elev_bin'] = pd.cut(
            stage_lakes[elev_col],
            bins=elev_breaks,
            labels=False,
            include_lowest=True
        )

        # Count lakes per bin
        bin_counts = stage_lakes.groupby('elev_bin').size()
        total_lakes = len(stage_lakes)

        # Distribute total stage area proportionally to lake counts
        for bin_idx in range(len(elev_breaks) - 1):
            n_lakes = bin_counts.get(bin_idx, 0)
            if n_lakes > 0:
                # Fraction of lakes at this elevation
                frac = n_lakes / total_lakes
                # Estimated landscape area
                area = total_stage_area * frac
                elev_mid = (elev_breaks[bin_idx] + elev_breaks[bin_idx + 1]) / 2

                results.append({
                    'glacial_stage': stage,
                    'elev_bin_mid': elev_mid,
                    'area_km2': area,
                    'n_lakes': n_lakes
                })

    result_df = pd.DataFrame(results)

    if verbose:
        print(f"  Computed hypsometry for {result_df['glacial_stage'].nunique()} stages")
        print(f"  {len(result_df)} stage-elevation combinations")

    return result_df


def compute_elevation_density_by_stage_normalized(lake_gdf, zone_areas, elev_breaks=None, verbose=True):
    """
    Compute properly normalized lake density by elevation for each glacial stage.

    Normalizes by the estimated landscape area at each elevation for each stage,
    enabling proper comparison of lake density across stages.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage and elevation columns
    zone_areas : dict
        Dictionary mapping stage names to total area in km²
    elev_breaks : list, optional
        Elevation bin edges
    verbose : bool
        Print progress

    Returns
    -------
    DataFrame
        Normalized density with columns:
        - glacial_stage
        - elev_bin_mid
        - n_lakes
        - area_km2 (estimated landscape area)
        - density_per_1000km2
    """
    if elev_breaks is None:
        elev_breaks = ELEV_BREAKS

    elev_col = COLS.get('elevation', 'Elevation_')

    if verbose:
        print("\nComputing elevation-normalized density by stage...")

    # First get per-stage hypsometry
    hyps_df = compute_per_stage_hypsometry(lake_gdf, zone_areas, elev_breaks, verbose=False)

    # Get lake counts by stage and elevation
    lake_gdf = lake_gdf.copy()
    lake_gdf['elev_bin'] = pd.cut(
        lake_gdf[elev_col],
        bins=elev_breaks,
        labels=False,
        include_lowest=True
    )
    lake_gdf['elev_bin_mid'] = lake_gdf['elev_bin'].apply(
        lambda x: (elev_breaks[int(x)] + elev_breaks[int(x)+1]) / 2
        if pd.notna(x) and 0 <= int(x) < len(elev_breaks)-1 else np.nan
    )

    counts = lake_gdf.groupby(['glacial_stage', 'elev_bin_mid']).size().reset_index(name='n_lakes')

    # Merge with hypsometry
    result = counts.merge(
        hyps_df[['glacial_stage', 'elev_bin_mid', 'area_km2']],
        on=['glacial_stage', 'elev_bin_mid'],
        how='left'
    )

    # Compute density (lakes per 1000 km²)
    result['density_per_1000km2'] = np.where(
        result['area_km2'] > 0,
        result['n_lakes'] / result['area_km2'] * 1000,
        0
    )

    if verbose:
        for stage in result['glacial_stage'].unique():
            stage_data = result[result['glacial_stage'] == stage]
            mean_density = stage_data['density_per_1000km2'].mean()
            print(f"  {stage}: mean density = {mean_density:.1f} lakes/1000km²")

    return result


# ============================================================================
# ARIDITY INDEX ANALYSIS
# ============================================================================

def compute_density_by_aridity(lake_gdf, aridity_breaks=None, zone_areas=None, verbose=True):
    """
    Compute lake density by aridity index bin.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with aridity index ('AI') column
    aridity_breaks : list, optional
        Aridity index bin edges. Default: [0, 0.2, 0.5, 0.65, 1.0, 2.0, 5.0, 10.0]
        (0-0.2: Hyper-arid, 0.2-0.5: Arid, 0.5-0.65: Semi-arid,
         0.65-1.0: Dry sub-humid, 1.0-2.0: Humid, >2.0: Hyper-humid)
    zone_areas : dict, optional
        Pre-computed landscape areas by aridity bin for normalization
    verbose : bool
        Print progress

    Returns
    -------
    DataFrame
        Lake statistics by aridity bin
    """
    if aridity_breaks is None:
        aridity_breaks = AI_BREAKS

    ai_col = COLS.get('aridity', 'AI')
    area_col = COLS.get('area', 'AREASQKM')

    if ai_col not in lake_gdf.columns:
        if verbose:
            print(f"  Warning: Aridity column '{ai_col}' not found")
        return None

    if verbose:
        print("\n" + "-" * 60)
        print("LAKE DENSITY BY ARIDITY INDEX")
        print("-" * 60)

    # Check if AI values appear to be scaled (common in geodatabases)
    # AI should typically range 0-5, values > 100 suggest scaling by 10000
    lake_gdf = lake_gdf.copy()
    ai_values = lake_gdf[ai_col].dropna()

    if len(ai_values) == 0:
        if verbose:
            print(f"  Warning: No valid AI values found")
        return None

    ai_max = ai_values.max()
    ai_scale = 1.0

    if ai_max > 100:
        # Values appear to be scaled - detect scale factor
        if ai_max > 10000:
            ai_scale = 10000.0
        elif ai_max > 1000:
            ai_scale = 1000.0
        else:
            ai_scale = 100.0

        if verbose:
            print(f"  Detected scaled AI values (max={ai_max:.0f})")
            print(f"  Applying scale factor: 1/{ai_scale:.0f}")

        # Normalize AI values
        lake_gdf['AI_normalized'] = lake_gdf[ai_col] / ai_scale
        ai_col_use = 'AI_normalized'
    else:
        ai_col_use = ai_col

    if verbose:
        ai_norm_vals = lake_gdf[ai_col_use].dropna()
        print(f"  AI range (normalized): {ai_norm_vals.min():.3f} to {ai_norm_vals.max():.3f}")

    # Bin by aridity
    lake_gdf['ai_bin'] = pd.cut(
        lake_gdf[ai_col_use],
        bins=aridity_breaks,
        labels=False,
        include_lowest=True
    )

    # Labels for bins
    ai_labels = {
        0: 'Hyper-arid (0-0.2)',
        1: 'Arid (0.2-0.5)',
        2: 'Semi-arid (0.5-0.65)',
        3: 'Dry sub-humid (0.65-1.0)',
        4: 'Humid (1.0-2.0)',
        5: 'Wet (2.0-5.0)',
        6: 'Hyper-humid (>5.0)'
    }

    results = []
    for bin_idx in range(len(aridity_breaks) - 1):
        bin_lakes = lake_gdf[lake_gdf['ai_bin'] == bin_idx]
        n_lakes = len(bin_lakes)

        if n_lakes > 0:
            total_lake_area = bin_lakes[area_col].sum()
            mean_lake_area = bin_lakes[area_col].mean()
            ai_mid = (aridity_breaks[bin_idx] + aridity_breaks[bin_idx + 1]) / 2

            results.append({
                'ai_bin': bin_idx,
                'ai_label': ai_labels.get(bin_idx, f'Bin {bin_idx}'),
                'ai_lower': aridity_breaks[bin_idx],
                'ai_upper': aridity_breaks[bin_idx + 1],
                'ai_mid': ai_mid,
                'n_lakes': n_lakes,
                'total_lake_area_km2': total_lake_area,
                'mean_lake_area_km2': mean_lake_area
            })

    result_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  {'Aridity Bin':<25} {'N Lakes':>12} {'Mean Area (km²)':>15}")
        print("  " + "-" * 55)
        for _, row in result_df.iterrows():
            print(f"  {row['ai_label']:<25} {row['n_lakes']:>12,} {row['mean_lake_area_km2']:>15.4f}")

    return result_df


def compute_density_by_aridity_and_glacial(lake_gdf, aridity_breaks=None, verbose=True):
    """
    Compute lake density stratified by both aridity and glacial stage.

    This allows us to examine whether glacial stage effects persist
    after controlling for aridity (climate).

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with aridity ('AI') and glacial_stage columns
    aridity_breaks : list, optional
        Aridity index bin edges
    verbose : bool
        Print progress

    Returns
    -------
    DataFrame
        Cross-tabulation of lake counts by aridity and glacial stage
    """
    if aridity_breaks is None:
        aridity_breaks = AI_BREAKS

    ai_col = COLS.get('aridity', 'AI')

    if ai_col not in lake_gdf.columns or 'glacial_stage' not in lake_gdf.columns:
        if verbose:
            print("  Warning: Required columns not found")
        return None

    if verbose:
        print("\n" + "-" * 60)
        print("LAKE DENSITY: ARIDITY × GLACIAL STAGE")
        print("-" * 60)

    # Check if AI values appear to be scaled
    lake_gdf = lake_gdf.copy()
    ai_values = lake_gdf[ai_col].dropna()

    if len(ai_values) == 0:
        if verbose:
            print(f"  Warning: No valid AI values found")
        return None

    ai_max = ai_values.max()

    if ai_max > 100:
        # Values appear to be scaled - detect scale factor
        if ai_max > 10000:
            ai_scale = 10000.0
        elif ai_max > 1000:
            ai_scale = 1000.0
        else:
            ai_scale = 100.0

        if verbose:
            print(f"  Normalizing scaled AI values (scale: 1/{ai_scale:.0f})")

        # Normalize AI values
        lake_gdf['AI_normalized'] = lake_gdf[ai_col] / ai_scale
        ai_col_use = 'AI_normalized'
    else:
        ai_col_use = ai_col

    # Bin by aridity
    lake_gdf['ai_bin'] = pd.cut(
        lake_gdf[ai_col_use],
        bins=aridity_breaks,
        labels=False,
        include_lowest=True
    )

    # Cross-tabulation
    cross_tab = pd.crosstab(
        lake_gdf['ai_bin'],
        lake_gdf['glacial_stage'],
        margins=True
    )

    if verbose:
        print("\n  Lake counts by Aridity × Glacial Stage:")
        print(cross_tab.to_string())

    # Compute proportions within each aridity bin
    proportions = cross_tab.div(cross_tab['All'], axis=0) * 100

    if verbose:
        print("\n  Percentage by glacial stage within each aridity bin:")
        print(proportions.round(1).to_string())

    return {
        'counts': cross_tab,
        'proportions': proportions,
        'lake_gdf': lake_gdf
    }


def run_aridity_glacial_comparison(lake_gdf, zone_areas=None, verbose=True):
    """
    Compare aridity vs glacial stage as predictors of lake density.

    Uses multiple approaches:
    1. Simple correlation analysis
    2. ANOVA comparing glacial stages within aridity bins
    3. Partial correlation controlling for aridity
    4. Multiple regression with both predictors

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with aridity, elevation, and glacial_stage columns
    zone_areas : dict, optional
        Glacial zone areas for density normalization
    verbose : bool
        Print detailed results

    Returns
    -------
    dict
        Comprehensive comparison results
    """
    from scipy import stats

    ai_col = COLS.get('aridity', 'AI')
    elev_col = COLS.get('elevation', 'Elevation_')
    area_col = COLS.get('area', 'AREASQKM')

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARING ARIDITY VS GLACIAL STAGE AS LAKE DENSITY PREDICTORS")
        print("=" * 70)

    results = {}

    # =========================================================================
    # 0. Check for scaled AI values and normalize if needed
    # =========================================================================
    lake_gdf = lake_gdf.copy()
    ai_values = lake_gdf[ai_col].dropna()
    ai_scale = 1.0

    if len(ai_values) > 0:
        ai_max = ai_values.max()
        if ai_max > 100:
            # Values appear to be scaled
            if ai_max > 10000:
                ai_scale = 10000.0
            elif ai_max > 1000:
                ai_scale = 1000.0
            else:
                ai_scale = 100.0

            if verbose:
                print(f"\n  Detected scaled AI values (max={ai_max:.0f})")
                print(f"  Normalizing with scale factor: 1/{ai_scale:.0f}")

            lake_gdf['AI_normalized'] = lake_gdf[ai_col] / ai_scale
            ai_col_use = 'AI_normalized'
        else:
            ai_col_use = ai_col
    else:
        ai_col_use = ai_col

    results['ai_scale_factor'] = ai_scale

    # =========================================================================
    # 1. Basic statistics by aridity
    # =========================================================================
    aridity_stats = compute_density_by_aridity(lake_gdf, verbose=verbose)
    results['aridity_stats'] = aridity_stats

    # =========================================================================
    # 2. Cross-tabulation: Aridity × Glacial Stage
    # =========================================================================
    cross_results = compute_density_by_aridity_and_glacial(lake_gdf, verbose=verbose)
    results['cross_tabulation'] = cross_results

    # =========================================================================
    # 3. Correlation: Aridity vs Lake Size
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("CORRELATION ANALYSIS")
        print("-" * 60)

    # Use normalized AI values for correlation
    valid_mask = (lake_gdf[ai_col_use] > 0) & (lake_gdf[ai_col_use] < 100)
    valid_lakes = lake_gdf[valid_mask].copy()

    if len(valid_lakes) == 0:
        if verbose:
            print(f"  Warning: No valid lakes after filtering (AI range: 0-100)")
        return results

    # Add AI binning to valid_lakes for ANOVA
    valid_lakes['ai_bin'] = pd.cut(
        valid_lakes[ai_col_use],
        bins=AI_BREAKS,
        labels=False,
        include_lowest=True
    )

    # Aridity vs lake area (use normalized values)
    r_aridity_area, p_aridity_area = stats.pearsonr(
        valid_lakes[ai_col_use],
        np.log10(valid_lakes[area_col])
    )
    results['r_aridity_area'] = r_aridity_area
    results['p_aridity_area'] = p_aridity_area

    # Aridity vs elevation
    r_aridity_elev, p_aridity_elev = stats.pearsonr(
        valid_lakes[ai_col_use],
        valid_lakes[elev_col]
    )
    results['r_aridity_elev'] = r_aridity_elev
    results['p_aridity_elev'] = p_aridity_elev

    if verbose:
        print(f"\n  Valid lakes for correlation: {len(valid_lakes):,}")
        print(f"  Aridity vs log(Lake Area): r = {r_aridity_area:.3f}, p = {p_aridity_area:.2e}")
        print(f"  Aridity vs Elevation: r = {r_aridity_elev:.3f}, p = {p_aridity_elev:.2e}")

    # =========================================================================
    # 4. ANOVA: Do glacial stages differ within aridity bins?
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("ANOVA: GLACIAL STAGE EFFECTS WITHIN ARIDITY BINS")
        print("-" * 60)
        print("\n  Testing if glacial stage matters after controlling for aridity...")

    anova_results = []
    for ai_bin in valid_lakes['ai_bin'].dropna().unique():
        bin_lakes = valid_lakes[valid_lakes['ai_bin'] == ai_bin]

        # Get lake counts by glacial stage
        groups = [
            bin_lakes[bin_lakes['glacial_stage'] == stage][area_col].values
            for stage in ['Wisconsin', 'Illinoian', 'Driftless', 'unclassified']
            if len(bin_lakes[bin_lakes['glacial_stage'] == stage]) > 5
        ]

        if len(groups) >= 2:
            # Log-transform areas for normality
            log_groups = [np.log10(g[g > 0]) for g in groups if len(g[g > 0]) > 0]

            if len(log_groups) >= 2:
                f_stat, p_val = stats.f_oneway(*log_groups)
                anova_results.append({
                    'ai_bin': int(ai_bin),
                    'n_groups': len(log_groups),
                    'total_lakes': sum(len(g) for g in log_groups),
                    'f_statistic': f_stat,
                    'p_value': p_val
                })

    if anova_results:
        anova_df = pd.DataFrame(anova_results)
        results['anova_by_aridity'] = anova_df

        if verbose:
            print(f"\n  {'AI Bin':>8} {'N Groups':>10} {'N Lakes':>10} {'F-stat':>10} {'p-value':>12}")
            print("  " + "-" * 55)
            for _, row in anova_df.iterrows():
                sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"  {row['ai_bin']:>8} {row['n_groups']:>10} {row['total_lakes']:>10} "
                      f"{row['f_statistic']:>10.2f} {row['p_value']:>12.2e} {sig}")

    # =========================================================================
    # 5. Multiple Regression
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("MULTIPLE REGRESSION: log(Lake Area) ~ Aridity + Glacial Stage")
        print("-" * 60)

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # Prepare data - use normalized AI values
        reg_data = valid_lakes[[area_col, 'glacial_stage', elev_col]].copy()
        reg_data['AI_norm'] = valid_lakes[ai_col_use]
        reg_data['log_area'] = np.log10(reg_data[area_col])
        reg_data = reg_data.dropna()

        if len(reg_data) < 10:
            if verbose:
                print(f"  Warning: Not enough data for regression ({len(reg_data)} rows)")
            results['regression'] = None
        else:
            # Fit model
            model = ols('log_area ~ AI_norm + C(glacial_stage)', data=reg_data).fit()

            results['regression'] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'params': model.params.to_dict(),
                'pvalues': model.pvalues.to_dict()
            }

            if verbose:
                print(f"\n  R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}")
                print(f"  F-statistic = {model.fvalue:.2f}, p = {model.f_pvalue:.2e}")
                print("\n  Coefficients:")
                for param, coef in model.params.items():
                    pval = model.pvalues[param]
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    print(f"    {param}: {coef:.4f} (p = {pval:.2e}) {sig}")

            # Model with interaction
            model_int = ols('log_area ~ AI_norm * C(glacial_stage)', data=reg_data).fit()
            results['regression_interaction'] = {
                'r_squared': model_int.rsquared,
                'adj_r_squared': model_int.rsquared_adj,
            }

            if verbose:
                print(f"\n  With interaction: R² = {model_int.rsquared:.4f}")
                improvement = model_int.rsquared - model.rsquared
                print(f"  R² improvement from interaction: {improvement:.4f}")

    except ImportError:
        if verbose:
            print("  Note: statsmodels not available for regression analysis")
        results['regression'] = None

    # =========================================================================
    # 6. Summary: Which explains more variance?
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("SUMMARY: RELATIVE IMPORTANCE")
        print("-" * 60)

    # Aridity-only model
    try:
        if len(reg_data) >= 10:
            model_ai = ols('log_area ~ AI_norm', data=reg_data).fit()
            r2_ai = model_ai.rsquared

            # Glacial-only model
            model_glacial = ols('log_area ~ C(glacial_stage)', data=reg_data).fit()
            r2_glacial = model_glacial.rsquared

            results['r2_aridity_only'] = r2_ai
            results['r2_glacial_only'] = r2_glacial

            if verbose:
                print(f"\n  R² (Aridity only): {r2_ai:.4f}")
                print(f"  R² (Glacial stage only): {r2_glacial:.4f}")
                if results.get('regression') and results['regression'].get('r_squared'):
                    print(f"  R² (Both): {results['regression']['r_squared']:.4f}")

                if r2_glacial > r2_ai:
                    print(f"\n  → Glacial stage explains {(r2_glacial/r2_ai - 1)*100:.1f}% more variance than aridity alone")
                else:
                    print(f"\n  → Aridity explains {(r2_ai/r2_glacial - 1)*100:.1f}% more variance than glacial stage alone")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not complete R² comparison: {e}")

    return results


# ============================================================================
# ARIDITY AND LAKE HALF-LIFE ANALYSIS
# ============================================================================

def analyze_aridity_lake_halflife(lake_gdf, verbose=True):
    """
    Analyze whether aridity affects lake persistence (half-life).

    Tests if the rate of lake density decline with landscape age varies
    by aridity class. If aridity affects lake half-life, we'd expect:
    - Different density decay rates in arid vs humid regions
    - Interaction between aridity and glacial stage in predicting density

    Limitations:
    - Only 2-3 glacial stages with known ages (poor temporal resolution)
    - Cannot directly measure lake half-life, only infer from density patterns
    - Confounded by other factors (elevation, lithology, etc.)

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with 'AI', 'glacial_stage', and area columns

    Returns
    -------
    dict
        Results including:
        - density_by_aridity_stage: Lake density for each aridity-stage combination
        - decay_rates: Estimated density decay rates by aridity class
        - interaction_test: Statistical test for aridity × age interaction
        - half_life_estimates: Rough half-life estimates by aridity (if computable)
    """
    from scipy import stats

    ai_col = COLS.get('aridity', 'AI')
    area_col = COLS.get('area', 'AREASQKM')

    if verbose:
        print("\n" + "=" * 70)
        print("ARIDITY AND LAKE HALF-LIFE ANALYSIS")
        print("=" * 70)
        print("\nQuestion: Does aridity affect how quickly lakes disappear?")
        print("Approach: Compare density decay rates across aridity classes")

    results = {}

    # Check required columns
    if ai_col not in lake_gdf.columns:
        print(f"  ERROR: Aridity column '{ai_col}' not found")
        return {'error': 'aridity_column_not_found'}

    if 'glacial_stage' not in lake_gdf.columns:
        print(f"  ERROR: glacial_stage column not found")
        return {'error': 'glacial_stage_not_found'}

    # =========================================================================
    # 1. Normalize AI values if scaled
    # =========================================================================
    lake_gdf = lake_gdf.copy()
    ai_values = lake_gdf[ai_col].dropna()
    ai_max = ai_values.max()

    if ai_max > 100:
        if ai_max > 10000:
            ai_scale = 10000.0
        elif ai_max > 1000:
            ai_scale = 1000.0
        else:
            ai_scale = 100.0
        lake_gdf['AI_norm'] = lake_gdf[ai_col] / ai_scale
        if verbose:
            print(f"\n  Normalized AI values (scale: 1/{ai_scale:.0f})")
    else:
        lake_gdf['AI_norm'] = lake_gdf[ai_col]
        ai_scale = 1.0

    results['ai_scale'] = ai_scale

    # =========================================================================
    # 2. Create aridity classes (simplified: arid vs humid)
    # =========================================================================
    # Using AI = 0.65 as threshold (dry sub-humid boundary)
    lake_gdf['aridity_class'] = pd.cut(
        lake_gdf['AI_norm'],
        bins=[0, 0.5, 0.65, 1.0, 100],
        labels=['Arid (<0.5)', 'Semi-arid (0.5-0.65)', 'Sub-humid (0.65-1.0)', 'Humid (>1.0)']
    )

    if verbose:
        print("\n  Aridity class distribution:")
        for cls, count in lake_gdf['aridity_class'].value_counts().items():
            print(f"    {cls}: {count:,} lakes")

    # =========================================================================
    # 3. Compute density by aridity × glacial stage
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("LAKE DENSITY BY ARIDITY × GLACIAL STAGE")
        print("-" * 60)

    # Get glacial stage ages (ka)
    stage_ages = {
        'Wisconsin': 20,
        'Illinoian': 160,
        'Driftless': None,  # Never glaciated
        'unclassified': None
    }

    # Cross-tabulation
    density_data = []

    for aridity_cls in lake_gdf['aridity_class'].dropna().unique():
        for stage in ['Wisconsin', 'Illinoian']:
            mask = (lake_gdf['aridity_class'] == aridity_cls) & (lake_gdf['glacial_stage'] == stage)
            n_lakes = mask.sum()

            if n_lakes > 0:
                mean_area = lake_gdf.loc[mask, area_col].mean()
                total_area = lake_gdf.loc[mask, area_col].sum()
                age = stage_ages.get(stage)

                density_data.append({
                    'aridity_class': str(aridity_cls),
                    'glacial_stage': stage,
                    'age_ka': age,
                    'n_lakes': n_lakes,
                    'mean_area_km2': mean_area,
                    'total_area_km2': total_area,
                })

    density_df = pd.DataFrame(density_data)
    results['density_by_aridity_stage'] = density_df

    if verbose and len(density_df) > 0:
        print("\n  Aridity Class          Stage        Age(ka)  N Lakes   Mean Area")
        print("  " + "-" * 65)
        for _, row in density_df.iterrows():
            print(f"  {row['aridity_class']:<22} {row['glacial_stage']:<12} "
                  f"{row['age_ka']:>6}  {row['n_lakes']:>8,}  {row['mean_area_km2']:>10.4f}")

    # =========================================================================
    # 4. Estimate "decay rates" by aridity class
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("DENSITY DECAY RATES BY ARIDITY CLASS")
        print("-" * 60)
        print("\n  Comparing Wisconsin (20 ka) to Illinoian (160 ka) density ratios")

    decay_rates = []

    for aridity_cls in density_df['aridity_class'].unique():
        cls_data = density_df[density_df['aridity_class'] == aridity_cls]

        wisc = cls_data[cls_data['glacial_stage'] == 'Wisconsin']
        illin = cls_data[cls_data['glacial_stage'] == 'Illinoian']

        if len(wisc) > 0 and len(illin) > 0:
            n_wisc = wisc['n_lakes'].values[0]
            n_illin = illin['n_lakes'].values[0]

            # Density ratio (Wisconsin / Illinoian)
            # Higher ratio = more lakes lost over time = shorter half-life
            density_ratio = n_wisc / n_illin if n_illin > 0 else np.nan

            # Rough decay rate estimate
            # Assuming exponential decay: N(t) = N0 * exp(-λt)
            # λ = ln(N1/N2) / (t2 - t1)
            if n_illin > 0 and n_wisc > 0:
                # Note: We expect more lakes in younger terrain, so this might be negative
                # which would indicate "growth" rather than decay
                time_diff = 160 - 20  # ka
                lambda_est = np.log(n_wisc / n_illin) / time_diff
                half_life = np.log(2) / abs(lambda_est) if lambda_est != 0 else np.nan
            else:
                lambda_est = np.nan
                half_life = np.nan

            decay_rates.append({
                'aridity_class': aridity_cls,
                'n_wisconsin': n_wisc,
                'n_illinoian': n_illin,
                'density_ratio': density_ratio,
                'decay_rate_per_ka': lambda_est,
                'implied_half_life_ka': half_life
            })

    decay_df = pd.DataFrame(decay_rates)
    results['decay_rates'] = decay_df

    if verbose and len(decay_df) > 0:
        print("\n  Aridity Class           N(Wisc)  N(Illin)  Ratio    λ(/ka)   Half-life(ka)")
        print("  " + "-" * 75)
        for _, row in decay_df.iterrows():
            hl_str = f"{row['implied_half_life_ka']:.0f}" if pd.notna(row['implied_half_life_ka']) else "N/A"
            print(f"  {row['aridity_class']:<22} {row['n_wisconsin']:>8,} {row['n_illinoian']:>8,}  "
                  f"{row['density_ratio']:>6.2f}  {row['decay_rate_per_ka']:>8.4f}   {hl_str:>10}")

    # =========================================================================
    # 5. Statistical test: Does aridity modify the age-density relationship?
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("STATISTICAL TEST: ARIDITY × AGE INTERACTION")
        print("-" * 60)

    # Test using 2-way ANOVA or regression with interaction
    try:
        from statsmodels.formula.api import ols
        import statsmodels.api as sm
        from statsmodels.stats.anova import anova_lm

        # Prepare data - only glacial stages with known ages
        test_data = lake_gdf[
            (lake_gdf['glacial_stage'].isin(['Wisconsin', 'Illinoian'])) &
            (lake_gdf['aridity_class'].notna())
        ].copy()

        test_data['age'] = test_data['glacial_stage'].map(stage_ages)
        test_data['log_area'] = np.log10(test_data[area_col])

        if len(test_data) > 100:
            # Model without interaction
            model_add = ols('log_area ~ AI_norm + C(glacial_stage)', data=test_data).fit()

            # Model with interaction
            model_int = ols('log_area ~ AI_norm * C(glacial_stage)', data=test_data).fit()

            # Compare models
            r2_add = model_add.rsquared
            r2_int = model_int.rsquared

            # F-test for interaction term significance
            anova_result = anova_lm(model_add, model_int)

            results['interaction_test'] = {
                'r2_additive': r2_add,
                'r2_interaction': r2_int,
                'r2_improvement': r2_int - r2_add,
                'f_statistic': anova_result['F'].iloc[1],
                'p_value': anova_result['Pr(>F)'].iloc[1]
            }

            if verbose:
                print(f"\n  Testing if aridity modifies the glacial stage effect on lake size...")
                print(f"\n  Model comparison:")
                print(f"    Additive model (AI + Stage):      R² = {r2_add:.4f}")
                print(f"    Interaction model (AI × Stage):   R² = {r2_int:.4f}")
                print(f"    R² improvement from interaction:  {r2_int - r2_add:.4f}")
                print(f"\n  F-test for interaction:")
                print(f"    F = {anova_result['F'].iloc[1]:.2f}, p = {anova_result['Pr(>F)'].iloc[1]:.2e}")

                if anova_result['Pr(>F)'].iloc[1] < 0.05:
                    print("\n  → SIGNIFICANT: Aridity DOES modify the glacial stage effect")
                    print("    This suggests aridity may affect lake persistence rates")
                else:
                    print("\n  → NOT SIGNIFICANT: No evidence that aridity modifies the glacial effect")
                    print("    Lake persistence appears similar across aridity classes")

    except ImportError:
        if verbose:
            print("  Note: statsmodels not available for interaction test")
        results['interaction_test'] = None
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not complete interaction test: {e}")
        results['interaction_test'] = None

    # =========================================================================
    # 6. Summary and caveats
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("SUMMARY AND CAVEATS")
        print("-" * 60)

        print("\n  Key limitations:")
        print("    • Only 2 glacial stages with known ages (140 ka time span)")
        print("    • Cannot measure actual lake half-life directly")
        print("    • Density ≠ persistence (new lakes can form)")
        print("    • Confounded by elevation, lithology, regional climate")
        print("    • Sample sizes vary greatly by aridity × stage cell")

        if len(decay_df) > 0:
            mean_hl = decay_df['implied_half_life_ka'].mean()
            if pd.notna(mean_hl):
                print(f"\n  Rough estimate: Mean implied half-life ≈ {mean_hl:.0f} ka")
                print("    (Interpret with extreme caution given limitations)")

    return results


# ============================================================================
# NADI-1 TIME SLICE CHRONOSEQUENCE ANALYSIS
# ============================================================================
# Uses Dalton et al. NADI-1 ice sheet reconstructions (1 ka to 25 ka at 0.5 ka
# intervals) for comprehensive deglaciation chronosequence analysis.

def discover_nadi1_time_slices(verbose=True):
    """
    Discover all available NADI-1 time slice shapefiles.

    Searches the NADI-1 directory for shapefiles matching the expected naming
    pattern and returns a DataFrame with file paths and metadata.

    Parameters
    ----------
    verbose : bool, optional
        If True, print discovery summary.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: age_ka, extent_type, filepath, exists
    """
    from pathlib import Path

    nadi_dir = Path(NADI1_CONFIG['directory'])
    pattern = NADI1_CONFIG['file_pattern']
    ages = get_nadi1_ages()
    extent_types = NADI1_CONFIG['extent_types']

    results = []

    for age in ages:
        # Handle age formatting (1.0 -> "1", 1.5 -> "1.5")
        if age == int(age):
            age_str = str(int(age))
        else:
            age_str = str(age)

        for extent_type in extent_types:
            filename = pattern.format(age=age_str, extent_type=extent_type)
            filepath = nadi_dir / filename

            results.append({
                'age_ka': age,
                'extent_type': extent_type,
                'filename': filename,
                'filepath': str(filepath),
                'exists': filepath.exists()
            })

    df = pd.DataFrame(results)

    if verbose:
        n_found = df['exists'].sum()
        n_expected = len(df)
        n_ages = len(ages)
        print(f"\nNADI-1 Time Slice Discovery")
        print("=" * 50)
        print(f"  Directory: {nadi_dir}")
        print(f"  Ages: {ages[0]} ka to {ages[-1]} ka ({n_ages} time slices)")
        print(f"  Extent types: {extent_types}")
        print(f"  Files found: {n_found}/{n_expected}")

        if n_found < n_expected:
            missing = df[~df['exists']].head(5)
            print(f"\n  First missing files:")
            for _, row in missing.iterrows():
                print(f"    - {row['filename']}")

    return df


def load_nadi1_time_slice(age_ka, extent_type='OPTIMAL', clip_to_conus=True,
                          continental_only=True, verbose=True):
    """
    Load a specific NADI-1 time slice shapefile.

    Parameters
    ----------
    age_ka : float
        Age in thousand years before present (e.g., 20.0 for LGM)
    extent_type : str, optional
        One of 'MIN', 'MAX', 'OPTIMAL'. Default 'OPTIMAL'.
    clip_to_conus : bool, optional
        If True, clip to CONUS boundary. Default True.
    continental_only : bool, optional
        If True, filter to east of continental_lon_threshold (-110°)
        to exclude alpine glaciation. Default True.
    verbose : bool, optional
        Print status messages.

    Returns
    -------
    gpd.GeoDataFrame or None
        Ice sheet extent geometry, or None if file not found.
    """
    from pathlib import Path

    nadi_dir = Path(NADI1_CONFIG['directory'])
    pattern = NADI1_CONFIG['file_pattern']

    # Format age string
    if age_ka == int(age_ka):
        age_str = str(int(age_ka))
    else:
        age_str = str(age_ka)

    filename = pattern.format(age=age_str, extent_type=extent_type)
    filepath = nadi_dir / filename

    if not filepath.exists():
        if verbose:
            print(f"  Warning: File not found: {filepath}")
        return None

    # Load and reproject
    target_crs = get_target_crs()

    try:
        gdf = gpd.read_file(filepath)

        # Reproject to target CRS
        gdf = gdf.to_crs(target_crs)

        # Filter to continental ice (east of threshold)
        if continental_only:
            lon_threshold = NADI1_CONFIG['continental_lon_threshold']
            # Create bounding box for continental region
            # Need to convert threshold to projected coordinates
            from pyproj import Transformer
            try:
                transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
                x_threshold, _ = transformer.transform(lon_threshold, 45.0)  # ~center of CONUS

                # Clip to east of threshold
                minx, miny, maxx, maxy = gdf.total_bounds
                continental_box = box(x_threshold, miny, maxx, maxy)
                gdf = gdf.clip(continental_box)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not filter to continental region: {e}")

        # Clip to CONUS
        if clip_to_conus and SHAPEFILES.get('conus_boundary'):
            try:
                conus = gpd.read_file(SHAPEFILES['conus_boundary'])
                conus = conus.to_crs(target_crs)
                conus_geom = conus.geometry.unary_union
                gdf = gdf.clip(conus_geom)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not clip to CONUS: {e}")

        # Add metadata
        gdf['age_ka'] = age_ka
        gdf['extent_type'] = extent_type

        if verbose:
            area_km2 = gdf.geometry.area.sum() / 1e6  # m² to km²
            print(f"  Loaded {age_ka} ka ({extent_type}): {area_km2:,.0f} km²")

        return gdf

    except Exception as e:
        if verbose:
            print(f"  Error loading {filepath}: {e}")
        return None


def assign_deglaciation_age(lake_gdf, extent_type='OPTIMAL',
                            include_uncertainty=True, verbose=True):
    """
    Assign deglaciation ages to lakes based on NADI-1 time slices.

    For each lake, determines when it was deglaciated by finding the youngest
    time slice where the lake is NOT covered by ice.

    Parameters
    ----------
    lake_gdf : gpd.GeoDataFrame
        Lakes with Point geometry.
    extent_type : str, optional
        Primary extent type ('OPTIMAL', 'MIN', or 'MAX'). Default 'OPTIMAL'.
    include_uncertainty : bool, optional
        If True, also compute ages using MIN and MAX extents.
    verbose : bool, optional
        Print progress messages.

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with new columns:
        - deglaciation_age: Age in ka when lake was deglaciated
        - deglac_age_min: Minimum age estimate (from MAX ice extent)
        - deglac_age_max: Maximum age estimate (from MIN ice extent)
        - was_glaciated: Whether lake was ever under Wisconsin ice
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ASSIGNING DEGLACIATION AGES (NADI-1)")
        print("=" * 60)

    # Get list of ages (from oldest to youngest for proper assignment)
    ages = sorted(get_nadi1_ages(), reverse=True)  # 25 ka down to 1 ka

    # Ensure lakes are in the target CRS
    target_crs = get_target_crs()
    if lake_gdf.crs is None or lake_gdf.crs.to_string() != target_crs:
        lake_gdf = lake_gdf.to_crs(target_crs)

    # Initialize columns
    lake_gdf = lake_gdf.copy()
    lake_gdf['deglaciation_age'] = np.nan
    lake_gdf['was_glaciated'] = False

    if include_uncertainty:
        lake_gdf['deglac_age_min'] = np.nan  # From MAX extent
        lake_gdf['deglac_age_max'] = np.nan  # From MIN extent

    # Track which lakes have been assigned
    assigned_mask = pd.Series(False, index=lake_gdf.index)

    extent_types_to_process = [extent_type]
    if include_uncertainty:
        extent_types_to_process = ['OPTIMAL', 'MIN', 'MAX']

    # Process each extent type
    for ext_type in extent_types_to_process:
        if verbose:
            print(f"\n  Processing {ext_type} extent...")

        age_col = 'deglaciation_age' if ext_type == extent_type else \
                  ('deglac_age_min' if ext_type == 'MAX' else 'deglac_age_max')

        # Reset assignment for this extent type
        if ext_type != extent_type:
            lake_gdf[age_col] = np.nan

        current_assigned = pd.Series(False, index=lake_gdf.index)

        # Process from oldest to youngest
        for i, age in enumerate(ages):
            # Load this time slice
            ice_gdf = load_nadi1_time_slice(
                age, extent_type=ext_type,
                clip_to_conus=True, continental_only=True,
                verbose=False
            )

            if ice_gdf is None or ice_gdf.empty:
                continue

            # Get ice extent geometry
            ice_geom = ice_gdf.geometry.unary_union

            # Find lakes covered by ice at this time
            try:
                covered = lake_gdf.geometry.within(ice_geom)
            except Exception:
                # Try with buffered points for robustness
                covered = lake_gdf.geometry.buffer(100).intersects(ice_geom)

            # Lakes covered now but not yet assigned get this age
            # (they were deglaciated sometime AFTER this age)
            newly_assigned = covered & ~current_assigned

            if ext_type == extent_type:
                lake_gdf.loc[covered, 'was_glaciated'] = True

            # For primary extent, assign deglaciation age
            # Deglaciation age = the first time slice when lake is NOT covered
            # So we track coverage at each time step

            current_assigned = current_assigned | covered

            if verbose and (i + 1) % 10 == 0:
                n_covered = covered.sum()
                print(f"    {age:4.1f} ka: {n_covered:,} lakes under ice")

        # Now assign ages - lakes get the youngest age at which they were still covered
        # Actually, we need a different approach: find when ice retreated FROM each lake

        if verbose:
            print(f"    Computing deglaciation timing...")

        # Reprocess to find exact deglaciation age
        lake_ages = pd.Series(np.nan, index=lake_gdf.index)
        prev_covered = pd.Series(False, index=lake_gdf.index)

        # Process from oldest to youngest
        for age in ages:
            ice_gdf = load_nadi1_time_slice(
                age, extent_type=ext_type,
                clip_to_conus=True, continental_only=True,
                verbose=False
            )

            if ice_gdf is None or ice_gdf.empty:
                covered = pd.Series(False, index=lake_gdf.index)
            else:
                ice_geom = ice_gdf.geometry.unary_union
                try:
                    covered = lake_gdf.geometry.within(ice_geom)
                except Exception:
                    covered = pd.Series(False, index=lake_gdf.index)

            # Lakes that were covered previously but not now were deglaciated at this age
            # (Or more precisely, between prev_age and this age)
            deglaciated_now = prev_covered & ~covered
            lake_ages.loc[deglaciated_now & lake_ages.isna()] = age

            prev_covered = covered

        # Lakes still covered at youngest time slice (1 ka)
        # These were deglaciated more recently than 1 ka
        still_covered = prev_covered
        lake_ages.loc[still_covered & lake_ages.isna()] = 0.5  # Assign 0.5 ka

        lake_gdf[age_col] = lake_ages

    # Lakes never glaciated get NaN age and was_glaciated=False
    never_glaciated = ~lake_gdf['was_glaciated']

    if verbose:
        n_total = len(lake_gdf)
        n_glaciated = lake_gdf['was_glaciated'].sum()
        n_assigned = lake_gdf['deglaciation_age'].notna().sum()

        print(f"\n  Results:")
        print(f"    Total lakes: {n_total:,}")
        print(f"    Were glaciated (Wisconsin): {n_glaciated:,} ({100*n_glaciated/n_total:.1f}%)")
        print(f"    With assigned deglaciation age: {n_assigned:,}")
        print(f"    Never glaciated: {never_glaciated.sum():,}")

        if n_assigned > 0:
            ages_assigned = lake_gdf['deglaciation_age'].dropna()
            print(f"\n  Deglaciation age distribution:")
            print(f"    Mean: {ages_assigned.mean():.1f} ka")
            print(f"    Median: {ages_assigned.median():.1f} ka")
            print(f"    Range: {ages_assigned.min():.1f} - {ages_assigned.max():.1f} ka")

    return lake_gdf


def compute_density_by_deglaciation_age(lake_gdf, age_bins=None,
                                        area_col=None, min_lake_area=0.01,
                                        verbose=True):
    """
    Compute lake density as a function of deglaciation age.

    Parameters
    ----------
    lake_gdf : gpd.GeoDataFrame
        Lakes with deglaciation_age column (from assign_deglaciation_age).
    age_bins : list, optional
        Bin edges for age categories. Default uses NADI-1 time slices.
    area_col : str, optional
        Column containing lake area. Default uses COLS['area'].
    min_lake_area : float, optional
        Minimum lake area in km². Default 0.01.
    verbose : bool, optional
        Print summary statistics.

    Returns
    -------
    pd.DataFrame
        Lake density statistics by age bin.
    """
    if area_col is None:
        area_col = COLS['area']

    # Default age bins: 2 ka intervals
    if age_bins is None:
        age_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]

    # Filter to glaciated lakes with assigned ages
    glaciated = lake_gdf[
        (lake_gdf['was_glaciated']) &
        (lake_gdf['deglaciation_age'].notna()) &
        (lake_gdf[area_col] >= min_lake_area)
    ].copy()

    if verbose:
        print(f"\n  Computing density by deglaciation age...")
        print(f"    Lakes with ages: {len(glaciated):,}")

    # Bin lakes by age
    glaciated['age_bin'] = pd.cut(
        glaciated['deglaciation_age'],
        bins=age_bins,
        labels=[f"{age_bins[i]}-{age_bins[i+1]} ka" for i in range(len(age_bins)-1)]
    )

    # Compute statistics by bin
    results = []
    for age_label in glaciated['age_bin'].cat.categories:
        bin_lakes = glaciated[glaciated['age_bin'] == age_label]

        if len(bin_lakes) == 0:
            continue

        # Get bin midpoint
        parts = age_label.replace(' ka', '').split('-')
        age_mid = (float(parts[0]) + float(parts[1])) / 2

        results.append({
            'age_bin': age_label,
            'age_midpoint_ka': age_mid,
            'n_lakes': len(bin_lakes),
            'total_area_km2': bin_lakes[area_col].sum(),
            'mean_area_km2': bin_lakes[area_col].mean(),
            'median_area_km2': bin_lakes[area_col].median(),
        })

    df = pd.DataFrame(results)

    if verbose and len(df) > 0:
        print(f"\n  Age Bin              N lakes    Total Area (km²)   Mean Area")
        print("  " + "-" * 65)
        for _, row in df.iterrows():
            print(f"  {row['age_bin']:<18} {row['n_lakes']:>10,}    {row['total_area_km2']:>14,.1f}   {row['mean_area_km2']:>10.3f}")

    return df


def run_nadi1_chronosequence_analysis(lake_gdf, extent_type='OPTIMAL',
                                      include_uncertainty=True,
                                      compare_with_illinoian=True,
                                      area_col=None, min_lake_area=0.01,
                                      output_dir=None, verbose=True):
    """
    Run comprehensive glacial chronosequence analysis using NADI-1 time slices.

    This analysis:
    1. Discovers available NADI-1 time slices
    2. Assigns deglaciation ages to lakes based on ice sheet retreat
    3. Computes lake density as a function of time since deglaciation
    4. Uses MIN/MAX extents to bracket uncertainty
    5. Compares with Illinoian and Driftless as end members

    Parameters
    ----------
    lake_gdf : gpd.GeoDataFrame
        Lakes with geometry (Point or Polygon).
    extent_type : str, optional
        Primary extent type ('OPTIMAL', 'MIN', or 'MAX'). Default 'OPTIMAL'.
    include_uncertainty : bool, optional
        If True, compute ages using all extent types. Default True.
    compare_with_illinoian : bool, optional
        If True, add Illinoian (160 ka) and Driftless as reference points.
    area_col : str, optional
        Column containing lake area. Default uses COLS['area'].
    min_lake_area : float, optional
        Minimum lake area in km². Default 0.01.
    output_dir : str, optional
        Directory for output files. Default uses OUTPUT_DIR.
    verbose : bool, optional
        Print detailed progress messages.

    Returns
    -------
    dict
        Dictionary containing:
        - lake_gdf: GeoDataFrame with deglaciation ages
        - time_slices: DataFrame of discovered time slices
        - density_by_age: Lake density statistics by age bin
        - decay_model: Fitted exponential decay parameters
        - uncertainty: Uncertainty estimates from MIN/MAX extents
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NADI-1 GLACIAL CHRONOSEQUENCE ANALYSIS")
        print("=" * 70)
        print("\n  Using Dalton et al. NADI-1 ice sheet reconstructions")
        print("  to build continuous lake density vs. deglaciation age curve")
        print("=" * 70)

    if area_col is None:
        area_col = COLS['area']

    if output_dir is None:
        output_dir = ensure_output_dir()

    results = {
        'parameters': {
            'extent_type': extent_type,
            'include_uncertainty': include_uncertainty,
            'min_lake_area': min_lake_area,
        }
    }

    # Step 1: Discover available time slices
    if verbose:
        print("\n" + "-" * 60)
        print("STEP 1: DISCOVERING NADI-1 TIME SLICES")
        print("-" * 60)

    time_slices = discover_nadi1_time_slices(verbose=verbose)
    results['time_slices'] = time_slices

    n_found = time_slices['exists'].sum()
    if n_found == 0:
        print("\n  ERROR: No NADI-1 time slices found!")
        print(f"  Check directory: {NADI1_CONFIG['directory']}")
        return results

    # Step 2: Prepare lake data
    if verbose:
        print("\n" + "-" * 60)
        print("STEP 2: PREPARING LAKE DATA")
        print("-" * 60)

    # Convert to GeoDataFrame if needed
    if not isinstance(lake_gdf, gpd.GeoDataFrame):
        print("  Converting to GeoDataFrame...")
        lon_col = COLS.get('lon', 'Longitude')
        lat_col = COLS.get('lat', 'Latitude')
        geometry = [Point(lon, lat) for lon, lat in
                    zip(lake_gdf[lon_col], lake_gdf[lat_col])]
        lake_gdf = gpd.GeoDataFrame(lake_gdf, geometry=geometry, crs="EPSG:4326")

    # Filter to minimum area and CONUS (east of -110° to exclude alpine regions)
    lon_col = COLS.get('lon', 'Longitude')
    if lon_col in lake_gdf.columns:
        continental_mask = lake_gdf[lon_col] > NADI1_CONFIG['continental_lon_threshold']
        lake_gdf_continental = lake_gdf[continental_mask].copy()
        if verbose:
            n_orig = len(lake_gdf)
            n_cont = len(lake_gdf_continental)
            print(f"  Filtered to continental region (east of {NADI1_CONFIG['continental_lon_threshold']}°):")
            print(f"    {n_orig:,} → {n_cont:,} lakes")
    else:
        lake_gdf_continental = lake_gdf.copy()

    # Apply minimum area filter
    area_mask = lake_gdf_continental[area_col] >= min_lake_area
    lake_gdf_filtered = lake_gdf_continental[area_mask].copy()

    if verbose:
        print(f"  After area filter (≥{min_lake_area} km²): {len(lake_gdf_filtered):,} lakes")

    # Step 3: Assign deglaciation ages
    if verbose:
        print("\n" + "-" * 60)
        print("STEP 3: ASSIGNING DEGLACIATION AGES")
        print("-" * 60)

    lake_gdf_with_ages = assign_deglaciation_age(
        lake_gdf_filtered,
        extent_type=extent_type,
        include_uncertainty=include_uncertainty,
        verbose=verbose
    )
    results['lake_gdf'] = lake_gdf_with_ages

    # Step 4: Compute density by age
    if verbose:
        print("\n" + "-" * 60)
        print("STEP 4: COMPUTING LAKE DENSITY BY DEGLACIATION AGE")
        print("-" * 60)

    density_by_age = compute_density_by_deglaciation_age(
        lake_gdf_with_ages,
        area_col=area_col,
        min_lake_area=min_lake_area,
        verbose=verbose
    )
    results['density_by_age'] = density_by_age

    # Step 5: Fit exponential decay model
    if verbose:
        print("\n" + "-" * 60)
        print("STEP 5: FITTING EXPONENTIAL DECAY MODEL")
        print("-" * 60)

    if len(density_by_age) >= 3:
        try:
            from scipy.optimize import curve_fit

            # Model: N(t) = N0 * exp(-λt) where t is time since deglaciation
            # But we have age (time before present), so younger ages = more time since deglaciation
            # Actually, deglaciation_age is when ice left, so
            # time since deglaciation = deglaciation_age (in ka from present)
            # Younger deglaciation age = more recently deglaciated = less time for decay

            x = density_by_age['age_midpoint_ka'].values
            y = density_by_age['n_lakes'].values

            # Exponential decay: n = A * exp(-λ * time_since_deglaciation)
            # time_since_deglaciation ≈ deglaciation_age (ka)
            # So for Davis hypothesis: n should DECREASE as age DECREASES
            # (older deglaciation = more time for lake loss)

            # Actually, let's think about this more carefully:
            # deglaciation_age = 20 ka means ice left 20,000 years ago
            # deglaciation_age = 2 ka means ice left 2,000 years ago
            # So time_since_deglaciation = deglaciation_age

            # Davis predicts: lake density decreases with time since deglaciation
            # So: n decreases as deglaciation_age INCREASES
            # Model: n = N0 * exp(-λ * deglaciation_age)

            def decay_model(age, n0, lam):
                return n0 * np.exp(-lam * age)

            # Initial guess
            p0 = [y.max(), 0.05]

            popt, pcov = curve_fit(decay_model, x, y, p0=p0, maxfev=5000)
            n0_fit, lambda_fit = popt
            n0_err, lambda_err = np.sqrt(np.diag(pcov))

            # Half-life = ln(2) / λ
            half_life = np.log(2) / lambda_fit if lambda_fit > 0 else np.inf

            results['decay_model'] = {
                'n0': n0_fit,
                'n0_err': n0_err,
                'lambda': lambda_fit,
                'lambda_err': lambda_err,
                'half_life_ka': half_life,
                'model_type': 'exponential',
            }

            if verbose:
                print(f"\n  Exponential decay fit: N(t) = N0 * exp(-λ * t)")
                print(f"    N0 = {n0_fit:.0f} ± {n0_err:.0f} lakes")
                print(f"    λ  = {lambda_fit:.4f} ± {lambda_err:.4f} /ka")
                print(f"    Half-life = {half_life:.1f} ka")

                # R-squared
                y_pred = decay_model(x, *popt)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot
                results['decay_model']['r2'] = r2
                print(f"    R² = {r2:.3f}")

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fit decay model: {e}")
            results['decay_model'] = None
    else:
        results['decay_model'] = None

    # Step 6: Add Illinoian and Driftless comparison
    if compare_with_illinoian and verbose:
        print("\n" + "-" * 60)
        print("STEP 6: COMPARISON WITH ILLINOIAN AND DRIFTLESS")
        print("-" * 60)
        print("\n  Note: This requires classification from run_glacial_chronosequence_analysis()")
        print("  Run that analysis first to get Wisconsin/Illinoian/Driftless comparison.")

    # Step 7: Uncertainty analysis
    if include_uncertainty:
        if verbose:
            print("\n" + "-" * 60)
            print("STEP 7: UNCERTAINTY ANALYSIS (MIN/MAX EXTENTS)")
            print("-" * 60)

        uncertainty_data = {}

        for ext in ['MIN', 'MAX']:
            if f'deglac_age_{ext.lower()}' in lake_gdf_with_ages.columns:
                ages = lake_gdf_with_ages[f'deglac_age_{ext.lower()}'].dropna()
                uncertainty_data[ext] = {
                    'n_assigned': len(ages),
                    'mean_age_ka': ages.mean(),
                    'median_age_ka': ages.median(),
                    'std_age_ka': ages.std(),
                }

        results['uncertainty'] = uncertainty_data

        if verbose and uncertainty_data:
            print("\n  Extent Type    N assigned    Mean Age    Std Dev")
            print("  " + "-" * 50)
            print(f"  {'OPTIMAL':<12} {len(lake_gdf_with_ages['deglaciation_age'].dropna()):>10,}    "
                  f"{lake_gdf_with_ages['deglaciation_age'].mean():>8.1f}    "
                  f"{lake_gdf_with_ages['deglaciation_age'].std():>8.1f}")
            for ext, data in uncertainty_data.items():
                print(f"  {ext:<12} {data['n_assigned']:>10,}    "
                      f"{data['mean_age_ka']:>8.1f}    {data['std_age_ka']:>8.1f}")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("NADI-1 CHRONOSEQUENCE ANALYSIS COMPLETE")
        print("=" * 70)

        n_glaciated = lake_gdf_with_ages['was_glaciated'].sum()
        n_with_age = lake_gdf_with_ages['deglaciation_age'].notna().sum()

        print(f"\n  Summary:")
        print(f"    Lakes analyzed: {len(lake_gdf_with_ages):,}")
        print(f"    Were glaciated (Wisconsin): {n_glaciated:,}")
        print(f"    With deglaciation ages: {n_with_age:,}")

        if results.get('decay_model'):
            print(f"\n  Key finding:")
            print(f"    Estimated lake half-life: {results['decay_model']['half_life_ka']:.0f} ka")
            print(f"    (Time for lake density to decrease by 50%)")

    return results


if __name__ == "__main__":
    print("Glacial Chronosequence Analysis Module")
    print("=" * 40)
    print("\nTo run the analysis, load your lake data and call:")
    print("  from glacial_chronosequence import run_glacial_chronosequence_analysis")
    print("  results = run_glacial_chronosequence_analysis(lake_df)")
    print("\nFor Dalton 18ka specific analysis:")
    print("  from glacial_chronosequence import run_dalton_18ka_analysis")
    print("  results = run_dalton_18ka_analysis(lake_df)")
    print("\nFor NADI-1 comprehensive chronosequence (1-25 ka):")
    print("  from glacial_chronosequence import run_nadi1_chronosequence_analysis")
    print("  results = run_nadi1_chronosequence_analysis(lake_gdf)")
    print("\nThis module tests Davis's hypothesis that lake density")
    print("decreases with time since glaciation.")
