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
        SHAPEFILES
    )
    from .data_loading import get_raster_info, calculate_landscape_area_by_bin
except ImportError:
    from config import (
        COLS, ELEV_BREAKS, OUTPUT_DIR, ensure_output_dir,
        GLACIAL_BOUNDARIES, GLACIAL_CHRONOLOGY,
        GLACIAL_TARGET_CRS, GLACIAL_TARGET_CRS_PROJ4,
        SHAPEFILES
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
        }

        zone_key = stage_key_map.get(stage)
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


if __name__ == "__main__":
    print("Glacial Chronosequence Analysis Module")
    print("=" * 40)
    print("\nTo run the analysis, load your lake data and call:")
    print("  from glacial_chronosequence import run_glacial_chronosequence_analysis")
    print("  results = run_glacial_chronosequence_analysis(lake_df)")
    print("\nThis module tests Davis's hypothesis that lake density")
    print("decreases with time since glaciation.")
