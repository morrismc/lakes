"""
Configuration settings for Lake Distribution Analysis
======================================================

This module contains all paths, parameters, and constants for the analysis.
Users should modify the PATHS section for their specific system.

Project: Lake Distribution Analysis - Elevation-Normalized Density
Author: [Your Name]
"""

import os
from pathlib import Path

# ============================================================================
# PATHS - USER MODIFIES THESE FOR THEIR SYSTEM
# ============================================================================

# Lake geodatabase path (File Geodatabase)
LAKE_GDB_PATH = r"F:\Lakes\GIS\MyProject.gdb"
LAKE_FEATURE_CLASS = "Lakes_with_all_details"

# Alternative: If you export to parquet later for faster loading
LAKE_PARQUET_PATH = r"F:\Lakes\Data\lakes.parquet"

# CONUS-clipped lake dataset (created once via create_conus_lake_dataset())
# This excludes Alaska, Hawaii, and territories for Lower 48 analysis
LAKE_CONUS_PARQUET_PATH = r"F:\Lakes\Data\lakes_conus.parquet"

# Raster paths
# NOTE: For ESRI Grid rasters, point to the FOLDER (not the .adf file inside)
# For other formats (.bil, .asc), point to the actual file
RASTERS = {
    # Topographic variables (NAD_1983_Albers projection, ~93.7m cells)
    # WARNING: These are LARGE (8GB each uncompressed) - use chunked processing!
    'elevation': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_ea",
    'slope': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_slope",
    'relief_5km': r"F:\Lakes\GIS\rasters\Lower_48_DEM\lwr_48_cmplt\srtm_rlif_5k",

    # Climate variables (Geographic CRS - different from topographic!)
    'PET': r"F:\Lakes\GIS\rasters\PET_he_annual\pet_he_yr",
    'precip_4km': r"F:\Lakes\GIS\rasters\PRISM_ppt_30yr_normal_4kmM2_annual_asc\PRISM_ppt_30yr_normal_4kmM2_annual_asc.asc",
    'precip_800m': r"F:\Lakes\GIS\rasters\PRISM_ppt_30yr_normal_800mM2_annual_bil\PRISM_ppt_30yr_normal_800mM2_annual_bil.bil",
    'aridity': r"F:\Lakes\GIS\rasters\AI_annual\ai_yr",
}

# Raster metadata - NoData values and CRS info for each raster
# CRITICAL: Different rasters have different NoData values!
RASTER_METADATA = {
    'elevation': {
        'nodata': 32767,
        'crs': 'NAD_1983_Albers',  # Projected (meters)
        'units': 'meters',
        'cell_size_m': 93.7,
        'size_gb': 8.05,
    },
    'slope': {
        'nodata': -3.4028235e+38,  # Float min (special value)
        'crs': 'NAD_1983_Albers',
        'units': 'degrees',
        'cell_size_m': 93.7,
        'size_gb': 8.05,
    },
    'relief_5km': {
        'nodata': -32768,
        'crs': 'NAD_1983_Albers',
        'units': 'meters',
        'cell_size_m': 93.7,
        'size_gb': 3.96,
    },
    'PET': {
        'nodata': -32768,
        'crs': 'EPSG:4326',  # WGS 1984 (geographic, degrees)
        'units': 'mm',
        'cell_size_deg': 0.0083,  # ~1km at equator
        'size_gb': 1.45,
    },
    'precip_4km': {
        'nodata': -9999,
        'crs': 'EPSG:4269',  # NAD 1983 (geographic, degrees)
        'units': 'mm',
        'cell_size_deg': 0.0417,  # ~4km
        'size_gb': 0.003,
    },
    'precip_800m': {
        'nodata': -9999,  # Assumed - verify
        'crs': 'EPSG:4269',  # NAD 1983
        'units': 'mm',
        'cell_size_deg': 0.0083,  # ~800m
    },
    'aridity': {
        'nodata': -32768,  # Assumed - verify
        'crs': 'EPSG:4326',  # WGS 1984
        'units': 'dimensionless',
        'cell_size_deg': 0.0083,
    },
}

# ============================================================================
# SHAPEFILES
# ============================================================================
# Additional vector data for analysis

SHAPEFILES = {
    # Glacial extent boundaries by stage (Quaternary glaciation)
    # TODO: Update path when shapefile is ready
    'glacial_stages': None,  # e.g., r"F:\Lakes\GIS\glacial\glacial_stages.shp"

    # CONUS boundary for clipping (Census Bureau 2024, 1:5m resolution)
    # CRS: NAD83 (EPSG:4269) - geographic coordinates
    'conus_boundary': r"F:\Lakes\GIS\shapefiles\cb_2024_us_all_5m\cb_2024_us_nation_5m\cb_2024_us_nation_5m.shp",
}

# ============================================================================
# GLACIAL CHRONOSEQUENCE DATASETS
# ============================================================================
# Glacial boundary datasets for testing Davis's hypothesis that lake density
# decreases with time since glaciation (landscape "maturity")

# Target CRS for all glacial spatial operations (USA Contiguous Albers Equal Area)
# This is an equal-area projection essential for density calculations
GLACIAL_TARGET_CRS = "ESRI:102039"  # USA Contiguous Albers Equal Area Conic

# Alternative PROJ4 specification if ESRI code doesn't work
GLACIAL_TARGET_CRS_PROJ4 = (
    "+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 "
    "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)

# Glacial boundary file paths
GLACIAL_BOUNDARIES = {
    # Dalton 2020 North American Ice Sheets at 18ka (Last Glacial Maximum)
    # CRS: NAD 1983 Canada Atlas Lambert (EPSG:3978)
    # NOTE: Covers all of North America - must be clipped to CONUS
    'dalton_18ka': {
        'path': r"F:\Lakes\GIS\shapefiles\Dalton_2020_NA_IceSheets\shapefiles\18ka_21.7cal_MOCA_15_Aug_2019.shp",
        'crs': 'EPSG:3978',
        'description': 'Dalton 2020 NA ice sheet extent at 18ka (LGM)',
        'age_ka': 18,  # thousand years before present
    },

    # Wisconsin glaciation extent (most recent major glaciation)
    # CRS: USA Contiguous Albers (WKID:102039)
    'wisconsin': {
        'path': r"F:\Lakes\GIS\MyProject.gdb",
        'layer': 'Wisconsin_area',
        'crs': 'ESRI:102039',
        'description': 'Wisconsin glaciation maximum extent',
        'age_ka': (15, 25),  # ~15,000-25,000 years BP
    },

    # Illinoian glaciation extent (older glaciation)
    # CRS: USA Contiguous Albers (WKID:102039)
    'illinoian': {
        'path': r"F:\Lakes\GIS\MyProject.gdb",
        'layer': 'illinoian_glacial_extent',
        'crs': 'ESRI:102039',
        'description': 'Illinoian glaciation maximum extent',
        'age_ka': (130, 190),  # ~130,000-190,000 years BP
    },

    # Driftless Area - definitely never glaciated
    # CRS: NAD 1983 UTM 15N (EPSG:26915)
    'driftless_definite': {
        'path': r"F:\Lakes\GIS\MyProject.gdb",
        'layer': 'definite_driftless_area_never_glaciated',
        'crs': 'EPSG:26915',
        'description': 'Driftless Area - never glaciated (definite)',
        'age_ka': None,  # Never glaciated
    },

    # Larger Driftless Area - may have pre-Illinoian till
    # CRS: NAD 1983 UTM 15N (EPSG:26915)
    'driftless_larger': {
        'path': r"F:\Lakes\GIS\MyProject.gdb",
        'layer': 'larger_driftless_area_may_have_beenglaciatiated',
        'crs': 'EPSG:26915',
        'description': 'Larger Driftless Area - may have pre-Illinoian till',
        'age_ka': (300, None),  # Possibly >300,000 years BP if glaciated
    },
}

# Glacial stage classification order (youngest to oldest)
GLACIAL_CHRONOLOGY = {
    'wisconsin': {
        'name': 'Wisconsin',
        'age_ka': (15, 25),
        'color': '#1f77b4',  # Blue
        'order': 1,
    },
    'illinoian': {
        'name': 'Illinoian',
        'age_ka': (130, 190),
        'color': '#ff7f0e',  # Orange
        'order': 2,
    },
    'pre_illinoian': {
        'name': 'Pre-Illinoian',
        'age_ka': (300, None),
        'color': '#2ca02c',  # Green
        'order': 3,
    },
    'driftless': {
        'name': 'Driftless (Never Glaciated)',
        'age_ka': None,
        'color': '#d62728',  # Red
        'order': 4,
    },
    'alpine': {
        'name': 'Western Alpine (Dalton)',
        'age_ka': 18,
        'color': '#9467bd',  # Purple
        'order': 0,  # Special category
    },
}

# Shapefile metadata
SHAPEFILE_METADATA = {
    'conus_boundary': {
        'crs': 'EPSG:4269',  # NAD83 geographic
        'source': 'US Census Bureau TIGER/Line 2024',
        'resolution': '1:5,000,000',
        'notes': 'Nation boundary - includes Alaska, Hawaii, territories. Filter for CONUS only.',
    },
}

# Glacial stage classification
# These will be used when glacial_stages shapefile is provided
GLACIAL_STAGES = {
    'LGM': 'Last Glacial Maximum (LGM)',
    'Wisconsin': 'Wisconsin glaciation',
    'Illinoian': 'Illinoian glaciation',
    'Pre-Illinoian': 'Pre-Illinoian glaciation',
    'unglaciated': 'Never glaciated',
}

# Output directory (will be created if it doesn't exist)
OUTPUT_DIR = r"F:\Lakes\Analysis\outputs"

# ============================================================================
# CRS COMPATIBILITY NOTES
# ============================================================================
# IMPORTANT: Your rasters are in DIFFERENT coordinate systems!
#
# PROJECTED (NAD_1983_Albers, units=meters):
#   - elevation, slope, relief_5km
#   - Cell size: ~93.7m
#   - Good for area calculations (equal-area projection)
#
# GEOGRAPHIC (degrees):
#   - PET (WGS 1984 / EPSG:4326)
#   - precip_4km, precip_800m (NAD 1983 / EPSG:4269)
#   - aridity (WGS 1984 / EPSG:4326)
#   - Cell area varies by latitude!
#
# For 2D analyses (elevation × slope), use only Albers rasters (same CRS).
# For 1D analyses, the code handles geographic CRS pixel area calculation.

# ============================================================================
# COLUMN NAME MAPPING
# ============================================================================
# Column names in the geodatabase (case-sensitive!)
# Note: 'Elevation_' has a trailing underscore

COLS = {
    'area': 'AREASQKM',           # Lake surface area (km²)
    'elevation': 'Elevation_',     # Elevation at lake centroid (m) - NOTE trailing underscore!
    'slope': 'Slope',              # Terrain slope at lake location
    'relief_5km': 'F5km_relief',   # Local relief within 5km radius (m)
    'MAT': 'MAT',                  # Mean annual temperature (°C)
    'precip': 'precip_mm',         # Mean annual precipitation (mm)
    'aridity': 'AI',               # Aridity Index (dimensionless)
    'PET': 'PET',                  # Potential evapotranspiration (mm)
    'lat': 'Latitude',
    'lon': 'Longitude',
}

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Elevation binning (meters)
ELEV_BREAKS = list(range(0, 4200, 200))  # 0 to 4000m in 200m bins

# Slope binning (degrees)
SLOPE_BREAKS = list(range(0, 48, 3))  # 0 to 45° in 3° bins

# Relief binning (meters)
RELIEF_BREAKS = list(range(0, 2100, 100))  # 0 to 2000m in 100m bins

# Temperature binning (°C)
TEMP_BREAKS = list(range(-10, 30, 2))  # -10 to 28°C in 2° bins

# Precipitation binning (mm)
PRECIP_BREAKS = list(range(0, 4100, 200))  # 0 to 4000mm in 200mm bins

# Aridity Index binning (dimensionless)
AI_BREAKS = [0, 0.2, 0.5, 0.65, 1.0, 2.0, 5.0, 10.0]  # Arid to humid

# ============================================================================
# DATA QUALITY PARAMETERS
# ============================================================================

# NoData value used in the dataset
NODATA_VALUE = -9999

# Minimum lake area for reliable mapping (km²)
# Lakes below this threshold may have significant area uncertainty
# 0.024 km² is the NHD standard minimum for consistent mapping across states
# This avoids heterogeneity where some states mapped smaller water bodies
# Note: Cael & Seekell (2016) used 0.46 km² for global power law analysis
MIN_LAKE_AREA = 0.024  # NHD consistent threshold across CONUS

# Power law analysis parameters
POWERLAW_XMIN_THRESHOLD = 0.46  # km², from Cael & Seekell (2016)
MIN_LAKES_FOR_POWERLAW = 100    # Minimum lakes needed for reliable fitting

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Tile size for chunked raster processing (pixels)
RASTER_TILE_SIZE = 5000

# Target CRS for analysis (NAD83 Albers Equal Area - good for area calculations)
# EPSG:5070 - NAD83 / Conus Albers (meters)
TARGET_CRS = "EPSG:5070"

# Alternative: If your data is in a geographic CRS
# TARGET_CRS = "EPSG:4326"  # WGS84

# ============================================================================
# ELEVATION DOMAINS FOR DOMAIN-SPECIFIC ANALYSIS
# ============================================================================

ELEVATION_DOMAINS = {
    'coastal_floodplain': (0, 200),
    'low_elevation': (200, 600),
    'mid_elevation': (600, 1200),
    'high_elevation': (1200, 2000),
    'alpine': (2000, 4500),
}

# Glacial chronosequence boundaries (if available)
GLACIAL_AGES = {
    'Wisconsin': (15000, 25000),      # years BP
    'Illinoian': (130000, 190000),
    'pre_Illinoian': (300000, None),
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

PLOT_STYLE = 'seaborn-v0_8-whitegrid'

PLOT_PARAMS = {
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': (10, 6),
}

# Color schemes for different plot types
COLORMAPS = {
    'heatmap': 'magma',
    'diverging': 'RdBu_r',
    'sequential': 'viridis',
    'elevation': 'terrain',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_raster_path(raster_name):
    """Get path for a named raster, with existence check."""
    if raster_name not in RASTERS:
        raise ValueError(f"Unknown raster: {raster_name}. Available: {list(RASTERS.keys())}")
    return RASTERS[raster_name]


def print_config_summary():
    """Print summary of current configuration."""
    print("=" * 60)
    print("LAKE DISTRIBUTION ANALYSIS - Configuration Summary")
    print("=" * 60)
    print(f"\nLake Data:")
    print(f"  Geodatabase: {LAKE_GDB_PATH}")
    print(f"  Feature Class: {LAKE_FEATURE_CLASS}")
    print(f"\nRasters Available:")
    for name, path in RASTERS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  [{exists}] {name}: {path}")
    print(f"\nTarget CRS: {TARGET_CRS}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
