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

    # Southern Appalachians DEM for hypsometry-normalized density
    'sapp_dem': r"F:\Lakes\GIS\rasters\S_App_DEM",  # S. Appalachian elevation

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

    # Southern Appalachians - never glaciated, mountainous region
    # CRS: UNKNOWN - will be detected and reprojected on load
    # Note: This is a comparison region with different hypsometry than glaciated lowlands
    'southern_appalachians': {
        'path': r"F:\Lakes\GIS\rasters\S_App_Lakes.dbf",
        'crs': None,  # Will be auto-detected from .prj file if available
        'description': 'Southern Appalachian lakes - never glaciated highlands',
        'age_ka': None,  # Never glaciated (Paleozoic, >250 Ma)
        'notes': 'Different lake formation processes (fluvial, structural) vs. glacial'
    },
}

# ============================================================================
# NADI-1 TIME SLICE DATASET (Dalton et al. QSR)
# ============================================================================
# Comprehensive glacial reconstructions from 1 ka to 25 ka at 0.5 ka intervals
# Each time slice has MIN, MAX, and OPTIMAL ice extent estimates
# Note: LGM is closer to 20 ka (not 25 ka) - the 25 ka slice is pre-LGM buildup

NADI1_CONFIG = {
    # Directory containing all NADI-1 shapefiles
    'directory': r"F:\Lakes\GIS\shapefiles\NADI-1 shapefiles Dalton et al. QSR",

    # File naming pattern: {age}ka_cal_{type}_NADI-1_Dalton_etal_QSR.shp
    # Ages: 1, 1.5, 2, 2.5, ... 25 (0.5 ka intervals)
    # Types: MIN, MAX, OPTIMAL
    'file_pattern': "{age}ka_cal_{extent_type}_NADI-1_Dalton_etal_QSR.shp",

    # Age range and interval (in ka)
    'age_start': 1.0,      # 1 ka (youngest)
    'age_end': 25.0,       # 25 ka (oldest, pre-LGM)
    'age_interval': 0.5,   # 0.5 ka steps

    # Note: LGM is approximately 20 ka, not 25 ka
    'lgm_age': 20.0,

    # Extent types available
    'extent_types': ['MIN', 'MAX', 'OPTIMAL'],

    # CRS of the NADI-1 shapefiles (NAD 1983 Canada Atlas Lambert)
    'crs': 'EPSG:3978',

    # Longitude threshold for continental vs alpine glaciation
    # Focus on continental ice east of -110° to exclude western alpine glaciation
    'continental_lon_threshold': -110.0,

    # Reference ages for comparison (from existing datasets)
    'reference_ages': {
        'wisconsin': 20.0,    # ka - use LGM age for comparison
        'illinoian': 160.0,   # ka - mid-point of Illinoian
        'driftless': None,    # Never glaciated
    },

    # Description
    'description': (
        'NADI-1 (North American Deglaciation Ice-sheet) reconstruction '
        'from Dalton et al. (Quaternary Science Reviews). Provides ice sheet '
        'extent at 0.5 ka intervals from 1 ka to 25 ka BP, with MIN, MAX, '
        'and OPTIMAL estimates for uncertainty quantification.'
    ),
}

# Generate list of all NADI-1 ages
def get_nadi1_ages():
    """Generate list of all NADI-1 time slice ages (in ka)."""
    import numpy as np
    ages = np.arange(
        NADI1_CONFIG['age_start'],
        NADI1_CONFIG['age_end'] + NADI1_CONFIG['age_interval'],
        NADI1_CONFIG['age_interval']
    )
    return ages.tolist()

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
# SIZE-STRATIFIED ANALYSIS PARAMETERS
# ============================================================================

# Default size bins for size-stratified analysis (min, max in km², label)
# These bins are chosen to balance:
# - Sufficient lakes in each bin for statistical power
# - Fine enough resolution to detect size-dependent trends
# - Detection limits across all glacial stages
SIZE_STRATIFIED_BINS = [
    (0.05, 0.1, 'tiny'),
    (0.1, 0.25, 'very_small'),
    (0.25, 0.5, 'small'),
    (0.5, 1.0, 'medium_small'),
    (1.0, 2.5, 'medium'),
    (2.5, 10.0, 'large'),
    (10.0, float('inf'), 'very_large')
]

# Default landscape areas (km²) for glacial stages
# These are approximate values and should be recalculated from actual boundaries
SIZE_STRATIFIED_LANDSCAPE_AREAS = {
    'Wisconsin': 1225000,           # Adjust based on actual Wisconsin boundary area
    'Illinoian': 145000,            # Adjust based on actual Illinoian boundary area
    'Driftless': 25500,             # Adjust based on actual Driftless boundary area
    'Southern_Appalachians': None   # Will be computed from DEM or boundary
}

# Age estimates for glacial stages (ka = thousands of years before present)
SIZE_STRATIFIED_AGE_ESTIMATES = {
    'Wisconsin': {'mean': 20, 'std': 5},       # ~15-25 ka
    'Illinoian': {'mean': 160, 'std': 30},     # ~130-190 ka
    'Driftless': {'mean': 1500, 'std': 500}    # >1500 ka (never glaciated)
}

# Bayesian sampling parameters for size-stratified analysis
SIZE_STRATIFIED_BAYESIAN_PARAMS = {
    'n_samples': 2000,       # Number of posterior samples per chain
    'n_tune': 1000,          # Number of tuning/warmup samples
    'n_chains': 4,           # Number of MCMC chains
    'target_accept': 0.95    # Target acceptance rate (higher = slower but more accurate)
}

# Minimum number of lakes needed in a size class for Bayesian analysis
SIZE_STRATIFIED_MIN_LAKES = 10

# Color scheme for glacial stages in size-stratified plots
SIZE_STRATIFIED_STAGE_COLORS = {
    'Wisconsin': '#3498db',             # Blue
    'Illinoian': '#e74c3c',             # Red
    'Driftless': '#2ecc71',             # Green
    'Southern_Appalachians': '#8B4513', # Brown (non-glacial highlands)
    'unclassified': '#95a5a6'           # Gray
}

# ============================================================================
# BAYESIAN HALF-LIFE ANALYSIS DEFAULTS
# ============================================================================

# Default parameters for Bayesian half-life analysis
BAYESIAN_HALFLIFE_DEFAULTS = {
    'run_overall': True,           # Run overall half-life analysis
    'run_size_stratified': True,   # Run size-stratified analysis
    'min_lake_area': 0.01,         # Minimum lake size (km²) - CRITICAL: Use 0.01 for 661 ka half-life
    'max_lake_area': 20000,        # Maximum lake size (km²) - excludes Great Lakes
    'min_lakes_per_class': 10,     # Min lakes per size class for Bayesian fit
    'test_thresholds': False,      # Test sensitivity to min_lake_area threshold
    'threshold_values': [0.01, 0.024, 0.05, 0.1],  # Thresholds to test if enabled
    'n_samples': 2000,             # PyMC samples per chain
    'n_tune': 1000,                # PyMC tuning samples
    'n_chains': 4,                 # PyMC MCMC chains
    'target_accept': 0.95          # PyMC target acceptance rate
}

# Glacial stages for half-life analysis (supports future expansion)
GLACIAL_STAGES_CONFIG = {
    'Wisconsin': {
        'age_mean_ka': 20,
        'age_std_ka': 5,
        'boundary_key': 'wisconsin',
        'required': True,
        'description': 'Most recent glaciation (~15-25 ka)'
    },
    'Illinoian': {
        'age_mean_ka': 160,
        'age_std_ka': 30,
        'boundary_key': 'illinoian',
        'required': True,
        'description': 'Older glaciation (~130-190 ka)'
    },
    'Pre-Illinoian': {
        'age_mean_ka': 500,
        'age_std_ka': 100,
        'boundary_key': 'pre_illinoian',
        'required': False,  # Not yet available - placeholder for future
        'description': 'Pre-Illinoian glaciation (>~500 ka)'
    },
    'Driftless': {
        'age_mean_ka': 1500,
        'age_std_ka': 500,
        'boundary_key': 'driftless',
        'required': True,
        'description': 'Never glaciated (>1.5 Ma)'
    }
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
