"""
Lake Distribution Analysis Package
===================================

A Python package for analyzing the geomorphological controls on lake
distributions across the continental United States.

Core Innovation: Elevation-normalization of lake density to reveal
bimodal patterns controlled by glacial and floodplain processes.

Modules:
    config           - Configuration settings and paths
    data_loading     - Load data from geodatabase and rasters
    normalization    - Compute normalized lake density
    visualization    - Publication-quality plotting
    powerlaw_analysis - Power law fitting with MLE
    main             - Orchestration and pipeline

Quick Start:
    >>> from lake_analysis import load_data, analyze_elevation
    >>> lakes = load_data()
    >>> results = analyze_elevation(lakes)

Author: [Your Name]
Project: Lake Scaling and Occurrence Analysis
"""

__version__ = '0.1.0'

# Import key functions for convenient access
from .config import (
    COLS, ELEV_BREAKS, SLOPE_BREAKS, RELIEF_BREAKS,
    ensure_output_dir, print_config_summary
)

from .data_loading import (
    load_lake_data_from_gdb,
    load_lake_data_from_parquet,
    calculate_landscape_area_by_bin,
    quick_data_check,
    summarize_lake_data
)

from .normalization import (
    compute_1d_normalized_density,
    compute_2d_normalized_density,
    classify_lake_domains
)

from .visualization import (
    plot_raw_vs_normalized,
    plot_1d_density,
    plot_2d_heatmap,
    plot_powerlaw_rank_size,
    setup_plot_style
)

from .powerlaw_analysis import (
    full_powerlaw_analysis,
    fit_powerlaw_by_elevation_bands
)

from .main import (
    load_data,
    analyze_elevation,
    analyze_slope,
    analyze_relief,
    analyze_powerlaw,
    run_full_analysis,
    quick_start
)
