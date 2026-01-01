"""
Lake Distribution Analysis Package
===================================

A Python package for analyzing the geomorphological controls on lake
distributions across the continental United States.

Core Innovation: Elevation-normalization of lake density to reveal
bimodal patterns controlled by glacial and floodplain processes.

Modules:
    config                  - Configuration settings and paths
    data_loading            - Load data from geodatabase and rasters
    normalization           - Compute normalized lake density
    visualization           - Publication-quality plotting
    powerlaw_analysis       - Power law fitting with MLE
    glacial_chronosequence  - Glacial chronosequence analysis (Davis's hypothesis)
    main                    - Orchestration and pipeline

Quick Start:
    >>> from lake_analysis import load_data, analyze_elevation
    >>> lakes = load_data()
    >>> results = analyze_elevation(lakes)

Glacial Chronosequence Analysis:
    >>> from lake_analysis import analyze_glacial_chronosequence
    >>> results = analyze_glacial_chronosequence(lakes)
    # Tests Davis's hypothesis that lake density decreases with landscape age

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
    analyze_glacial_chronosequence,
    run_full_analysis,
    quick_start
)

# Glacial chronosequence analysis
from .glacial_chronosequence import (
    run_glacial_chronosequence_analysis,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    compute_lake_density_by_glacial_stage,
    test_davis_hypothesis,
    # Enhanced analysis functions
    classify_ice_types,
    create_mutually_exclusive_zones,
    power_law_by_glacial_zone,
    decompose_bimodal_by_glacial_status,
    western_alpine_analysis,
    validate_glacial_boundaries,
)

# Glacial visualization functions
from .visualization import (
    plot_density_by_glacial_stage,
    plot_elevation_histogram_by_glacial_stage,
    plot_davis_hypothesis_test,
    plot_glacial_extent_map,
    plot_glacial_chronosequence_summary,
    plot_bimodal_decomposition,
    plot_power_law_by_glacial_zone,
    # Enhanced glacial visualizations
    plot_normalized_density_with_glacial_overlay,
    plot_glacial_powerlaw_comparison,
    plot_glacial_lake_size_histograms,
    plot_glacial_xmin_sensitivity,
    plot_glacial_geographic_lakes,
    plot_glacial_comprehensive_summary
)
