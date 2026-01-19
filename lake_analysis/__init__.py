"""
Lake Distribution Analysis Package
===================================

A Python package for analyzing the geomorphological controls on lake
distributions across the continental United States.

Core Innovation: Elevation-normalization of lake density to reveal
bimodal patterns controlled by glacial and floodplain processes.

Modules:
    config                     - Configuration settings and paths
    data_loading               - Load data from geodatabase and rasters
    normalization              - Compute normalized lake density
    visualization              - Publication-quality plotting
    powerlaw_analysis          - Power law fitting with MLE
    glacial_chronosequence     - Glacial chronosequence analysis (Davis's hypothesis)
    size_stratified_analysis   - Size-stratified lake half-life analysis
    main                       - Orchestration and pipeline

Quick Start:
    >>> from lake_analysis import load_data, analyze_elevation
    >>> lakes = load_data()
    >>> results = analyze_elevation(lakes)

Glacial Chronosequence Analysis:
    >>> from lake_analysis import analyze_glacial_chronosequence
    >>> results = analyze_glacial_chronosequence(lakes)
    # Tests Davis's hypothesis that lake density decreases with landscape age

Aridity Analysis (standalone):
    >>> from lake_analysis import analyze_aridity
    >>> results = analyze_aridity()
    # Compares aridity vs glacial stage as lake density predictors
    # Runs glacial classification first, then aridity comparison

Size-Stratified Half-Life Analysis:
    >>> from lake_analysis import run_size_stratified_analysis
    >>> results = run_size_stratified_analysis(lakes)
    # Tests whether lake half-lives vary by lake size
    # Fits Bayesian exponential decay models for each size class

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
    load_conus_lake_data,
    calculate_landscape_area_by_bin,
    quick_data_check,
    summarize_lake_data,
    sample_raster_at_points,
    sample_raster_at_coords
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
    fit_powerlaw_by_elevation_bands,
    # Bayesian estimation for small samples
    bayesian_powerlaw_estimate,
    adaptive_powerlaw_estimate,
    compute_sample_size_power
)

from .main import (
    load_data,
    analyze_elevation,
    analyze_slope,
    analyze_relief,
    analyze_powerlaw,
    analyze_glacial_chronosequence,
    analyze_aridity,
    analyze_bayesian_halflife,
    analyze_nadi1_chronosequence,
    run_full_analysis,
    quick_start
)

# NADI-1 configuration
from .config import NADI1_CONFIG, get_nadi1_ages

# Glacial chronosequence analysis
from .glacial_chronosequence import (
    run_glacial_chronosequence_analysis,
    load_all_glacial_boundaries,
    load_wisconsin_extent,
    load_illinoian_extent,
    load_driftless_area,
    load_southern_appalachian_lakes,  # New: S. Appalachian comparison region
    compute_sapp_land_area_from_dem,
    add_sapp_to_density_comparison,
    compute_sapp_hypsometry_normalized_density,
    convert_lakes_to_gdf,
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
    # Dalton 18ka analysis
    run_dalton_18ka_analysis,
    compare_wisconsin_vs_dalton_18ka,
    xmin_sensitivity_by_glacial_zone,
    # NADI-1 time slice analysis
    discover_nadi1_time_slices,
    load_nadi1_time_slice,
    assign_deglaciation_age,
    compute_density_by_deglaciation_age,
    compute_glaciated_area_timeseries,
    compute_density_by_deglaciation_age_with_area,
    run_nadi1_chronosequence_analysis,
    # Bayesian decay model
    fit_bayesian_decay_model,
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
    plot_glacial_comprehensive_summary,
    # Dalton 18ka visualizations
    plot_dalton_18ka_comparison,
    plot_wisconsin_vs_dalton_comparison,
    plot_glacial_zone_xmin_sensitivity
)

# Spatial scaling analysis
from .spatial_scaling import (
    analyze_latitudinal_scaling,
    analyze_longitudinal_scaling,
    compare_glacial_vs_nonglacial_scaling,
    analyze_elevation_size_scaling,
    run_spatial_scaling_analysis,
    create_hypothesis_summary_table
)

# Spatial scaling visualization functions
from .visualization import (
    plot_latitudinal_scaling,
    plot_glacial_vs_nonglacial_comparison,
    plot_colorful_hypothesis_table,
    plot_spatial_scaling_summary
)

# NADI-1 chronosequence visualization functions
from .visualization import (
    plot_nadi1_chronosequence,
    plot_deglaciation_age_histogram,
    plot_glaciated_area_timeseries,
    plot_density_with_uncertainty,
    plot_nadi1_density_decay
)

# Bayesian visualization functions
from .visualization import (
    plot_bayesian_posteriors,
    plot_bayesian_decay_curves,
    plot_bayesian_covariance,
    plot_bayesian_summary,
    plot_sapp_hypsometry_normalized_density
)

# Size-stratified analysis
from .size_stratified_analysis import (
    # Main pipeline
    run_size_stratified_analysis,
    # Individual components
    detection_limit_diagnostics,
    calculate_size_stratified_densities,
    plot_size_stratified_densities,
    fit_size_stratified_halflife_models,
    plot_bayesian_halflife_results,
    test_halflife_size_relationship,
    # Overall Bayesian half-life (not size-stratified)
    fit_overall_bayesian_halflife,
    plot_overall_bayesian_halflife
)

# Size-stratified analysis configuration
from .config import (
    SIZE_STRATIFIED_BINS,
    SIZE_STRATIFIED_LANDSCAPE_AREAS,
    SIZE_STRATIFIED_AGE_ESTIMATES,
    SIZE_STRATIFIED_BAYESIAN_PARAMS,
    SIZE_STRATIFIED_MIN_LAKES,
    SIZE_STRATIFIED_STAGE_COLORS,
    # Bayesian half-life configuration
    BAYESIAN_HALFLIFE_DEFAULTS,
    GLACIAL_STAGES_CONFIG
)
