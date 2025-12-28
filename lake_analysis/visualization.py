"""
Visualization Module for Lake Distribution Analysis
=====================================================

This module provides publication-quality plotting functions for:
- Raw vs normalized density comparisons
- 1D density curves (elevation, slope, relief, etc.)
- 2D heatmaps (elevation × slope space, etc.)
- Power law diagnostics
- Domain comparison plots
- Bimodality visualization

All plots are designed for publication quality with customizable styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, LogLocator
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .config import (
        PLOT_STYLE, PLOT_PARAMS, COLORMAPS, OUTPUT_DIR,
        COLS, ELEVATION_DOMAINS, ensure_output_dir
    )
except ImportError:
    from config import (
        PLOT_STYLE, PLOT_PARAMS, COLORMAPS, OUTPUT_DIR,
        COLS, ELEVATION_DOMAINS, ensure_output_dir
    )


# ============================================================================
# PLOT SETUP
# ============================================================================

def setup_plot_style():
    """Apply publication-quality plot settings."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(PLOT_PARAMS)


def get_figure_path(filename, create_dir=True):
    """Get full path for saving figure."""
    if create_dir:
        ensure_output_dir()
    return Path(OUTPUT_DIR) / filename


# ============================================================================
# RAW VS NORMALIZED COMPARISON
# ============================================================================

def plot_raw_vs_normalized(result_df, value_column, units='',
                            figsize=(14, 5), save_path=None):
    """
    Side-by-side comparison of raw counts vs normalized density.

    This is the KEY visualization for showing why normalization matters.
    The bimodal pattern should only appear after normalization.

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_1d_normalized_density()
    value_column : str
        Name for x-axis labeling (e.g., 'Elevation')
    units : str
        Units for x-axis (e.g., 'm', '°')
    figsize : tuple
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    midpoints = (result_df['bin_lower'] + result_df['bin_upper']) / 2
    bar_width = (result_df['bin_upper'] - result_df['bin_lower']).iloc[0] * 0.8

    # Left panel: Raw counts
    ax1 = axes[0]
    ax1.bar(midpoints, result_df['n_lakes'],
            width=bar_width, alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xlabel(f'{value_column} ({units})' if units else value_column, fontsize=14)
    ax1.set_ylabel('Lake Count (raw)', fontsize=14)
    ax1.set_title('A) Raw Lake Counts', fontsize=16, fontweight='bold')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add total count annotation
    total = result_df['n_lakes'].sum()
    ax1.annotate(f'Total: {total:,}', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Normalized density
    ax2 = axes[1]
    ax2.plot(midpoints, result_df['normalized_density'],
             'o-', linewidth=2.5, markersize=6, color='darkred')
    ax2.fill_between(midpoints, result_df['normalized_density'],
                     alpha=0.2, color='darkred')
    ax2.set_xlabel(f'{value_column} ({units})' if units else value_column, fontsize=14)
    ax2.set_ylabel('Lakes per 1,000 km²', fontsize=14)
    ax2.set_title('B) Normalized Lake Density', fontsize=16, fontweight='bold')

    # Find and annotate peaks
    peaks = find_local_peaks(result_df['normalized_density'].values, prominence=0.1)
    if len(peaks) > 0:
        for peak_idx in peaks[:3]:  # Top 3 peaks
            peak_x = midpoints.iloc[peak_idx]
            peak_y = result_df['normalized_density'].iloc[peak_idx]
            ax2.annotate(f'{peak_y:.1f}',
                        xy=(peak_x, peak_y),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def find_local_peaks(values, prominence=0.1):
    """Find indices of local maxima in an array, handling NaN values."""
    # Convert to numpy array and handle NaN
    values = np.asarray(values, dtype=float)

    # Check if all values are NaN
    if np.all(np.isnan(values)):
        return []

    try:
        from scipy.signal import find_peaks
        # Replace NaN with -inf for peak finding (NaN breaks comparisons)
        clean_values = np.where(np.isnan(values), -np.inf, values)
        max_val = np.nanmax(values)
        if np.isnan(max_val) or max_val <= 0:
            return []
        peaks, properties = find_peaks(clean_values, prominence=prominence * max_val)
        # Sort by height (using original values)
        if len(peaks) > 0:
            # Filter out peaks that correspond to NaN positions
            valid_peaks = [p for p in peaks if not np.isnan(values[p])]
            if len(valid_peaks) > 0:
                sorted_idx = np.argsort([values[p] for p in valid_peaks])[::-1]
                return [valid_peaks[i] for i in sorted_idx]
        return []
    except ImportError:
        # Simple fallback without scipy - handle NaN in comparisons
        peaks = []
        for i in range(1, len(values) - 1):
            # Skip if any value is NaN
            if np.isnan(values[i]) or np.isnan(values[i-1]) or np.isnan(values[i+1]):
                continue
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)
        return sorted(peaks, key=lambda x: values[x], reverse=True)


# ============================================================================
# 1D DENSITY CURVES
# ============================================================================

def plot_1d_density(result_df, value_column, units='',
                    show_landscape=True, log_y=False,
                    figsize=(10, 6), save_path=None):
    """
    Plot 1D normalized density curve with optional landscape area overlay.

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_1d_normalized_density()
    value_column : str
        Name for x-axis
    units : str
        Units for x-axis
    show_landscape : bool
        If True, show landscape area as secondary y-axis
    log_y : bool
        If True, use log scale for y-axis
    """
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=figsize)

    midpoints = (result_df['bin_lower'] + result_df['bin_upper']) / 2

    # Main plot: normalized density
    color1 = 'darkred'
    ax1.plot(midpoints, result_df['normalized_density'],
             'o-', linewidth=2.5, markersize=5, color=color1, label='Normalized Density')
    ax1.set_xlabel(f'{value_column} ({units})' if units else value_column, fontsize=14)
    ax1.set_ylabel('Lakes per 1,000 km²', fontsize=14, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    if log_y:
        ax1.set_yscale('log')

    # Secondary axis: landscape area
    if show_landscape:
        ax2 = ax1.twinx()
        color2 = 'gray'
        ax2.bar(midpoints, result_df['area_km2'] / 1e3,  # Convert to 1000s km²
                alpha=0.3, width=(midpoints.iloc[1] - midpoints.iloc[0]) * 0.8,
                color=color2, label='Landscape Area')
        ax2.set_ylabel('Landscape Area (×10³ km²)', fontsize=12, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(f'Lake Density vs {value_column}', fontsize=16)
    ax1.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if show_landscape:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax1


def plot_multiple_1d_densities(results_dict, value_column, units='',
                                figsize=(12, 7), save_path=None):
    """
    Plot multiple 1D density curves on the same axes.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping labels to DataFrames from compute_1d_normalized_density()
    value_column : str
        Name for x-axis
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_dict)))

    for (label, result_df), color in zip(results_dict.items(), colors):
        midpoints = (result_df['bin_lower'] + result_df['bin_upper']) / 2
        ax.plot(midpoints, result_df['normalized_density'],
                'o-', linewidth=2, markersize=4, label=label, color=color)

    ax.set_xlabel(f'{value_column} ({units})' if units else value_column, fontsize=14)
    ax.set_ylabel('Lakes per 1,000 km²', fontsize=14)
    ax.set_title(f'Lake Density vs {value_column} by Category', fontsize=16)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_density_by_size_class(result_df, value_column, units='',
                                figsize=(12, 8), save_path=None):
    """
    Plot normalized density separately for different lake size classes.

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_1d_density_with_size_classes()
    """
    setup_plot_style()

    size_classes = result_df['size_class'].unique()
    n_classes = len(size_classes)

    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i, size_class in enumerate(size_classes):
        ax = axes[i]
        subset = result_df[result_df['size_class'] == size_class]
        midpoints = (subset['bin_lower'] + subset['bin_upper']) / 2

        ax.plot(midpoints, subset['normalized_density'],
                'o-', linewidth=2, markersize=4, color='darkblue')
        ax.fill_between(midpoints, subset['normalized_density'], alpha=0.3)

        ax.set_title(f'Size: {size_class} km²', fontsize=12)
        ax.set_ylabel('Lakes/1000 km²' if i % 2 == 0 else '')
        ax.grid(True, alpha=0.3)

        # Count
        n_lakes = subset['n_lakes'].sum()
        ax.annotate(f'n={n_lakes:,}', xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=10)

    # Remove unused axes
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel(f'{value_column} ({units})' if units else value_column, fontsize=14)
    fig.suptitle(f'Lake Density by Size Class', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ============================================================================
# 2D HEATMAPS
# ============================================================================

def plot_2d_heatmap(result_df, var1_name, var2_name,
                    var1_units='', var2_units='',
                    log_scale=True, cmap=None,
                    figsize=(12, 9), save_path=None):
    """
    Plot 2D normalized density heatmap.

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_2d_normalized_density()
    var1_name, var2_name : str
        Variable names for axis labels
    log_scale : bool
        If True, use log10 of density for coloring
    cmap : str
        Colormap name (default from config)
    """
    setup_plot_style()

    if cmap is None:
        cmap = COLORMAPS['heatmap']

    # Pivot to 2D array
    pivot_df = result_df.pivot_table(
        index=f'{var1_name}_mid',
        columns=f'{var2_name}_mid',
        values='normalized_density'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    data = pivot_df.values
    if log_scale:
        # Use log scale, handling zeros/NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.log10(data)
        data[~np.isfinite(data)] = np.nan
        cbar_label = 'log₁₀(Lakes per 1,000 km²)'
    else:
        cbar_label = 'Lakes per 1,000 km²'

    # Create heatmap
    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                   extent=[pivot_df.columns.min(), pivot_df.columns.max(),
                          pivot_df.index.min(), pivot_df.index.max()],
                   interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=12)

    # Labels
    var1_label = f'{var1_name} ({var1_units})' if var1_units else var1_name
    var2_label = f'{var2_name} ({var2_units})' if var2_units else var2_name
    ax.set_ylabel(var1_label, fontsize=14)
    ax.set_xlabel(var2_label, fontsize=14)
    ax.set_title(f'Lake Density in {var1_name} × {var2_name} Space', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_2d_heatmap_with_contours(result_df, var1_name, var2_name,
                                   var1_units='', var2_units='',
                                   n_contours=8, figsize=(12, 9), save_path=None):
    """
    2D heatmap with contour lines for clearer visualization.
    """
    setup_plot_style()

    pivot_df = result_df.pivot_table(
        index=f'{var1_name}_mid',
        columns=f'{var2_name}_mid',
        values='normalized_density'
    )

    fig, ax = plt.subplots(figsize=figsize)

    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    # Log scale
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_log = np.log10(Z)
    Z_log[~np.isfinite(Z_log)] = np.nan

    # Filled contours
    levels = np.linspace(np.nanmin(Z_log), np.nanmax(Z_log), n_contours)
    cf = ax.contourf(X, Y, Z_log, levels=levels, cmap=COLORMAPS['heatmap'])
    ax.contour(X, Y, Z_log, levels=levels, colors='white', linewidths=0.5, alpha=0.5)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('log₁₀(Lakes per 1,000 km²)', fontsize=12)

    var1_label = f'{var1_name} ({var1_units})' if var1_units else var1_name
    var2_label = f'{var2_name} ({var2_units})' if var2_units else var2_name
    ax.set_ylabel(var1_label, fontsize=14)
    ax.set_xlabel(var2_label, fontsize=14)
    ax.set_title(f'Lake Density Contours', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ============================================================================
# POWER LAW PLOTS
# ============================================================================

def plot_powerlaw_rank_size(lake_areas, xmin=None, alpha=None,
                            title="Lake Size Distribution",
                            figsize=(10, 8), save_path=None):
    """
    Plot lake size distribution as rank-size plot with power law fit.

    Parameters
    ----------
    lake_areas : array-like
        Lake areas in km²
    xmin : float, optional
        Power law lower cutoff
    alpha : float, optional
        Power law exponent
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Sort descending for rank-size
    areas = np.array(lake_areas)
    areas = areas[areas > 0]  # Remove zeros
    sorted_areas = np.sort(areas)[::-1]
    ranks = np.arange(1, len(sorted_areas) + 1)

    # Subsample for plotting efficiency if large
    if len(ranks) > 10000:
        idx = np.unique(np.logspace(0, np.log10(len(ranks)-1), 5000).astype(int))
        ranks_plot = ranks[idx]
        areas_plot = sorted_areas[idx]
    else:
        ranks_plot = ranks
        areas_plot = sorted_areas

    # Plot data
    ax.scatter(areas_plot, ranks_plot, s=8, alpha=0.5, c='steelblue', label='Data')

    # Add power law fit if provided
    if xmin is not None and alpha is not None:
        # For rank-size: rank ~ area^(1-alpha)
        x_line = np.logspace(np.log10(xmin), np.log10(max(areas)), 100)
        n_above_xmin = np.sum(areas >= xmin)
        y_line = n_above_xmin * (x_line / xmin) ** (1 - alpha)

        ax.plot(x_line, y_line, 'r--', linewidth=2.5,
                label=f'Power Law (α={alpha:.2f})')

        # Mark xmin
        ax.axvline(xmin, color='green', linestyle=':', linewidth=2,
                   label=f'x_min = {xmin:.3f} km²')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('Rank', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotations
    ax.annotate(f'N = {len(areas):,}', xy=(0.05, 0.05), xycoords='axes fraction',
               fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_powerlaw_ccdf(lake_areas, xmin=None, alpha=None,
                       figsize=(10, 8), save_path=None):
    """
    Plot complementary cumulative distribution function (CCDF).

    CCDF = P(X >= x), the probability of observing a value >= x.
    For power laws, this is a straight line on log-log axes.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    areas = np.array(lake_areas)
    areas = areas[areas > 0]
    sorted_areas = np.sort(areas)

    # CCDF: P(X >= x)
    ccdf = 1 - np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)

    ax.scatter(sorted_areas, ccdf, s=5, alpha=0.5, c='steelblue', label='Data')

    if xmin is not None and alpha is not None:
        # Theoretical CCDF for power law: P(X >= x) = (x/xmin)^(1-alpha)
        x_line = np.logspace(np.log10(xmin), np.log10(max(areas)), 100)
        y_line = (x_line / xmin) ** (1 - alpha)

        ax.plot(x_line, y_line, 'r--', linewidth=2.5,
                label=f'Power Law Fit (α={alpha:.2f})')
        ax.axvline(xmin, color='green', linestyle=':', linewidth=2,
                   label=f'x_min = {xmin:.3f} km²')

    # Add reference line for Cael & Seekell global
    x_ref = np.logspace(-1, 2, 50)
    y_ref = (x_ref / 0.46) ** (1 - 2.14)
    ax.plot(x_ref, y_ref, 'k--', linewidth=1.5, alpha=0.5,
            label='Global ref. (τ=2.14)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('P(X ≥ x)', fontsize=14)
    ax.set_title('Lake Size CCDF', fontsize=16)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_powerlaw_by_domain(fit_results, figsize=(10, 6), save_path=None):
    """
    Plot power law exponents across domains (e.g., elevation bands).

    Parameters
    ----------
    fit_results : DataFrame
        Output from fit_powerlaw_by_domain() with columns: domain, alpha, xmin
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    domains = fit_results['domain'].astype(str)
    alphas = fit_results['alpha']

    ax.scatter(domains, alphas, s=100, c='darkblue', zorder=3)

    # Reference line from Cael & Seekell (2016)
    ax.axhline(2.14, color='red', linestyle='--', linewidth=2,
               label='Cael & Seekell (2016): τ = 2.14')

    ax.set_xlabel('Domain', fontsize=14)
    ax.set_ylabel('Power Law Exponent (α)', fontsize=14)
    ax.set_title('Power Law Exponent by Domain', fontsize=16)
    ax.legend(loc='best')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ============================================================================
# DOMAIN COMPARISON PLOTS
# ============================================================================

def plot_domain_comparison(lake_df, domain_col='domain',
                           figsize=(14, 10), save_path=None):
    """
    Multi-panel comparison of lake characteristics across domains.
    """
    setup_plot_style()

    domains = lake_df[domain_col].unique()
    n_domains = len(domains)
    colors = plt.cm.Set2(np.linspace(0, 1, n_domains))
    color_map = dict(zip(domains, colors))

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    area_col = COLS['area']
    elev_col = COLS['elevation']

    # Panel A: Lake count by domain
    ax = axes[0, 0]
    counts = lake_df[domain_col].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[color_map[d] for d in counts.index])
    ax.set_ylabel('Number of Lakes', fontsize=12)
    ax.set_title('A) Lake Count by Domain', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Panel B: Area distribution by domain
    ax = axes[0, 1]
    for domain in domains:
        subset = lake_df[lake_df[domain_col] == domain][area_col]
        ax.hist(np.log10(subset), bins=30, alpha=0.5,
                label=domain, color=color_map[domain])
    ax.set_xlabel('log₁₀(Area km²)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('B) Lake Size Distribution', fontsize=14)
    ax.legend(loc='upper right')

    # Panel C: Elevation distribution by domain
    ax = axes[1, 0]
    positions = np.arange(len(domains))
    bp_data = [lake_df[lake_df[domain_col] == d][elev_col].values for d in domains]
    bp = ax.boxplot(bp_data, positions=positions, patch_artist=True)
    for patch, domain in zip(bp['boxes'], domains):
        patch.set_facecolor(color_map[domain])
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title('C) Elevation by Domain', fontsize=14)

    # Panel D: Total area by domain
    ax = axes[1, 1]
    total_areas = lake_df.groupby(domain_col)[area_col].sum()
    ax.bar(total_areas.index, total_areas.values / 1000,  # Convert to 1000s km²
           color=[color_map[d] for d in total_areas.index])
    ax.set_ylabel('Total Lake Area (×10³ km²)', fontsize=12)
    ax.set_title('D) Total Lake Area by Domain', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ============================================================================
# BIMODALITY ANALYSIS
# ============================================================================

def plot_bimodality_test(result_df, value_column, units='',
                          figsize=(12, 5), save_path=None):
    """
    Visualization for testing bimodality in normalized density.

    Shows:
    - Left: Density curve with peak annotations
    - Right: Second derivative to identify inflection points
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    midpoints = (result_df['bin_lower'] + result_df['bin_upper']) / 2
    density = result_df['normalized_density'].values

    # Left: Density with peaks
    ax1 = axes[0]
    ax1.plot(midpoints, density, 'o-', linewidth=2, markersize=5, color='darkred')
    ax1.fill_between(midpoints, density, alpha=0.2, color='darkred')

    # Find and mark peaks
    peaks = find_local_peaks(density)
    for peak_idx in peaks[:3]:
        ax1.axvline(midpoints.iloc[peak_idx], color='green', linestyle='--', alpha=0.7)
        ax1.scatter([midpoints.iloc[peak_idx]], [density[peak_idx]],
                   s=100, c='green', zorder=5)

    ax1.set_xlabel(f'{value_column} ({units})' if units else value_column)
    ax1.set_ylabel('Lakes per 1,000 km²')
    ax1.set_title('A) Normalized Density with Peaks')

    # Right: Second derivative (curvature)
    ax2 = axes[1]
    # Smooth first if needed
    try:
        from scipy.ndimage import gaussian_filter1d
        smooth_density = gaussian_filter1d(density, sigma=1)
    except ImportError:
        smooth_density = density

    # First derivative
    d1 = np.gradient(smooth_density)
    # Second derivative
    d2 = np.gradient(d1)

    ax2.plot(midpoints, d2, 'b-', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(midpoints, d2, 0, where=d2 > 0, alpha=0.3, color='green', label='Concave up')
    ax2.fill_between(midpoints, d2, 0, where=d2 < 0, alpha=0.3, color='red', label='Concave down')

    ax2.set_xlabel(f'{value_column} ({units})' if units else value_column)
    ax2.set_ylabel('Second Derivative')
    ax2.set_title('B) Curvature Analysis')
    ax2.legend(loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ============================================================================
# SUMMARY FIGURE
# ============================================================================

def create_summary_figure(elev_results, slope_results=None, relief_results=None,
                           density_2d=None, figsize=(16, 12), save_path=None):
    """
    Create comprehensive summary figure for publication.

    Parameters
    ----------
    elev_results : DataFrame
        Elevation normalized density results
    slope_results, relief_results : DataFrame, optional
        Other 1D results
    density_2d : DataFrame, optional
        2D density results for heatmap
    """
    setup_plot_style()

    n_rows = 2
    n_cols = 2 if density_2d is None else 3

    fig = plt.figure(figsize=figsize)

    # Panel A: Elevation density
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    midpoints = (elev_results['bin_lower'] + elev_results['bin_upper']) / 2
    ax1.plot(midpoints, elev_results['normalized_density'],
             'o-', linewidth=2, markersize=4, color='darkred')
    ax1.fill_between(midpoints, elev_results['normalized_density'], alpha=0.2, color='darkred')
    ax1.set_xlabel('Elevation (m)')
    ax1.set_ylabel('Lakes per 1,000 km²')
    ax1.set_title('A) Density vs Elevation', fontsize=14, fontweight='bold')

    # Panel B: Slope density (if provided)
    if slope_results is not None:
        ax2 = fig.add_subplot(n_rows, n_cols, 2)
        midpoints = (slope_results['bin_lower'] + slope_results['bin_upper']) / 2
        ax2.plot(midpoints, slope_results['normalized_density'],
                 'o-', linewidth=2, markersize=4, color='darkblue')
        ax2.fill_between(midpoints, slope_results['normalized_density'], alpha=0.2, color='darkblue')
        ax2.set_xlabel('Slope (°)')
        ax2.set_ylabel('Lakes per 1,000 km²')
        ax2.set_title('B) Density vs Slope', fontsize=14, fontweight='bold')

    # Panel C: Relief density (if provided)
    if relief_results is not None:
        ax3 = fig.add_subplot(n_rows, n_cols, n_cols + 1)
        midpoints = (relief_results['bin_lower'] + relief_results['bin_upper']) / 2
        ax3.plot(midpoints, relief_results['normalized_density'],
                 'o-', linewidth=2, markersize=4, color='darkgreen')
        ax3.fill_between(midpoints, relief_results['normalized_density'], alpha=0.2, color='darkgreen')
        ax3.set_xlabel('Relief (m)')
        ax3.set_ylabel('Lakes per 1,000 km²')
        ax3.set_title('C) Density vs Relief', fontsize=14, fontweight='bold')

    # Panel D: 2D heatmap (if provided)
    if density_2d is not None:
        ax4 = fig.add_subplot(n_rows, n_cols, n_cols + 2)
        # Simplified heatmap
        var1_col = [c for c in density_2d.columns if '_mid' in c][0]
        var2_col = [c for c in density_2d.columns if '_mid' in c][1]
        pivot = density_2d.pivot_table(
            index=var1_col, columns=var2_col, values='normalized_density'
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.log10(pivot.values)
        data[~np.isfinite(data)] = np.nan

        im = ax4.imshow(data, aspect='auto', cmap='magma', origin='lower',
                       extent=[pivot.columns.min(), pivot.columns.max(),
                              pivot.index.min(), pivot.index.max()])
        plt.colorbar(im, ax=ax4, label='log₁₀(Density)')
        ax4.set_xlabel(var2_col.replace('_mid', ''))
        ax4.set_ylabel(var1_col.replace('_mid', ''))
        ax4.set_title('D) 2D Density Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ============================================================================
# ENHANCED POWER LAW VISUALIZATIONS
# ============================================================================

def plot_powerlaw_by_elevation_multipanel(lake_df, elev_bands, area_col=None,
                                           elev_col=None, figsize=(16, 12),
                                           save_path=None):
    """
    Create multi-panel plot showing power law fits for each elevation band.

    Each panel shows the CCDF (complementary cumulative distribution function)
    with the fitted power law line.

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with area and elevation columns
    elev_bands : list
        Elevation bin edges (e.g., [0, 500, 1000, 1500, ...])
    area_col, elev_col : str
        Column names (defaults from config)

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    if area_col is None:
        area_col = COLS['area']
    if elev_col is None:
        elev_col = COLS['elevation']

    # Determine grid layout
    n_bands = len(elev_bands) - 1
    n_cols = min(3, n_bands)
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_bands == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_bands))

    for i in range(n_bands):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        elev_low, elev_high = elev_bands[i], elev_bands[i+1]
        mask = (lake_df[elev_col] >= elev_low) & (lake_df[elev_col] < elev_high)
        areas = lake_df.loc[mask, area_col].values
        areas = areas[areas > 0]

        if len(areas) < 50:
            ax.text(0.5, 0.5, f'n={len(areas)}\n(insufficient data)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{elev_low}-{elev_high} m', fontsize=12)
            continue

        # Compute CCDF
        sorted_areas = np.sort(areas)[::-1]
        ranks = np.arange(1, len(sorted_areas) + 1)
        ccdf = ranks / len(sorted_areas)

        # Plot CCDF
        ax.scatter(sorted_areas, ccdf, s=3, alpha=0.5, c=[colors[i]])

        # Fit power law (simple MLE for display)
        xmin = 0.1  # Use consistent threshold for visual comparison
        tail = areas[areas >= xmin]
        if len(tail) > 10:
            alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))

            # Plot fit line
            x_line = np.logspace(np.log10(xmin), np.log10(sorted_areas.max()), 50)
            y_line = (x_line / xmin) ** (1 - alpha)
            ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'α = {alpha:.2f}')
            ax.legend(loc='lower left', fontsize=10)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Area (km²)', fontsize=10)
        ax.set_ylabel('P(X ≥ x)', fontsize=10)
        ax.set_title(f'{elev_low}-{elev_high} m (n={len(areas):,})', fontsize=12)
        ax.grid(True, alpha=0.3, which='both')

    # Hide unused axes
    for i in range(n_bands, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Lake Size Distribution (CCDF) by Elevation Band', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_powerlaw_overlay(lake_df, elev_bands, area_col=None, elev_col=None,
                          figsize=(12, 8), save_path=None):
    """
    Overlay CCDFs from all elevation bands on single plot for comparison.

    This makes it easy to visually compare how the power law exponent
    varies with elevation.

    Parameters
    ----------
    lake_df : DataFrame
    elev_bands : list
    """
    setup_plot_style()

    if area_col is None:
        area_col = COLS['area']
    if elev_col is None:
        elev_col = COLS['elevation']

    fig, ax = plt.subplots(figsize=figsize)

    n_bands = len(elev_bands) - 1
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_bands))

    legend_entries = []

    for i in range(n_bands):
        elev_low, elev_high = elev_bands[i], elev_bands[i+1]
        mask = (lake_df[elev_col] >= elev_low) & (lake_df[elev_col] < elev_high)
        areas = lake_df.loc[mask, area_col].values
        areas = areas[areas > 0]

        if len(areas) < 50:
            continue

        # Compute CCDF
        sorted_areas = np.sort(areas)[::-1]
        ranks = np.arange(1, len(sorted_areas) + 1)
        ccdf = ranks / len(sorted_areas)

        # Subsample for plotting efficiency
        if len(sorted_areas) > 2000:
            idx = np.unique(np.logspace(0, np.log10(len(sorted_areas)-1), 1000).astype(int))
            sorted_areas = sorted_areas[idx]
            ccdf = ccdf[idx]

        # Fit alpha for label
        xmin = 0.1
        tail = areas[areas >= xmin]
        alpha = 1 + len(tail) / np.sum(np.log(tail / xmin)) if len(tail) > 10 else np.nan

        label = f'{elev_low}-{elev_high}m (α={alpha:.2f})' if not np.isnan(alpha) else f'{elev_low}-{elev_high}m'
        ax.plot(sorted_areas, ccdf, '-', linewidth=2, color=colors[i], alpha=0.8, label=label)

    # Add reference slope BEFORE setting legend
    x_ref = np.logspace(-2, 2, 50)
    y_ref = (x_ref / 0.5) ** (-1.14)  # α=2.14 means slope of -(α-1)=-1.14
    ax.plot(x_ref, y_ref, 'k--', linewidth=2.5, alpha=0.7, label='Cael & Seekell (τ=2.14)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('P(X ≥ x)', fontsize=14)
    ax.set_title('Lake Size CCDF Comparison Across Elevation Bands', fontsize=16)
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_powerlaw_explained(results, figsize=(14, 10), save_path=None):
    """
    Create an educational figure explaining power law parameters.

    Shows:
    - What α (alpha) means: steepness of decline
    - What x_min means: where power law begins
    - Comparison to Cael & Seekell (2016) global result

    Parameters
    ----------
    results : dict
        Output from full_powerlaw_analysis()
    """
    setup_plot_style()

    fig = plt.figure(figsize=figsize)

    # Panel A: What different alpha values look like
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.logspace(-1, 3, 100)
    for alpha, color, label in [(1.5, 'blue', 'α=1.5 (shallow)'),
                                  (2.0, 'green', 'α=2.0 (moderate)'),
                                  (2.14, 'red', 'α=2.14 (global)'),
                                  (2.5, 'purple', 'α=2.5 (steep)'),
                                  (3.0, 'orange', 'α=3.0 (very steep)')]:
        y = (x / 0.5) ** (1 - alpha)
        ax1.plot(x, y, color=color, linewidth=2, label=label)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Lake Area (km²)')
    ax1.set_ylabel('P(X ≥ x) - CCDF')
    ax1.set_title('A) What α (alpha) means', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_xlim(0.1, 1000)
    ax1.grid(True, alpha=0.3)

    # Add explanation text
    ax1.text(0.98, 0.98, 'Higher α → fewer large lakes\nLower α → more large lakes',
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: What x_min means
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.logspace(-2, 3, 100)
    alpha = 2.14

    for xmin, color, label in [(0.1, 'blue', 'x_min=0.1 km²'),
                                 (0.46, 'red', 'x_min=0.46 km² (global)'),
                                 (1.0, 'green', 'x_min=1.0 km²')]:
        y = np.where(x >= xmin, (x / xmin) ** (1 - alpha), np.nan)
        ax2.plot(x, y, color=color, linewidth=2, label=label)
        ax2.axvline(xmin, color=color, linestyle=':', alpha=0.5)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Lake Area (km²)')
    ax2.set_ylabel('P(X ≥ x) - CCDF')
    ax2.set_title('B) What x_min means', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.text(0.98, 0.98, 'x_min = minimum size where\npower law behavior begins',
             transform=ax2.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel C: Key metrics table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    if results:
        table_data = [
            ['Parameter', 'Value', 'Interpretation'],
            ['α (alpha)', f"{results.get('alpha', 'N/A'):.3f}", 'Power law exponent'],
            ['x_min', f"{results.get('xmin', 'N/A'):.4f} km²", 'Lower cutoff'],
            ['n (tail)', f"{results.get('n_tail', 'N/A'):,}", 'Lakes in power law regime'],
            ['KS statistic', f"{results.get('ks_statistic', 'N/A'):.4f}", 'Goodness of fit'],
            ['Global α', '2.14', 'Cael & Seekell (2016)'],
        ]

        table = ax3.table(cellText=table_data, loc='center', cellLoc='left',
                          colWidths=[0.25, 0.3, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header
        for j in range(3):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax3.set_title('C) Key Power Law Parameters', fontsize=14, fontweight='bold', y=0.95)

    # Panel D: Interpretation guide
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    interpretation_text = """
    INTERPRETING POWER LAW RESULTS

    Power Law: P(X ≥ x) ∝ x^(1-α)

    • α < 2.14: More large lakes than global average
      → May indicate tectonic basins or large glacial lakes

    • α ≈ 2.14: Similar to global distribution
      → "Typical" lake-forming processes

    • α > 2.14: Fewer large lakes than expected
      → May indicate young landscapes or constrained basins

    WHAT AFFECTS α?

    • Geologic age: Older landscapes tend toward α ≈ 2
    • Lake-forming processes: Glacial vs fluvial vs tectonic
    • Basin constraints: Topography limits maximum size
    • Scale of analysis: Regional vs continental
    """

    ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('D) Interpretation Guide', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


# ============================================================================
# ENHANCED 2D DENSITY WITH MARGINAL DISTRIBUTIONS
# ============================================================================

def plot_2d_heatmap_with_marginals(result_df, var1_name, var2_name,
                                    var1_units='', var2_units='',
                                    log_scale=True, figsize=(12, 10),
                                    save_path=None, title=None):
    """
    Create 2D heatmap with marginal density distributions along each axis.

    This is a key visualization showing:
    - Central heatmap: 2D normalized lake density
    - Top margin: 1D density vs var2 (integrated over var1)
    - Right margin: 1D density vs var1 (integrated over var2)

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_2d_normalized_density()
    var1_name, var2_name : str
        Variable names (e.g., 'Elevation_', 'Slope')
    var1_units, var2_units : str
        Units for axis labels
    log_scale : bool
        If True, use log10 color scale
    title : str, optional
        Custom title for the plot
    """
    setup_plot_style()

    # Create figure with gridspec for layout - add padding to prevent overlap
    fig = plt.figure(figsize=figsize)
    # Adjusted ratios with proper spacing to prevent axis label overlap
    gs = fig.add_gridspec(3, 3, width_ratios=[0.2, 1, 0.06], height_ratios=[0.25, 1, 0.06],
                          hspace=0.08, wspace=0.08)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])

    # Marginal axes - NOT shared to allow proper spacing
    ax_top = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Pivot data for heatmap
    var1_mid = f'{var1_name}_mid'
    var2_mid = f'{var2_name}_mid'

    pivot_df = result_df.pivot_table(
        index=var1_mid,
        columns=var2_mid,
        values='normalized_density'
    )

    # Prepare data
    data = pivot_df.values.copy()
    if log_scale:
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.log10(data)
        data[~np.isfinite(data)] = np.nan
        cbar_label = 'log₁₀(Lakes per 1,000 km²)'
    else:
        cbar_label = 'Lakes per 1,000 km²'

    # Main heatmap
    extent = [pivot_df.columns.min(), pivot_df.columns.max(),
              pivot_df.index.min(), pivot_df.index.max()]
    im = ax_main.imshow(data, aspect='auto', cmap='magma', origin='lower',
                        extent=extent, interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=11)

    # Top marginal: density vs var2 (sum over var1)
    marginal_top = result_df.groupby(var2_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index()
    marginal_top['density'] = (marginal_top['n_lakes'] / marginal_top['area_km2']) * 1000

    # Apply smoothing for cleaner marginal PDFs
    try:
        from scipy.ndimage import gaussian_filter1d
        density_smoothed = gaussian_filter1d(marginal_top['density'].values, sigma=1.5)
    except ImportError:
        density_smoothed = marginal_top['density'].values

    ax_top.fill_between(marginal_top[var2_mid], density_smoothed,
                        alpha=0.4, color='steelblue')
    ax_top.plot(marginal_top[var2_mid], density_smoothed,
                '-', color='steelblue', linewidth=2)
    ax_top.set_ylabel('Density\n(lakes/1000 km²)', fontsize=9)
    ax_top.tick_params(labelbottom=False)
    # Set proper x limits to match main plot
    ax_top.set_xlim(extent[0], extent[1])
    ax_top.set_ylim(bottom=0)
    # Remove spines for cleaner look
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Set title
    clean_var1 = var1_name.replace('_', '')
    clean_var2 = var2_name.replace('_', '')
    plot_title = title if title else f'Lake Density in {clean_var1} × {clean_var2} Space'
    ax_top.set_title(plot_title, fontsize=14, fontweight='bold', pad=10)

    # Right marginal: density vs var1 (sum over var2)
    marginal_right = result_df.groupby(var1_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index()
    marginal_right['density'] = (marginal_right['n_lakes'] / marginal_right['area_km2']) * 1000

    # Apply smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        density_smoothed_r = gaussian_filter1d(marginal_right['density'].values, sigma=1.5)
    except ImportError:
        density_smoothed_r = marginal_right['density'].values

    ax_right.fill_betweenx(marginal_right[var1_mid], density_smoothed_r,
                           alpha=0.4, color='darkred')
    ax_right.plot(density_smoothed_r, marginal_right[var1_mid],
                  '-', color='darkred', linewidth=2)
    ax_right.set_xlabel('Density\n(lakes/1000 km²)', fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.invert_xaxis()  # So it reads left-to-right from the heatmap
    # Set proper y limits to match main plot
    ax_right.set_ylim(extent[2], extent[3])
    ax_right.set_xlim(left=0)
    # Remove spines for cleaner look
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # Labels - clean up variable names
    var1_label = f'{clean_var1} ({var1_units})' if var1_units else clean_var1
    var2_label = f'{clean_var2} ({var2_units})' if var2_units else clean_var2
    ax_main.set_xlabel(var2_label, fontsize=12)
    ax_main.set_ylabel(var1_label, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax_main, ax_top, ax_right)


def plot_2d_contour_with_domains(result_df, var1_name, var2_name,
                                  var1_units='', var2_units='',
                                  figsize=(12, 9), save_path=None):
    """
    2D density contour plot with annotated geomorphic process domains.

    Shows conceptual regions where different lake-forming processes dominate.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    var1_mid = f'{var1_name}_mid'
    var2_mid = f'{var2_name}_mid'

    pivot_df = result_df.pivot_table(
        index=var1_mid,
        columns=var2_mid,
        values='normalized_density'
    )

    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    # Log transform
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_log = np.log10(Z)
    Z_log[~np.isfinite(Z_log)] = np.nan

    # Filled contours
    levels = np.linspace(np.nanmin(Z_log), np.nanmax(Z_log), 15)
    cf = ax.contourf(X, Y, Z_log, levels=levels, cmap='magma', extend='both')
    ax.contour(X, Y, Z_log, levels=levels, colors='white', linewidths=0.3, alpha=0.5)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('log₁₀(Lakes per 1,000 km²)', fontsize=12)

    # Add domain annotations (if elevation × slope)
    if 'elev' in var1_name.lower() or 'elev' in var1_mid.lower():
        # Glacial domain
        ax.annotate('GLACIAL\nDOMAIN', xy=(5, 2000), fontsize=11, fontweight='bold',
                   ha='center', color='white',
                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.5))

        # Floodplain domain
        ax.annotate('FLOODPLAIN\nDOMAIN', xy=(2, 150), fontsize=11, fontweight='bold',
                   ha='center', color='white',
                   bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))

        # "Dead zone" - steep slopes
        ax.annotate('STEEP SLOPES\n(few lakes)', xy=(25, 1000), fontsize=10,
                   ha='center', color='white', style='italic',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    var1_label = f'{var1_name.replace("_", "")} ({var1_units})' if var1_units else var1_name
    var2_label = f'{var2_name.replace("_", "")} ({var2_units})' if var2_units else var2_name
    ax.set_ylabel(var1_label, fontsize=14)
    ax.set_xlabel(var2_label, fontsize=14)
    ax.set_title('Lake Density with Process Domain Annotations', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


# ============================================================================
# ADDITIONAL USEFUL VISUALIZATIONS
# ============================================================================

def plot_lake_size_histogram_by_elevation(lake_df, elev_bands, area_col=None,
                                           elev_col=None, figsize=(14, 8),
                                           save_path=None):
    """
    Line plot showing lake size frequency distribution at different elevations.
    Uses line plots instead of filled histograms for better visibility.
    """
    setup_plot_style()

    if area_col is None:
        area_col = COLS['area']
    if elev_col is None:
        elev_col = COLS['elevation']

    fig, ax = plt.subplots(figsize=figsize)

    n_bands = len(elev_bands) - 1
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_bands))

    # Create log-spaced bins for area
    area_bins = np.logspace(-3, 2, 40)
    bin_centers = np.sqrt(area_bins[:-1] * area_bins[1:])  # Geometric mean of bin edges

    for i in range(n_bands):
        elev_low, elev_high = elev_bands[i], elev_bands[i+1]
        mask = (lake_df[elev_col] >= elev_low) & (lake_df[elev_col] < elev_high)
        areas = lake_df.loc[mask, area_col].values
        areas = areas[areas > 0]

        if len(areas) > 0:
            # Compute histogram counts
            counts, _ = np.histogram(areas, bins=area_bins)
            # Plot as line
            ax.plot(bin_centers, counts, '-', linewidth=2, color=colors[i],
                   label=f'{elev_low}-{elev_high}m (n={len(areas):,})')
            # Add markers for actual data points (sparse for readability)
            marker_idx = np.arange(0, len(bin_centers), 4)
            ax.plot(bin_centers[marker_idx], counts[marker_idx], 'o',
                   markersize=5, color=colors[i], alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')  # Log scale for y to show power law behavior
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('Number of Lakes (frequency)', fontsize=14)
    ax.set_title('Lake Size Frequency Distribution by Elevation', fontsize=16)
    ax.legend(loc='upper right', fontsize=10, ncol=2 if n_bands > 4 else 1)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_cumulative_area_by_size(lake_df, area_col=None, figsize=(10, 6),
                                  save_path=None):
    """
    Plot cumulative lake area as function of minimum lake size.

    Shows what fraction of total lake area comes from different size classes.
    """
    setup_plot_style()

    if area_col is None:
        area_col = COLS['area']

    fig, ax = plt.subplots(figsize=figsize)

    areas = lake_df[area_col].values
    areas = areas[areas > 0]
    sorted_areas = np.sort(areas)[::-1]  # Largest first

    cumsum_area = np.cumsum(sorted_areas)
    total_area = cumsum_area[-1]

    # X-axis: lake size at that rank
    ax.plot(sorted_areas, cumsum_area / total_area * 100, 'b-', linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('Cumulative % of Total Lake Area', fontsize=14)
    ax.set_title('Cumulative Lake Area Distribution', fontsize=16)
    ax.grid(True, alpha=0.3, which='both')

    # Add reference lines
    for pct in [50, 90, 99]:
        idx = np.searchsorted(cumsum_area / total_area * 100, pct)
        if idx < len(sorted_areas):
            area_threshold = sorted_areas[idx]
            ax.axhline(pct, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(area_threshold, color='gray', linestyle='--', alpha=0.5)
            ax.annotate(f'{pct}% from lakes ≥{area_threshold:.2f} km²',
                       xy=(area_threshold, pct), xytext=(10, 5),
                       textcoords='offset points', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_geographic_density_map(lake_df, lat_col=None, lon_col=None,
                                 grid_size=0.5, figsize=(14, 10),
                                 save_path=None):
    """
    Create geographic heatmap of lake density.

    Parameters
    ----------
    grid_size : float
        Size of grid cells in degrees
    """
    setup_plot_style()

    if lat_col is None:
        lat_col = COLS.get('lat', 'Latitude')
    if lon_col is None:
        lon_col = COLS.get('lon', 'Longitude')

    if lat_col not in lake_df.columns or lon_col not in lake_df.columns:
        print(f"Warning: Lat/Lon columns not found. Available: {list(lake_df.columns)}")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Create grid
    lat_bins = np.arange(lake_df[lat_col].min(), lake_df[lat_col].max() + grid_size, grid_size)
    lon_bins = np.arange(lake_df[lon_col].min(), lake_df[lon_col].max() + grid_size, grid_size)

    # Count lakes per cell
    H, xedges, yedges = np.histogram2d(lake_df[lon_col], lake_df[lat_col],
                                        bins=[lon_bins, lat_bins])

    # Plot
    with np.errstate(divide='ignore'):
        H_log = np.log10(H.T + 1)  # +1 to handle zeros

    im = ax.imshow(H_log, origin='lower', aspect='auto', cmap='YlOrRd',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('log₁₀(Lake Count + 1)', fontsize=12)

    ax.set_xlabel('Longitude (°)', fontsize=14)
    ax.set_ylabel('Latitude (°)', fontsize=14)
    ax.set_title('Geographic Distribution of Lakes (CONUS)', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


# ============================================================================
# α-ELEVATION PHASE DIAGRAM
# ============================================================================

def plot_alpha_elevation_phase_diagram(fit_results, figsize=(12, 8), save_path=None):
    """
    Create α-Elevation phase diagram showing power law exponent vs elevation.

    This is a key visualization for understanding how lake size distributions
    change with elevation, with comparison to theoretical predictions.

    Parameters
    ----------
    fit_results : DataFrame
        Output from fit_powerlaw_by_elevation_bands() with columns:
        domain, alpha, alpha_ci_lower, alpha_ci_upper, n_tail
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Extract elevation midpoints from domain labels
    valid_results = fit_results.dropna(subset=['alpha'])

    if len(valid_results) == 0:
        ax.text(0.5, 0.5, 'No valid power law fits', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig, ax

    # Parse elevation bins from domain labels
    x_positions = []
    x_labels = []
    for domain in valid_results['domain']:
        domain_str = str(domain)
        # Parse interval notation like "(0, 500]" or "0-500"
        try:
            if ',' in domain_str:
                parts = domain_str.replace('(', '').replace('[', '').replace(']', '').replace(')', '').split(',')
                low, high = float(parts[0]), float(parts[1])
            elif '-' in domain_str:
                parts = domain_str.split('-')
                low, high = float(parts[0]), float(parts[1])
            else:
                low, high = 0, 1000
            midpoint = (low + high) / 2
            x_positions.append(midpoint)
            x_labels.append(f'{int(low)}-{int(high)}')
        except:
            x_positions.append(len(x_positions) * 500)
            x_labels.append(str(domain))

    alphas = valid_results['alpha'].values

    # Error bars from bootstrap CIs
    if 'alpha_ci_lower' in valid_results.columns and 'alpha_ci_upper' in valid_results.columns:
        yerr_lower = alphas - valid_results['alpha_ci_lower'].values
        yerr_upper = valid_results['alpha_ci_upper'].values - alphas
        yerr = [yerr_lower, yerr_upper]
        # Replace NaN with 0 for plotting
        yerr = [np.nan_to_num(y, nan=0) for y in yerr]
    else:
        yerr = None

    # Plot data points with error bars
    ax.errorbar(x_positions, alphas, yerr=yerr,
                fmt='o', markersize=10, capsize=5, capthick=2,
                color='darkblue', ecolor='steelblue', linewidth=2,
                label='Observed α', zorder=3)

    # Connect points with line
    sorted_idx = np.argsort(x_positions)
    ax.plot(np.array(x_positions)[sorted_idx], np.array(alphas)[sorted_idx],
            '-', color='darkblue', alpha=0.5, linewidth=1.5)

    # Reference lines - with better visibility and positioning
    # Percolation theory prediction (τ = 2.05 for 2D percolation)
    ax.axhline(2.05, color='red', linestyle='--', linewidth=2.5, zorder=1,
               label='Percolation theory (τ = 2.05)')

    # Cael & Seekell global estimate
    ax.axhline(2.14, color='green', linestyle=':', linewidth=2.5, zorder=1,
               label='Global estimate (τ = 2.14)')

    # Shade region of uncertainty around theoretical value (no legend entry)
    ax.axhspan(2.0, 2.1, alpha=0.08, color='red', zorder=0)

    # Set y limits to provide space for annotations at the top
    current_ylim = ax.get_ylim()
    y_range = current_ylim[1] - current_ylim[0]
    ax.set_ylim(current_ylim[0], current_ylim[1] + y_range * 0.15)

    # Annotations for process domains - positioned at top of plot, above data
    if len(x_positions) > 0:
        x_range = max(x_positions) - min(x_positions)
        ylim = ax.get_ylim()
        y_top = ylim[1] - 0.02 * (ylim[1] - ylim[0])  # Near top of plot

        # Low elevation annotation
        ax.annotate('Floodplain\nDominated',
                   xy=(min(x_positions) + x_range*0.15, y_top),
                   fontsize=9, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.3))

        # High elevation annotation
        ax.annotate('Alpine/Glacial\nDominated',
                   xy=(max(x_positions) - x_range*0.15, y_top),
                   fontsize=9, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7, pad=0.3))

    ax.set_xlabel('Elevation (m)', fontsize=14)
    ax.set_ylabel('Power Law Exponent (α)', fontsize=14)
    ax.set_title('α-Elevation Phase Diagram', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add sample size annotations
    for i, (x, alpha, n) in enumerate(zip(x_positions, alphas, valid_results['n_tail'])):
        if pd.notna(n) and n > 0:
            ax.annotate(f'n={int(n):,}', xy=(x, alpha), xytext=(5, -15),
                       textcoords='offset points', fontsize=8, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_alpha_by_process_domain(fit_results, figsize=(14, 8), save_path=None):
    """
    Plot power law exponents by geomorphic process domain.

    Parameters
    ----------
    fit_results : DataFrame
        Output from fit_powerlaw_by_domain() with domain names and alpha values
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    valid_results = fit_results.dropna(subset=['alpha'])

    if len(valid_results) == 0:
        return fig, axes

    # Left panel: Bar chart with error bars
    ax1 = axes[0]
    domains = valid_results['domain'].astype(str).values
    alphas = valid_results['alpha'].values

    # Error bars
    if 'alpha_se' in valid_results.columns:
        yerr = valid_results['alpha_se'].values * 1.96  # 95% CI
        yerr = np.nan_to_num(yerr, nan=0)
    else:
        yerr = None

    x_pos = np.arange(len(domains))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(domains)))

    bars = ax1.bar(x_pos, alphas, yerr=yerr, capsize=5, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Reference lines - positioned to not overlap bars
    # Get y range for better line positioning
    y_min = min(alphas) - 0.3 if len(alphas) > 0 else 1.5
    y_max = max(alphas) + 0.3 if len(alphas) > 0 else 2.5
    ax1.set_ylim(y_min, y_max + 0.2)  # Extra space for legend

    ax1.axhline(2.05, color='red', linestyle='--', linewidth=2, zorder=1,
               label='Percolation (τ=2.05)')
    ax1.axhline(2.14, color='green', linestyle=':', linewidth=2, zorder=1,
               label='Global (τ=2.14)')

    ax1.set_xticks(x_pos)
    # Shorter domain labels to prevent overlap
    short_domains = [d.replace('_', ' ').replace('elevation', 'elev.') for d in domains]
    ax1.set_xticklabels(short_domains, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax1.set_title('A) Power Law by Process Domain', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    # Add extra bottom margin for rotated labels
    ax1.tick_params(axis='x', pad=5)

    # Right panel: Sample size context
    ax2 = axes[1]
    n_total = valid_results['n_total'].values if 'n_total' in valid_results.columns else valid_results['n_tail'].values
    n_tail = valid_results['n_tail'].values

    x = np.arange(len(domains))
    width = 0.35

    ax2.bar(x - width/2, n_total, width, label='Total lakes', color='lightblue', edgecolor='navy')
    ax2.bar(x + width/2, n_tail, width, label='In power law tail', color='darkblue', edgecolor='navy')

    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Number of Lakes', fontsize=12)
    ax2.set_title('B) Sample Size by Domain', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


# ============================================================================
# SLOPE-RELIEF HEATMAP
# ============================================================================

def plot_slope_relief_heatmap(result_df, slope_name='Slope', relief_name='F5km_relief',
                               slope_units='°', relief_units='m',
                               figsize=(14, 10), save_path=None):
    """
    Create Slope-Relief 2D heatmap with normalized lake density.

    This identifies "sweet spots" for lake formation in slope-relief space.

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_2d_normalized_density() for slope × relief
    slope_name, relief_name : str
        Column name prefixes
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, (ax_main, ax_top, ax_right)
    """
    setup_plot_style()

    # Create figure with gridspec for marginals - improved spacing
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, width_ratios=[0.2, 1, 0.06], height_ratios=[0.25, 1, 0.06],
                          hspace=0.08, wspace=0.08)

    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Find the column names
    slope_mid = f'{slope_name}_mid'
    relief_mid = f'{relief_name}_mid'

    # Check if columns exist, try alternatives
    if slope_mid not in result_df.columns:
        slope_cols = [c for c in result_df.columns if 'slope' in c.lower() and 'mid' in c.lower()]
        if slope_cols:
            slope_mid = slope_cols[0]
    if relief_mid not in result_df.columns:
        relief_cols = [c for c in result_df.columns if 'relief' in c.lower() and 'mid' in c.lower()]
        if relief_cols:
            relief_mid = relief_cols[0]

    # Pivot data
    pivot_df = result_df.pivot_table(
        index=relief_mid,
        columns=slope_mid,
        values='normalized_density'
    )

    # Log transform
    data = pivot_df.values.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        data_log = np.log10(data)
    data_log[~np.isfinite(data_log)] = np.nan

    # Main heatmap
    extent = [pivot_df.columns.min(), pivot_df.columns.max(),
              pivot_df.index.min(), pivot_df.index.max()]
    im = ax_main.imshow(data_log, aspect='auto', cmap='magma', origin='lower',
                        extent=extent, interpolation='nearest')

    # Find and mark "sweet spot" (maximum density)
    max_idx = np.unravel_index(np.nanargmax(data), data.shape)
    sweet_slope = pivot_df.columns[max_idx[1]]
    sweet_relief = pivot_df.index[max_idx[0]]
    ax_main.scatter([sweet_slope], [sweet_relief], s=200, c='lime', marker='*',
                   edgecolors='white', linewidths=2, zorder=5,
                   label=f'Peak: ({sweet_slope:.0f}°, {sweet_relief:.0f}m)')
    ax_main.legend(loc='upper right', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('log₁₀(Lakes per 1,000 km²)', fontsize=11)

    # Top marginal: density vs slope
    marginal_top = result_df.groupby(slope_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index()
    marginal_top['density'] = (marginal_top['n_lakes'] / marginal_top['area_km2']) * 1000

    # Apply smoothing for cleaner marginal PDFs
    try:
        from scipy.ndimage import gaussian_filter1d
        density_smooth_top = gaussian_filter1d(marginal_top['density'].values, sigma=1.5)
    except ImportError:
        density_smooth_top = marginal_top['density'].values

    ax_top.fill_between(marginal_top[slope_mid], density_smooth_top,
                        alpha=0.4, color='steelblue')
    ax_top.plot(marginal_top[slope_mid], density_smooth_top,
                '-', color='steelblue', linewidth=2)
    ax_top.set_ylabel('Density\n(lakes/1000 km²)', fontsize=9)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(extent[0], extent[1])
    ax_top.set_ylim(bottom=0)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.set_title('Lake Density in Slope × Relief Space', fontsize=14, fontweight='bold', pad=10)

    # Right marginal: density vs relief
    marginal_right = result_df.groupby(relief_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index()
    marginal_right['density'] = (marginal_right['n_lakes'] / marginal_right['area_km2']) * 1000

    # Apply smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        density_smooth_right = gaussian_filter1d(marginal_right['density'].values, sigma=1.5)
    except ImportError:
        density_smooth_right = marginal_right['density'].values

    ax_right.fill_betweenx(marginal_right[relief_mid], density_smooth_right,
                           alpha=0.4, color='darkred')
    ax_right.plot(density_smooth_right, marginal_right[relief_mid],
                  '-', color='darkred', linewidth=2)
    ax_right.set_xlabel('Density\n(lakes/1000 km²)', fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.invert_xaxis()
    ax_right.set_ylim(extent[2], extent[3])
    ax_right.set_xlim(left=0)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # Labels
    ax_main.set_xlabel(f'Slope ({slope_units})', fontsize=12)
    ax_main.set_ylabel(f'Relief ({relief_units})', fontsize=12)

    # Add annotations for process domains
    ax_main.annotate('Low gradient\nfloodplains', xy=(3, 100), fontsize=9,
                    ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.6))
    ax_main.annotate('Moderate terrain\n(optimal)', xy=(10, 400), fontsize=9,
                    ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.6))
    ax_main.annotate('Steep terrain\n(few lakes)', xy=(25, 1200), fontsize=9,
                    ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.6))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax_main, ax_top, ax_right)


# ============================================================================
# SENSITIVITY ANALYSIS VISUALIZATION
# ============================================================================

def plot_xmin_sensitivity(sensitivity_results, figsize=(14, 10), save_path=None):
    """
    Visualize sensitivity of power law parameters to x_min threshold.

    Parameters
    ----------
    sensitivity_results : DataFrame
        Contains columns: xmin, alpha, alpha_se, n_tail, ks_statistic
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    xmins = sensitivity_results['xmin'].values
    alphas = sensitivity_results['alpha'].values

    # Panel A: Alpha vs x_min
    ax1 = axes[0, 0]
    ax1.plot(xmins, alphas, 'o-', linewidth=2, markersize=6, color='darkblue')

    if 'alpha_se' in sensitivity_results.columns:
        se = sensitivity_results['alpha_se'].values
        ax1.fill_between(xmins, alphas - 1.96*se, alphas + 1.96*se,
                        alpha=0.3, color='steelblue', label='95% CI')

    # Reference lines
    ax1.axhline(2.14, color='red', linestyle='--', linewidth=2,
               label='Cael & Seekell (τ=2.14)')
    ax1.axhline(2.05, color='green', linestyle=':', linewidth=2,
               label='Percolation (τ=2.05)')
    ax1.axvline(0.46, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label='C&S threshold (0.46 km²)')

    ax1.set_xlabel('x_min (km²)', fontsize=12)
    ax1.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax1.set_title('A) α Sensitivity to Minimum Area', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Sample size vs x_min
    ax2 = axes[0, 1]
    n_tail = sensitivity_results['n_tail'].values
    ax2.plot(xmins, n_tail, 's-', linewidth=2, markersize=6, color='darkgreen')

    ax2.set_xlabel('x_min (km²)', fontsize=12)
    ax2.set_ylabel('Number of Lakes in Tail', fontsize=12)
    ax2.set_title('B) Sample Size vs Minimum Area', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.axvline(0.46, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.grid(True, alpha=0.3)

    # Panel C: KS statistic vs x_min
    ax3 = axes[1, 0]
    if 'ks_statistic' in sensitivity_results.columns:
        ks = sensitivity_results['ks_statistic'].values
        ax3.plot(xmins, ks, 'd-', linewidth=2, markersize=6, color='purple')

        # Mark optimal (minimum KS)
        min_ks_idx = np.nanargmin(ks)
        ax3.scatter([xmins[min_ks_idx]], [ks[min_ks_idx]], s=150, c='red',
                   marker='*', zorder=5, label=f'Optimal: {xmins[min_ks_idx]:.3f} km²')
        ax3.legend(loc='best', fontsize=10)

    ax3.set_xlabel('x_min (km²)', fontsize=12)
    ax3.set_ylabel('KS Statistic', fontsize=12)
    ax3.set_title('C) Goodness of Fit vs Minimum Area', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.axvline(0.46, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax3.grid(True, alpha=0.3)

    # Panel D: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary statistics
    key_thresholds = [0.024, 0.1, 0.46, 1.0]
    table_data = [['x_min (km²)', 'α', 'n (tail)', 'KS stat']]

    for thresh in key_thresholds:
        idx = np.argmin(np.abs(xmins - thresh))
        row = [
            f'{xmins[idx]:.3f}',
            f'{alphas[idx]:.3f}' if pd.notna(alphas[idx]) else 'N/A',
            f'{n_tail[idx]:,.0f}' if pd.notna(n_tail[idx]) else 'N/A',
        ]
        if 'ks_statistic' in sensitivity_results.columns:
            row.append(f'{ks[idx]:.4f}' if pd.notna(ks[idx]) else 'N/A')
        else:
            row.append('N/A')
        table_data.append(row)

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.2, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('D) Key Threshold Comparison', fontsize=14, fontweight='bold', y=0.85)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


# ============================================================================
# SIGNIFICANCE TESTING VISUALIZATIONS
# ============================================================================

def plot_significance_tests(test_results, figsize=(16, 10), save_path=None):
    """
    Comprehensive visualization of statistical significance tests.

    Parameters
    ----------
    test_results : dict
        Contains test statistics, p-values, and effect sizes
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Panel A: P-value summary
    ax1 = axes[0, 0]
    if 'p_values' in test_results:
        tests = list(test_results['p_values'].keys())
        p_vals = list(test_results['p_values'].values())

        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_vals]
        bars = ax1.barh(tests, p_vals, color=colors, edgecolor='black')

        # Significance thresholds
        ax1.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax1.axvline(0.10, color='orange', linestyle=':', linewidth=2, label='α = 0.10')

        ax1.set_xlabel('p-value', fontsize=12)
        ax1.set_title('A) Statistical Significance', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(0, 1)
    else:
        ax1.text(0.5, 0.5, 'No p-values provided', ha='center', va='center',
                transform=ax1.transAxes)

    # Panel B: Effect sizes
    ax2 = axes[0, 1]
    if 'effect_sizes' in test_results:
        effects = test_results['effect_sizes']
        tests = list(effects.keys())
        sizes = list(effects.values())

        colors = plt.cm.RdYlGn_r(np.array(sizes) / max(max(sizes), 1))
        ax2.barh(tests, sizes, color=colors, edgecolor='black')

        ax2.set_xlabel('Effect Size', fontsize=12)
        ax2.set_title('B) Effect Sizes', fontsize=14, fontweight='bold')

    # Panel C: Bootstrap distributions
    ax3 = axes[0, 2]
    if 'bootstrap_distributions' in test_results:
        for name, dist in test_results['bootstrap_distributions'].items():
            ax3.hist(dist, bins=50, alpha=0.5, label=name, density=True)
        ax3.set_xlabel('Parameter Value', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('C) Bootstrap Distributions', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)

    # Panel D: Confidence intervals comparison
    ax4 = axes[1, 0]
    if 'confidence_intervals' in test_results:
        ci_data = test_results['confidence_intervals']
        names = list(ci_data.keys())

        for i, (name, ci) in enumerate(ci_data.items()):
            mid = (ci[0] + ci[1]) / 2
            err = [[mid - ci[0]], [ci[1] - mid]]
            ax4.errorbar([mid], [i], xerr=err, fmt='o', markersize=8, capsize=5,
                        color=plt.cm.tab10(i), label=name)

        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names)
        ax4.set_xlabel('Value', fontsize=12)
        ax4.set_title('D) 95% Confidence Intervals', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

    # Panel E: Likelihood ratio tests
    ax5 = axes[1, 1]
    if 'likelihood_ratios' in test_results:
        lr_data = test_results['likelihood_ratios']
        comparisons = list(lr_data.keys())
        lr_values = [lr_data[c]['ratio'] for c in comparisons]
        favors = [lr_data[c]['favors'] for c in comparisons]

        colors = ['green' if 'power' in f.lower() else 'red' for f in favors]
        ax5.barh(comparisons, lr_values, color=colors, edgecolor='black')
        ax5.axvline(0, color='black', linewidth=1)
        ax5.set_xlabel('Log Likelihood Ratio', fontsize=12)
        ax5.set_title('E) Distribution Comparison', fontsize=14, fontweight='bold')

        # Add interpretation
        ax5.annotate('Favors\nPower Law', xy=(ax5.get_xlim()[1]*0.7, 0.5),
                    fontsize=10, ha='center', color='green')
        ax5.annotate('Favors\nAlternative', xy=(ax5.get_xlim()[0]*0.7, 0.5),
                    fontsize=10, ha='center', color='red')

    # Panel F: Summary interpretation
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = """
INTERPRETATION GUIDE

Significance Levels:
• p < 0.05: Strong evidence
• p < 0.10: Moderate evidence
• p ≥ 0.10: Weak/no evidence

Power Law Fit:
• KS p > 0.1: Power law plausible
• KS p < 0.1: Power law may be rejected

Effect Size Guidelines:
• Small: < 0.2
• Medium: 0.2 - 0.5
• Large: > 0.5

Key Results:
"""
    if 'summary' in test_results:
        summary_text += test_results['summary']

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('F) Interpretation Guide', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_powerlaw_gof_summary(gof_results, figsize=(12, 8), save_path=None):
    """
    Visualize power law goodness-of-fit results.

    Parameters
    ----------
    gof_results : dict
        Contains p_value, observed_ks, synthetic_ks distribution
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: KS distribution with observed value
    ax1 = axes[0, 0]
    if 'synthetic_ks_values' in gof_results:
        ax1.hist(gof_results['synthetic_ks_values'], bins=50, density=True,
                alpha=0.7, color='steelblue', edgecolor='navy',
                label='Synthetic KS values')
        ax1.axvline(gof_results['observed_ks'], color='red', linewidth=3,
                   label=f'Observed KS = {gof_results["observed_ks"]:.4f}')
        ax1.set_xlabel('KS Statistic', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('A) Goodness-of-Fit Test', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)

    # Panel B: P-value interpretation
    ax2 = axes[0, 1]
    p_val = gof_results.get('p_value', 0.5)

    # Create a gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # Background arc
    ax2.fill_between(theta, 0, r, alpha=0.1, color='gray')

    # Colored regions
    ax2.fill_between(theta[theta < np.pi*0.1], 0, r, alpha=0.5, color='red')
    ax2.fill_between(theta[(theta >= np.pi*0.1) & (theta < np.pi*0.5)], 0, r,
                    alpha=0.5, color='orange')
    ax2.fill_between(theta[theta >= np.pi*0.5], 0, r, alpha=0.5, color='green')

    # Needle for p-value
    needle_angle = np.pi * p_val
    ax2.plot([0, np.cos(needle_angle)], [0, np.sin(needle_angle)],
            'k-', linewidth=3)
    ax2.plot([np.cos(needle_angle)], [np.sin(needle_angle)], 'ko', markersize=10)

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'B) p-value = {p_val:.3f}', fontsize=14, fontweight='bold')

    # Interpretation text
    if p_val >= 0.1:
        interp = "Power law is PLAUSIBLE"
        color = 'green'
    elif p_val >= 0.05:
        interp = "Power law is MARGINAL"
        color = 'orange'
    else:
        interp = "Power law is REJECTED"
        color = 'red'

    ax2.text(0, -0.1, interp, ha='center', fontsize=14, fontweight='bold', color=color)

    # Panel C: CCDF with fit
    ax3 = axes[1, 0]
    if 'data' in gof_results and 'xmin' in gof_results:
        data = gof_results['data']
        xmin = gof_results['xmin']
        alpha = gof_results['alpha']

        sorted_data = np.sort(data)[::-1]
        ranks = np.arange(1, len(sorted_data) + 1)
        ccdf = ranks / len(sorted_data)

        ax3.scatter(sorted_data, ccdf, s=5, alpha=0.5, c='steelblue', label='Data')

        # Fit line
        x_line = np.logspace(np.log10(xmin), np.log10(sorted_data.max()), 100)
        y_line = (len(data[data >= xmin]) / len(data)) * (x_line / xmin) ** (1 - alpha)
        ax3.plot(x_line, y_line, 'r--', linewidth=2, label=f'Fit (α={alpha:.2f})')
        ax3.axvline(xmin, color='green', linestyle=':', linewidth=2, label=f'x_min={xmin:.3f}')

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Value', fontsize=12)
        ax3.set_ylabel('P(X ≥ x)', fontsize=12)
        ax3.set_title('C) CCDF with Fit', fontsize=14, fontweight='bold')
        ax3.legend(loc='lower left', fontsize=10)
        ax3.grid(True, alpha=0.3)

    # Panel D: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_lines = [
        f"POWER LAW FIT SUMMARY",
        f"{'='*40}",
        f"",
        f"Fitted Parameters:",
        f"  x_min = {gof_results.get('xmin', 'N/A'):.4f}",
        f"  α = {gof_results.get('alpha', 'N/A'):.3f}",
        f"  n (tail) = {gof_results.get('n_tail', 'N/A'):,}",
        f"",
        f"Goodness of Fit:",
        f"  KS statistic = {gof_results.get('observed_ks', 'N/A'):.4f}",
        f"  p-value = {gof_results.get('p_value', 'N/A'):.3f}",
        f"",
        f"Interpretation:",
        f"  {interp}",
    ]

    ax4.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('D) Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


# ============================================================================
# NEW 3-PANEL SUMMARY FIGURE
# ============================================================================

def plot_three_panel_summary(lake_df, elev_density, landscape_area,
                              area_col=None, elev_col=None,
                              figsize=(16, 5), save_path=None):
    """
    Create 3-panel summary figure showing:
    A) Lake distribution on the landscape (geographic map or elevation histogram)
    B) Hypsometry of the lower 48 (landscape area by elevation)
    C) Normalized lake density by elevation

    Parameters
    ----------
    lake_df : DataFrame
        Lake data with area and elevation columns
    elev_density : DataFrame
        Output from compute_1d_normalized_density() for elevation
    landscape_area : DataFrame
        Landscape area by elevation bin
    area_col, elev_col : str, optional
        Column names (defaults from config)
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    if area_col is None:
        area_col = COLS['area']
    if elev_col is None:
        elev_col = COLS['elevation']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: Lake count by elevation (distribution on landscape)
    ax1 = axes[0]
    midpoints = (elev_density['bin_lower'] + elev_density['bin_upper']) / 2
    bar_width = (elev_density['bin_upper'] - elev_density['bin_lower']).iloc[0] * 0.8

    ax1.bar(midpoints, elev_density['n_lakes'],
            width=bar_width, alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Elevation (m)', fontsize=12)
    ax1.set_ylabel('Number of Lakes', fontsize=12)
    ax1.set_title('A) Lake Distribution by Elevation', fontsize=14, fontweight='bold')

    # Add total count annotation
    total = elev_density['n_lakes'].sum()
    ax1.annotate(f'Total: {total:,} lakes', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: Hypsometry (landscape area by elevation)
    ax2 = axes[1]
    if 'area_km2' in landscape_area.columns:
        hyps_area = landscape_area['area_km2'].values / 1000  # Convert to 1000s km²
    elif 'area_km2' in elev_density.columns:
        hyps_area = elev_density['area_km2'].values / 1000
    else:
        hyps_area = np.ones(len(midpoints))  # Fallback

    ax2.fill_between(midpoints, hyps_area, alpha=0.5, color='sienna')
    ax2.plot(midpoints, hyps_area, '-', color='sienna', linewidth=2)
    ax2.set_xlabel('Elevation (m)', fontsize=12)
    ax2.set_ylabel('Landscape Area (×10³ km²)', fontsize=12)
    ax2.set_title('B) Hypsometry (Lower 48)', fontsize=14, fontweight='bold')

    # Add total area annotation
    total_area = sum(hyps_area) * 1000  # Convert back to km²
    ax2.annotate(f'Total: {total_area/1e6:.2f}M km²', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel C: Normalized lake density
    ax3 = axes[2]
    ax3.plot(midpoints, elev_density['normalized_density'],
             'o-', linewidth=2.5, markersize=6, color='darkred')
    ax3.fill_between(midpoints, elev_density['normalized_density'],
                     alpha=0.2, color='darkred')
    ax3.set_xlabel('Elevation (m)', fontsize=12)
    ax3.set_ylabel('Lakes per 1,000 km²', fontsize=12)
    ax3.set_title('C) Normalized Lake Density', fontsize=14, fontweight='bold')

    # Find and annotate peaks
    peaks = find_local_peaks(elev_density['normalized_density'].values, prominence=0.1)
    if len(peaks) > 0:
        for peak_idx in peaks[:2]:  # Top 2 peaks
            peak_x = midpoints.iloc[peak_idx]
            peak_y = elev_density['normalized_density'].iloc[peak_idx]
            ax3.annotate(f'{peak_y:.1f}',
                        xy=(peak_x, peak_y),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')

    # Add conceptual annotation
    ax3.annotate('Normalization reveals\ntrue lake concentration',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Key functions:")
    print("  - plot_raw_vs_normalized()")
    print("  - plot_powerlaw_by_elevation_multipanel()")
    print("  - plot_powerlaw_overlay()")
    print("  - plot_powerlaw_explained()")
    print("  - plot_2d_heatmap_with_marginals()")
    print("  - plot_geographic_density_map()")
    print("  - plot_1d_density()")
    print("  - plot_2d_heatmap()")
    print("  - plot_powerlaw_rank_size()")
    print("  - plot_domain_comparison()")
    print("  - plot_three_panel_summary()            # NEW")
    print("  - plot_alpha_elevation_phase_diagram()")
    print("  - plot_slope_relief_heatmap()")
    print("  - plot_xmin_sensitivity()")
    print("  - plot_significance_tests()")
