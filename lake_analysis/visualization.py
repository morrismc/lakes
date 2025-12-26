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

from .config import (
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

    ax.scatter(sorted_areas, ccdf, s=5, alpha=0.5, c='steelblue')

    if xmin is not None and alpha is not None:
        # Theoretical CCDF for power law: P(X >= x) = (x/xmin)^(1-alpha)
        x_line = np.logspace(np.log10(xmin), np.log10(max(areas)), 100)
        y_line = (x_line / xmin) ** (1 - alpha)

        ax.plot(x_line, y_line, 'r--', linewidth=2.5,
                label=f'Power Law (α={alpha:.2f})')
        ax.axvline(xmin, color='green', linestyle=':', linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('P(X ≥ x)', fontsize=14)
    ax.set_title('Lake Size CCDF', fontsize=16)
    ax.legend()
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

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('P(X ≥ x)', fontsize=14)
    ax.set_title('Lake Size CCDF Comparison Across Elevation Bands', fontsize=16)
    ax.legend(loc='lower left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # Add reference slope
    x_ref = np.logspace(-2, 2, 50)
    y_ref = (x_ref / 0.5) ** (-1.14)  # α=2.14 means slope of -(α-1)=-1.14
    ax.plot(x_ref, y_ref, 'k--', linewidth=2, alpha=0.5, label='Cael & Seekell (α=2.14)')

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
                                    save_path=None):
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
    """
    setup_plot_style()

    # Create figure with gridspec for layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, width_ratios=[0.15, 1, 0.05], height_ratios=[0.3, 1, 0.05],
                          hspace=0.05, wspace=0.05)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])

    # Marginal axes
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 0], sharey=ax_main)
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
    cbar.set_label(cbar_label, fontsize=12)

    # Top marginal: density vs var2 (sum over var1)
    marginal_top = result_df.groupby(var2_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index()
    marginal_top['density'] = (marginal_top['n_lakes'] / marginal_top['area_km2']) * 1000

    ax_top.fill_between(marginal_top[var2_mid], marginal_top['density'],
                        alpha=0.5, color='steelblue')
    ax_top.plot(marginal_top[var2_mid], marginal_top['density'],
                'o-', color='steelblue', linewidth=2, markersize=3)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(f'Lake Density in {var1_name} × {var2_name} Space', fontsize=14)

    # Right marginal: density vs var1 (sum over var2)
    marginal_right = result_df.groupby(var1_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index()
    marginal_right['density'] = (marginal_right['n_lakes'] / marginal_right['area_km2']) * 1000

    ax_right.fill_betweenx(marginal_right[var1_mid], marginal_right['density'],
                           alpha=0.5, color='darkred')
    ax_right.plot(marginal_right['density'], marginal_right[var1_mid],
                  'o-', color='darkred', linewidth=2, markersize=3)
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.invert_xaxis()  # So it reads left-to-right from the heatmap

    # Labels
    var1_label = f'{var1_name} ({var1_units})' if var1_units else var1_name
    var2_label = f'{var2_name} ({var2_units})' if var2_units else var2_name
    ax_main.set_xlabel(var2_label, fontsize=12)
    ax_main.set_ylabel(var1_label, fontsize=12)

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
    Stacked histogram showing lake size distribution at different elevations.
    """
    setup_plot_style()

    if area_col is None:
        area_col = COLS['area']
    if elev_col is None:
        elev_col = COLS['elevation']

    fig, ax = plt.subplots(figsize=figsize)

    n_bands = len(elev_bands) - 1
    colors = plt.cm.terrain(np.linspace(0.2, 0.8, n_bands))

    # Create log-spaced bins for area
    area_bins = np.logspace(-3, 2, 40)

    for i in range(n_bands):
        elev_low, elev_high = elev_bands[i], elev_bands[i+1]
        mask = (lake_df[elev_col] >= elev_low) & (lake_df[elev_col] < elev_high)
        areas = lake_df.loc[mask, area_col].values
        areas = areas[areas > 0]

        if len(areas) > 0:
            ax.hist(areas, bins=area_bins, alpha=0.6, color=colors[i],
                   label=f'{elev_low}-{elev_high}m (n={len(areas):,})')

    ax.set_xscale('log')
    ax.set_xlabel('Lake Area (km²)', fontsize=14)
    ax.set_ylabel('Number of Lakes', fontsize=14)
    ax.set_title('Lake Size Distribution by Elevation', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
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
