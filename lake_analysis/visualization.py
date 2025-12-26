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


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Key functions:")
    print("  - plot_raw_vs_normalized()")
    print("  - plot_1d_density()")
    print("  - plot_2d_heatmap()")
    print("  - plot_powerlaw_rank_size()")
    print("  - plot_domain_comparison()")
