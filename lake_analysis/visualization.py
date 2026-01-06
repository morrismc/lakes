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

    # CCDF: P(X >= x) - use (n-i)/n to avoid log(0) at the end
    n = len(sorted_areas)
    ccdf = (n - np.arange(n)) / n

    ax.scatter(sorted_areas, ccdf, s=5, alpha=0.5, c='steelblue', label='Data')

    if xmin is not None and alpha is not None:
        # Theoretical CCDF for power law: P(X >= x) = p_tail * (x/xmin)^(1-alpha)
        # where p_tail is the fraction of data >= xmin
        p_tail = np.sum(areas >= xmin) / len(areas)
        x_line = np.logspace(np.log10(xmin), np.log10(max(areas)), 100)
        y_line = p_tail * (x_line / xmin) ** (1 - alpha)

        ax.plot(x_line, y_line, 'r--', linewidth=2.5,
                label=f'Power Law Fit (α={alpha:.2f})')
        ax.axvline(xmin, color='green', linestyle=':', linewidth=2,
                   label=f'x_min = {xmin:.3f} km²')

        # Add reference line for Cael & Seekell global (scaled to same p_tail for comparison)
        x_ref = np.logspace(np.log10(xmin), np.log10(max(areas)), 50)
        y_ref = p_tail * (x_ref / xmin) ** (1 - 2.14)
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

    # Panel B: Area distribution by domain - LINE PLOT for clarity
    ax = axes[0, 1]
    bins = np.linspace(-3, 3, 50)  # Log10 area bins
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for domain in domains:
        subset = lake_df[lake_df[domain_col] == domain][area_col]
        log_areas = np.log10(subset[subset > 0])
        counts, _ = np.histogram(log_areas, bins=bins)
        ax.plot(bin_centers, counts, '-', linewidth=2,
                label=domain, color=color_map[domain])

    ax.set_xlabel('log₁₀(Area km²)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('B) Lake Size Distribution', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale('log')  # Log scale helps see all domains

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

            # Plot fit line - MUST scale by fraction of data in tail
            p_tail = len(tail) / len(areas)  # Fraction of data >= xmin
            x_line = np.logspace(np.log10(xmin), np.log10(sorted_areas.max()), 50)
            # CCDF for power law: P(X >= x) = p_tail * (x/xmin)^(1-alpha) for x >= xmin
            y_line = p_tail * (x_line / xmin) ** (1 - alpha)
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
    - Left margin: 1D density vs var1 (integrated over var2)

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

    # Create figure with gridspec for layout
    # Layout: [left marginal | main | colorbar] x [top marginal | main]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.25, 1, 0.06], height_ratios=[0.2, 1],
                          hspace=0.08, wspace=0.12)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])

    # Marginal axes
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
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
    }).reset_index().sort_values(var2_mid)
    marginal_top['density'] = (marginal_top['n_lakes'] / marginal_top['area_km2']) * 1000
    marginal_top['density'] = marginal_top['density'].fillna(0)

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
    ax_top.set_xlim(extent[0], extent[1])
    ax_top.set_ylim(bottom=0)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Set title
    clean_var1 = var1_name.replace('_', '')
    clean_var2 = var2_name.replace('_', '')
    plot_title = title if title else f'Lake Density in {clean_var1} × {clean_var2} Space'
    ax_top.set_title(plot_title, fontsize=14, fontweight='bold', pad=10)

    # Left marginal: density vs var1 (sum over var2)
    marginal_left = result_df.groupby(var1_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index().sort_values(var1_mid)
    marginal_left['density'] = (marginal_left['n_lakes'] / marginal_left['area_km2']) * 1000
    marginal_left['density'] = marginal_left['density'].fillna(0)

    # Apply smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        density_smoothed_left = gaussian_filter1d(marginal_left['density'].values, sigma=1.5)
    except ImportError:
        density_smoothed_left = marginal_left['density'].values

    # Plot left marginal - density on x-axis (inverted), var1 on y-axis
    ax_left.fill_betweenx(marginal_left[var1_mid], density_smoothed_left,
                          alpha=0.4, color='darkred')
    ax_left.plot(density_smoothed_left, marginal_left[var1_mid],
                 '-', color='darkred', linewidth=2)
    ax_left.set_xlabel('Density\n(lakes/1000 km²)', fontsize=9)
    ax_left.tick_params(labelleft=False)
    ax_left.invert_xaxis()  # Density increases to the left
    ax_left.set_ylim(extent[2], extent[3])
    ax_left.set_xlim(left=0)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)

    # Labels - clean up variable names
    var1_label = f'{clean_var1} ({var1_units})' if var1_units else clean_var1
    var2_label = f'{clean_var2} ({var2_units})' if var2_units else clean_var2
    ax_main.set_xlabel(var2_label, fontsize=12)
    ax_main.set_ylabel(var1_label, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax_main, ax_top, ax_left)


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

    # Add sample size annotations - show both total and tail counts if available
    n_tail_col = valid_results['n_tail'] if 'n_tail' in valid_results.columns else None
    n_total_col = valid_results['n_total'] if 'n_total' in valid_results.columns else None

    for i, (x, alpha) in enumerate(zip(x_positions, alphas)):
        # Build annotation text
        if n_total_col is not None and pd.notna(n_total_col.iloc[i]):
            n_total = int(n_total_col.iloc[i])
            if n_tail_col is not None and pd.notna(n_tail_col.iloc[i]):
                n_tail = int(n_tail_col.iloc[i])
                label = f'n≥xmin: {n_tail:,}\n(of {n_total:,})'
            else:
                label = f'n={n_total:,}'
        elif n_tail_col is not None and pd.notna(n_tail_col.iloc[i]):
            n_tail = int(n_tail_col.iloc[i])
            label = f'n≥xmin: {n_tail:,}'
        else:
            continue

        ax.annotate(label, xy=(x, alpha), xytext=(5, -22),
                   textcoords='offset points', fontsize=7, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

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
    fig, (ax_main, ax_top, ax_left)
    """
    setup_plot_style()

    # Create figure with gridspec for marginals
    # Increased left margin width ratio for more space and better label visibility
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.25, 1, 0.05], height_ratios=[0.2, 1],
                          hspace=0.08, wspace=0.12)

    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
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

    # Pivot data - slope on x (columns), relief on y (index)
    pivot_df = result_df.pivot_table(
        index=relief_mid,
        columns=slope_mid,
        values='normalized_density'
    )

    # Sort indices to ensure correct ordering
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df[sorted(pivot_df.columns)]

    # Log transform
    data = pivot_df.values.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        data_log = np.log10(data)
    data_log[~np.isfinite(data_log)] = np.nan

    # Main heatmap using pcolormesh for accurate coordinate mapping
    slope_edges = np.array(sorted(pivot_df.columns))
    relief_edges = np.array(sorted(pivot_df.index))

    # Create mesh grid
    im = ax_main.pcolormesh(slope_edges, relief_edges, data_log,
                            cmap='magma', shading='nearest')

    # Find and mark "sweet spot" (maximum density) - use linear data for finding max
    valid_mask = np.isfinite(data)
    if valid_mask.any():
        max_idx = np.unravel_index(np.nanargmax(data), data.shape)
        sweet_relief = pivot_df.index[max_idx[0]]
        sweet_slope = pivot_df.columns[max_idx[1]]
        ax_main.scatter([sweet_slope], [sweet_relief], s=200, c='lime', marker='*',
                       edgecolors='white', linewidths=2, zorder=5,
                       label=f'Peak: ({sweet_slope:.0f}°, {sweet_relief:.0f}m)')
        ax_main.legend(loc='upper right', fontsize=10)

    # Set axis limits based on data
    ax_main.set_xlim(slope_edges.min(), slope_edges.max())
    ax_main.set_ylim(relief_edges.min(), relief_edges.max())

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('log₁₀(Lakes per 1,000 km²)', fontsize=11)

    # Top marginal: density vs slope
    marginal_top = result_df.groupby(slope_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index().sort_values(slope_mid)
    marginal_top['density'] = (marginal_top['n_lakes'] / marginal_top['area_km2']) * 1000
    marginal_top['density'] = marginal_top['density'].fillna(0)

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
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_ylim(bottom=0)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.set_title('Lake Density in Slope × Relief Space', fontsize=14, fontweight='bold', pad=10)

    # Left marginal: density vs relief (horizontal bar going left)
    marginal_left = result_df.groupby(relief_mid).agg({
        'n_lakes': 'sum',
        'area_km2': 'sum'
    }).reset_index().sort_values(relief_mid)
    marginal_left['density'] = (marginal_left['n_lakes'] / marginal_left['area_km2']) * 1000
    marginal_left['density'] = marginal_left['density'].fillna(0)

    # Apply smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        density_smooth_left = gaussian_filter1d(marginal_left['density'].values, sigma=1.5)
    except ImportError:
        density_smooth_left = marginal_left['density'].values

    # Plot relief marginal - density on x-axis (inverted), relief on y-axis
    ax_left.fill_betweenx(marginal_left[relief_mid], density_smooth_left,
                          alpha=0.4, color='darkred')
    ax_left.plot(density_smooth_left, marginal_left[relief_mid],
                 '-', color='darkred', linewidth=2)
    ax_left.set_xlabel('Density\n(lakes/1000 km²)', fontsize=9)
    ax_left.tick_params(labelleft=False)
    ax_left.invert_xaxis()  # Density increases to the left
    ax_left.set_ylim(ax_main.get_ylim())
    ax_left.set_xlim(left=0)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)

    # Labels
    ax_main.set_xlabel(f'Slope ({slope_units})', fontsize=12)
    ax_main.set_ylabel(f'Relief ({relief_units})', fontsize=12)

    # Add annotations for process domains - position based on actual data range
    slope_range = slope_edges.max() - slope_edges.min()
    relief_range = relief_edges.max() - relief_edges.min()

    ax_main.annotate('Low gradient\nfloodplains',
                    xy=(slope_edges.min() + slope_range*0.1, relief_edges.min() + relief_range*0.1),
                    fontsize=9, ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.6))
    ax_main.annotate('Moderate terrain\n(optimal)',
                    xy=(slope_edges.min() + slope_range*0.25, relief_edges.min() + relief_range*0.35),
                    fontsize=9, ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.6))
    ax_main.annotate('Steep terrain\n(few lakes)',
                    xy=(slope_edges.min() + slope_range*0.6, relief_edges.min() + relief_range*0.7),
                    fontsize=9, ha='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.6))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax_main, ax_top, ax_left)


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
                              figsize=(14, 12), save_path=None):
    """
    Create 3-panel summary figure showing:
    A) Lake count by elevation (distribution on landscape)
    B) Hypsometry of the lower 48 (landscape/terrain area by elevation)
    C) Normalized lake density by elevation

    All panels are arranged vertically and share the same x-axis (Elevation)
    for easy comparison.

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

    # Create figure with shared x-axis (elevation) - vertical layout
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Calculate midpoints and common x-axis limits
    midpoints = (elev_density['bin_lower'] + elev_density['bin_upper']) / 2
    bar_width = (elev_density['bin_upper'] - elev_density['bin_lower']).iloc[0] * 0.8
    x_min, x_max = elev_density['bin_lower'].min(), elev_density['bin_upper'].max()

    # Panel A: Lake count by elevation
    ax1 = axes[0]
    ax1.bar(midpoints, elev_density['n_lakes'],
            width=bar_width, alpha=0.7, color='steelblue', edgecolor='darkblue', linewidth=0.5)
    ax1.set_ylabel('Number of Lakes', fontsize=12)
    ax1.set_title('A) Lake Distribution', fontsize=14, fontweight='bold')

    # Add total count annotation
    total = elev_density['n_lakes'].sum()
    ax1.annotate(f'n = {total:,}', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Panel B: Hypsometry (landscape area by elevation)
    ax2 = axes[1]
    if 'area_km2' in landscape_area.columns:
        hyps_area = landscape_area['area_km2'].values / 1000  # Convert to 1000s km²
    elif 'area_km2' in elev_density.columns:
        hyps_area = elev_density['area_km2'].values / 1000
    else:
        hyps_area = np.ones(len(midpoints))  # Fallback

    ax2.bar(midpoints, hyps_area, width=bar_width, alpha=0.7, color='sienna',
            edgecolor='saddlebrown', linewidth=0.5)
    ax2.set_ylabel('Landscape Area\n(×10³ km²)', fontsize=12)
    ax2.set_title('B) Hypsometry (Lower 48)', fontsize=14, fontweight='bold')

    # Add total area annotation
    total_area = sum(hyps_area) * 1000  # Convert back to km²
    ax2.annotate(f'{total_area/1e6:.2f}M km²', xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Panel C: Normalized lake density
    ax3 = axes[2]
    ax3.bar(midpoints, elev_density['normalized_density'], width=bar_width,
            alpha=0.7, color='darkred', edgecolor='maroon', linewidth=0.5)
    ax3.set_ylabel('Lakes per 1,000 km²', fontsize=12)
    ax3.set_xlabel('Elevation (m)', fontsize=12)
    ax3.set_title('C) Normalized Lake Density', fontsize=14, fontweight='bold')

    # Find and annotate peaks
    peaks = find_local_peaks(elev_density['normalized_density'].values, prominence=0.5)
    if len(peaks) > 0:
        for peak_idx in peaks[:2]:  # Top 2 peaks
            if peak_idx < len(midpoints):
                peak_x = midpoints.iloc[peak_idx]
                peak_y = elev_density['normalized_density'].iloc[peak_idx]
                ax3.annotate(f'{peak_y:.1f}',
                            xy=(peak_x, peak_y),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold',
                            color='darkred')

    # Add explanation
    ax3.annotate('A ÷ B = C', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    # Set common x-axis label and limits
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3, axis='y')

    # Add main title
    fig.suptitle('Lake Density Normalization:\nCorrecting for Landscape Availability',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


# ============================================================================
# X_MIN SENSITIVITY BY ELEVATION VISUALIZATIONS
# ============================================================================

def plot_xmin_sensitivity_by_elevation(xmin_results, figsize=(16, 12), save_path=None):
    """
    Multi-panel visualization of x_min sensitivity analysis by elevation band.

    Creates a grid of panels, one per elevation band, showing:
    - KS statistic vs x_min
    - Optimal x_min marked
    - α values at key thresholds

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation() containing:
        - 'by_elevation': dict of DataFrames per elevation band
        - 'elevation_bands': list of (min, max) tuples
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    by_elevation = xmin_results.get('by_elevation', {})
    n_bands = len(by_elevation)

    if n_bands == 0:
        print("Warning: No elevation band results to plot")
        return None, None

    # Determine grid layout
    n_cols = min(3, n_bands)
    n_rows = int(np.ceil(n_bands / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_bands == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color palette for consistent coloring
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_bands - 1)) for i in range(n_bands)]

    for idx, (band_name, band_data) in enumerate(by_elevation.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        if 'sensitivity' not in band_data or band_data['sensitivity'] is None:
            ax.text(0.5, 0.5, f'{band_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sens = band_data['sensitivity']
        xmins = sens['xmin'].values
        ks_vals = sens['ks_statistic'].values
        alphas = sens['alpha'].values

        # Plot KS curve
        ax.plot(xmins, ks_vals, 'o-', linewidth=2, markersize=4,
                color=colors[idx], label='KS statistic')

        # Mark optimal x_min
        optimal = band_data.get('optimal', {})
        if optimal and 'xmin' in optimal:
            opt_xmin = optimal['xmin']
            opt_ks = optimal.get('ks', np.nanmin(ks_vals))
            ax.scatter([opt_xmin], [opt_ks], s=150, c='red', marker='*', zorder=10,
                      label=f'Optimal: {opt_xmin:.3f}')

        # Mark tolerance range if available
        if 'tolerance_range' in band_data:
            tol = band_data['tolerance_range']
            ax.axvspan(tol.get('xmin_lower', xmins[0]), tol.get('xmin_upper', xmins[-1]),
                      alpha=0.2, color='green', label='Within tolerance')

        # Reference lines
        ax.axvline(0.46, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(0.024, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)

        # Labels and formatting
        ax.set_xlabel('x_min (km²)', fontsize=10)
        ax.set_ylabel('KS Statistic', fontsize=10)
        ax.set_title(f'{band_name}', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Add sample size annotation
        n_total = band_data.get('n_total', 0)
        n_tail = band_data.get('n_tail_at_optimal', optimal.get('n_tail', 0))
        ax.annotate(f'n={n_total:,}\n(tail: {n_tail:,})',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   ha='left', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for idx in range(n_bands, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('KS Statistic vs x_min by Elevation Band', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_ks_curves_overlay(xmin_results, figsize=(12, 8), save_path=None):
    """
    Overlay KS curves for all elevation bands on a single plot.

    Allows direct comparison of how optimal x_min varies by elevation.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    by_elevation = xmin_results.get('by_elevation', {})
    n_bands = len(by_elevation)

    if n_bands == 0:
        print("Warning: No elevation band results to plot")
        return fig, ax

    # Color palette
    cmap = plt.cm.plasma
    colors = [cmap(i / max(1, n_bands - 1)) for i in range(n_bands)]

    # Track optimal points for summary
    optimal_points = []

    for idx, (band_name, band_data) in enumerate(by_elevation.items()):
        if 'sensitivity' not in band_data or band_data['sensitivity'] is None:
            continue

        sens = band_data['sensitivity']
        xmins = sens['xmin'].values
        ks_vals = sens['ks_statistic'].values

        ax.plot(xmins, ks_vals, '-', linewidth=2, color=colors[idx],
                label=band_name, alpha=0.8)

        # Mark optimal
        optimal = band_data.get('optimal', {})
        if optimal and 'xmin' in optimal:
            opt_xmin = optimal['xmin']
            opt_ks = optimal.get('ks', np.nanmin(ks_vals))
            ax.scatter([opt_xmin], [opt_ks], s=100, c=[colors[idx]],
                      marker='o', edgecolor='black', linewidth=1.5, zorder=10)
            optimal_points.append((opt_xmin, opt_ks, band_name))

    # Reference lines
    ax.axvline(0.46, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Cael & Seekell (0.46 km²)')
    ax.axvline(0.024, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label='NHD threshold (0.024 km²)')

    ax.set_xlabel('x_min (km²)', fontsize=12)
    ax.set_ylabel('KS Statistic', fontsize=12)
    ax.set_title('KS Curves by Elevation: Finding Optimal x_min', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    # Add interpretation note
    ax.annotate('Lower KS = Better fit\nCircles mark optimal x_min for each band',
               xy=(0.02, 0.02), xycoords='axes fraction',
               ha='left', va='bottom', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_optimal_xmin_vs_elevation(xmin_results, figsize=(12, 8), save_path=None):
    """
    Plot how optimal x_min varies with elevation.

    This reveals whether different geomorphic processes at different
    elevations produce lakes with different size distributions.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    by_elevation = xmin_results.get('by_elevation', {})

    # Extract data for plotting
    elevations = []
    opt_xmins = []
    opt_alphas = []
    n_tails = []
    ks_stats = []

    for band_name, band_data in by_elevation.items():
        # Parse elevation from band name (e.g., "0-500m" -> 250)
        try:
            parts = band_name.replace('m', '').split('-')
            elev_mid = (float(parts[0]) + float(parts[1])) / 2
        except:
            continue

        optimal = band_data.get('optimal', {})
        if optimal and 'xmin' in optimal:
            elevations.append(elev_mid)
            opt_xmins.append(optimal['xmin'])
            opt_alphas.append(optimal.get('alpha', np.nan))
            n_tails.append(optimal.get('n_tail', np.nan))
            ks_stats.append(optimal.get('ks', np.nan))

    if len(elevations) == 0:
        print("Warning: No optimal x_min data to plot")
        return fig, axes

    elevations = np.array(elevations)
    opt_xmins = np.array(opt_xmins)
    opt_alphas = np.array(opt_alphas)

    # Panel A: Optimal x_min vs elevation
    ax1 = axes[0, 0]
    ax1.plot(elevations, opt_xmins, 'o-', linewidth=2, markersize=10, color='darkblue')
    ax1.axhline(0.46, color='red', linestyle='--', linewidth=2, label='C&S threshold')
    ax1.axhline(0.024, color='gray', linestyle=':', linewidth=2, label='NHD threshold')
    ax1.set_xlabel('Elevation (m)', fontsize=12)
    ax1.set_ylabel('Optimal x_min (km²)', fontsize=12)
    ax1.set_title('A) Optimal x_min by Elevation', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel B: α at optimal x_min vs elevation
    ax2 = axes[0, 1]
    ax2.plot(elevations, opt_alphas, 's-', linewidth=2, markersize=10, color='darkgreen')
    ax2.axhline(2.14, color='red', linestyle='--', linewidth=2, label='Cael & Seekell (2.14)')
    ax2.axhline(2.05, color='orange', linestyle=':', linewidth=2, label='Percolation (2.05)')
    ax2.set_xlabel('Elevation (m)', fontsize=12)
    ax2.set_ylabel('α at Optimal x_min', fontsize=12)
    ax2.set_title('B) Power Law Exponent by Elevation', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Panel C: Sample size in tail vs elevation
    ax3 = axes[1, 0]
    ax3.bar(elevations, n_tails, width=200, alpha=0.7, color='steelblue', edgecolor='darkblue')
    ax3.set_xlabel('Elevation (m)', fontsize=12)
    ax3.set_ylabel('n (tail at optimal x_min)', fontsize=12)
    ax3.set_title('C) Tail Sample Size by Elevation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Summary scatter (x_min vs α colored by elevation)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(opt_xmins, opt_alphas, c=elevations, s=150,
                          cmap='viridis', edgecolor='black', linewidth=1)
    cbar = plt.colorbar(scatter, ax=ax4, label='Elevation (m)')

    # Reference lines
    ax4.axhline(2.14, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(2.05, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax4.axvline(0.46, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax4.set_xlabel('Optimal x_min (km²)', fontsize=12)
    ax4.set_ylabel('α at Optimal x_min', fontsize=12)
    ax4.set_title('D) x_min vs α (colored by elevation)', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Does Optimal x_min Vary with Elevation?', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_alpha_stability_by_elevation(xmin_results, figsize=(14, 8), save_path=None):
    """
    Visualize α stability across x_min choices for each elevation band.

    Shows the range of α values within the "acceptable" x_min range
    (within tolerance of optimal KS).

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Handle both old nested structure and direct band structure
    by_elevation = xmin_results.get('by_elevation', {})
    if not by_elevation:
        # Data might be directly keyed by band names (e.g., "0-500m")
        skip_keys = {'method_comparison', 'robustness', 'summary_table',
                     'hypothesis_tests', 'hypothesis_report'}
        by_elevation = {k: v for k, v in xmin_results.items()
                        if k not in skip_keys and isinstance(v, dict) and 'status' in v}

    bands = []
    alphas_opt = []
    alphas_low = []
    alphas_high = []
    alphas_fixed_046 = []
    alphas_fixed_024 = []

    for band_name, band_data in by_elevation.items():
        # Skip if status is not success
        if band_data.get('status') != 'success':
            continue

        bands.append(band_name)

        # Handle both nested structure and flat structure
        if 'optimal' in band_data:
            # Nested structure
            optimal = band_data.get('optimal', {})
            opt_alpha = optimal.get('alpha', np.nan)
        else:
            # Flat structure from xmin_sensitivity_by_elevation
            opt_alpha = band_data.get('optimal_alpha', np.nan)

        alphas_opt.append(opt_alpha)

        # Get α range from tolerance range or acceptable_range
        if 'tolerance_range' in band_data:
            tol = band_data.get('tolerance_range', {})
            alphas_low.append(tol.get('alpha_min', opt_alpha - 0.05))
            alphas_high.append(tol.get('alpha_max', opt_alpha + 0.05))
        elif 'alpha_stability' in band_data and pd.notna(band_data.get('alpha_stability')):
            # Use alpha_stability as half-range
            stability = band_data['alpha_stability']
            alphas_low.append(opt_alpha - stability/2)
            alphas_high.append(opt_alpha + stability/2)
        else:
            alphas_low.append(opt_alpha - 0.05)
            alphas_high.append(opt_alpha + 0.05)

        # Get α at fixed thresholds
        if 'fixed_xmin' in band_data:
            fixed = band_data.get('fixed_xmin', {})
            alphas_fixed_046.append(fixed.get(0.46, {}).get('alpha', np.nan))
            alphas_fixed_024.append(fixed.get(0.024, {}).get('alpha', np.nan))
        elif 'alpha_at_fixed_xmin' in band_data:
            fixed = band_data.get('alpha_at_fixed_xmin', {})
            alphas_fixed_046.append(fixed.get(0.46, {}).get('alpha', np.nan) if isinstance(fixed.get(0.46), dict) else np.nan)
            alphas_fixed_024.append(fixed.get(0.024, {}).get('alpha', np.nan) if isinstance(fixed.get(0.024), dict) else np.nan)
        else:
            alphas_fixed_046.append(np.nan)
            alphas_fixed_024.append(np.nan)

    # Check if we have any data to plot
    if len(bands) == 0:
        ax.text(0.5, 0.5, 'No elevation band data available\nfor x_min sensitivity analysis',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('α Stability: How Sensitive is α to x_min Choice?', fontsize=14, fontweight='bold')
        return fig, ax

    x = np.arange(len(bands))
    width = 0.25

    # Plot bars for different α estimates
    ax.bar(x - width, alphas_opt, width, label='α at optimal x_min', color='steelblue', alpha=0.8)
    ax.bar(x, alphas_fixed_046, width, label='α at x_min=0.46', color='coral', alpha=0.8)
    ax.bar(x + width, alphas_fixed_024, width, label='α at x_min=0.024', color='lightgreen', alpha=0.8)

    # Add error bars for tolerance range
    errors_low = np.array(alphas_opt) - np.array(alphas_low)
    errors_high = np.array(alphas_high) - np.array(alphas_opt)
    ax.errorbar(x - width, alphas_opt, yerr=[errors_low, errors_high],
                fmt='none', capsize=4, capthick=2, color='black', zorder=10)

    # Reference lines
    ax.axhline(2.14, color='red', linestyle='--', linewidth=2, label='Cael & Seekell (2.14)')
    ax.axhline(2.05, color='orange', linestyle=':', linewidth=2, label='Percolation (2.05)')

    ax.set_xlabel('Elevation Band', fontsize=12)
    ax.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax.set_title('α Stability: How Sensitive is α to x_min Choice?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation
    ax.annotate('Error bars show α range within\nKS tolerance of optimal',
               xy=(0.02, 0.98), xycoords='axes fraction',
               ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_alpha_vs_xmin_by_elevation(xmin_results, figsize=(16, 12), save_path=None):
    """
    Multi-panel visualization of alpha sensitivity to x_min by elevation band.

    Creates a grid of panels, one per elevation band, showing:
    - Alpha vs x_min curve with confidence intervals
    - Reference lines for percolation theory and Cael & Seekell
    - Optimal x_min marked

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation() containing:
        - 'by_elevation': dict of DataFrames per elevation band
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    by_elevation = xmin_results.get('by_elevation', {})
    n_bands = len(by_elevation)

    if n_bands == 0:
        print("Warning: No elevation band results to plot")
        return None, None

    # Determine grid layout
    n_cols = min(3, n_bands)
    n_rows = int(np.ceil(n_bands / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_bands == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color palette
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_bands - 1)) for i in range(n_bands)]

    for idx, (band_name, band_data) in enumerate(by_elevation.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        if 'sensitivity' not in band_data or band_data['sensitivity'] is None:
            ax.text(0.5, 0.5, f'{band_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sens = band_data['sensitivity']
        xmins = sens['xmin'].values
        alphas = sens['alpha'].values

        # Plot alpha vs x_min curve
        ax.plot(xmins, alphas, 'o-', linewidth=2, markersize=4,
                color=colors[idx], label='α vs x_min')

        # Add confidence interval if available
        if 'alpha_se' in sens.columns:
            se = sens['alpha_se'].values
            valid_se = np.isfinite(se) & np.isfinite(alphas)
            if valid_se.any():
                ax.fill_between(xmins[valid_se],
                               (alphas - 1.96*se)[valid_se],
                               (alphas + 1.96*se)[valid_se],
                               alpha=0.2, color=colors[idx])

        # Mark optimal x_min
        optimal = band_data.get('optimal', {})
        if optimal and 'xmin' in optimal:
            opt_xmin = optimal['xmin']
            opt_alpha = optimal.get('alpha', np.nan)
            ax.scatter([opt_xmin], [opt_alpha], s=150, c='red', marker='*', zorder=10,
                      label=f'Optimal: α={opt_alpha:.2f}')

        # Reference lines
        ax.axhline(2.14, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='Cael & Seekell (2.14)')
        ax.axhline(2.05, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                  label='Percolation (2.05)')
        ax.axvline(0.46, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Labels and formatting
        ax.set_xlabel('x_min (km²)', fontsize=10)
        ax.set_ylabel('α', fontsize=10)
        ax.set_title(f'{band_name}', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Add sample size annotation
        n_total = band_data.get('n_total', 0)
        ax.annotate(f'n={n_total:,}',
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   ha='left', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)

    # Hide unused axes
    for idx in range(n_bands, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('α Sensitivity to x_min by Elevation Band', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_xmin_elevation_summary(xmin_results, figsize=(16, 12), save_path=None):
    """
    Comprehensive summary figure for x_min sensitivity by elevation analysis.

    Creates a 6-panel figure answering key questions:
    1. Does optimal x_min differ by elevation?
    2. How sensitive is α to x_min choice?
    3. Are conclusions robust to threshold choice?

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    by_elevation = xmin_results.get('by_elevation', {})
    comparison = xmin_results.get('method_comparison', {})
    robustness = xmin_results.get('robustness', {})

    # Extract data
    bands = list(by_elevation.keys())
    n_bands = len(bands)

    elevations = []
    opt_xmins = []
    opt_alphas = []
    fixed_alphas_046 = []
    n_totals = []

    for band_name, band_data in by_elevation.items():
        try:
            parts = band_name.replace('m', '').split('-')
            elev_mid = (float(parts[0]) + float(parts[1])) / 2
        except:
            elev_mid = 0

        elevations.append(elev_mid)

        optimal = band_data.get('optimal', {})
        opt_xmins.append(optimal.get('xmin', np.nan))
        opt_alphas.append(optimal.get('alpha', np.nan))
        n_totals.append(band_data.get('n_total', 0))

        fixed = band_data.get('fixed_xmin', {})
        fixed_alphas_046.append(fixed.get(0.46, {}).get('alpha', np.nan))

    elevations = np.array(elevations)
    opt_xmins = np.array(opt_xmins)
    opt_alphas = np.array(opt_alphas)

    # Panel A: Optimal x_min vs elevation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(elevations, opt_xmins, 'o-', linewidth=2, markersize=10, color='darkblue')
    ax1.axhline(0.46, color='red', linestyle='--', linewidth=2, alpha=0.7, label='C&S (0.46)')
    ax1.axhline(0.024, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='NHD (0.024)')
    ax1.set_xlabel('Elevation (m)', fontsize=11)
    ax1.set_ylabel('Optimal x_min (km²)', fontsize=11)
    ax1.set_title('A) Does optimal x_min vary?', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: α at optimal vs fixed x_min
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(n_bands)
    width = 0.35
    ax2.bar(x - width/2, opt_alphas, width, label='α at optimal x_min', color='steelblue')
    ax2.bar(x + width/2, fixed_alphas_046, width, label='α at x_min=0.46', color='coral')
    ax2.axhline(2.14, color='red', linestyle='--', linewidth=2)
    ax2.axhline(2.05, color='orange', linestyle=':', linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('α', fontsize=11)
    ax2.set_title('B) How does α change with x_min method?', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: KS curves overlay
    ax3 = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_bands - 1)) for i in range(n_bands)]

    for idx, (band_name, band_data) in enumerate(by_elevation.items()):
        if 'sensitivity' not in band_data or band_data['sensitivity'] is None:
            continue
        sens = band_data['sensitivity']
        ax3.plot(sens['xmin'].values, sens['ks_statistic'].values,
                '-', linewidth=2, color=colors[idx], label=band_name, alpha=0.8)

    ax3.axvline(0.46, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('x_min (km²)', fontsize=11)
    ax3.set_ylabel('KS Statistic', fontsize=11)
    ax3.set_title('C) KS curves by elevation', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panel D: α vs x_min curves overlay
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, (band_name, band_data) in enumerate(by_elevation.items()):
        if 'sensitivity' not in band_data or band_data['sensitivity'] is None:
            continue
        sens = band_data['sensitivity']
        ax4.plot(sens['xmin'].values, sens['alpha'].values,
                '-', linewidth=2, color=colors[idx], label=band_name, alpha=0.8)

    ax4.axhline(2.14, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.axhline(2.05, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax4.axvline(0.46, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('x_min (km²)', fontsize=11)
    ax4.set_ylabel('α', fontsize=11)
    ax4.set_title('D) α sensitivity to x_min', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    # Panel E: Sample size by elevation
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar(elevations, n_totals, width=200, alpha=0.7, color='teal', edgecolor='darkcyan')
    ax5.set_xlabel('Elevation (m)', fontsize=11)
    ax5.set_ylabel('Total Lakes', fontsize=11)
    ax5.set_title('E) Sample size by elevation', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add total count
    total = sum(n_totals)
    ax5.annotate(f'Total: {total:,}', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel F: Robustness summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Create summary text
    summary_lines = [
        "Key Findings:",
        "",
    ]

    if robustness:
        robust = robustness.get('robust_bands', [])
        sensitive = robustness.get('sensitive_bands', [])
        summary_lines.extend([
            f"• Robust bands: {len(robust)}",
            f"• Sensitive bands: {len(sensitive)}",
        ])

    if comparison:
        agreement = comparison.get('methods_agree', 'Unknown')
        summary_lines.append(f"• Methods agree: {agreement}")

    # Add interpretation
    mean_opt_alpha = np.nanmean(opt_alphas)
    mean_fixed_alpha = np.nanmean(fixed_alphas_046)
    diff = abs(mean_opt_alpha - mean_fixed_alpha)

    summary_lines.extend([
        "",
        f"Mean α (optimal x_min): {mean_opt_alpha:.3f}",
        f"Mean α (fixed 0.46): {mean_fixed_alpha:.3f}",
        f"Difference: {diff:.3f}",
        "",
        "Conclusion:" if diff < 0.1 else "Warning:",
        "Results are ROBUST to x_min" if diff < 0.1 else "Results SENSITIVE to x_min"
    ])

    summary_text = '\n'.join(summary_lines)
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    ax6.set_title('F) Robustness Summary', fontsize=12, fontweight='bold')

    fig.suptitle('x_min Sensitivity Analysis by Elevation', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax1, ax2, ax3, ax4, ax5, ax6)


# ============================================================================
# HYPOTHESIS TEST VISUALIZATIONS
# ============================================================================

def plot_hypothesis_test_summary(xmin_results, hypothesis_results, figsize=(16, 14), save_path=None):
    """
    Create 4-panel visualization of hypothesis test results, similar to
    x_min sensitivity figures.

    Panels:
    A) α sensitivity to x_min by elevation (with 95% CI, reference lines)
    B) Sample size vs x_min by elevation (log scale)
    C) KS statistic vs x_min (goodness of fit)
    D) Key threshold comparison table (colored)

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    hypothesis_results : dict
        Output from run_all_hypothesis_tests()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Get elevation colors
    elevation_bands = [k for k in xmin_results.keys() if 'method_comparison' not in k
                       and 'robustness' not in k and 'summary' not in k
                       and 'hypothesis' not in k]
    n_bands = len(elevation_bands)
    cmap = plt.cm.viridis
    colors = {band: cmap(i / max(1, n_bands - 1)) for i, band in enumerate(elevation_bands)}

    # Panel A: α sensitivity to x_min
    ax1 = fig.add_subplot(gs[0, 0])
    for band_name in elevation_bands:
        band_data = xmin_results.get(band_name, {})
        if band_data.get('status') != 'success':
            continue
        full_results = band_data.get('full_results')
        if full_results is None or len(full_results) == 0:
            continue

        xmin_vals = full_results['xmin'].values
        alpha_vals = full_results['alpha'].values

        # Plot with confidence interval
        ax1.plot(xmin_vals, alpha_vals, '-o', linewidth=2, markersize=4,
                color=colors[band_name], label=band_name, alpha=0.8)

        # Add CI if available
        if 'alpha_se' in full_results.columns:
            se = full_results['alpha_se'].values
            valid = ~np.isnan(alpha_vals) & ~np.isnan(se)
            ax1.fill_between(xmin_vals[valid],
                           alpha_vals[valid] - 1.96*se[valid],
                           alpha_vals[valid] + 1.96*se[valid],
                           alpha=0.15, color=colors[band_name])

    # Reference lines
    ax1.axhline(2.14, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Cael & Seekell (τ=2.14)')
    ax1.axhline(2.05, color='green', linestyle=':', linewidth=2, alpha=0.8, label='Percolation (τ=2.05)')
    ax1.axvline(0.46, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='C&S threshold (0.46 km²)')

    ax1.set_xlabel('x_min (km²)', fontsize=12)
    ax1.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax1.set_title('A) α Sensitivity to Minimum Area', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='lower right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel B: Sample size vs x_min
    ax2 = fig.add_subplot(gs[0, 1])
    for band_name in elevation_bands:
        band_data = xmin_results.get(band_name, {})
        if band_data.get('status') != 'success':
            continue
        full_results = band_data.get('full_results')
        if full_results is None or len(full_results) == 0:
            continue

        xmin_vals = full_results['xmin'].values
        n_tail = full_results['n_tail'].values

        ax2.plot(xmin_vals, n_tail, '-s', linewidth=2, markersize=4,
                color=colors[band_name], label=band_name, alpha=0.8)

    ax2.axvline(0.46, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('x_min (km²)', fontsize=12)
    ax2.set_ylabel('Number of Lakes in Tail', fontsize=12)
    ax2.set_title('B) Sample Size vs Minimum Area', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Panel C: KS statistic vs x_min
    ax3 = fig.add_subplot(gs[1, 0])
    for band_name in elevation_bands:
        band_data = xmin_results.get(band_name, {})
        if band_data.get('status') != 'success':
            continue
        full_results = band_data.get('full_results')
        if full_results is None or len(full_results) == 0:
            continue

        xmin_vals = full_results['xmin'].values
        ks_vals = full_results['ks_statistic'].values

        ax3.plot(xmin_vals, ks_vals, '-d', linewidth=2, markersize=4,
                color=colors[band_name], label=band_name, alpha=0.8)

        # Mark optimal x_min
        optimal_xmin = band_data.get('optimal_xmin')
        optimal_ks = band_data.get('optimal_ks')
        if optimal_xmin is not None and optimal_ks is not None:
            ax3.scatter([optimal_xmin], [optimal_ks], marker='*', s=150,
                       color=colors[band_name], edgecolors='black', zorder=5)

    ax3.axvline(0.46, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('x_min (km²)', fontsize=12)
    ax3.set_ylabel('KS Statistic', fontsize=12)
    ax3.set_title('C) Goodness of Fit vs Minimum Area', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.annotate('★ = optimal x_min', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=9, ha='left', va='top')

    # Panel D: Colored table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Build table data
    table_data = []
    cell_colors = []
    headers = ['Elevation\nBand', 'x_min\n(km²)', 'α', 'n (tail)', 'KS\nstat', 'SE']

    for band_name in elevation_bands:
        band_data = xmin_results.get(band_name, {})
        if band_data.get('status') != 'success':
            continue

        opt_xmin = band_data.get('optimal_xmin', np.nan)
        opt_alpha = band_data.get('optimal_alpha', np.nan)
        opt_ks = band_data.get('optimal_ks', np.nan)

        # Get n_tail from full_results at optimal x_min
        full_results = band_data.get('full_results')
        if full_results is not None and len(full_results) > 0:
            opt_idx = (full_results['xmin'] - opt_xmin).abs().idxmin() if pd.notna(opt_xmin) else 0
            n_tail = full_results.loc[opt_idx, 'n_tail'] if opt_idx in full_results.index else 0
        else:
            n_tail = 0

        # Compute SE
        se = (opt_alpha - 1) / np.sqrt(n_tail) if n_tail > 0 and pd.notna(opt_alpha) else np.nan

        row = [
            band_name,
            f'{opt_xmin:.3f}' if pd.notna(opt_xmin) else 'N/A',
            f'{opt_alpha:.3f}' if pd.notna(opt_alpha) else 'N/A',
            f'{n_tail:,}' if n_tail > 0 else 'N/A',
            f'{opt_ks:.4f}' if pd.notna(opt_ks) else 'N/A',
            f'{se:.4f}' if pd.notna(se) else 'N/A',
        ]
        table_data.append(row)

        # Color based on alpha value
        if pd.notna(opt_alpha):
            if opt_alpha < 1.9:
                alpha_color = '#FFB3B3'  # Light red - below percolation
            elif opt_alpha < 2.05:
                alpha_color = '#FFFFB3'  # Light yellow - near percolation
            elif opt_alpha < 2.14:
                alpha_color = '#B3FFB3'  # Light green - near C&S
            else:
                alpha_color = '#B3D9FF'  # Light blue - above C&S
        else:
            alpha_color = '#FFFFFF'

        row_colors = ['#FFFFFF', '#FFFFFF', alpha_color, '#FFFFFF', '#FFFFFF', '#FFFFFF']
        cell_colors.append(row_colors)

    if table_data:
        table = ax4.table(cellText=table_data,
                         colLabels=headers,
                         cellColours=cell_colors,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('D) Key Threshold Comparison', fontsize=14, fontweight='bold', pad=20)

    # Add legend for colors
    legend_text = ('α colors: Red < 1.9 | Yellow 1.9-2.05 | '
                   'Green 2.05-2.14 | Blue > 2.14')
    ax4.text(0.5, -0.05, legend_text, transform=ax4.transAxes,
            fontsize=9, ha='center', va='top', style='italic')

    fig.suptitle('x_min Sensitivity Analysis by Elevation Band', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax1, ax2, ax3, ax4)


def plot_hypothesis_test_results(hypothesis_results, figsize=(14, 10), save_path=None):
    """
    Create visual summary of all four hypothesis tests with colored indicators.

    Parameters
    ----------
    hypothesis_results : dict
        Output from run_all_hypothesis_tests()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    setup_plot_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Test 1: x_min variation by elevation
    ax1 = fig.add_subplot(gs[0, 0])
    test1 = hypothesis_results.get('xmin_variation', {})

    xmin_by_band = test1.get('xmin_by_band', {})
    if xmin_by_band:
        bands = list(xmin_by_band.keys())
        xmins = list(xmin_by_band.values())

        # Create bar plot
        y_pos = np.arange(len(bands))
        bars = ax1.barh(y_pos, xmins, alpha=0.7, color='steelblue', edgecolor='darkblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(bands, fontsize=9)
        ax1.set_xlabel('Optimal x_min (km²)', fontsize=11)

        # Add mean line
        mean_xmin = test1.get('xmin_mean', np.nan)
        if pd.notna(mean_xmin):
            ax1.axvline(mean_xmin, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_xmin:.3f}')

        # Add CV annotation
        cv = test1.get('xmin_cv', np.nan)
        if pd.notna(cv):
            color = 'green' if cv < 0.3 else ('orange' if cv < 0.5 else 'red')
            ax1.annotate(f'CV = {cv:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                        ha='right', va='top', fontsize=11, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='x')

    ax1.set_title('Test 1: Does x_min Vary by Elevation?', fontsize=12, fontweight='bold')

    # Test 2: KS curve behavior
    ax2 = fig.add_subplot(gs[0, 1])
    test2 = hypothesis_results.get('ks_behavior', {})

    if test2:
        bands = []
        flatness_scores = []
        shapes = []
        shape_colors = {'flat': 'green', 'moderate': 'orange', 'sharp': 'red', 'unknown': 'gray'}

        for band_name, band_data in test2.items():
            if band_data.get('ks_curve_shape') != 'unknown':
                bands.append(band_name)
                flatness_scores.append(band_data.get('flatness_score', np.nan))
                shapes.append(band_data.get('ks_curve_shape', 'unknown'))

        if bands:
            y_pos = np.arange(len(bands))
            bar_colors = [shape_colors.get(s, 'gray') for s in shapes]
            bars = ax2.barh(y_pos, flatness_scores, alpha=0.7, color=bar_colors, edgecolor='black')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(bands, fontsize=9)
            ax2.set_xlabel('Flatness Score (lower = more robust)', fontsize=11)

            # Add thresholds
            ax2.axvline(0.1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Flat')
            ax2.axvline(0.25, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Sharp')

            ax2.legend(loc='lower right', fontsize=9)
            ax2.grid(True, alpha=0.3, axis='x')

    ax2.set_title('Test 2: KS Curve Shape by Band', fontsize=12, fontweight='bold')

    # Test 3: Alpha trajectory/convergence
    ax3 = fig.add_subplot(gs[1, 0])
    test3 = hypothesis_results.get('alpha_trajectory', {})

    alpha_low = test3.get('alpha_at_low_xmin', {})
    alpha_high = test3.get('alpha_at_high_xmin', {})

    if alpha_low and alpha_high:
        common_bands = [b for b in alpha_low if b in alpha_high]
        if common_bands:
            x = np.arange(len(common_bands))
            width = 0.35

            low_vals = [alpha_low[b] for b in common_bands]
            high_vals = [alpha_high[b] for b in common_bands]

            bars1 = ax3.bar(x - width/2, low_vals, width, label='Low x_min (≤0.2)', alpha=0.7, color='steelblue')
            bars2 = ax3.bar(x + width/2, high_vals, width, label='High x_min (≥2.0)', alpha=0.7, color='coral')

            ax3.set_xticks(x)
            ax3.set_xticklabels(common_bands, fontsize=9, rotation=45, ha='right')
            ax3.set_ylabel('α', fontsize=11)
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')

            # Add reference lines
            ax3.axhline(2.14, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            ax3.axhline(2.05, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

            # Add convergence annotation
            conv_ratio = test3.get('convergence_ratio', np.nan)
            if pd.notna(conv_ratio):
                if conv_ratio < 0.5:
                    verdict = 'CONVERGE'
                    color = 'orange'
                elif conv_ratio > 1.5:
                    verdict = 'DIVERGE'
                    color = 'green'
                else:
                    verdict = 'STABLE'
                    color = 'blue'
                ax3.annotate(f'Bands {verdict}\n(ratio={conv_ratio:.2f})',
                            xy=(0.95, 0.05), xycoords='axes fraction',
                            ha='right', va='bottom', fontsize=10, fontweight='bold',
                            color=color,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_title('Test 3: α Trajectory (Low vs High x_min)', fontsize=12, fontweight='bold')

    # Test 4: n_tail stability
    ax4 = fig.add_subplot(gs[1, 1])
    test4 = hypothesis_results.get('ntail_stability', {})

    if test4:
        bands = []
        n_tails = []
        ses = []
        stable_flags = []

        for band_name, band_data in test4.items():
            n_tail = band_data.get('n_tail_at_optimal', 0)
            if n_tail > 0:
                bands.append(band_name)
                n_tails.append(n_tail)
                ses.append(band_data.get('se_at_optimal', np.nan))
                stable_flags.append(band_data.get('stable', False))

        if bands:
            y_pos = np.arange(len(bands))
            bar_colors = ['green' if s else 'red' for s in stable_flags]
            bars = ax4.barh(y_pos, n_tails, alpha=0.7, color=bar_colors, edgecolor='black')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(bands, fontsize=9)
            ax4.set_xlabel('n (tail) at optimal x_min', fontsize=11)

            # Add SE annotations
            for i, (n, se) in enumerate(zip(n_tails, ses)):
                if pd.notna(se):
                    ax4.annotate(f'SE={se:.3f}', xy=(n, i), xytext=(5, 0),
                                textcoords='offset points', fontsize=8, va='center')

            ax4.grid(True, alpha=0.3, axis='x')

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', alpha=0.7, label='Stable'),
                              Patch(facecolor='red', alpha=0.7, label='Unstable')]
            ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax4.set_title('Test 4: Sample Size Stability', fontsize=12, fontweight='bold')

    fig.suptitle('Hypothesis Test Results Summary', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax1, ax2, ax3, ax4)


def plot_colored_summary_table(xmin_results, hypothesis_results=None,
                                figsize=(12, 8), save_path=None):
    """
    Create a publication-ready colored table visualization of x_min results.

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_elevation()
    hypothesis_results : dict, optional
        Output from run_all_hypothesis_tests() for additional annotations
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Build table data
    elevation_bands = [k for k in xmin_results.keys() if 'method_comparison' not in k
                       and 'robustness' not in k and 'summary' not in k
                       and 'hypothesis' not in k]

    headers = ['Elevation\nBand', 'n\n(total)', 'Optimal\nx_min', 'α\n(optimal)',
               'KS\nstat', 'SE', 'x_min\nRange', 'α\nStability']

    table_data = []
    cell_colors = []

    for band_name in elevation_bands:
        band_data = xmin_results.get(band_name, {})
        if band_data.get('status') != 'success':
            row = [band_name, '-', '-', '-', '-', '-', '-', '-']
            table_data.append(row)
            cell_colors.append(['#E0E0E0'] * 8)
            continue

        n_total = band_data.get('n_total', 0)
        opt_xmin = band_data.get('optimal_xmin', np.nan)
        opt_alpha = band_data.get('optimal_alpha', np.nan)
        opt_ks = band_data.get('optimal_ks', np.nan)
        acc_range = band_data.get('acceptable_range', (np.nan, np.nan))
        alpha_stability = band_data.get('alpha_stability', np.nan)

        # Get n_tail for SE calculation
        full_results = band_data.get('full_results')
        if full_results is not None and len(full_results) > 0 and pd.notna(opt_xmin):
            opt_idx = (full_results['xmin'] - opt_xmin).abs().idxmin()
            n_tail = full_results.loc[opt_idx, 'n_tail'] if opt_idx in full_results.index else 0
        else:
            n_tail = 0

        se = (opt_alpha - 1) / np.sqrt(n_tail) if n_tail > 0 and pd.notna(opt_alpha) else np.nan

        row = [
            band_name,
            f'{n_total:,}',
            f'{opt_xmin:.3f}' if pd.notna(opt_xmin) else '-',
            f'{opt_alpha:.3f}' if pd.notna(opt_alpha) else '-',
            f'{opt_ks:.4f}' if pd.notna(opt_ks) else '-',
            f'{se:.4f}' if pd.notna(se) else '-',
            f'[{acc_range[0]:.2f}, {acc_range[1]:.2f}]' if pd.notna(acc_range[0]) else '-',
            f'±{alpha_stability/2:.3f}' if pd.notna(alpha_stability) else '-',
        ]
        table_data.append(row)

        # Color coding
        row_colors = ['#FFFFFF'] * 8

        # Color α based on value
        if pd.notna(opt_alpha):
            if opt_alpha < 1.9:
                row_colors[3] = '#FFB3B3'  # Light red
            elif opt_alpha < 2.05:
                row_colors[3] = '#FFFFB3'  # Light yellow
            elif opt_alpha < 2.14:
                row_colors[3] = '#B3FFB3'  # Light green
            else:
                row_colors[3] = '#B3D9FF'  # Light blue

        # Color KS based on quality
        if pd.notna(opt_ks):
            if opt_ks < 0.05:
                row_colors[4] = '#B3FFB3'  # Good fit
            elif opt_ks < 0.1:
                row_colors[4] = '#FFFFB3'  # OK fit
            else:
                row_colors[4] = '#FFB3B3'  # Poor fit

        # Color SE based on precision
        if pd.notna(se):
            if se < 0.03:
                row_colors[5] = '#B3FFB3'  # Very precise
            elif se < 0.05:
                row_colors[5] = '#FFFFB3'  # Adequate
            else:
                row_colors[5] = '#FFB3B3'  # Imprecise

        # Color stability
        if pd.notna(alpha_stability):
            if alpha_stability < 0.05:
                row_colors[7] = '#B3FFB3'  # Very stable
            elif alpha_stability < 0.1:
                row_colors[7] = '#FFFFB3'  # Moderately stable
            else:
                row_colors[7] = '#FFB3B3'  # Unstable

        cell_colors.append(row_colors)

    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellColours=cell_colors,
                        loc='center',
                        cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2.0)

        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=10)

    ax.set_title('x_min Sensitivity Analysis Summary by Elevation\n',
                fontsize=16, fontweight='bold', pad=20)

    # Add color legend
    legend_text = ('Color coding:\n'
                   'α: Red (<1.9) | Yellow (1.9-2.05) | Green (2.05-2.14) | Blue (>2.14)\n'
                   'KS: Green (<0.05) | Yellow (0.05-0.1) | Red (>0.1)\n'
                   'SE: Green (<0.03) | Yellow (0.03-0.05) | Red (>0.05)\n'
                   'Stability: Green (<0.05) | Yellow (0.05-0.1) | Red (>0.1)')
    ax.text(0.5, -0.02, legend_text, transform=ax.transAxes,
           fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


# ============================================================================
# GLACIAL CHRONOSEQUENCE VISUALIZATIONS
# ============================================================================

def plot_density_by_glacial_stage(density_df, figsize=(12, 6), save_path=None):
    """
    Plot lake density by glacial stage as a bar chart.

    Tests Davis's hypothesis that lake density decreases with landscape age.

    Parameters
    ----------
    density_df : DataFrame
        Output from compute_lake_density_by_glacial_stage()
        Columns: glacial_stage, n_lakes, density_per_1000km2, age_ka, color
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by age (youngest first, handle None age for Driftless)
    df = density_df.copy()
    df['sort_age'] = df['age_ka'].fillna(999999)
    df = df.sort_values('sort_age')

    # Extract data
    stages = df['glacial_stage'].values
    densities = df['density_per_1000km2'].values
    colors = df['color'].values if 'color' in df.columns else ['#1f77b4'] * len(df)
    n_lakes = df['n_lakes'].values

    # Create bar chart
    bars = ax.bar(range(len(stages)), densities, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, n, d) in enumerate(zip(bars, n_lakes, densities)):
        if pd.notna(d):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{d:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    f'n={n:,}',
                    ha='center', va='center', fontsize=9, color='white')

    # Labels and formatting
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=0, ha='center', fontsize=12)
    ax.set_ylabel('Lake Density (lakes per 1,000 km²)', fontsize=14)
    ax.set_title("Lake Density by Glacial Stage\nTesting Davis's Hypothesis", fontsize=16, fontweight='bold')

    # Add age labels below stage names (positioned higher to avoid overlap)
    for i, row in enumerate(df.itertuples()):
        if pd.notna(row.age_ka):
            age_str = f'{row.age_ka:.0f} ka' if row.age_ka < 1000 else f'{row.age_ka/1000:.0f} Ma'
        else:
            age_str = 'Never glaciated'
        ax.text(i, -0.08, age_str, ha='center', va='top', fontsize=9,
                transform=ax.get_xaxis_transform(), style='italic', color='gray')

    # Add xlabel with extra padding to avoid overlap with age labels
    ax.set_xlabel('Glacial Stage (youngest → oldest)', fontsize=14, labelpad=25)

    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

    # Adjust subplot to leave room for age labels
    plt.subplots_adjust(bottom=0.18)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_elevation_histogram_by_glacial_stage(elevation_df, figsize=(14, 8), save_path=None):
    """
    Plot elevation histograms showing lake counts with glacial stage overlays.

    Creates stacked bar chart showing how lakes at each elevation are
    distributed across different glacial stages.

    Parameters
    ----------
    elevation_df : DataFrame
        Output from compute_elevation_binned_density_by_stage()
        Columns: elev_bin_mid, glacial_stage, n_lakes, pct_of_stage
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get unique stages and their colors
    try:
        from .config import GLACIAL_CHRONOLOGY
    except ImportError:
        from config import GLACIAL_CHRONOLOGY

    stages = elevation_df['glacial_stage'].unique()
    stage_colors = {}
    for stage in stages:
        stage_lower = stage.lower()
        for key, chrono in GLACIAL_CHRONOLOGY.items():
            if key in stage_lower or chrono['name'].lower() in stage_lower:
                stage_colors[stage] = chrono['color']
                break
        if stage not in stage_colors:
            stage_colors[stage] = '#808080'

    # Get elevation bins
    elev_bins = sorted(elevation_df['elev_bin_mid'].dropna().unique())
    bar_width = elev_bins[1] - elev_bins[0] if len(elev_bins) > 1 else 100

    # Left panel: Stacked bar chart (absolute counts)
    ax1 = axes[0]
    bottom = np.zeros(len(elev_bins))

    for stage in stages:
        stage_data = elevation_df[elevation_df['glacial_stage'] == stage]
        counts = []
        for elev in elev_bins:
            match = stage_data[stage_data['elev_bin_mid'] == elev]
            counts.append(match['n_lakes'].values[0] if len(match) > 0 else 0)

        ax1.bar(elev_bins, counts, width=bar_width*0.9, bottom=bottom,
                label=stage, color=stage_colors[stage], edgecolor='white', linewidth=0.5)
        bottom += np.array(counts)

    ax1.set_xlabel('Elevation (m)', fontsize=14)
    ax1.set_ylabel('Lake Count', fontsize=14)
    ax1.set_title('A) Lake Count by Elevation\n(Stacked by Glacial Stage)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Overlaid density curves (normalized within each stage)
    ax2 = axes[1]

    for stage in stages:
        stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
        if len(stage_data) > 0:
            ax2.plot(stage_data['elev_bin_mid'], stage_data['pct_of_stage'],
                     'o-', linewidth=2, markersize=4, label=stage, color=stage_colors[stage])

    ax2.set_xlabel('Elevation (m)', fontsize=14)
    ax2.set_ylabel('Percent of Stage Total (%)', fontsize=14)
    ax2.set_title('B) Elevation Distribution Within Each Stage\n(Normalized)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_elevation_glacial_multipanel(elevation_df, hypsometry_df=None, dalton_df=None,
                                       normalized_density_df=None, zone_areas=None,
                                       figsize=(10, 18), save_path=None):
    """
    Create single-column multi-panel figure for elevation analysis by glacial stage.

    Creates a vertical 3-panel figure:
    A) Lake count by elevation (stacked bar chart by glacial stage)
    B) Percent of stage total by elevation (normalized within each stage)
    C) Area-normalized lake density by elevation (lakes per km² of landscape)
        with Dalton 18ka data shown as a separate color

    Parameters
    ----------
    elevation_df : DataFrame
        Output from compute_elevation_binned_density_by_stage()
        Required columns: elev_bin_mid, glacial_stage, n_lakes, pct_of_stage
    hypsometry_df : DataFrame, optional
        Per-stage hypsometry (area by elevation for each stage)
        Required columns: elev_bin_mid, glacial_stage, area_km2
    dalton_df : DataFrame, optional
        Dalton 18ka data with same structure as elevation_df for separate plotting
    normalized_density_df : DataFrame, optional
        Pre-computed normalized density with columns:
        glacial_stage, elev_bin_mid, density_per_1000km2
    zone_areas : dict, optional
        Glacial zone areas for computing per-stage normalization
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Get glacial stage colors from config
    try:
        from .config import GLACIAL_CHRONOLOGY
    except ImportError:
        from config import GLACIAL_CHRONOLOGY

    stages = elevation_df['glacial_stage'].unique()
    stage_colors = {}
    for stage in stages:
        stage_lower = stage.lower()
        for key, chrono in GLACIAL_CHRONOLOGY.items():
            if key in stage_lower or chrono['name'].lower() in stage_lower:
                stage_colors[stage] = chrono['color']
                break
        if stage not in stage_colors:
            stage_colors[stage] = '#808080'

    # Get elevation bins
    elev_bins = sorted(elevation_df['elev_bin_mid'].dropna().unique())
    if len(elev_bins) < 2:
        for ax in axes:
            ax.text(0.5, 0.5, 'Insufficient elevation data',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig, axes

    bar_width = elev_bins[1] - elev_bins[0] if len(elev_bins) > 1 else 100

    # ====== Panel A: Lake Count by Elevation (stacked by glacial stage) ======
    ax1 = axes[0]
    bottom = np.zeros(len(elev_bins))

    for stage in stages:
        stage_data = elevation_df[elevation_df['glacial_stage'] == stage]
        counts = []
        for elev in elev_bins:
            match = stage_data[stage_data['elev_bin_mid'] == elev]
            counts.append(match['n_lakes'].values[0] if len(match) > 0 else 0)

        ax1.bar(elev_bins, counts, width=bar_width*0.9, bottom=bottom,
                label=stage, color=stage_colors[stage], edgecolor='white', linewidth=0.5)
        bottom += np.array(counts)

    ax1.set_ylabel('Lake Count', fontsize=12)
    ax1.set_title('A) Lake Count by Elevation (Stacked by Glacial Stage)',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ====== Panel B: Percent of Stage Total by Elevation ======
    ax2 = axes[1]

    for stage in stages:
        stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
        if len(stage_data) > 0:
            ax2.plot(stage_data['elev_bin_mid'], stage_data['pct_of_stage'],
                     'o-', linewidth=2, markersize=4, label=stage, color=stage_colors[stage])

    ax2.set_ylabel('Percent of Stage Total (%)', fontsize=12)
    ax2.set_title('B) Elevation Distribution Within Each Stage (Normalized)',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ====== Panel C: Area-Normalized Lake Density by Elevation ======
    ax3 = axes[2]

    # Priority 1: Use pre-computed normalized density if provided
    if normalized_density_df is not None and 'density_per_1000km2' in normalized_density_df.columns:
        for stage in stages:
            stage_data = normalized_density_df[
                normalized_density_df['glacial_stage'] == stage
            ].sort_values('elev_bin_mid')
            if len(stage_data) > 0:
                ax3.plot(stage_data['elev_bin_mid'], stage_data['density_per_1000km2'],
                        'o-', linewidth=2, markersize=4, label=stage, color=stage_colors[stage])

    # Priority 2: Compute normalization using zone_areas (per-stage total area)
    elif zone_areas is not None:
        for stage in stages:
            stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
            if len(stage_data) > 0:
                stage_area = zone_areas.get(stage.lower(), 0)
                if stage_area > 0:
                    # Total lakes in this stage
                    total_lakes_in_stage = stage_data['n_lakes'].sum()
                    # For each elevation bin, compute density as:
                    # (n_lakes / total_lakes_in_stage) * (total_lakes_in_stage / stage_area) * 1000
                    # = n_lakes / stage_area * 1000
                    # But we want to show the elevation distribution, so we use:
                    # density = n_lakes / (stage_area * pct_of_stage/100) * 1000
                    # This normalizes by the estimated area at each elevation
                    densities = []
                    elevs = []
                    for _, row in stage_data.iterrows():
                        n_lakes = row['n_lakes']
                        pct = row.get('pct_of_stage', 100 * n_lakes / total_lakes_in_stage)
                        if pct > 0:
                            # Estimated area at this elevation for this stage
                            est_area = stage_area * pct / 100
                            density = n_lakes / est_area * 1000
                            densities.append(density)
                            elevs.append(row['elev_bin_mid'])

                    if len(densities) > 0:
                        ax3.plot(elevs, densities, 'o-', linewidth=2, markersize=4,
                                label=stage, color=stage_colors[stage])

    # Priority 3: Use per-stage hypsometry if provided
    elif hypsometry_df is not None and 'area_km2' in hypsometry_df.columns:
        # Check if hypsometry is per-stage or overall
        if 'glacial_stage' in hypsometry_df.columns:
            # Per-stage hypsometry
            for stage in stages:
                stage_elev = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
                stage_hyps = hypsometry_df[hypsometry_df['glacial_stage'] == stage]
                if len(stage_elev) > 0 and len(stage_hyps) > 0:
                    merged = stage_elev.merge(
                        stage_hyps[['elev_bin_mid', 'area_km2']],
                        on='elev_bin_mid', how='inner'
                    )
                    merged['density'] = np.where(
                        merged['area_km2'] > 0,
                        merged['n_lakes'] / merged['area_km2'] * 1000,
                        0
                    )
                    if len(merged) > 0:
                        ax3.plot(merged['elev_bin_mid'], merged['density'],
                                'o-', linewidth=2, markersize=4, label=stage, color=stage_colors[stage])
        else:
            # Overall hypsometry (old behavior)
            for stage in stages:
                stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
                if len(stage_data) > 0:
                    densities = []
                    elevs = []
                    for _, row in stage_data.iterrows():
                        elev_mid = row['elev_bin_mid']
                        n_lakes = row['n_lakes']
                        hyps_match = hypsometry_df[hypsometry_df['elev_bin_mid'] == elev_mid]
                        if len(hyps_match) > 0 and hyps_match['area_km2'].values[0] > 0:
                            density = n_lakes / hyps_match['area_km2'].values[0] * 1000
                            densities.append(density)
                            elevs.append(elev_mid)

                    if len(densities) > 0:
                        ax3.plot(elevs, densities, 'o-', linewidth=2, markersize=4,
                                label=stage, color=stage_colors[stage])
    else:
        # Fallback: Just show raw counts (but note this isn't normalized)
        ax3.text(0.5, 0.95, '(Raw counts - not area-normalized)', transform=ax3.transAxes,
                fontsize=10, ha='center', va='top', style='italic', color='red')
        for stage in stages:
            stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
            if len(stage_data) > 0:
                ax3.plot(stage_data['elev_bin_mid'], stage_data['n_lakes'],
                        'o-', linewidth=2, markersize=4, label=stage, color=stage_colors[stage])

    # Add Dalton data if provided
    if dalton_df is not None and len(dalton_df) > 0:
        dalton_color = '#9467bd'  # Purple for Dalton
        dalton_sorted = dalton_df.sort_values('elev_bin_mid')

        # Try to normalize Dalton data too if we have zone_areas
        if zone_areas is not None and 'dalton_18ka' in zone_areas:
            dalton_area = zone_areas['dalton_18ka']
            if dalton_area > 0:
                total_dalton = dalton_sorted['n_lakes'].sum()
                densities = []
                elevs = []
                for _, row in dalton_sorted.iterrows():
                    n_lakes = row['n_lakes']
                    pct = row.get('pct_of_stage', 100 * n_lakes / total_dalton if total_dalton > 0 else 0)
                    if pct > 0:
                        est_area = dalton_area * pct / 100
                        density = n_lakes / est_area * 1000
                        densities.append(density)
                        elevs.append(row['elev_bin_mid'])
                if len(densities) > 0:
                    ax3.plot(elevs, densities, 'o-', linewidth=2.5, markersize=6,
                            label='Dalton 18ka', color=dalton_color, linestyle='--')
        else:
            # Fallback: plot raw Dalton counts
            ax3.plot(dalton_sorted['elev_bin_mid'], dalton_sorted['n_lakes'],
                    'o-', linewidth=2.5, markersize=6, label='Dalton 18ka',
                    color=dalton_color, linestyle='--')

    ax3.set_xlabel('Elevation (m)', fontsize=12)
    ax3.set_ylabel('Lake Density (per 1,000 km²)', fontsize=12)
    ax3.set_title('C) Area-Normalized Lake Density by Elevation',
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_davis_hypothesis_test(davis_results, density_df=None, figsize=(10, 8), save_path=None):
    """
    Visualize Davis's hypothesis test results.

    Shows the relationship between glacial stage age and lake density
    with regression line and statistical annotations.

    Parameters
    ----------
    davis_results : dict
        Output from test_davis_hypothesis()
    density_df : DataFrame, optional
        Density data for additional annotations
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    ages = np.array(davis_results.get('ages', []))
    densities = np.array(davis_results.get('densities', []))

    if len(ages) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for visualization',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        return fig, ax

    # Plot data points
    try:
        from .config import GLACIAL_CHRONOLOGY
    except ImportError:
        from config import GLACIAL_CHRONOLOGY

    if density_df is not None:
        for _, row in density_df.iterrows():
            if pd.notna(row['age_ka']) and pd.notna(row['density_per_1000km2']):
                color = row.get('color', '#1f77b4')
                ax.scatter(row['age_ka'], row['density_per_1000km2'],
                          s=200, c=color, edgecolors='black', linewidth=2, zorder=5)
                ax.annotate(row['glacial_stage'],
                           (row['age_ka'], row['density_per_1000km2']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=11, fontweight='bold')
    else:
        ax.scatter(ages, densities, s=150, c='steelblue', edgecolors='black', linewidth=2, zorder=5)

    # Add regression line
    slope = davis_results.get('slope')
    intercept = davis_results.get('intercept')
    if slope is not None and intercept is not None:
        x_line = np.linspace(min(ages)*0.9, max(ages)*1.1, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, label='Regression line', zorder=3)

        # Add confidence band (approximate)
        r_squared = davis_results.get('r_squared', 0)
        if r_squared > 0:
            residual_std = np.std(densities - (slope * ages + intercept))
            ax.fill_between(x_line, y_line - 1.96*residual_std, y_line + 1.96*residual_std,
                           alpha=0.2, color='red', label='95% CI')

    # Add statistics annotation
    stats_text = (
        f"Pearson r = {davis_results.get('correlation', np.nan):.3f}\n"
        f"p-value = {davis_results.get('p_value', np.nan):.4f}\n"
        f"R² = {davis_results.get('r_squared', np.nan):.3f}\n"
        f"Slope = {davis_results.get('slope', np.nan):.4f}\n"
        f"n = {davis_results.get('n_stages', 0)} stages"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Conclusion box
    supports = davis_results.get('supports_hypothesis')
    if supports:
        conclusion = "SUPPORTS Davis's Hypothesis"
        box_color = '#90EE90'
    elif supports is False:
        conclusion = "Does NOT support Davis's Hypothesis"
        box_color = '#FFB6C1'
    else:
        conclusion = "Insufficient data"
        box_color = '#FFE4B5'

    ax.text(0.5, 0.02, conclusion, transform=ax.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='bottom',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=box_color, edgecolor='black', linewidth=2))

    # Labels
    ax.set_xlabel('Glacial Stage Age (ka = thousands of years BP)', fontsize=14)
    ax.set_ylabel('Lake Density (lakes per 1,000 km²)', fontsize=14)
    ax.set_title("Davis's Hypothesis Test:\nLake Density vs. Landscape Age", fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_glacial_extent_map(lake_gdf, boundaries, figsize=(14, 10), save_path=None):
    """
    Create a geographic map showing glacial extents and lake locations.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lakes with glacial_stage column
    boundaries : dict
        Dictionary of glacial boundary GeoDataFrames
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    try:
        from .config import GLACIAL_CHRONOLOGY
    except ImportError:
        from config import GLACIAL_CHRONOLOGY

    # Plot glacial boundaries in order (oldest first, so newest is on top)
    boundary_order = ['driftless', 'illinoian', 'wisconsin', 'dalton_18ka']
    alphas = {'driftless': 0.3, 'illinoian': 0.4, 'wisconsin': 0.5, 'dalton_18ka': 0.6}

    for key in boundary_order:
        gdf = boundaries.get(key)
        if gdf is None:
            continue

        chrono = GLACIAL_CHRONOLOGY.get(key.replace('_18ka', '').replace('dalton', 'alpine'), {})
        color = chrono.get('color', '#808080')
        name = chrono.get('name', key)
        alpha = alphas.get(key, 0.5)

        gdf.plot(ax=ax, color=color, alpha=alpha, edgecolor='black',
                linewidth=0.5, label=name)

    # Plot lakes colored by glacial stage
    if 'glacial_stage' in lake_gdf.columns:
        for stage in lake_gdf['glacial_stage'].unique():
            stage_lakes = lake_gdf[lake_gdf['glacial_stage'] == stage]
            stage_lower = stage.lower()

            # Find color
            color = '#808080'
            for key, chrono in GLACIAL_CHRONOLOGY.items():
                if key in stage_lower or chrono['name'].lower() in stage_lower:
                    color = chrono['color']
                    break

            stage_lakes.plot(ax=ax, color=color, markersize=1, alpha=0.5, label=f'Lakes - {stage}')

    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Glacial Extents and Lake Distribution\n(USA Contiguous Albers Equal Area)', fontsize=14, fontweight='bold')

    # Add legend with proper handling of many items
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_glacial_chronosequence_summary(results, figsize=(16, 12), save_path=None):
    """
    Create a multi-panel summary figure for glacial chronosequence analysis.

    Parameters
    ----------
    results : dict
        Output from run_glacial_chronosequence_analysis()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig = plt.figure(figsize=figsize)

    # Create 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel A: Density by glacial stage (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    density_df = results.get('density_by_stage')
    if density_df is not None:
        df = density_df.copy()
        df['sort_age'] = df['age_ka'].fillna(999999)
        df = df.sort_values('sort_age')

        stages = df['glacial_stage'].values
        densities = df['density_per_1000km2'].values
        colors = df['color'].values if 'color' in df.columns else ['#1f77b4'] * len(df)

        bars = ax1.bar(range(len(stages)), densities, color=colors, edgecolor='black', linewidth=1.5)

        for i, (bar, d) in enumerate(zip(bars, densities)):
            if pd.notna(d):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{d:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels(stages, rotation=30, ha='right', fontsize=10)
        ax1.set_ylabel('Lakes per 1,000 km²', fontsize=12)
        ax1.set_title('A) Lake Density by Glacial Stage', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Panel B: Davis hypothesis test
    ax2 = fig.add_subplot(gs[0, 1])
    davis_results = results.get('davis_test', {})
    ages = np.array(davis_results.get('ages', []))
    densities_test = np.array(davis_results.get('densities', []))

    if len(ages) > 0:
        ax2.scatter(ages, densities_test, s=120, c='steelblue', edgecolors='black', linewidth=2, zorder=5)

        slope = davis_results.get('slope')
        intercept = davis_results.get('intercept')
        if slope is not None and intercept is not None:
            x_line = np.linspace(min(ages)*0.9, max(ages)*1.1, 100)
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, 'r--', linewidth=2)

        r = davis_results.get('correlation', np.nan)
        p = davis_results.get('p_value', np.nan)
        ax2.text(0.95, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                transform=ax2.transAxes, fontsize=11, va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        ax2.set_xlabel('Age (ka)', fontsize=12)
        ax2.set_ylabel('Lake Density', fontsize=12)
        ax2.set_title("B) Davis's Hypothesis Test", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # Panel C: Elevation distribution by stage
    ax3 = fig.add_subplot(gs[1, 0])
    elevation_df = results.get('elevation_by_stage')
    if elevation_df is not None:
        try:
            from .config import GLACIAL_CHRONOLOGY
        except ImportError:
            from config import GLACIAL_CHRONOLOGY

        for stage in elevation_df['glacial_stage'].unique():
            stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
            if len(stage_data) > 0:
                stage_lower = stage.lower()
                color = '#808080'
                for key, chrono in GLACIAL_CHRONOLOGY.items():
                    if key in stage_lower or chrono['name'].lower() in stage_lower:
                        color = chrono['color']
                        break
                ax3.plot(stage_data['elev_bin_mid'], stage_data['pct_of_stage'],
                        'o-', linewidth=2, markersize=3, label=stage, color=color)

        ax3.set_xlabel('Elevation (m)', fontsize=12)
        ax3.set_ylabel('Percent of Stage (%)', fontsize=12)
        ax3.set_title('C) Elevation Distribution by Stage', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)

    # Panel D: Lake count summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    if density_df is not None:
        table_data = []
        headers = ['Stage', 'n Lakes', 'Density', 'Mean Area', 'Age (ka)']

        for _, row in density_df.iterrows():
            table_data.append([
                row['glacial_stage'],
                f"{row['n_lakes']:,}",
                f"{row['density_per_1000km2']:.1f}" if pd.notna(row['density_per_1000km2']) else '-',
                f"{row['mean_lake_area_km2']:.4f}" if pd.notna(row['mean_lake_area_km2']) else '-',
                f"{row['age_ka']:.0f}" if pd.notna(row['age_ka']) else 'Never',
            ])

        table = ax4.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax4.set_title('D) Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    supports = davis_results.get('supports_hypothesis')
    if supports:
        conclusion = "Results SUPPORT Davis's Hypothesis"
    elif supports is False:
        conclusion = "Results do NOT support Davis's Hypothesis"
    else:
        conclusion = "Insufficient data for hypothesis test"

    fig.suptitle(f'Glacial Chronosequence Analysis\n{conclusion}',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, [ax1, ax2, ax3, ax4]


def plot_bimodal_decomposition(bimodal_results, figsize=(16, 10), save_path=None):
    """
    Visualize bimodal pattern decomposition by glacial status.

    Shows how glaciated vs never-glaciated regions contribute to the
    bimodal lake distribution pattern observed in elevation-normalized density.

    Parameters
    ----------
    bimodal_results : dict
        Output from decompose_bimodal_by_glacial_status() containing:
        - elevation_bins : array of elevation bin edges
        - glaciated_density : density from glaciated regions
        - never_glaciated_density : density from never-glaciated regions
        - total_density : total normalized density
        - lowland_peak_stats : stats for low-elevation peak
        - highland_peak_stats : stats for high-elevation peak
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: Stacked area plot showing contributions
    ax1 = axes[0, 0]

    elev_bins = bimodal_results.get('elevation_bins', [])
    glaciated = bimodal_results.get('glaciated_density', [])
    never_glac = bimodal_results.get('never_glaciated_density', [])

    if len(elev_bins) > 1:
        # Use bin midpoints for plotting
        midpoints = [(elev_bins[i] + elev_bins[i+1])/2 for i in range(len(elev_bins)-1)]

        # Ensure arrays match length
        if len(glaciated) == len(midpoints) and len(never_glac) == len(midpoints):
            ax1.fill_between(midpoints, 0, glaciated, alpha=0.7,
                           label='Glaciated regions', color='#1f77b4')
            ax1.fill_between(midpoints, glaciated,
                           np.array(glaciated) + np.array(never_glac),
                           alpha=0.7, label='Never glaciated', color='#ff7f0e')
            ax1.set_xlabel('Elevation (m)', fontsize=12)
            ax1.set_ylabel('Normalized Lake Density', fontsize=12)
            ax1.set_title('A) Bimodal Pattern Decomposition', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)

    # Panel B: Line plot comparing patterns
    ax2 = axes[0, 1]

    total = bimodal_results.get('total_density', [])
    if len(elev_bins) > 1 and len(total) == len(midpoints):
        ax2.plot(midpoints, total, 'k-', linewidth=2.5, label='Total', zorder=3)
        if len(glaciated) == len(midpoints):
            ax2.plot(midpoints, glaciated, '--', linewidth=2,
                    label='Glaciated', color='#1f77b4')
        if len(never_glac) == len(midpoints):
            ax2.plot(midpoints, never_glac, ':', linewidth=2,
                    label='Never glaciated', color='#ff7f0e')

        ax2.set_xlabel('Elevation (m)', fontsize=12)
        ax2.set_ylabel('Normalized Lake Density', fontsize=12)
        ax2.set_title('B) Pattern Comparison', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

    # Panel C: Peak statistics
    ax3 = axes[1, 0]
    ax3.axis('off')

    lowland = bimodal_results.get('lowland_peak_stats', {})
    highland = bimodal_results.get('highland_peak_stats', {})

    table_data = []
    headers = ['Metric', 'Lowland Peak', 'Highland Peak']

    metrics = [
        ('Peak elevation (m)', 'peak_elevation'),
        ('Peak density', 'peak_density'),
        ('Glaciated fraction', 'glaciated_fraction'),
        ('Primary driver', 'primary_driver'),
    ]

    for label, key in metrics:
        row = [label]
        for peak in [lowland, highland]:
            val = peak.get(key, 'N/A')
            if isinstance(val, float):
                row.append(f'{val:.3f}')
            else:
                row.append(str(val))
        table_data.append(row)

    if table_data:
        table = ax3.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax3.set_title('C) Peak Statistics', fontsize=14, fontweight='bold', y=0.9)

    # Panel D: Interpretation
    ax4 = axes[1, 1]
    ax4.axis('off')

    interpretation = bimodal_results.get('interpretation', {})
    text_lines = [
        'Bimodal Pattern Interpretation:',
        '',
        f"• Lowland peak: {interpretation.get('lowland_explanation', 'N/A')}",
        f"• Highland peak: {interpretation.get('highland_explanation', 'N/A')}",
        '',
        'Key Finding:',
        interpretation.get('key_finding', 'Analysis pending'),
    ]

    ax4.text(0.1, 0.9, '\n'.join(text_lines), transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_title('D) Interpretation', fontsize=14, fontweight='bold', y=0.95)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_power_law_by_glacial_zone(power_law_results, figsize=(14, 10), save_path=None):
    """
    Visualize power law analysis results by glacial zone.

    Shows how power law exponents (α) and x_min values vary across
    glacial chronosequence stages.

    Parameters
    ----------
    power_law_results : dict
        Output from power_law_by_glacial_zone() containing fit results
        for each glacial stage
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data from results
    stages = []
    alphas = []
    alpha_errors = []
    xmins = []
    n_tails = []
    ages = []

    for stage, data in power_law_results.items():
        if isinstance(data, dict) and 'alpha' in data:
            stages.append(stage)
            alphas.append(data.get('alpha', np.nan))
            # Error from bootstrap CI
            alpha_ci = data.get('alpha_ci', (np.nan, np.nan))
            if isinstance(alpha_ci, (list, tuple)) and len(alpha_ci) == 2:
                alpha_errors.append((data['alpha'] - alpha_ci[0], alpha_ci[1] - data['alpha']))
            else:
                alpha_errors.append((0, 0))
            xmins.append(data.get('x_min', np.nan))
            n_tails.append(data.get('n_tail', 0))
            ages.append(data.get('age_ka', np.nan))

    if len(stages) == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No valid power law fits', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        return fig, axes

    # Sort by age
    sort_idx = np.argsort([a if not np.isnan(a) else 999999 for a in ages])
    stages = [stages[i] for i in sort_idx]
    alphas = [alphas[i] for i in sort_idx]
    alpha_errors = [alpha_errors[i] for i in sort_idx]
    xmins = [xmins[i] for i in sort_idx]
    n_tails = [n_tails[i] for i in sort_idx]
    ages = [ages[i] for i in sort_idx]

    x_pos = np.arange(len(stages))

    # Panel A: Alpha by glacial zone
    ax1 = axes[0, 0]
    yerr_lower = [e[0] for e in alpha_errors]
    yerr_upper = [e[1] for e in alpha_errors]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(stages)))
    ax1.bar(x_pos, alphas, yerr=[yerr_lower, yerr_upper], capsize=5,
           color=colors, edgecolor='black', linewidth=1.5)

    # Reference lines
    ax1.axhline(2.05, color='red', linestyle='--', linewidth=2,
               label='Percolation (τ=2.05)')
    ax1.axhline(2.14, color='green', linestyle=':', linewidth=2,
               label='Global (τ=2.14)')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stages, rotation=30, ha='right', fontsize=10)
    ax1.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax1.set_title('A) Power Law Exponent by Glacial Stage', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: x_min by glacial zone
    ax2 = axes[0, 1]
    ax2.bar(x_pos, xmins, color=colors, edgecolor='black', linewidth=1.5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stages, rotation=30, ha='right', fontsize=10)
    ax2.set_ylabel('x_min (km²)', fontsize=12)
    ax2.set_title('B) Optimal x_min by Glacial Stage', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Panel C: Alpha vs Age scatter
    ax3 = axes[1, 0]
    valid_mask = [not np.isnan(a) for a in ages]
    valid_ages = [ages[i] for i in range(len(ages)) if valid_mask[i]]
    valid_alphas = [alphas[i] for i in range(len(alphas)) if valid_mask[i]]
    valid_stages = [stages[i] for i in range(len(stages)) if valid_mask[i]]

    if len(valid_ages) > 1:
        ax3.scatter(valid_ages, valid_alphas, s=120, c='steelblue',
                   edgecolors='black', linewidth=2, zorder=5)

        # Add stage labels
        for age, alpha, stage in zip(valid_ages, valid_alphas, valid_stages):
            ax3.annotate(stage, (age, alpha), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)

        # Fit regression line
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(valid_ages, valid_alphas)
        x_line = np.linspace(min(valid_ages)*0.9, max(valid_ages)*1.1, 100)
        y_line = slope * x_line + intercept
        ax3.plot(x_line, y_line, 'r--', linewidth=2)

        ax3.text(0.95, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                transform=ax3.transAxes, fontsize=11, va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax3.set_xlabel('Age (ka)', fontsize=12)
    ax3.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax3.set_title('C) α vs Landscape Age', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel D: Sample size by zone
    ax4 = axes[1, 1]
    ax4.bar(x_pos, n_tails, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, n in enumerate(n_tails):
        ax4.text(i, n, f'{n:,}', ha='center', va='bottom', fontsize=10)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stages, rotation=30, ha='right', fontsize=10)
    ax4.set_ylabel('n Lakes in Power Law Tail', fontsize=12)
    ax4.set_title('D) Sample Size by Glacial Stage', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_normalized_density_with_glacial_overlay(lake_gdf, density_df=None, elev_col='Elevation',
                                                   area_col='AreaSqKm', elev_bins=None,
                                                   figsize=(14, 8), save_path=None):
    """
    Create bar chart of lake counts by elevation with glacial fraction overlay.

    Shows lake counts by elevation bin with a line overlay indicating what fraction
    of lakes in each bin are within glaciated terrain.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lake data with glacial_stage classification and elevation column
    density_df : DataFrame, optional
        Optional normalized density results (not currently used, kept for compatibility)
    elev_col : str
        Elevation column name
    area_col : str
        Lake area column name
    elev_bins : array-like, optional
        Elevation bin edges (default: 0-4000m in 250m bins)
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, (ax1, ax2))
    """
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=figsize)

    if elev_bins is None:
        elev_bins = np.arange(0, 4250, 250)

    # Handle case where arguments might be swapped (for backwards compatibility)
    if hasattr(lake_gdf, 'geometry'):
        # Correct order: lake_gdf is a GeoDataFrame
        gdf = lake_gdf
    elif hasattr(density_df, 'geometry') if density_df is not None else False:
        # Arguments were swapped
        gdf = density_df
    else:
        # Neither is a GeoDataFrame, try to work with lake_gdf as DataFrame
        gdf = lake_gdf

    # Check if we have the required columns
    if elev_col not in gdf.columns:
        # Try common alternatives (including 'Elevation_' with trailing underscore from config)
        for alt_col in ['Elevation_', 'Elevation', 'elevation', 'ELEV', 'elev', 'Mean_Elevation', 'elev_m']:
            if alt_col in gdf.columns:
                elev_col = alt_col
                break

    # Also fix area column if needed
    if area_col not in gdf.columns:
        for alt_col in ['AREASQKM', 'AreaSqKm', 'area_km2', 'Area', 'AREA']:
            if alt_col in gdf.columns:
                area_col = alt_col
                break

    # Check if elevation column was found
    if elev_col not in gdf.columns:
        print(f"Warning: Elevation column not found. Tried: {['Elevation_', 'Elevation', 'elevation', 'ELEV', 'elev', 'Mean_Elevation', 'elev_m']}")
        print(f"Available columns: {list(gdf.columns[:20])}...")  # Show first 20 columns
        ax1.text(0.5, 0.5, f'Elevation column not found in data\nAvailable: {list(gdf.columns[:10])}...',
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)
        ax1.set_title('Lake Distribution by Elevation with Glacial Origin Overlay',
                     fontsize=16, fontweight='bold')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, (ax1, ax1)

    if 'glacial_stage' not in gdf.columns:
        print("Warning: glacial_stage column not found in data")
        # Create figure with just lake count by elevation
        if elev_col in gdf.columns:
            counts, _ = np.histogram(gdf[elev_col].dropna(), bins=elev_bins)
            bin_mids = [(elev_bins[i] + elev_bins[i+1]) / 2 for i in range(len(elev_bins)-1)]
            ax1.bar(bin_mids, counts, width=200, color='steelblue', alpha=0.7,
                   edgecolor='navy', linewidth=1)
            ax1.set_xlabel('Elevation (m)', fontsize=14)
            ax1.set_ylabel('Lake Count', fontsize=14)
            ax1.set_title('Lake Count by Elevation', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, (ax1, ax1)

    # Calculate lake counts and glacial fractions by elevation bin
    lake_counts = []
    glacial_fractions = []
    glacial_counts = []
    bin_mids = []

    for i in range(len(elev_bins) - 1):
        low, high = elev_bins[i], elev_bins[i+1]
        mid = (low + high) / 2
        bin_mids.append(mid)

        # Count lakes in this elevation bin
        if elev_col in gdf.columns:
            mask = (gdf[elev_col] >= low) & (gdf[elev_col] < high)
            bin_lakes = gdf[mask]
            total = len(bin_lakes)
            lake_counts.append(total)

            if total > 0:
                # Count glaciated lakes (not Driftless/never glaciated/unclassified)
                glaciated_mask = ~bin_lakes['glacial_stage'].str.lower().str.contains(
                    'driftless|never|unglaciated|unclassified', na=True)
                n_glaciated = glaciated_mask.sum()
                glacial_counts.append(n_glaciated)
                glacial_fractions.append(n_glaciated / total)
            else:
                glacial_counts.append(0)
                glacial_fractions.append(np.nan)
        else:
            lake_counts.append(0)
            glacial_counts.append(0)
            glacial_fractions.append(np.nan)

    # Bar chart of lake counts
    bars = ax1.bar(bin_mids, lake_counts, width=200, color='steelblue',
                  alpha=0.7, edgecolor='navy', linewidth=1,
                  label='Total Lakes')

    ax1.set_xlabel('Elevation (m)', fontsize=14)
    ax1.set_ylabel('Lake Count', fontsize=14, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Create second y-axis for glacial fraction
    ax2 = ax1.twinx()

    # Plot glacial fraction line
    glacial_fractions = np.array(glacial_fractions)
    valid_mask = ~np.isnan(glacial_fractions)
    if np.any(valid_mask):
        ax2.plot(np.array(bin_mids)[valid_mask],
                glacial_fractions[valid_mask] * 100,
                'o-', color='darkred', linewidth=2.5, markersize=8,
                label='% Glaciated Lakes')

        ax2.set_ylabel('% Lakes in Glaciated Terrain', fontsize=14, color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.set_ylim(0, 105)

        # Add annotation for peak glacial fraction
        valid_fracs = glacial_fractions[valid_mask]
        valid_mids = np.array(bin_mids)[valid_mask]
        if len(valid_fracs) > 0:
            max_idx = np.nanargmax(valid_fracs)
            if not np.isnan(valid_fracs[max_idx]):
                ax2.annotate(f'Peak: {valid_fracs[max_idx]*100:.1f}%\nat {valid_mids[max_idx]:.0f}m',
                            xy=(valid_mids[max_idx], valid_fracs[max_idx]*100),
                            xytext=(30, -20), textcoords='offset points',
                            fontsize=10, ha='left',
                            arrowprops=dict(arrowstyle='->', color='darkred'),
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_title('Lake Distribution by Elevation with Glacial Origin Overlay',
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(elev_bins[0], elev_bins[-1])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax1, ax2)


def plot_glacial_powerlaw_comparison(lake_gdf, area_col='AreaSqKm',
                                      min_area=0.01, figsize=(16, 12),
                                      save_path=None):
    """
    Create comprehensive power law comparison across glacial stages.

    Four-panel figure showing:
    - A) Rank-size plots (log-log) for each glacial stage
    - B) Complementary CDF comparison
    - C) Probability density functions
    - D) Alpha comparison bar chart with confidence intervals

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lake data with 'glacial_stage' column
    area_col : str
        Lake area column
    min_area : float
        Minimum lake area to include (lower threshold for more data)
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if lake_gdf is None or 'glacial_stage' not in lake_gdf.columns:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No glacial classification available',
                   ha='center', va='center', transform=ax.transAxes)
        return fig, axes

    # Get unique stages and assign colors
    stages = lake_gdf['glacial_stage'].dropna().unique()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stages)))
    stage_colors = dict(zip(stages, colors))

    # Filter by minimum area
    lake_gdf = lake_gdf[lake_gdf[area_col] >= min_area].copy()

    # Panel A: Rank-size plots
    ax1 = axes[0, 0]
    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]
        if len(areas) > 10:
            sorted_areas = np.sort(areas)[::-1]
            ranks = np.arange(1, len(sorted_areas) + 1)
            ax1.loglog(ranks, sorted_areas, '-', linewidth=2,
                      color=stage_colors[stage], label=f'{stage} (n={len(areas):,})')

    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Lake Area (km²)', fontsize=12)
    ax1.set_title('A) Rank-Size Distribution by Glacial Stage', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel B: Complementary CDF
    ax2 = axes[0, 1]
    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]
        if len(areas) > 10:
            sorted_areas = np.sort(areas)
            n = len(sorted_areas)
            # CCDF: P(X > x) - use (n-i)/n to avoid log(0) at the end
            ccdf = (n - np.arange(n)) / n
            ax2.loglog(sorted_areas, ccdf, '-', linewidth=2,
                      color=stage_colors[stage], label=stage)

    ax2.set_xlabel('Lake Area (km²)', fontsize=12)
    ax2.set_ylabel('P(X > x)', fontsize=12)
    ax2.set_title('B) Complementary CDF by Glacial Stage', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Panel C: Probability density (histogram)
    ax3 = axes[1, 0]
    area_bins = np.logspace(np.log10(min_area), np.log10(lake_gdf[area_col].max()), 40)

    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]
        if len(areas) > 10:
            counts, _ = np.histogram(areas, bins=area_bins)
            bin_widths = np.diff(area_bins)
            bin_centers = (area_bins[:-1] + area_bins[1:]) / 2
            # Normalize to get PDF
            pdf = counts / (len(areas) * bin_widths)
            ax3.loglog(bin_centers, pdf, 'o-', markersize=4, linewidth=1.5,
                      color=stage_colors[stage], label=stage)

    ax3.set_xlabel('Lake Area (km²)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('C) Size Distribution (PDF) by Glacial Stage', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # Panel D: Alpha comparison (simplified MLE estimate)
    ax4 = axes[1, 1]
    stage_names = []
    alphas = []
    alpha_errors = []

    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas >= min_area]
        if len(areas) > 30:
            # Simple MLE estimate: alpha = 1 + n / sum(log(x/xmin))
            xmin = min_area
            log_ratios = np.log(areas / xmin)
            n = len(areas)
            alpha = 1 + n / np.sum(log_ratios)
            # Standard error approximation
            se = (alpha - 1) / np.sqrt(n)
            stage_names.append(stage)
            alphas.append(alpha)
            alpha_errors.append(se * 1.96)  # 95% CI

    if len(stage_names) > 0:
        x_pos = np.arange(len(stage_names))
        colors_bar = [stage_colors[s] for s in stage_names]
        ax4.bar(x_pos, alphas, yerr=alpha_errors, capsize=5,
               color=colors_bar, edgecolor='black', linewidth=1.5)

        # Reference lines
        ax4.axhline(2.05, color='red', linestyle='--', linewidth=2,
                   label='Percolation (τ=2.05)')
        ax4.axhline(2.14, color='green', linestyle=':', linewidth=2,
                   label='Global (τ=2.14)')

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(stage_names, rotation=30, ha='right', fontsize=10)
        ax4.set_ylabel('Power Law Exponent (α)', fontsize=12)
        ax4.set_title(f'D) Power Law Exponent (x_min={min_area} km²)',
                     fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_glacial_lake_size_histograms(lake_gdf, area_col='AreaSqKm',
                                        min_area=0.001, figsize=(16, 10),
                                        save_path=None):
    """
    Create stacked histograms showing lake size distribution by glacial stage.

    Shows raw counts and cumulative area contribution by size class.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lake data with 'glacial_stage' column
    area_col : str
        Lake area column
    min_area : float
        Minimum area threshold (lower = more lakes)
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if lake_gdf is None or 'glacial_stage' not in lake_gdf.columns:
        return fig, axes

    stages = lake_gdf['glacial_stage'].dropna().unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(stages)))
    stage_colors = dict(zip(stages, colors))

    # Filter by minimum area
    lake_gdf = lake_gdf[lake_gdf[area_col] >= min_area].copy()

    # Define size classes
    size_classes = [
        (0.001, 0.01, 'Tiny\n(<0.01 km²)'),
        (0.01, 0.1, 'Small\n(0.01-0.1 km²)'),
        (0.1, 1.0, 'Medium\n(0.1-1 km²)'),
        (1.0, 10.0, 'Large\n(1-10 km²)'),
        (10.0, 100.0, 'Very Large\n(10-100 km²)'),
        (100.0, 10000.0, 'Massive\n(>100 km²)')
    ]

    # Panel A: Count by size class (stacked)
    ax1 = axes[0, 0]
    x_pos = np.arange(len(size_classes))
    width = 0.8 / len(stages)
    bottom = np.zeros(len(size_classes))

    for j, stage in enumerate(stages):
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        counts = []
        for low, high, _ in size_classes:
            mask = (stage_data[area_col] >= low) & (stage_data[area_col] < high)
            counts.append(mask.sum())
        ax1.bar(x_pos + j*width, counts, width, label=stage,
               color=stage_colors[stage], edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x_pos + width * (len(stages)-1) / 2)
    ax1.set_xticklabels([s[2] for s in size_classes], fontsize=9)
    ax1.set_ylabel('Number of Lakes', fontsize=12)
    ax1.set_title('A) Lake Counts by Size Class', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: Proportion by size class
    ax2 = axes[0, 1]
    for j, stage in enumerate(stages):
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        total = len(stage_data)
        if total > 0:
            proportions = []
            for low, high, _ in size_classes:
                mask = (stage_data[area_col] >= low) & (stage_data[area_col] < high)
                proportions.append(mask.sum() / total * 100)
            ax2.bar(x_pos + j*width, proportions, width, label=stage,
                   color=stage_colors[stage], edgecolor='black', linewidth=0.5)

    ax2.set_xticks(x_pos + width * (len(stages)-1) / 2)
    ax2.set_xticklabels([s[2] for s in size_classes], fontsize=9)
    ax2.set_ylabel('Percent of Stage (%)', fontsize=12)
    ax2.set_title('B) Lake Size Distribution (% of each stage)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Panel C: Cumulative area by stage
    ax3 = axes[1, 0]
    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = np.sort(stage_data[area_col].values)[::-1]
        if len(areas) > 0:
            cumsum = np.cumsum(areas)
            pct_cumsum = cumsum / cumsum[-1] * 100
            ax3.semilogx(areas, pct_cumsum, '-', linewidth=2,
                        color=stage_colors[stage], label=stage)

    ax3.set_xlabel('Lake Area (km²)', fontsize=12)
    ax3.set_ylabel('Cumulative % of Total Area', fontsize=12)
    ax3.set_title('C) Cumulative Area by Lake Size', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(90, color='gray', linestyle='--', alpha=0.5)

    # Panel D: Mean and median lake size
    ax4 = axes[1, 1]
    stage_names = []
    means = []
    medians = []
    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        if len(stage_data) > 0:
            stage_names.append(stage)
            means.append(stage_data[area_col].mean())
            medians.append(stage_data[area_col].median())

    if len(stage_names) > 0:
        x_pos = np.arange(len(stage_names))
        width = 0.35
        ax4.bar(x_pos - width/2, means, width, label='Mean',
               color='steelblue', edgecolor='black')
        ax4.bar(x_pos + width/2, medians, width, label='Median',
               color='coral', edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(stage_names, rotation=30, ha='right', fontsize=10)
        ax4.set_ylabel('Lake Area (km²)', fontsize=12)
        ax4.set_title('D) Mean vs Median Lake Size', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.set_yscale('log')
        ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_glacial_xmin_sensitivity(lake_gdf, area_col='AreaSqKm',
                                   xmin_range=None, figsize=(16, 10),
                                   save_path=None):
    """
    Analyze x_min sensitivity for power law fits by glacial stage.

    Shows how alpha and goodness-of-fit vary with x_min choice for each stage.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lake data with 'glacial_stage' column
    area_col : str
        Lake area column
    xmin_range : array-like, optional
        Range of x_min values to test
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if lake_gdf is None or 'glacial_stage' not in lake_gdf.columns:
        return fig, axes

    if xmin_range is None:
        xmin_range = np.logspace(-3, 1, 20)  # 0.001 to 10 km²

    stages = lake_gdf['glacial_stage'].dropna().unique()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stages)))
    stage_colors = dict(zip(stages, colors))

    # Panel A: Alpha vs x_min
    ax1 = axes[0, 0]
    optimal_xmins = {}

    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]

        alphas = []
        valid_xmins = []
        n_tails = []

        for xmin in xmin_range:
            tail_data = areas[areas >= xmin]
            if len(tail_data) > 30:
                log_ratios = np.log(tail_data / xmin)
                n = len(tail_data)
                alpha = 1 + n / np.sum(log_ratios)
                alphas.append(alpha)
                valid_xmins.append(xmin)
                n_tails.append(n)

        if len(valid_xmins) > 3:
            ax1.semilogx(valid_xmins, alphas, 'o-', linewidth=2, markersize=4,
                        color=stage_colors[stage], label=stage)

    ax1.axhline(2.05, color='red', linestyle='--', linewidth=2, label='τ=2.05')
    ax1.axhline(2.14, color='green', linestyle=':', linewidth=2, label='τ=2.14')
    ax1.set_xlabel('x_min (km²)', fontsize=12)
    ax1.set_ylabel('Power Law Exponent (α)', fontsize=12)
    ax1.set_title('A) α Sensitivity to x_min', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Sample size vs x_min
    ax2 = axes[0, 1]
    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]

        n_above = [np.sum(areas >= xmin) for xmin in xmin_range]
        ax2.loglog(xmin_range, n_above, 'o-', linewidth=2, markersize=4,
                  color=stage_colors[stage], label=stage)

    ax2.axhline(100, color='gray', linestyle='--', alpha=0.5, label='n=100')
    ax2.axhline(30, color='gray', linestyle=':', alpha=0.5, label='n=30')
    ax2.set_xlabel('x_min (km²)', fontsize=12)
    ax2.set_ylabel('Sample Size (n ≥ x_min)', fontsize=12)
    ax2.set_title('B) Sample Size vs x_min', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Panel C: Alpha stability (standard deviation in sliding window)
    ax3 = axes[1, 0]
    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]

        alphas = []
        valid_xmins = []

        for xmin in xmin_range:
            tail_data = areas[areas >= xmin]
            if len(tail_data) > 30:
                log_ratios = np.log(tail_data / xmin)
                n = len(tail_data)
                alpha = 1 + n / np.sum(log_ratios)
                alphas.append(alpha)
                valid_xmins.append(xmin)

        if len(alphas) > 5:
            # Rolling standard deviation
            window = 3
            rolling_std = []
            for i in range(len(alphas) - window + 1):
                rolling_std.append(np.std(alphas[i:i+window]))
            ax3.semilogx(valid_xmins[window-1:], rolling_std, 'o-',
                        linewidth=2, markersize=4,
                        color=stage_colors[stage], label=stage)

    ax3.set_xlabel('x_min (km²)', fontsize=12)
    ax3.set_ylabel('α Variability (rolling σ)', fontsize=12)
    ax3.set_title('C) α Stability (lower = more stable)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel D: Recommended x_min summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate recommended x_min for each stage (where alpha stabilizes)
    summary_text = "Recommended x_min by Stage:\n" + "=" * 30 + "\n\n"

    for stage in stages:
        stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
        areas = stage_data[area_col].values
        areas = areas[areas > 0]
        n_total = len(areas)

        # Find x_min where we have at least 100 samples
        recommended = None
        for xmin in xmin_range:
            if np.sum(areas >= xmin) >= 100:
                recommended = xmin
                break

        if recommended:
            tail_n = np.sum(areas >= recommended)
            log_ratios = np.log(areas[areas >= recommended] / recommended)
            alpha = 1 + tail_n / np.sum(log_ratios)
            summary_text += f"{stage}:\n"
            summary_text += f"  x_min = {recommended:.3f} km²\n"
            summary_text += f"  α = {alpha:.2f}, n = {tail_n:,}\n\n"
        else:
            summary_text += f"{stage}: Insufficient data\n\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('D) Recommended Settings', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_glacial_geographic_lakes(lake_gdf, boundaries=None, figsize=(16, 12),
                                   save_path=None):
    """
    Create geographic visualization of lakes colored by glacial stage.

    Shows lake locations with different colors for each glacial stage,
    overlaid on glacial boundary outlines.

    Parameters
    ----------
    lake_gdf : GeoDataFrame
        Lake data with geometry and 'glacial_stage' column
    boundaries : dict, optional
        Dictionary of glacial boundary GeoDataFrames
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    if lake_gdf is None or 'glacial_stage' not in lake_gdf.columns:
        ax.text(0.5, 0.5, 'No glacial data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        return fig, ax

    stages = lake_gdf['glacial_stage'].dropna().unique()

    # Define colors for stages
    stage_color_map = {
        'Wisconsin': '#1f77b4',
        'Illinoian': '#ff7f0e',
        'Pre-Illinoian': '#2ca02c',
        'Driftless': '#d62728',
        'Alpine (Dalton 18ka)': '#9467bd',
        'Never Glaciated': '#8c564b',
    }

    # Plot glacial boundaries first (as outlines)
    if boundaries:
        boundary_colors = {
            'wisconsin': '#1f77b4',
            'illinoian': '#ff7f0e',
            'driftless': '#d62728',
            'dalton_18ka': '#9467bd',
        }
        for name, gdf in boundaries.items():
            if gdf is not None and len(gdf) > 0:
                color = boundary_colors.get(name, 'gray')
                gdf.boundary.plot(ax=ax, color=color, linewidth=1.5,
                                 alpha=0.7, label=f'{name} boundary')

    # Plot lakes by stage
    for stage in stages:
        stage_lakes = lake_gdf[lake_gdf['glacial_stage'] == stage]
        if len(stage_lakes) > 0:
            # Get color
            color = stage_color_map.get(stage, 'gray')
            for key, c in stage_color_map.items():
                if key.lower() in stage.lower():
                    color = c
                    break

            # Plot as points (centroid if polygon)
            if 'geometry' in stage_lakes.columns:
                try:
                    centroids = stage_lakes.geometry.centroid
                    ax.scatter(centroids.x, centroids.y, s=3, c=color,
                              alpha=0.6, label=f'{stage} (n={len(stage_lakes):,})')
                except:
                    pass

    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Geographic Distribution of Lakes by Glacial Stage',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, markerscale=3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_glacial_comprehensive_summary(results, lake_gdf=None, figsize=(20, 16),
                                         save_path=None):
    """
    Create comprehensive 6-panel summary of glacial chronosequence analysis.

    Panels:
    A) Lake density by glacial stage (bar chart)
    B) Davis's hypothesis test (scatter + regression)
    C) Power law comparison (rank-size)
    D) Size distribution by stage
    E) Elevation distribution by stage
    F) Summary statistics table

    Parameters
    ----------
    results : dict
        Full glacial analysis results
    lake_gdf : GeoDataFrame, optional
        Classified lake data
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Get data
    density_df = results.get('density_by_stage')
    davis_test = results.get('davis_test', {})
    if lake_gdf is None:
        lake_gdf = results.get('lake_gdf')

    # Panel A: Density by stage
    ax1 = fig.add_subplot(gs[0, 0])
    if density_df is not None and len(density_df) > 0:
        df = density_df.copy()
        df['sort_age'] = df['age_ka'].fillna(999999)
        df = df.sort_values('sort_age')

        stages = df['glacial_stage'].values
        densities = df['density_per_1000km2'].values
        colors = df['color'].values if 'color' in df.columns else plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))

        bars = ax1.bar(range(len(stages)), densities, color=colors, edgecolor='black', linewidth=1.5)

        for i, (bar, d) in enumerate(zip(bars, densities)):
            if pd.notna(d):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{d:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels(stages, rotation=30, ha='right', fontsize=10)
        ax1.set_ylabel('Lakes per 1,000 km²', fontsize=12)
        ax1.set_title('A) Lake Density by Glacial Stage', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Panel B: Davis hypothesis test
    ax2 = fig.add_subplot(gs[0, 1])
    ages = np.array(davis_test.get('ages', []))
    densities_test = np.array(davis_test.get('densities', []))

    if len(ages) > 0 and len(densities_test) > 0:
        ax2.scatter(ages, densities_test, s=150, c='steelblue', edgecolors='black',
                   linewidth=2, zorder=5)

        # Regression line
        slope = davis_test.get('slope')
        intercept = davis_test.get('intercept')
        if slope is not None and intercept is not None:
            x_line = np.linspace(min(ages)*0.9, max(ages)*1.1, 100)
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, 'r--', linewidth=2.5)

        r = davis_test.get('correlation', np.nan)
        p = davis_test.get('p_value', np.nan)
        ax2.text(0.95, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                transform=ax2.transAxes, fontsize=12, va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        ax2.set_xlabel('Age (ka)', fontsize=12)
        ax2.set_ylabel('Lake Density', fontsize=12)
        ax2.set_title("B) Davis's Hypothesis Test", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # Panel C: Power law (rank-size)
    ax3 = fig.add_subplot(gs[1, 0])
    if lake_gdf is not None and 'glacial_stage' in lake_gdf.columns:
        stages = lake_gdf['glacial_stage'].dropna().unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(stages)))

        # Find area column (case-insensitive)
        area_col = None
        for col in lake_gdf.columns:
            if col.lower() in ['areasqkm', 'area_km2', 'area', 'lakeareasqkm']:
                area_col = col
                break
        if area_col is None:
            # Fallback: look for columns containing 'area'
            for col in lake_gdf.columns:
                if 'area' in col.lower() and lake_gdf[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    area_col = col
                    break
        if area_col is None:
            area_col = lake_gdf.select_dtypes(include=[np.number]).columns[0]
            print(f"Warning: Using '{area_col}' as area column (no area column found)")

        for i, stage in enumerate(stages):
            stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
            areas = stage_data[area_col].dropna().values
            areas = areas[areas > 0.01]  # Lower threshold
            if len(areas) > 10:
                sorted_areas = np.sort(areas)[::-1]
                ranks = np.arange(1, len(sorted_areas) + 1)
                ax3.loglog(ranks, sorted_areas, '-', linewidth=2,
                          color=colors[i], label=f'{stage} (n={len(areas):,})')

        ax3.set_xlabel('Rank', fontsize=12)
        ax3.set_ylabel('Lake Area (km²)', fontsize=12)
        ax3.set_title('C) Rank-Size Distribution', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')

    # Panel D: Size distribution (CCDF)
    ax4 = fig.add_subplot(gs[1, 1])
    if lake_gdf is not None and 'glacial_stage' in lake_gdf.columns:
        for i, stage in enumerate(stages):
            stage_data = lake_gdf[lake_gdf['glacial_stage'] == stage]
            areas = stage_data[area_col].dropna().values
            areas = areas[areas > 0.001]
            if len(areas) > 10:
                sorted_areas = np.sort(areas)
                n = len(sorted_areas)
                # CCDF: P(X > x) - use (n-i)/n to avoid log(0) at the end
                ccdf = (n - np.arange(n)) / n
                ax4.loglog(sorted_areas, ccdf, '-', linewidth=2,
                          color=colors[i], label=stage)

        ax4.set_xlabel('Lake Area (km²)', fontsize=12)
        ax4.set_ylabel('P(X > x)', fontsize=12)
        ax4.set_title('D) Complementary CDF', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, which='both')

    # Panel E: Elevation distribution
    ax5 = fig.add_subplot(gs[2, 0])
    elevation_df = results.get('elevation_by_stage')
    if elevation_df is not None:
        for stage in elevation_df['glacial_stage'].unique():
            stage_data = elevation_df[elevation_df['glacial_stage'] == stage].sort_values('elev_bin_mid')
            if len(stage_data) > 0:
                ax5.plot(stage_data['elev_bin_mid'], stage_data['pct_of_stage'],
                        'o-', linewidth=2, markersize=4, label=stage)

        ax5.set_xlabel('Elevation (m)', fontsize=12)
        ax5.set_ylabel('Percent of Stage (%)', fontsize=12)
        ax5.set_title('E) Elevation Distribution by Stage', fontsize=14, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(True, alpha=0.3)

    # Panel F: Summary table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    if density_df is not None and len(density_df) > 0:
        table_data = []
        headers = ['Stage', 'n Lakes', 'Density', 'Mean (km²)', 'Median (km²)']

        for _, row in density_df.iterrows():
            # Get median if available
            median_val = row.get('median_lake_area_km2', '-')
            if pd.notna(median_val) and median_val != '-':
                median_str = f"{median_val:.4f}"
            else:
                median_str = '-'

            table_data.append([
                row['glacial_stage'],
                f"{row['n_lakes']:,}",
                f"{row['density_per_1000km2']:.1f}" if pd.notna(row['density_per_1000km2']) else '-',
                f"{row['mean_lake_area_km2']:.4f}" if pd.notna(row['mean_lake_area_km2']) else '-',
                median_str,
            ])

        table = ax6.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)

        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Add conclusion
    supports = davis_test.get('supports_hypothesis')
    if supports:
        conclusion = "SUPPORTS Davis's Hypothesis"
        color = 'green'
    elif supports is False:
        conclusion = "Does NOT support Davis's Hypothesis"
        color = 'red'
    else:
        conclusion = "Insufficient data for test"
        color = 'gray'

    ax6.set_title(f'F) Summary: {conclusion}', fontsize=14, fontweight='bold',
                 color=color, y=0.95)

    fig.suptitle('Glacial Chronosequence Analysis - Comprehensive Summary',
                fontsize=18, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]


# ============================================================================
# SPATIAL SCALING VISUALIZATIONS
# ============================================================================

def plot_latitudinal_scaling(lat_results, figsize=(16, 10), save_path=None):
    """
    Visualize latitudinal scaling patterns in lake properties.

    Parameters
    ----------
    lat_results : dict
        Output from analyze_latitudinal_scaling()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    binned = lat_results.get('binned_stats')
    if binned is None or len(binned) == 0:
        return fig, axes

    # Panel A: Lake count by latitude
    ax1 = axes[0, 0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(binned)))
    bars = ax1.bar(range(len(binned)), binned['n_lakes'], color=colors, edgecolor='black')
    ax1.set_xticks(range(len(binned)))
    ax1.set_xticklabels(binned['lat_band'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Number of Lakes', fontsize=12)
    ax1.set_title('A) Lake Count by Latitude', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: Mean lake size by latitude
    ax2 = axes[0, 1]
    ax2.errorbar(binned['lat_mid'], binned['mean_area_km2'],
                fmt='o-', color='steelblue', linewidth=2, markersize=10,
                capsize=0, label='Mean')
    ax2.plot(binned['lat_mid'], binned['median_area_km2'],
            's--', color='coral', linewidth=2, markersize=8, label='Median')
    ax2.set_xlabel('Latitude (°N)', fontsize=12)
    ax2.set_ylabel('Lake Area (km²)', fontsize=12)
    ax2.set_title('B) Lake Size vs Latitude', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    if 'regression' in lat_results:
        reg = lat_results['regression']
        x_line = np.linspace(binned['lat_mid'].min(), binned['lat_mid'].max(), 100)
        y_line = 10**(reg['slope'] * x_line + reg['intercept'])
        ax2.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
                label=f"Trend (R²={reg['r_squared']:.3f})")
        ax2.legend(loc='upper left', fontsize=9)

    # Panel C: Power law α by latitude
    ax3 = axes[1, 0]
    valid = binned.dropna(subset=['alpha'])
    if len(valid) > 0:
        ax3.errorbar(valid['lat_mid'], valid['alpha'],
                    yerr=valid['alpha_se']*1.96, fmt='o-',
                    color='darkgreen', linewidth=2, markersize=10, capsize=5)
        ax3.axhline(2.05, color='red', linestyle='--', linewidth=2, label='τ=2.05')
        ax3.axhline(2.14, color='orange', linestyle=':', linewidth=2, label='τ=2.14')
        ax3.set_xlabel('Latitude (°N)', fontsize=12)
        ax3.set_ylabel('Power Law Exponent (α)', fontsize=12)
        ax3.set_title('C) α vs Latitude', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)

    # Panel D: Correlation summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    corr = lat_results.get('correlation', {})
    reg = lat_results.get('regression', {})
    alpha_test = lat_results.get('alpha_latitude_test', {})

    summary_text = "LATITUDINAL SCALING SUMMARY\n" + "=" * 35 + "\n\n"
    summary_text += f"Lake Size ~ Latitude:\n"
    summary_text += f"  r = {corr.get('r', np.nan):.4f}\n"
    summary_text += f"  p = {corr.get('p_value', np.nan):.2e}\n"
    summary_text += f"  → {corr.get('interpretation', 'N/A')}\n\n"

    summary_text += f"Regression (log₁₀ Area ~ Lat):\n"
    summary_text += f"  R² = {reg.get('r_squared', np.nan):.4f}\n"
    summary_text += f"  slope = {reg.get('slope', np.nan):.4f}\n\n"

    if alpha_test:
        summary_text += f"α ~ Latitude:\n"
        summary_text += f"  r = {alpha_test.get('r', np.nan):.4f}\n"
        summary_text += f"  → {alpha_test.get('interpretation', 'N/A')}\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title('D) Summary', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_glacial_vs_nonglacial_comparison(comparison_results, figsize=(16, 12),
                                           save_path=None):
    """
    Visualize comparison between glaciated and non-glaciated lake distributions.

    Parameters
    ----------
    comparison_results : dict
        Output from compare_glacial_vs_nonglacial_scaling()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    glac = comparison_results.get('glaciated_stats', {})
    nonglac = comparison_results.get('non_glaciated_stats', {})

    # Panel A: Size distribution comparison (CCDF)
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.5, 'CCDF comparison\n(requires lake data)',
            ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title('A) Size Distribution (CCDF)', fontsize=14, fontweight='bold')

    # Panel B: Summary statistics bar chart
    ax2 = axes[0, 1]
    metrics = ['Mean Area', 'Median Area', 'Max Area']
    glac_vals = [glac.get('mean', 0), glac.get('median', 0), glac.get('max', 0)]
    nonglac_vals = [nonglac.get('mean', 0), nonglac.get('median', 0), nonglac.get('max', 0)]

    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, glac_vals, width, label='Glaciated', color='steelblue', edgecolor='black')
    ax2.bar(x + width/2, nonglac_vals, width, label='Non-Glaciated', color='coral', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylabel('Lake Area (km²)', fontsize=12)
    ax2.set_title('B) Size Statistics Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)

    # Panel C: Power law comparison
    ax3 = axes[1, 0]
    pl = comparison_results.get('power_law_comparison', {})

    if pl.get('glaciated_alpha') and pl.get('non_glaciated_alpha'):
        categories = ['Glaciated', 'Non-Glaciated']
        alphas = [pl['glaciated_alpha'], pl['non_glaciated_alpha']]
        errors = [pl.get('glaciated_se', 0) * 1.96, pl.get('non_glaciated_se', 0) * 1.96]
        colors = ['steelblue', 'coral']

        bars = ax3.bar(categories, alphas, yerr=errors, capsize=10,
                      color=colors, edgecolor='black', linewidth=2)
        ax3.axhline(2.05, color='red', linestyle='--', linewidth=2, label='τ=2.05')
        ax3.axhline(2.14, color='green', linestyle=':', linewidth=2, label='τ=2.14')

        # Add significance annotation
        if pl.get('significant'):
            max_y = max(alphas) + max(errors) + 0.1
            ax3.annotate('*', xy=(0.5, max_y), fontsize=24, ha='center', fontweight='bold')
            ax3.plot([0, 1], [max_y - 0.02, max_y - 0.02], 'k-', linewidth=2)

        ax3.set_ylabel('Power Law Exponent (α)', fontsize=12)
        ax3.set_title('C) Power Law Exponent Comparison', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

    # Panel D: Colorful hypothesis test summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create colorful table
    table_data = []
    colors_table = []

    # Mann-Whitney test
    mw = comparison_results.get('mann_whitney_test', {})
    if mw:
        sig = mw.get('significant', False)
        table_data.append(['Size Distribution', f"U = {mw.get('U_statistic', 0):.0f}",
                          f"{mw.get('p_value', 1):.2e}", 'Yes' if sig else 'No'])
        colors_table.append(['lightgreen' if sig else 'lightcoral'] * 4)

    # KS test
    ks = comparison_results.get('ks_test', {})
    if ks:
        sig = ks.get('significant', False)
        table_data.append(['Distribution Shape', f"D = {ks.get('D_statistic', 0):.4f}",
                          f"{ks.get('p_value', 1):.2e}", 'Yes' if sig else 'No'])
        colors_table.append(['lightgreen' if sig else 'lightcoral'] * 4)

    # Power law
    if pl.get('p_value'):
        sig = pl.get('significant', False)
        table_data.append(['α Difference', f"z = {pl.get('z_statistic', 0):.2f}",
                          f"{pl.get('p_value', 1):.4f}", 'Yes' if sig else 'No'])
        colors_table.append(['lightgreen' if sig else 'lightcoral'] * 4)

    if table_data:
        headers = ['Test', 'Statistic', 'p-value', 'Significant?']
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellColours=colors_table,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('D) Hypothesis Test Results', fontsize=14, fontweight='bold', y=0.95)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_colorful_hypothesis_table(hypothesis_df, figsize=(14, 8), save_path=None):
    """
    Create a colorful, publication-quality hypothesis test summary table.

    Parameters
    ----------
    hypothesis_df : DataFrame
        Table with columns: Hypothesis, Test, Statistic, p-value, Significant, Conclusion
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    if hypothesis_df is None or len(hypothesis_df) == 0:
        ax.text(0.5, 0.5, 'No hypothesis tests available',
               ha='center', va='center', fontsize=14)
        return fig, ax

    # Prepare data
    columns = ['Hypothesis', 'Test', 'Statistic', 'p-value', 'Conclusion']
    cell_text = []
    cell_colors = []

    for _, row in hypothesis_df.iterrows():
        sig = row.get('Significant', False)
        p_val = row.get('p-value', 1)

        # Format p-value
        if p_val < 0.001:
            p_str = f"{p_val:.2e}"
        else:
            p_str = f"{p_val:.4f}"

        cell_text.append([
            row.get('Hypothesis', ''),
            row.get('Test', ''),
            row.get('Statistic', ''),
            p_str,
            row.get('Conclusion', '')
        ])

        # Color based on significance
        if sig:
            # Gradient based on p-value (more significant = more green)
            if p_val < 0.001:
                color = '#90EE90'  # Light green
            elif p_val < 0.01:
                color = '#98FB98'  # Pale green
            else:
                color = '#F0FFF0'  # Honeydew
        else:
            color = '#FFE4E1'  # Misty rose

        cell_colors.append([color] * len(columns))

    # Create table
    table = ax.table(cellText=cell_text,
                    colLabels=columns,
                    cellColours=cell_colors,
                    loc='center',
                    cellLoc='center')

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)

    # Style header
    for i, col in enumerate(columns):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=11)

    # Add legend
    ax.text(0.02, 0.02, '■ Significant (p < 0.05)', color='green',
           transform=ax.transAxes, fontsize=10, fontweight='bold')
    ax.text(0.25, 0.02, '■ Not Significant', color='red',
           transform=ax.transAxes, fontsize=10, fontweight='bold')

    ax.set_title('Hypothesis Test Summary',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_spatial_scaling_summary(spatial_results, figsize=(20, 16), save_path=None):
    """
    Create comprehensive 6-panel summary of spatial scaling analysis.

    Parameters
    ----------
    spatial_results : dict
        Output from run_spatial_scaling_analysis()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Panel A: Latitudinal pattern
    ax1 = fig.add_subplot(gs[0, 0])
    lat_results = spatial_results.get('latitudinal', {})
    if 'binned_stats' in lat_results:
        binned = lat_results['binned_stats']
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(binned)))
        ax1.bar(range(len(binned)), binned['n_lakes'], color=colors, edgecolor='black')
        ax1.set_xticks(range(len(binned)))
        ax1.set_xticklabels(binned['lat_band'], rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Number of Lakes', fontsize=11)
        ax1.set_title('A) Lake Count by Latitude', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Panel B: Longitudinal pattern
    ax2 = fig.add_subplot(gs[0, 1])
    lon_results = spatial_results.get('longitudinal', {})
    if 'binned_stats' in lon_results:
        binned = lon_results['binned_stats']
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(binned)))
        ax2.bar(range(len(binned)), binned['n_lakes'], color=colors, edgecolor='black')
        ax2.set_xticks(range(len(binned)))
        ax2.set_xticklabels(binned['lon_band'], rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Number of Lakes', fontsize=11)
        ax2.set_title('B) Lake Count by Longitude', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

    # Panel C: Elevation pattern
    ax3 = fig.add_subplot(gs[1, 0])
    elev_results = spatial_results.get('elevation', {})
    if 'binned_stats' in elev_results:
        binned = elev_results['binned_stats']
        ax3.plot(binned['elev_mid'], binned['mean_area_km2'], 'o-',
                color='steelblue', linewidth=2, markersize=8, label='Mean')
        ax3.plot(binned['elev_mid'], binned['median_area_km2'], 's--',
                color='coral', linewidth=2, markersize=6, label='Median')
        ax3.set_xlabel('Elevation (m)', fontsize=11)
        ax3.set_ylabel('Lake Area (km²)', fontsize=11)
        ax3.set_title('C) Lake Size vs Elevation', fontsize=12, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)

    # Panel D: Glacial comparison
    ax4 = fig.add_subplot(gs[1, 1])
    glac_results = spatial_results.get('glacial_comparison', {})
    if glac_results and 'glaciated_stats' in glac_results:
        glac = glac_results['glaciated_stats']
        nonglac = glac_results['non_glaciated_stats']

        categories = ['Glaciated', 'Non-Glaciated']
        means = [glac.get('mean', 0), nonglac.get('mean', 0)]
        medians = [glac.get('median', 0), nonglac.get('median', 0)]

        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, means, width, label='Mean', color='steelblue', edgecolor='black')
        ax4.bar(x + width/2, medians, width, label='Median', color='coral', edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, fontsize=11)
        ax4.set_ylabel('Lake Area (km²)', fontsize=11)
        ax4.set_title('D) Glaciated vs Non-Glaciated', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_yscale('log')
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Glacial comparison\nnot available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('D) Glaciated vs Non-Glaciated', fontsize=12, fontweight='bold')

    # Panel E: Alpha by latitude
    ax5 = fig.add_subplot(gs[2, 0])
    if 'binned_stats' in lat_results:
        binned = lat_results['binned_stats']
        valid = binned.dropna(subset=['alpha'])
        if len(valid) > 0:
            ax5.errorbar(valid['lat_mid'], valid['alpha'],
                        yerr=valid['alpha_se']*1.96, fmt='o-',
                        color='darkgreen', linewidth=2, markersize=8, capsize=5)
            ax5.axhline(2.05, color='red', linestyle='--', linewidth=2, label='τ=2.05')
            ax5.set_xlabel('Latitude (°N)', fontsize=11)
            ax5.set_ylabel('Power Law α', fontsize=11)
            ax5.set_title('E) Power Law Exponent vs Latitude', fontsize=12, fontweight='bold')
            ax5.legend(loc='best', fontsize=9)
            ax5.grid(True, alpha=0.3)

    # Panel F: Summary table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Build summary
    summary_data = []
    summary_colors = []

    # Latitudinal
    if 'correlation' in lat_results:
        corr = lat_results['correlation']
        sig = corr.get('p_value', 1) < 0.05
        summary_data.append(['Latitude ~ Size', f"r={corr.get('r', 0):.3f}",
                            f"{corr.get('p_value', 1):.2e}", 'Yes' if sig else 'No'])
        summary_colors.append(['lightgreen' if sig else 'lightcoral'] * 4)

    # Longitudinal
    if 'correlation' in lon_results:
        corr = lon_results['correlation']
        sig = corr.get('p_value', 1) < 0.05
        summary_data.append(['Longitude ~ Size', f"r={corr.get('r', 0):.3f}",
                            f"{corr.get('p_value', 1):.2e}", 'Yes' if sig else 'No'])
        summary_colors.append(['lightgreen' if sig else 'lightcoral'] * 4)

    # Elevation
    if 'size_elevation_correlation' in elev_results:
        corr = elev_results['size_elevation_correlation']
        sig = corr.get('p_value', 1) < 0.05
        summary_data.append(['Elevation ~ Size', f"r={corr.get('r', 0):.3f}",
                            f"{corr.get('p_value', 1):.2e}", 'Yes' if sig else 'No'])
        summary_colors.append(['lightgreen' if sig else 'lightcoral'] * 4)

    # Glacial
    if glac_results and 'mann_whitney_test' in glac_results:
        mw = glac_results['mann_whitney_test']
        sig = mw.get('significant', False)
        summary_data.append(['Glacial Effect', f"U={mw.get('U_statistic', 0):.0f}",
                            f"{mw.get('p_value', 1):.2e}", 'Yes' if sig else 'No'])
        summary_colors.append(['lightgreen' if sig else 'lightcoral'] * 4)

    if summary_data:
        table = ax6.table(cellText=summary_data,
                         colLabels=['Pattern', 'Statistic', 'p-value', 'Significant'],
                         cellColours=summary_colors,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('F) Hypothesis Test Summary', fontsize=12, fontweight='bold', y=0.95)

    fig.suptitle('Spatial Scaling Analysis - Comprehensive Summary',
                fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]


# ============================================================================
# DALTON 18KA VISUALIZATIONS
# ============================================================================

def plot_dalton_18ka_comparison(dalton_results, figsize=(16, 12), save_path=None):
    """
    Create comprehensive visualization of Dalton 18ka analysis results.

    Four-panel figure showing:
    - A) Density comparison (18ka glaciated vs unglaciated)
    - B) Power law CCDFs for both categories
    - C) Alpha values with confidence intervals
    - D) x_min sensitivity comparison

    Parameters
    ----------
    dalton_results : dict
        Output from run_dalton_18ka_analysis()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    colors = {'18ka_glaciated': '#1f77b4', '18ka_unglaciated': '#d62728'}

    # A) Density comparison bar chart
    density_comp = dalton_results.get('density_comparison', {})
    if density_comp:
        categories = ['18ka_glaciated', '18ka_unglaciated']
        densities = [density_comp.get(cat, {}).get('density_per_1000km2', 0) for cat in categories]
        n_lakes = [density_comp.get(cat, {}).get('n_lakes', 0) for cat in categories]

        bars = ax1.bar(range(len(categories)), densities,
                      color=[colors[c] for c in categories],
                      edgecolor='black', linewidth=1.5)

        for i, (bar, n, d) in enumerate(zip(bars, n_lakes, densities)):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{d:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    f'n={n:,}', ha='center', va='center', fontsize=10, color='white')

        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(['Within 18ka Ice', 'Outside 18ka Ice'], fontsize=12)
        ax1.set_ylabel('Lake Density (per 1,000 km²)', fontsize=12)
        ax1.set_title('A) Lake Density: 18ka Glaciated vs Unglaciated', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # B) Power law CCDFs
    lake_gdf = dalton_results.get('lake_gdf')
    if lake_gdf is not None and 'dalton_classification' in lake_gdf.columns:
        # Find area column (case-insensitive)
        area_col = None
        for col in lake_gdf.columns:
            if col.lower() in ['areasqkm', 'area_km2', 'area', 'lakeareasqkm']:
                area_col = col
                break
        if area_col is None:
            for col in lake_gdf.columns:
                if 'area' in col.lower() and lake_gdf[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    area_col = col
                    break
        if area_col is None:
            area_col = lake_gdf.select_dtypes(include=[np.number]).columns[0]

        for classification in ['18ka_glaciated', '18ka_unglaciated']:
            mask = lake_gdf['dalton_classification'] == classification
            areas = lake_gdf.loc[mask, area_col].dropna().values
            areas = areas[areas > 0]

            if len(areas) > 10:
                sorted_areas = np.sort(areas)[::-1]
                n = len(sorted_areas)
                # CCDF: P(X > x) - use (n-i)/n to avoid log(0)
                ccdf = (n - np.arange(n)) / n

                label = 'In 18ka ice' if classification == '18ka_glaciated' else 'Outside 18ka ice'
                ax2.loglog(sorted_areas, ccdf, 'o-', color=colors[classification],
                          alpha=0.7, markersize=3, label=f'{label} (n={len(areas):,})')

        ax2.set_xlabel('Lake Area (km²)', fontsize=12)
        ax2.set_ylabel('P(X > x) CCDF', fontsize=12)
        ax2.set_title('B) Lake Size Distributions (CCDF)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

    # C) Alpha comparison
    pl_results = dalton_results.get('power_law_results', [])
    if pl_results:
        classifications = [r['classification'] for r in pl_results]
        alphas = [r['alpha'] for r in pl_results]
        alpha_ses = [r['alpha_se'] for r in pl_results]

        x_pos = range(len(classifications))
        ax3.bar(x_pos, alphas, yerr=[1.96*se for se in alpha_ses],
               color=[colors.get(c, 'gray') for c in classifications],
               edgecolor='black', linewidth=1.5, capsize=5)

        # Add reference line for percolation theory
        ax3.axhline(y=2.05, color='green', linestyle='--', linewidth=2,
                   label='Percolation τ ≈ 2.05')

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['In 18ka ice', 'Outside 18ka'], fontsize=11)
        ax3.set_ylabel('Power Law Exponent (α)', fontsize=12)
        ax3.set_title('C) Power Law Exponents', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        # Add alpha values on bars
        for i, (a, se) in enumerate(zip(alphas, alpha_ses)):
            ax3.text(i, a + 0.1, f'α={a:.2f}±{se:.2f}', ha='center', fontsize=10)

    # D) x_min sensitivity
    if pl_results:
        for r in pl_results:
            classification = r['classification']
            xmin_sens = r.get('xmin_sensitivity', [])
            if xmin_sens:
                xmins = [x['xmin'] for x in xmin_sens]
                alphas_sens = [x['alpha'] for x in xmin_sens]

                label = 'In 18ka ice' if classification == '18ka_glaciated' else 'Outside 18ka'
                ax4.semilogx(xmins, alphas_sens, 'o-', color=colors.get(classification, 'gray'),
                           linewidth=2, markersize=6, label=label)

        ax4.axhline(y=2.05, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label='Percolation τ ≈ 2.05')
        ax4.set_xlabel('x_min (km²)', fontsize=12)
        ax4.set_ylabel('Power Law Exponent (α)', fontsize=12)
        ax4.set_title('D) x_min Sensitivity', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Dalton 18ka (LGM) Glacial Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_wisconsin_vs_dalton_comparison(comparison_results, figsize=(18, 10), save_path=None):
    """
    Visualize comparison between Wisconsin max extent and Dalton 18ka.

    Creates a multi-panel figure showing:
    - A) Area comparison
    - B) Lake counts by classification
    - C) Density comparison
    - D) Power law comparison

    Parameters
    ----------
    comparison_results : dict
        Output from compare_wisconsin_vs_dalton_18ka()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    comparison = comparison_results.get('comparison', {})
    lake_gdf = comparison_results.get('lake_gdf')

    colors = {
        'wisconsin': '#2ca02c',
        'dalton_18ka': '#1f77b4',
        'wisconsin_only': '#ff7f0e',
        'both': '#9467bd',
        'neither': '#d62728'
    }

    # A) Area comparison
    areas = {
        'Wisconsin\n(~20 ka max)': comparison.get('wisconsin', {}).get('area_km2', 0),
        'Dalton\n(18 ka precise)': comparison.get('dalton_18ka', {}).get('area_km2', 0),
        'Wisconsin\nmargins only': comparison.get('wisconsin_only', {}).get('area_km2', 0)
    }

    ax1.bar(range(len(areas)), list(areas.values()),
           color=['#2ca02c', '#1f77b4', '#ff7f0e'],
           edgecolor='black', linewidth=1.5)

    for i, (name, area) in enumerate(areas.items()):
        ax1.text(i, area, f'{area/1000:.0f}k km²', ha='center', va='bottom', fontsize=11)

    ax1.set_xticks(range(len(areas)))
    ax1.set_xticklabels(list(areas.keys()), fontsize=11)
    ax1.set_ylabel('Area (km²)', fontsize=12)
    ax1.set_title('A) Ice Extent Areas', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # B) Lake counts by classification
    if lake_gdf is not None and 'combined_classification' in lake_gdf.columns:
        class_counts = lake_gdf['combined_classification'].value_counts()

        class_labels = {
            'both': 'Both extents\n(core glaciated)',
            'wisconsin_only': 'Wisconsin only\n(marginal)',
            'neither': 'Neither\n(unglaciated)',
            'dalton_only': 'Dalton only\n(rare)'
        }

        classes = [c for c in ['both', 'wisconsin_only', 'neither', 'dalton_only'] if c in class_counts.index]
        counts = [class_counts[c] for c in classes]
        labels = [class_labels.get(c, c) for c in classes]

        ax2.bar(range(len(classes)), counts,
               color=[colors.get(c, 'gray') for c in classes],
               edgecolor='black', linewidth=1.5)

        for i, count in enumerate(counts):
            pct = 100 * count / len(lake_gdf)
            ax2.text(i, count, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel('Number of Lakes', fontsize=12)
        ax2.set_title('B) Lake Classification', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

    # C) Density comparison
    densities = {
        'Wisconsin': comparison.get('wisconsin', {}).get('density', 0),
        'Dalton 18ka': comparison.get('dalton_18ka', {}).get('density', 0),
        'Margins only': comparison.get('wisconsin_only', {}).get('density', 0)
    }

    ax3.bar(range(len(densities)), list(densities.values()),
           color=['#2ca02c', '#1f77b4', '#ff7f0e'],
           edgecolor='black', linewidth=1.5)

    for i, d in enumerate(densities.values()):
        ax3.text(i, d, f'{d:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax3.set_xticks(range(len(densities)))
    ax3.set_xticklabels(list(densities.keys()), fontsize=11)
    ax3.set_ylabel('Lakes per 1,000 km²', fontsize=12)
    ax3.set_title('C) Lake Density Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # D) Power law comparison
    pl_results = comparison_results.get('power_law_comparison', [])
    if pl_results:
        extents = [r['extent'] for r in pl_results]
        alphas = [r['alpha'] for r in pl_results]
        alpha_ses = [r['alpha_se'] for r in pl_results]

        x_labels = ['Wisconsin\n(~20 ka)' if e == 'wisconsin' else 'Dalton\n(18 ka)' for e in extents]
        bar_colors = [colors.get(e, 'gray') for e in extents]

        ax4.bar(range(len(extents)), alphas, yerr=[1.96*se for se in alpha_ses],
               color=bar_colors, edgecolor='black', linewidth=1.5, capsize=5)

        ax4.axhline(y=2.05, color='red', linestyle='--', linewidth=2,
                   label='Percolation τ ≈ 2.05')

        for i, (a, se) in enumerate(zip(alphas, alpha_ses)):
            ax4.text(i, a + 0.1, f'α={a:.2f}', ha='center', fontsize=11, fontweight='bold')

        ax4.set_xticks(range(len(extents)))
        ax4.set_xticklabels(x_labels, fontsize=11)
        ax4.set_ylabel('Power Law Exponent (α)', fontsize=12)
        ax4.set_title('D) Power Law Comparison', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Wisconsin Maximum vs Dalton 18ka Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_glacial_zone_xmin_sensitivity(xmin_results, figsize=(16, 12), save_path=None):
    """
    Visualize x_min sensitivity analysis by glacial zone.

    Creates a multi-panel figure showing:
    - Alpha vs x_min curves for each glacial zone
    - KS statistics by x_min
    - Optimal x_min comparison

    Parameters
    ----------
    xmin_results : dict
        Output from xmin_sensitivity_by_glacial_zone()
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    tuple (fig, axes)
    """
    setup_plot_style()

    zones = list(xmin_results.keys())
    if not zones:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig, [ax]

    n_zones = len(zones)
    n_cols = min(3, n_zones)
    n_rows = (n_zones + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_zones == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color map for zones
    zone_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_zones)))

    for i, (zone, data) in enumerate(xmin_results.items()):
        ax = axes[i]

        sensitivity_df = data.get('sensitivity')
        if sensitivity_df is None or len(sensitivity_df) == 0:
            ax.text(0.5, 0.5, f'{zone}\nNo data', ha='center', va='center')
            continue

        # Plot alpha vs x_min
        ax.semilogx(sensitivity_df['xmin'], sensitivity_df['alpha'],
                   'o-', color=zone_colors[i], linewidth=2, markersize=5)

        # Mark optimal x_min
        optimal_xmin = data.get('optimal_xmin')
        optimal_alpha = data.get('optimal_alpha')
        if optimal_xmin is not None and optimal_alpha is not None:
            ax.axvline(x=optimal_xmin, color='red', linestyle='--', alpha=0.7)
            ax.plot(optimal_xmin, optimal_alpha, 'r*', markersize=15,
                   label=f'Optimal: x_min={optimal_xmin:.3f}')

        # Reference line
        ax.axhline(y=2.05, color='green', linestyle=':', alpha=0.7,
                  label='τ ≈ 2.05')

        ax.set_xlabel('x_min (km²)', fontsize=10)
        ax.set_ylabel('α', fontsize=10)
        ax.set_title(f'{zone}\n(n={data.get("n_lakes", 0):,})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_zones, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('x_min Sensitivity by Glacial Zone', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


# ============================================================================
# ARIDITY INDEX ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_aridity_lake_density(aridity_stats, figsize=(12, 8), save_path=None):
    """
    Plot lake density by aridity index bin.

    Parameters
    ----------
    aridity_stats : DataFrame
        Output from compute_density_by_aridity()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    if aridity_stats is None or len(aridity_stats) == 0:
        ax.text(0.5, 0.5, 'No aridity data available',
               ha='center', va='center', fontsize=14)
        return fig, ax

    # Color gradient from red (arid) to blue (humid)
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(aridity_stats)))

    bars = ax.bar(range(len(aridity_stats)),
                  aridity_stats['n_lakes'],
                  color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(aridity_stats)))
    ax.set_xticklabels(aridity_stats['ai_label'], rotation=45, ha='right')
    ax.set_ylabel('Number of Lakes', fontsize=12)
    ax.set_xlabel('Aridity Index Category', fontsize=12)
    ax.set_title('Lake Distribution by Aridity Index', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, count in zip(bars, aridity_stats['n_lakes']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{count:,}', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_aridity_glacial_heatmap(cross_results, figsize=(14, 10), save_path=None):
    """
    Plot heatmap of lake counts by aridity × glacial stage.

    Parameters
    ----------
    cross_results : dict
        Output from compute_density_by_aridity_and_glacial()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if cross_results is None:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig, axes

    counts = cross_results['counts']
    proportions = cross_results['proportions']

    # Remove margins for heatmap
    counts_data = counts.drop('All', axis=1).drop('All', axis=0) if 'All' in counts.columns else counts
    props_data = proportions.drop('All', axis=1).drop('All', axis=0) if 'All' in proportions.columns else proportions

    # AI bin labels
    ai_labels = {
        0: 'Hyper-arid',
        1: 'Arid',
        2: 'Semi-arid',
        3: 'Dry sub-humid',
        4: 'Humid',
        5: 'Wet',
        6: 'Hyper-humid'
    }
    row_labels = [ai_labels.get(i, f'Bin {i}') for i in counts_data.index]

    # Panel A: Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(counts_data.values, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(counts_data.columns)))
    ax1.set_xticklabels(counts_data.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(row_labels)))
    ax1.set_yticklabels(row_labels)
    ax1.set_title('A) Lake Counts', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Count')

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(counts_data.columns)):
            val = counts_data.values[i, j]
            color = 'white' if val > counts_data.values.max() * 0.5 else 'black'
            ax1.text(j, i, f'{val:,.0f}', ha='center', va='center', fontsize=8, color=color)

    # Panel B: Proportions
    ax2 = axes[1]
    im2 = ax2.imshow(props_data.values, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(props_data.columns)))
    ax2.set_xticklabels(props_data.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels(row_labels)
    ax2.set_title('B) Percentage by Glacial Stage', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Percent')

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(props_data.columns)):
            val = props_data.values[i, j]
            color = 'white' if val > 50 else 'black'
            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=8, color=color)

    fig.suptitle('Lake Distribution: Aridity × Glacial Stage', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_aridity_glacial_comparison(comparison_results, figsize=(18, 14), save_path=None):
    """
    Create comprehensive multi-panel figure comparing aridity vs glacial stage effects.

    Parameters
    ----------
    comparison_results : dict
        Output from run_aridity_glacial_comparison()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    setup_plot_style()
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel A: Lake counts by aridity
    ax1 = fig.add_subplot(gs[0, 0])
    aridity_stats = comparison_results.get('aridity_stats')
    if aridity_stats is not None and len(aridity_stats) > 0:
        colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(aridity_stats)))
        bars = ax1.bar(range(len(aridity_stats)), aridity_stats['n_lakes'],
                      color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xticks(range(len(aridity_stats)))
        ax1.set_xticklabels([s[:10] for s in aridity_stats['ai_label']], rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Lake Count')
        ax1.set_title('A) Lakes by Aridity Index', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Panel B: Aridity × Glacial heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    cross_results = comparison_results.get('cross_tabulation')
    if cross_results is not None and cross_results.get('proportions') is not None:
        props = cross_results['proportions']
        if 'All' in props.columns:
            props = props.drop('All', axis=1)
        if 'All' in props.index:
            props = props.drop('All', axis=0)

        im = ax2.imshow(props.values, cmap='Blues', aspect='auto', vmin=0, vmax=100)
        ax2.set_xticks(range(len(props.columns)))
        ax2.set_xticklabels(props.columns, rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(props.index)))
        ai_labels = ['Hyper-arid', 'Arid', 'Semi-arid', 'Dry sub-humid', 'Humid', 'Wet', 'Hyper-humid']
        ax2.set_yticklabels([ai_labels[i] if i < len(ai_labels) else f'Bin {i}' for i in props.index], fontsize=8)
        ax2.set_title('B) % by Glacial Stage per Aridity Bin', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='%', shrink=0.8)

    # Panel C: ANOVA results by aridity bin
    ax3 = fig.add_subplot(gs[1, 0])
    anova_df = comparison_results.get('anova_by_aridity')
    if anova_df is not None and len(anova_df) > 0:
        # Plot F-statistics
        bars = ax3.bar(range(len(anova_df)), anova_df['f_statistic'], color='steelblue', edgecolor='black')
        ax3.set_xticks(range(len(anova_df)))
        ai_labels = ['Hyper-arid', 'Arid', 'Semi-arid', 'Dry sub-humid', 'Humid', 'Wet', 'Hyper-humid']
        ax3.set_xticklabels([ai_labels[int(i)] if int(i) < len(ai_labels) else f'Bin {i}' for i in anova_df['ai_bin']],
                          rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('F-statistic')
        ax3.set_title('C) ANOVA: Glacial Stage Effect within Aridity Bins', fontsize=11, fontweight='bold')

        # Add significance markers
        for i, (bar, pval) in enumerate(zip(bars, anova_df['p_value'])):
            marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    marker, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

    # Panel D: R² comparison
    ax4 = fig.add_subplot(gs[1, 1])
    r2_ai = comparison_results.get('r2_aridity_only', 0)
    r2_glacial = comparison_results.get('r2_glacial_only', 0)
    reg = comparison_results.get('regression', {})
    r2_both = reg.get('r_squared', 0) if reg else 0

    r2_values = [r2_ai, r2_glacial, r2_both]
    labels = ['Aridity\nOnly', 'Glacial Stage\nOnly', 'Both\nCombined']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    bars = ax4.bar(range(3), r2_values, color=colors, edgecolor='black', linewidth=1)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('R² (Variance Explained)')
    ax4.set_title('D) Predictive Power Comparison', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, max(r2_values) * 1.3 if max(r2_values) > 0 else 0.1)

    for bar, val in zip(bars, r2_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    # Panel E: Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    summary_text = "SUMMARY: Aridity vs Glacial Stage as Lake Density Predictors\n"
    summary_text += "=" * 70 + "\n\n"

    if r2_glacial > r2_ai:
        winner = "GLACIAL STAGE"
        ratio = (r2_glacial / r2_ai - 1) * 100 if r2_ai > 0 else float('inf')
        summary_text += f"→ {winner} explains {ratio:.1f}% more variance than aridity alone\n"
    else:
        winner = "ARIDITY"
        ratio = (r2_ai / r2_glacial - 1) * 100 if r2_glacial > 0 else float('inf')
        summary_text += f"→ {winner} explains {ratio:.1f}% more variance than glacial stage alone\n"

    summary_text += f"\nR² Aridity only: {r2_ai:.4f}\n"
    summary_text += f"R² Glacial stage only: {r2_glacial:.4f}\n"
    summary_text += f"R² Both combined: {r2_both:.4f}\n"

    if anova_df is not None and len(anova_df) > 0:
        sig_bins = (anova_df['p_value'] < 0.05).sum()
        summary_text += f"\nANOVA: Glacial stage significant in {sig_bins}/{len(anova_df)} aridity bins\n"
        summary_text += "This suggests glacial history effects persist after controlling for climate."

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=11, fontfamily='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Comparing Controls on Lake Density: Climate vs Glacial History',
                fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, None


# ============================================================================
# NADI-1 CHRONOSEQUENCE VISUALIZATION
# ============================================================================

def plot_nadi1_chronosequence(results, save_path=None, show_uncertainty=True):
    """
    Plot lake density vs deglaciation age from NADI-1 analysis.

    Creates a multi-panel figure showing:
    - Lake count by deglaciation age with decay model fit
    - Uncertainty bounds from MIN/MAX extents
    - Comparison with Illinoian and Driftless reference points

    Parameters
    ----------
    results : dict
        Output from run_nadi1_chronosequence_analysis()
    save_path : str, optional
        Path to save the figure.
    show_uncertainty : bool, optional
        If True, show MIN/MAX uncertainty bands.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    setup_plot_style()

    density_by_age = results.get('density_by_age')
    decay_model = results.get('decay_model')
    uncertainty = results.get('uncertainty', {})

    if density_by_age is None or len(density_by_age) == 0:
        print("No density data available for plotting")
        return None

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel A: Lake count by deglaciation age
    ax1 = fig.add_subplot(gs[0, 0])

    x = density_by_age['age_midpoint_ka'].values
    y = density_by_age['n_lakes'].values

    # Scatter plot of actual data
    ax1.scatter(x, y, s=100, c='steelblue', alpha=0.8, edgecolors='black',
                linewidths=1.5, zorder=5, label='Observed')

    # Fit line if decay model available
    if decay_model and decay_model.get('n0') is not None:
        x_fit = np.linspace(0, max(x) * 1.1, 100)
        n0 = decay_model['n0']
        lam = decay_model['lambda']
        y_fit = n0 * np.exp(-lam * x_fit)
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label='Exponential fit', zorder=3)

        # Add half-life annotation
        half_life = decay_model.get('half_life_ka', np.inf)
        if np.isfinite(half_life):
            ax1.axhline(n0/2, color='gray', linestyle='--', alpha=0.5, zorder=1)
            ax1.axvline(half_life, color='gray', linestyle='--', alpha=0.5, zorder=1)
            ax1.annotate(f't₁/₂ = {half_life:.1f} ka',
                        xy=(half_life, n0/2),
                        xytext=(half_life + 2, n0/2 + y.max()*0.1),
                        fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax1.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax1.set_ylabel('Number of Lakes', fontsize=12)
    ax1.set_title('A) Lake Count vs Deglaciation Age', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(x) * 1.1)

    # Panel B: Total lake area by age
    ax2 = fig.add_subplot(gs[0, 1])

    y_area = density_by_age['total_area_km2'].values
    ax2.scatter(x, y_area, s=100, c='forestgreen', alpha=0.8, edgecolors='black',
                linewidths=1.5, zorder=5)

    ax2.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax2.set_ylabel('Total Lake Area (km²)', fontsize=12)
    ax2.set_title('B) Total Lake Area vs Deglaciation Age', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel C: Mean lake size by age
    ax3 = fig.add_subplot(gs[1, 0])

    y_mean = density_by_age['mean_area_km2'].values
    ax3.scatter(x, y_mean, s=100, c='darkorange', alpha=0.8, edgecolors='black',
                linewidths=1.5, zorder=5)

    ax3.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax3.set_ylabel('Mean Lake Area (km²)', fontsize=12)
    ax3.set_title('C) Mean Lake Size vs Deglaciation Age', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add trend line if significant
    if len(x) >= 3:
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, _ = stats.linregress(x, y_mean)
            if p_value < 0.05:
                x_trend = np.array([min(x), max(x)])
                y_trend = slope * x_trend + intercept
                ax3.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.7,
                        label=f'Trend (p={p_value:.3f})')
                ax3.legend()
        except Exception:
            pass

    # Panel D: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Build summary text
    lake_gdf = results.get('lake_gdf')
    summary_lines = ["NADI-1 CHRONOSEQUENCE SUMMARY", "=" * 40, ""]

    if lake_gdf is not None:
        n_total = len(lake_gdf)
        n_glaciated = lake_gdf['was_glaciated'].sum()
        n_assigned = lake_gdf['deglaciation_age'].notna().sum()

        summary_lines.extend([
            f"Total lakes analyzed: {n_total:,}",
            f"Were glaciated (Wisconsin): {n_glaciated:,} ({100*n_glaciated/n_total:.1f}%)",
            f"With deglaciation ages: {n_assigned:,}",
            ""
        ])

    if decay_model:
        summary_lines.extend([
            "Exponential Decay Model:",
            f"  N(t) = {decay_model.get('n0', 0):.0f} × e^(-{decay_model.get('lambda', 0):.4f} × t)",
            f"  Half-life: {decay_model.get('half_life_ka', np.inf):.1f} ka",
            f"  R²: {decay_model.get('r2', 0):.3f}",
            ""
        ])

    if uncertainty:
        summary_lines.extend([
            "Uncertainty from MIN/MAX extents:",
        ])
        for ext, data in uncertainty.items():
            summary_lines.append(f"  {ext}: mean age = {data.get('mean_age_ka', 0):.1f} ± {data.get('std_age_ka', 0):.1f} ka")

    summary_lines.extend([
        "",
        "Note: LGM ≈ 20 ka, not 25 ka",
        "Continental ice only (east of -110°)"
    ])

    summary_text = "\n".join(summary_lines)

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    fig.suptitle('Lake Density Decay with Deglaciation Age (NADI-1 Analysis)',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_deglaciation_age_histogram(lake_gdf, save_path=None):
    """
    Plot histogram of deglaciation ages.

    Parameters
    ----------
    lake_gdf : gpd.GeoDataFrame
        Lakes with deglaciation_age column.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    ages = lake_gdf['deglaciation_age'].dropna()

    if len(ages) == 0:
        print("No deglaciation ages to plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(ages, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(ages.median(), color='red', linestyle='--', linewidth=2,
                label=f'Median = {ages.median():.1f} ka')
    ax1.axvline(20, color='orange', linestyle=':', linewidth=2,
                label='LGM (~20 ka)')
    ax1.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax1.set_ylabel('Number of Lakes', fontsize=12)
    ax1.set_title('Distribution of Deglaciation Ages', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_ages = np.sort(ages)
    cumulative = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages)
    ax2.plot(sorted_ages, cumulative, 'b-', linewidth=2)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(ages.median(), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax2.set_ylabel('Cumulative Fraction of Lakes', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Deglaciation Timing of Continental US Lakes',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_bayesian_posteriors(bayesian_results, save_path=None):
    """
    Plot Bayesian posterior distributions for decay model parameters.

    Parameters
    ----------
    bayesian_results : dict
        Output from fit_bayesian_decay_model()
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    if bayesian_results is None:
        print("No Bayesian results to plot")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # D0 posterior
    ax = axes[0, 0]
    D0_samples = bayesian_results['D0']['samples']
    ax.hist(D0_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(bayesian_results['D0']['mean'], color='red', linestyle='-', linewidth=2,
               label=f"Mean: {bayesian_results['D0']['mean']:.1f}")
    ax.axvline(bayesian_results['D0']['ci_lower'], color='red', linestyle='--', alpha=0.7)
    ax.axvline(bayesian_results['D0']['ci_upper'], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('D₀ (initial lake count)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Posterior: Initial Lake Density', fontsize=12, fontweight='bold')
    ax.legend()

    # Half-life posterior
    ax = axes[0, 1]
    hl_samples = bayesian_results['half_life']['samples']
    ax.hist(hl_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(bayesian_results['half_life']['mean'], color='red', linestyle='-', linewidth=2,
               label=f"Mean: {bayesian_results['half_life']['mean']:.0f} ka")
    ax.axvline(bayesian_results['half_life']['ci_lower'], color='red', linestyle='--', alpha=0.7)
    ax.axvline(bayesian_results['half_life']['ci_upper'], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Half-life (ka)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Posterior: Lake Density Half-life', fontsize=12, fontweight='bold')
    ax.legend()

    # k posterior
    ax = axes[1, 0]
    k_samples = bayesian_results['k']['samples']
    ax.hist(k_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(bayesian_results['k']['mean'], color='red', linestyle='-', linewidth=2,
               label=f"Mean: {bayesian_results['k']['mean']:.6f}")
    ax.set_xlabel('Decay rate k (per ka)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Posterior: Decay Rate', fontsize=12, fontweight='bold')
    ax.legend()

    # k vs half-life correlation
    ax = axes[1, 1]
    ax.scatter(k_samples[::10], hl_samples[::10], alpha=0.3, s=5, color='steelblue')
    ax.set_xlabel('Decay rate k (per ka)', fontsize=11)
    ax.set_ylabel('Half-life (ka)', fontsize=11)
    ax.set_title('Posterior: k vs Half-life\n(inherent inverse relationship)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_bayesian_decay_curves(bayesian_results, save_path=None):
    """
    Plot Bayesian posterior predictive decay curves with credible intervals.

    Parameters
    ----------
    bayesian_results : dict
        Output from fit_bayesian_decay_model()
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    if bayesian_results is None or 'curves' not in bayesian_results:
        print("No Bayesian curves to plot")
        return None

    curves = bayesian_results['curves']
    data = bayesian_results['data']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax = axes[0]
    age_grid = curves['age_grid']

    # Credible intervals
    ax.fill_between(age_grid, curves['ci_lower_95'], curves['ci_upper_95'],
                    color='steelblue', alpha=0.2, label='95% CI')
    ax.fill_between(age_grid, curves['ci_lower_50'], curves['ci_upper_50'],
                    color='steelblue', alpha=0.4, label='50% CI')
    ax.plot(age_grid, curves['median'], '-', color='darkblue', linewidth=2, label='Median')

    # Plot data points with error bars
    stages = data['stages']
    densities = data['densities']
    ages_point = data['ages_point']
    ages_lower = data['ages_lower']
    ages_upper = data['ages_upper']
    density_sigma = data['density_sigma']

    for i, stage in enumerate(stages):
        xerr_low = ages_point[i] - ages_lower[i]
        xerr_high = ages_upper[i] - ages_point[i]
        color = '#B2182B' if 'driftless' in stage.lower() else '#2166AC'
        ax.errorbar(ages_point[i], densities[i],
                    xerr=[[xerr_low], [xerr_high]],
                    yerr=density_sigma[i],
                    fmt='o', markersize=10,
                    color=color, capsize=5,
                    markeredgecolor='black', markeredgewidth=1.5,
                    elinewidth=1.5, zorder=10)

    ax.set_xlabel('Landscape Age (ka)', fontsize=12)
    ax.set_ylabel('Lake Density', fontsize=12)
    ax.set_title('Bayesian Posterior: Exponential Decay Model\n(Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, max(ages_point) * 1.2)
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    mask = age_grid > 5

    ax.fill_between(age_grid[mask], curves['ci_lower_95'][mask], curves['ci_upper_95'][mask],
                    color='steelblue', alpha=0.2, label='95% CI')
    ax.fill_between(age_grid[mask], curves['ci_lower_50'][mask], curves['ci_upper_50'][mask],
                    color='steelblue', alpha=0.4, label='50% CI')
    ax.plot(age_grid[mask], curves['median'][mask], '-', color='darkblue', linewidth=2, label='Median')

    for i, stage in enumerate(stages):
        xerr_low = ages_point[i] - ages_lower[i]
        xerr_high = ages_upper[i] - ages_point[i]
        color = '#B2182B' if 'driftless' in stage.lower() else '#2166AC'
        ax.errorbar(ages_point[i], densities[i],
                    xerr=[[xerr_low], [xerr_high]],
                    yerr=density_sigma[i],
                    fmt='o', markersize=10,
                    color=color, capsize=5,
                    markeredgecolor='black', markeredgewidth=1.5,
                    elinewidth=1.5, zorder=10)

    ax.set_xscale('log')
    ax.set_xlabel('Landscape Age (ka, log scale)', fontsize=12)
    ax.set_ylabel('Lake Density', fontsize=12)
    ax.set_title('Bayesian Posterior: Exponential Decay Model\n(Log Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_bayesian_covariance(bayesian_results, save_path=None):
    """
    Plot covariance between Driftless age and half-life posteriors.

    Parameters
    ----------
    bayesian_results : dict
        Output from fit_bayesian_decay_model()
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    if bayesian_results is None:
        print("No Bayesian results to plot")
        return None

    # Check if we have Driftless age
    driftless_key = None
    for key in bayesian_results['age_posteriors'].keys():
        if 'driftless' in key.lower():
            driftless_key = key
            break

    if driftless_key is None:
        print("No Driftless age in results")
        return None

    driftless_samples = bayesian_results['age_posteriors'][driftless_key]['samples']
    hl_samples = bayesian_results['half_life']['samples']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter with density coloring
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([driftless_samples, hl_samples])
        z = gaussian_kde(xy)(xy)

        idx_sort = z.argsort()
        x_sorted = driftless_samples[idx_sort]
        y_sorted = hl_samples[idx_sort]
        z_sorted = z[idx_sort]

        scatter = ax.scatter(x_sorted[::10], y_sorted[::10], c=z_sorted[::10],
                            s=15, alpha=0.5, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Density')
    except ImportError:
        ax.scatter(driftless_samples[::5], hl_samples[::5], alpha=0.2, s=10, color='steelblue')

    ax.set_xlabel('Inferred Driftless Age (ka)', fontsize=12)
    ax.set_ylabel('Inferred Half-life (ka)', fontsize=12)
    ax.set_title('Posterior Covariance: Driftless Age vs Half-life\n(Older Driftless → Longer Half-life)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = bayesian_results.get('driftless_halflife_correlation',
                                np.corrcoef(driftless_samples, hl_samples)[0, 1])
    ax.annotate(f'Correlation: r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold', va='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_bayesian_summary(bayesian_results, save_path=None):
    """
    Plot comprehensive Bayesian analysis summary (4-panel figure).

    Parameters
    ----------
    bayesian_results : dict
        Output from fit_bayesian_decay_model()
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    if bayesian_results is None:
        print("No Bayesian results to plot")
        return None

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    curves = bayesian_results['curves']
    data = bayesian_results['data']

    # Panel A: Decay curves with credible intervals
    ax = fig.add_subplot(gs[0, 0])
    age_grid = curves['age_grid']

    ax.fill_between(age_grid, curves['ci_lower_95'], curves['ci_upper_95'],
                    color='steelblue', alpha=0.2, label='95% CI')
    ax.fill_between(age_grid, curves['ci_lower_50'], curves['ci_upper_50'],
                    color='steelblue', alpha=0.4, label='50% CI')
    ax.plot(age_grid, curves['median'], '-', color='darkblue', linewidth=2, label='Median')

    for i, stage in enumerate(data['stages']):
        xerr_low = data['ages_point'][i] - data['ages_lower'][i]
        xerr_high = data['ages_upper'][i] - data['ages_point'][i]
        ax.errorbar(data['ages_point'][i], data['densities'][i],
                    xerr=[[xerr_low], [xerr_high]],
                    yerr=data['density_sigma'][i],
                    fmt='o', markersize=8, color='#2166AC', capsize=4,
                    markeredgecolor='black', markeredgewidth=1, elinewidth=1.5, zorder=10)

    ax.set_xlabel('Landscape Age (ka)', fontsize=11)
    ax.set_ylabel('Lake Density', fontsize=11)
    ax.set_title('A) Bayesian Decay Model', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Half-life posterior
    ax = fig.add_subplot(gs[0, 1])
    hl_samples = bayesian_results['half_life']['samples']
    ax.hist(hl_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(bayesian_results['half_life']['mean'], color='red', linestyle='-', linewidth=2,
               label=f"Mean: {bayesian_results['half_life']['mean']:.0f} ka")
    ax.axvline(bayesian_results['half_life']['ci_lower'], color='red', linestyle='--', alpha=0.7)
    ax.axvline(bayesian_results['half_life']['ci_upper'], color='red', linestyle='--', alpha=0.7,
               label=f"95% CI: [{bayesian_results['half_life']['ci_lower']:.0f}, {bayesian_results['half_life']['ci_upper']:.0f}]")
    ax.set_xlabel('Half-life (ka)', fontsize=11)
    ax.set_ylabel('Posterior Density', fontsize=11)
    ax.set_title('B) Half-life Posterior', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel C: D0 posterior
    ax = fig.add_subplot(gs[1, 0])
    D0_samples = bayesian_results['D0']['samples']
    ax.hist(D0_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(bayesian_results['D0']['mean'], color='red', linestyle='-', linewidth=2,
               label=f"Mean: {bayesian_results['D0']['mean']:.1f}")
    ax.set_xlabel('D₀ (initial density)', fontsize=11)
    ax.set_ylabel('Posterior Density', fontsize=11)
    ax.set_title('C) Initial Density Posterior', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel D: Summary text
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    summary_lines = [
        "BAYESIAN DECAY MODEL SUMMARY",
        "=" * 40,
        "",
        "Posterior Estimates (mean [95% CI]):",
        f"  D₀: {bayesian_results['D0']['mean']:.1f} [{bayesian_results['D0']['ci_lower']:.1f}, {bayesian_results['D0']['ci_upper']:.1f}]",
        f"  k:  {bayesian_results['k']['mean']:.6f} [{bayesian_results['k']['ci_lower']:.6f}, {bayesian_results['k']['ci_upper']:.6f}] /ka",
        f"  Half-life: {bayesian_results['half_life']['mean']:.0f} [{bayesian_results['half_life']['ci_lower']:.0f}, {bayesian_results['half_life']['ci_upper']:.0f}] ka",
        "",
        "Model:",
        "  D(t) = D₀ × exp(-k × t)",
        "",
        "Interpretation:",
        f"  Lake density decays by 50% every",
        f"  ~{bayesian_results['half_life']['mean']:.0f} thousand years.",
    ]

    if 'driftless_halflife_correlation' in bayesian_results:
        corr = bayesian_results['driftless_halflife_correlation']
        summary_lines.extend([
            "",
            f"Age-Halflife Correlation: r = {corr:.3f}",
            "  (Older assumed ages → longer half-life)"
        ])

    summary_text = "\n".join(summary_lines)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    fig.suptitle('Bayesian Analysis of Lake Density Decay',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_glaciated_area_timeseries(area_df, save_path=None):
    """
    Plot glaciated area over time with MIN/MAX uncertainty.

    Creates a time series plot showing how glaciated area changed over time,
    with error bars from MIN and MAX ice extent reconstructions.

    Parameters
    ----------
    area_df : pd.DataFrame
        Output from compute_glaciated_area_timeseries() with columns:
        - age_ka: Time in ka
        - area_min_km2, area_max_km2, area_optimal_km2
        - area_error_km2
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    if area_df is None or len(area_df) == 0:
        print("No area data available for plotting")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Glaciated area time series
    ax = axes[0]
    x = area_df['age_ka'].values
    y_opt = area_df['area_optimal_km2'].values / 1e6  # Convert to million km²
    y_min = area_df['area_min_km2'].values / 1e6
    y_max = area_df['area_max_km2'].values / 1e6

    # Fill between MIN and MAX
    ax.fill_between(x, y_min, y_max, alpha=0.3, color='steelblue',
                    label='MIN-MAX range')

    # Plot OPTIMAL line
    ax.plot(x, y_opt, 'b-', linewidth=2, marker='o', markersize=4,
            label='OPTIMAL', zorder=5)

    # Mark LGM
    lgm_idx = np.argmax(y_opt)
    lgm_age = x[lgm_idx]
    lgm_area = y_opt[lgm_idx]
    ax.annotate(f'LGM\n{lgm_age:.0f} ka\n{lgm_area:.2f} M km²',
                xy=(lgm_age, lgm_area), xytext=(lgm_age + 2, lgm_area * 0.9),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    ax.set_xlabel('Time (ka BP)', fontsize=12)
    ax.set_ylabel('Continental Ice Area (million km²)', fontsize=12)
    ax.set_title('A) Laurentide Ice Sheet Area Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(x) + 1)
    ax.set_ylim(0, max(y_max) * 1.1)
    ax.invert_xaxis()  # Older ages on left

    # Panel B: Rate of deglaciation
    ax = axes[1]

    # Calculate rate of area change (km²/ka)
    if len(x) > 1:
        # Sort by age (young to old)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y_opt[sort_idx]

        # Calculate rate: dA/dt
        rate = np.diff(y_sorted) / np.diff(x_sorted)  # million km²/ka
        x_rate = (x_sorted[:-1] + x_sorted[1:]) / 2

        ax.bar(x_rate, -rate, width=0.4, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Mark peak deglaciation rate
        peak_idx = np.argmax(-rate)
        peak_rate = -rate[peak_idx]
        peak_age = x_rate[peak_idx]
        ax.annotate(f'Peak: {peak_rate:.3f} M km²/ka\nat {peak_age:.0f} ka',
                    xy=(peak_age, peak_rate), xytext=(peak_age + 3, peak_rate * 1.1),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    ax.set_xlabel('Time (ka BP)', fontsize=12)
    ax.set_ylabel('Deglaciation Rate (million km²/ka)', fontsize=12)
    ax.set_title('B) Rate of Ice Sheet Retreat', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle('NADI-1 Ice Sheet Reconstruction (Continental Ice, East of -110°)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_density_with_uncertainty(density_df, save_path=None, fit_model=True):
    """
    Plot lake density vs deglaciation age with MIN/MAX uncertainty.

    Creates a plot showing lake density (lakes per 1000 km²) over time,
    with proper error bars from MIN and MAX ice extent uncertainties.

    Parameters
    ----------
    density_df : pd.DataFrame
        Output from compute_density_by_deglaciation_age_with_area() with columns:
        - age_midpoint_ka: Time bin midpoint in ka
        - density_opt, density_min, density_max: Lake densities
        - density_error: Half-width of MIN-MAX range
        - n_lakes: Number of lakes per bin
        - landscape_area_km2_opt: Landscape area per bin
    save_path : str, optional
        Path to save the figure.
    fit_model : bool
        If True, fit and plot exponential decay model.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    setup_plot_style()

    if density_df is None or len(density_df) == 0:
        print("No density data available for plotting")
        return None

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel A: Lake density vs age with error bars
    ax = fig.add_subplot(gs[0, 0])

    x = density_df['age_midpoint_ka'].values
    y = density_df['density_opt'].values
    y_min = density_df['density_min'].values
    y_max = density_df['density_max'].values
    yerr_lower = y - y_min
    yerr_upper = y_max - y

    # Plot with asymmetric error bars
    ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper],
                fmt='o', markersize=10, color='steelblue', capsize=5,
                markeredgecolor='black', markeredgewidth=1.5, elinewidth=2,
                label='Observed density', zorder=5)

    # Fit exponential decay if requested
    if fit_model and len(x) >= 3:
        try:
            def exp_decay(t, D0, k):
                return D0 * np.exp(-k * t)

            # Weight by inverse variance
            valid = ~np.isnan(y) & ~np.isnan(y_min) & ~np.isnan(y_max)
            x_fit = x[valid]
            y_fit = y[valid]
            yerr_fit = (y_max[valid] - y_min[valid]) / 2
            yerr_fit = np.maximum(yerr_fit, 0.01)  # Avoid zero weights

            popt, pcov = curve_fit(exp_decay, x_fit, y_fit,
                                   p0=[max(y_fit), 0.1],
                                   sigma=yerr_fit,
                                   absolute_sigma=True,
                                   bounds=([0, 0], [np.inf, 1]))

            D0_fit, k_fit = popt
            half_life = np.log(2) / k_fit

            # Plot fit
            x_model = np.linspace(0, max(x) * 1.2, 100)
            y_model = exp_decay(x_model, D0_fit, k_fit)
            ax.plot(x_model, y_model, 'r-', linewidth=2, zorder=3,
                    label=f'Fit: D₀={D0_fit:.1f}, t½={half_life:.1f} ka')

            # Half-life marker
            ax.axhline(D0_fit/2, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(half_life, color='gray', linestyle='--', alpha=0.5)

        except Exception as e:
            print(f"Warning: Could not fit exponential decay: {e}")

    ax.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax.set_ylabel('Lake Density (lakes per 1000 km²)', fontsize=12)
    ax.set_title('A) Lake Density Decay with Age', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(x) * 1.15)
    ax.set_ylim(0, max(y_max) * 1.1 if np.any(~np.isnan(y_max)) else None)

    # Panel B: Landscape area deglaciated per bin
    ax = fig.add_subplot(gs[0, 1])

    area_opt = density_df['landscape_area_km2_opt'].values / 1e3  # thousand km²
    area_min = density_df['landscape_area_km2_min'].values / 1e3
    area_max = density_df['landscape_area_km2_max'].values / 1e3

    bar_width = (x.max() - x.min()) / (len(x) * 1.5) if len(x) > 1 else 2

    ax.bar(x, area_opt, width=bar_width, color='forestgreen', edgecolor='black',
           alpha=0.7, label='OPTIMAL')
    ax.errorbar(x, area_opt, yerr=[area_opt - area_min, area_max - area_opt],
                fmt='none', color='black', capsize=4, elinewidth=1.5)

    ax.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax.set_ylabel('Landscape Area Deglaciated (thousand km²)', fontsize=12)
    ax.set_title('B) Landscape Area per Time Bin', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Number of lakes per bin
    ax = fig.add_subplot(gs[1, 0])

    n_lakes = density_df['n_lakes'].values

    ax.bar(x, n_lakes, width=bar_width, color='steelblue', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Deglaciation Age (ka BP)', fontsize=12)
    ax.set_ylabel('Number of Lakes', fontsize=12)
    ax.set_title('C) Lake Count per Time Bin', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel D: Summary table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    summary_lines = [
        "LAKE DENSITY ANALYSIS SUMMARY",
        "=" * 45,
        "",
        "Age Bin    N Lakes    Area (km²)    Density",
        "-" * 45,
    ]

    for _, row in density_df.iterrows():
        density_str = f"{row['density_opt']:.2f}" if not np.isnan(row['density_opt']) else "N/A"
        summary_lines.append(
            f"{row['age_bin']:<10} {row['n_lakes']:>8,}   {row['landscape_area_km2_opt']:>11,.0f}   {density_str:>8}"
        )

    summary_lines.extend([
        "-" * 45,
        f"Total:     {density_df['n_lakes'].sum():>8,}   {density_df['landscape_area_km2_opt'].sum():>11,.0f}",
        "",
        "Note: Density = lakes per 1000 km²",
        "Error bars show MIN-MAX ice extent uncertainty"
    ])

    summary_text = "\n".join(summary_lines)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    fig.suptitle('Lake Density by Deglaciation Age (NADI-1 Time Slices)',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_nadi1_density_decay(density_df, bayesian_results=None, save_path=None):
    """
    Plot lake density decay with NADI-1 time slice data and Bayesian model fit.

    Creates a publication-quality figure showing:
    - Lake density (per 1000 km²) vs deglaciation age
    - MIN/MAX uncertainty from ice extent reconstructions
    - Bayesian credible intervals on decay curve

    Parameters
    ----------
    density_df : pd.DataFrame
        Output from compute_density_by_deglaciation_age_with_area()
    bayesian_results : dict, optional
        Output from fit_bayesian_decay_model()
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    if density_df is None or len(density_df) == 0:
        print("No density data available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    x = density_df['age_midpoint_ka'].values
    y = density_df['density_opt'].values
    y_min = density_df['density_min'].values
    y_max = density_df['density_max'].values
    yerr_lower = y - y_min
    yerr_upper = y_max - y

    # Plot Bayesian credible intervals if available
    if bayesian_results is not None and 'curves' in bayesian_results:
        curves = bayesian_results['curves']
        age_grid = curves['age_grid']

        # 95% credible interval
        ax.fill_between(age_grid, curves['ci_95_lower'], curves['ci_95_upper'],
                        alpha=0.15, color='red', label='95% credible interval')

        # 50% credible interval
        ax.fill_between(age_grid, curves['ci_50_lower'], curves['ci_50_upper'],
                        alpha=0.25, color='red', label='50% credible interval')

        # Median curve
        ax.plot(age_grid, curves['median'], 'r-', linewidth=2,
                label='Posterior median', zorder=4)

        # Half-life annotation
        hl = bayesian_results['half_life']['mean']
        hl_lower = bayesian_results['half_life']['ci_lower']
        hl_upper = bayesian_results['half_life']['ci_upper']
        D0 = bayesian_results['D0']['mean']

        ax.axhline(D0/2, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(hl, color='gray', linestyle='--', alpha=0.5, zorder=1)

        ax.annotate(f't½ = {hl:.0f} ka\n[{hl_lower:.0f}, {hl_upper:.0f}]',
                    xy=(hl, D0/2), xytext=(hl + 3, D0/2 + 5),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    # Plot data points with error bars
    ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper],
                fmt='o', markersize=12, color='#2166AC', capsize=5,
                markeredgecolor='black', markeredgewidth=1.5, elinewidth=2,
                label='Observed (OPTIMAL)', zorder=10)

    # Add MIN and MAX points (smaller, for reference)
    ax.scatter(x, y_min, marker='v', s=40, color='lightblue', edgecolors='gray',
               alpha=0.6, label='MIN extent', zorder=6)
    ax.scatter(x, y_max, marker='^', s=40, color='darkblue', edgecolors='gray',
               alpha=0.6, label='MAX extent', zorder=6)

    ax.set_xlabel('Deglaciation Age (ka BP)', fontsize=14)
    ax.set_ylabel('Lake Density (lakes per 1000 km²)', fontsize=14)
    ax.set_title('Lake Density Decay with Landscape Age\n(NADI-1 Ice Sheet Reconstruction)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(x) * 1.15)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


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
    print("  - plot_three_panel_summary()")
    print("  - plot_alpha_elevation_phase_diagram()")
    print("  - plot_slope_relief_heatmap()")
    print("  - plot_xmin_sensitivity()")
    print("  - plot_significance_tests()")
    print("  # x_min by elevation visualizations:")
    print("  - plot_xmin_sensitivity_by_elevation()")
    print("  - plot_ks_curves_overlay()")
    print("  - plot_optimal_xmin_vs_elevation()")
    print("  # NADI-1 visualizations:")
    print("  - plot_glaciated_area_timeseries()")
    print("  - plot_density_with_uncertainty()")
    print("  - plot_nadi1_density_decay()")
    print("  - plot_alpha_stability_by_elevation()")
    print("  - plot_xmin_elevation_summary()")
    print("  - plot_alpha_vs_xmin_by_elevation()")
    print("  # Hypothesis test visualizations:")
    print("  - plot_hypothesis_test_summary()")
    print("  - plot_hypothesis_test_results()")
    print("  - plot_colored_summary_table()")
    print("  # Glacial chronosequence visualizations:")
    print("  - plot_density_by_glacial_stage()")
    print("  - plot_elevation_histogram_by_glacial_stage()")
    print("  - plot_davis_hypothesis_test()")
    print("  - plot_glacial_extent_map()")
    print("  - plot_glacial_chronosequence_summary()")
    print("  - plot_bimodal_decomposition()")
    print("  - plot_power_law_by_glacial_zone()")
    print("  # Enhanced glacial visualizations:")
    print("  - plot_normalized_density_with_glacial_overlay()")
    print("  - plot_glacial_powerlaw_comparison()")
    print("  - plot_glacial_lake_size_histograms()")
    print("  - plot_glacial_xmin_sensitivity()")
    print("  - plot_glacial_geographic_lakes()")
    print("  - plot_glacial_comprehensive_summary()")
    print("  # Spatial scaling visualizations:")
    print("  - plot_latitudinal_scaling()")
    print("  - plot_longitudinal_scaling()")
    print("  - plot_glacial_vs_nonglacial_comparison()")
    print("  - plot_spatial_scaling_summary()")
    print("  - plot_colorful_hypothesis_table()")
