"""
Visualization functions for multivariate analysis results.

Generates publication-quality figures for:
- Correlation matrices
- PCA biplots
- Variance partitioning Venn diagrams
- Partial correlation networks
- Variable importance plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import warnings


def plot_correlation_matrix(corr_matrix, figsize=(10, 8), save_path=None):
    """
    Plot correlation matrix as heatmap.

    Parameters
    ----------
    corr_matrix : DataFrame
        Correlation matrix
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Spearman ρ'},
        ax=ax
    )

    ax.set_title('Variable Correlation Matrix\n(Spearman rank correlation)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_pca_biplot(pca_results, figsize=(14, 10), save_path=None):
    """
    PCA biplot showing both scores (observations) and loadings (variables).

    Parameters
    ----------
    pca_results : dict
        PCA results from run_pca_analysis()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Principal Component Analysis\nIdentifying Dominant Environmental Gradients',
                fontsize=16, fontweight='bold', y=0.995)

    scores = pca_results['scores']
    loadings = pca_results['loadings']
    var_names = pca_results['variable_names']
    explained_var = pca_results['explained_variance']

    # Determine observation type (grid cells or individual lakes)
    obs_type = 'Grid Cells' if scores.shape[0] < 10000 else 'Individual Lakes'

    # Color by glacial stage if available
    if 'glacial_stage' in pca_results:
        stages = pca_results['glacial_stage']
        unique_stages = np.unique(stages)
        colors = plt.cm.tab10(np.arange(len(unique_stages)))
        stage_colors = {stage: colors[i] for i, stage in enumerate(unique_stages)}
    else:
        stages = None

    # Panel A: PC1 vs PC2 scores
    ax = axes[0, 0]
    if stages is not None:
        # Plot in order: unclassified first (behind), then glaciated stages
        plot_order = ['unclassified'] + [s for s in unique_stages if s != 'unclassified']

        for stage in plot_order:
            if stage not in unique_stages:
                continue

            mask = stages == stage
            n_points = np.sum(mask)

            # Subsample unclassified points if too many
            if stage == 'unclassified' and n_points > 5000:
                indices = np.where(mask)[0]
                subsample_idx = np.random.choice(indices, size=5000, replace=False)
                mask_plot = np.zeros(len(mask), dtype=bool)
                mask_plot[subsample_idx] = True
                alpha_val = 0.05
                label_str = f'{stage} (showing 5000/{n_points})'
            else:
                mask_plot = mask
                alpha_val = 0.4 if stage != 'unclassified' else 0.1
                label_str = stage

            ax.scatter(scores[mask_plot, 0], scores[mask_plot, 1],
                      c=[stage_colors[stage]], label=label_str,
                      alpha=alpha_val, s=2 if stage != 'unclassified' else 1,
                      rasterized=True, edgecolors='none')

        # Legend outside plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), markerscale=3,
                 frameon=True, fancybox=True, shadow=True)
    else:
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.3, s=1, rasterized=True)

    ax.set_xlabel(f'PC1 ({100*explained_var[0]:.1f}% var)')
    ax.set_ylabel(f'PC2 ({100*explained_var[1]:.1f}% var)')
    ax.set_title(f'A) Sample Scores ({obs_type})', fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Panel B: PC1 vs PC2 loadings (biplot arrows)
    ax = axes[0, 1]
    for i, var in enumerate(var_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var,
               fontsize=10, ha='center', va='center', fontweight='bold')

    ax.set_xlabel(f'PC1 Loadings ({100*explained_var[0]:.1f}% var)')
    ax.set_ylabel(f'PC2 Loadings ({100*explained_var[1]:.1f}% var)')
    ax.set_title('B) Variable Loadings (Environmental Axes)', fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_aspect('equal')

    # Panel C: Scree plot
    ax = axes[1, 0]
    n_components = len(explained_var)
    ax.bar(range(1, n_components + 1), explained_var * 100, color='steelblue', edgecolor='black')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('C) Scree Plot (Variance Explained)', fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)

    # Add cumulative variance line
    ax2 = ax.twinx()
    cumulative_var = np.cumsum(explained_var) * 100
    ax2.plot(range(1, n_components + 1), cumulative_var, 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Cumulative Variance (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 105])

    # Panel D: Loading heatmap
    ax = axes[1, 1]
    loading_matrix = loadings[:, :min(3, n_components)]
    im = ax.imshow(loading_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-np.abs(loading_matrix).max(), vmax=np.abs(loading_matrix).max())
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.set_yticks(range(min(3, n_components)))
    ax.set_yticklabels([f'PC{i+1}' for i in range(min(3, n_components))])
    ax.set_title('D) Loading Heatmap (First 3 PCs)', fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax, label='Loading')

    # Add loading values as text
    for i in range(len(var_names)):
        for j in range(min(3, n_components)):
            text = ax.text(i, j, f'{loading_matrix[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if abs(loading_matrix[i, j]) > 0.5 * np.abs(loading_matrix).max() else 'black',
                          fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


def plot_variance_partitioning(vp_results, figsize=(10, 8), save_path=None):
    """
    Variance partitioning Venn diagram showing pure and shared effects.

    Parameters
    ----------
    vp_results : dict
        Variance partitioning results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract values
    pure_glac = vp_results['pure_glacial']
    pure_clim = vp_results['pure_climate']
    pure_topo = vp_results['pure_topo']
    shared = vp_results['shared']
    r2_total = vp_results['r2_total']

    # Convert to percentages
    pct_pure_glac = 100 * pure_glac / r2_total if r2_total > 0 else 0
    pct_pure_clim = 100 * pure_clim / r2_total if r2_total > 0 else 0
    pct_pure_topo = 100 * pure_topo / r2_total if r2_total > 0 else 0
    pct_shared = 100 * shared / r2_total if r2_total > 0 else 0

    # Bar plot
    categories = ['Pure\nGlaciation', 'Pure\nClimate', 'Pure\nTopography', 'Shared\nVariance']
    values = [pct_pure_glac, pct_pure_clim, pct_pure_topo, pct_shared]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Proportion of Explained Variance (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Variance Partitioning Analysis\nTotal R² = {r2_total:.3f}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, max(values) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add interpretation text
    interpretation = "Interpretation:\n"
    if pct_pure_glac > pct_pure_clim and pct_pure_glac > pct_pure_topo:
        interpretation += "→ Glaciation is the PRIMARY control on lake density\n"
        interpretation += f"  (explains {pct_pure_glac:.1f}% after controlling for other factors)"
    elif pct_pure_clim > pct_pure_glac and pct_pure_clim > pct_pure_topo:
        interpretation += "→ Climate is the PRIMARY control on lake density\n"
        interpretation += f"  (explains {pct_pure_clim:.1f}% after controlling for other factors)"
    elif pct_pure_topo > pct_pure_glac and pct_pure_topo > pct_pure_clim:
        interpretation += "→ Topography is the PRIMARY control on lake density\n"
        interpretation += f"  (explains {pct_pure_topo:.1f}% after controlling for other factors)"
    else:
        interpretation += "→ No single factor dominates\n"
        interpretation += "  Effects are roughly balanced"

    ax.text(0.02, 0.98, interpretation,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_variable_importance(reg_results, figsize=(10, 6), save_path=None):
    """
    Variable importance plot from multiple regression.

    Parameters
    ----------
    reg_results : dict
        Regression results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    predictors = reg_results['predictors']
    coefficients = reg_results['coefficients']
    p_values = reg_results['p_values']

    # Sort by absolute coefficient
    sort_idx = np.argsort(np.abs(coefficients))[::-1]
    predictors_sorted = [predictors[i] for i in sort_idx]
    coef_sorted = coefficients[sort_idx]
    p_sorted = p_values[sort_idx]

    # Color by significance
    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'gray'
             for p in p_sorted]

    # Horizontal bar plot
    y_pos = np.arange(len(predictors_sorted))
    bars = ax.barh(y_pos, coef_sorted, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors_sorted, fontsize=11)
    ax.set_xlabel('Standardized Coefficient (β)', fontsize=12, fontweight='bold')
    ax.set_title(f'Variable Importance in Multiple Regression\nR² = {reg_results["r2"]:.3f}',
                fontsize=14, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linewidth=2)
    ax.grid(axis='x', alpha=0.3)

    # Add significance stars
    for i, (bar, p) in enumerate(zip(bars, p_sorted)):
        width = bar.get_width()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {sig}',
               ha='left' if width > 0 else 'right',
               va='center', fontsize=10, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='p < 0.001 ***'),
        Patch(facecolor='orange', edgecolor='black', label='p < 0.01 **'),
        Patch(facecolor='yellow', edgecolor='black', label='p < 0.05 *'),
        Patch(facecolor='gray', edgecolor='black', label='p ≥ 0.05 (ns)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_multivariate_summary(results, output_dir=None):
    """
    Generate complete multivariate analysis figure set.

    Parameters
    ----------
    results : dict
        Complete multivariate analysis results
    output_dir : str
        Output directory for figures

    Returns
    -------
    dict
        Dictionary of (fig, axes) tuples for each plot
    """
    import os

    if output_dir is None:
        try:
            from .config import OUTPUT_DIR
            output_dir = OUTPUT_DIR
        except:
            from config import OUTPUT_DIR
            output_dir = OUTPUT_DIR

    figures = {}

    # 1. Correlation matrix
    if 'correlation_matrix' in results:
        save_path = os.path.join(output_dir, 'multivariate_correlation_matrix.png')
        fig, ax = plot_correlation_matrix(results['correlation_matrix'], save_path=save_path)
        figures['correlation_matrix'] = (fig, ax)

    # 2. PCA biplot
    if 'pca' in results:
        save_path = os.path.join(output_dir, 'multivariate_pca_biplot.png')
        fig, axes = plot_pca_biplot(results['pca'], save_path=save_path)
        figures['pca'] = (fig, axes)

    # 3. Variance partitioning
    if 'variance_partitioning' in results and results['variance_partitioning'] is not None:
        save_path = os.path.join(output_dir, 'multivariate_variance_partitioning.png')
        fig, ax = plot_variance_partitioning(results['variance_partitioning'], save_path=save_path)
        figures['variance_partitioning'] = (fig, ax)

    # 4. Variable importance
    if 'regression' in results:
        save_path = os.path.join(output_dir, 'multivariate_variable_importance.png')
        fig, ax = plot_variable_importance(results['regression'], save_path=save_path)
        figures['variable_importance'] = (fig, ax)

    print(f"\nAll multivariate figures saved to: {output_dir}")

    return figures


# Export functions
__all__ = [
    'plot_correlation_matrix',
    'plot_pca_biplot',
    'plot_variance_partitioning',
    'plot_variable_importance',
    'plot_multivariate_summary',
]
