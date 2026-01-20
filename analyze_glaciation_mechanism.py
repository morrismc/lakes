"""
Mechanistic analysis: How does glaciation prepare topography for lakes?

This script tests the hypothesis that glaciation acts INDIRECTLY through topography:
    Glaciation → Creates favorable topography → Supports higher lake density

Key questions:
1. Do glaciated regions have different topographic characteristics?
2. Does glaciation's effect mediated through topography?
3. How does this integrate with the temporal decay pattern (half-life)?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from lake_analysis import (
    load_conus_lake_data,
    convert_lakes_to_gdf,
    load_all_glacial_boundaries,
    classify_lakes_by_glacial_extent,
    create_gridded_density_dataset
)

# Load and classify
print("Loading and classifying lakes...")
lakes = load_conus_lake_data()
lakes_gdf = convert_lakes_to_gdf(lakes)
boundaries = load_all_glacial_boundaries()
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)

# Create gridded dataset
print("\nCreating gridded density dataset...")
grid_data = create_gridded_density_dataset(
    lakes_classified,
    grid_size_deg=0.5,
    min_lake_area=0.005,
    max_lake_area=20000,
    min_lakes_per_cell=5,
    verbose=True
)

print("\n" + "="*80)
print("MECHANISTIC ANALYSIS: GLACIATION → TOPOGRAPHY → LAKES")
print("="*80)

# ============================================================================
# ANALYSIS 1: Topographic Characteristics by Glacial Stage
# ============================================================================

print("\n" + "="*80)
print("1. TOPOGRAPHIC CHARACTERISTICS BY GLACIAL STAGE")
print("="*80)

topographic_vars = ['elevation', 'slope', 'relief']
stages = ['Wisconsin', 'Illinoian', 'Driftless', 'unclassified']

# Summary statistics
print("\nMean values by glacial stage:")
print("-" * 80)
for stage in stages:
    if stage not in grid_data['glacial_stage'].values:
        continue

    stage_data = grid_data[grid_data['glacial_stage'] == stage]
    n_cells = len(stage_data)
    density = stage_data['density'].mean()

    print(f"\n{stage} (n={n_cells} cells, density={density:.1f} lakes/1000 km²):")
    for var in topographic_vars:
        if var in stage_data.columns:
            mean_val = stage_data[var].mean()
            std_val = stage_data[var].std()
            print(f"  {var:12s}: {mean_val:8.1f} ± {std_val:6.1f}")

# Statistical tests
print("\n" + "-"*80)
print("Statistical comparison (Kruskal-Wallis test):")
print("-"*80)

for var in topographic_vars:
    if var not in grid_data.columns:
        continue

    groups = [grid_data[grid_data['glacial_stage'] == stage][var].values
              for stage in stages if stage in grid_data['glacial_stage'].values]

    h_stat, p_val = stats.kruskal(*groups)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{var:12s}: H = {h_stat:6.2f}, p = {p_val:.4e} {sig}")

# ============================================================================
# ANALYSIS 2: Glaciated vs Non-Glaciated Comparison
# ============================================================================

print("\n" + "="*80)
print("2. GLACIATED vs NON-GLACIATED REGIONS")
print("="*80)

# Define glaciated (Wisconsin + Illinoian + Driftless) vs non-glaciated (unclassified)
grid_data['glaciated'] = grid_data['glacial_stage'].isin(['Wisconsin', 'Illinoian', 'Driftless'])

glaciated = grid_data[grid_data['glaciated']]
nonglaciated = grid_data[~grid_data['glaciated']]

print(f"\nGlaciated regions: {len(glaciated)} cells")
print(f"Non-glaciated regions: {len(nonglaciated)} cells")

print("\nMean lake density:")
print(f"  Glaciated:     {glaciated['density'].mean():6.1f} lakes/1000 km²")
print(f"  Non-glaciated: {nonglaciated['density'].mean():6.1f} lakes/1000 km²")
print(f"  Ratio:         {glaciated['density'].mean() / nonglaciated['density'].mean():.2f}x")

# Statistical test
u_stat, p_val = stats.mannwhitneyu(glaciated['density'], nonglaciated['density'])
print(f"  Mann-Whitney U test: U = {u_stat:.0f}, p = {p_val:.4e}")

print("\nMean topographic characteristics:")
for var in topographic_vars:
    if var in grid_data.columns:
        glac_mean = glaciated[var].mean()
        nonglac_mean = nonglaciated[var].mean()
        u_stat, p_val = stats.mannwhitneyu(glaciated[var], nonglaciated[var])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {var:12s}: Glaciated={glac_mean:7.1f}, Non-glaciated={nonglac_mean:7.1f}, p={p_val:.4e} {sig}")

# ============================================================================
# ANALYSIS 3: Mediation Analysis - Does topography mediate glaciation effect?
# ============================================================================

print("\n" + "="*80)
print("3. MEDIATION ANALYSIS: Glaciation → Topography → Density")
print("="*80)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Encode glacial stage
le = LabelEncoder()
grid_data['glacial_encoded'] = le.fit_transform(grid_data['glacial_stage'])

# Path c: Total effect (Glaciation → Density)
X_glac = grid_data[['glacial_encoded']].values
y_density = grid_data['density'].values

model_c = LinearRegression().fit(X_glac, y_density)
total_effect = model_c.coef_[0]
r2_total = model_c.score(X_glac, y_density)

print(f"\nTotal effect (c):")
print(f"  Glaciation → Density: β = {total_effect:.3f}, R² = {r2_total:.3f}")

# Path a: Glaciation → Topography (for each topographic variable)
print(f"\nPath a (Glaciation → Topography):")
for var in topographic_vars:
    if var in grid_data.columns:
        y_topo = grid_data[var].values
        model_a = LinearRegression().fit(X_glac, y_topo)
        effect_a = model_a.coef_[0]
        r2_a = model_a.score(X_glac, y_topo)
        print(f"  Glaciation → {var:12s}: β = {effect_a:8.3f}, R² = {r2_a:.3f}")

# Path b: Topography → Density (controlling for glaciation)
print(f"\nPath b (Topography → Density | Glaciation):")
topo_cols = [v for v in topographic_vars if v in grid_data.columns]
X_topo_glac = grid_data[topo_cols + ['glacial_encoded']].values

model_b = LinearRegression().fit(X_topo_glac, y_density)
r2_mediated = model_b.score(X_topo_glac, y_density)

for i, var in enumerate(topo_cols):
    effect_b = model_b.coef_[i]
    print(f"  {var:12s} → Density: β = {effect_b:8.3f}")

direct_effect = model_b.coef_[-1]  # Coefficient for glacial_encoded
print(f"  Glaciation → Density (direct): β = {direct_effect:8.3f}")

# Calculate indirect effect and proportion mediated
# For simplicity, use average topographic effect
print(f"\nMediation summary:")
print(f"  Total effect (c):   {total_effect:.3f}")
print(f"  Direct effect (c'): {direct_effect:.3f}")
print(f"  Indirect effect:    {total_effect - direct_effect:.3f}")
print(f"  Proportion mediated: {100 * (total_effect - direct_effect) / total_effect:.1f}%")

print(f"\nR² comparison:")
print(f"  Glaciation alone:            R² = {r2_total:.3f}")
print(f"  Glaciation + Topography:     R² = {r2_mediated:.3f}")
print(f"  Topography explains extra:   ΔR² = {r2_mediated - r2_total:.3f}")

# ============================================================================
# VISUALIZATION 1: Topographic characteristics by glacial stage
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Topographic Characteristics by Glacial Stage\nTesting: Does glaciation create favorable topography for lakes?',
             fontsize=14, fontweight='bold', y=0.995)

stages_plot = ['Wisconsin', 'Illinoian', 'Driftless', 'unclassified']
colors = {'Wisconsin': '#2E7D32', 'Illinoian': '#FF6F00', 'Driftless': '#1565C0', 'unclassified': '#757575'}

# Panel A: Lake density
ax = axes[0, 0]
densities = []
labels = []
for stage in stages_plot:
    if stage in grid_data['glacial_stage'].values:
        data = grid_data[grid_data['glacial_stage'] == stage]['density']
        densities.append(data)
        labels.append(f"{stage}\n(n={len(data)})")

bp = ax.boxplot(densities, labels=labels, patch_artist=True, showfliers=False)
for patch, stage in zip(bp['boxes'], stages_plot):
    if stage in grid_data['glacial_stage'].values:
        patch.set_facecolor(colors[stage])
        patch.set_alpha(0.6)

ax.set_ylabel('Lake Density (lakes/1000 km²)', fontweight='bold')
ax.set_title('A) Lake Density by Glacial Stage', fontweight='bold', loc='left')
ax.grid(axis='y', alpha=0.3)

# Panel B: Elevation
ax = axes[0, 1]
elevations = []
for stage in stages_plot:
    if stage in grid_data['glacial_stage'].values:
        data = grid_data[grid_data['glacial_stage'] == stage]['elevation']
        elevations.append(data)

bp = ax.boxplot(elevations, labels=labels, patch_artist=True, showfliers=False)
for patch, stage in zip(bp['boxes'], stages_plot):
    if stage in grid_data['glacial_stage'].values:
        patch.set_facecolor(colors[stage])
        patch.set_alpha(0.6)

ax.set_ylabel('Elevation (m)', fontweight='bold')
ax.set_title('B) Elevation by Glacial Stage', fontweight='bold', loc='left')
ax.grid(axis='y', alpha=0.3)

# Panel C: Slope
ax = axes[1, 0]
slopes = []
for stage in stages_plot:
    if stage in grid_data['glacial_stage'].values:
        data = grid_data[grid_data['glacial_stage'] == stage]['slope']
        slopes.append(data)

bp = ax.boxplot(slopes, labels=labels, patch_artist=True, showfliers=False)
for patch, stage in zip(bp['boxes'], stages_plot):
    if stage in grid_data['glacial_stage'].values:
        patch.set_facecolor(colors[stage])
        patch.set_alpha(0.6)

ax.set_ylabel('Slope (degrees)', fontweight='bold')
ax.set_title('C) Slope by Glacial Stage', fontweight='bold', loc='left')
ax.grid(axis='y', alpha=0.3)

# Panel D: Relief
ax = axes[1, 1]
reliefs = []
for stage in stages_plot:
    if stage in grid_data['glacial_stage'].values:
        data = grid_data[grid_data['glacial_stage'] == stage]['relief']
        reliefs.append(data)

bp = ax.boxplot(reliefs, labels=labels, patch_artist=True, showfliers=False)
for patch, stage in zip(bp['boxes'], stages_plot):
    if stage in grid_data['glacial_stage'].values:
        patch.set_facecolor(colors[stage])
        patch.set_alpha(0.6)

ax.set_ylabel('Relief (m)', fontweight='bold')
ax.set_title('D) Relief by Glacial Stage', fontweight='bold', loc='left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('F:/Lakes/Analysis/outputs/topography_by_glacial_stage.png', dpi=300, bbox_inches='tight')
print("\nSaved: F:/Lakes/Analysis/outputs/topography_by_glacial_stage.png")

# ============================================================================
# VISUALIZATION 2: Mediation pathway diagram
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.suptitle('Mediation Analysis: Glaciation → Topography → Lake Density\nExploring the mechanism of glacial control',
             fontsize=14, fontweight='bold')

ax.text(0.5, 0.95, 'Does glaciation affect lakes DIRECTLY or through TOPOGRAPHY?',
        ha='center', va='top', fontsize=12, style='italic', transform=ax.transAxes)

# Draw pathway diagram (will create proper visualization)
ax.axis('off')

# Add text summary
y_pos = 0.80
ax.text(0.1, y_pos, 'PATHWAY ANALYSIS:', fontsize=12, fontweight='bold', transform=ax.transAxes)
y_pos -= 0.08

ax.text(0.1, y_pos, f'Total Effect (c): Glaciation → Density', fontsize=11, transform=ax.transAxes)
y_pos -= 0.05
ax.text(0.15, y_pos, f'β = {total_effect:.3f}, R² = {r2_total:.3f}', fontsize=10,
        family='monospace', transform=ax.transAxes)
y_pos -= 0.08

ax.text(0.1, y_pos, f'Direct Effect (c\'): Glaciation → Density (after controlling for topography)',
        fontsize=11, transform=ax.transAxes)
y_pos -= 0.05
ax.text(0.15, y_pos, f'β = {direct_effect:.3f}', fontsize=10, family='monospace', transform=ax.transAxes)
y_pos -= 0.08

ax.text(0.1, y_pos, f'Indirect Effect: Glaciation → Topography → Density', fontsize=11, transform=ax.transAxes)
y_pos -= 0.05
ax.text(0.15, y_pos, f'β = {total_effect - direct_effect:.3f} ({100 * (total_effect - direct_effect) / total_effect:.1f}% of total effect)',
        fontsize=10, family='monospace', transform=ax.transAxes)
y_pos -= 0.10

# Interpretation
ax.text(0.1, y_pos, 'INTERPRETATION:', fontsize=12, fontweight='bold', transform=ax.transAxes)
y_pos -= 0.06

proportion_mediated = 100 * (total_effect - direct_effect) / total_effect
if proportion_mediated > 50:
    interp = f'Glaciation acts PRIMARILY through topography ({proportion_mediated:.0f}% mediated)'
elif proportion_mediated > 25:
    interp = f'Glaciation acts PARTIALLY through topography ({proportion_mediated:.0f}% mediated)'
else:
    interp = f'Glaciation has a strong DIRECT effect ({100-proportion_mediated:.0f}% direct)'

ax.text(0.1, y_pos, f'→ {interp}', fontsize=11, style='italic', transform=ax.transAxes)
y_pos -= 0.06
ax.text(0.1, y_pos, f'→ Adding topography increases R² from {r2_total:.3f} to {r2_mediated:.3f} (ΔR² = {r2_mediated - r2_total:.3f})',
        fontsize=11, style='italic', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('F:/Lakes/Analysis/outputs/mediation_pathway_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: F:/Lakes/Analysis/outputs/mediation_pathway_analysis.png")

plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - See output figures for visualization")
print("="*80)
