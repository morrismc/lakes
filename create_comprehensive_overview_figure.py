"""
Comprehensive visualization of all analyses and tests performed.

Creates a multi-panel figure showing:
1. Analysis pipeline flowchart
2. Key findings from each analysis
3. How analyses integrate together
4. Geographic scope of different tests
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Create figure
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Comprehensive Analysis Pipeline: Glacial Controls on Lake Density\nIntegrating Temporal, Spatial, and Mechanistic Perspectives',
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# Panel 1: Analysis Pipeline Flowchart (TOP)
# ============================================================================

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('A) Analysis Pipeline: From Data to Mechanistic Understanding',
              fontsize=14, fontweight='bold', loc='left', pad=20)

# Data preparation box
data_box = FancyBboxPatch((0.2, 8), 1.5, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
ax1.add_patch(data_box)
ax1.text(0.95, 8.8, 'DATA\nPREPARATION', ha='center', va='center', fontsize=9, fontweight='bold')
ax1.text(0.95, 8.3, '• 4.9M lakes\n• Parquet format\n• Area filters', ha='center', va='center', fontsize=7)

# Descriptive analyses
desc_box = FancyBboxPatch((2.2, 8), 1.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='lightgreen', linewidth=2)
ax1.add_patch(desc_box)
ax1.text(2.95, 8.8, 'DESCRIPTIVE\nANALYSIS', ha='center', va='center', fontsize=9, fontweight='bold')
ax1.text(2.95, 8.3, '• Elevation\n• Slope\n• Relief\n• Power law', ha='center', va='center', fontsize=7)

# Temporal analyses
temp_box = FancyBboxPatch((4.2, 8), 1.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='lightyellow', linewidth=2)
ax1.add_patch(temp_box)
ax1.text(4.95, 8.8, 'TEMPORAL\nANALYSIS', ha='center', va='center', fontsize=9, fontweight='bold')
ax1.text(4.95, 8.3, '• Half-life\n• Chronosequence\n• Decay model', ha='center', va='center', fontsize=7)

# Spatial analyses
spat_box = FancyBboxPatch((6.2, 8), 1.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='lightcoral', linewidth=2)
ax1.add_patch(spat_box)
ax1.text(6.95, 8.8, 'SPATIAL\nANALYSIS', ha='center', va='center', fontsize=9, fontweight='bold')
ax1.text(6.95, 8.3, '• Multivariate\n• PCA\n• Variance part.', ha='center', va='center', fontsize=7)

# Mechanistic
mech_box = FancyBboxPatch((8.2, 8), 1.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='plum', linewidth=2)
ax1.add_patch(mech_box)
ax1.text(8.95, 8.8, 'MECHANISTIC\nANALYSIS', ha='center', va='center', fontsize=9, fontweight='bold')
ax1.text(8.95, 8.3, '• Mediation\n• Pathways\n• Integration', ha='center', va='center', fontsize=7)

# Arrows connecting boxes
arrow1 = FancyArrowPatch((1.7, 8.6), (2.2, 8.6), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax1.add_patch(arrow1)
arrow2 = FancyArrowPatch((3.7, 8.6), (4.2, 8.6), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax1.add_patch(arrow2)
arrow3 = FancyArrowPatch((5.7, 8.6), (6.2, 8.6), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax1.add_patch(arrow3)
arrow4 = FancyArrowPatch((7.7, 8.6), (8.2, 8.6), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax1.add_patch(arrow4)

# Add detail boxes below main pipeline
# Temporal details
ax1.add_patch(FancyBboxPatch((3.7, 6.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor='orange', facecolor='#FFF8DC', linewidth=1.5, linestyle='--'))
ax1.text(5.0, 7.3, 'TEMPORAL PERSPECTIVE', ha='center', fontsize=8, fontweight='bold')
ax1.text(5.0, 6.9, '1. Bayesian half-life: t½ ≈ 660 ka', ha='center', fontsize=7)
ax1.text(5.0, 6.7, '2. Wisconsin > Illinoian > Driftless', ha='center', fontsize=7)

# Spatial details
ax1.add_patch(FancyBboxPatch((6.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor='blue', facecolor='#E6F2FF', linewidth=1.5, linestyle='--'))
ax1.text(7.75, 7.3, 'SPATIAL PERSPECTIVE', ha='center', fontsize=8, fontweight='bold')
ax1.text(7.75, 6.9, '1. Topography: 30.5% pure effect', ha='center', fontsize=7)
ax1.text(7.75, 6.7, '2. Climate: 16.1%, Glaciation: 9.5%', ha='center', fontsize=7)

# Integration arrow
integration_arrow = FancyArrowPatch((6.2, 7.1), (6.5, 7.1), arrowstyle='<->',
                                   mutation_scale=15, linewidth=2, color='red')
ax1.add_patch(integration_arrow)
ax1.text(6.35, 7.4, 'Integrate', ha='center', fontsize=7, style='italic', color='red', fontweight='bold')

# ============================================================================
# Panel 2: Key Findings by Analysis Type (BOTTOM LEFT)
# ============================================================================

ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('B) Key Findings by Analysis Type', fontsize=12, fontweight='bold', loc='left', pad=10)

y_pos = 9.5

# Temporal analyses
ax2.text(0.5, y_pos, 'TEMPORAL ANALYSES', fontsize=10, fontweight='bold', color='darkgreen')
y_pos -= 0.5

analyses_temporal = [
    ('Bayesian Half-Life', 't½ = 660 ka [95% CI: 418-1505 ka]'),
    ('Chronosequence', 'Wisconsin: 228 → Illinoian: 202 → Driftless: 69 lakes/1000 km²'),
    ('Size-Stratified', 'Half-life independent of lake size'),
    ('NADI-1 Time Slices', 'High-resolution (0.5 ka) deglaciation chronology'),
    ('Threshold Sensitivity', 'Half-life plateaus at thresholds >1 km²'),
]

for name, finding in analyses_temporal:
    ax2.text(0.7, y_pos, f'• {name}:', fontsize=8, fontweight='bold')
    y_pos -= 0.35
    ax2.text(1.0, y_pos, finding, fontsize=7, style='italic')
    y_pos -= 0.5

y_pos -= 0.3

# Spatial analyses
ax2.text(0.5, y_pos, 'SPATIAL ANALYSES', fontsize=10, fontweight='bold', color='darkblue')
y_pos -= 0.5

analyses_spatial = [
    ('Multivariate Regression', 'R² = 0.128, all predictors significant'),
    ('Variance Partitioning', 'Topography 30.5%, Climate 16.1%, Glaciation 9.5%'),
    ('PCA', 'PC1 (40.4%): Climate gradient, PC2 (25.1%): Topographic roughness'),
    ('Correlation Matrix', 'Strong climate coupling: precip ↔ aridity (ρ=0.91)'),
]

for name, finding in analyses_spatial:
    ax2.text(0.7, y_pos, f'• {name}:', fontsize=8, fontweight='bold')
    y_pos -= 0.35
    ax2.text(1.0, y_pos, finding, fontsize=7, style='italic')
    y_pos -= 0.5

y_pos -= 0.3

# Comparative analyses
ax2.text(0.5, y_pos, 'COMPARATIVE ANALYSES', fontsize=10, fontweight='bold', color='darkred')
y_pos -= 0.5

analyses_comp = [
    ('S. Appalachian', 'Density (101.9) > Illinoian (69.3) but < Wisconsin (228.2)'),
    ('Aridity', 'Climate important but not dominant'),
    ('Spatial Scaling', 'Latitudinal and elevational gradients'),
]

for name, finding in analyses_comp:
    ax2.text(0.7, y_pos, f'• {name}:', fontsize=8, fontweight='bold')
    y_pos -= 0.35
    ax2.text(1.0, y_pos, finding, fontsize=7, style='italic')
    y_pos -= 0.5

# ============================================================================
# Panel 3: Mechanistic Integration (BOTTOM MIDDLE)
# ============================================================================

ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=1, rowspan=2)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('C) Mechanistic Integration: How Glaciation Controls Lakes',
              fontsize=12, fontweight='bold', loc='left', pad=10)

# Draw causal pathway
y_start = 8.5

# Step 1: Glaciation
box1 = FancyBboxPatch((2, y_start), 6, 0.8, boxstyle="round,pad=0.1",
                      edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax3.add_patch(box1)
ax3.text(5, y_start+0.4, 'GLACIATION', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5, y_start+0.1, '(Ultimate Cause)', ha='center', va='center', fontsize=8, style='italic')

# Arrow down
arrow_down1 = FancyArrowPatch((5, y_start), (5, y_start-0.8), arrowstyle='->',
                             mutation_scale=25, linewidth=3, color='black')
ax3.add_patch(arrow_down1)
ax3.text(6.2, y_start-0.5, 'Creates', fontsize=9, style='italic')

y_start -= 1.5

# Step 2: Topography
box2 = FancyBboxPatch((2, y_start), 6, 1.0, boxstyle="round,pad=0.1",
                      edgecolor='darkblue', facecolor='lightblue', linewidth=2)
ax3.add_patch(box2)
ax3.text(5, y_start+0.6, 'FAVORABLE TOPOGRAPHY', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5, y_start+0.2, 'Depressions, basins, low relief', ha='center', va='center', fontsize=8)
ax3.text(5, y_start-0.05, '(Proximate Cause)', ha='center', va='center', fontsize=8, style='italic')

# Arrow down
arrow_down2 = FancyArrowPatch((5, y_start), (5, y_start-0.8), arrowstyle='->',
                             mutation_scale=25, linewidth=3, color='black')
ax3.add_patch(arrow_down2)
ax3.text(6.2, y_start-0.5, 'Supports', fontsize=9, style='italic')

y_start -= 1.5

# Step 3: High Density
box3 = FancyBboxPatch((2, y_start), 6, 0.8, boxstyle="round,pad=0.1",
                      edgecolor='darkorange', facecolor='lightyellow', linewidth=2)
ax3.add_patch(box3)
ax3.text(5, y_start+0.4, 'HIGH LAKE DENSITY', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5, y_start+0.1, '~208 lakes/1000 km² (Wisconsin)', ha='center', va='center', fontsize=8)

# Arrow down
arrow_down3 = FancyArrowPatch((5, y_start), (5, y_start-0.8), arrowstyle='->',
                             mutation_scale=25, linewidth=3, color='red')
ax3.add_patch(arrow_down3)
ax3.text(6.5, y_start-0.5, 't½ ≈ 660 ka', fontsize=9, style='italic', color='red', fontweight='bold')

y_start -= 1.5

# Step 4: Degradation
box4 = FancyBboxPatch((2, y_start), 6, 1.0, boxstyle="round,pad=0.1",
                      edgecolor='purple', facecolor='#E6D5F0', linewidth=2)
ax3.add_patch(box4)
ax3.text(5, y_start+0.6, 'TOPOGRAPHIC DEGRADATION', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5, y_start+0.2, 'Depressions fill in', ha='center', va='center', fontsize=8)
ax3.text(5, y_start-0.05, 'Relief increases over time', ha='center', va='center', fontsize=8)

# Arrow down
arrow_down4 = FancyArrowPatch((5, y_start), (5, y_start-0.8), arrowstyle='->',
                             mutation_scale=25, linewidth=3, color='red')
ax3.add_patch(arrow_down4)
ax3.text(6.2, y_start-0.5, 'Results in', fontsize=9, style='italic', color='red')

y_start -= 1.5

# Step 5: Low Density
box5 = FancyBboxPatch((2, y_start), 6, 0.8, boxstyle="round,pad=0.1",
                      edgecolor='darkred', facecolor='#FFE6E6', linewidth=2)
ax3.add_patch(box5)
ax3.text(5, y_start+0.4, 'LOW LAKE DENSITY', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5, y_start+0.1, '~69 lakes/1000 km² (Driftless)', ha='center', va='center', fontsize=8)

# Add side annotation
ax3.text(0.5, 5, 'TIME', fontsize=14, fontweight='bold', color='red', rotation=90, va='center', ha='center')
arrow_time = FancyArrowPatch((1, 8), (1, 2), arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='red', linestyle='--')
ax3.add_patch(arrow_time)

# ============================================================================
# Panel 4: Analysis Coverage & Geographic Scope (BOTTOM RIGHT)
# ============================================================================

ax4 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('D) Analysis Coverage & Key Statistics', fontsize=12, fontweight='bold', loc='left', pad=10)

y_pos = 9.5

# Dataset overview
ax4.text(0.5, y_pos, 'DATASET OVERVIEW', fontsize=10, fontweight='bold', color='darkblue')
y_pos -= 0.5

dataset_stats = [
    'Total lakes (CONUS): 4,896,094',
    'After filtering (0.005-20000 km²): 1,173,057',
    'Grid cells (0.5° resolution): 3,589',
    'Spatial coverage: Continental U.S.',
]

for stat in dataset_stats:
    ax4.text(0.7, y_pos, f'• {stat}', fontsize=8)
    y_pos -= 0.4

y_pos -= 0.4

# Glacial stages
ax4.text(0.5, y_pos, 'GLACIAL STAGE BREAKDOWN', fontsize=10, fontweight='bold', color='darkgreen')
y_pos -= 0.5

stage_stats = [
    ('Wisconsin', '621 cells', '206.6 lakes/1000 km²'),
    ('Illinoian', '54 cells', '225.2 lakes/1000 km²'),
    ('Driftless', '8 cells', '78.0 lakes/1000 km²'),
    ('Unclassified', '2,906 cells', '122.3 lakes/1000 km²'),
    ('S. Appalachian', 'separate analysis', '101.9 lakes/1000 km²'),
]

for stage, cells, density in stage_stats:
    ax4.text(0.7, y_pos, f'• {stage}:', fontsize=8, fontweight='bold')
    y_pos -= 0.3
    ax4.text(1.0, y_pos, f'{cells}, density: {density}', fontsize=7)
    y_pos -= 0.4

y_pos -= 0.4

# Variables analyzed
ax4.text(0.5, y_pos, 'VARIABLES ANALYZED', fontsize=10, fontweight='bold', color='darkred')
y_pos -= 0.5

var_categories = [
    ('Response', 'Lake density (lakes/1000 km²)'),
    ('Glaciation', 'Glacial stage (W/I/D/U)'),
    ('Topography', 'Elevation, slope, relief'),
    ('Climate', 'Aridity index, precipitation'),
]

for category, vars in var_categories:
    ax4.text(0.7, y_pos, f'• {category}:', fontsize=8, fontweight='bold')
    y_pos -= 0.3
    ax4.text(1.0, y_pos, vars, fontsize=7)
    y_pos -= 0.4

y_pos -= 0.4

# Key statistical findings
ax4.text(0.5, y_pos, 'KEY STATISTICAL FINDINGS', fontsize=10, fontweight='bold', color='purple')
y_pos -= 0.5

stat_findings = [
    'Total variance explained: R² = 0.128',
    'Shared variance: 43.9% (high collinearity)',
    'All predictors significant: p < 0.001',
    'Glaciation effect: β = -22.9 (indirect)',
]

for finding in stat_findings:
    ax4.text(0.7, y_pos, f'• {finding}', fontsize=8)
    y_pos -= 0.4

plt.tight_layout()
plt.savefig('F:/Lakes/Analysis/outputs/comprehensive_analysis_overview.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS OVERVIEW FIGURE CREATED")
print("="*80)
print("\nSaved: F:/Lakes/Analysis/outputs/comprehensive_analysis_overview.png")
print("\nThis figure shows:")
print("  1. Complete analysis pipeline from data to mechanistic understanding")
print("  2. Key findings from 15+ different analyses")
print("  3. Mechanistic pathway: Glaciation → Topography → Lakes → Decay")
print("  4. Dataset coverage and statistical summary")
print("\n" + "="*80)

plt.show()
