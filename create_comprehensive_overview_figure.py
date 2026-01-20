"""
IMPROVED Comprehensive visualization of all analyses and tests performed.

Creates a cleaner, more aesthetic multi-panel figure showing:
1. Code structure flowchart (how modules are organized)
2. Analysis pipeline flowchart
3. Key findings from each analysis
4. Mechanistic integration with mediation results
5. Geographic scope of different tests
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch

# Create figure with better layout
fig = plt.figure(figsize=(22, 16))
fig.suptitle('Comprehensive Analysis Overview: Glacial Controls on Lake Density\n' +
             'Integrating Temporal, Spatial, and Mechanistic Perspectives',
             fontsize=20, fontweight='bold', y=0.985)

# Define color palette (more professional)
colors = {
    'data': '#B3E5FC',      # Light blue
    'desc': '#C8E6C9',      # Light green
    'temp': '#FFF9C4',      # Light yellow
    'spat': '#FFCCBC',      # Light orange
    'mech': '#E1BEE7',      # Light purple
    'code': '#F0F4C3',      # Light lime
}

# ============================================================================
# Panel 1: CODE STRUCTURE (TOP LEFT)
# ============================================================================

ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=1, rowspan=1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('A) Code Structure: Module Organization',
              fontsize=13, fontweight='bold', loc='left', pad=15)

# Main modules
y = 9.0
module_height = 0.7
module_width = 8.5

modules = [
    ('config.py', 'Configuration & Paths', colors['code']),
    ('data_loading.py', 'Load Lakes & Rasters', colors['data']),
    ('glacial_chronosequence.py', 'Temporal Analysis', colors['temp']),
    ('multivariate_analysis.py', 'Spatial Analysis', colors['spat']),
    ('visualization.py', 'Plotting Functions', colors['desc']),
]

for module, desc, color in modules:
    box = FancyBboxPatch((0.5, y-module_height), module_width, module_height,
                         boxstyle="round,pad=0.05", edgecolor='black',
                         facecolor=color, linewidth=1.5, alpha=0.8)
    ax1.add_patch(box)
    ax1.text(1.5, y-module_height/2, module, fontsize=9, fontweight='bold', va='center')
    ax1.text(5.5, y-module_height/2, desc, fontsize=8, va='center', style='italic')
    y -= 1.1

# Main entry point
y -= 0.3
box = FancyBboxPatch((0.5, y-module_height), module_width, module_height,
                     boxstyle="round,pad=0.05", edgecolor='darkblue',
                     facecolor='#BBDEFB', linewidth=2.5, alpha=0.9)
ax1.add_patch(box)
ax1.text(4.75, y-module_height/2, 'main.py (Orchestration)',
         fontsize=9, fontweight='bold', va='center', ha='center')

# Standalone scripts
y -= 1.3
ax1.text(0.5, y, 'Standalone Scripts:', fontsize=9, fontweight='bold')
y -= 0.6
scripts = [
    'run_multivariate_analysis.py',
    'analyze_glaciation_mechanism.py',
    'run_bayesian_with_sapp.py',
]
for script in scripts:
    ax1.text(1.2, y, f'• {script}', fontsize=7)
    y -= 0.5

# ============================================================================
# Panel 2: ANALYSIS PIPELINE (TOP MIDDLE-RIGHT)
# ============================================================================

ax2 = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=1)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('B) Analysis Pipeline: From Raw Data to Mechanistic Understanding',
              fontsize=13, fontweight='bold', loc='left', pad=15)

# Define pipeline boxes with better spacing
pipeline_y = 7.5
box_width = 1.6
box_height = 1.8
x_spacing = 2.0

stages = [
    ('DATA\nPREP', '4.9M lakes\nArea filters\nGeoDataFrame', colors['data']),
    ('DESCRIPTIVE', 'Elevation\nSlope\nRelief', colors['desc']),
    ('TEMPORAL', 'Half-life\nDecay model\nt½≈660 ka', colors['temp']),
    ('SPATIAL', 'Multivariate\nPCA\nR²=0.128', colors['spat']),
    ('MECHANISTIC', 'Mediation\n53% indirect\nPathway', colors['mech']),
]

boxes = []
for i, (title, desc, color) in enumerate(stages):
    x = 0.5 + i * x_spacing
    box = FancyBboxPatch((x, pipeline_y), box_width, box_height,
                         boxstyle="round,pad=0.08", edgecolor='black',
                         facecolor=color, linewidth=2, alpha=0.85)
    ax2.add_patch(box)

    # Title
    ax2.text(x + box_width/2, pipeline_y + box_height - 0.3, title,
             ha='center', va='center', fontsize=9, fontweight='bold')

    # Description (multi-line)
    lines = desc.split('\n')
    y_text = pipeline_y + box_height/2 - 0.1
    for line in lines:
        ax2.text(x + box_width/2, y_text, line,
                ha='center', va='center', fontsize=7)
        y_text -= 0.35

    boxes.append((x, pipeline_y))

    # Arrow to next stage
    if i < len(stages) - 1:
        arrow = FancyArrowPatch((x + box_width, pipeline_y + box_height/2),
                               (x + x_spacing, pipeline_y + box_height/2),
                               arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color='black', zorder=10)
        ax2.add_patch(arrow)

# Key findings callouts (no overlap)
# Temporal perspective
temp_box = FancyBboxPatch((3.8, 5.2), 2.8, 1.3, boxstyle="round,pad=0.08",
                         edgecolor='#F57C00', facecolor='#FFF3E0',
                         linewidth=2, linestyle='--', alpha=0.9)
ax2.add_patch(temp_box)
ax2.text(5.2, 6.2, 'Temporal View', ha='center', fontsize=9, fontweight='bold', color='#E65100')
ax2.text(5.2, 5.85, 'W > I > D density', ha='center', fontsize=7.5)
ax2.text(5.2, 5.55, 'Decay: t½ ≈ 660 ka', ha='center', fontsize=7.5)

# Spatial perspective
spat_box = FancyBboxPatch((7.0, 5.2), 2.8, 1.3, boxstyle="round,pad=0.08",
                         edgecolor='#1976D2', facecolor='#E3F2FD',
                         linewidth=2, linestyle='--', alpha=0.9)
ax2.add_patch(spat_box)
ax2.text(8.4, 6.2, 'Spatial View', ha='center', fontsize=9, fontweight='bold', color='#0D47A1')
ax2.text(8.4, 5.85, 'Topography: 30.5%', ha='center', fontsize=7.5)
ax2.text(8.4, 5.55, 'Climate: 16.1%', ha='center', fontsize=7.5)

# Integration indicator
ax2.annotate('', xy=(7.0, 5.85), xytext=(6.6, 5.85),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
ax2.text(6.8, 6.3, 'INTEGRATE', ha='center', fontsize=8,
        fontweight='bold', color='red', style='italic')

# ============================================================================
# Panel 3: MECHANISTIC PATHWAY (LEFT MIDDLE)
# ============================================================================

ax3 = plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=2)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('C) Mechanistic Pathway: Glaciation → Topography → Lake Density\n' +
              'Mediation Analysis Results (Key Finding: 53% mediated through topography)',
              fontsize=12, fontweight='bold', loc='left', pad=15)

# Draw pathway with actual mediation results
y_start = 8.8
box_w = 8.5
box_h = 0.9

# Glaciation
box1 = FancyBboxPatch((0.7, y_start), box_w, box_h, boxstyle="round,pad=0.08",
                      edgecolor='#2E7D32', facecolor='#C8E6C9', linewidth=2.5)
ax3.add_patch(box1)
ax3.text(5.0, y_start+box_h/2, 'GLACIATION (Ultimate Cause)', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Total effect annotation
ax3.text(0.2, y_start+box_h/2, 'c', ha='center', va='center',
        fontsize=10, fontweight='bold', style='italic', color='blue')

y_start -= 1.4

# Arrow with mediation stats
arrow1 = FancyArrowPatch((5.0, y_start+1.4), (5.0, y_start+0.9),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3.5, color='#1976D2')
ax3.add_patch(arrow1)
ax3.text(6.5, y_start+1.15, 'Creates favorable\ntopography',
        fontsize=8.5, style='italic', color='#1976D2', va='center')

# Topography
box2 = FancyBboxPatch((0.7, y_start), box_w, 1.1, boxstyle="round,pad=0.08",
                      edgecolor='#1565C0', facecolor='#BBDEFB', linewidth=2.5)
ax3.add_patch(box2)
ax3.text(5.0, y_start+0.75, 'FAVORABLE TOPOGRAPHY (Proximate Cause)',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5.0, y_start+0.35, 'Depressions, basins, low relief',
        ha='center', va='center', fontsize=9, style='italic')

y_start -= 1.7

# Arrow
arrow2 = FancyArrowPatch((5.0, y_start+1.7), (5.0, y_start+0.9),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3.5, color='#1976D2')
ax3.add_patch(arrow2)
ax3.text(6.3, y_start+1.3, 'Supports high\nlake density',
        fontsize=8.5, style='italic', color='#1976D2', va='center')

# High Density
box3 = FancyBboxPatch((0.7, y_start), box_w, 0.9, boxstyle="round,pad=0.08",
                      edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2.5)
ax3.add_patch(box3)
ax3.text(5.0, y_start+box_h/2, 'HIGH LAKE DENSITY (~208 lakes/1000 km²)',
        ha='center', va='center', fontsize=11, fontweight='bold')

y_start -= 1.4

# Decay arrow (RED for time)
arrow3 = FancyArrowPatch((5.0, y_start+1.4), (5.0, y_start+0.9),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3.5, color='#D32F2F')
ax3.add_patch(arrow3)
ax3.text(6.5, y_start+1.15, 'TIME\nt½ ≈ 660 ka',
        fontsize=9, fontweight='bold', style='italic', color='#D32F2F', va='center')

# Degradation
box4 = FancyBboxPatch((0.7, y_start), box_w, 1.1, boxstyle="round,pad=0.08",
                      edgecolor='#7B1FA2', facecolor='#E1BEE7', linewidth=2.5)
ax3.add_patch(box4)
ax3.text(5.0, y_start+0.75, 'TOPOGRAPHIC DEGRADATION',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(5.0, y_start+0.35, 'Depressions fill in, relief increases',
        ha='center', va='center', fontsize=9, style='italic')

y_start -= 1.7

# Arrow
arrow4 = FancyArrowPatch((5.0, y_start+1.7), (5.0, y_start+0.9),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3.5, color='#D32F2F')
ax3.add_patch(arrow4)
ax3.text(6.3, y_start+1.3, 'Results in',
        fontsize=8.5, style='italic', color='#D32F2F', va='center')

# Low Density
box5 = FancyBboxPatch((0.7, y_start), box_w, 0.9, boxstyle="round,pad=0.08",
                      edgecolor='#C62828', facecolor='#FFCDD2', linewidth=2.5)
ax3.add_patch(box5)
ax3.text(5.0, y_start+box_h/2, 'LOW LAKE DENSITY (~69 lakes/1000 km²)',
        ha='center', va='center', fontsize=11, fontweight='bold')

# MEDIATION RESULTS BOX
med_box = FancyBboxPatch((0.5, 0.3), 9.0, 2.0, boxstyle="round,pad=0.1",
                        edgecolor='darkred', facecolor='#FFF9C4',
                        linewidth=3, alpha=0.95)
ax3.add_patch(med_box)

ax3.text(5.0, 2.0, 'MEDIATION ANALYSIS RESULTS', ha='center', va='center',
        fontsize=11, fontweight='bold', color='darkred')

results_text = [
    'Total Effect (c):  β = -66.987,  R² = 0.029  →  Glaciation reduces density',
    'Direct Effect (c′): β = -31.812                  →  After controlling for topography',
    'Indirect Effect:     β = -35.175  (52.5% mediated) →  Through topographic pathway',
    '',
    'KEY FINDING: Glaciation acts PRIMARILY through creating favorable topography!',
    'Adding topography to model: ΔR² = +0.078 (R²: 0.029 → 0.108)',
]

y_text = 1.65
for line in results_text:
    if 'KEY FINDING' in line:
        ax3.text(5.0, y_text, line, ha='center', va='center',
                fontsize=9, fontweight='bold', color='darkred', style='italic')
    else:
        ax3.text(5.0, y_text, line, ha='center', va='center',
                fontsize=8, family='monospace')
    y_text -= 0.3

# ============================================================================
# Panel 4: KEY FINDINGS INVENTORY (RIGHT MIDDLE)
# ============================================================================

ax4 = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=2)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('D) Key Findings: 15+ Analyses Performed',
              fontsize=12, fontweight='bold', loc='left', pad=15)

y_pos = 9.5
x_left = 0.3

# Function to add category header
def add_header(text, color):
    global y_pos
    ax4.text(x_left, y_pos, text, fontsize=10, fontweight='bold', color=color)
    y_pos -= 0.45

# Function to add finding
def add_finding(name, result):
    global y_pos
    ax4.text(x_left + 0.3, y_pos, f'• {name}:', fontsize=8.5, fontweight='bold')
    y_pos -= 0.32
    ax4.text(x_left + 0.6, y_pos, result, fontsize=7.5, style='italic')
    y_pos -= 0.42

# Temporal analyses
add_header('TEMPORAL ANALYSES', '#2E7D32')
add_finding('Bayesian Half-Life', 't½ = 660 ka [95% CI: 418-1505 ka]')
add_finding('Chronosequence', 'Wisconsin (228) > Illinoian (202) > Driftless (69)')
add_finding('Size-Stratified', 'Half-life independent of lake size')
add_finding('NADI-1 Time Slices', '0-25 ka at 0.5 ka intervals')
add_finding('Threshold Sensitivity', 'Plateaus at thresholds >1 km²')

y_pos -= 0.3

# Spatial analyses
add_header('SPATIAL ANALYSES', '#1565C0')
add_finding('Multivariate Regression', 'R² = 0.128, all predictors p<0.001')
add_finding('Variance Partitioning', 'Topo 30.5%, Climate 16.1%, Glac 9.5%')
add_finding('PCA', 'PC1 (40.4%): Climate, PC2 (25.1%): Topography')
add_finding('Shared Variance', '43.9% (glaciation ↔ topography coupled)')

y_pos -= 0.3

# Mechanistic analyses
add_header('MECHANISTIC ANALYSES', '#7B1FA2')
add_finding('Mediation Analysis', '53% of effect through topography')
add_finding('Topographic Comparison', 'Glaciated regions smoother (lower relief)')
add_finding('Direct vs Indirect', 'Indirect pathway dominant')

y_pos -= 0.3

# Comparative analyses
add_header('COMPARATIVE ANALYSES', '#C62828')
add_finding('S. Appalachian', 'Density 101.9, between Illinoian & Wisconsin')
add_finding('Aridity Analysis', 'Climate important but not dominant')
add_finding('Spatial Scaling', 'Latitudinal & elevational gradients')

y_pos -= 0.5

# Dataset summary box
summary_box = FancyBboxPatch((0.2, y_pos-2.8), 9.6, 2.7, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#E8F5E9',
                            linewidth=2, alpha=0.9)
ax4.add_patch(summary_box)

ax4.text(5.0, y_pos-0.4, 'DATASET & COVERAGE', ha='center', va='center',
        fontsize=10, fontweight='bold')

y_pos -= 0.75

summary_items = [
    ('Total lakes (CONUS):', '4,896,094'),
    ('After filtering (0.005-20000 km²):', '1,173,057'),
    ('Grid cells (0.5° resolution):', '3,589'),
    ('Wisconsin cells:', '621 (density: 206.6)'),
    ('Illinoian cells:', '54 (density: 225.2)'),
    ('Driftless cells:', '8 (density: 78.0)'),
    ('Unclassified cells:', '2,906 (density: 122.3)'),
    ('Variables analyzed:', 'Elevation, slope, relief, aridity, precip'),
]

for label, value in summary_items:
    ax4.text(x_left + 0.5, y_pos, label, fontsize=7.5, fontweight='bold')
    ax4.text(7.5, y_pos, value, fontsize=7.5, ha='right')
    y_pos -= 0.3

# ============================================================================
# Panel 5: INTEGRATION MESSAGE (BOTTOM)
# ============================================================================

ax5 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')

# Large integration box
integ_box = FancyBboxPatch((0.2, 2.0), 9.6, 7.5, boxstyle="round,pad=0.15",
                          edgecolor='darkblue', facecolor='#E3F2FD',
                          linewidth=3.5, alpha=0.95)
ax5.add_patch(integ_box)

ax5.text(5.0, 9.0, 'THE COMPLETE STORY: Integration of All Analyses',
        ha='center', va='center', fontsize=14, fontweight='bold', color='darkblue')

story_text = [
    ('TEMPORAL PERSPECTIVE (Half-Life Analysis):', '#2E7D32', 'bold'),
    ('Lake density decays exponentially with time since glaciation (t½ ≈ 660 ka).', '#2E7D32', 'normal'),
    ('Wisconsin (20 ka): 228 lakes/1000 km²  →  Illinoian (160 ka): 202  →  Driftless (>1.5 Ma): 69', '#2E7D32', 'italic'),
    ('', 'black', 'normal'),
    ('SPATIAL PERSPECTIVE (Multivariate Analysis):', '#1565C0', 'bold'),
    ('At any snapshot in time, topography is the strongest predictor of lake density (30.5%).', '#1565C0', 'normal'),
    ('Climate (16.1%) and glaciation (9.5%) have smaller pure effects.', '#1565C0', 'normal'),
    ('Large shared variance (43.9%) shows these factors are intertwined.', '#1565C0', 'italic'),
    ('', 'black', 'normal'),
    ('MECHANISTIC PERSPECTIVE (Mediation Analysis):', '#7B1FA2', 'bold'),
    ('Glaciation acts 53% INDIRECTLY through creating favorable topography.', '#7B1FA2', 'normal'),
    ('Favorable topography (depressions, low relief) supports high lake density.', '#7B1FA2', 'normal'),
    ('Over time (~660 ka), topography degrades and lakes disappear.', '#7B1FA2', 'italic'),
    ('', 'black', 'normal'),
    ('UNIFIED NARRATIVE:', 'darkred', 'bold'),
    ('Glaciation (ultimate cause) → Creates favorable topography (proximate cause) →', 'darkred', 'normal'),
    ('High lake density → Topographic degradation → Low lake density (t½ ≈ 660 ka)', 'darkred', 'normal'),
]

y_text = 8.2
for text, color, weight in story_text:
    if weight == 'bold':
        ax5.text(0.5, y_text, text, fontsize=10, fontweight='bold', color=color)
    elif weight == 'italic':
        ax5.text(0.8, y_text, text, fontsize=9, style='italic', color=color)
    else:
        ax5.text(0.8, y_text, text, fontsize=9, color=color)
    y_text -= 0.55

# Bottom conclusion
ax5.text(5.0, 2.8, 'All three perspectives (temporal, spatial, mechanistic) are COMPLEMENTARY,',
        ha='center', fontsize=10, fontweight='bold', color='darkred')
ax5.text(5.0, 2.3, 'not contradictory. Together they reveal the complete process of glacial control on lake density.',
        ha='center', fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('/home/user/lakes/output/comprehensive_analysis_overview.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("IMPROVED COMPREHENSIVE ANALYSIS OVERVIEW CREATED")
print("="*80)
print("\nSaved: /home/user/lakes/output/comprehensive_analysis_overview.png")
print("\nImprovements:")
print("  ✓ Fixed overlapping labels")
print("  ✓ Better spacing and layout")
print("  ✓ Added code structure flowchart (Panel A)")
print("  ✓ Incorporated mediation results (53% indirect effect)")
print("  ✓ More professional color scheme")
print("  ✓ Clearer visual hierarchy")
print("  ✓ Integration message at bottom ties everything together")
print("\n" + "="*80)

plt.show()
