# =============================================================================
# Bayesian Half-Life Analysis Integration - Implementation Plan
# =============================================================================
#
# This document outlines the complete integration of Bayesian half-life analysis
# into the main workflow, with support for:
# 1. Running as part of run_full_analysis()
# 2. Running standalone (just the Bayesian analysis)
# 3. Overall half-life (all lakes in each glacial stage)
# 4. Size-stratified half-life (broken down by lake size)
# 5. Future pre-Illinoian glacial boundaries
#
# =============================================================================

## COMPONENTS TO ADD/MODIFY:

### 1. NEW FUNCTION: analyze_bayesian_halflife() in main.py
"""
Standalone function that runs the complete Bayesian half-life analysis.
Can be called independently or as part of run_full_analysis().

Parameters:
- lakes: DataFrame (will be converted to GeoDataFrame internally)
- run_overall: bool - Run overall half-life for all lakes per stage
- run_size_stratified: bool - Run size-stratified analysis
- min_lake_area: float - Minimum lake size
- max_lake_area: float - Maximum lake size (exclude Great Lakes)
- min_lakes_per_class: int - Min lakes needed per size class
- save_figures: bool - Generate visualizations
- verbose: bool - Print progress

Returns:
- dict with both 'overall' and 'size_stratified' results
"""

### 2. MODIFY: run_full_analysis() in main.py
"""
Add new parameter:
- include_bayesian_halflife: bool (default=True)

Add new step (after glacial chronosequence):
- Step N: Bayesian Half-Life Analysis
  - Runs overall half-life if glacial analysis ran
  - Runs size-stratified if requested
"""

### 3. NEW FUNCTION: fit_overall_bayesian_halflife() in size_stratified_analysis.py
"""
Fits Bayesian exponential decay model to ALL lakes in each glacial stage
(not broken down by size).

This is the "main" half-life analysis that estimates:
- D₀: Initial lake density (Wisconsin baseline)
- k: Decay rate
- t½: Half-life = ln(2) / k

Uses PyMC to fit: D(t) = D₀ × exp(-k × t)
Where t = age of glacial stage (ka)
"""

### 4. CONFIGURATION: Add to config.py
"""
# Bayesian half-life analysis defaults
BAYESIAN_HALFLIFE_DEFAULTS = {
    'run_overall': True,           # Run overall half-life analysis
    'run_size_stratified': True,   # Run size-stratified analysis
    'min_lake_area': 0.05,         # Minimum lake size (km²)
    'max_lake_area': 20000,        # Maximum lake size (km²)
    'min_lakes_per_class': 10,     # Min lakes per size class
    'n_samples': 2000,             # PyMC samples per chain
    'n_tune': 1000,                # PyMC tuning samples
    'n_chains': 4,                 # PyMC chains
    'target_accept': 0.95          # PyMC target acceptance rate
}

# Supported glacial stages (for future expansion)
GLACIAL_STAGES = {
    'Wisconsin': {
        'age_mean_ka': 20,
        'age_std_ka': 5,
        'boundary_key': 'wisconsin',
        'required': True
    },
    'Illinoian': {
        'age_mean_ka': 160,
        'age_std_ka': 30,
        'boundary_key': 'illinoian',
        'required': True
    },
    'Pre-Illinoian': {
        'age_mean_ka': 500,
        'age_std_ka': 100,
        'boundary_key': 'pre_illinoian',
        'required': False  # Not yet available
    },
    'Driftless': {
        'age_mean_ka': 1500,
        'age_std_ka': 500,
        'boundary_key': 'driftless',
        'required': True
    }
}
"""

## WORKFLOW DIAGRAMS:

### A. INTEGRATED WORKFLOW (run_full_analysis):
```
1-11. [Existing analyses...]
12. Glacial Chronosequence
    ↓ (produces lakes_classified with 'glacial_stage' column)
13. Bayesian Half-Life Analysis ← NEW!
    ├── Overall Half-Life
    │   ├── Fit D(t) = D₀ × exp(-k × t)
    │   ├── Estimate t½ for all stages combined
    │   └── Generate: bayesian_overall_halflife.png
    └── Size-Stratified (optional)
        ├── Split lakes by size bins
        ├── Fit separate model per size class
        ├── Test: t½ ~ lake size relationship
        └── Generate: 3 figures (detection, density, halflife)
14. Spatial Scaling
15. Aridity Analysis
16. Summary Figures
```

### B. STANDALONE WORKFLOW:
```python
from lake_analysis import analyze_bayesian_halflife

# Option 1: Run everything
results = analyze_bayesian_halflife(
    lakes,
    run_overall=True,
    run_size_stratified=True
)

# Option 2: Run only overall half-life
results = analyze_bayesian_halflife(
    lakes,
    run_overall=True,
    run_size_stratified=False
)

# Option 3: Run only size-stratified
results = analyze_bayesian_halflife(
    lakes,
    run_overall=False,
    run_size_stratified=True
)
```

## OUTPUT STRUCTURE:

```python
results = {
    'overall': {
        'halflife_mean': 543.2,      # ka
        'halflife_median': 521.8,
        'halflife_ci_low': 458.1,
        'halflife_ci_high': 649.3,
        'D0': {...},
        'k': {...},
        'trace': <PyMC InferenceData>,
        'summary': <DataFrame>,
        'figures': {
            'decay_curve': 'bayesian_overall_halflife.png',
            'posteriors': 'bayesian_posteriors.png'
        }
    },
    'size_stratified': {
        'density_df': <DataFrame>,
        'halflife_df': <DataFrame>,
        'traces': {...},
        'statistics': {...},
        'figures': {
            'detection': 'detection_limit_diagnostics.png',
            'density': 'size_stratified_density_patterns.png',
            'halflife': 'size_stratified_bayesian_results.png'
        }
    }
}
```

## FILES TO MODIFY:

1. **lake_analysis/main.py** (200 lines to add)
   - Add analyze_bayesian_halflife() function
   - Modify run_full_analysis() to include Bayesian step
   - Update print_analysis_summary() to show half-life results

2. **lake_analysis/size_stratified_analysis.py** (150 lines to add)
   - Add fit_overall_bayesian_halflife() function
   - Add plot_overall_bayesian_halflife() function
   - Modify run_size_stratified_analysis() to optionally run overall first

3. **lake_analysis/config.py** (50 lines to add)
   - Add BAYESIAN_HALFLIFE_DEFAULTS
   - Add GLACIAL_STAGES with future pre-Illinoian support

4. **lake_analysis/__init__.py** (5 lines to add)
   - Export analyze_bayesian_halflife
   - Export fit_overall_bayesian_halflife
   - Export BAYESIAN_HALFLIFE_DEFAULTS

5. **CLAUDE.md** (update workflow section)
6. **COMPLETE_WORKFLOW.md** (add Bayesian section)

## IMPLEMENTATION PRIORITY:

1. **HIGH**: fit_overall_bayesian_halflife() - Core functionality
2. **HIGH**: analyze_bayesian_halflife() - Standalone function
3. **HIGH**: Integration into run_full_analysis()
4. **MEDIUM**: Visualization for overall half-life
5. **MEDIUM**: Update documentation
6. **LOW**: Future pre-Illinoian support (placeholder for now)

## TESTING CHECKLIST:

- [ ] Overall half-life runs standalone
- [ ] Size-stratified runs standalone
- [ ] Both run together
- [ ] Integrated into run_full_analysis()
- [ ] Works with Wisconsin + Illinoian + Driftless
- [ ] Gracefully handles missing PyMC
- [ ] Figures saved correctly
- [ ] Results included in summary
- [ ] Code handles missing glacial_stage column
- [ ] Pre-Illinoian placeholder doesn't break anything
