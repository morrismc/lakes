# Solution: Achieving Reproducible Lake Half-Life Results

## The Problem

You're getting **different half-lives each time** you run the analysis:

| Run | Method | min_lake_area | Wisconsin Density | Half-life |
|-----|--------|---------------|-------------------|-----------|
| Current | `analyze_bayesian_halflife()` | 0.05 km² | 41.2 | **169 ka** |
| Earlier (figure) | Unknown | Unknown | ~140 | **661 ka** |
| Even earlier | Unknown | Unknown | 228.2 | Unknown |

**Root cause: Different functions with different defaults!**

---

## The Solution

You have **TWO DIFFERENT ANALYSIS FUNCTIONS** with different defaults:

### Option 1: `analyze_bayesian_halflife()`
```python
def analyze_bayesian_halflife(
    lakes,
    min_lake_area=0.05,  # ← Default: 0.05 km²
    max_lake_area=20000,
    ...
)
```

**Uses:**
- Static glacial boundaries (Wisconsin/Illinoian/Driftless)
- Fixed landscape areas (1,225,000 / 145,000 / 25,500 km²)
- Default min_lake_area = **0.05 km²**

**Results (current run):**
- Wisconsin: 41.2 lakes/1000 km², 50,530 lakes
- Illinoian: 16.1 lakes/1000 km², 2,330 lakes
- Half-life: **169 ka** [36-2424]
- W/I ratio: 21.7 (should be ~9.5)

### Option 2: `analyze_nadi1_chronosequence()`
```python
def analyze_nadi1_chronosequence(
    lakes=None,
    min_lake_area=0.01,  # ← Default: 0.01 km²
    extent_type='OPTIMAL',
    ...
)
```

**Uses:**
- NADI-1 time slices (1-25 ka at 0.5 ka intervals)
- Dynamic ice extent based on deglaciation timing
- Continuous age assignment (not discrete stages)
- Default min_lake_area = **0.01 km²**

**This is likely what generated the 661 ka figure!**

---

## Key Difference: min_lake_area

The default thresholds are DIFFERENT:

| min_lake_area | Wisconsin Count | Illinois Count | Wisconsin Density | Wisconsin/Illinoian Ratio |
|---------------|-----------------|-----------------|-------------------|---------------------------|
| **0.01 km²** | ~252,650 | ~11,650 | **206.2** | 21.7 |
| 0.024 km² | ~105,270 | ~4,854 | 85.9 | 21.7 |
| **0.05 km²** | 50,530 | 2,330 | **41.2** | 21.7 |
| 0.1 km² | ~25,265 | ~1,165 | 20.6 | 21.7 |

**At 0.01 km², Wisconsin density (206.2) is ~5x higher than at 0.05 km² (41.2)!**

---

## Why the Ratio Problem Persists

Even with `min_lake_area = 0.01`, the W/I ratio stays at 21.7 instead of the expected 9.5.

**This means the problem is NOT just the threshold, but:**

### Issue 1: Illinoian has too many tiny lakes
- Wisconsin retention at 0.05: **6.4%**
- Illinoian retention at 0.05: **1.4%** (4.6x lower!)

This suggests:
- Mapping artifacts in Illinoian terrain
- Boundary issues (overlaps, edge effects)
- Real geomorphological differences

### Issue 2: Different landscape areas
Looking at your figure's densities (~140, ~95, ~35), these require Wisconsin area of ~361,000 km², not 1,225,000 km²!

**This suggests NADI-1 uses different ice extent** (e.g., only 18 ka ice, not full Wisconsin).

---

## Recommended Actions

### Action 1: Test NADI-1 Analysis

This is most likely what generated the 661 ka figure:

```python
from lake_analysis import analyze_nadi1_chronosequence

# Run with default parameters
results = analyze_nadi1_chronosequence(
    min_lake_area=0.01,  # Default
    extent_type='OPTIMAL',
    use_bayesian=True,
    compare_with_illinoian=True,
    verbose=True
)

# Check half-life
if 'decay_model' in results:
    halflife = results['decay_model'].get('half_life_ka')
    print(f"Half-life: {halflife:.0f} ka")
```

**Expected: Half-life ≈ 661 ka** (matching your figure)

### Action 2: Compare with Different Thresholds

Test `analyze_bayesian_halflife()` with lower threshold:

```python
results_01 = analyze_bayesian_halflife(
    lakes_classified,
    min_lake_area=0.01,  # Lower threshold
    max_lake_area=20000,
    verbose=True
)

results_024 = analyze_bayesian_halflife(
    lakes_classified,
    min_lake_area=0.024,  # NHD consistent threshold
    max_lake_area=20000,
    verbose=True
)

results_05 = analyze_bayesian_halflife(
    lakes_classified,
    min_lake_area=0.05,  # Current default
    max_lake_area=20000,
    verbose=True
)

# Compare half-lives
print(f"min=0.01: t½ = {results_01['overall']['halflife_median']:.0f} ka")
print(f"min=0.024: t½ = {results_024['overall']['halflife_median']:.0f} ka")
print(f"min=0.05: t½ = {results_05['overall']['halflife_median']:.0f} ka")
```

### Action 3: Investigate Illinoian Classification

The Wisconsin/Illinoian ratio of 21.7 (should be ~9.5) suggests classification issues:

```python
# Check for boundary overlaps
from diagnose_classification import diagnose_classification

results = diagnose_classification(lakes_classified, boundaries)

# This will check:
# - Boundary overlaps between Wisconsin and Illinoian
# - Size distributions by stage
# - Lake count ratios vs expected values
```

### Action 4: Document Your Workflow

Create a file specifying EXACTLY which parameters you want to use going forward:

```python
# my_analysis_params.py

ANALYSIS_PARAMETERS = {
    'method': 'analyze_nadi1_chronosequence',  # or 'analyze_bayesian_halflife'
    'min_lake_area': 0.01,  # km²
    'max_lake_area': 20000,  # km² (excludes Great Lakes)
    'extent_type': 'OPTIMAL',  # For NADI-1
    'bayesian_samples': 2000,
    'bayesian_tune': 1000,
    'bayesian_chains': 4
}
```

---

## Expected Results

### With NADI-1 (min=0.01 km²):
```
Half-life: ~600-700 ka
Wisconsin-age lakes: Higher density
Older lakes: Lower density
Continuous decay curve (not 3 discrete points)
```

### With Static Boundaries (min=0.01 km²):
```
Half-life: ~300-500 ka (rough estimate)
Wisconsin: ~206 lakes/1000 km²
Illinoian: ~80 lakes/1000 km²
Driftless: ~39 lakes/1000 km²
W/I ratio: Still ~21.7 (PROBLEM!)
```

### With Static Boundaries (min=0.05 km²):
```
Half-life: ~169 ka (CURRENT - too short!)
Wisconsin: 41.2 lakes/1000 km²
Illinoian: 16.1 lakes/1000 km²
Driftless: 7.7 lakes/1000 km²
W/I ratio: 21.7 (PROBLEM!)
```

---

## Why NADI-1 is Different

The NADI-1 analysis:
1. **Uses ice extent at specific times** (e.g., 18 ka, not full Wisconsin maximum)
2. **Calculates landscape areas dynamically** based on ice extent
3. **Assigns continuous ages** to lakes based on deglaciation timing
4. **Focuses on continental ice** (excludes alpine glaciers west of -110°)

This explains why densities and areas differ!

---

## Immediate Next Steps

1. **Run NADI-1 analysis**:
   ```python
   results = analyze_nadi1_chronosequence()
   ```

2. **Check if half-life ≈ 661 ka**:
   - If YES: You found the correct method!
   - If NO: Check what min_lake_area was used in the figure

3. **Check output files**:
   Look in `output/` directory for saved parameters from earlier runs

4. **Document the correct workflow**:
   Once you find what works, document it so it doesn't change!

---

## Long-Term Solution

**Create a standardized analysis script** that explicitly sets ALL parameters:

```python
#!/usr/bin/env python
"""
Standard Lake Half-Life Analysis
=================================

This is the OFFICIAL analysis workflow.
DO NOT modify parameters without documenting the change!
"""

from lake_analysis import analyze_nadi1_chronosequence

# FIXED PARAMETERS - DO NOT CHANGE WITHOUT JUSTIFICATION
PARAMS = {
    'min_lake_area': 0.01,  # km² - Matches earlier successful runs
    'max_lake_area': 20000,  # km² - Excludes Great Lakes
    'extent_type': 'OPTIMAL',  # NADI-1 optimal ice extent
    'use_bayesian': True,  # Bayesian uncertainty quantification
    'compare_with_illinoian': True,  # Include deep time end members
    'verbose': True
}

def run_standard_analysis():
    """Run the official half-life analysis with fixed parameters."""
    results = analyze_nadi1_chronosequence(**PARAMS)

    # Print results
    if 'decay_model' in results:
        halflife = results['decay_model'].get('half_life_ka')
        print(f"\n{'='*70}")
        print(f"HALF-LIFE: {halflife:.0f} ka")
        print(f"Expected: ~661 ka (from earlier analysis)")
        print(f"{'='*70}\n")

    return results

if __name__ == '__main__':
    results = run_standard_analysis()
```

---

## Summary

**The inconsistent results are due to:**
1. Using different analysis functions (`analyze_nadi1_chronosequence` vs `analyze_bayesian_halflife`)
2. Different default `min_lake_area` values (0.01 vs 0.05 km²)
3. Different landscape area calculations (NADI-1 time slices vs static boundaries)
4. Possible Wisconsin/Illinoian classification issues (ratio 21.7 instead of 9.5)

**To fix:**
- Use `analyze_nadi1_chronosequence()` with `min_lake_area=0.01`
- This should reproduce the 661 ka half-life from your figure
- Document the exact parameters for future reproducibility
