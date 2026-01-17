# Lake Density Inconsistency - Root Cause Analysis

## Problem Statement

The Bayesian half-life analysis is producing **different results each time** using supposedly the same dataset:

### Run 1 (Current - Just Now):
```
Wisconsin:  41.25 lakes/1000 km²
Illinoian:  16.07 lakes/1000 km²
Driftless:   7.73 lakes/1000 km²
Half-life: 169 ka [36-2424]
```

### Run 2 (From earlier figure):
```
Wisconsin: ~140 lakes/1000 km²
Illinoian:  ~95 lakes/1000 km²
Driftless:  ~35 lakes/1000 km²
Half-life: 661 ka [465-1131]
```

### Run 3 (From conversation summary):
```
Wisconsin: 228.2 lakes/1000 km²
Illinoian: 202.8 lakes/1000 km²
Driftless:  69.4 lakes/1000 km²
```

**These densities vary by 3-6x between runs! This is unacceptable for reproducible science.**

---

## Critical Finding: Differential Lake Retention

From your logs, I extracted this alarming pattern:

### BEFORE Size Filtering (Total Classified):
```
Wisconsin: 788,332 lakes
Illinoian: 161,964 lakes
Driftless:   8,746 lakes

Ratio: Wisconsin/Illinoian = 4.9
```

### AFTER Size Filtering (min=0.05 km², max=20,000 km²):
```
Wisconsin: 50,530 lakes  (6.4% retained)
Illinoian:  2,330 lakes  (1.4% retained)  ← PROBLEM!
Driftless:    197 lakes  (2.2% retained)

Ratio: Wisconsin/Illinoian = 21.7  ← WRONG! Should be ~9.5
```

**Illinoian is only retaining 1.4% of lakes, while Wisconsin retains 6.4%!**

This means:
1. Illinoian has **WAY MORE tiny lakes** (< 0.05 km²) proportionally
2. When we filter by min_lake_area=0.05, Illinoian loses most lakes
3. This artificially lowers Illinoian density
4. The ratio changes from 4.9 to 21.7!

---

## Why This Happens: Classification BEFORE Filtering

The current workflow:

```python
# Step 1: Classify ALL lakes (including tiny ones)
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries)
# Wisconsin: 788,332 (includes tiny lakes)
# Illinoian: 161,964 (includes MANY tiny lakes)

# Step 2: Filter by size in analyze_bayesian_halflife()
lakes_filtered = lakes[
    (lakes['area'] >= 0.05) &
    (lakes['area'] <= 20000)
]
# Wisconsin: 50,530 (lost 93.6% - mostly tiny)
# Illinoian: 2,330 (lost 98.6% - almost all tiny!)
```

**The problem**: If Illinoian has more tiny lakes proportionally, the filtered ratio becomes wrong!

---

## Why Does Illinoian Have More Tiny Lakes?

Possible reasons:

### 1. Mapping/Detection Bias
- Illinoian terrain is older, more vegetated
- Harder to detect small water bodies in imagery
- NHD may have artificially inflated tiny lake counts
- Could be flooded fields, temporary ponds misclassified

### 2. Boundary Issues
- Illinoian boundary may include marginal areas
- Edge effects where lakes are partially in/out
- Boundary resolution creates "sliver" polygons

### 3. Real Geomorphological Difference
- Illinoian terrain actually HAS more tiny ponds
- Different glacial processes created different size distributions
- But this seems unlikely given the extreme difference (6.4% vs 1.4%)

### 4. **MOST LIKELY: Different min_lake_area in Earlier Analyses**

Looking at Run 3 (Wisconsin/Illinoian ratio should be ~9.5):
- If we need ratio = 9.5 instead of 21.7
- We need to INCLUDE more Illinoian lakes
- This means using a LOWER min_lake_area threshold!

Let me calculate what threshold was likely used:

```
Current (min=0.05 km²):
  Wisconsin: 50,530
  Illinoian:  2,330
  Ratio: 21.7

To get ratio = 9.5:
  Wisconsin: 50,530
  Illinoian: 50,530 / 9.5 = 5,319

We need 5,319 Illinoian lakes, but we only have 2,330!
```

Wait, this means we need to include MORE Illinoian lakes by lowering the threshold!

Or... **we need FEWER Wisconsin lakes**!

Let me recalculate:
```
Total classified (no size filter):
  Wisconsin: 788,332
  Illinoian: 161,964
  Ratio: 4.9

To get ratio 9.5, we'd need:
  Wisconsin: N
  Illinoian: N / 9.5

But classified ratio is only 4.9, which is LESS than 9.5!
```

This is impossible unless:
1. **The boundaries are WRONG**
2. **Earlier analyses used DIFFERENT boundaries**
3. **The landscape areas are WRONG**

---

## The Real Problem: Landscape Areas

Looking at your logs more carefully:

```
Wisconsin zone_area_km2: 1,225,000 km²
Illinoian zone_area_km2:   145,000 km²

Density = (lake count) / (area) × 1000

Wisconsin: 50,530 / 1,225,000 × 1000 = 41.25 ✓
Illinoian:  2,330 /   145,000 × 1000 = 16.07 ✓
```

The math is correct! But let me check what densities we'd get with Run 3's expected values:

**Run 3 expected:**
```
Wisconsin: 228.2 lakes/1000 km²
Illinoian: 202.8 lakes/1000 km²
```

If these densities are correct, and we have:
- Wisconsin: 50,530 lakes
- Illinoian: 2,330 lakes

Then the areas would need to be:
```
Wisconsin area = 50,530 / (228.2/1000) = 221,400 km²  ← NOT 1,225,000!
Illinoian area =  2,330 / (202.8/1000) =  11,500 km²  ← NOT 145,000!
```

**These are 5.5x and 12.6x SMALLER than current landscape areas!**

This means either:
1. **The landscape areas in config are WRONG**
2. **Earlier analyses used DIFFERENT lake counts**
3. **Earlier analyses used DIFFERENT boundaries** (much smaller extent)

---

## What Changed Between Analyses?

### Theory 1: Different `min_lake_area`
Earlier analysis might have used `min_lake_area = 0.01` or `0.024` km²

Let me estimate lake counts at different thresholds using retention rates:

**At min = 0.01 km²** (assuming uniform distribution):
```
Wisconsin: Could have ~150,000 lakes (rough estimate)
Illinoian: Could have ~15,000 lakes (rough estimate)
Ratio: ~10 ✓ (close to expected 9.5!)

With areas (1,225,000 and 145,000 km²):
  Wisconsin: 150,000 / 1,225,000 × 1000 = 122.4
  Illinoian:  15,000 /   145,000 × 1000 = 103.4
```

Still not matching Run 2's ~140 and ~95...

### Theory 2: Different Landscape Areas
What if earlier analyses calculated areas differently?

**If Run 2 used smaller Wisconsin area:**
```
140 lakes/1000 km² with 50,530 lakes
  → Area = 50,530 / 0.140 = 361,000 km²  (NOT 1,225,000!)
```

**If Run 3 used even smaller area:**
```
228.2 lakes/1000 km² with 50,530 lakes
  → Area = 50,530 / 0.2282 = 221,400 km²
```

### Theory 3: Different Boundaries Entirely
Maybe earlier analyses used:
- Only specific NADI-1 time slices (e.g., 18 ka)
- Different glacial extent definitions
- Continental ice only (excluding Cordilleran)

---

## The Most Likely Explanation

Looking at the **figure you provided** (Bayesian Analysis of Lake Density Decay), I notice:
- Only 3 data points are shown
- The ages are different from current
- The analysis appears to use **NADI-1 time slices** not **static glacial boundaries**!

**I believe the earlier successful analysis used:**
1. **NADI-1 chronosequence** (not Wisconsin/Illinoian/Driftless boundaries)
2. **Continuous deglaciation ages** (not discrete stages)
3. **Dynamic ice extent** based on time slices
4. **Different landscape area calculations** based on NADI-1 extents

This would explain why the densities are so different!

---

## Diagnosis: Wrong Analysis Function!

You're running: `analyze_bayesian_halflife()`
- Uses: Wisconsin/Illinoian/Driftless static boundaries
- Lake counts: 50,530 / 2,330 / 197
- Areas: 1,225,000 / 145,000 / 25,500 km²

You SHOULD be running: `analyze_nadi1_chronosequence()`
- Uses: NADI-1 time slices (1-25 ka)
- Assigns continuous deglaciation ages
- Uses NADI-1 extent areas (not static boundaries)
- This matches the figure you provided!

---

## Solution: Run NADI-1 Analysis

```python
from lake_analysis import analyze_nadi1_chronosequence

# Run the correct analysis
results = analyze_nadi1_chronosequence(
    min_lake_area=0.01,  # Or 0.024, or 0.05 - need to determine
    max_lake_area=20000,
    use_bayesian=True,
    compare_with_illinoian=True
)
```

This should reproduce the earlier results!

---

## Action Items

1. **Verify which analysis was used for the 661 ka figure**
   - Check if it was NADI-1 chronosequence
   - Check what min_lake_area was used

2. **Determine correct min_lake_area threshold**
   - Test 0.01, 0.024, 0.05 km²
   - Compare Wisconsin/Illinoian ratios
   - Match with earlier lake counts

3. **Reconcile landscape areas**
   - Verify Wisconsin area: 1,225,000 km² vs ~361,000 km² vs ~221,400 km²
   - Check if NADI-1 uses different area calculations
   - Understand why areas differ so much

4. **Document the correct workflow**
   - Specify which analysis to use (NADI-1 vs static boundaries)
   - Document the correct parameters
   - Ensure reproducibility

---

## Immediate Test

Run this to see which matches the 661 ka result:

```python
# Option 1: NADI-1 chronosequence (most likely)
results1 = analyze_nadi1_chronosequence()

# Option 2: Static boundaries with different threshold
results2 = analyze_bayesian_halflife(
    lakes_classified,
    min_lake_area=0.01  # Lower threshold
)

# Option 3: Check what parameters were in the figure
# Look for saved parameters or logs from earlier session
```

**My bet: The 661 ka figure came from NADI-1 chronosequence, not static boundaries!**
