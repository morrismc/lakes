#!/usr/bin/env python
"""
Investigate Density Inconsistency
==================================

The half-life and lake densities are changing between analyses!

Current run:
- Wisconsin: 41.25, Illinoian: 16.07, Driftless: 7.73
- Half-life: 169 ka

Earlier run (from figure):
- Wisconsin: ~140, Illinoian: ~95, Driftless: ~35
- Half-life: 661 ka

Even earlier (from conversation):
- Wisconsin: 228.2, Illinoian: 202.8, Driftless: 69.4

This script investigates:
1. What min_lake_area was used in earlier analyses?
2. Are the glacial boundaries consistent?
3. Is the area calculation method different?
4. Why is the Wisconsin/Illinoian ratio 21.7 instead of 9.5?
"""

import numpy as np
import pandas as pd
from lake_analysis import (
    load_lake_data_from_parquet,
    convert_lakes_to_gdf,
    load_wisconsin_extent,
    load_illinoian_extent,
    load_driftless_area,
    classify_lakes_by_glacial_extent,
    compute_lake_density_by_glacial_stage,
    SIZE_STRATIFIED_LANDSCAPE_AREAS,
    COLS
)

print("=" * 80)
print("INVESTIGATING DENSITY INCONSISTENCY")
print("=" * 80)

# Load and classify
print("\n[1/5] Loading and classifying lakes...")
lakes = load_lake_data_from_parquet()
lakes_gdf = convert_lakes_to_gdf(lakes)
boundaries = {
    'wisconsin': load_wisconsin_extent(),
    'illinoian': load_illinoian_extent(),
    'driftless': load_driftless_area()
}
lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries, verbose=False)

area_col = COLS.get('area', 'AREASQKM')

print(f"\nTotal classified lakes: {len(lakes_classified):,}")
for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
    n = len(lakes_classified[lakes_classified['glacial_stage'] == stage])
    print(f"  {stage}: {n:,}")

# Test different min_lake_area thresholds
print("\n[2/5] Testing different min_lake_area thresholds...")
print("-" * 80)

thresholds = [0.01, 0.024, 0.05, 0.1]
results = []

for min_area in thresholds:
    # Filter by min area only (still exclude Great Lakes with max)
    lakes_filtered = lakes_classified[
        (lakes_classified[area_col] >= min_area) &
        (lakes_classified[area_col] <= 20000)
    ].copy()

    # Get counts
    wisc_count = len(lakes_filtered[lakes_filtered['glacial_stage'] == 'Wisconsin'])
    ill_count = len(lakes_filtered[lakes_filtered['glacial_stage'] == 'Illinoian'])
    drift_count = len(lakes_filtered[lakes_filtered['glacial_stage'] == 'Driftless'])

    # Calculate densities using landscape areas
    zone_areas = {
        'wisconsin': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Wisconsin'),
        'illinoian': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Illinoian'),
        'driftless': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Driftless')
    }

    wisc_density = (wisc_count / zone_areas['wisconsin']) * 1000 if zone_areas['wisconsin'] else np.nan
    ill_density = (ill_count / zone_areas['illinoian']) * 1000 if zone_areas['illinoian'] else np.nan
    drift_density = (drift_count / zone_areas['driftless']) * 1000 if zone_areas['driftless'] else np.nan

    ratio = wisc_count / ill_count if ill_count > 0 else np.nan

    results.append({
        'min_area': min_area,
        'wisc_count': wisc_count,
        'ill_count': ill_count,
        'drift_count': drift_count,
        'wisc_density': wisc_density,
        'ill_density': ill_density,
        'drift_density': drift_density,
        'wisc_ill_ratio': ratio
    })

    print(f"\nmin_area = {min_area} km²:")
    print(f"  Lake counts: Wisconsin={wisc_count:,}, Illinoian={ill_count:,}, Driftless={drift_count:,}")
    print(f"  Ratio (W/I): {ratio:.2f}")
    print(f"  Densities (per 1000 km²):")
    print(f"    Wisconsin: {wisc_density:.1f}")
    print(f"    Illinoian: {ill_density:.1f}")
    print(f"    Driftless: {drift_density:.1f}")

    # Check against expected values
    print(f"  Comparison to earlier results:")
    if abs(wisc_density - 228.2) < 20:
        print(f"    Wisconsin: MATCHES 228.2 ✓")
    elif abs(wisc_density - 140) < 20:
        print(f"    Wisconsin: MATCHES 140 from figure ✓")
    else:
        print(f"    Wisconsin: NO MATCH (expected 228.2 or 140)")

    if abs(ratio - 9.5) < 2:
        print(f"    W/I ratio: MATCHES expected 9.5 ✓")
    else:
        print(f"    W/I ratio: NO MATCH (expected 9.5)")

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Test what min_area gives Wisconsin/Illinoian ratio of 9.5
print("\n[3/5] Finding min_area that gives W/I ratio ≈ 9.5...")
print("-" * 80)

target_ratio = 9.5
best_match = None
best_diff = float('inf')

for min_area_test in np.arange(0.001, 0.5, 0.005):
    lakes_test = lakes_classified[
        (lakes_classified[area_col] >= min_area_test) &
        (lakes_classified[area_col] <= 20000)
    ].copy()

    wisc_count = len(lakes_test[lakes_test['glacial_stage'] == 'Wisconsin'])
    ill_count = len(lakes_test[lakes_test['glacial_stage'] == 'Illinoian'])

    if ill_count > 0:
        ratio = wisc_count / ill_count
        diff = abs(ratio - target_ratio)

        if diff < best_diff:
            best_diff = diff
            best_match = {
                'min_area': min_area_test,
                'ratio': ratio,
                'wisc': wisc_count,
                'ill': ill_count
            }

if best_match:
    print(f"\nClosest match to W/I ratio = 9.5:")
    print(f"  min_area = {best_match['min_area']:.3f} km²")
    print(f"  Ratio = {best_match['ratio']:.2f}")
    print(f"  Wisconsin: {best_match['wisc']:,} lakes")
    print(f"  Illinoian: {best_match['ill']:,} lakes")

    # Calculate densities at this threshold
    zone_areas = {
        'wisconsin': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Wisconsin'),
        'illinoian': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Illinoian'),
        'driftless': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Driftless')
    }

    lakes_best = lakes_classified[
        (lakes_classified[area_col] >= best_match['min_area']) &
        (lakes_classified[area_col] <= 20000)
    ].copy()

    drift_count = len(lakes_best[lakes_best['glacial_stage'] == 'Driftless'])

    wisc_density = (best_match['wisc'] / zone_areas['wisconsin']) * 1000
    ill_density = (best_match['ill'] / zone_areas['illinoian']) * 1000
    drift_density = (drift_count / zone_areas['driftless']) * 1000

    print(f"\n  Resulting densities:")
    print(f"    Wisconsin: {wisc_density:.1f} (expected: 228.2)")
    print(f"    Illinoian: {ill_density:.1f} (expected: 202.8)")
    print(f"    Driftless: {drift_density:.1f} (expected: 69.4)")

# Check landscape area calculations
print("\n[4/5] Checking landscape area calculations...")
print("-" * 80)

print("\nFrom SIZE_STRATIFIED_LANDSCAPE_AREAS (config.py):")
for key, val in SIZE_STRATIFIED_LANDSCAPE_AREAS.items():
    print(f"  {key}: {val:,.0f} km²")

print("\nFrom actual boundary GeoDataFrames:")
for name, boundary_key in [('Wisconsin', 'wisconsin'), ('Illinoian', 'illinoian'), ('Driftless', 'driftless')]:
    boundary = boundaries[boundary_key]
    actual_area = boundary.geometry.area.sum() / 1e6  # Convert m² to km²
    config_area = SIZE_STRATIFIED_LANDSCAPE_AREAS.get(name)
    diff = actual_area - config_area if config_area else np.nan
    diff_pct = (diff / config_area * 100) if config_area else np.nan

    print(f"  {name}: {actual_area:,.0f} km² (diff: {diff:+,.0f} km², {diff_pct:+.1f}%)")

# Check if issue is boundary-related
print("\n[5/5] Checking for boundary classification issues...")
print("-" * 80)

# Check size distribution by stage
print("\nSize distribution by glacial stage:")
for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
    stage_lakes = lakes_classified[lakes_classified['glacial_stage'] == stage]

    if len(stage_lakes) > 0:
        sizes = stage_lakes[area_col].values

        print(f"\n{stage}:")
        print(f"  Total: {len(stage_lakes):,} lakes")
        print(f"  Min: {sizes.min():.4f} km²")
        print(f"  5th percentile: {np.percentile(sizes, 5):.4f} km²")
        print(f"  Median: {np.median(sizes):.4f} km²")
        print(f"  Mean: {np.mean(sizes):.4f} km²")
        print(f"  95th percentile: {np.percentile(sizes, 95):.4f} km²")
        print(f"  Max: {sizes.max():.1f} km²")

        # Count by size bins
        n_tiny = np.sum((sizes >= 0.001) & (sizes < 0.01))
        n_very_small = np.sum((sizes >= 0.01) & (sizes < 0.05))
        n_small = np.sum((sizes >= 0.05) & (sizes < 0.1))
        n_medium = np.sum((sizes >= 0.1) & (sizes < 1.0))
        n_large = np.sum(sizes >= 1.0)

        print(f"  Size breakdown:")
        print(f"    < 0.01 km²: {n_tiny:,} ({n_tiny/len(stage_lakes)*100:.1f}%)")
        print(f"    0.01-0.05 km²: {n_very_small:,} ({n_very_small/len(stage_lakes)*100:.1f}%)")
        print(f"    0.05-0.1 km²: {n_small:,} ({n_small/len(stage_lakes)*100:.1f}%)")
        print(f"    0.1-1.0 km²: {n_medium:,} ({n_medium/len(stage_lakes)*100:.1f}%)")
        print(f"    >= 1.0 km²: {n_large:,} ({n_large/len(stage_lakes)*100:.1f}%)")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

print("\nKey Findings:")
print("-" * 80)

# Find which threshold matches earlier results
best_threshold = None
for r in results:
    if abs(r['wisc_ill_ratio'] - 9.5) < 2:
        best_threshold = r['min_area']
        print(f"✓ min_area = {best_threshold} km² gives W/I ratio ≈ 9.5")
        print(f"  Densities: Wisconsin={r['wisc_density']:.1f}, Illinoian={r['ill_density']:.1f}")
        break

if not best_threshold:
    print("✗ None of the tested thresholds match expected W/I ratio of 9.5")
    print(f"  Current (min_area=0.05): ratio={results[2]['wisc_ill_ratio']:.1f}")
    print(f"  This suggests the earlier analysis may have used DIFFERENT boundaries")
    print(f"  or DIFFERENT landscape area calculations")

print("\nRecommendation:")
print("-" * 80)
print("The inconsistent results suggest one of the following:")
print("  1. Earlier analyses used a different min_lake_area threshold")
print("  2. The glacial boundary shapefiles have changed")
print("  3. The landscape area calculations are different")
print("  4. The classification method has changed")
print("\nTo resolve this, we need to:")
print("  - Identify which min_lake_area was used in earlier successful analyses")
print("  - Verify the glacial boundary files haven't changed")
print("  - Ensure landscape areas match the actual boundary areas")
