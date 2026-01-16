#!/usr/bin/env python
"""
Diagnostic Script: Investigate Glacial Classification Issues
===========================================================

This script helps diagnose why Illinoian shows unexpectedly high lake density.

Checks:
1. Lake count ratios (should match earlier analysis)
2. Boundary overlaps (Wisconsin shouldn't overlap Illinoian)
3. Size distribution differences between stages
4. Spatial distribution of classifications

Run this after classify_lakes_by_glacial_extent() to check for issues.
"""

import numpy as np
import pandas as pd

def diagnose_classification(lakes_gdf, boundaries, min_lake_area=0.05, max_lake_area=20000):
    """
    Diagnose potential issues with glacial classification.

    Parameters
    ----------
    lakes_gdf : GeoDataFrame
        Classified lakes with 'glacial_stage' column
    boundaries : dict
        Dictionary with boundary GeoDataFrames
    min_lake_area : float
        Minimum lake area used in analysis
    max_lake_area : float
        Maximum lake area used in analysis
    """

    print("=" * 80)
    print("GLACIAL CLASSIFICATION DIAGNOSTIC")
    print("=" * 80)

    # Filter to analysis range
    area_col = 'AREASQKM'
    lakes_filtered = lakes_gdf[
        (lakes_gdf[area_col] >= min_lake_area) &
        (lakes_gdf[area_col] <= max_lake_area)
    ].copy()

    print(f"\nLakes in analysis range ({min_lake_area} - {max_lake_area} km²):")
    print(f"  Total: {len(lakes_filtered):,}")

    # Count by stage
    print("\n" + "-" * 80)
    print("LAKE COUNTS BY GLACIAL STAGE")
    print("-" * 80)

    stage_counts = lakes_filtered['glacial_stage'].value_counts()
    for stage in ['Wisconsin', 'Illinoian', 'Driftless', 'unclassified']:
        if stage in stage_counts.index:
            count = stage_counts[stage]
            pct = 100 * count / len(lakes_filtered)
            print(f"  {stage:15s}: {count:>10,} ({pct:5.1f}%)")

    # Check ratios
    print("\n" + "-" * 80)
    print("LAKE COUNT RATIOS (for validation)")
    print("-" * 80)

    wisc_count = stage_counts.get('Wisconsin', 0)
    ill_count = stage_counts.get('Illinoian', 0)
    drift_count = stage_counts.get('Driftless', 0)

    if wisc_count > 0 and ill_count > 0:
        ratio_wisc_ill = wisc_count / ill_count
        print(f"  Wisconsin / Illinoian: {ratio_wisc_ill:.2f}")
        print(f"    Expected from earlier analysis: ~9.5")
        if ratio_wisc_ill < 7:
            print(f"    ⚠️  WARNING: Ratio too low! Illinoian has too many lakes.")

    if wisc_count > 0 and drift_count > 0:
        ratio_wisc_drift = wisc_count / drift_count
        print(f"  Wisconsin / Driftless: {ratio_wisc_drift:.2f}")
        print(f"    Expected from earlier analysis: ~158")
        if ratio_wisc_drift < 100:
            print(f"    ⚠️  WARNING: Ratio too low! Driftless has too many lakes.")

    # Check boundary areas
    print("\n" + "-" * 80)
    print("BOUNDARY AREAS (from GIS)")
    print("-" * 80)

    for name, gdf in boundaries.items():
        if gdf is not None:
            area_km2 = gdf.geometry.area.sum() / 1e6  # m² to km²
            print(f"  {name:15s}: {area_km2:>12,.0f} km²")

    # Check for potential boundary overlaps
    print("\n" + "-" * 80)
    print("BOUNDARY OVERLAP CHECK")
    print("-" * 80)

    if 'wisconsin' in boundaries and 'illinoian' in boundaries:
        wisc = boundaries['wisconsin']
        ill = boundaries['illinoian']

        if wisc is not None and ill is not None:
            try:
                # Check if any Wisconsin polygons intersect Illinoian
                wisc_union = wisc.geometry.union_all()
                ill_union = ill.geometry.union_all()

                if wisc_union.intersects(ill_union):
                    intersection = wisc_union.intersection(ill_union)
                    overlap_area = intersection.area / 1e6  # m² to km²
                    print(f"  ⚠️  WARNING: Wisconsin and Illinoian boundaries overlap!")
                    print(f"  Overlap area: {overlap_area:,.0f} km²")
                    print(f"  This could cause classification errors.")
                else:
                    print(f"  ✓ No overlap detected between Wisconsin and Illinoian")
            except Exception as e:
                print(f"  Could not check overlap: {e}")

    # Check size distribution by stage
    print("\n" + "-" * 80)
    print("SIZE DISTRIBUTION BY STAGE")
    print("-" * 80)

    for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
        stage_lakes = lakes_filtered[lakes_filtered['glacial_stage'] == stage]
        if len(stage_lakes) > 0:
            sizes = stage_lakes[area_col]
            print(f"\n  {stage}:")
            print(f"    Count: {len(stage_lakes):,}")
            print(f"    Mean size: {sizes.mean():.4f} km²")
            print(f"    Median size: {sizes.median():.4f} km²")
            print(f"    25th percentile: {sizes.quantile(0.25):.4f} km²")
            print(f"    75th percentile: {sizes.quantile(0.75):.4f} km²")

    # Check if there are lakes classified as multiple stages (shouldn't happen!)
    print("\n" + "-" * 80)
    print("MULTIPLE CLASSIFICATION CHECK")
    print("-" * 80)

    # This would require the original classification process to have stored multiple stages
    # For now, just note that each lake should have exactly one stage
    unique_stages_per_lake = lakes_filtered.groupby(lakes_filtered.index)['glacial_stage'].nunique()
    multi_classified = unique_stages_per_lake[unique_stages_per_lake > 1]

    if len(multi_classified) > 0:
        print(f"  ⚠️  WARNING: {len(multi_classified)} lakes classified as multiple stages!")
    else:
        print(f"  ✓ All lakes have single classification")

    # Density calculation check
    print("\n" + "-" * 80)
    print("DENSITY CALCULATION CHECK")
    print("-" * 80)

    # Expected densities from earlier analysis (with proper filtering)
    expected = {
        'Wisconsin': 228.2,
        'Illinoian': 202.8,
        'Driftless': 69.4
    }

    # Calculated densities
    landscape_areas = {
        'Wisconsin': 1_225_000,
        'Illinoian': 145_000,
        'Driftless': 25_500
    }

    print("\n  Comparing with earlier analysis:")
    for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
        count = stage_counts.get(stage, 0)
        area = landscape_areas[stage]
        density = (count / area) * 1000

        exp_density = expected[stage]
        ratio = density / exp_density

        print(f"\n  {stage}:")
        print(f"    Current density: {density:.1f} lakes/1000 km²")
        print(f"    Expected density: {exp_density:.1f} lakes/1000 km²")
        print(f"    Ratio (current/expected): {ratio:.2f}x")

        if ratio > 1.5 or ratio < 0.7:
            print(f"    ⚠️  WARNING: Density differs by >50% from earlier analysis!")

    print("\n" + "=" * 80)
    print("END DIAGNOSTIC")
    print("=" * 80)

    # Return summary for further analysis
    return {
        'stage_counts': stage_counts,
        'ratios': {
            'wisconsin_illinoian': wisc_count / ill_count if ill_count > 0 else None,
            'wisconsin_driftless': wisc_count / drift_count if drift_count > 0 else None
        }
    }


if __name__ == "__main__":
    print("This is a diagnostic module.")
    print("\nUsage:")
    print("  from diagnose_classification import diagnose_classification")
    print("  results = diagnose_classification(lakes_classified, boundaries)")
