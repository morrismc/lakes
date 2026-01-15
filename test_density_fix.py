#!/usr/bin/env python
"""
Test Density Fix
================

Quick test to verify that the min_lake_area filtering fix
restores correct lake density ordering:

Expected:
- Wisconsin > Illinoian > Driftless
- Wisconsin/Illinoian ratio ~ 9.5

Previous (WRONG):
- Illinoian (1117) > Wisconsin (643) > Driftless (343)

After fix (should be):
- Wisconsin (~228) > Illinoian (~203) > Driftless (~69)
"""

print("=" * 80)
print("TESTING DENSITY FIX - Min Lake Area Filtering")
print("=" * 80)

# Test 1: Import and load data
print("\n[1/4] Loading data...")
try:
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

    lakes = load_lake_data_from_parquet()
    print(f"  Loaded {len(lakes):,} lakes")

    lakes_gdf = convert_lakes_to_gdf(lakes)
    print(f"  Converted to GeoDataFrame")

except Exception as e:
    print(f"  ERROR: {e}")
    import sys
    sys.exit(1)

# Test 2: Load boundaries and classify
print("\n[2/4] Classifying lakes by glacial extent...")
try:
    boundaries = {
        'wisconsin': load_wisconsin_extent(),
        'illinoian': load_illinoian_extent(),
        'driftless': load_driftless_area()
    }

    lakes_classified = classify_lakes_by_glacial_extent(lakes_gdf, boundaries, verbose=False)
    print(f"  Classified {len(lakes_classified):,} lakes")

except Exception as e:
    print(f"  ERROR: {e}")
    import sys
    sys.exit(1)

# Test 3: Apply filters as in analyze_bayesian_halflife
print("\n[3/4] Applying size filters...")
min_lake_area = 0.05
max_lake_area = 20000
area_col = COLS.get('area', 'AREASQKM')

n_original = len(lakes_classified)

# THIS IS THE CRITICAL FIX - Apply BOTH filters
lakes_filtered = lakes_classified[
    (lakes_classified[area_col] >= min_lake_area) &
    (lakes_classified[area_col] <= max_lake_area)
].copy()

n_filtered = len(lakes_filtered)

print(f"  Original lakes: {n_original:,}")
print(f"  Min area filter: {min_lake_area} km²")
print(f"  Max area filter: {max_lake_area} km²")
print(f"  Filtered lakes: {n_filtered:,}")
print(f"  Removed: {n_original - n_filtered:,} ({100*(n_original-n_filtered)/n_original:.1f}%)")

# Test 4: Compute densities
print("\n[4/4] Computing lake densities by glacial stage...")
try:
    zone_areas = {
        'wisconsin': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Wisconsin'),
        'illinoian': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Illinoian'),
        'driftless': SIZE_STRATIFIED_LANDSCAPE_AREAS.get('Driftless')
    }

    density_by_stage = compute_lake_density_by_glacial_stage(
        lakes_filtered,
        zone_areas=zone_areas,
        verbose=False
    )

    print("\nRESULTS:")
    print("-" * 80)

    # Extract densities
    if density_by_stage is not None and len(density_by_stage) > 0:
        print("\nLake Densities (lakes per 1000 km²):")
        print()

        densities = {}
        for _, row in density_by_stage.iterrows():
            stage = row['glacial_stage']
            density = row['density_per_1000km2']
            count = row['n_lakes']
            area = row['zone_area_km2']
            densities[stage] = density
            print(f"  {stage:15s}: {density:8.1f} ({count:,} lakes / {area:,.0f} km²)")

        # Check ordering
        print("\n" + "-" * 80)
        print("VALIDATION:")
        print("-" * 80)

        wisc = densities.get('Wisconsin', 0)
        ill = densities.get('Illinoian', 0)
        drift = densities.get('Driftless', 0)

        # Expected from earlier analysis
        expected = {
            'Wisconsin': 228.2,
            'Illinoian': 202.8,
            'Driftless': 69.4
        }

        print("\nComparison with earlier results:")
        for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
            current = densities.get(stage, 0)
            exp = expected[stage]
            diff_pct = 100 * (current - exp) / exp
            status = "✓" if abs(diff_pct) < 20 else "✗"
            print(f"  {stage:15s}: {current:8.1f} vs {exp:8.1f} (expected)  [{diff_pct:+6.1f}%] {status}")

        print("\nDensity ordering:")
        if wisc > ill > drift:
            print("  ✓ CORRECT: Wisconsin > Illinoian > Driftless")
        else:
            print("  ✗ WRONG ordering:")
            print(f"    Wisconsin: {wisc:.1f}")
            print(f"    Illinoian: {ill:.1f}")
            print(f"    Driftless: {drift:.1f}")

        print("\nLake count ratios:")
        wisc_count = density_by_stage[density_by_stage['glacial_stage'] == 'Wisconsin']['n_lakes'].values[0]
        ill_count = density_by_stage[density_by_stage['glacial_stage'] == 'Illinoian']['n_lakes'].values[0]
        drift_count = density_by_stage[density_by_stage['glacial_stage'] == 'Driftless']['n_lakes'].values[0]

        ratio_wisc_ill = wisc_count / ill_count
        ratio_wisc_drift = wisc_count / drift_count

        print(f"  Wisconsin / Illinoian: {ratio_wisc_ill:.2f}")
        print(f"    Expected: ~9.5")
        if 7 < ratio_wisc_ill < 12:
            print(f"    ✓ Within expected range")
        else:
            print(f"    ✗ Outside expected range")

        print(f"  Wisconsin / Driftless: {ratio_wisc_drift:.2f}")
        print(f"    Expected: ~158")
        if 100 < ratio_wisc_drift < 200:
            print(f"    ✓ Within expected range")
        else:
            print(f"    ✗ Outside expected range")

    else:
        print("ERROR: No density data returned")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
