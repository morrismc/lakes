#!/usr/bin/env python
"""
Analyze Impact of min_lake_area Threshold
==========================================

Test different min_lake_area thresholds to see which one
matches the earlier successful analyses.

Key metrics to match:
1. Wisconsin/Illinoian ratio should be ~9.5
2. Half-life should be ~661 ka
3. Densities should be reasonable

From user's logs, we know:
- Total classified: Wisconsin=788,332, Illinoian=161,964, Driftless=8,746
- At min=0.05: Wisconsin=50,530 (6.4%), Illinoian=2,330 (1.4%), ratio=21.7
- Expected ratio: ~9.5
"""

import numpy as np

# Data from user's logs
classified_counts = {
    'Wisconsin': 788332,
    'Illinoian': 161964,
    'Driftless': 8746
}

# After filtering with min=0.05, max=20000
filtered_05_counts = {
    'Wisconsin': 50530,
    'Illinoian': 2330,
    'Driftless': 197
}

# Landscape areas
areas_km2 = {
    'Wisconsin': 1225000,
    'Illinoian': 145000,
    'Driftless': 25500
}

print("=" * 80)
print("ANALYZING min_lake_area THRESHOLD IMPACT")
print("=" * 80)

# Calculate retention rates at min=0.05
print("\n[1] Current state (min_lake_area = 0.05 km²):")
print("-" * 80)

for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
    total = classified_counts[stage]
    filtered = filtered_05_counts[stage]
    retention = filtered / total * 100
    density = (filtered / areas_km2[stage]) * 1000

    print(f"\n{stage}:")
    print(f"  Total classified: {total:,}")
    print(f"  After filtering:  {filtered:,}")
    print(f"  Retention rate:   {retention:.1f}%")
    print(f"  Density:          {density:.1f} lakes/1000 km²")

wisc_ill_ratio = filtered_05_counts['Wisconsin'] / filtered_05_counts['Illinoian']
print(f"\nWisconsin/Illinoian ratio: {wisc_ill_ratio:.1f}")
print(f"Expected ratio: ~9.5")
print(f"Error: {abs(wisc_ill_ratio - 9.5):.1f} (should be < 2)")

# The problem: Illinoian has MANY more tiny lakes proportionally
print("\n[2] Why Illinoian retention is so low:")
print("-" * 80)

print("\nIllinoian retains only 1.4% of lakes vs 6.4% for Wisconsin.")
print("This means Illinoian has proportionally MORE tiny lakes (< 0.05 km²)")
print("\nPossible explanations:")
print("  1. Mapping artifacts (NHD detection issues in Illinoian terrain)")
print("  2. Real geomorphology (Illinoian actually has more tiny ponds)")
print("  3. Boundary issues (Illinoian boundary includes marginal areas)")

# Estimate lake counts at different thresholds
print("\n[3] Estimating counts at different thresholds:")
print("-" * 80)

# Assume power-law size distribution: N(>x) ∝ x^(-α)
# Typical α for lakes is ~0.9-1.1 (Cael & Seekell 2016)
# Use α = 1.0 for rough estimate

alpha = 1.0  # Power-law exponent

def estimate_count_at_threshold(count_at_05, alpha, new_threshold):
    """
    Estimate count at new threshold using power-law scaling.
    N(>x) ∝ x^(-α)

    N(>x_new) / N(>x_old) = (x_new / x_old)^(-α)
    """
    ratio = (new_threshold / 0.05) ** (-alpha)
    return count_at_05 * ratio

thresholds = [0.01, 0.024, 0.05, 0.1]

print(f"\nAssuming power-law size distribution with α = {alpha}:")
print()

results = []

for threshold in thresholds:
    print(f"min_lake_area = {threshold} km²:")

    estimates = {}
    for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
        count_05 = filtered_05_counts[stage]
        estimated = estimate_count_at_threshold(count_05, alpha, threshold)
        estimates[stage] = int(estimated)

    # Don't exceed total classified
    for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
        estimates[stage] = min(estimates[stage], classified_counts[stage])

    # Calculate densities
    densities = {}
    for stage in ['Wisconsin', 'Illinoian', 'Driftless']:
        densities[stage] = (estimates[stage] / areas_km2[stage]) * 1000

    ratio = estimates['Wisconsin'] / estimates['Illinoian']

    print(f"  Estimated counts: Wisconsin={estimates['Wisconsin']:,}, "
          f"Illinoian={estimates['Illinoian']:,}, Driftless={estimates['Driftless']:,}")
    print(f"  W/I ratio: {ratio:.1f}")
    print(f"  Densities: Wisconsin={densities['Wisconsin']:.1f}, "
          f"Illinoian={densities['Illinoian']:.1f}, Driftless={densities['Driftless']:.1f}")

    # Check match with earlier results
    match_228 = abs(densities['Wisconsin'] - 228.2) < 30
    match_140 = abs(densities['Wisconsin'] - 140) < 30
    match_ratio = abs(ratio - 9.5) < 2

    if match_228:
        print(f"  ✓ Wisconsin density matches 228.2!")
    if match_140:
        print(f"  ✓ Wisconsin density matches 140 (from figure)!")
    if match_ratio:
        print(f"  ✓ W/I ratio matches expected 9.5!")

    print()

    results.append({
        'threshold': threshold,
        'wisc_count': estimates['Wisconsin'],
        'ill_count': estimates['Illinoian'],
        'drift_count': estimates['Driftless'],
        'wisc_density': densities['Wisconsin'],
        'ill_density': densities['Illinoian'],
        'drift_density': densities['Driftless'],
        'ratio': ratio,
        'match_228': match_228,
        'match_140': match_140,
        'match_ratio': match_ratio
    })

# Find best match
print("\n[4] Best threshold matches:")
print("-" * 80)

print("\nFor Run 3 (Wisconsin=228.2, Illinoian=202.8, Driftless=69.4):")
for r in results:
    if r['match_228'] and r['match_ratio']:
        print(f"  ✓ min_lake_area = {r['threshold']} km²")
        print(f"    Densities: W={r['wisc_density']:.1f}, I={r['ill_density']:.1f}, "
              f"D={r['drift_density']:.1f}")
        print(f"    Ratio: {r['ratio']:.1f}")

print("\nFor Run 2 (Wisconsin=~140, Illinoian=~95, Driftless=~35):")
for r in results:
    if r['match_140'] and r['match_ratio']:
        print(f"  ✓ min_lake_area = {r['threshold']} km²")
        print(f"    Densities: W={r['wisc_density']:.1f}, I={r['ill_density']:.1f}, "
              f"D={r['drift_density']:.1f}")
        print(f"    Ratio: {r['ratio']:.1f}")

# The real issue
print("\n[5] The Real Problem:")
print("-" * 80)

print("\nEven at min_lake_area = 0.01 km², we can't reach Wisconsin density of 228.2")
print("because we'd need more lakes than we have classified!")
print("\nThis suggests one of:")
print("  1. The earlier analysis used DIFFERENT BOUNDARIES (not Wisconsin/Illinoian)")
print("  2. The earlier analysis used NADI-1 chronosequence (time slices)")
print("  3. The landscape AREAS are different (not 1,225,000 km²)")
print("  4. The figure is from a completely different dataset")

print("\n[6] Testing if AREA is the issue:")
print("-" * 80)

# What Wisconsin area would give density=228.2 with current counts?
for threshold, r in zip(thresholds, results):
    required_area = (r['wisc_count'] / (228.2/1000))
    print(f"\nmin_lake_area = {threshold} km²:")
    print(f"  Wisconsin count: {r['wisc_count']:,}")
    print(f"  To get density=228.2, area would need to be: {required_area:,.0f} km²")
    print(f"  Current area: {areas_km2['Wisconsin']:,.0f} km²")
    print(f"  Ratio: {required_area / areas_km2['Wisconsin']:.2f}x")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("\nThe density mismatch is TOO LARGE to explain by min_lake_area alone.")
print("\nMost likely explanations:")
print("  1. Earlier analysis used NADI-1 chronosequence, not static boundaries")
print("  2. Earlier analysis used DIFFERENT landscape area calculations")
print("  3. Earlier analysis used SUBSET of Wisconsin extent (e.g., only 18 ka ice)")
print("\nRecommendation:")
print("  - Check if the 661 ka figure came from analyze_nadi1_chronosequence()")
print("  - Verify the landscape areas match the boundary shapefiles")
print("  - Document which min_lake_area was used in earlier successful runs")
