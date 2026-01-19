"""
Test script to diagnose S. Appalachian lakes projection issue.
"""

from lake_analysis import load_southern_appalachian_lakes

print("=" * 70)
print("TESTING S. APPALACHIAN LAKES PROJECTION")
print("=" * 70)

# Load S. Appalachian lakes
sapp_lakes = load_southern_appalachian_lakes()

if sapp_lakes is not None:
    print(f"\n✓ Loaded {len(sapp_lakes):,} lakes")
    print(f"  CRS: {sapp_lakes.crs}")

    # Check coordinate ranges
    bounds = sapp_lakes.total_bounds
    print(f"\n  Bounding box:")
    print(f"    X (easting):  {bounds[0]:,.2f} to {bounds[2]:,.2f} meters")
    print(f"    Y (northing): {bounds[1]:,.2f} to {bounds[3]:,.2f} meters")

    # Expected ranges for S. Appalachians in ESRI:102039 (Albers Equal Area)
    # Roughly: X = 1,400,000 to 1,800,000 m, Y = 1,300,000 to 1,900,000 m
    print(f"\n  Expected S. Appalachian ranges in Albers projection:")
    print(f"    X (easting):  1,400,000 to 1,800,000 meters")
    print(f"    Y (northing): 1,300,000 to 1,900,000 meters")

    # Check if coordinates look reasonable
    x_ok = 1_000_000 < bounds[0] < 2_500_000 and 1_000_000 < bounds[2] < 2_500_000
    y_ok = 800_000 < bounds[1] < 2_500_000 and 800_000 < bounds[3] < 2_500_000

    if x_ok and y_ok:
        print(f"\n✓ Coordinates look correct (in meters, Albers projection)")
    else:
        print(f"\n✗ WARNING: Coordinates look wrong!")
        if abs(bounds[0]) < 200 and abs(bounds[1]) < 200:
            print(f"  → Coordinates appear to still be in degrees (lat/lon)")
            print(f"  → Reprojection may have failed")

    # Sample a few lakes
    print(f"\n  Sample coordinates (first 5 lakes):")
    for i in range(min(5, len(sapp_lakes))):
        geom = sapp_lakes.geometry.iloc[i]
        print(f"    Lake {i+1}: X={geom.x:,.2f}, Y={geom.y:,.2f}")
