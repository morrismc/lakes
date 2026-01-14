#!/usr/bin/env python
"""
Test Bayesian Half-Life Integration
====================================

Simple test to verify that Bayesian half-life analysis is properly integrated.
Tests imports, function signatures, and basic structure.

Author: Lake Analysis Project
"""

import sys

print("=" * 80)
print("BAYESIAN HALF-LIFE INTEGRATION TEST")
print("=" * 80)

# Test 1: Import core functions
print("\n[Test 1] Testing imports...")
try:
    from lake_analysis import (
        analyze_bayesian_halflife,
        fit_overall_bayesian_halflife,
        plot_overall_bayesian_halflife,
        run_size_stratified_analysis,
        BAYESIAN_HALFLIFE_DEFAULTS,
        GLACIAL_STAGES_CONFIG
    )
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check configuration
print("\n[Test 2] Testing configuration...")
try:
    assert isinstance(BAYESIAN_HALFLIFE_DEFAULTS, dict), "BAYESIAN_HALFLIFE_DEFAULTS should be dict"
    assert 'run_overall' in BAYESIAN_HALFLIFE_DEFAULTS, "Missing 'run_overall' key"
    assert 'run_size_stratified' in BAYESIAN_HALFLIFE_DEFAULTS, "Missing 'run_size_stratified' key"
    print(f"  ✓ BAYESIAN_HALFLIFE_DEFAULTS has {len(BAYESIAN_HALFLIFE_DEFAULTS)} parameters")

    assert isinstance(GLACIAL_STAGES_CONFIG, dict), "GLACIAL_STAGES_CONFIG should be dict"
    assert 'Wisconsin' in GLACIAL_STAGES_CONFIG, "Missing 'Wisconsin' stage"
    assert 'Illinoian' in GLACIAL_STAGES_CONFIG, "Missing 'Illinoian' stage"
    assert 'Pre-Illinoian' in GLACIAL_STAGES_CONFIG, "Missing 'Pre-Illinoian' stage"
    assert 'Driftless' in GLACIAL_STAGES_CONFIG, "Missing 'Driftless' stage"

    # Check Pre-Illinoian is marked as not required
    assert GLACIAL_STAGES_CONFIG['Pre-Illinoian']['required'] is False, "Pre-Illinoian should not be required"
    print(f"  ✓ GLACIAL_STAGES_CONFIG has {len(GLACIAL_STAGES_CONFIG)} stages")
    print(f"  ✓ Pre-Illinoian is correctly marked as not required")
except AssertionError as e:
    print(f"  ✗ Configuration test failed: {e}")
    sys.exit(1)

# Test 3: Check function signatures
print("\n[Test 3] Testing function signatures...")
try:
    import inspect

    # Check analyze_bayesian_halflife
    sig = inspect.signature(analyze_bayesian_halflife)
    params = list(sig.parameters.keys())
    required_params = ['lakes', 'run_overall', 'run_size_stratified']
    for param in required_params:
        assert param in params, f"Missing parameter: {param}"
    print(f"  ✓ analyze_bayesian_halflife has correct signature")

    # Check fit_overall_bayesian_halflife
    sig = inspect.signature(fit_overall_bayesian_halflife)
    params = list(sig.parameters.keys())
    assert 'density_by_stage' in params, "Missing 'density_by_stage' parameter"
    print(f"  ✓ fit_overall_bayesian_halflife has correct signature")

    # Check plot_overall_bayesian_halflife
    sig = inspect.signature(plot_overall_bayesian_halflife)
    params = list(sig.parameters.keys())
    assert 'results' in params, "Missing 'results' parameter"
    print(f"  ✓ plot_overall_bayesian_halflife has correct signature")

except AssertionError as e:
    print(f"  ✗ Signature test failed: {e}")
    sys.exit(1)

# Test 4: Check run_full_analysis parameter
print("\n[Test 4] Testing run_full_analysis integration...")
try:
    from lake_analysis import run_full_analysis
    sig = inspect.signature(run_full_analysis)
    params = list(sig.parameters.keys())
    assert 'include_bayesian_halflife' in params, "Missing 'include_bayesian_halflife' parameter"

    # Check default value
    default = sig.parameters['include_bayesian_halflife'].default
    assert default is True, "include_bayesian_halflife should default to True"
    print(f"  ✓ run_full_analysis has 'include_bayesian_halflife' parameter")
    print(f"  ✓ Default value is True")
except AssertionError as e:
    print(f"  ✗ Integration test failed: {e}")
    sys.exit(1)

# Test 5: Check docstrings
print("\n[Test 5] Testing documentation...")
try:
    assert analyze_bayesian_halflife.__doc__ is not None, "Missing docstring for analyze_bayesian_halflife"
    assert "Overall" in analyze_bayesian_halflife.__doc__, "Docstring should mention 'Overall' mode"
    assert "size-stratified" in analyze_bayesian_halflife.__doc__.lower(), "Docstring should mention size-stratified mode"
    print(f"  ✓ analyze_bayesian_halflife has comprehensive docstring")

    assert fit_overall_bayesian_halflife.__doc__ is not None, "Missing docstring for fit_overall_bayesian_halflife"
    print(f"  ✓ fit_overall_bayesian_halflife has docstring")

    assert plot_overall_bayesian_halflife.__doc__ is not None, "Missing docstring for plot_overall_bayesian_halflife"
    print(f"  ✓ plot_overall_bayesian_halflife has docstring")
except AssertionError as e:
    print(f"  ✗ Documentation test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
print("\nBayesian half-life analysis is correctly integrated!")
print("\nNext steps:")
print("  1. Run with actual data to test functionality")
print("  2. Verify outputs are generated correctly")
print("  3. Check that figures and CSV files are created")
print("\nStandalone usage:")
print("  from lake_analysis import analyze_bayesian_halflife")
print("  results = analyze_bayesian_halflife(lakes)")
print("\nIntegrated usage:")
print("  from lake_analysis import run_full_analysis")
print("  results = run_full_analysis(include_bayesian_halflife=True)")
print("\n" + "=" * 80)
