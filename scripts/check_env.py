"""
Environment smoke test for the lakes project.

Run from the repo root (works the same from a terminal or Spyder's "Run file"):

    conda run -n lakes python scripts/check_env.py

Checks: package versions, that `import lake_analysis` works, that every
configured data path exists under DATA_ROOT, that the CONUS parquet loads,
and that PyMC can compile and sample a tiny model.
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def section(title):
    print(f"\n=== {title} ===")


section("Python / packages")
print("python", sys.version.split()[0], "@", sys.executable)
import importlib
for name in ["numpy", "pandas", "geopandas", "shapely", "pyogrio", "fiona",
             "pyproj", "rasterio", "pyarrow", "scipy", "statsmodels",
             "sklearn", "matplotlib", "seaborn", "pymc", "pytensor", "arviz"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:12s} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:12s} MISSING ({exc})")

section("import lake_analysis")
t = time.time()
import lake_analysis  # noqa: E402
from lake_analysis import config  # noqa: E402
print(f"  ok ({time.time() - t:.1f} s)")
print("  DATA_ROOT  =", config.DATA_ROOT)
print("  OUTPUT_DIR =", config.OUTPUT_DIR)

section("Configured paths")
paths = {"LAKE_GDB_PATH": config.LAKE_GDB_PATH,
         "LAKE_CONUS_PARQUET_PATH": config.LAKE_CONUS_PARQUET_PATH,
         "OUTPUT_DIR": config.OUTPUT_DIR}
paths.update({f"RASTERS[{k}]": v for k, v in config.RASTERS.items()})
for k, v in getattr(config, "SHAPEFILES", {}).items():
    if v:
        paths[f"SHAPEFILES[{k}]"] = v
for k, v in getattr(config, "GLACIAL_BOUNDARIES", {}).items():
    if isinstance(v, dict) and v.get("path"):
        paths[f"GLACIAL_BOUNDARIES[{k}]"] = v["path"]
if getattr(config, "NADI1_CONFIG", None):
    paths["NADI1_CONFIG[directory]"] = config.NADI1_CONFIG.get("directory")
missing = [k for k, v in paths.items() if not v or not os.path.exists(v)]
print(f"  {len(paths) - len(missing)}/{len(paths)} exist")
for k in missing:
    print("  MISSING:", k, "->", paths[k])

section("Load CONUS parquet")
from lake_analysis.data_loading import load_conus_lake_data  # noqa: E402
t = time.time()
lakes = load_conus_lake_data()
print(f"  {len(lakes):,} rows, {lakes.shape[1]} columns ({time.time() - t:.1f} s)")

section("PyMC compile + sample")
import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
t = time.time()
with pm.Model():
    mu = pm.Normal("mu", 0, 10)
    pm.Normal("y", mu, 1, observed=np.random.default_rng(0).normal(3, 1, 50))
    idata = pm.sample(200, tune=200, chains=2, cores=1, progressbar=False,
                      random_seed=0)
print(f"  posterior mean mu = {float(idata.posterior['mu'].mean()):.2f} "
      f"(expect ~3) ({time.time() - t:.1f} s)")

print("\nALL CHECKS DONE" + ("" if not missing else " (with missing paths)"))
