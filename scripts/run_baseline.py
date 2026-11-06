"""
Baseline re-run of the glacial chronosequence on the current code.

Runs, with the code's own defaults except where noted:
  A. analyze_glacial_chronosequence  -- dual track (all lakes / Qal excluded),
     min_lake_area = 0.005 km2 (the threshold the 2026-04-03 outputs used).
  B. analyze_nadi1_chronosequence    -- NADI-1 path incl. Bayesian decay fit,
     min_lake_area = 0.01 km2 (its default).

Everything is written to a NEW folder (never the April outputs):
    <DATA_ROOT>\\Analysis\\outputs_baseline_<date>_<commit>\\

Run from the repo root, in a terminal or Spyder (same `lakes` env):
    conda run -n lakes python scripts/run_baseline.py
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GLACIAL_MIN_AREA = 0.005
NADI1_MIN_AREA = 0.01


def git_commit():
    try:
        out = subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "rev-parse", "--short", "HEAD"],
            cwd=REPO, capture_output=True, text=True, check=True)
        dirty = subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "status", "--porcelain",
             "--untracked-files=no"],
            cwd=REPO, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def redirect_outputs(new_dir):
    """Point OUTPUT_DIR at new_dir in config and every module that copied it."""
    import lake_analysis  # noqa: F401  (imports all submodules)
    n = 0
    for name, mod in list(sys.modules.items()):
        if (name == "lake_analysis" or name.startswith("lake_analysis.")) and hasattr(mod, "OUTPUT_DIR"):
            setattr(mod, "OUTPUT_DIR", str(new_dir))
            n += 1
    return n


def main():
    from lake_analysis import config
    commit = git_commit()
    today = dt.date.today().isoformat()
    out_dir = Path(config.DATA_ROOT) / "Analysis" / f"outputs_baseline_{today}_{commit}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_mod = redirect_outputs(out_dir)
    print(f"Output folder: {out_dir}  (OUTPUT_DIR patched in {n_mod} modules)")

    import numpy, pandas, geopandas, pymc, arviz  # noqa: E401
    info = {
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "glacial_min_lake_area_km2": GLACIAL_MIN_AREA,
        "nadi1_min_lake_area_km2": NADI1_MIN_AREA,
        "versions": {m.__name__: m.__version__ for m in (numpy, pandas, geopandas, pymc, arviz)},
        "python": sys.version.split()[0],
    }

    from lake_analysis.data_loading import load_conus_lake_data
    from lake_analysis.main import analyze_glacial_chronosequence, analyze_nadi1_chronosequence

    lakes = load_conus_lake_data()
    info["n_lakes_loaded"] = int(len(lakes))

    # ---------------- A. dual-track glacial chronosequence ----------------
    t = time.time()
    res_g = analyze_glacial_chronosequence(lakes, save_figures=True, verbose=True,
                                           exclude_qal=True, min_lake_area=GLACIAL_MIN_AREA)
    info["glacial_runtime_min"] = round((time.time() - t) / 60, 1)
    summary = {}
    for track in ("all_lakes", "qal_excluded"):
        r = (res_g or {}).get(track)
        if not r:
            summary[track] = "FAILED"
            continue
        dens = r.get("density_by_stage")
        if dens is not None:
            dens.drop(columns=[c for c in ("color",) if c in dens.columns]).to_csv(
                out_dir / f"baseline_density_{track}.csv", index=False)
        davis = r.get("davis_test") or r.get("davis_hypothesis") or {}
        summary[track] = {k: (float(v) if hasattr(v, "__float__") and not isinstance(v, bool) else v)
                          for k, v in davis.items() if not hasattr(v, "__len__") or isinstance(v, str)}
    info["glacial_davis_tests"] = summary
    info["glacial_result_keys"] = {k: sorted(v.keys()) if isinstance(v, dict) else type(v).__name__
                                   for k, v in (res_g or {}).items() if k in ("all_lakes", "qal_excluded")}

    # ---------------- B. NADI-1 chronosequence + Bayesian ----------------
    t = time.time()
    res_n = analyze_nadi1_chronosequence(lakes=lakes, min_lake_area=NADI1_MIN_AREA,
                                         save_figures=True, verbose=True)
    info["nadi1_runtime_min"] = round((time.time() - t) / 60, 1)
    if isinstance(res_n, dict):
        info["nadi1_result_keys"] = sorted(res_n.keys())
        dm = res_n.get("decay_model")
        if dm:
            info["nadi1_decay_model"] = {k: (list(map(float, v)) if isinstance(v, tuple) else
                                             float(v) if hasattr(v, "__float__") else str(v))
                                         for k, v in dm.items()}
        for key in ("end_member_data", "end_members", "density_by_stage"):
            v = res_n.get(key)
            if v is not None:
                try:
                    import pandas as pd
                    pd.DataFrame(v).to_csv(out_dir / f"baseline_nadi1_{key}.csv", index=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"could not save {key}: {exc}")
        dba = res_n.get("density_by_age")
        if dba is not None:
            dba.to_csv(out_dir / "baseline_nadi1_density_by_age.csv", index=False)

    info["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print("\n" + "=" * 70 + "\nBASELINE RUN INFO\n" + json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # no GUI windows when run from a terminal
    main()
