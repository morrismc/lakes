"""
Size-binned half-life: the canonical headline model (floor_nb) fitted
separately within NON-CUMULATIVE lake-size bins.

Why: the canonical threshold sweep (>= cutoff) gave t1/2 from ~850 ka
(>= 0.005 km2) down to ~45 ka (>= 0.1 km2). Cumulative cutoffs mix size
classes; bins isolate each one, so we can see whether the long half-life
belongs to the smallest lakes alone (constructed ponds / real persistence)
or is spread across sizes.

Reuses geometry, Qal mask (age_consistent), grid cells, priors and model from
regenerate_canonical.py, so the only thing that changes is which lakes count.

Usage (repo root, `lakes` env):
  conda run -n lakes python scripts/size_binned_halflife.py               # fits
  conda run -n lakes python scripts/size_binned_halflife.py --counts-only # just densities
From Claude Code also set PYTENSOR_FLAGS=base_compiledir=D:/Lakes/.pytensor
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import regenerate_canonical as rc  # noqa: E402

TRACK = "age_consistent"
LAKE_SETS = ["all", "no_intermittent"]
BINS = [(0.005, 0.01), (0.01, 0.024), (0.024, 0.05), (0.05, 0.1), (0.1, 1.0), (1.0, 20000.0)]
ZONE_ORDER = [z for z, _ in rc.ZONES]
# Reference categorical palette, slots 1-4 in fixed order (dataviz skill palette.md, light mode)
ZONE_COLORS = {"Wisconsin": "#2a78d6", "Illinoian": "#eb6834", "Pre-Illinoian": "#1baf7a", "Driftless": "#eda100"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"


def bin_label(lo, hi):
    return f"{lo:g}–{hi:g}" if hi < 20000 else f"≥{lo:g}"


def bin_mask(pts, lake_set, lo, hi):
    m = (pts.AREASQKM >= lo) & (pts.AREASQKM < hi)
    if lake_set == "no_intermittent":
        m &= ~pts.intermittent
    return m


def plot(dens, fits, path):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    labels = [bin_label(lo, hi) for lo, hi in BINS]
    x = np.arange(len(BINS))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # A: density per zone by bin (log y), one line per zone, direct-labelled
    d = dens[dens.lake_set == "all"]
    for z in ZONE_ORDER:
        y = d[d.zone == z].set_index("bin").reindex(labels)["density"].to_numpy()
        y = np.where(y > 0, y, np.nan)
        a1.plot(x, y, color=ZONE_COLORS[z], lw=2, marker="o", ms=5, label=z)
        last = np.where(~np.isnan(y))[0]
        if len(last):
            a1.annotate(z, (x[last[-1]], y[last[-1]]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    a1.set_yscale("log")
    a1.set_xticks(x, labels, rotation=30, ha="right")
    a1.set_xlabel("Lake size bin (km²)")
    a1.set_ylabel("Lakes per 1000 km² (log scale)")
    a1.set_title("A. Density by size bin (all lakes)", loc="left", color=INK)
    a1.grid(axis="y", color=GRID, lw=0.6)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    a1.set_xlim(-0.3, len(BINS) - 0.2)

    # B: half-life by bin with 95% interval, both lake sets
    f = fits[fits.model == "floor_nb"]
    for i, (ls, col, off) in enumerate([("all", "#2a78d6", -0.08), ("no_intermittent", "#eb6834", 0.08)]):
        g = f[f.lake_set == ls].set_index("bin").reindex(labels)
        med, lo, hi = g.t_half_median.to_numpy(), g.t_half_lo95.to_numpy(), g.t_half_hi95.to_numpy()
        a2.errorbar(x + off, med, yerr=[med - lo, hi - med], fmt="o", ms=6, color=col, ecolor=col,
                    elinewidth=1.5, capsize=3, label="All lakes" if ls == "all" else "No intermittent ponds")
    a2.set_yscale("log")
    a2.set_xticks(x, labels, rotation=30, ha="right")
    a2.set_xlabel("Lake size bin (km²)")
    a2.set_ylabel("Half-life, ka (log scale)")
    a2.set_title("B. Half-life by size bin (median, 95% interval)", loc="left", color=INK)
    a2.grid(axis="y", color=GRID, lw=0.6)
    a2.legend(frameon=False, fontsize=8)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_md(out_dir, info, dens, fits):
    labels = [bin_label(lo, hi) for lo, hi in BINS]
    L = [f"# Size-binned half-life — {info['date']}\n",
         f"- Commit `{info['commit']}`{' (**dirty**)' if info['dirty'] else ''}; script sha256 `{info['script_sha']}`; "
         f"runtime {info['runtime_min']:.1f} min",
         f"- Track `{TRACK}` (D2 mask), model `floor_nb` (D1 floor, D3 ages, D5 negative binomial on "
         f"{rc.CELL_KM:.0f} km cells) — identical to the canonical headline except for which lakes are counted.",
         "- Bins are non-cumulative: lo ≤ area < hi (km²). Upper bin capped at 20,000 km² (excludes the Great Lakes).\n"]
    for ls in LAKE_SETS:
        d = dens[dens.lake_set == ls]
        L.append(f"## Densities (per 1000 km²), lake set `{ls}`\n")
        L.append("| Bin | " + " | ".join(ZONE_ORDER) + " | W÷I | I÷PI |\n|---|" + "---|" * (len(ZONE_ORDER) + 2))
        for b in labels:
            r = d[d.bin == b].set_index("zone")["density"]
            n = d[d.bin == b].set_index("zone")["n_lakes"]
            cells = " | ".join(f"{r[z]:.2f} ({n[z]:,})" for z in ZONE_ORDER)
            wi = r["Wisconsin"] / r["Illinoian"] if r["Illinoian"] > 0 else np.nan
            ip = r["Illinoian"] / r["Pre-Illinoian"] if r["Pre-Illinoian"] > 0 else np.nan
            L.append(f"| {b} | {cells} | {wi:.2f} | {ip:.2f} |")
        L.append("\nCells show density (lake count).\n")
    L.append("## Half-life by bin (`floor_nb`), median [95%] ka\n")
    L.append("| Bin | all lakes | D∞ (all) | no intermittent | D∞ (no int.) | r̂ max | divergences |\n|---|---|---|---|---|---|---|")
    f = fits[fits.model == "floor_nb"]
    for b in labels:
        ra = f[(f.bin == b) & (f.lake_set == "all")]
        rn = f[(f.bin == b) & (f.lake_set == "no_intermittent")]
        def ci(r, v, nd=0):
            if not len(r):
                return "—"
            r = r.iloc[0]
            return f"{r[f'{v}_median']:.{nd}f} [{r[f'{v}_lo95']:.{nd}f}–{r[f'{v}_hi95']:.{nd}f}]"
        rh = max([x.iloc[0].r_hat_max for x in (ra, rn) if len(x)] or [np.nan])
        dv = sum(int(x.iloc[0].divergences) for x in (ra, rn) if len(x))
        L.append(f"| {b} | {ci(ra, 't_half')} | {ci(ra, 'Dinf', 2)} | {ci(rn, 't_half')} | {ci(rn, 'Dinf', 2)} | {rh:.3f} | {dv} |")
    L.append("\n![size bins](size_bins.png)\n")
    L.append("## Reading this\n")
    L.append("- If only the smallest bin(s) show a very long t½ while larger bins agree on a shorter one, "
             "the long cumulative half-lives come from small ponds. That points to the small-pond population "
             "(including constructed ponds) rather than to a general slowing of lake loss.")
    L.append("- W÷I and I÷PI are model-free. A single exponential with these ages predicts I÷PI ≈ (W÷I)^(490/140) "
             "before the floor. Ratios far from that relationship mean one decay rate cannot describe that bin.")
    L.append("- In large bins, counts on old surfaces are small. The Driftless count for those bins is near zero, "
             "so D∞ is set mostly by the Pre-Illinoian cells.")
    (out_dir / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    from lake_analysis import config as cfg
    import geopandas as gpd
    commit, dirty = rc.git_state()
    today = dt.date.today().isoformat()
    out_dir = Path(cfg.DATA_ROOT) / "Analysis" / f"canonical_sizebins_{today}_{commit}{'-dirty' if dirty else ''}"
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    rc.log(f"output -> {out_dir}")

    masked, areas = rc.build_geometry(cfg)
    pts = rc.classify_lakes(cfg, masked)
    area = areas[areas.track == TRACK].set_index("zone")["masked_area_km2"]

    rows = []
    for ls in LAKE_SETS:
        for lo, hi in BINS:
            vc = pts.loc[bin_mask(pts, ls, lo, hi), f"zone_{TRACK}"].value_counts()
            for z in ZONE_ORDER:
                n = int(vc.get(z, 0))
                rows.append(dict(lake_set=ls, bin=bin_label(lo, hi), lo=lo, hi=hi, zone=z, n_lakes=n,
                                 area_km2=area[z], density=1000 * n / area[z]))
    dens = pd.DataFrame(rows)
    dens.to_csv(out_dir / "densities_by_bin.csv", index=False)
    print(dens[dens.lake_set == "all"].pivot(index="bin", columns="zone", values="n_lakes")
          .reindex([bin_label(lo, hi) for lo, hi in BINS])[ZONE_ORDER].to_string())
    if args.counts_only:
        return

    cells = rc.build_cells(masked[TRACK])
    lk = pts[pts[f"zone_{TRACK}"].notna()][["geometry"]]
    j = gpd.sjoin(lk, cells[["geometry"]].reset_index(names="cid"), predicate="within", how="inner")
    j = j[~j.index.duplicated()]
    lake_cell = j["cid"].reindex(pts.index)

    fits = []
    for ls in LAKE_SETS:
        for lo, hi in BINS:
            b = bin_label(lo, hi)
            sel = bin_mask(pts, ls, lo, hi) & lake_cell.notna()
            n_c = np.bincount(lake_cell[sel].astype(int).to_numpy(), minlength=len(cells))
            rc.log(f"fit floor_nb {ls:16s} bin {b}  ({n_c.sum():,} lakes)")
            idata = rc.fit_negbin_cells(cells, n_c, floor=True)
            r = rc.summarize(idata, "floor_nb", dict(lake_set=ls, bin=b, lo=lo, hi=hi))
            r["n_lakes_in_cells"] = int(n_c.sum())
            fits.append(r)
            idata.to_netcdf(out_dir / "traces" / f"floor_nb_{ls}_{lo:g}_{hi:g}.nc")
            pd.DataFrame(fits).to_csv(out_dir / "fits_by_bin.csv", index=False)
    fits = pd.DataFrame(fits)
    plot(dens, fits, out_dir / "size_bins.png")
    info = dict(date=today, commit=commit, dirty=bool(dirty), script_sha=rc.file_sha(__file__),
                runtime_min=(time.time() - t0) / 60, bins=BINS, track=TRACK)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    write_md(out_dir, info, dens, fits)
    rc.log(f"done in {info['runtime_min']:.1f} min -> {out_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
