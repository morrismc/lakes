"""
Canonical glacial-chronosequence numbers, built to the 2026-11-05 decisions
(ORIENTATION_2026-11-04.md §8). This script owns its own classification,
areas, thresholds and model so the headline numbers come from ONE place.

Decisions implemented
---------------------
D1  Driftless is the floor:  D(t) = Dinf + (D0 - Dinf) * exp(-k t).
    k is constrained by Wisconsin, Illinoian and Pre-Illinoian; Driftless
    density informs Dinf. Driftless is not given an age.
D2  Age-consistent Qal mask: Wisconsin zone masks only floodplain alluvium
    ('al'); Illinoian, Pre-Illinoian and Driftless mask 'al' + 'gs' (outwash)
    + '(ag)' (mixed glacial). Masked land is removed from zone areas as well
    as its lakes.
D3  Pre-Illinoian age prior centred on 650 ka (MIS 16), ~450-900 ka.
D4  Headline min lake area 0.01 km2; sweep reported as sensitivity.

Zones are made non-overlapping hierarchically (Wisconsin > Illinoian >
Pre-Illinoian > Driftless-definite), in an equal-area CRS. Lakes are
assigned by their stored centroid (Latitude/Longitude), as the pipeline does.

Models (PyMC; exposure = masked area)
  floor_nb       HEADLINE + sweep (decided 2026-11-05). D1 model with a
                 negative-binomial likelihood on ~50 km grid-cell counts, so
                 spatial heterogeneity sets the uncertainty.
  exp3_nb        plain exponential, glacial cells only, no Driftless
                 (headline comparison: what D1 changes)
  floor_poisson  D1 model, Poisson on whole-zone counts (headline comparison;
  exp3_poisson   not adopted: pins each density to ~1% and bends the ages)

Outputs go to  <DATA_ROOT>/Analysis/canonical_<date>_<commit>/ :
  RESULTS.md, counts_areas.csv, fits.csv, decay_headline.png, traces/*.nc

Usage (repo root, `lakes` env):
  conda run -n lakes python scripts/regenerate_canonical.py            # full
  conda run -n lakes python scripts/regenerate_canonical.py --quick    # headline only
From Claude Code also set PYTENSOR_FLAGS=base_compiledir=D:/Lakes/.pytensor
(see RUNNING.md).
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# SETTINGS (everything that defines the canonical numbers lives here)
# --------------------------------------------------------------------------
EQUAL_AREA_CRS = "ESRI:102039"          # USA Contiguous Albers Equal Area (USGS)
THRESHOLDS = [0.005, 0.01, 0.024, 0.05, 0.1]   # km2
HEADLINE = dict(track="age_consistent", lake_set="all", threshold=0.01)

ZONES = [  # (name, gdb layer) in hierarchy order
    ("Wisconsin", "Wisconsin_area"),
    ("Illinoian", "illinoian_glacial_extent"),
    ("Pre-Illinoian", "Pre_illinoian_glacial_extent"),
    ("Driftless", "definite_driftless_area_never_glaciated"),
]
GLACIAL = ["Wisconsin", "Illinoian", "Pre-Illinoian"]

QAL_UNITS = {"al": "alluvium", "gs": "outwash", "(ag)": "mixed_glacial"}
ALL_UNITS = ["alluvium", "outwash", "mixed_glacial"]
TRACKS = {  # zone -> Qal units masked out (lakes AND land)
    "age_consistent": {"Wisconsin": ["alluvium"], "Illinoian": ALL_UNITS,
                       "Pre-Illinoian": ALL_UNITS, "Driftless": ALL_UNITS},
    "full_mask": {z: ALL_UNITS for z, _ in ZONES},
    "no_mask": {z: [] for z, _ in ZONES},
}

# Intermittent NHD LakePond FCODEs (39001 intermittent; 39005/39006 intermittent
# with high-water / photo-date elevation). 46600 (swamp/marsh) also dropped.
INTERMITTENT_FCODES = {39001, 39005, 39006, 46600}
LAKE_SETS = ["all", "no_intermittent"]

# Age priors, ka.  Normal(mu, sd) unless 'lower' given (truncated normal).
AGE_PRIORS = {
    "Wisconsin": dict(mu=20.0, sd=2.5),                   # ~15-25 ka
    "Illinoian": dict(mu=160.0, sd=15.0),                 # ~130-190 ka
    "Pre-Illinoian": dict(mu=650.0, sd=115.0, lower=400.0),  # MIS 16, ~425-875 ka
}
HALF_LIFE_PRIOR = dict(mu=np.log(500.0), sigma=1.5)      # LogNormal, ka
DENSITY_PRIOR_SD = dict(Dinf=300.0, delta=1000.0, D0=1000.0)  # HalfNormal, lakes/1000 km2

CELL_KM = 50.0          # grid for the negative-binomial variant
MIN_CELL_KM2 = 25.0     # drop slivers smaller than this
SAMPLER = dict(draws=2000, tune=2000, chains=4, cores=1, target_accept=0.95,
               random_seed=20261105, progressbar=False)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def git_state():
    g = ["git", "-c", f"safe.directory={REPO.as_posix()}"]
    try:
        commit = subprocess.run(g + ["rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(g + ["status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return commit, dirty
    except Exception:  # noqa: BLE001
        return "unknown", ""


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def q(x, p):
    return float(np.quantile(np.asarray(x).ravel(), p))


# --------------------------------------------------------------------------
# 1. geometry: hierarchical zones, Qal units, masked zones per track
# --------------------------------------------------------------------------
def build_geometry(cfg):
    import geopandas as gpd
    from shapely import make_valid

    def load(path, layer=None):
        g = gpd.read_file(path, layer=layer)
        g = g.to_crs(EQUAL_AREA_CRS)
        g["geometry"] = g.geometry.apply(make_valid)
        return g

    gdb = cfg.GLACIAL_BOUNDARIES["wisconsin"]["path"]
    raw = {z: load(gdb, layer).union_all() for z, layer in ZONES}
    hier, used = {}, None
    for z, _ in ZONES:
        hier[z] = raw[z] if used is None else raw[z].difference(used)
        used = raw[z] if used is None else used.union(raw[z])

    qal = load(cfg.GLACIAL_BOUNDARIES["quaternary_alluvium"]["path"])
    qal["unit"] = qal["GEOUNIT"].map(QAL_UNITS)
    if qal["unit"].isna().any():
        raise ValueError(f"Unknown Qal GEOUNIT values: {qal.loc[qal.unit.isna(), 'GEOUNIT'].unique()}")
    units = {u: g.union_all() for u, g in qal.groupby("unit")}

    masked, area_rows = {}, []
    for track, rule in TRACKS.items():
        masked[track] = {}
        for z, _ in ZONES:
            geom = hier[z]
            for u in rule[z]:
                geom = geom.difference(units[u])
            masked[track][z] = geom
            area_rows.append(dict(track=track, zone=z,
                                  raw_area_km2=raw[z].area / 1e6,
                                  hier_area_km2=hier[z].area / 1e6,
                                  masked_area_km2=geom.area / 1e6))
    return masked, pd.DataFrame(area_rows)


# --------------------------------------------------------------------------
# 2. lakes -> zone label per track
# --------------------------------------------------------------------------
def classify_lakes(cfg, masked):
    import geopandas as gpd
    cols = ["AREASQKM", "FCODE", "Latitude", "Longitude"]
    lakes = pd.read_parquet(cfg.LAKE_CONUS_PARQUET_PATH, columns=cols)
    lakes = lakes[lakes.AREASQKM >= min(THRESHOLDS)].reset_index(drop=True)
    pts = gpd.GeoDataFrame(lakes, geometry=gpd.points_from_xy(lakes.Longitude, lakes.Latitude),
                           crs="EPSG:4326").to_crs(EQUAL_AREA_CRS)
    for track, zones in masked.items():
        zg = gpd.GeoDataFrame({"zone": list(zones)}, geometry=list(zones.values()), crs=EQUAL_AREA_CRS)
        j = gpd.sjoin(pts[["geometry"]], zg, predicate="within", how="left")
        j = j[~j.index.duplicated()]
        pts[f"zone_{track}"] = j["zone"]
    pts["intermittent"] = pts["FCODE"].isin(INTERMITTENT_FCODES)
    return pts


def lake_mask(pts, lake_set, threshold):
    m = pts.AREASQKM >= threshold
    if lake_set == "no_intermittent":
        m &= ~pts.intermittent
    return m


def count_table(pts, areas):
    rows = []
    for track in TRACKS:
        a = areas[areas.track == track].set_index("zone")["masked_area_km2"]
        for lake_set in LAKE_SETS:
            for thr in THRESHOLDS:
                sub = pts.loc[lake_mask(pts, lake_set, thr), f"zone_{track}"]
                vc = sub.value_counts()
                for z, _ in ZONES:
                    n = int(vc.get(z, 0))
                    rows.append(dict(track=track, lake_set=lake_set, threshold=thr, zone=z,
                                     n_lakes=n, area_km2=a[z], density=1000 * n / a[z]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. models
# --------------------------------------------------------------------------
def _age_rvs(pm, zones):
    out = []
    for z in zones:
        p = AGE_PRIORS[z]
        name = f"age_{z}"
        if "lower" in p:
            out.append(pm.TruncatedNormal(name, mu=p["mu"], sigma=p["sd"], lower=p["lower"]))
        else:
            out.append(pm.Normal(name, mu=p["mu"], sigma=p["sd"]))
    return out


def fit_floor_poisson(counts):
    """counts: dict zone -> (n, area_km2) for the 3 glacial zones + Driftless."""
    import pymc as pm
    import pytensor.tensor as pt
    with pm.Model() as m:
        t_half = pm.LogNormal("t_half", **HALF_LIFE_PRIOR)
        k = pm.Deterministic("k", np.log(2) / t_half)
        Dinf = pm.HalfNormal("Dinf", sigma=DENSITY_PRIOR_SD["Dinf"])
        delta = pm.HalfNormal("delta", sigma=DENSITY_PRIOR_SD["delta"])
        D0 = pm.Deterministic("D0", Dinf + delta)
        ages = pt.stack(_age_rvs(pm, GLACIAL))
        D_g = pm.Deterministic("D_glacial", Dinf + delta * pt.exp(-k * ages))
        n_g = np.array([counts[z][0] for z in GLACIAL])
        a_g = np.array([counts[z][1] for z in GLACIAL]) / 1000.0
        pm.Poisson("N_glacial", mu=a_g * D_g, observed=n_g)
        nD, aD = counts["Driftless"]
        pm.Poisson("N_driftless", mu=aD / 1000.0 * Dinf, observed=nD)
        idata = pm.sample(**SAMPLER)
    return idata


def fit_exp3_poisson(counts):
    import pymc as pm
    import pytensor.tensor as pt
    with pm.Model():
        t_half = pm.LogNormal("t_half", **HALF_LIFE_PRIOR)
        k = pm.Deterministic("k", np.log(2) / t_half)
        D0 = pm.HalfNormal("D0", sigma=DENSITY_PRIOR_SD["D0"])
        ages = pt.stack(_age_rvs(pm, GLACIAL))
        D_g = pm.Deterministic("D_glacial", D0 * pt.exp(-k * ages))
        n_g = np.array([counts[z][0] for z in GLACIAL])
        a_g = np.array([counts[z][1] for z in GLACIAL]) / 1000.0
        pm.Poisson("N_glacial", mu=a_g * D_g, observed=n_g)
        idata = pm.sample(**SAMPLER)
    return idata


def build_cells(masked_track):
    import geopandas as gpd
    from shapely.geometry import box
    zg = gpd.GeoDataFrame({"zone": list(masked_track)}, geometry=list(masked_track.values()),
                          crs=EQUAL_AREA_CRS)
    minx, miny, maxx, maxy = zg.total_bounds
    s = CELL_KM * 1000
    xs = np.arange(minx, maxx + s, s)
    ys = np.arange(miny, maxy + s, s)
    grid = gpd.GeoDataFrame(geometry=[box(x, y, x + s, y + s) for x in xs for y in ys], crs=EQUAL_AREA_CRS)
    cells = gpd.overlay(zg, grid.reset_index(names="cell"), how="intersection", keep_geom_type=True)
    cells["area_km2"] = cells.area / 1e6
    return cells[cells.area_km2 >= MIN_CELL_KM2].reset_index(drop=True)


def fit_negbin_cells(cells, n_cells, floor=True):
    """Negative binomial on grid-cell counts, exposure = cell area.
    floor=True : D1 model, Driftless cells inform Dinf (headline / sweep model).
    floor=False: plain exponential on the glacial cells only (Driftless dropped)."""
    import pymc as pm
    import pytensor.tensor as pt
    zone_names = GLACIAL + (["Driftless"] if floor else [])
    keep = cells["zone"].isin(zone_names).to_numpy()
    zi = cells.loc[keep, "zone"].map({z: i for i, z in enumerate(zone_names)}).to_numpy()
    expo = cells.loc[keep, "area_km2"].to_numpy() / 1000.0
    obs = np.asarray(n_cells)[keep]
    with pm.Model():
        t_half = pm.LogNormal("t_half", **HALF_LIFE_PRIOR)
        k = pm.Deterministic("k", np.log(2) / t_half)
        ages = pt.stack(_age_rvs(pm, GLACIAL))
        if floor:
            Dinf = pm.HalfNormal("Dinf", sigma=DENSITY_PRIOR_SD["Dinf"])
            delta = pm.HalfNormal("delta", sigma=DENSITY_PRIOR_SD["delta"])
            pm.Deterministic("D0", Dinf + delta)
            D_g = pm.Deterministic("D_glacial", Dinf + delta * pt.exp(-k * ages))
            D_z = pt.concatenate([D_g, pt.stack([Dinf])])
        else:
            D0 = pm.HalfNormal("D0", sigma=DENSITY_PRIOR_SD["D0"])
            D_z = pm.Deterministic("D_glacial", D0 * pt.exp(-k * ages))
        alpha = pm.LogNormal("alpha", mu=0.0, sigma=1.5, shape=len(zone_names))
        pm.NegativeBinomial("N_cell", mu=expo * D_z[zi], alpha=alpha[zi], observed=obs)
        idata = pm.sample(**SAMPLER)
    return idata


def summarize(idata, model, key):
    import arviz as az
    post = idata.posterior
    out = dict(key, model=model)
    for v in ["t_half", "k", "D0", "Dinf"]:
        if v in post:
            x = post[v].values
            out[f"{v}_median"] = q(x, 0.5)
            out[f"{v}_lo95"] = q(x, 0.025)
            out[f"{v}_hi95"] = q(x, 0.975)
    for z in GLACIAL:
        x = post[f"age_{z}"].values
        out[f"age_{z}_median"] = q(x, 0.5)
    dg = post["D_glacial"].values.reshape(-1, len(GLACIAL))
    for i, z in enumerate(GLACIAL):
        out[f"Dfit_{z}_median"] = float(np.median(dg[:, i]))
    if "alpha" in post:  # NB shape per zone (smaller = more spatially clustered)
        a = post["alpha"].values.reshape(-1, post["alpha"].shape[-1])
        for i, z in enumerate((GLACIAL + ["Driftless"])[:a.shape[1]]):
            out[f"alpha_{z}_median"] = float(np.median(a[:, i]))
    s = az.summary(idata, var_names=[v for v in ["t_half", "D0", "Dinf"] if v in post],
                   kind="diagnostics")
    out["r_hat_max"] = float(s["r_hat"].max())
    out["ess_bulk_min"] = float(s["ess_bulk"].min())
    out["divergences"] = int(idata.sample_stats["diverging"].sum())
    return out


# --------------------------------------------------------------------------
# 4. figure + report
# --------------------------------------------------------------------------
def plot_headline(idata_floor, idata_exp3, dens, path):
    import matplotlib.pyplot as plt
    post = idata_floor.posterior
    t = np.linspace(0, 1500, 301)
    th = post["t_half"].values.ravel()
    k = np.log(2) / th
    Dinf = post["Dinf"].values.ravel()
    delta = post["delta"].values.ravel()
    curves = Dinf[:, None] + delta[:, None] * np.exp(-k[:, None] * t[None, :])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.fill_between(t, np.quantile(curves, 0.025, 0), np.quantile(curves, 0.975, 0),
                    color="#2c7fb8", alpha=0.25, lw=0, label="Floor model (neg. binomial cells), 95% band")
    ax.plot(t, np.median(curves, 0), color="#2c7fb8", lw=2, label="Floor model, median")
    p3 = idata_exp3.posterior
    k3 = np.log(2) / p3["t_half"].values.ravel()
    c3 = p3["D0"].values.ravel()[:, None] * np.exp(-k3[:, None] * t[None, :])
    ax.plot(t, np.median(c3, 0), color="#636363", lw=1.5, ls="--", label="Plain exponential (3 glacial zones)")
    ax.axhspan(q(Dinf, 0.025), q(Dinf, 0.975), color="#d95f0e", alpha=0.15, lw=0)
    ax.axhline(np.median(Dinf), color="#d95f0e", lw=1, label="D∞ (Driftless)")
    for i, z in enumerate(GLACIAL):
        age = post[f"age_{z}"].values.ravel()
        y = float(dens.loc[z])
        ax.errorbar(np.median(age), y, xerr=[[np.median(age) - q(age, .025)], [q(age, .975) - np.median(age)]],
                    fmt="o", color="black", ms=6, capsize=3)
        ax.annotate(z, (np.median(age), y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Surface age (ka)")
    ax.set_ylabel("Lakes per 1000 km²")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, None)
    ax.set_title(f"Headline: {HEADLINE['track']} mask, ≥{HEADLINE['threshold']} km², {HEADLINE['lake_set']} lakes")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmt_ci(r, v, nd=0):
    return f"{r[f'{v}_median']:.{nd}f} [{r[f'{v}_lo95']:.{nd}f}–{r[f'{v}_hi95']:.{nd}f}]"


def write_results(out_dir, info, areas, counts, fits):
    L = []
    L.append(f"# Canonical chronosequence results — {info['date']}\n")
    L.append(f"- Commit: `{info['commit']}`{' (**working tree dirty**)' if info['dirty'] else ''}; "
             f"script sha256 `{info['script_sha']}`")
    L.append(f"- Generated by `scripts/regenerate_canonical.py` ({'quick' if info['quick'] else 'full'} run), "
             f"runtime {info['runtime_min']:.1f} min")
    L.append(f"- Environment: python {info['versions']['python']}, pymc {info['versions']['pymc']}, "
             f"geopandas {info['versions']['geopandas']}, pandas {info['versions']['pandas']}")
    L.append(f"- Lakes: `{info['lake_file']}` ({info['n_lakes_min_thr']:,} lakes ≥ {min(THRESHOLDS)} km²)")
    L.append("- Decisions: ORIENTATION_2026-11-04.md §8 (D1 floor model, D2 age-consistent Qal mask, "
             "D3 Pre-Illinoian 650 ka, D4 headline 0.01 km²)\n")

    h = HEADLINE
    hc = counts[(counts.track == h["track"]) & (counts.lake_set == h["lake_set"]) & (counts.threshold == h["threshold"])]
    L.append("## Headline densities\n")
    L.append(f"Track `{h['track']}`, lakes ≥ {h['threshold']} km², lake set `{h['lake_set']}`.\n")
    L.append("| Zone | Lakes | Masked area (km²) | Density (per 1000 km²) |\n|---|---|---|---|")
    for _, r in hc.iterrows():
        L.append(f"| {r.zone} | {r.n_lakes:,} | {r.area_km2:,.0f} | {r.density:.1f} |")

    hf = fits[(fits.track == h["track"]) & (fits.lake_set == h["lake_set"]) & (fits.threshold == h["threshold"])]
    L.append("\n## Headline fits\n")
    L.append("Median [95% equal-tailed interval]. t½ in ka; densities per 1000 km². **`floor_nb` is the "
             "headline model** (negative binomial on grid-cell counts). The others are comparisons: `exp3_nb` "
             "drops Driftless and uses a plain exponential, and the `*_poisson` models use whole-zone counts.\n")
    L.append("| Model | t½ | D₀ | D∞ | Illinoian age | Pre-Ill. age | Driftless used as | r̂ max | divergences |\n"
             "|---|---|---|---|---|---|---|---|---|")
    role = {"floor_nb": "floor (D∞)", "exp3_nb": "not used",
            "floor_poisson": "floor (D∞)", "exp3_poisson": "not used"}
    for _, r in hf.iterrows():
        dinf = fmt_ci(r, "Dinf", 1) if not pd.isna(r.get("Dinf_median", np.nan)) else "—"
        L.append(f"| {r.model} | {fmt_ci(r, 't_half')} | {fmt_ci(r, 'D0', 1)} | {dinf} | "
                 f"{r['age_Illinoian_median']:.0f} | {r['age_Pre-Illinoian_median']:.0f} | {role[r.model]} | "
                 f"{r.r_hat_max:.3f} | {r.divergences} |")
    obs = hc.set_index("zone")["density"]
    L.append("\n**Fitted vs observed zone density** (per 1000 km²; observed = lakes ÷ masked area). "
             "A gap here means the curve cannot pass through all the zones at their prior ages.\n")
    L.append("| Model | " + " | ".join(GLACIAL) + " |\n|---|" + "---|" * len(GLACIAL))
    L.append("| observed | " + " | ".join(f"{obs[z]:.1f}" for z in GLACIAL) + " |")
    for _, r in hf.iterrows():
        L.append(f"| {r.model} | " + " | ".join(f"{r[f'Dfit_{z}_median']:.1f}" for z in GLACIAL) + " |")
    L.append(f"\nAge priors: Illinoian {AGE_PRIORS['Illinoian']['mu']:.0f} ± {AGE_PRIORS['Illinoian']['sd']:.0f} ka; "
             f"Pre-Illinoian {AGE_PRIORS['Pre-Illinoian']['mu']:.0f} ± {AGE_PRIORS['Pre-Illinoian']['sd']:.0f} ka "
             f"(truncated at {AGE_PRIORS['Pre-Illinoian']['lower']:.0f}). If a posterior age sits far from its prior, "
             "the model is bending the age to fit the curve.")
    nb = hf[hf.model == "floor_nb"]
    if len(nb):
        r = nb.iloc[0]
        al = ", ".join(f"{z} {r[f'alpha_{z}_median']:.2f}" for z in GLACIAL + ["Driftless"]
                       if f"alpha_{z}_median" in r and not pd.isna(r[f"alpha_{z}_median"]))
        L.append(f"\n`floor_nb`: {int(r.n_cells):,} cells of {CELL_KM:.0f} km (slivers < {MIN_CELL_KM2:.0f} km² dropped; "
                 f"{100 * r.lakes_in_cells_frac:.1f}% of zone lakes retained). NB shape α by zone (smaller = more "
                 f"clustered): {al}.")
    L.append("\n![headline decay](decay_headline.png)\n")

    L.append("## Masked zone areas (km²)\n")
    piv = areas.pivot(index="zone", columns="track", values="masked_area_km2").reindex([z for z, _ in ZONES])
    raw = areas.drop_duplicates("zone").set_index("zone").reindex([z for z, _ in ZONES])
    L.append("| Zone | raw polygon | non-overlapping | " + " | ".join(TRACKS) + " |\n|---|---|---|" + "---|" * len(TRACKS))
    for z in piv.index:
        L.append(f"| {z} | {raw.loc[z, 'raw_area_km2']:,.0f} | {raw.loc[z, 'hier_area_km2']:,.0f} | "
                 + " | ".join(f"{piv.loc[z, t]:,.0f}" for t in TRACKS) + " |")

    fl = fits[fits.model == "floor_nb"]
    if len(fl) > 1:
        L.append("\n## Sensitivity: headline-model (`floor_nb`) t½ (ka), median [95%]\n")
        for lake_set in LAKE_SETS:
            L.append(f"\nLake set `{lake_set}`:\n")
            L.append("| Threshold (km²) | " + " | ".join(TRACKS) + " |\n|---|" + "---|" * len(TRACKS))
            for thr in THRESHOLDS:
                cells = []
                for tr in TRACKS:
                    r = fl[(fl.track == tr) & (fl.lake_set == lake_set) & (fl.threshold == thr)]
                    cells.append(fmt_ci(r.iloc[0], "t_half") if len(r) else "—")
                L.append(f"| {thr} | " + " | ".join(cells) + " |")

        L.append("\n## Sensitivity: densities (per 1000 km²), lake set `all`\n")
        for tr in TRACKS:
            c = counts[(counts.track == tr) & (counts.lake_set == "all")]
            p = c.pivot(index="threshold", columns="zone", values="density")[[z for z, _ in ZONES]]
            L.append(f"\nTrack `{tr}`:\n")
            L.append("| Threshold | " + " | ".join(p.columns) + " |\n|---|" + "---|" * len(p.columns))
            for thr, row in p.iterrows():
                L.append(f"| {thr} | " + " | ".join(f"{v:.1f}" for v in row) + " |")

    L.append("\n## Caveats\n")
    L.append("- The headline uses a negative binomial on ~50 km cells so that spatial heterogeneity widens the "
             "interval. Cells are treated as independent, though spatial autocorrelation exists, so the "
             "interval is probably still somewhat too narrow. The `*_poisson` rows show the alternative: "
             "whole-zone counts pin each density to about 1%, and the model then bends the ages to fit.")
    L.append("- Driftless density falls much faster with threshold than the glacial zones do. That is "
             "consistent with many small constructed ponds (NHD LakePond includes stock ponds), so D∞ may "
             "partly measure farm-pond density rather than a natural asymptote.")
    L.append("- The floor model assumes the Driftless upland density equals the long-time asymptote of the "
             "glacial surfaces (D1). D∞ comes almost entirely from the Driftless count.")
    L.append("- Lakes are assigned by stored centroid point. Zone polygons, the Qal map (1:2.5M) and NHD-HR "
             "differ in scale, so lakes near boundaries are uncertain.")
    L.append("- The Wisconsin age is a single value (20 ± 2.5 ka) for the whole zone. The NADI-1 per-lake "
             "deglaciation ages are not used here.")
    (out_dir / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="headline settings only (headline + comparison models), for testing")
    ap.add_argument("--out", default=None, help="override output folder")
    args = ap.parse_args()

    t0 = time.time()
    from lake_analysis import config as cfg
    import geopandas, pymc  # noqa: E401
    commit, dirty = git_state()
    today = dt.date.today().isoformat()
    out_dir = Path(args.out) if args.out else Path(cfg.DATA_ROOT) / "Analysis" / f"canonical_{today}_{commit}{'-dirty' if dirty else ''}{'-quick' if args.quick else ''}"
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    log(f"output -> {out_dir}")

    log("building geometry (hierarchical zones, Qal units, masks)")
    masked, areas = build_geometry(cfg)
    areas.to_csv(out_dir / "zone_areas.csv", index=False)

    log("classifying lakes")
    pts = classify_lakes(cfg, masked)
    counts = count_table(pts, areas)
    counts.to_csv(out_dir / "counts_areas.csv", index=False)
    log(f"  {len(pts):,} lakes >= {min(THRESHOLDS)} km2")

    import geopandas as gpd
    # grid cells per track, and each lake's cell id (once per track)
    log(f"building {CELL_KM:.0f} km cells per track")
    cells, lake_cell = {}, {}
    for tr in TRACKS:
        cells[tr] = build_cells(masked[tr])
        lk = pts[pts[f"zone_{tr}"].notna()][["geometry"]]
        j = gpd.sjoin(lk, cells[tr][["geometry"]].reset_index(names="cid"), predicate="within", how="inner")
        j = j[~j.index.duplicated()]
        lake_cell[tr] = j["cid"].reindex(pts.index)   # NaN = outside zones or in a dropped sliver
        log(f"  {tr}: {len(cells[tr]):,} cells")

    head = (HEADLINE["track"], HEADLINE["lake_set"], HEADLINE["threshold"])
    combos = [head] if args.quick else \
        [(tr, ls, thr) for tr in TRACKS for ls in LAKE_SETS for thr in THRESHOLDS]
    fits, idatas = [], {}
    for tr, ls, thr in combos:
        key = dict(track=tr, lake_set=ls, threshold=thr)
        is_head = (tr, ls, thr) == head
        sel = lake_mask(pts, ls, thr) & lake_cell[tr].notna()
        n_c = np.bincount(lake_cell[tr][sel].astype(int).to_numpy(), minlength=len(cells[tr]))
        n_zone = int((lake_mask(pts, ls, thr) & pts[f"zone_{tr}"].notna()).sum())

        log(f"fit floor_nb   {tr:15s} {ls:16s} {thr}")
        idnb = fit_negbin_cells(cells[tr], n_c, floor=True)
        r = summarize(idnb, "floor_nb", key)
        r["n_cells"] = len(cells[tr])
        r["lakes_in_cells_frac"] = float(n_c.sum() / max(1, n_zone))
        fits.append(r)

        if is_head:
            idatas["floor_nb"] = idnb
            idnb.to_netcdf(out_dir / "traces" / "headline_floor_nb.nc")
            log("fit exp3_nb (headline comparison: Driftless not used)")
            id3 = fit_negbin_cells(cells[tr], n_c, floor=False)
            fits.append(summarize(id3, "exp3_nb", key))
            idatas["exp3_nb"] = id3
            id3.to_netcdf(out_dir / "traces" / "headline_exp3_nb.nc")
            # Poisson on whole-zone counts, kept to show why it was not adopted
            c = counts[(counts.track == tr) & (counts.lake_set == ls) & (counts.threshold == thr)].set_index("zone")
            cd = {z: (int(c.loc[z, "n_lakes"]), float(c.loc[z, "area_km2"])) for z, _ in ZONES}
            log("fit floor_poisson / exp3_poisson (headline comparison)")
            fits.append(summarize(fit_floor_poisson(cd), "floor_poisson", key))
            fits.append(summarize(fit_exp3_poisson(cd), "exp3_poisson", key))
        pd.DataFrame(fits).to_csv(out_dir / "fits.csv", index=False)

    fits = pd.DataFrame(fits)
    hc = counts[(counts.track == HEADLINE["track"]) & (counts.lake_set == HEADLINE["lake_set"])
                & (counts.threshold == HEADLINE["threshold"])].set_index("zone")["density"]
    plot_headline(idatas["floor_nb"], idatas["exp3_nb"], hc, out_dir / "decay_headline.png")

    info = dict(date=today, commit=commit, dirty=bool(dirty), script_sha=file_sha(__file__),
                quick=args.quick, runtime_min=(time.time() - t0) / 60,
                versions=dict(python=sys.version.split()[0], pymc=pymc.__version__,
                              geopandas=geopandas.__version__, pandas=pd.__version__),
                lake_file=cfg.LAKE_CONUS_PARQUET_PATH, n_lakes_min_thr=len(pts),
                settings=dict(HEADLINE=HEADLINE, THRESHOLDS=THRESHOLDS, TRACKS=TRACKS,
                              AGE_PRIORS=AGE_PRIORS, HALF_LIFE_PRIOR=HALF_LIFE_PRIOR,
                              DENSITY_PRIOR_SD=DENSITY_PRIOR_SD, CELL_KM=CELL_KM,
                              INTERMITTENT_FCODES=sorted(INTERMITTENT_FCODES), SAMPLER=SAMPLER))
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    write_results(out_dir, info, areas, counts, fits)
    log(f"done in {info['runtime_min']:.1f} min -> {out_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
