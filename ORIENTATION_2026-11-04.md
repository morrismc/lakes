# Lakes project: local orientation (2026-11-04)

This note builds on `BRIEFING_2026-10-28.md` using what the repo, the data drive, and the last outputs show. Nothing has been changed, committed, or re-run yet; every number here comes from reading code or files on disk, or from quick read-only geometry checks.

---

## 1. Where things live (the drive is now `D:`, not `F:`)

| What | Path |
|---|---|
| Git repo (`morrismc/lakes`, `main` @ `a83144d`, 2026-04-03) | `D:\Lakes\Code\lakes` |
| Package | `D:\Lakes\Code\lakes\lake_analysis\` (≈35k lines; the biggest files are `visualization` 8.4k, `glacial_chronosequence` 5.7k, `main` 4.4k) |
| Lake tables | `D:\Lakes\Data\lakes_conus.parquet` (4,896,094 rows), `lakes.parquet` (5,488,564) |
| Geodatabase (lakes + zone polygons) | `D:\Lakes\GIS\MyProject.gdb` |
| Shapefiles (NADI-1 ×147, Dalton 18 ka, Qal, ES-057, CONUS) | `D:\Lakes\GIS\shapefiles\` |
| Rasters (SRTM 90 m elevation/slope/relief, PRISM, AI, PET) | `D:\Lakes\GIS\rasters\` |
| Outputs from earlier runs | `D:\Lakes\Analysis\outputs\` (573 files; last run 2026-04-03) |
| Older R / Python prototypes (2024) | `D:\Lakes\Code\*.Rmd`, `Lake_raster_normalization*.py` |

**Path check:** every path in `config.py` resolves once `F:` is swapped for `D:`. All six gdb layers that config names exist, and all 147 NADI-1 shapefiles are present. Nothing is missing.

---

## 2. What changed since the briefing's source chats

| Date | Commit | Change |
|---|---|---|
| 03-14 → 03-16 | `cb40bc8` … `0fb1d19` | Pre-Illinoian added as a 4th end member; layer-name and age-tuple crash fixes |
| 03-27 | `3ef68e1` | Qal exclusion (`Qal_comprehensive_plus_glacial_alluvial.shp`) |
| 04-02 | `24b3654` | Dual-track run: with and without Qal |
| 04-03 | `a8b26f2`, `32363b3` | `min_lake_area` argument on the glacial track; **Driftless entered at 1500 ka in the Davis regression** (this is option (b) in briefing §5.1) |
| **06-26 (NOT merged)** | `d739ccf` on `origin/claude/add-pre-illinoian-data-b44IB` | (1) Qal comparison figure was pairing stages by row position, which made Driftless look like it *gained* density after Qal removal; it actually falls **53.9 → 25.4 (−53%)**. (2) The power-law vs lognormal/exponential comparison used a χ² test for nested models; replaced with Vuong's test. (3) The NADI-1 path never classified Pre-Illinoian lakes; now it does. |

No commits since June. The local checkout does not include `d739ccf`.

---

## 3. How the pipeline is wired

`main.run_full_analysis()` runs the steps in this order: load → H1 elevation → H2 slope → H3 relief → H4 elev×slope → H5 power law → H6 slope×relief → relief×elev → power-law sensitivity → x_min by elevation → domains → **glacial chronosequence (two tracks)** → **Bayesian half-life** → spatial scaling → aridity → multivariate (CONUS-wide and glacial-only) → summary figures.

### The decay / half-life results come from three separate code paths

Each path classifies lakes, computes areas, filters by lake size, and fits decay differently. That is the most likely reason the briefing lists half-lives of 242, 660, 822, and 1,118 ka.

| | **A. Glacial track** `analyze_glacial_chronosequence` → `run_glacial_chronosequence_analysis` | **B. Bayesian t½** `analyze_bayesian_halflife` → `fit_overall_bayesian_halflife` | **C. NADI-1** `analyze_nadi1_chronosequence` → `fit_bayesian_decay_model` |
|---|---|---|---|
| Zone areas | Computed from gdb polygons | **Hard-coded in config** (W 1,225,000; I 145,000; **Pre-Ill `None`**; D 25,500) | Computed from polygons (own routine) |
| Stages actually fit | W, I, PI, D | **W, I, D only.** Pre-Ill area is `None`, so its density is NaN and the stage is dropped without a message | **W, I, D only** on `main` (PI never classified); fixed in `d739ccf` |
| Qal | Both tracks, but the **area is not removed** from the denominator (§4.2) | Uses track A's Qal-excluded lakes with the hard-coded, Qal-inclusive areas | No Qal handling |
| Min lake area | `None` unless passed (the full pipeline passes nothing) | 0.01 | 0.01 |
| Other filters | none | max 20,000 km² | lon < −110° removed; optional max area |
| "Decay" model | **Linear** Pearson/OLS on 4 points | Normal likelihood on densities + free `sigma_obs`; t½ ~ LogNormal(ln 500, 1.5) | Normal likelihood, fixed 10% CV; D₀ ~ N(max+10, 30); k ~ HalfNormal(0.002) |
| Ages | W 20, I 160, PI 500, D 1500 (fixed) | Normal: PI 500±100, D 1500±500 | Truncated normals; D = 1500 + Exp(mean 500) |

None of the three uses the Poisson-counts-with-area-exposure likelihood or the MIS-16 prior (650 ka) that briefing §6 recommends. `config.py` still has Pre-Illinoian at 500±100 ka.

---

## 4. Issues found, roughly by impact

1. **`import lake_analysis` fails.** `sensitivity_analysis.py:35/47` imports `convert_lakes_to_gdf` from `data_loading`, but that function is defined in `glacial_chronosequence`. Any `from lake_analysis import ...` raises. Running `main.py` directly in Spyder skips `__init__`, which may be why this went unnoticed. One-line fix.
2. **The Qal track removes Qal lakes but keeps Qal land in each zone's area.** Measured with the gdb polygons and the Qal shapefile in ESRI:102039:

   | Zone | Zone area (km²) | Qal inside zone | Qal share | Density bias in the Qal-excluded track |
   |---|---|---|---|---|
   | Wisconsin | 1,224,969 | 51,854 | 4.2% | −4% |
   | Illinoian | 145,018 | 11,908 | 8.2% | −8% |
   | Pre-Illinoian | 286,833 | 38,077 | 13.3% | −13% |
   | Driftless (definite) | 25,452 | 3,309 | 13.0% | −13% |

   Older surfaces are biased low about three times more than Wisconsin, which steepens the decay. This is the mask-induced contrast that briefing §5.5 warned about.
3. **Pre-Illinoian is not used by the Bayesian half-life (path B).** The "4 end-member" claim does not hold for the headline t½ result.
4. **Size thresholds are still scattered across the code** (briefing lesson #1 is not enforced). Values in use: 0.001 (size histograms), 0.005 (multivariate default), 0.01 (Bayesian, NADI-1), 0.024 (`config.MIN_LAKE_AREA`, which the pipeline never reads), 0.05 (size-stratified), and `None` (glacial track). One `run_full_analysis()` call therefore filters lakes four different ways.
5. **The two Qal tracks write their CSVs to the same folder.** `density_by_glacial_stage.csv` etc. land in `glacial_chronosequence/` for both tracks, so the all-lakes CSVs are overwritten. Only the figures are split into subfolders.
6. **Driftless is a dated point at 1500 ka**, and the Davis "test" in path A is a linear correlation on n = 4. The asymptote alternative (briefing §5.1a) is not implemented.
7. **The zone polygons overlap slightly.** W∩I is 6,815 km² (4.7% of Illinoian), W∩PI and I∩PI are about 6,768 km² each, and PI∩Driftless is 702 km². Lakes are assigned hierarchically, but the overlaps still count in both zones' areas. Small, but it is a denominator error.
8. **The Qal source needs confirming.** The file is `Qal_comprehensive_plus_glacial_alluvial.shp` (CRS EPSG:4267, NAD27; config says ESRI:102039, which does no harm because the code reads the file's own CRS). The briefing names `Qal_soller_2004`. If "glacial alluvial" includes outwash plains, the mask also removes real kettle lakes.
9. **Silent fuzzy layer matching.** `load_and_reproject` falls back to the closest layer name when a layer isn't found. It could load the wrong layer (for example the `*_glacial_limit` line layer) without failing.
10. **Facts about the lake table that bear on detection:** it is NHD **High Resolution** (`RESOLUTION = 2` for every row). It contains `FTYPE 390` (LakePond) only, so reservoirs are excluded. Median area is 0.002 km². **20% of lakes ≥ 0.01 km² are intermittent (FCODE 39001).** Including intermittent ponds is a choice to make deliberately, because intermittency follows climate. FDATE runs 2001–2017.

---

## 5. Reconciling the numbers

Last on-disk run (2026-04-03), `Analysis/outputs/glacial_chronosequence/density_by_glacial_stage.csv`. That file is from the **Qal-excluded** track (it has an `in_qal` column). Its threshold appears to be about **0.005 km²**, inferred from the total lake count and from lakes of 0.008 km² appearing in the export.

| Stage | n lakes | On disk (per 1000 km²) | Same, with Qal land removed from area (rough) | Briefing values |
|---|---|---|---|---|
| Wisconsin | 267,486 | 218.4 | ~228 | 228.2 (Jan) |
| Illinoian | 25,435 | 175.4 | ~191 | 202.8 (Jan) / ~95 (Feb) |
| Pre-Illinoian | 32,453 | 113.1 | ~130 | n/a |
| Driftless | 646 | 25.4 | ~29 | 69.4 (Jan) |

The corrected column only re-divides by (zone area − Qal area). It is for scale, not a canonical value. The Driftless value moves between 54 and 25 depending on Qal handling, more than any other stage. Whatever the Driftless area ends up representing in the model, its density depends heavily on the Qal mask.

### 5a. Baseline re-run, 2026-11-04 (commit c5855ca, after merging d739ccf)

The outputs are in `D:\Lakes\Analysis\outputs_baseline_2026-11-04_c5855ca\`. `run_info.json` there records the settings, and the logs are `Analysis\baseline_run_2026-11-04_v2.log`. The script is `scripts/run_baseline.py`.

**Glacial track, minimum lake area 0.005 km².** The Qal-excluded track reproduces April exactly: identical lake counts, densities and zone areas. The threshold is now confirmed as exactly 0.005.

| Stage | All lakes: n | All lakes: density | Qal excl.: n | Qal excl.: density | Change |
|---|---|---|---|---|---|
| Wisconsin | 279,582 | 228.2 | 267,486 | 218.4 | −4.3% |
| Illinoian | 29,404 | 202.8 | 25,435 | 175.4 | −13.5% |
| Pre-Illinoian | 41,177 | 143.6 | 32,453 | 113.1 | −21.2% |
| Driftless | 1,373 | 53.9 | 646 | 25.4 | −52.9% |

- The briefing's January values (Wisconsin 228.2, Illinoian 202.8) are the **all-lakes** track at 0.005.
- Qal exclusion removes a larger fraction of lakes than the fraction of area that is Qal: 21% of lakes against 13% of the area for Pre-Illinoian, and 53% against 13% for Driftless. Qal ground on the older surfaces is therefore *lake-rich*, mostly floodplain ponds. This makes area handling the deciding question for the old end of the curve.

**NADI-1 path, minimum lake area 0.01 km², lakes west of −110° removed. Now 4 points.**
- End-member densities: Wisconsin 140.4, Illinoian 94.1, Pre-Illinoian 60.2, Driftless 35.4. The 94.1 is the briefing's February "~95" Illinoian value.
- Bayesian fit (Normal likelihood with a fixed 10% CV; ages from config):
  - **t½ = 587 ka [95% CI 452–807]**, D₀ = 126.5 [107–146], k = 0.00121 per ka.
  - This replaces the 3-point 660 [418–1505] ka in CLAUDE.md.
- Warning signs:
  - The Driftless age posterior is pulled down to **1117 ka**, against the 1000 ka floor and well below its nominal 1500 ka. Its correlation with half-life is r = 0.73.
  - The Pre-Illinoian age is pulled up to 568 ka.
  - D₀ (126) is below the Wisconsin density (140).
  - Together these say the single exponential cannot fit all four points. The Driftless density is higher than the curve predicts at 1.5 Ma, which is the shape an asymptote (D∞) model would produce. Read this as the model bending the ages to fit, not as evidence about the ages themselves.
- The narrow CI comes partly from the fixed 10% CV. A Poisson-count likelihood would set the uncertainty from lake counts instead.

**Environment note.** The first attempt's Bayesian fit failed, and the code quietly fell back to a frequentist fit (533 ka). The cause was that, under the Claude desktop app, PyTensor's compile cache resolves to a path at Windows' 260-character limit. Claude runs therefore set `PYTENSOR_FLAGS=base_compiledir=D:/Lakes/.pytensor`. Spyder runs should not hit this.

---

## 6. Running the code here vs. in Spyder

**Current state of Python on this machine**
- No `python` on PATH (only the Microsoft Store stub).
- Miniforge is at `C:\Users\mmorriss\AppData\Local\miniforge3`. The `lakes` and `bayesian` envs are **empty shells** (created 2026-05-04, no Python installed).
- `pygis` (Python 3.12) has the full geospatial stack (geopandas 1.1, pyogrio, rasterio, shapely 2.1, sklearn, statsmodels, pyarrow) but **lacks PyMC, ArviZ, and spyder-kernels**, and has **pandas 3.0**, while this code was written against pandas 2.
- There is no Spyder executable in any env. Where does Spyder run from now (standalone installer, a different machine)?
- The env has to be activated (`conda run -n <env>`). Calling `python.exe` directly fails with a DLL error.

**Proposal: one shared environment so that my runs and Spyder runs give identical numbers**
1. Create `lakes` from an `environment.yml` committed to the repo: python 3.11, pandas < 3, numpy, scipy, geopandas, pyogrio, fiona, rasterio, pyproj, shapely, pyarrow, matplotlib, seaborn, scikit-learn, statsmodels, pymc, arviz, ipykernel, **spyder-kernels**.
2. Point Spyder at it: *Tools → Preferences → Python interpreter → Use the following interpreter →* `...\miniforge3\envs\lakes\python.exe`.
3. Replace the hard-coded `F:\` paths with one `DATA_ROOT` that reads the `LAKES_DATA_ROOT` environment variable and defaults to `D:\Lakes`. This avoids editing paths every time the drive letter changes.
4. Put run scripts in `scripts/`. They run as `conda run -n lakes python scripts/xyz.py` here (long jobs in the background, with a log) and as `%runfile` in Spyder. Each run writes to a timestamped output folder so the April outputs are never overwritten.

---

## 7. Next steps (briefing §7, adjusted to what turned up)

1. ~~Decide on `d739ccf`~~ Merged into `main` on 2026-11-04 (c5855ca).
2. Small fixes: ~~the import bug~~ and ~~`DATA_ROOT`~~ are done (PR #73). Per-track CSV folders (§4.5) are still open.
3. ~~Build the `lakes` env~~ Done (§6, `RUNNING.md`).
4. Write `scripts/regenerate_canonical.py`, which uses **one** classification, **one** set of areas computed from polygons (overlaps resolved, Qal land removed when Qal lakes are), and **one** threshold taken from config. It sweeps [0.005, 0.01, 0.024, 0.05, 0.1] km², with and without Qal, writes counts, areas, densities, and t½ (exponential fit, Poisson likelihood) to `RESULTS.md` with the commit hash, and also produces the Qal-bias table by zone and size class (briefing task 4).
5. Then the modelling work: asymptote model vs dated Driftless, MIS-16 prior, the x_min test for 1000–1500 m, and decomposition of the low-elevation peak.

## 8. Decisions (Matthew, 2026-11-05)

These are the spec for `scripts/regenerate_canonical.py`.

| # | Question | Decision |
|---|---|---|
| D1 | What Driftless represents | **Floor (D∞).** Fit D(t) = D∞ + (D₀ − D∞)·e^(−kt). k comes from the three glacial stages, and the Driftless density informs D∞. Driftless gets no age. |
| D2 | Qal mask | **Age-consistent.** In the Wisconsin zone, mask only floodplain alluvium (`al`). In the Illinoian, Pre-Illinoian and Driftless zones, mask all three units (`al`, `gs` outwash, `(ag)` mixed glacial). Masked land is **removed from the zone areas** as well as the lakes. |
| D3 | Pre-Illinoian age | **650 ka (MIS 16)**, with a fairly wide prior (about 450–900 ka). |
| D4 | Headline threshold | **0.01 km².** The full sweep [0.005, 0.01, 0.024, 0.05, 0.1] is reported as sensitivity. |

Still open: keep or drop intermittent ponds (FCODE 39001, 20% of lakes ≥ 0.01 km²). The default is to keep them and also report a run without them. The likelihood is Poisson on counts, with the exposure being the masked zone area, per briefing §7.

**What the Qal layer actually is (checked 2026-11-05).** `Qal_comprehensive_plus_glacial_alluvial.shp` has 1,339 polygons in EPSG:4267 (the config's crs entry says ESRI:102039, which is wrong). The `GEOUNIT` field holds three units, all late Wisconsin to Holocene:
- `al`: channel and flood-plain alluvium, 1,200 polygons.
- `gs`: outwash sand, 77 polygons.
- `(ag)`: alluvial, outwash, ice-contact and glacial-lake deposits, 62 polygons.

Lake density per 1000 km² by unit (lakes ≥ 0.005 km², zones assigned W > I > PI > D with overlaps removed). The table is saved as `outputs_baseline_2026-11-04_c5855ca\qal_by_unit_and_zone.csv`.

| Zone | alluvium | outwash | mixed glacial | outside Qal |
|---|---|---|---|---|
| Wisconsin | 160 | 254 | 297 | 228.0 |
| Illinoian | 340 | 130 | 344 | 201.4 |
| Pre-Illinoian | 231 | 216 | 166 | 134.1 |
| Driftless | 211 | 279 | — | 29.6 |

The lake counts outside Qal equal the April Qal-excluded counts exactly. 36% of Driftless lakes (≥ 0.005) sit on Wisconsin-age outwash, which is valley train, and another 17% sit on alluvium.

**Likelihood (D5, 2026-11-05):** a negative binomial on 50 km grid-cell counts (`floor_nb`) is the headline and sweep model. Poisson on whole-zone counts was rejected: it fixes each density to about 1%, and the model then pushes the ages to their prior limits (Pre-Illinoian to 414 ka, Illinoian to 193 ka).

### 8a. First canonical run, 2026-11-05

The run is `D:\Lakes\Analysis\canonical_2026-11-05_c5855ca-dirty\RESULTS.md`, from `scripts/regenerate_canonical.py`, which was uncommitted at the time.

- **Headline** (age-consistent mask, ≥ 0.01 km², all lakes):
  - Densities: Wisconsin 141.1, Illinoian 90.0, Pre-Illinoian 53.4, Driftless 12.2 per 1000 km².
  - `floor_nb`: **t½ = 383 ka [262–581]**, D∞ = 13.5. Without Driftless, `exp3_nb` gives 463 [319–705].
- **No single decay rate fits.** The decay from Wisconsin to Illinoian implies t½ ≈ 220 ka, and from Illinoian to Pre-Illinoian (at 650 ka) ≈ 650 ka. The NB fit misses the observed densities (Wisconsin fitted 125 vs observed 141, Illinoian 99 vs 90). The Poisson fit matches them by moving the ages.
- **The half-life is set mainly by the lake-size threshold.** `floor_nb`, age-consistent mask, t½ in ka:

| Min area (km²) | all lakes | no intermittent |
|---|---|---|
| 0.005 | 851 [528–1394] | 2176 [1079–7736] |
| 0.01 | 383 [262–581] | 605 [391–931] |
| 0.024 | 109 [49–256] | 228 [83–359] |
| 0.05 | 54 [35–81] | 62 [40–101] |
| 0.1 | 44 [27–64] | 48 [32–70] |

  - The Qal mask choice changes these values by less than 10%. The exceptions are the less reliable fits flagged below.
  - Above 0.05 km², the half-life is about 45–65 ka whatever the mask or lake set. Lakes of that size are largely gone by the Illinoian, and the Pre-Illinoian zone sits at the floor. At those thresholds D∞ is set by the Pre-Illinoian cells rather than the Driftless: D∞ is 3.9 while the Driftless density is 1.1.
  - Small lakes give very long half-lives, and removing intermittent ponds makes them longer still. Two explanations fit this, and these data cannot separate them:
    - (a) small lakes really persist longer;
    - (b) many small perennial ponds on older, agricultural surfaces are constructed (stock or farm ponds) and do not belong to the glacial lake population at all.
- **Less reliable fits:**
  - `no_mask/all/0.1`: 33 divergences, ESS 261.
  - `full_mask/all/0.024`: ESS 317, with a broad, uneven posterior.
  - Every other fit has r̂ ≤ 1.01 and 0–5 divergences.

*Original question list (now answered):* Driftless role; Qal layer identity and whether it includes outwash; intermittent ponds; headline threshold; Pre-Illinoian prior.
