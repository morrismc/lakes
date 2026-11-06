# Running the lakes code (terminal, Spyder, or Claude Code)

All three routes use the **same conda environment, `lakes`**, built from
[`environment.yml`](environment.yml). That way a number Claude reports and a
number you get in Spyder come from identical package versions.

## 1. One-time setup

The env is already built on this machine (Miniforge at
`C:\Users\mmorriss\AppData\Local\miniforge3`). To rebuild it elsewhere or after editing
`environment.yml`:

```bash
conda env create -f environment.yml
```

```bash
conda env update -n lakes -f environment.yml --prune
```

Data location: `lake_analysis/config.py` builds every path from `DATA_ROOT`,
which defaults to `D:\Lakes`. If the drive letter changes, set the
environment variable instead of editing the file:

```bash
setx LAKES_DATA_ROOT "F:\Lakes"
```

## 2. Check the environment

From the repo root (`D:\Lakes\Code\lakes`):

```bash
conda run -n lakes python scripts/check_env.py
```

It prints package versions, confirms `import lake_analysis` works, checks every
configured path exists, loads the CONUS parquet, and samples a tiny PyMC model.

## 3. Spyder

Spyder can use the `lakes` env either way:

- **Spyder installed separately (standalone installer or its own env):**
  Tools → Preferences → Python interpreter → "Use the following interpreter" →
  `C:\Users\mmorriss\AppData\Local\miniforge3\envs\lakes\python.exe`.
  Restart the console. `spyder-kernels` is already in the env.
- **Spyder inside the env:** `conda install -n lakes -c conda-forge spyder`,
  then launch with `conda run -n lakes spyder`.

Then set the working directory to `D:\Lakes\Code\lakes` (Run → Configuration per
file → "Working directory"), so `import lake_analysis` finds the package.

PyMC note: Spyder's IPython console on Windows can hang when PyMC sampling uses
multiprocessing. If sampling stalls, pass `cores=1` or run the script from a
terminal.

## 4. Claude Code

Claude runs the same env non-interactively:

```bash
C:\Users\mmorriss\AppData\Local\miniforge3\Scripts\conda.exe run -n lakes python <script>
```

(Calling `envs\lakes\python.exe` directly without activation fails on this
machine with a DLL error; `conda run` activates the env first.)

Claude's runs also set `PYTENSOR_FLAGS=base_compiledir=D:/Lakes/.pytensor`.
Inside the Claude desktop app, PyTensor's default compile cache resolves to a
path that hits Windows' 260-character limit. When that happens the NADI-1
Bayesian fit fails and the code silently falls back to a frequentist fit (look
for "Bayesian model failed" in the log). Spyder runs use a shorter cache path
and shouldn't need this.

Long runs (full pipeline, Bayesian fits) go in the background with their log
written under `D:\Lakes\Analysis\outputs\`.
