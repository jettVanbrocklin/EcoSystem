# Local setup

Run these PowerShell commands from the repository root, the folder containing [README.md](./README.md). The existing environment is one directory above it. The verified setup uses Python 3.11.5 and CPU computation.

## Use the existing environment

```powershell
& ../.venv/Scripts/python.exe -m pip install -r requirements.txt
& ../.venv/Scripts/python.exe -m pip check
```

If recreating the environment on another computer, first install Python 3.11 and run `py -3.11 -m venv ../.venv`. Installing packages requires network access; using the installed environment does not. [requirements.txt](./requirements.txt) pins the project's direct dependencies; [the complete environment snapshot](./reports/environment.freeze.txt) records all packages present during verification, including unrelated packages already in the existing environment. The snapshot is an audit record for this Windows/Python setup, not a cross-platform lockfile.

## Run the notebook locally

In VS Code, open [the data-audit notebook](./notebooks/01_data_audit.ipynb), choose **Select Kernel → Python Environments**, and select the existing parent environment. [The interpreter setting](./.vscode/settings.json) supplies the default when opening this repository as a workspace; an existing VS Code selection may override that default. The notebook's first cell prints the interpreter and package versions so the executing kernel can be verified directly.

A project-local kernel is also registered under `.jupyter/share/jupyter/kernels/ecosystem-local`. To use it with JupyterLab, launch with its search path:

```powershell
$env:JUPYTER_PATH = (Join-Path $PWD '.jupyter/share/jupyter')
& ../.venv/Scripts/python.exe -m jupyterlab
```

Choose **EcoSystem (local CPU)**. The kernel registration is local and ignored by Git because it contains a machine-specific interpreter path. Regenerate it after moving or recreating the environment:

```powershell
& ../.venv/Scripts/python.exe -m ipykernel install --prefix .jupyter --name ecosystem-local --display-name 'EcoSystem (local CPU)'
```

The first notebook verifies setup, snapshot provenance, and saved development boundaries, then explores development-train only. It embeds nine figures with interpretations and regenerates the EDA report/summary. To verify it from a fresh kernel and regenerate saved outputs and environment metadata:

```powershell
$env:JUPYTER_PATH = (Join-Path $PWD '.jupyter/share/jupyter')
& ../.venv/Scripts/python.exe -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.kernel_name=ecosystem-local notebooks/01_data_audit.ipynb
```

## Acquire once, then load offline

Run the acquisition command once with internet access:

```powershell
& ../.venv/Scripts/python.exe -m src.download_data
```

[The downloader](./src/download_data.py) uses Python's standard library and the fixed dataset revision `d7b9f23bce2f5cf02c3be85c38506cf386d9d0b2`. No Hugging Face package, token, or account is required. It preserves original CSV/card bytes under [data/raw/](./data/raw/), with a manifest recording SHA-256 hashes, byte sizes, source URLs, revision, UTC acquisition time, and source cautions. [The shareable metadata copy](./reports/dataset_snapshot.json) stays outside the ignored raw-data directory. The dataset card and repository do not declare a license at this revision; the metadata records that absence without assuming reuse rights.

A repeat acquisition command verifies the existing snapshot and makes no network request or raw-file rewrite. If a partial snapshot or hash mismatch is found, it fails rather than overwriting files. To obtain a separate copy, use `--output-dir` with a new empty directory and `--metadata-output` with a separate provenance destination. Keep the original snapshot and manifest together.

Verify local loading with:

```powershell
& ../.venv/Scripts/python.exe -m src.data
& ../.venv/Scripts/python.exe -m unittest discover -s tests -v
```

[The loader](./src/data.py) exposes `load_split("train")` and `load_split("test")`, returns pandas DataFrames, and never downloads anything. It checks snapshot identity, hashes, the 14-column schema, parseable numeric fields, and the expected 2,000/500 row counts. Numeric values are loaded as float64, textual columns as pandas strings, and missing values are retained for the later audit. It returns the complete raw table, including labels and IDs; downstream feature selection must explicitly exclude labels and IDs. Step 3 verifies test structure only, without class counts or feature-distribution analysis.

The self-contained tests create temporary synthetic file fixtures; they do not require internet or the downloaded dataset. They verify offline loading and repeat acquisition, unchanged raw bytes/modification times, clear missing-setup errors, corruption detection, schema/row-count rejection, and partial-snapshot protection.

## Preserve development/test boundaries

Step 4 fixes eight numeric and three categorical predictors in [src/schema.py](./src/schema.py). `health_class`, `health_score`, and `record_id` are excluded from every model input. The raw loader returns all columns, so use the explicit `features_target_ids(frame)` helper to construct inputs safely.

Generate the deterministic memberships and structural report:

```powershell
& ../.venv/Scripts/python.exe -m src.splits
```

This checks duplicate/overlapping IDs and complete predictor rows, then splits the provided training data into 1,600/400 stratified development-training/validation records with seed 42. The provided 500 test records remain reserved. Memberships and metadata are saved under [data/processed/](./data/processed/); a shareable summary is saved in [reports/data_boundaries.json](./reports/data_boundaries.json). Regeneration preserves raw files and produces the same memberships.

Use `load_development_split("development_train")` and `load_development_split("validation")` from [src/splits.py](./src/splits.py) in subsequent analysis/modeling. The helper verifies source and membership hashes and does not load the test table. [The boundary report](./reports/DATA_BOUNDARIES.md) documents units, unknown meanings, observed vocabularies, missingness, and why a stratified random split is appropriate for this snapshot. Test data has been used only for structural and overlap checks; do not compute test-label summaries or feature distributions before final evaluation.

## Explore development-train locally

After acquiring the raw snapshot and generating saved memberships, run:

```powershell
& ../.venv/Scripts/python.exe -m src.eda
```

[The EDA module](./src/eda.py) loads only `development_train` and writes [EDA_REPORT.md](./reports/EDA_REPORT.md), [eda_summary.json](./reports/eda_summary.json), and nine plots under [reports/figures/](./reports/figures/). It does not summarize validation/test rows or fit transformations/models. The report covers class balance, sensor/context distributions, missingness, correlations, documented bounds/IQR flags, and excluded-score leakage evidence. The [audit notebook](./notebooks/01_data_audit.ipynb) reruns the analysis and embeds plots with interpretations so they remain visible without the ignored PNGs.

All 1,600 development-training records were retained with no corrections, conversions, or imputations. The majority-class accuracy is a descriptive development-training reference; held-out baseline metrics remain Step 7. Score thresholds describe observed synthetic label generation and must never be used as score-derived predictor features. Re-running the EDA command updates generated reports/plots without changing raw data or saved memberships.

## Reproducibility and local outputs

[src/config.py](./src/config.py) defines the project root and split/model/bootstrap seeds, all initially 42. Use these constants explicitly in future splitters, randomized models, and `numpy.random.default_rng`; a shared constant alone does not seed future algorithms automatically. The notebook demonstrates a seeded generator and prints the settings.

[reports/environment.json](./reports/environment.json) records the interpreter, platform, seeds, and package versions. The first notebook regenerates that file and [the full package snapshot](./reports/environment.freeze.txt). Copy [the run-metadata example](./reports/run_metadata.example.json) for each future experiment and fill in the dataset revision/hashes, features, class order, model settings, and actual metrics. No model results are implied by that example.

| Location | Purpose and regeneration |
| --- | --- |
| [data/raw/](./data/raw/) | Preserved source CSVs, source card, and hash manifest; regenerate as a separate pinned snapshot using the acquisition command. |
| [data/processed/](./data/processed/) | Derived data and saved split membership; regenerate from raw data using the later data workflow. |
| [notebooks/](./notebooks/) | Ordered, restartable analysis notebooks. |
| [src/](./src/) | Importable shared project code and seed configuration. |
| [models/](./models/) | Future serialized pipelines and metadata; regenerate through the later training workflow. |
| [reports/](./reports/) | Project explanation, environment snapshots, and metadata examples. |
| [reports/figures/](./reports/figures/) | Generated plots; regenerate from the later analysis/evaluation notebooks. |

[.gitignore](./.gitignore) keeps caches, environments, downloaded data, fitted models, generated figures, and per-run output directories local. Directory placeholders remain tracked. Training and inference commands will be documented when those workflows exist.

JAX and Optax are deliberately deferred: they are optional and are not required for the scikit-learn CPU workflow. NumPy is available for the optional mathematical implementation. If JAX is added later, verify Windows/Python compatibility and CPU execution before adopting the assignment pins, and enable x64 before creating arrays for numerical gradient checks.
