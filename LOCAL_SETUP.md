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

The first notebook verifies setup, snapshot provenance, and saved development boundaries, explores development-train only, and verifies training-fitted preprocessing against validation inputs. It embeds nine figures with interpretations and regenerates EDA/preprocessing summaries. To verify it from a fresh kernel and regenerate saved outputs and environment metadata:

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

## Verify the reusable preprocessing pipeline

```powershell
& ../.venv/Scripts/python.exe -m src.pipeline
& ../.venv/Scripts/python.exe -m unittest discover -s tests -v
```

[The pipeline module](./src/pipeline.py) provides `build_preprocessing_pipeline()` and `build_model_pipeline(classifier)`. The allowlist drops extra fields before median imputation/standardization of numeric features and most-frequent imputation/one-hot encoding of categorical features. Unknown categories are ignored. [The preprocessing report](./reports/PREPROCESSING.md) documents the input contract, 27 named outputs, fixed probability-column order, and validation checks.

The verification command fits only on development-train, transforms validation without refitting, checks missing/unseen inputs, and saves [the intermediate preprocessor](./models/development_preprocessor.joblib) plus [a shareable metadata summary](./reports/preprocessing_summary.json). No project classifier is trained. Regenerate this artifact from raw data and saved memberships with the command above. For future model search, pass the complete fresh model pipeline into cross-validation; do not reuse this fitted intermediate preprocessor across folds. Use `predict_class_probabilities` to align a fitted model's probability columns to the documented project order.

## Run the initial model comparison

```powershell
& ../.venv/Scripts/python.exe -m src.compare_models
```

[The comparison module](./src/compare_models.py) fits fresh complete pipelines for the hard-majority dummy, a prior-probability dummy, L2 multinomial logistic regression, and a depth-4 decision tree. It uses the 1,600 development-training rows for fitting and the 400 validation rows for development evaluation; it does not load the test table or tune settings.

Open [02_model_comparison.ipynb](./notebooks/02_model_comparison.ipynb) with the same **EcoSystem (local CPU)** kernel to reproduce the comparison and view embedded confusion matrices, train/validation metric plots, and per-class recall. [The report](./reports/MODEL_COMPARISON.md), [CSV table](./reports/model_comparison.csv), and [full JSON results](./reports/model_comparison.json) document metrics, settings, package versions, seed, fit times, warning/iteration diagnostics, and source/membership hashes. Local development model pipelines and per-record validation predictions are regenerated by the command above. Fitted candidates are development artifacts, with final selection and test evaluation still pending.

The first fixed comparison gives logistic regression validation macro F1 of 0.5327 and accuracy 0.6000; the tree gives macro F1 of 0.4496 and accuracy 0.6350. The tree misses all healthy validation records, which explains why accuracy and macro F1 rank the models differently. Prior-probability and hard-majority predictions have the same class metrics but very different log loss; the hard-majority result depends on numerical clipping of zero probabilities.

## Connect the classifier to the math

Read [SOFTMAX_MATH.md](./reports/SOFTMAX_MATH.md), then execute [02b_softmax_math.ipynb](./notebooks/02b_softmax_math.ipynb) with the existing local CPU kernel, or run:

```powershell
& ../.venv/Scripts/python.exe -m src.softmax_experiment
```

[The NumPy implementation](./src/softmax.py) uses float64 stable log-softmax, the analytic cross-entropy/L2 gradients, and full-batch gradient descent with training-only Armijo backtracking. The experiment checks central finite differences, extreme logits, probability sums, a simple synthetic learning problem, and agreement with a tighter L-BFGS reference on the same transformed development inputs. Local source inspection establishes lambda=1/(C*n)=0.000625 for this installed unweighted L-BFGS convention. No JAX/Optax installation is needed; the already-installed threadpoolctl package is now pinned explicitly because this experiment imports it to limit BLAS threads.

[The measured summary](./reports/softmax_math_summary.json) records objective/probability gaps, optimizer history, regularization scaling, precision, seeds, package versions, and source/membership hashes. The command updates the generated numerical section of the derivation report and a convergence plot. [The educational artifact](./models/educational_softmax.joblib) contains its own training-fitted preprocessing, NumPy parameters, class mapping, and lambda; it does not overwrite any Step 7 candidate. Validation is only a numerical comparison, never an optimizer stopping/tuning input, and the test table is not loaded.

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
