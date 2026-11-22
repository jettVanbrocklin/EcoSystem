# Data and leakage boundaries — Step 4

The pinned snapshot has been checked for structural independence. Its provided training split is now divided into **1,600 development-training records and 400 validation records**, stratified by `health_class` with seed 42. The provided **500-record test split stays reserved**. No preprocessing or model fitting has occurred.

## Modeling inputs and labels

[The feature allowlist](../src/schema.py) contains eight numeric and three categorical predictors. It deliberately differs from the raw loader's numeric-column list, which also contains the separate `health_score` label. Use `features_target_ids(frame)` to obtain separate copies of the predictor table, target, and traceability IDs; never use all raw numeric columns as predictors.

| Field | Role | Loaded dtype | Units / documented valid values | Observed development-training range or categories |
| --- | --- | --- | --- | --- |
| `air_humidity_pct` | Numeric predictor | float64 | Relative humidity, %; percentage bounds 0–100 | 25.0–93.9 |
| `soil_humidity_pct` | Numeric predictor | float64 | Soil moisture, %; measurement/calibration method unspecified | 8.1–90.0 |
| `ethylene_ppm` | Numeric predictor | float64 | Name suggests ppm; card says atmospheric gas reading without confirming the unit | 0.001–2.496 |
| `co2_ppm` | Numeric predictor | float64 | Name suggests ppm; card does not independently establish the unit | 330.5–1249.7 |
| `o2_ppm` | Numeric predictor | float64 | Actual unit unknown: values around 18–22 resemble percentages despite the name; preserve values without conversion | 18.2–21.5 |
| `ldr_on_plant` | Numeric predictor | float64 | Simulated LDR reading, 0–1023; not a documented lux or duration measurement | 30–1023 |
| `leaf_color_index` | Numeric predictor | float64 | Simulated greenness/color index, 0–100; no physical unit specified | 20.1–100.0 |
| `temperature_c` | Numeric predictor | float64 | Degrees Celsius; no source-specified admissible temperature range | 8.0–38.0 |
| `plant_type` | Categorical predictor | pandas string | Plant type; no exhaustive allowed vocabulary declared | 8 categories |
| `habitat` | Categorical predictor | pandas string | Growing context; no exhaustive allowed vocabulary declared | 6 categories |
| `season` | Categorical predictor | pandas string | Seasonal category; no exhaustive allowed vocabulary declared | 5 categories |
| `health_class` | Target, excluded from predictors | pandas string | `healthy`, `moderate_stress`, `high_stress` | All three confirmed in provided training data |
| `health_score` | Separate label, excluded from predictors | float64 | Synthetic score, 0–100 | No score/class relationship analyzed in Step 4 |
| `record_id` | Traceability only, excluded from predictors | pandas string | Source record identifier; not a verified persistent plant identifier | Unique and nonmissing across all partitions |

All 14 fields have **zero missing values in development-train**. The observed ranges above describe this sample, not guaranteed valid bounds for future records. Numeric dtypes describe the explicit loader representation; integer-valued readings/scores are intentionally represented as float64. No units were converted, values removed, or data imputed.

Observed categorical vocabularies in development-train are:

- `plant_type`: `aloe_vera`, `basil`, `lettuce`, `mint`, `pothos`, `rose`, `snake_plant`, `tomato`.
- `habitat`: `arid_garden`, `garden`, `greenhouse`, `hydroponic`, `indoor`, `shade_house`.
- `season`: `autumn`, `monsoon`, `spring`, `summer`, `winter`.

These are observed values rather than an exhaustive future-input contract. Later one-hot encoding must handle unseen categories. Missingness, category counts, dtypes, and observed sensor ranges are also saved in [the machine-readable audit](./data_boundaries.json); its descriptive scope is development-train only.

## Independence and test access

The structural audit found zero missing/duplicate record IDs, zero repeated complete predictor rows within either provided split, and zero exact ID or complete-predictor overlap across them. There were zero predictor groups with conflicting labels in the provided training data. Predictors used in duplicate checks exclude both labels and the record ID. Conflicting test labels were not inspected; with no duplicated predictor rows or cross-split predictor overlap, label comparisons were unnecessary for this check.

These checks establish exact-record separation, not statistical independence or real-world representativeness. They do not rule out near-duplicates or hidden relationships. The source provides record IDs and categorical plant types, but no persistent individual-plant ID or measurement timestamp. Its card describes synthetic records generated with broad plant-specific ranges and a deterministic seed; it does not establish IID sampling or a longitudinal/grouped sampling design. Accordingly, retain the planned stratified random development split. Do not treat plant type as an individual-plant group or invent time ordering from record numbers. Revisit the protocol if actual plant identities or temporal data become available.

Test access in Step 4 was limited to schema/row-count validation, ID checks, and exact predictor-overlap checks. No test class counts, score summaries, feature distributions, subgroup metrics, or fitted-model results were inspected. Development-loading code loads only the raw training table; test file hashes may be verified without loading the test table.

## Reproducible membership and reuse

[The split module](../src/splits.py) produces [split_membership.csv](../data/processed/split_membership.csv) with source split, original zero-based row position, record ID, and assigned partition for all 2,500 records. It saves [split_metadata.json](../data/processed/split_metadata.json) with source hashes/revision, seed, splitting method, membership hash, feature/label definitions, and structural evidence. [The report copy](./data_boundaries.json) keeps this summary available when generated local files are not committed. Raw CSVs remain unchanged.

The split process checks that every source row appears exactly once, every record ID is unique, partition counts match the declared 80/20 design, and test rows cannot enter development partitions. Repeated generation produces identical membership bytes. The loader verifies saved membership against the raw snapshot and rejects altered membership, source identity, or record-to-row mappings.

Run from the repository root:

```powershell
& ../.venv/Scripts/python.exe -m src.splits
```

Then use `load_development_split("development_train")` or `load_development_split("validation")` from [src/splits.py](../src/splits.py). Saved memberships determine the ordering; both development partitions are returned in original source-row order. Obtain model inputs through [features_target_ids](../src/schema.py). The raw test loader remains an explicit separate operation for the permitted integrity checks and eventual final evaluation.

The [split tests](../tests/test_splits.py) check the allowlist, overlap/conflict rejection, reproducible persistence, test-row exclusion, integrity checks without test labels/scores, and development-only loading. The notebook now links the [problem definition](./PROJECT_REPORT.md), verifies saved boundaries, and displays development-training schema evidence. All 12 data/split tests and fresh-kernel notebook execution pass.

Step 4 is complete. Step 5 will explore development-training distributions, class balance, correlations, and possible score/class leakage. All learned transformations must remain fitted within development-train (or the training folds used later).
