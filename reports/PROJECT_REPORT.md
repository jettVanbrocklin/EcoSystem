# Plant-health classification project report

## Problem definition — Step 1

Status: defined before implementation or inspection of model results. This section records the decisions required by Step 1 of [the working checklist](../ATTACK_PLAN.md), using [the project plan](../PLAN.md) and the course materials below.

### Problem and scope

Build a reproducible supervised multiclass classifier that predicts `health_class` from a single record of plant context and environmental measurements. Use the planned `soumikmahato/plant-health-points-synthetic` dataset, preserve its provided train/test separation, and perform development, training, evaluation, and inference locally on a CPU.

The expected target classes are `healthy`, `moderate_stress`, and `high_stress`. These are nominal classification labels for this workflow: do not encode them as a continuous regression target or assume that the spacing between stress levels is measurable. The labels and schema are provisional until verified against the local dataset snapshot in Steps 3–4; no dataset has been downloaded or audited as part of Step 1.

The planned inputs are:

| Input type | Columns |
| --- | --- |
| Categorical context | `plant_type`, `habitat`, `season` |
| Numeric measurements | `air_humidity_pct`, `soil_humidity_pct`, `ethylene_ppm`, `co2_ppm`, `o2_ppm`, `ldr_on_plant`, `leaf_color_index`, `temperature_c` |

Exclude `health_class` from the predictors because it is the target. Exclude `health_score` because it is a separate label and a potential source of target leakage. Retain `record_id` only for traceability and split-integrity checks. Verify measurement units and meanings during the data audit rather than inferring them solely from column names.

The final local pipeline should accept the documented input columns and return a predicted class and a probability for each class, with the probability-column order saved explicitly. Sensor hardware, watering automation, a user interface, and hosting are outside the present scope.

### The three course ingredients

| Ingredient | Decision |
| --- | --- |
| Model class | Multiclass logistic regression: one linear score per class, converted to probabilities with softmax. Fit imputation, standardization, and categorical encoding using training data only. |
| Objective | Minimize mean multiclass cross-entropy plus an L2 penalty on the weight matrix, leaving the intercept unpenalized. |
| Optimizer | Use numerical optimization, initially scikit-learn's L-BFGS solver. A manually implemented gradient-descent classifier is an optional learning exercise. |

For a transformed input row `x_i`, define the class scores as `z_i = x_i W + b` and probabilities as `p_ik = softmax(z_i)_k`. With `n` training examples, use the mathematical convention:

```text
J(W, b) = -(1/n) sum_i log(p_i,y_i) + (lambda/2) ||W||_F^2
```

Here `lambda > 0` controls regularization. This equation defines the educational objective; verify the library's penalty scaling before comparing its `C` parameter with a hand-coded `lambda`. Softmax scores share an offset redundancy, so equivalent predictions need not imply identical coefficient matrices.

Regularization constrains coefficient growth and helps when synthetic labels make the classes nearly separable. Compare this primary model against a majority-class baseline and a shallow decision tree. The tree provides a nonlinear comparison and uses greedy impurity-based splitting rather than the logistic regression objective.

### Evaluation decisions fixed in advance

Use **macro F1** as the primary selection metric: calculate F1 separately for each of the three expected classes and average them equally. Always supply the full class list when calculating metrics. Record undefined precision/F1 as zero under a documented `zero_division=0` convention and show class support so missing or rare classes are visible.

Supporting measures are per-class precision and recall, accuracy, multiclass log loss, and a labeled confusion matrix. Pay particular attention to recall for `high_stress` and errors between stress categories, while keeping macro F1 as the selection criterion. High accuracy alone can conceal failure on a minority class. Log loss measures probability quality and penalizes confident mistakes; it does not replace macro F1 as the primary criterion.

Define a negligible macro-F1 difference as **0.005 absolute** (half a percentage point on the 0–1 scale). This is a predeclared practical tolerance, not a claim of statistical equivalence. Apply the selection rule as follows:

1. Find the highest validation macro F1 among the declared candidates.
2. Keep candidates within 0.005 of that score, including the best candidate.
3. Prefer the simpler model family in this fixed order: majority baseline, logistic regression, shallow decision tree. Within logistic regression prefer stronger L2 regularization; within trees prefer lower depth, then larger minimum leaf size.
4. If candidates remain tied in simplicity, prefer lower validation log loss, calculated using aligned class probabilities. If still tied, use the first candidate in the recorded search order. Record any clipped zero probabilities used for log-loss calculation, especially for the majority baseline.

Use a fixed stratified development-train/validation split within the provided training data, with seed 42. Select preprocessing choices, features, regularization, tree complexity, and any optional calibration using development data only. If cross-validation is needed, use the same folds for all compared configurations and fit preprocessing separately within each fold; apply the selection rule to mean fold macro F1. Record the adopted protocol before searching.

Keep the provided test data reserved for the final evaluation stage after the settings are frozen. Early test access is limited to structural integrity checks such as schema and duplicate overlap. Refit the selected configuration on all provided training data before final evaluation. Do not choose settings from test results.

Predict the class with the greatest probability. No asymmetric error costs or custom stress threshold are assumed. Any future cost-based decision rule must be defined and selected on development data before final evaluation.

### Interpretation and limitations

This is an **educational modeling experiment on synthetic data**. The existing project plan describes the source as deterministic and synthetic; preserve that limitation and confirm the source documentation during dataset acquisition. Good held-out performance would show success at learning patterns in this synthetic dataset. It would not establish biological causation, validated real-plant diagnosis, or suitability for plant-care decisions.

Environmental measurements may omit disease, pests, root damage, and other causes of poor health. Deployment claims would require independently labeled real observations and external validation. There is no minimum accuracy target that justifies changing the evaluation protocol or hiding a disappointing result.

### Course connections

- [Lecture notes](../documents/COE379L_lecture_notes%20%281%29.pdf): Chapter 1 supplies the model/objective/optimizer framing; Sections 7.5–7.6 connect conditional likelihood to prediction; Sections 8.3–8.5 develop cross-entropy, class decisions, and softmax.
- [Lecture notes](../documents/COE379L_lecture_notes%20%281%29.pdf): Section 6.8.1 separates training, validation, and final test evaluation; Example 8.4.2 motivates class-specific metrics; Appendix C.7 supports local execution.
- [Assignment 3, Q5](../Assigments/assignment_3_3.py): selecting a validation winner creates optimism, motivating an untouched final test set.
- [Assignment 3, Q10–Q11](../Assigments/assignment_3_3.py): the cross-entropy gradient and separability motivate the classifier and its regularization.
- [Assignment 2, Q5](../Assigments/assignment_2_1.py): base-rate effects motivate reporting precision and recall alongside accuracy.

### Completion and handoff

Step 1 is complete: the problem, expected classes, model ingredients, selection metric and tie rule, and synthetic-data limitation are documented. Dataset validation remains scheduled for Steps 3–4. No training, evaluation, or performance claims have been made.

## Local workspace and environment — Step 2

Step 2 is complete. The existing parent-directory environment uses Python 3.11.5, and the required scientific/Jupyter packages have been installed and pinned in [requirements.txt](../requirements.txt). `pip check` reported no broken requirements. A project-local **EcoSystem (local CPU)** kernel points to that same interpreter, and [the first notebook](../notebooks/01_data_audit.ipynb) executed successfully in a fresh kernel with its outputs saved.

[The environment metadata](./environment.json) records the verified interpreter, platform, direct package versions, and seeds; [the full snapshot](./environment.freeze.txt) records all installed packages. [The shared configuration](../src/config.py) defines explicit split, model, and bootstrap seeds of 42. The seeded NumPy example produced `[8, 77, 65, 43, 43]`.

The planned local directories, ignore rules, and [run-metadata example](./run_metadata.example.json) are present. [Local setup instructions](../LOCAL_SETUP.md) explain interpreter/kernel selection, environment recreation, snapshot regeneration, and generated-output locations. Optional JAX/Optax are deferred; NumPy and scikit-learn provide the core CPU workflow. Course links were updated after the assignments moved inside the repository.

At the end of Step 2, no dataset had been downloaded and no model had been trained.

## Local dataset snapshot — Step 3

Step 3 is complete. [The acquisition module](../src/download_data.py) downloaded the original train/test CSVs and source card from dataset revision `d7b9f23bce2f5cf02c3be85c38506cf386d9d0b2`. [The provenance record](./dataset_snapshot.json) includes the exact source URLs, UTC download time, SHA-256 hashes, byte sizes, and structural counts. Original bytes are preserved locally under [data/raw/](../data/raw/); no cleaning or re-splitting has occurred.

The source files match the plan: **2,000 training rows and 500 test rows, with 14 columns each**. The source card describes deterministic synthetic generation and cautions against scientific, agronomic, or plant-care use. Neither the card nor the repository files declare a license at this revision; this is recorded as unspecified, with no reuse license inferred. The expected label names are documented by the card; empirical label auditing remains Step 4.

[The local loader](../src/data.py) checks snapshot identity, file hashes, schema, numeric parsing, and row counts. It never downloads files. Both real splits loaded successfully with network calls blocked, and repeated acquisition verified the existing files without rewriting them. Six self-contained [integrity/offline tests](../tests/test_data.py) passed, covering missing setup, corruption, changed schema/counts, offline reuse, and partial-snapshot overwrite protection.

Test-data access was limited to integrity and structural validation. No test-label summaries, feature distributions, training, or performance evaluation were performed. [Setup instructions](../LOCAL_SETUP.md) now include acquisition, offline verification, and test commands. Next: establish feature/leakage boundaries and save development splits in Step 4.

## Data and leakage boundaries — Step 4

Step 4 is complete. [The boundary report](./DATA_BOUNDARIES.md) documents the eight numeric and three categorical predictors, explicit exclusions for both labels and the ID, source-confirmed units and remaining ambiguities, development-training dtypes/missingness/vocabularies, and the structural integrity audit. The training label vocabulary matches the three expected classes. Unknown gas-reading units are preserved as unknown rather than silently converted.

The audit found no duplicate IDs, exact duplicate predictor rows, conflicting training labels for identical predictors, or exact train/test ID/predictor overlap. No persistent plant IDs or timestamps are supplied, so the declared seed-42 stratified random split was retained. [Saved membership](../data/processed/split_membership.csv) covers all 2,500 source records exactly once: 1,600 development-training, 400 validation, and 500 reserved test records. [Metadata](./data_boundaries.json) binds membership to the source revision and hashes; repeated generation gives identical membership bytes.

[The feature helper](../src/schema.py) returns predictors, target, and IDs separately. [The development loader](../src/splits.py) verifies saved membership and loads only training-source records. Twelve data/split tests passed, including rejection of overlap, membership tampering, and test-row entry into development. The notebook executes in a fresh kernel and displays development-training schema evidence without test distributions or test-label summaries.

No preprocessing or model fitting has occurred. Next: explore development-training data in Step 5.

## Development-training exploration — Step 5

Step 5 is complete in [the EDA report](./EDA_REPORT.md), [machine-readable summary](./eda_summary.json), and [executed notebook](../notebooks/01_data_audit.ipynb). All exploratory statistics and nine labeled figures use only the saved 1,600 development-training rows; validation and test data are excluded from this analysis.

Development-train contains 192 healthy records (12.00%), 743 moderate-stress records (46.44%), and 665 high-stress records (41.56%). A constant-majority prediction would achieve 46.44% accuracy on these same records; this is a descriptive reference, not held-out performance. These proportions reinforce the predeclared macro-F1 criterion and the need to report per-class recall.

Sensor-to-sensor Pearson correlations are weak: the largest absolute off-diagonal value is about 0.044. This does not establish independence or rule out nonlinear relationships. Class-conditional plots show shifts in leaf color, ethylene, and oxygen with overlap; context categories also differ in class composition. These are associations in synthetic data, not causal biological findings.

There are no missing/nonfinite sensor values, documented-bound violations, or overall-development 1.5-IQR flags. Within-class boxplot fences differ and can flag observations without contradicting that overall screen. Unknown gas units remain unresolved. No corrections, conversions, imputations, or row removals were performed.

The excluded `health_score` occupies nonoverlapping observed ranges: high stress 0–54, moderate stress 55–79, and healthy 80–100. Thresholds at 55 and 80 reproduce every development-training label; no observed score belongs to multiple classes. This is post-hoc evidence of synthetic label-generation leakage, not an independently evaluated model. The score remains excluded from all predictor lists.

The notebook embeds all nine figures with interpretations, and [local instructions](../LOCAL_SETUP.md) document report/figure regeneration. Raw files and saved membership are unchanged. No model or preprocessing has been fitted; next is Step 6, the reusable preprocessing pipeline.
