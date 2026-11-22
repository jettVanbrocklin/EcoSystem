# Plant Health Classification Plan

## 1. Objective and scope

Build a reproducible supervised multiclass classification workflow for the Hugging Face dataset [`soumikmahato/plant-health-points-synthetic`](https://huggingface.co/datasets/soumikmahato/plant-health-points-synthetic). The model will predict `health_class` with the classes `healthy`, `moderate_stress`, and `high_stress` from plant context and environmental measurements.

This is an educational modeling exercise. The dataset card describes the data as deterministic and synthetic, and says it is not appropriate for scientific conclusions, production agronomy, or plant-care decisions. The final report must preserve that limitation.

The implementation should satisfy the project requirements in [`README.md`](./README.md) and follow the lecture notes' framing of a supervised learning problem: define the data and model class, define an objective, optimize it, and evaluate generalization on held-out data.

## 2. Dataset and target audit

1. Download the dataset through the Hugging Face `datasets` API or its CSV files and record the dataset revision/date used.
2. Confirm the two provided splits: `train` (2,000 rows) and `test` (500 rows). Keep the provided test split untouched until final evaluation.
3. Document the columns, units, ranges, dtypes, and missing-value counts:
   - Categorical context: `plant_type`, `habitat`, `season`.
   - Numeric measurements: `air_humidity_pct`, `soil_humidity_pct`, `ethylene_ppm`, `co2_ppm`, `o2_ppm`, `ldr_on_plant`, `leaf_color_index`, `temperature_c`.
   - Labels: `health_class` for classification and `health_score` as a separate regression label.
4. Verify class counts and proportions in both splits. Check whether class proportions differ materially between train and test.
5. Verify duplicate or near-duplicate records and inspect `record_id`. Use it only for traceability; do not use it as a predictive feature.
6. Treat `health_score` as target leakage for this task. Do not include it in `X`, because it is another label and the dataset card says `health_class` is derived from the synthetic health signal. If useful, analyze its relationship with `health_class` separately as a limitation, not as a model input.
7. Check whether the synthetic construction creates deterministic rules, unusually separable classes, or feature ranges that differ by class. Report these findings rather than interpreting high performance as real-world validity.

## 3. Experimental design

Use the provided `train` split for model development and the provided `test` split exactly once for the final comparison. Within `train`, create a stratified train/validation split, for example 80/20 with a fixed seed. All preprocessing must be fitted only on the training portion and applied through a single pipeline.

Because the lecture notes emphasize reproducibility, record:

- Python and package versions.
- Dataset revision or download date.
- Random seed(s), split seed, and model settings.
- The exact feature list and excluded columns.
- Any class weighting, threshold, or hyperparameter-selection rule.

The default multiclass decision rule will be the class with the largest predicted probability. If the project later assigns different costs to false alarms versus missed stress, define those costs first and choose an operating rule from validation data; do not silently change the threshold after seeing test results.

## 4. Exploratory analysis

Create an exploratory notebook or script that:

1. Summarizes numeric distributions and categorical cardinalities.
2. Visualizes class balance and feature distributions by `health_class`.
3. Checks missing values, invalid ranges, duplicate IDs, and possible outliers.
4. Examines correlations and simple pairwise plots, while noting that correlation is not evidence of causal plant-health relationships.
5. Compares train and test feature distributions to identify possible split shift.
6. Checks whether `health_score` nearly determines `health_class`, documenting this as label-generation evidence rather than using it for training.

## 5. Preprocessing pipeline

Use scikit-learn `ColumnTransformer` and `Pipeline` so preprocessing cannot leak validation or test information:

- Numeric columns: median imputation if needed, followed by standardization for linear and neural models.
- Categorical columns: most-frequent imputation if needed, followed by one-hot encoding with unknown categories ignored.
- Drop `record_id` and `health_score` from the feature matrix.

Preserve feature names after transformation so coefficients and model diagnostics can be interpreted. Fit the final preprocessing object together with the selected classifier and save them as one deployable artifact.

## 6. Models and lecture-note connection

Compare progressively more flexible models:

1. Majority-class classifier as a minimal reference.
2. Multinomial logistic regression with cross-entropy loss. This is the primary lecture-aligned model: the lecture notes motivate probabilities through softmax and train by minimizing negative log-likelihood. Use regularization to control overfitting and address separability.
3. A tree-based baseline such as a shallow decision tree or random forest to test nonlinear interactions and provide a non-linear comparison.
4. Optional: a small MLP only if the simpler models leave a meaningful gap. Use a validation-selected architecture and early stopping/regularization; it should not replace the interpretable baselines.

For each candidate, state the model class, objective/loss, optimizer or fitting procedure, and relevant assumptions. Hyperparameters must be selected using only the development split, preferably with stratified cross-validation inside the training data.

## 7. Evaluation

Report validation results during model selection and final held-out test results after the model is frozen. Include:

- Accuracy, as a general summary only.
- Per-class precision, recall, and F1.
- Macro F1, so each health class receives equal weight.
- Confusion matrix with labeled axes and class support.
- Balanced accuracy if class imbalance is material.
- Cross-entropy/log loss and predicted-probability calibration, when probabilities are used.

Compare against the majority baseline and include uncertainty where practical, such as bootstrap confidence intervals on the test metrics. Explain which class errors matter most for the intended use and why accuracy alone can be misleading, following the lecture notes' imbalanced-classification discussion.

Also test robustness with at least one of:

- Repeated stratified validation with different seeds.
- Performance by plant type or habitat.
- A train/test distribution-shift check.
- A sensitivity analysis that removes categorical context or selected sensors.

## 8. Interpretation and limitations

Explain the selected model with coefficients, permutation importance, or feature-importance plots appropriate to that model. Clearly distinguish predictive association from biological causation. Discuss:

- Synthetic and deterministic data generation.
- Potential label leakage through `health_score`.
- Whether random variation makes the task easier than real sensor data.
- Missing phenomena such as pests, disease, root damage, and visual symptoms.
- Whether the provided test split represents future plants or future measurements.
- Why performance cannot be generalized to real plants without independently labeled observations and external validation.

## 9. Reproducible project deliverables

Create the following implementation artifacts in later work:

- `notebooks/01_data_audit.ipynb`: loading, schema checks, EDA, and leakage audit.
- `notebooks/02_model_comparison.ipynb`: baseline, logistic regression, candidate nonlinear models, and validation results.
- `notebooks/03_final_evaluation.ipynb`: one-time test evaluation, plots, interpretation, and limitations.
- `src/data.py`: dataset loading and schema validation.
- `src/pipeline.py`: preprocessing and model construction.
- `src/evaluate.py`: metrics, confusion matrix, and calibration utilities.
- `models/`: serialized final pipeline and metadata, if model files are committed or explicitly exported.
- `requirements.txt` or `pyproject.toml` with pinned/recorded versions.
- An updated project README containing setup, commands, feature definitions, target definition, evaluation protocol, and limitations.

Each notebook must run from a fresh kernel from top to bottom, include labeled figures and short interpretations, and avoid hidden state. The final selected pipeline must accept a new row with the documented input columns and return both a predicted class and class probabilities.

## 10. Definition of done

The classification task is complete when:

- The dataset schema, units, labels, split strategy, and leakage decisions are documented.
- A reproducible preprocessing pipeline and majority baseline exist.
- Logistic regression is trained with a probability-based multiclass objective and compared with at least one nonlinear model.
- Model selection uses only training/validation data.
- The untouched test split is evaluated once with accuracy, per-class precision/recall/F1, macro F1, and a confusion matrix.
- Results include reproducibility metadata and an honest discussion of the synthetic-data limitations.
- The chosen model and preprocessing steps are saved together and can score new measurements.

## References

- Project requirements: [`README.md`](./README.md)
- Course notes: [`COE379L_lecture_notes (1).pdf`](./documents/COE379L_lecture_notes%20%281%29.pdf)
- Dataset: [`soumikmahato/plant-health-points-synthetic`](https://huggingface.co/datasets/soumikmahato/plant-health-points-synthetic)
