# Reusable preprocessing — Step 6

Step 6 is complete. [The pipeline module](../src/pipeline.py) builds a fresh scikit-learn `Pipeline` containing an explicit feature selector and a `ColumnTransformer`. The verified development preprocessor was fitted on **1,600 development-training records only**, then applied to the **400 validation records without refitting**. No project classifier was trained, and test records were not loaded for this step.

## Transformation contract

| Stage | Operation | Learned from |
| --- | --- | --- |
| Input boundary | Require all 11 named predictors; copy/reorder the allowlist, drop extra fields, normalize missing values | No fitted statistics |
| Numeric imputation | Median imputation for the eight numeric predictors | Development-train only |
| Numeric scaling | Standardize imputed values using fitted means and standard deviations | Development-train only |
| Categorical imputation | Most-frequent imputation for plant type, habitat, and season | Development-train only |
| Categorical encoding | One-hot encode fitted categories; unknown categories become all-zero blocks | Development-train only |

Inputs must be pandas DataFrames. The selector drops any extra columns, including `health_class`, `health_score`, and `record_id`; they cannot enter the numeric or categorical transformations. Required missing columns, duplicate column names, unparseable numeric inputs, and infinite numeric values produce explicit errors. Numeric strings are parsed without changing measurement units. Categorical values are normalized to strings, while `None`, `NaN`, and `pd.NA` remain missing for the imputers.

Both imputers retain entirely missing training columns so output width stays stable. Scikit-learn uses a zero fallback for such empty features under these strategies; this is a technical fallback, not a measured median or meaningful category. The pinned snapshot has no entirely missing columns, but that edge case is covered by a synthetic test. Zero-variance numeric columns use the scaler's unit scale rather than producing division-by-zero values.

The resulting dense feature matrix contains **27 columns**: eight numeric outputs, eight plant-type indicators, six habitat indicators, and five season indicators. Dense output is small enough for this dataset. [The verification summary](./preprocessing_summary.json) stores every transformed name, the fitted numeric medians/means/scales/variances, and categorical modes/vocabularies. Numeric names begin with `numeric__`; categorical indicator names begin with `categorical__`.

## Fixed probability-column order

The preprocessing stage does not produce class probabilities. For later fitted classifiers, use `predict_class_probabilities` from [the pipeline module](../src/pipeline.py), which checks the estimator's actual class labels and reorders `predict_proba` columns to:

| Probability column | Class |
| ---: | --- |
| 0 | `healthy` |
| 1 | `moderate_stress` |
| 2 | `high_stress` |

Do not relabel a classifier's raw probability array with this order without reordering it. Many estimators sort class names alphabetically, placing `high_stress` before `moderate_stress`. The helper rejects unexpected classes or invalid probability arrays. A synthetic DummyClassifier test with unequal class proportions verifies the reorder; it is not a project model comparison.

## Verification and saved intermediate artifact

Run from the repository root:

```powershell
& ../.venv/Scripts/python.exe -m src.pipeline
& ../.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The verification command fits on the saved development-training partition, records learned state, transforms validation, and verifies the state remains unchanged. It also transforms a single valid row, a row with all predictor values missing, and a row with unseen values in all three categorical fields. All outputs are finite with the same width; unseen categories produce zero indicator blocks. No labels are required to fit the preprocessing stage.

The command saves [the development preprocessor](../models/development_preprocessor.joblib) and [its local metadata](../models/development_preprocessor.metadata.json). The shareable [summary](./preprocessing_summary.json) includes source revision/hashes, membership hash, transformation names/state, class mapping, and check results. Reloaded artifact output matches the original validation transformation. This is an intermediate development artifact; final model selection/refitting will save a complete classifier-plus-preprocessing pipeline later.

All **19 tests pass**, including seven focused [preprocessing tests](../tests/test_pipeline.py). They verify training-only fitted state despite extreme held-out values, exclusion of labels/IDs/extras, missing/unseen inputs, stable empty-feature handling, input errors, probability ordering, and fold-specific fitting of the complete pipeline. Fresh-kernel notebook execution also passes.

## Use in the next modeling step

Use `build_preprocessing_pipeline()` for an unfitted preprocessing object. Use `build_model_pipeline(classifier)` to append a cloned classifier to fresh preprocessing. Pass the **entire model pipeline** into cross-validation/search; candidate parameter names use the `classifier__` prefix. Each fold must learn its own medians, scaling, modes, and categories from that fold's training rows. Never transform all development rows with the saved fitted preprocessor before cross-validation.

The next step is the baseline/logistic-regression/tree comparison in Step 7. Validation has only been transformed in this step; no validation scores or test metrics have been computed. Standardization connects to the conditioning/optimization material in [Assignment 2, Q1–Q3 and Q11](../Assigments/assignment_2_1.py).
