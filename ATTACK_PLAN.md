# Local plan of attack: plant-health classification

Use this as the working progress checklist alongside [PLAN.md](./PLAN.md). The original plan defines the project; this document puts the work in execution order. Check a box only when its output exists and has been verified. All implementation tasks start unchecked because the repository currently contains planning documents and course material, rather than an implemented training workflow.

The default path is a CPU-based project on your own computer: local data files, local Jupyter notebooks, local training, and local saved results. Internet access is needed initially to obtain packages and the dataset; after those are saved, the analysis, training, evaluation, and inference should work offline. No hosted notebook, paid API, cloud training, deployment, or GPU is needed for the core project.

## 1. Fix the problem definition before coding

- [x] Read [PLAN.md](./PLAN.md) and write a short problem statement: predict `health_class` from plant context and environmental measurements.
- [x] Record the three classes: `healthy`, `moderate_stress`, and `high_stress`. Confirmed against the provided training data in Step 4.
- [x] Define the three course ingredients: model class = regularized multiclass logistic regression; objective = cross-entropy plus an L2 penalty; fitting procedure = numerical optimization. A nonlinear classifier will provide a comparison.
- [x] Choose macro F1 as the primary model-selection metric, with per-class recall, accuracy, and log loss as supporting measures. Define negligible as within 0.005 of the best macro F1; prefer the simpler model, then lower validation log loss, using the fixed ordering in the report.
- [x] Record that this is an educational experiment on synthetic data, with no claim of validated real-plant diagnosis.

**Output completed:** [Project report: problem definition](./reports/PROJECT_REPORT.md#problem-definition--step-1). Course connection: lecture Chapter 1, Sections 7.5–7.6, and Section 8.5.

## 2. Prepare the local workspace and environment

- [x] Work inside the Git repository containing [README.md](./README.md). Project paths are relative to that repository.
- [x] Configure the existing parent-directory Python environment, `../.venv/Scripts/python.exe` (Python 3.11.5), using the environment configuration tool and [workspace defaults](./.vscode/settings.json). Inspected its packages before installation; no additional environment was created.
- [x] Create [data/raw/](./data/raw/), [data/processed/](./data/processed/), [notebooks/](./notebooks/), [src/](./src/), [models/](./models/), [reports/](./reports/), and [reports/figures/](./reports/figures/).
- [x] Install NumPy, pandas, SciPy, scikit-learn, Matplotlib, Seaborn, JupyterLab, ipykernel, and joblib. Verified imports together and passed `pip check`; dataset-download libraries remain deferred until a download route is chosen.
- [x] Register the project-local **EcoSystem (local CPU)** kernel using the existing environment. Execute [the first notebook](./notebooks/01_data_audit.ipynb) in a fresh kernel and save its interpreter, Python version, package versions, and seed banner.
- [x] Pin direct dependencies in [requirements.txt](./requirements.txt). Save [environment metadata](./reports/environment.json) and [the complete resolved package snapshot](./reports/environment.freeze.txt).
- [x] Define seed 42 and explicit split/model/bootstrap constants in [src/config.py](./src/config.py); verify a seeded NumPy generator. Future algorithms must receive these seeds explicitly when implemented.
- [x] Add [.gitignore](./.gitignore), [a run-metadata example](./reports/run_metadata.example.json), and [local setup/regeneration instructions](./LOCAL_SETUP.md). Verify generated data, models, plots, and kernel registrations are ignored.
- [x] Resolve the optional JAX decision: defer JAX/Optax and use the NumPy/scikit-learn CPU workflow. Compatibility and x64 verification are required only if that optional extension is undertaken later.

**Output completed:** a working local kernel, an executed setup notebook, and [documented environment](./LOCAL_SETUP.md). The fresh-kernel run and `pip check` passed. Lecture Appendix C.3 covers precision, C.6 reproducibility, and C.7 local execution. Next: Step 3, dataset acquisition.

## 3. Obtain and preserve a local dataset snapshot

- [x] Download the provided CSV splits into [data/raw/](./data/raw/) at revision `d7b9f23bce2f5cf02c3be85c38506cf386d9d0b2` using [the explicit acquisition command](./src/download_data.py).
- [x] Preserve the source dataset card and raw-file manifest locally. Record revision, UTC download time, source URLs, SHA-256 hashes, and use cautions in [dataset metadata](./reports/dataset_snapshot.json). No license is declared at this revision; do not infer one.
- [x] Confirm the expected sizes from [PLAN.md](./PLAN.md): 2,000 training rows and 500 test rows, each with the expected 14 columns. No discrepancy was found.
- [x] Build [src/data.py](./src/data.py) with local-only loading, snapshot hash checks, required schema/type checks, and expected row counts. Acquisition remains a separate command and refuses to overwrite existing raw files.
- [x] Verify both saved splits load with network calls blocked, repeat acquisition reuses the snapshot without changing its bytes or modification times, and missing setup gives a clear acquisition instruction. All six [snapshot tests](./tests/test_data.py) pass.

**Output completed:** original local raw-file bytes, saved source documentation, and [versioned provenance](./reports/dataset_snapshot.json). Hash verification detects changes; no labels or feature distributions were summarized. Next: Step 4, data/leakage boundaries and development splitting.

## 4. Establish the data and leakage boundaries

- [x] Extend [the data-audit notebook](./notebooks/01_data_audit.ipynb) with the problem definition, provenance, seed/environment banner, and saved boundary/schema checks. Fresh-kernel execution passed.
- [x] Define all eight numeric predictors explicitly in [src/schema.py](./src/schema.py), separate from the raw numeric-column list that also contains `health_score`.
- [x] Define categorical predictors explicitly: `plant_type`, `habitat`, and `season`.
- [x] Create the target from `health_class`; [the allowlist helper](./src/schema.py) returns predictors, target, and IDs separately, excluding both labels and the ID from model inputs.
- [x] Document units, loaded dtypes, documented bounds, development-training ranges, missingness, category counts, and unknown meanings in [the boundary report](./reports/DATA_BOUNDARIES.md). Gas-reading units remain uncertain; no conversion was applied.
- [x] Audit duplicate IDs/predictor rows and conflicting training labels. All checks are zero, including exact train/test ID and predictor overlap; [the structural evidence](./reports/data_boundaries.json) records the result.
- [x] Reserve the provided test split: access only schema/counts, IDs, and exact predictor-overlap checks; do not summarize test labels/scores or feature distributions.
- [x] Save a seed-42 stratified 80/20 split of the provided training data: 1,600 development-training and 400 validation records.
- [x] Save [membership](./data/processed/split_membership.csv) and [metadata](./data/processed/split_metadata.json); verify complete/disjoint assignment, deterministic regeneration, and source/hash binding. No persistent individual-plant ID or timestamp is provided, so retain the planned random design and document that limitation.

**Output completed:** [feature/target definitions](./src/schema.py), [boundary documentation](./reports/DATA_BOUNDARIES.md), and reproducible memberships with [a development-only loader](./src/splits.py). All 12 data/split tests pass; no preprocessing or models are fitted. Next: Step 5, development-training exploration. Lecture Section 6.8.1 and [Assignment 3, Q5](./Assigments/assignment_3_3.py) explain why selecting on evaluation data creates optimism.

**Test-set rule:** this checklist adopts a stricter boundary than the exploratory test comparisons in [PLAN.md](./PLAN.md). Move those comparisons to the final evaluation/reporting step. No model, feature, threshold, or preprocessing choice may depend on them.

## 5. Explore the development data and explain what you see

- [x] Plot development-training class counts/proportions: healthy 192 (12.00%), moderate stress 743 (46.44%), high stress 665 (41.56%). Moderate stress is the majority class; its descriptive in-sample reference accuracy is 46.44%, not held-out baseline performance.
- [x] Plot numeric distributions, missing-value counts, category frequencies, numeric/context distributions by class, and selected sensor pairs using only development-train. Nine labeled figures are embedded in [the audit notebook](./notebooks/01_data_audit.ipynb).
- [x] Inspect sensor correlations: maximum absolute off-diagonal Pearson r is about 0.044. Explain nonlinear dependence and coefficient instability in [the EDA report](./reports/EDA_REPORT.md); weak linear correlation does not prove independence.
- [x] Check documented bounds, finite values, and overall 1.5-IQR fences. No violations or overall IQR flags were found; within-class boxplot flags use different fences. Record no corrections/conversions/imputations/removals; retain all 1,600 rows.
- [x] Audit the excluded score: observed class ranges are high stress 0–54, moderate stress 55–79, and healthy 80–100. Post-hoc thresholds reproduce all development-training labels, documenting leakage; the score remains excluded from model inputs.
- [x] Add interpretations for every plot, explaining class imbalance, overlapping sensor distributions, synthetic construction, uncertain units, and association versus causation. Save [the reproducible summary](./reports/eda_summary.json).

**Output completed:** [EDA report](./reports/EDA_REPORT.md), [machine-readable summary](./reports/eda_summary.json), and nine interpreted figures embedded in [the executed notebook](./notebooks/01_data_audit.ipynb). Offline/scope checks and fresh-kernel execution passed; raw data and split membership are unchanged. No model or preprocessing is fitted. Next: Step 6. Course connections: [Assignment 2, Q6 and Q11](./Assigments/assignment_2_1.py), probability and covariance in Chapter 5, and regularization in Chapter 6.

## 6. Build one reusable preprocessing pipeline

- [ ] Implement `src/pipeline.py` with a scikit-learn `ColumnTransformer` inside a `Pipeline`.
- [ ] For numeric features, fit median imputation and standardization on development-train only.
- [ ] For categorical features, fit most-frequent imputation and one-hot encoding with unknown categories ignored on development-train only.
- [ ] Preserve transformed feature names and a fixed class-to-probability-column mapping.
- [ ] Apply the fitted transformation to validation without refitting. When using cross-validation later, pass the entire pipeline into the search so each fold fits its own preprocessing.
- [ ] Verify missing values, an unseen category, and one valid new row pass through the pipeline. Verify predictor columns never include either label or the record ID.

**Output:** a reusable preprocessing object with focused leakage and input checks. Standardization supports the conditioning and optimization lessons from [Assignment 2, Q1–Q3 and Q11](./Assigments/assignment_2_1.py).

## 7. Train the minimum useful model comparison

- [ ] Start `notebooks/02_model_comparison.ipynb`, importing reusable functions from `src/`.
- [ ] Fit a majority-class `DummyClassifier` and record validation accuracy, macro F1, per-class recall, and a confusion matrix. If recording its log loss, note that hard majority probabilities assign zero probability to other classes; a prior-probability dummy is a useful separate reference for log loss.
- [ ] Fit regularized multinomial logistic regression with a suitable multiclass solver, such as L-BFGS. Record convergence warnings, iteration count, regularization, and validation results.
- [ ] Fit a shallow decision tree as the required nonlinear comparison. Start with a bounded depth rather than an unrestricted tree.
- [ ] Record each candidate's model class, objective, and fitting method. For the tree, describe greedy splits based on an impurity criterion; it does not optimize logistic regression's cross-entropy objective.
- [ ] Save a comparison table with training/validation metrics, fit time, and model settings. Investigate whether a large training/validation gap indicates overfitting.

**Output:** baseline, logistic regression, and one nonlinear model running locally. Course connections: lecture Sections 8.3–8.5; [Assignment 3, Q1 and Q10–Q12](./Assigments/assignment_3_3.py). The tree is a practical comparison beyond the methods developed in the supplied assignments.

## 8. Connect the classifier to the class math

Complete the explanation first. The implementation exercise below is optional and can follow the core workflow without blocking it.

- [ ] Write the multiclass model as `Z = XW + b`, with rows of `X` representing samples, three columns of logits, and softmax producing one probability vector per row.
- [ ] Write the learning objective as `J = -(1/n) sum_i log(P[i, y_i]) + (lambda/2)||W||_F^2`, leaving the intercept unpenalized. State that this convention defines the gradient and that library regularization scaling must be checked before comparing parameter values.
- [ ] Derive `dW = X.T @ (P - Y)/n + lambda*W` and `db = mean(P - Y, axis=0)`, with one-hot labels `Y`. Connect this to the binary gradient in [Assignment 3, Q10](./Assigments/assignment_3_3.py).
- [ ] Explain why regularization matters when synthetic classes are nearly separable, using [Assignment 3, Q11](./Assigments/assignment_3_3.py). Explain softmax's shared-offset redundancy using lecture Section 8.5.1.
- [ ] **Optional implementation:** create a small NumPy or JAX softmax classifier using development-train's fitted preprocessing, stable log-softmax/log-sum-exp, and float64 for numerical checks. If using JAX, enable x64 before creating arrays.
- [ ] **Optional verification:** compare analytic/autodiff gradients with central finite differences on a tiny seeded example; check probability sums, finite loss values, and learning on a simple synthetic classification problem.
- [ ] **Optional experiment:** train with full-batch gradient descent, then compare with the library classifier using the same splits and objective conventions. Compare probabilities and loss rather than expecting identical coefficient matrices.

**Output:** an explanation of what the model learns and, optionally, an independently checked implementation. Gradient-check connection: [Assignment 5, Q2](./Assigments/assignment_5.py). There is no need to rewrite every library component to demonstrate understanding.

## 9. Tune a small, declared search and select the model

- [ ] Use the fixed validation split for the first working comparison. Declare a compact search before running it: for example logistic `C` in `{0.01, 0.1, 1, 10}` and tree `max_depth` in `{2, 4, 6}` with `min_samples_leaf` in `{1, 5}`.
- [ ] If results are sensitive to the split, use stratified 5-fold cross-validation inside development-train for those candidates, then use the reserved validation portion as the development check. Fit all transformations separately within each fold.
- [ ] Select using the declared macro-F1 rule. Use validation log loss and class-specific behavior to explain the result; do not switch the primary metric after seeing a preferred outcome.
- [ ] If imbalance is material, compare class weighting as a documented validation experiment. Keep the ordinary argmax decision rule unless a separate cost-based rule is explicitly defined and validated.
- [ ] Investigate suspiciously perfect results for leakage or deterministic labeling. Accept a modest result if it is honestly evaluated; do not expand the search just to obtain a desired score.
- [ ] Record the final feature list, preprocessing, model settings, selection rationale, and development results before opening the test results.

**Output:** a selected configuration with a bounded local compute cost. Course connections: lecture Section 6.8, [Assignment 3, Q4–Q5](./Assigments/assignment_3_3.py), and the regularization/conditioning work in [Assignment 2, Q9–Q11](./Assigments/assignment_2_1.py).

## 10. Check robustness on development data

- [ ] Repeat the selected configuration's development evaluation across a few declared split seeds, such as 42, 43, and 44. Record variability without selecting the best seed for reporting.
- [ ] Report validation metrics by plant type or habitat with subgroup sample counts; flag small groups whose estimates are unstable.
- [ ] Compare the full predictor set against numeric sensors only, using the same splits. Explain whether context features dominate the synthetic task.
- [ ] Inspect validation errors between `moderate_stress` and `high_stress`. Discuss what missed stress and false alarms would mean, without claiming field validity.
- [ ] Record whether these checks support the chosen model. If they motivate a change, complete and document that change before final test evaluation.

**Output:** a short robustness section. Connect the per-class discussion to lecture Example 8.4.2 and the base-rate calculation in [Assignment 2, Q5](./Assigments/assignment_2_1.py).

## 11. Refit, freeze, and evaluate on the test split

- [ ] Refit the selected pipeline on all 2,000 provided training rows, if that count is confirmed, using the already selected settings.
- [ ] Freeze the artifact, class mapping, feature list, and metadata. Also declare any baseline configurations to include in the final comparison; fit them using training data only.
- [ ] Start `notebooks/03_final_evaluation.ipynb` and run one final evaluation stage on the reserved test data for the frozen model and declared baselines.
- [ ] Save accuracy, macro F1, per-class precision/recall/F1 and support, log loss, and a labeled confusion matrix. Include balanced accuracy if imbalance warrants it.
- [ ] Inspect probability reliability with per-class reliability plots and report uncertainty from the small test sample. Do not fit a calibrator on the test labels; any calibration must have been chosen using development data.
- [ ] If practical, bootstrap the fixed test predictions with a seeded procedure to estimate confidence intervals for accuracy and macro F1. Describe these as uncertainty conditional on this dataset/model, not proof of real-world performance.
- [ ] Now summarize test class balance and train/test distribution differences as reporting diagnostics. Do not use these observations to retune the model and reuse the same test set as independent evidence.
- [ ] Save metrics, predictions, figures, and the exact model/data metadata. If final results disappoint, report that outcome; a changed model needs a newly justified evaluation protocol.

**Output:** a frozen final comparison and final metrics. Reusing saved predictions for reporting or bootstrap calculations does not create another model-selection round. Lecture Section 6.8.1 is the governing evaluation protocol.

## 12. Save and use the model locally

- [ ] Save the complete preprocessing-plus-classifier pipeline together with joblib and save metadata alongside it, including package versions, data revision, feature names, classes, seed, and selection settings.
- [ ] Implement a local inference entry point that accepts the documented predictor columns from a CSV and returns a predicted class plus all three class probabilities.
- [ ] Add clear errors for missing required columns and invalid numeric inputs; document how missing values and unseen categories are handled.
- [ ] Reload the saved artifact in a fresh process and verify predictions match the original fitted pipeline on the same example rows.
- [ ] Demonstrate a prediction using a valid example record with no target, score, or ID required as an input.
- [ ] Verify loading and inference work without internet access, and record the commands needed to reproduce them from the repository root.

**Output:** a reusable local model and an offline inference example. A hosted service or user interface is outside the current project scope.

## 13. Finish the explanation and reproducibility checks

- [ ] Explain logistic coefficients or compute appropriate importance measures. Discuss coefficient instability with correlated predictors and distinguish predictive association from causation.
- [ ] Write the limitations: synthetic/deterministic labels, excluded score leakage, limited sample size, unobserved plant problems, and the need for independent real-plant validation.
- [ ] Update [README.md](./README.md) to reflect the selected dataset and implemented workflow; its current planning-stage text predates the choices in [PLAN.md](./PLAN.md).
- [ ] Document local setup, dataset acquisition, notebook execution order, training, evaluation, and inference commands, distinguishing one-time downloads from offline steps.
- [ ] Restart every notebook kernel and run all cells in order. Confirm figures have labeled axes, metrics have class labels, and each figure/table has a short interpretation.
- [ ] Reproduce the core workflow from local raw data in a clean process with recorded dependencies and seeds. Check that training does not implicitly download anything.
- [ ] Confirm the saved model, metrics, report, and notebook outputs describe the same final configuration and dataset revision.
- [ ] Review Git changes and save an intentional project checkpoint, leaving temporary files and local environments out of the commit.

**Output:** a reproducible local project with an interpretable final report. Lecture Appendix C.6–C.8 explains reproducibility and the expectation to interpret outputs.

## Optional extensions after the core workflow works

Mark each extension as completed or explicitly deferred. None is required to finish the core classification project, and all model comparisons must stay within the development-data protocol if the final test has not yet been opened.

- [ ] **Optimizer experiment:** compare full-batch GD, mini-batch SGD, momentum, and Adam on the same softmax problem. Tune learning rates separately, compare equal sample-evaluation budgets, and warm up JAX before timing. Use [Assignment 4, Q4–Q10](./Assigments/assignment_4_2.py) and lecture Chapter 10; start with a much smaller search than the assignment's repeated simulations.
- [ ] **Generative comparison:** try shrinkage LDA/GDA on numeric features and compare with softmax. Study covariance conditioning and explicitly state the Gaussian/shared-covariance assumptions. Avoid an unregularized full covariance over redundant one-hot features. Use [Assignment 4, Q1–Q3](./Assigments/assignment_4_2.py) and lecture Chapter 9.
- [ ] **PCA experiment:** fit standardization and PCA on training folds only; visualize the leading numeric sensor directions and compare validation classification with/without PCA. Choose component count using development results, since retained variance does not guarantee retained class information. Use [Assignment 5, Q7–Q10](./Assigments/assignment_5.py) and lecture Sections 13.2–13.5.
- [ ] **Small neural network:** only if useful for the learning objective or a demonstrated modeling gap, try one or two small ReLU hidden layers with regularization and early stopping. Keep CPU training bounded and compare against simpler models. Use [Assignment 5, Q1–Q6](./Assigments/assignment_5.py) and lecture Chapters 11–12.

The PDE/reduced-order-model exercises, deep residual networks, and generative models are useful course material but do not address a necessary part of this small tabular classification task.

## Course reference map

The assignments are Colab-exported Python files with notebook-only installation syntax and extensive top-level experiments. Read and adapt individual ideas; do not import or run the whole files as project modules.

| Source | What to carry into this project |
| --- | --- |
| [Assignment 2](./Assigments/assignment_2_1.py), Q1–Q3, Q5–Q6, Q9–Q11 | Gradient descent, conditioning, base rates, covariance, and regularization |
| [Assignment 3](./Assigments/assignment_3_3.py), Q1–Q6, Q10–Q12 | Generalization gaps, validation selection, sparsity/shrinkage, cross-entropy, and separability |
| [Assignment 4](./Assigments/assignment_4_2.py), Q1–Q10 | GDA assumptions, covariance stability, mini-batches, learning-rate schedules, and optimizer comparisons |
| [Assignment 5](./Assigments/assignment_5.py), Q1–Q10 | Gradient checks, neural-network training, and optional PCA |
| [Lecture notes](./documents/COE379L_lecture_notes%20%281%29.pdf), Chapters 1, 4, 6–10, 12–13, Appendix C | Model/objective/optimizer framing, statistical evaluation, optional extensions, and local reproducibility |

**First milestone:** finish Steps 1–7 to obtain a defensible local baseline comparison. Then complete the math explanation, selection, robustness, final evaluation, saved inference, and documentation in Steps 8–13.
