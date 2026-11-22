# Local plan of attack: plant-health classification

Use this as the working progress checklist alongside [PLAN.md](./PLAN.md). The original plan defines the project; this document puts the work in execution order. Check a box only when its output exists and has been verified. All implementation tasks start unchecked because the repository currently contains planning documents and course material, rather than an implemented training workflow.

The default path is a CPU-based project on your own computer: local data files, local Jupyter notebooks, local training, and local saved results. Internet access is needed initially to obtain packages and the dataset; after those are saved, the analysis, training, evaluation, and inference should work offline. No hosted notebook, paid API, cloud training, deployment, or GPU is needed for the core project.

## 1. Fix the problem definition before coding

- [x] Read [PLAN.md](./PLAN.md) and write a short problem statement: predict `health_class` from plant context and environmental measurements.
- [x] Record the three expected classes: `healthy`, `moderate_stress`, and `high_stress`. Their confirmation against downloaded data remains part of Steps 3–4; they are provisional until then.
- [x] Define the three course ingredients: model class = regularized multiclass logistic regression; objective = cross-entropy plus an L2 penalty; fitting procedure = numerical optimization. A nonlinear classifier will provide a comparison.
- [x] Choose macro F1 as the primary model-selection metric, with per-class recall, accuracy, and log loss as supporting measures. Define negligible as within 0.005 of the best macro F1; prefer the simpler model, then lower validation log loss, using the fixed ordering in the report.
- [x] Record that this is an educational experiment on synthetic data, with no claim of validated real-plant diagnosis.

**Output completed:** [Project report: problem definition](./reports/PROJECT_REPORT.md#problem-definition--step-1). Course connection: lecture Chapter 1, Sections 7.5–7.6, and Section 8.5.

## 2. Prepare the local workspace and environment

- [ ] Work inside the Git repository containing [README.md](./README.md). Keep paths relative to that repository so notebooks can be rerun elsewhere.
- [ ] Select the existing parent-directory Python environment in VS Code: `../.venv/Scripts/python.exe`. It currently uses Python 3.11.5. Verify its packages before installing anything or creating another environment.
- [ ] Create the planned directories: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `models/`, `reports/`, and `reports/figures/`.
- [ ] Install or confirm NumPy, pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Jupyter, ipykernel, and joblib. Add `datasets` or `huggingface_hub` only if using that download route.
- [ ] Register/select this environment as the notebook kernel. Print the interpreter, Python version, and relevant package versions in the first notebook.
- [ ] Record working dependency versions in `requirements.txt` or `pyproject.toml`. Save the actual resolved environment versions with the experiment results.
- [ ] Set a project seed, such as 42, and use explicit seeds for splitting, models, and bootstrap resampling.
- [ ] Add ignore rules for virtual environments, notebook checkpoints, caches, and generated artifacts that should stay local. Keep a small metadata example and instructions for regenerating ignored outputs.
- [ ] If doing the optional JAX work, verify a compatible CPU installation separately. The assignments pin JAX 0.4.35 and Optax 0.2.3; verify compatibility on this Windows environment before adopting those pins. NumPy is sufficient for the core math implementation if JAX installation is inconvenient.

**Output:** a working local kernel and documented environment. Lecture Appendix C.3 covers precision, C.6 reproducibility, and C.7 local execution. No environment/package changes are required merely to use this checklist.

## 3. Obtain and preserve a local dataset snapshot

- [ ] Download `soumikmahato/plant-health-points-synthetic` from [Hugging Face](https://huggingface.co/datasets/soumikmahato/plant-health-points-synthetic) once, saving its provided train and test splits under `data/raw/`.
- [ ] Save the dataset card locally and record the dataset revision, download date, source URL, file hashes, and any license or use restrictions.
- [ ] Confirm the expected sizes from [PLAN.md](./PLAN.md): 2,000 training rows and 500 test rows. Investigate discrepancies instead of silently assuming the plan is correct.
- [ ] Build `src/data.py` to read the local snapshot and validate the schema. Keep downloading as an explicit setup action, separate from loading data during an experiment.
- [ ] Verify that the loader can read the saved data without network access and gives a clear missing-file message if setup is incomplete.

**Output:** immutable local raw files and dataset metadata. Keep raw files unchanged; save derived files separately.

## 4. Establish the data and leakage boundaries

- [ ] Start `notebooks/01_data_audit.ipynb` with the problem definition, provenance, seed, and environment banner.
- [ ] Define numeric predictors explicitly: `air_humidity_pct`, `soil_humidity_pct`, `ethylene_ppm`, `co2_ppm`, `o2_ppm`, `ldr_on_plant`, `leaf_color_index`, and `temperature_c`.
- [ ] Define categorical predictors explicitly: `plant_type`, `habitat`, and `season`.
- [ ] Create the target from `health_class`. Exclude `health_class`, `health_score`, and `record_id` from every predictor list; retain IDs separately for traceability.
- [ ] Document confirmed units, data types, valid values, missing values, and category counts. Mark units or physical meanings as unknown when the dataset card does not establish them.
- [ ] Audit duplicate IDs, duplicate predictor rows, and conflicting labels. Check overlap across provided splits as a data-integrity check; if overlap compromises independence, resolve and document the evaluation protocol before modeling.
- [ ] Reserve the provided test split. Limit early test access to structural checks such as schema, row counts, and overlap; defer test-label summaries, feature distributions, and model metrics to final evaluation.
- [ ] Split the provided training data into stratified development-train/validation portions using an 80/20 split and the fixed seed: approximately 1,600/400 rows if the expected counts hold.
- [ ] Save the split membership and assert that development-train, validation, and test records are disjoint. If plant grouping or time dependence is actually present, replace the random split with a defensible grouped or chronological design.

**Output:** explicit feature/target definitions and reproducible split membership. Lecture Section 6.8.1 and [Assignment 3, Q5](../Assigments/assignment_3_3.py) explain why selecting on evaluation data creates optimism.

**Test-set rule:** this checklist adopts a stricter boundary than the exploratory test comparisons in [PLAN.md](./PLAN.md). Move those comparisons to the final evaluation/reporting step. No model, feature, threshold, or preprocessing choice may depend on them.

## 5. Explore the development data and explain what you see

- [ ] Plot class counts and calculate class proportions in development-train. Identify the majority class and its baseline accuracy.
- [ ] Plot numeric distributions, missing-value counts, category frequencies, and feature distributions by class using development-train.
- [ ] Inspect correlations and overlapping sensors. Explain why zero correlation does not prove independence and why correlated predictors can make individual coefficients unstable.
- [ ] Check unusual ranges and outliers against the dataset documentation before removing anything. Preserve an audit trail for any correction.
- [ ] Examine the relationship between `health_score` and `health_class` using development-train only to document likely label-generation leakage. Keep the score excluded from modeling.
- [ ] Write one or two sentences interpreting each figure, including whether apparent separation could reflect deterministic synthetic generation.

**Output:** a completed exploratory section with labeled figures. Course connections: [Assignment 2, Q6 and Q11](../Assigments/assignment_2_1.py), probability and covariance in Chapter 5, and regularization in Chapter 6.

## 6. Build one reusable preprocessing pipeline

- [ ] Implement `src/pipeline.py` with a scikit-learn `ColumnTransformer` inside a `Pipeline`.
- [ ] For numeric features, fit median imputation and standardization on development-train only.
- [ ] For categorical features, fit most-frequent imputation and one-hot encoding with unknown categories ignored on development-train only.
- [ ] Preserve transformed feature names and a fixed class-to-probability-column mapping.
- [ ] Apply the fitted transformation to validation without refitting. When using cross-validation later, pass the entire pipeline into the search so each fold fits its own preprocessing.
- [ ] Verify missing values, an unseen category, and one valid new row pass through the pipeline. Verify predictor columns never include either label or the record ID.

**Output:** a reusable preprocessing object with focused leakage and input checks. Standardization supports the conditioning and optimization lessons from [Assignment 2, Q1–Q3 and Q11](../Assigments/assignment_2_1.py).

## 7. Train the minimum useful model comparison

- [ ] Start `notebooks/02_model_comparison.ipynb`, importing reusable functions from `src/`.
- [ ] Fit a majority-class `DummyClassifier` and record validation accuracy, macro F1, per-class recall, and a confusion matrix. If recording its log loss, note that hard majority probabilities assign zero probability to other classes; a prior-probability dummy is a useful separate reference for log loss.
- [ ] Fit regularized multinomial logistic regression with a suitable multiclass solver, such as L-BFGS. Record convergence warnings, iteration count, regularization, and validation results.
- [ ] Fit a shallow decision tree as the required nonlinear comparison. Start with a bounded depth rather than an unrestricted tree.
- [ ] Record each candidate's model class, objective, and fitting method. For the tree, describe greedy splits based on an impurity criterion; it does not optimize logistic regression's cross-entropy objective.
- [ ] Save a comparison table with training/validation metrics, fit time, and model settings. Investigate whether a large training/validation gap indicates overfitting.

**Output:** baseline, logistic regression, and one nonlinear model running locally. Course connections: lecture Sections 8.3–8.5; [Assignment 3, Q1 and Q10–Q12](../Assigments/assignment_3_3.py). The tree is a practical comparison beyond the methods developed in the supplied assignments.

## 8. Connect the classifier to the class math

Complete the explanation first. The implementation exercise below is optional and can follow the core workflow without blocking it.

- [ ] Write the multiclass model as `Z = XW + b`, with rows of `X` representing samples, three columns of logits, and softmax producing one probability vector per row.
- [ ] Write the learning objective as `J = -(1/n) sum_i log(P[i, y_i]) + (lambda/2)||W||_F^2`, leaving the intercept unpenalized. State that this convention defines the gradient and that library regularization scaling must be checked before comparing parameter values.
- [ ] Derive `dW = X.T @ (P - Y)/n + lambda*W` and `db = mean(P - Y, axis=0)`, with one-hot labels `Y`. Connect this to the binary gradient in [Assignment 3, Q10](../Assigments/assignment_3_3.py).
- [ ] Explain why regularization matters when synthetic classes are nearly separable, using [Assignment 3, Q11](../Assigments/assignment_3_3.py). Explain softmax's shared-offset redundancy using lecture Section 8.5.1.
- [ ] **Optional implementation:** create a small NumPy or JAX softmax classifier using development-train's fitted preprocessing, stable log-softmax/log-sum-exp, and float64 for numerical checks. If using JAX, enable x64 before creating arrays.
- [ ] **Optional verification:** compare analytic/autodiff gradients with central finite differences on a tiny seeded example; check probability sums, finite loss values, and learning on a simple synthetic classification problem.
- [ ] **Optional experiment:** train with full-batch gradient descent, then compare with the library classifier using the same splits and objective conventions. Compare probabilities and loss rather than expecting identical coefficient matrices.

**Output:** an explanation of what the model learns and, optionally, an independently checked implementation. Gradient-check connection: [Assignment 5, Q2](../Assigments/assignment_5.py). There is no need to rewrite every library component to demonstrate understanding.

## 9. Tune a small, declared search and select the model

- [ ] Use the fixed validation split for the first working comparison. Declare a compact search before running it: for example logistic `C` in `{0.01, 0.1, 1, 10}` and tree `max_depth` in `{2, 4, 6}` with `min_samples_leaf` in `{1, 5}`.
- [ ] If results are sensitive to the split, use stratified 5-fold cross-validation inside development-train for those candidates, then use the reserved validation portion as the development check. Fit all transformations separately within each fold.
- [ ] Select using the declared macro-F1 rule. Use validation log loss and class-specific behavior to explain the result; do not switch the primary metric after seeing a preferred outcome.
- [ ] If imbalance is material, compare class weighting as a documented validation experiment. Keep the ordinary argmax decision rule unless a separate cost-based rule is explicitly defined and validated.
- [ ] Investigate suspiciously perfect results for leakage or deterministic labeling. Accept a modest result if it is honestly evaluated; do not expand the search just to obtain a desired score.
- [ ] Record the final feature list, preprocessing, model settings, selection rationale, and development results before opening the test results.

**Output:** a selected configuration with a bounded local compute cost. Course connections: lecture Section 6.8, [Assignment 3, Q4–Q5](../Assigments/assignment_3_3.py), and the regularization/conditioning work in [Assignment 2, Q9–Q11](../Assigments/assignment_2_1.py).

## 10. Check robustness on development data

- [ ] Repeat the selected configuration's development evaluation across a few declared split seeds, such as 42, 43, and 44. Record variability without selecting the best seed for reporting.
- [ ] Report validation metrics by plant type or habitat with subgroup sample counts; flag small groups whose estimates are unstable.
- [ ] Compare the full predictor set against numeric sensors only, using the same splits. Explain whether context features dominate the synthetic task.
- [ ] Inspect validation errors between `moderate_stress` and `high_stress`. Discuss what missed stress and false alarms would mean, without claiming field validity.
- [ ] Record whether these checks support the chosen model. If they motivate a change, complete and document that change before final test evaluation.

**Output:** a short robustness section. Connect the per-class discussion to lecture Example 8.4.2 and the base-rate calculation in [Assignment 2, Q5](../Assigments/assignment_2_1.py).

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

- [ ] **Optimizer experiment:** compare full-batch GD, mini-batch SGD, momentum, and Adam on the same softmax problem. Tune learning rates separately, compare equal sample-evaluation budgets, and warm up JAX before timing. Use [Assignment 4, Q4–Q10](../Assigments/assignment_4_2.py) and lecture Chapter 10; start with a much smaller search than the assignment's repeated simulations.
- [ ] **Generative comparison:** try shrinkage LDA/GDA on numeric features and compare with softmax. Study covariance conditioning and explicitly state the Gaussian/shared-covariance assumptions. Avoid an unregularized full covariance over redundant one-hot features. Use [Assignment 4, Q1–Q3](../Assigments/assignment_4_2.py) and lecture Chapter 9.
- [ ] **PCA experiment:** fit standardization and PCA on training folds only; visualize the leading numeric sensor directions and compare validation classification with/without PCA. Choose component count using development results, since retained variance does not guarantee retained class information. Use [Assignment 5, Q7–Q10](../Assigments/assignment_5.py) and lecture Sections 13.2–13.5.
- [ ] **Small neural network:** only if useful for the learning objective or a demonstrated modeling gap, try one or two small ReLU hidden layers with regularization and early stopping. Keep CPU training bounded and compare against simpler models. Use [Assignment 5, Q1–Q6](../Assigments/assignment_5.py) and lecture Chapters 11–12.

The PDE/reduced-order-model exercises, deep residual networks, and generative models are useful course material but do not address a necessary part of this small tabular classification task.

## Course reference map

The assignments are Colab-exported Python files with notebook-only installation syntax and extensive top-level experiments. Read and adapt individual ideas; do not import or run the whole files as project modules.

| Source | What to carry into this project |
| --- | --- |
| [Assignment 2](../Assigments/assignment_2_1.py), Q1–Q3, Q5–Q6, Q9–Q11 | Gradient descent, conditioning, base rates, covariance, and regularization |
| [Assignment 3](../Assigments/assignment_3_3.py), Q1–Q6, Q10–Q12 | Generalization gaps, validation selection, sparsity/shrinkage, cross-entropy, and separability |
| [Assignment 4](../Assigments/assignment_4_2.py), Q1–Q10 | GDA assumptions, covariance stability, mini-batches, learning-rate schedules, and optimizer comparisons |
| [Assignment 5](../Assigments/assignment_5.py), Q1–Q10 | Gradient checks, neural-network training, and optional PCA |
| [Lecture notes](./documents/COE379L_lecture_notes%20%281%29.pdf), Chapters 1, 4, 6–10, 12–13, Appendix C | Model/objective/optimizer framing, statistical evaluation, optional extensions, and local reproducibility |

**First milestone:** finish Steps 1–7 to obtain a defensible local baseline comparison. Then complete the math explanation, selection, robustness, final evaluation, saved inference, and documentation in Steps 8–13.
