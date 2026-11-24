# Initial model comparison — Step 7

Four fixed candidates were fitted on **1,600 development-training rows** and evaluated on **400 validation rows**. The extra prior-probability dummy complements the required hard-majority baseline. No tuning, validation-based setting changes, or test-table loading occurred. See [the comparison CSV](./model_comparison.csv), [full metrics/settings](./model_comparison.json), and [executed notebook](../notebooks/02_model_comparison.ipynb).

## Model classes, objectives, and fitting methods

| Candidate | Model/objective | Fitting and assumptions |
| --- | --- | --- |
| Majority | Constant class prediction; hard one-hot probabilities | Count training labels and always select the most frequent class; ignores predictors. |
| Prior probability | Constant empirical class distribution | Estimate class frequencies from training labels; same argmax labels as majority, but finite nonzero probabilities for every observed class. |
| Logistic regression | Linear softmax classifier; L2-regularized multinomial negative log-likelihood | L-BFGS, C=1, l1_ratio=0 (L2), tolerance 1e-4, maximum 1000 iterations; linear pairwise boundaries in transformed feature space. |
| Shallow tree | Piecewise-constant class probabilities in axis-aligned regions | Greedy Gini-impurity splits, maximum depth 4, minimum 5 training rows per leaf; no gradient optimizer or logistic-regression objective. |

Every candidate receives a fresh complete preprocessing/classifier pipeline. Imputation, scaling, and category vocabularies are fitted on development-train only. Numeric scaling is retained for a common pipeline; it is not essential to tree split ordering. No class weighting or probability calibration is used. The shared seed is 42.

## Aggregate comparison

| Model | Train accuracy | Validation accuracy | Train macro F1 | Validation macro F1 | Validation log loss | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.4644 | 0.4650 | 0.2114 | 0.2116 | 19.2834 | 0.0619 |
| prior_probability | 0.4644 | 0.4650 | 0.2114 | 0.2116 | 0.9755 | 0.0152 |
| logistic_regression | 0.6150 | 0.6000 | 0.5471 | 0.5327 | 0.8177 | 0.0350 |
| shallow_tree | 0.6637 | 0.6350 | 0.4694 | 0.4496 | 0.8646 | 0.0182 |

Fit times include preprocessing and are single local wall-time measurements, not a controlled benchmark. Macro F1 is still the primary selection criterion; these fixed candidates are an initial comparison, not a frozen final model.

## Per-class validation results

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| majority | healthy | 0.0000 | 0.0000 | 0.0000 | 48 |
| majority | moderate_stress | 0.4650 | 1.0000 | 0.6348 | 186 |
| majority | high_stress | 0.0000 | 0.0000 | 0.0000 | 166 |
| prior_probability | healthy | 0.0000 | 0.0000 | 0.0000 | 48 |
| prior_probability | moderate_stress | 0.4650 | 1.0000 | 0.6348 | 186 |
| prior_probability | high_stress | 0.0000 | 0.0000 | 0.0000 | 166 |
| logistic_regression | healthy | 0.5714 | 0.2500 | 0.3478 | 48 |
| logistic_regression | moderate_stress | 0.5735 | 0.6505 | 0.6096 | 186 |
| logistic_regression | high_stress | 0.6369 | 0.6446 | 0.6407 | 166 |
| shallow_tree | healthy | 0.0000 | 0.0000 | 0.0000 | 48 |
| shallow_tree | moderate_stress | 0.5741 | 0.8333 | 0.6798 | 186 |
| shallow_tree | high_stress | 0.7615 | 0.5964 | 0.6689 | 166 |

Healthy-class recall is 25.00% for logistic regression and 0.00% for the shallow tree, across 48 healthy validation records. The tree's higher overall accuracy therefore conceals complete failure to recover healthy records in this run. Logistic regression recovers some healthy records, but its 25% recall still leaves substantial minority-class error.

Precision/F1 use zero_division=0 where a class is never predicted. All metrics and confusion matrices preserve the fixed three-class list, including classes a model fails to predict.

## Probability quality and ordering

Probability columns are explicitly reordered to healthy/moderate stress/high stress. Log loss uses matching integer target codes so alphabetical class sorting cannot exchange stress-class columns. Hard-majority probabilities assign zero probability to every nonmajority label; sklearn clips these to float64 machine epsilon for finite log loss. That value is a numerical convention, not evidence of a useful probability forecast. The prior dummy is the more informative constant-probability reference.

## Convergence and train/validation gaps

- **majority:** training minus validation accuracy = -0.0006; macro-F1 gap = -0.0002. Fitting warnings: 0. Diagnostics: `{"fitted_class_priors_estimator_order": [0.12, 0.415625, 0.464375]}`.
- **prior_probability:** training minus validation accuracy = -0.0006; macro-F1 gap = -0.0002. Fitting warnings: 0. Diagnostics: `{"fitted_class_priors_estimator_order": [0.12, 0.415625, 0.464375]}`.
- **logistic_regression:** training minus validation accuracy = 0.0150; macro-F1 gap = 0.0144. Fitting warnings: 0. Diagnostics: `{"iterations": [42]}`.
- **shallow_tree:** training minus validation accuracy = 0.0287; macro-F1 gap = 0.0198. Fitting warnings: 0. Diagnostics: `{"actual_depth": 4, "leaves": 16}`.

The highest initial validation macro F1 belongs to **logistic_regression**. Training/validation gaps must be interpreted alongside absolute scores: a small gap can indicate underfitting, while a larger positive gap can indicate overfitting or split variability. The bounded tree cannot memorize arbitrary training records, and the logistic model imposes linear decision boundaries; residual errors may require richer features/interactions. No settings were changed after observing these results, and one validation split does not establish stable superiority.

## Figures

### Training and validation metrics

![Training and validation metrics](figures/model_comparison_metrics.png)

The train/validation gaps are a first check of generalization, not proof of it. Accuracy can hide poor healthy-class recall; compare macro F1 and the per-class results before judging the candidates.

### Validation confusion matrices

![Validation confusion matrices](figures/model_comparison_confusions.png)

Both dummy baselines predict only the majority class and miss every healthy/high-stress record. The logistic and tree matrices expose which class confusions remain; all panels use the same class order and color scale.

### Per-class recall

![Per-class recall](figures/model_comparison_recall.png)

Recall measures the fraction of each true class recovered. The minority healthy class must be assessed separately from aggregate accuracy; these estimates come from one 400-record validation partition and are not final test results.

## Saved outputs and reproduction

Candidate preprocessing/classifier pipelines are stored locally under the development-prefixed files in [models/](../models/). Reloading each pipeline reproduced its validation probabilities. Per-record validation labels/probabilities are saved locally in [the prediction CSV](../data/processed/model_comparison_validation_predictions.csv). The summary binds outputs to the raw revision/hashes, membership hash, package versions, seed, settings, class order, fit warnings, and metrics. These are development artifacts, not final production or final-test artifacts.

```powershell
& ../.venv/Scripts/python.exe -m src.compare_models
```

Run from the repository root. [The comparison module](../src/compare_models.py) imports [shared evaluation](../src/evaluate.py), [pipeline builders](../src/pipeline.py), and saved development splits. [The notebook](../notebooks/02_model_comparison.ipynb) runs the same workflow and embeds all figures with interpretations.

The synthetic/deterministic labels limit interpretation: validation success shows patterns learned within this synthetic snapshot, not validated real-plant diagnosis. Course connections: lecture Sections 8.3–8.5 and [Assignment 3, Q1 and Q10–Q12](../Assigments/assignment_3_3.py). The tree is a practical nonlinear comparison beyond the supplied derivations. Next is Step 8's math explanation; tuning/selection remains Step 9, and final test evaluation remains Step 11.
