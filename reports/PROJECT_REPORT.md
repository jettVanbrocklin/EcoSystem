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
- [Assignment 3, Q5](../../Assigments/assignment_3_3.py): selecting a validation winner creates optimism, motivating an untouched final test set.
- [Assignment 3, Q10–Q11](../../Assigments/assignment_3_3.py): the cross-entropy gradient and separability motivate the classifier and its regularization.
- [Assignment 2, Q5](../../Assigments/assignment_2_1.py): base-rate effects motivate reporting precision and recall alongside accuracy.

### Completion and handoff

Step 1 is complete: the problem, expected classes, model ingredients, selection metric and tie rule, and synthetic-data limitation are documented. Dataset validation remains scheduled for Steps 3–4. No training, evaluation, or performance claims have been made. Next: prepare the local workspace and environment in Step 2.
