"""Fixed-setting baseline comparison using development data only."""

import importlib.metadata
import json
import warnings
from datetime import datetime, timezone
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_SEED, PROJECT_ROOT
from src.evaluate import classification_metrics
from src.pipeline import build_model_pipeline, predict_class_probabilities, preprocessing_state
from src.schema import CLASSES, features_target_ids
from src.splits import PROCESSED_DIR, load_development_split


def model_candidates():
    """Predeclared settings, with no search or validation-driven retuning."""
    return {
        "majority": DummyClassifier(strategy="most_frequent", random_state=MODEL_SEED),
        "prior_probability": DummyClassifier(strategy="prior", random_state=MODEL_SEED),
        "logistic_regression": LogisticRegression(
            solver="lbfgs", C=1.0, l1_ratio=0.0, max_iter=1000,
            tol=1e-4, random_state=MODEL_SEED,
        ),
        "shallow_tree": DecisionTreeClassifier(
            criterion="gini", max_depth=4, min_samples_leaf=5, random_state=MODEL_SEED,
        ),
    }


def run_comparison():
    """Fit each complete pipeline once; evaluate train and validation, not test."""
    fit = load_development_split("development_train")
    validation = load_development_split("validation")
    X_fit, y_fit, _ = features_target_ids(fit)
    X_validation, y_validation, validation_ids = features_target_ids(validation)
    reports = PROJECT_ROOT / "reports"
    figures_dir, models_dir = reports / "figures", PROJECT_ROOT / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((PROCESSED_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    results, rows, predictions = {}, [], []
    for name, classifier in model_candidates().items():
        pipeline = build_model_pipeline(classifier)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            start = perf_counter()
            pipeline.fit(X_fit, y_fit)
            fit_seconds = perf_counter() - start
        state = preprocessing_state(pipeline)
        metrics = {}
        for partition, X, y in [("train", X_fit, y_fit), ("validation", X_validation, y_validation)]:
            probabilities = predict_class_probabilities(pipeline, X)
            predicted = pipeline.predict(X)
            metrics[partition] = classification_metrics(y, predicted, probabilities)
            if partition == "validation":
                output = pd.DataFrame({"model": name, "record_id": validation_ids.to_numpy(),
                                       "true_class": y.to_numpy(), "predicted_class": predicted})
                for label in CLASSES:
                    output[f"probability_{label}"] = probabilities[label].to_numpy()
                predictions.append(output)
        if state != preprocessing_state(pipeline):
            raise AssertionError("Evaluation altered fitted preprocessing state.")
        fitted = pipeline.named_steps["classifier"]
        diagnostics = {}
        if hasattr(fitted, "n_iter_"):
            diagnostics["iterations"] = fitted.n_iter_.tolist()
        if hasattr(fitted, "get_depth"):
            diagnostics.update({"actual_depth": int(fitted.get_depth()), "leaves": int(fitted.get_n_leaves())})
        if hasattr(fitted, "class_prior_"):
            diagnostics["fitted_class_priors_estimator_order"] = fitted.class_prior_.tolist()
        artifact = models_dir / f"development_{name}.joblib"
        joblib.dump(pipeline, artifact)
        restored = joblib.load(artifact)
        np.testing.assert_allclose(predict_class_probabilities(restored, X_validation),
                                   predict_class_probabilities(pipeline, X_validation))
        warning_messages = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in recorded]
        result = {
            "estimator": type(classifier).__name__, "parameters": classifier.get_params(deep=False),
            "fit_seconds": fit_seconds, "fit_warnings": warning_messages,
            "diagnostics": diagnostics, "estimator_class_order": pipeline.classes_.tolist(),
            "feature_names": pipeline[:-1].get_feature_names_out().tolist(),
            "metrics": metrics, "preprocessing_unchanged_during_evaluation": True,
            "artifact": f"models/{artifact.name}", "serialization_verified": True,
        }
        results[name] = result
        row = {"model": name, "fit_seconds": fit_seconds, "fit_warning_count": len(warning_messages),
               "settings": json.dumps(result["parameters"], sort_keys=True)}
        for partition in ["train", "validation"]:
            for metric in ["accuracy", "macro_f1", "balanced_accuracy", "log_loss"]:
                row[f"{partition}_{metric}"] = metrics[partition][metric]
            for label in CLASSES:
                row[f"{partition}_recall_{label}"] = metrics[partition]["per_class"][label]["recall"]
        row["accuracy_gap"] = metrics["train"]["accuracy"] - metrics["validation"]["accuracy"]
        row["macro_f1_gap"] = metrics["train"]["macro_f1"] - metrics["validation"]["macro_f1"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(PROCESSED_DIR / "model_comparison_validation_predictions.csv", index=False)
    summary = {
        "status": "initial_fixed_setting_comparison; no tuning or final model selection",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_partition": "development_train", "fit_rows": len(fit),
        "evaluation_partition": "validation", "evaluation_rows": len(validation),
        "test_table_loaded": False, "seed": MODEL_SEED,
        "dataset_revision": split["revision"], "source_sha256": split["source_sha256"],
        "membership_sha256": split["membership_sha256"],
        "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "class_order": list(CLASSES), "primary_metric": "macro_f1", "models": results,
        "predictions": "data/processed/model_comparison_validation_predictions.csv",
        "notes": [
            "All candidates fit fresh full preprocessing/classifier pipelines on development-train only.",
            "Hard predictions use estimator.predict; probabilities are explicitly reordered to project class order.",
            "Log loss uses integer-encoded targets aligned to project probability order; zero probabilities use float64 epsilon clipping.",
            "Fit times include preprocessing, are single-run wall times, and are not a rigorous speed benchmark.",
            "Validation is development evidence; the reserved test set remains for final evaluation after selection.",
        ],
    }
    figure_records = []
    names = list(results)
    short = ["Majority", "Prior", "Logistic", "Depth-4 tree"]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro F1"]):
        ax.bar(positions - 0.18, comparison[f"train_{metric}"], width=0.36, label="Development-train")
        ax.bar(positions + 0.18, comparison[f"validation_{metric}"], width=0.36, label="Validation")
        ax.set(xticks=positions, xticklabels=short, xlabel="Fixed candidate", ylabel=title, ylim=(0, 1))
        ax.legend()
    fig.suptitle("Initial model comparison — development data only")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"file": "figures/model_comparison_metrics.png", "title": "Training and validation metrics",
                           "interpretation": "The train/validation gaps are a first check of generalization, not proof of it. Accuracy can hide poor healthy-class recall; compare macro F1 and the per-class results before judging the candidates."})
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    maximum = max(max(max(row) for row in results[name]["metrics"]["validation"]["confusion_matrix"]) for name in names)
    for ax, name, title in zip(axes, names, short):
        display = ConfusionMatrixDisplay(np.asarray(results[name]["metrics"]["validation"]["confusion_matrix"]), display_labels=CLASSES)
        display.plot(ax=ax, colorbar=False, values_format="d", cmap="Blues", xticks_rotation=35, im_kw={"vmin": 0, "vmax": maximum})
        ax.set_title(title)
    fig.suptitle("Validation confusion matrices — rows are true classes, columns predicted")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_confusions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"file": "figures/model_comparison_confusions.png", "title": "Validation confusion matrices",
                           "interpretation": "Both dummy baselines predict only the majority class and miss every healthy/high-stress record. The logistic and tree matrices expose which class confusions remain; all panels use the same class order and color scale."})
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(CLASSES):
        recall = [results[name]["metrics"]["validation"]["per_class"][label]["recall"] for name in names]
        ax.bar(positions + (i-1)*0.23, recall, width=0.23, label=label)
    ax.set(xticks=positions, xticklabels=short, xlabel="Fixed candidate", ylabel="Validation recall", ylim=(0, 1.1))
    ax.legend(title="True class")
    fig.suptitle("Per-class validation recall — development data only")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"file": "figures/model_comparison_recall.png", "title": "Per-class recall",
                           "interpretation": "Recall measures the fraction of each true class recovered. The minority healthy class must be assessed separately from aggregate accuracy; these estimates come from one 400-record validation partition and are not final test results."})
    summary["figures"] = figure_records
    (reports / "model_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, comparison, reports)
    return summary


def write_report(summary, comparison, reports):
    models = summary["models"]
    lines = [
        "# Initial model comparison — Step 7", "",
        "Four fixed candidates were fitted on **1,600 development-training rows** and evaluated on **400 validation rows**. The extra prior-probability dummy complements the required hard-majority baseline. No tuning, validation-based setting changes, or test-table loading occurred. See [the comparison CSV](./model_comparison.csv), [full metrics/settings](./model_comparison.json), and [executed notebook](../notebooks/02_model_comparison.ipynb).", "",
        "## Model classes, objectives, and fitting methods", "",
        "| Candidate | Model/objective | Fitting and assumptions |", "| --- | --- | --- |",
        "| Majority | Constant class prediction; hard one-hot probabilities | Count training labels and always select the most frequent class; ignores predictors. |",
        "| Prior probability | Constant empirical class distribution | Estimate class frequencies from training labels; same argmax labels as majority, but finite nonzero probabilities for every observed class. |",
        "| Logistic regression | Linear softmax classifier; L2-regularized multinomial negative log-likelihood | L-BFGS, C=1, l1_ratio=0 (L2), tolerance 1e-4, maximum 1000 iterations; linear pairwise boundaries in transformed feature space. |",
        "| Shallow tree | Piecewise-constant class probabilities in axis-aligned regions | Greedy Gini-impurity splits, maximum depth 4, minimum 5 training rows per leaf; no gradient optimizer or logistic-regression objective. |", "",
        "Every candidate receives a fresh complete preprocessing/classifier pipeline. Imputation, scaling, and category vocabularies are fitted on development-train only. Numeric scaling is retained for a common pipeline; it is not essential to tree split ordering. No class weighting or probability calibration is used. The shared seed is 42.", "",
        "## Aggregate comparison", "",
        "| Model | Train accuracy | Validation accuracy | Train macro F1 | Validation macro F1 | Validation log loss | Fit seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.to_dict("records"):
        lines.append(f"| {row['model']} | {row['train_accuracy']:.4f} | {row['validation_accuracy']:.4f} | {row['train_macro_f1']:.4f} | {row['validation_macro_f1']:.4f} | {row['validation_log_loss']:.4f} | {row['fit_seconds']:.4f} |")
    lines.extend(["", "Fit times include preprocessing and are single local wall-time measurements, not a controlled benchmark. Macro F1 is still the primary selection criterion; these fixed candidates are an initial comparison, not a frozen final model.", "",
                  "## Per-class validation results", "", "| Model | Class | Precision | Recall | F1 | Support |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for name, model in models.items():
        for label, metrics in model["metrics"]["validation"]["per_class"].items():
            lines.append(f"| {name} | {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |")
    healthy_logistic = models["logistic_regression"]["metrics"]["validation"]["per_class"]["healthy"]
    healthy_tree = models["shallow_tree"]["metrics"]["validation"]["per_class"]["healthy"]
    lines.extend(["", f"Healthy-class recall is {healthy_logistic['recall']:.2%} for logistic regression and {healthy_tree['recall']:.2%} for the shallow tree, across {healthy_logistic['support']} healthy validation records. The tree's higher overall accuracy therefore conceals complete failure to recover healthy records in this run. Logistic regression recovers some healthy records, but its 25% recall still leaves substantial minority-class error."])
    lines.extend(["", "Precision/F1 use zero_division=0 where a class is never predicted. All metrics and confusion matrices preserve the fixed three-class list, including classes a model fails to predict.", "",
                  "## Probability quality and ordering", "",
                  "Probability columns are explicitly reordered to healthy/moderate stress/high stress. Log loss uses matching integer target codes so alphabetical class sorting cannot exchange stress-class columns. Hard-majority probabilities assign zero probability to every nonmajority label; sklearn clips these to float64 machine epsilon for finite log loss. That value is a numerical convention, not evidence of a useful probability forecast. The prior dummy is the more informative constant-probability reference.", "",
                  "## Convergence and train/validation gaps", ""])
    for name, model in models.items():
        training, validation = model["metrics"]["train"], model["metrics"]["validation"]
        lines.append(f"- **{name}:** training minus validation accuracy = {training['accuracy']-validation['accuracy']:.4f}; macro-F1 gap = {training['macro_f1']-validation['macro_f1']:.4f}. Fitting warnings: {len(model['fit_warnings'])}. Diagnostics: `{json.dumps(model['diagnostics'], sort_keys=True)}`.")
        for warning in model["fit_warnings"]:
            lines.append(f"  - {warning['category']}: {warning['message']}")
    best = max(models, key=lambda name: models[name]["metrics"]["validation"]["macro_f1"])
    lines.extend(["", f"The highest initial validation macro F1 belongs to **{best}**. Training/validation gaps must be interpreted alongside absolute scores: a small gap can indicate underfitting, while a larger positive gap can indicate overfitting or split variability. The bounded tree cannot memorize arbitrary training records, and the logistic model imposes linear decision boundaries; residual errors may require richer features/interactions. No settings were changed after observing these results, and one validation split does not establish stable superiority.", "",
                  "## Figures", ""])
    for figure in summary["figures"]:
        lines.extend([f"### {figure['title']}", "", f"![{figure['title']}]({figure['file']})", "", figure["interpretation"], ""])
    lines.extend(["## Saved outputs and reproduction", "",
                  "Candidate preprocessing/classifier pipelines are stored locally under the development-prefixed files in [models/](../models/). Reloading each pipeline reproduced its validation probabilities. Per-record validation labels/probabilities are saved locally in [the prediction CSV](../data/processed/model_comparison_validation_predictions.csv). The summary binds outputs to the raw revision/hashes, membership hash, package versions, seed, settings, class order, fit warnings, and metrics. These are development artifacts, not final production or final-test artifacts.", "",
                  "```powershell", "& ../.venv/Scripts/python.exe -m src.compare_models", "```", "",
                  "Run from the repository root. [The comparison module](../src/compare_models.py) imports [shared evaluation](../src/evaluate.py), [pipeline builders](../src/pipeline.py), and saved development splits. [The notebook](../notebooks/02_model_comparison.ipynb) runs the same workflow and embeds all figures with interpretations.", "",
                  "The synthetic/deterministic labels limit interpretation: validation success shows patterns learned within this synthetic snapshot, not validated real-plant diagnosis. Course connections: lecture Sections 8.3–8.5 and [Assignment 3, Q1 and Q10–Q12](../Assigments/assignment_3_3.py). The tree is a practical nonlinear comparison beyond the supplied derivations. Next is Step 8's math explanation; tuning/selection remains Step 9, and final test evaluation remains Step 11.", ""])
    (reports / "MODEL_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    comparison = run_comparison()
    for name, model in comparison["models"].items():
        print(name, json.dumps(model["metrics"]["validation"], sort_keys=True))
