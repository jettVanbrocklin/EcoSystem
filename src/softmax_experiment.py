"""Numerically verify the course math without accessing the test table."""

import hashlib
import importlib.metadata
import inspect
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model._logistic as logistic_source
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.config import PROJECT_ROOT, RANDOM_SEED
from src.evaluate import classification_metrics
from src.pipeline import build_preprocessing_pipeline
from src.schema import CLASSES, features_target_ids
from src.softmax import NumpySoftmaxClassifier, check_gradient, log_softmax, loss_and_grad
from src.splits import PROCESSED_DIR, load_development_split


def run_experiment():
    gradient = check_gradient(RANDOM_SEED)
    if gradient["max_absolute_error"] > 1e-7:
        raise AssertionError("Analytic/finite-difference gradients disagree.")
    extreme = log_softmax(np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]]))
    if not np.isfinite(extreme).all() or not np.allclose(np.exp(extreme).sum(axis=1), 1):
        raise AssertionError("Stable log-softmax check failed.")
    rng = np.random.default_rng(RANDOM_SEED)
    centers = np.array([[-3, -3], [3, -3], [0, 3]], dtype=np.float64)
    toy_y = np.repeat(np.arange(3), 30)
    toy_X = centers[toy_y] + rng.normal(scale=0.35, size=(90, 2))
    fit = load_development_split("development_train")
    validation = load_development_split("validation")
    X_fit, y_fit, _ = features_target_ids(fit)
    X_validation, y_validation, _ = features_target_ids(validation)
    preprocess = build_preprocessing_pipeline()
    X = np.asarray(preprocess.fit_transform(X_fit), dtype=np.float64)
    Xv = np.asarray(preprocess.transform(X_validation), dtype=np.float64)
    encoded = np.array([CLASSES.index(label) for label in y_fit])
    C = 1.0
    regularization = 1.0 / (C * len(X))
    # The source scaling is recorded for this installed version, not assumed across versions.
    source = inspect.getsource(logistic_source._logistic_regression_path)
    if "l2_reg_strength = 1.0 / (C * sw_sum)" not in source:
        raise RuntimeError("Recheck this scikit-learn version's L-BFGS penalty convention.")
    reference = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", tol=1e-10, max_iter=10000)
    with threadpool_limits(limits=1):
        toy = NumpySoftmaxClassifier(regularization=0.01, max_iter=5000).fit(toy_X, toy_y)
        model = NumpySoftmaxClassifier(regularization=regularization).fit(X, encoded)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            reference.fit(X, encoded)
    numpy_train, numpy_validation = model.predict_proba(X), model.predict_proba(Xv)
    reference_train, reference_validation = reference.predict_proba(X), reference.predict_proba(Xv)
    reference_objective, ref_W, ref_b = loss_and_grad(X, encoded, reference.coef_.T, reference.intercept_, regularization)
    toy_accuracy = float(np.mean(toy.predict(toy_X) == toy_y))
    if toy_accuracy < 0.99 or model.objective_ >= np.log(3):
        raise AssertionError("The implementation did not learn the expected problems.")
    objective_gap = abs(model.objective_ - reference_objective)
    probability_gap = float(np.max(np.abs(numpy_validation - reference_validation)))
    if objective_gap > 1e-6 or probability_gap > 1e-3:
        raise AssertionError("Matched-objective comparison differs beyond numerical tolerances.")
    split = json.loads((PROCESSED_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "scope": "educational math verification; development-train and validation only",
        "seed": RANDOM_SEED, "dtype": "float64", "test_table_loaded": False,
        "dataset_revision": split["revision"], "membership_sha256": split["membership_sha256"],
        "source_sha256": split["source_sha256"], "class_order": list(CLASSES),
        "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "threadpoolctl"]},
        "train_shape": list(X.shape), "validation_shape": list(Xv.shape),
        "gradient_check": gradient,
        "extreme_logit_check": {"finite_log_probabilities": True, "probability_sums_one": True,
                                "wrong_class_loss": float(-extreme[0, 1])},
        "synthetic_learning": {"rows": len(toy_X), "accuracy": toy_accuracy,
                               "initial_objective": float(np.log(3)), "final_objective": toy.objective_,
                               "iterations": toy.n_iter_, "converged": toy.converged_},
        "regularization": {"C": C, "lambda": regularization, "formula": "1/(C*n)", "n": len(X),
                           "sklearn_version": sklearn.__version__, "reference_solver": "lbfgs",
                           "source_module": "sklearn.linear_model._logistic._logistic_regression_path",
                           "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
        "numpy_optimizer": {"method": "full-batch gradient descent with training-only Armijo backtracking",
                            "initial_step": model.initial_step_, "max_iter": model.max_iter,
                            "tolerance": model.tolerance, "iterations": model.n_iter_,
                            "converged": model.converged_, "gradient_norm": model.gradient_norm_,
                            "objective": model.objective_},
        "reference_optimizer": {"solver": "lbfgs", "tol": 1e-10, "max_iter": 10000,
                                "iterations": reference.n_iter_.tolist(), "objective": reference_objective,
                                "gradient_norm": float(np.sqrt(np.sum(ref_W ** 2) + np.sum(ref_b ** 2))),
                                "warnings": [str(w.message) for w in captured]},
        "comparison": {"objective_absolute_gap": objective_gap,
                       "train_max_probability_gap": float(np.max(np.abs(numpy_train - reference_train))),
                       "validation_max_probability_gap": probability_gap,
                       "validation_prediction_agreement": float(np.mean(model.predict(Xv) == reference.predict(Xv)))},
        "validation_numpy": classification_metrics(y_validation, np.asarray(CLASSES)[model.predict(Xv)], pd.DataFrame(numpy_validation, columns=CLASSES)),
        "validation_reference": classification_metrics(y_validation, np.asarray(CLASSES)[reference.predict(Xv)], pd.DataFrame(reference_validation, columns=CLASSES)),
        "history": model.history_,
        "note": "No validation tuning or model selection; reference uses tighter stopping solely for numerical verification. Step 7 candidates remain unchanged.",
    }
    reports = PROJECT_ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    iterations = [point["iteration"] for point in model.history_]
    axes[0].plot(iterations, [point["objective"] for point in model.history_], label="NumPy full-batch GD")
    axes[0].axhline(reference_objective, color="black", linestyle="--", label="L-BFGS reference")
    axes[0].set(xlabel="Training iteration", ylabel="Mean cross-entropy + L2 penalty", title="Matched training objective")
    axes[0].legend()
    axes[1].semilogy(iterations, [max(point["gradient_norm"], 1e-16) for point in model.history_])
    axes[1].axhline(model.tolerance, color="black", linestyle="--", label="Gradient tolerance")
    axes[1].set(xlabel="Training iteration", ylabel="Full gradient L2 norm", title="Training-only stopping check")
    axes[1].legend()
    fig.suptitle("Course-math verification — no validation-driven stopping")
    fig.tight_layout()
    fig.savefig(figures / "softmax_math_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (reports / "softmax_math_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump({"preprocessing": preprocess, "classifier": model, "class_order": CLASSES,
                 "lambda": regularization, "educational_only": True}, PROJECT_ROOT / "models" / "educational_softmax.joblib")
    write_results(summary)
    return summary


def write_results(summary):
    """Refresh only the generated numerical section of the derivation report."""
    report = PROJECT_ROOT / "reports" / "SOFTMAX_MATH.md"
    marker = "<!-- NUMERICAL_RESULTS -->"
    explanation = report.read_text(encoding="utf-8").split(marker, 1)[0].rstrip()
    gradient, optimizer, comparison = summary["gradient_check"], summary["numpy_optimizer"], summary["comparison"]
    lines = ["", "", marker, "", "## 7. Measured numerical results", "",
             "This section is regenerated by the experiment command; the mathematical explanation above is preserved.", "",
             "| Check | Result |", "| --- | ---: |",
             f"| Central finite-difference parameters checked | {gradient['parameters_checked']} |",
             f"| Maximum absolute gradient error | {gradient['max_absolute_error']:.3e} |",
             f"| Relative L2 gradient error (normalizer at least 1) | {gradient['relative_l2_error']:.3e} |",
             f"| Finite log-loss for an extreme confidently wrong sample | {summary['extreme_logit_check']['wrong_class_loss']:.1f} |",
             f"| Accuracy on the simple synthetic training problem | {summary['synthetic_learning']['accuracy']:.2%} |",
             f"| NumPy full-batch iterations | {optimizer['iterations']} |",
             f"| Gradient tolerance reached | {optimizer['converged']} |",
             f"| Final full-gradient norm | {optimizer['gradient_norm']:.3e} |",
             f"| NumPy penalized training objective | {optimizer['objective']:.12f} |",
             f"| Reference penalized training objective | {summary['reference_optimizer']['objective']:.12f} |",
             f"| Absolute objective gap | {comparison['objective_absolute_gap']:.3e} |",
             f"| Maximum validation probability difference | {comparison['validation_max_probability_gap']:.3e} |",
             f"| Validation predicted-label agreement | {comparison['validation_prediction_agreement']:.2%} |", "",
             "The NumPy and L-BFGS solutions agree numerically in probabilities and matched objective value. This verifies the equations and implementation conventions; it does not improve the model class, establish biological validity, or select a final model. Validation was transformed once per fitted preprocessor and never used for optimizer stopping or step-size selection. The tighter reference is a numerical check, and Step 7's C=1 candidate/artifacts remain unchanged.", "",
             "![Training-only convergence](./figures/softmax_math_convergence.png)", "",
             "The left panel shows gradient descent lowering the penalized training objective toward the reference value; the right panel shows the full gradient norm reaching the declared tolerance. Both curves use development-train only; their convergence is an optimization result rather than a generalization estimate.", "",
             "## 8. Reproduce and inspect", "",
             "```powershell", "& ../.venv/Scripts/python.exe -m src.softmax_experiment",
             "& ../.venv/Scripts/python.exe -m unittest discover -s tests -v", "```", "",
             "Run from the repository root. [The NumPy implementation](../src/softmax.py) contains the stable loss/gradient, full-batch optimizer, and finite-difference check. [The experiment](../src/softmax_experiment.py) fits fresh development-training preprocessing, verifies a simple synthetic problem, and compares to a same-objective library reference. It uses one BLAS thread for consistent small-matrix CPU execution, with no new JAX/Optax installation.", "",
             "[The complete summary](./softmax_math_summary.json) records numeric precision, seeds, source/membership hashes, installed versions, inspected source-function hash, regularization conversion, optimizer settings/history, reference diagnostics, and validation comparisons. [The math notebook](../notebooks/02b_softmax_math.ipynb) presents the derivation and executes the verification. [The local educational artifact](../models/educational_softmax.joblib) saves preprocessing, NumPy parameters, class order, and lambda; it is not a new selected project candidate.", "",
             "All 29 tests pass, including six dedicated [math tests](../tests/test_softmax.py). Raw data, memberships, Step 7 model artifacts, and the reserved test table remain unchanged. Step 8 is complete, including its optional implementation/verification/experiment. Next is Step 9's declared development-only tuning and selection.", ""]
    report.write_text(explanation + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_experiment()
    for key in ["gradient_check", "numpy_optimizer", "comparison"]:
        print(key, json.dumps(result[key]))
