"""Classification metrics with an explicit, consistent class order."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from src.schema import CLASSES


def classification_metrics(y_true, y_pred, probabilities):
    """Evaluate probabilities whose columns follow the project CLASSES order.

    Encode targets numerically for log_loss so sklearn's alphabetical ordering
    cannot silently exchange moderate_stress and high_stress columns.
    """
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or predicted.shape != truth.shape or len(truth) == 0:
        raise ValueError("Metrics require matching nonempty one-dimensional labels.")
    if not np.isin(truth, CLASSES).all() or not np.isin(predicted, CLASSES).all():
        raise ValueError("Unexpected true or predicted class labels.")
    if hasattr(probabilities, "columns") and tuple(probabilities.columns) != CLASSES:
        raise ValueError("Probability columns must follow the explicit project class order.")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(truth), len(CLASSES)):
        raise ValueError("Unexpected probability matrix shape.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid class probabilities.")
    encoded = np.asarray([CLASSES.index(label) for label in truth])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(CLASSES), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "log_loss": float(log_loss(encoded, probabilities, labels=list(range(len(CLASSES))))),
        "log_loss_clip_epsilon": float(np.finfo(probabilities.dtype).eps),
        "zero_probability_true_labels": int((probabilities[np.arange(len(truth)), encoded] == 0).sum()),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(CLASSES)).tolist(),
        "class_order": list(CLASSES),
    }
