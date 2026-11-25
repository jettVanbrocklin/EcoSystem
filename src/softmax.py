"""Educational float64 softmax and full-batch gradient descent in NumPy."""

import numpy as np


def log_softmax(logits):
    """Stable log probabilities, avoiding exp of large positive logits."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("Logits must be a finite two-dimensional matrix.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def loss_and_grad(X, y, W, b, regularization):
    """Mean cross-entropy + lambda/2 ||W||^2; intercept is unpenalized."""
    log_probabilities = log_softmax(X @ W + b)
    count = len(X)
    loss = -log_probabilities[np.arange(count), y].mean() + 0.5 * regularization * np.sum(W * W)
    residual = np.exp(log_probabilities)
    residual[np.arange(count), y] -= 1
    residual /= count
    return float(loss), X.T @ residual + regularization * W, residual.sum(axis=0)


class NumpySoftmaxClassifier:
    """Three-class optimizer accepting transformed arrays and integer labels.

    Step sizes use a design-based bound followed by training-only Armijo
    backtracking. No validation data or validation stopping rule is accepted.
    """

    def __init__(self, regularization=0.01, max_iter=20000, tolerance=1e-8):
        self.regularization = regularization
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("Training predictors must be a nonempty finite matrix.")
        if y.shape != (len(X),) or y.dtype.kind not in "iu" or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("Labels must be integer class indices 0, 1, or 2.")
        if not np.isfinite(self.regularization) or self.regularization < 0 or self.max_iter < 1 or self.tolerance <= 0:
            raise ValueError("Invalid regularization, iteration budget, or tolerance.")
        self.W_ = np.zeros((X.shape[1], 3), dtype=np.float64)
        self.b_ = np.zeros(3, dtype=np.float64)
        augmented = np.column_stack([X, np.ones(len(X))])
        # Multiclass logit Hessian has spectral norm <= 1/2.
        smoothness_bound = 0.5 * np.linalg.norm(augmented, ord=2) ** 2 / len(X) + self.regularization
        step = 1.0 / smoothness_bound
        self.initial_step_ = step
        loss, grad_W, grad_b = loss_and_grad(X, y, self.W_, self.b_, self.regularization)
        self.history_ = []
        self.converged_ = False
        for iteration in range(self.max_iter + 1):
            norm_squared = float(np.sum(grad_W ** 2) + np.sum(grad_b ** 2))
            if iteration % 10 == 0 or norm_squared ** 0.5 <= self.tolerance or iteration == self.max_iter:
                self.history_.append({"iteration": iteration, "objective": loss, "gradient_norm": norm_squared ** 0.5})
            if norm_squared ** 0.5 <= self.tolerance:
                self.converged_ = True
                break
            if iteration == self.max_iter:
                break
            for _ in range(60):
                candidate_W = self.W_ - step * grad_W
                candidate_b = self.b_ - step * grad_b
                candidate_b -= candidate_b.mean()  # Fix the unpenalized common-intercept gauge.
                candidate_loss, next_W, next_b = loss_and_grad(X, y, candidate_W, candidate_b, self.regularization)
                if candidate_loss <= loss - 1e-4 * step * norm_squared:
                    break
                step *= 0.5
            else:
                raise RuntimeError("Armijo backtracking did not find a descent step.")
            self.W_, self.b_ = candidate_W, candidate_b
            loss, grad_W, grad_b = candidate_loss, next_W, next_b
            step = min(step * 1.5, 100.0)
        self.n_iter_ = iteration
        self.objective_ = loss
        self.gradient_norm_ = norm_squared ** 0.5
        return self

    def predict_proba(self, X):
        if not hasattr(self, "W_"):
            raise ValueError("Fit the NumPy classifier before prediction.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.W_.shape[0] or not np.isfinite(X).all():
            raise ValueError("Prediction matrix must be finite with the fitted feature width.")
        return np.exp(log_softmax(X @ self.W_ + self.b_))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def check_gradient(seed=42, step=1e-5):
    """Check all 15 parameters on a tiny seeded float64 example."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(5, 4)).astype(np.float64)
    y = np.array([0, 1, 2, 0, 2])
    W, b = rng.normal(scale=0.2, size=(4, 3)), rng.normal(scale=0.1, size=3)
    regularization = 0.13
    _, grad_W, grad_b = loss_and_grad(X, y, W, b, regularization)
    parameters = np.concatenate([W.ravel(), b])
    analytic = np.concatenate([grad_W.ravel(), grad_b])
    numerical = np.zeros_like(parameters)

    def objective(values):
        return loss_and_grad(X, y, values[:12].reshape(4, 3), values[12:], regularization)[0]

    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (objective(plus) - objective(minus)) / (2 * step)
    return {
        "seed": seed, "dtype": str(X.dtype), "step": step, "parameters_checked": len(parameters),
        "max_absolute_error": float(np.max(np.abs(analytic - numerical))),
        "relative_l2_error": float(np.linalg.norm(analytic - numerical) / max(1.0, np.linalg.norm(analytic), np.linalg.norm(numerical))),
    }
