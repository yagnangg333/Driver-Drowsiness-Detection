"""The learned model: L2-regularised logistic regression.

Trained by Newton's method (iteratively reweighted least squares) in NumPy, so
it needs no extra library; a trained model is a short JSON file of numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drowsiness.fusion.learned.features import FEATURES


@dataclass
class LogisticModel:
    """p(drowsy) = sigmoid(intercept + coef . (x - mean) / std)."""

    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    # Probability at which the model calls a frame drowsy (chosen on the
    # training data); 0.5 unless set after training.
    threshold: float = 0.5
    # What the model was trained on, for display; free text.
    trained_on: str = ""

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.intercept + ((np.asarray(x, float) - self.mean) / self.std) @ self.coef
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        l2: float = 1.0,
        weights: np.ndarray | None = None,
        features: tuple[str, ...] = FEATURES,
        iterations: int = 50,
    ) -> LogisticModel:
        """L2-regularised logistic regression by Newton's method (IRLS)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-9] = 1.0
        a = np.hstack([np.ones((len(x), 1)), (x - mean) / std])
        beta = np.zeros(a.shape[1])
        penalty = np.full(a.shape[1], l2)
        penalty[0] = 0.0  # the intercept is not regularised
        for _ in range(iterations):
            p = 1.0 / (1.0 + np.exp(-np.clip(a @ beta, -30, 30)))
            grad = a.T @ (w * (p - y)) + penalty * beta
            hess = (a * (w * p * (1 - p))[:, None]).T @ a + np.diag(penalty)
            step = np.linalg.solve(hess + 1e-9 * np.eye(len(beta)), grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        return cls(tuple(features), mean, std, beta[1:], float(beta[0]))

    def to_json(self, path: str | Path) -> None:
        data = {
            "features": list(self.features),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "trained_on": self.trained_on,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> LogisticModel:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["features"]) != FEATURES:
            raise ValueError(f"model {path} was trained on different features")
        return cls(
            tuple(data["features"]),
            np.array(data["mean"]),
            np.array(data["std"]),
            np.array(data["coef"]),
            float(data["intercept"]),
            float(data.get("threshold", 0.5)),
            str(data.get("trained_on", "")),
        )
