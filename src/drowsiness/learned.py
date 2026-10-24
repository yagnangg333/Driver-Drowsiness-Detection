"""Learned fusion: a logistic regression over windowed drowsiness features.

Instead of hand-set fusion weights, a small, explainable model is trained on
labelled recordings. The features are computed causally (only from the past)
by ``FeatureTracker``, the same code for training and for live use, from
signals the pipeline already produces. Safety rules (a microsleep or a lost
face forcing an alarm) stay in the alert manager, whatever the model says.

The model needs no extra library: training uses iteratively reweighted least
squares in NumPy, and a trained model is a short JSON file of numbers.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "closed_frac_10s",  # share of the last 10 s with eyes closed (smiles excluded)
    "closed_frac_30s",
    "closure_s",  # length of the current closure, capped at 3 s
    "long_closures_30s",  # closures of at least 0.5 s in the last 30 s
    "blinks_30s",  # closures started in the last 30 s
    "ear_rel_10s",  # mean EAR relative to the driver's open-eye EAR
    "yawn_score",  # yawn indicator (yawns in the last 60 s)
    "mar_10s",  # mean mouth opening
    "head_score",  # head-tilt indicator
    "pitch_std_10s",  # nodding: spread of head pitch, degrees
    "smile_10s",  # mean smile score
    "low_light",
    "eye_occluded",
)

_CLOSURE_CAP_S = 3.0
_LONG_CLOSURE_S = 0.5


class _Window:
    """Values from the last ``span`` seconds, with their mean and spread."""

    def __init__(self, span: float):
        self.span = span
        self.items: deque[tuple[float, float]] = deque()

    def add(self, t: float, value: float) -> None:
        self.items.append((t, value))
        while self.items and t - self.items[0][0] > self.span:
            self.items.popleft()

    def mean(self, default: float = 0.0) -> float:
        return sum(v for _, v in self.items) / len(self.items) if self.items else default

    def std(self) -> float:
        if len(self.items) < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((v - m) ** 2 for _, v in self.items) / len(self.items))


class FeatureTracker:
    """Per-frame feature vector from the past few seconds of signals."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._closed10, self._closed30 = _Window(10.0), _Window(30.0)
        self._ear, self._mar, self._pitch, self._smile = (_Window(10.0) for _ in range(4))
        self._onsets: deque[float] = deque()
        self._long: deque[float] = deque()
        self._closed_since: float | None = None
        self._was_closed = False
        self._counted_long = False

    def update(
        self,
        t: float,
        *,
        face: bool,
        closed: bool,
        ear_rel: float,
        mar: float,
        yawn_score: float,
        head_score: float,
        pitch_rel: float,
        smile: float,
        low_light: bool,
        eye_occluded: bool,
    ) -> np.ndarray:
        if face:
            self._closed10.add(t, float(closed))
            self._closed30.add(t, float(closed))
            for window, value in (
                (self._ear, ear_rel),
                (self._mar, mar),
                (self._pitch, pitch_rel),
                (self._smile, smile),
            ):
                if not math.isnan(value):
                    window.add(t, value)
            if closed and not self._was_closed:
                self._closed_since, self._counted_long = t, False
                self._onsets.append(t)
            if not closed:
                self._closed_since = None
            if (
                self._closed_since is not None
                and not self._counted_long
                and t - self._closed_since >= _LONG_CLOSURE_S
            ):
                self._long.append(t)
                self._counted_long = True
            self._was_closed = closed
        else:
            self._closed_since, self._was_closed = None, False
        for times in (self._onsets, self._long):
            while times and t - times[0] > 30.0:
                times.popleft()
        closure = 0.0 if self._closed_since is None else min(t - self._closed_since, _CLOSURE_CAP_S)
        return np.array(
            [
                self._closed10.mean(),
                self._closed30.mean(),
                closure,
                len(self._long),
                len(self._onsets),
                self._ear.mean(1.0),
                yawn_score,
                self._mar.mean(),
                head_score,
                self._pitch.std(),
                self._smile.mean(),
                float(low_light),
                float(eye_occluded),
            ]
        )


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


# Short names for the features, for showing the model's weights.
FEATURE_LABELS = {
    "closed_frac_10s": "eyes closed, last 10 s",
    "closed_frac_30s": "eyes closed, last 30 s",
    "closure_s": "current closure length",
    "long_closures_30s": "long closures, last 30 s",
    "blinks_30s": "blinks, last 30 s",
    "ear_rel_10s": "eye opening vs calibrated",
    "yawn_score": "yawn score",
    "mar_10s": "mouth opening",
    "head_score": "head-tilt score",
    "pitch_std_10s": "nodding (pitch spread)",
    "smile_10s": "smile",
    "low_light": "low light",
    "eye_occluded": "eyes occluded",
}


class LiveLearnedFusion:
    """Runs a saved model on the live pipeline's output, for comparison only.

    The features are built exactly as for training (``tools/learned_fusion.py``):
    eye closures as seen by the adaptive eye indicator, EAR relative to the
    calibrated open-eye EAR, head pitch relative to the neutral pose and the
    raw MediaPipe smile score. Returns None until calibration has finished.
    """

    def __init__(self, model: LogisticModel, ear_closed_ratio: float):
        self.model = model
        self._ear_closed_ratio = ear_closed_ratio
        self._tracker = FeatureTracker()

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, report) -> float | None:
        b = report.baseline
        if report.calibrating or b is None:
            return None
        s = report.signals if report.face_detected else None
        pose = s.pose if s else None
        open_ear = b.ear_threshold / self._ear_closed_ratio
        shapes = report.face.blendshapes if report.face_detected else {}
        smile = (shapes.get("mouthSmileLeft", 0.0) + shapes.get("mouthSmileRight", 0.0)) / 2
        x = self._tracker.update(
            report.timestamp,
            face=s is not None,
            closed=report.eye_adaptive.closed,
            ear_rel=s.ear / open_ear if s else math.nan,
            mar=s.mar if s else math.nan,
            yawn_score=report.yawn.score,
            head_score=report.head.score,
            pitch_rel=pose.pitch - b.pitch if pose else math.nan,
            smile=smile,
            low_light=report.conditions.low_light,
            eye_occluded=report.conditions.eye_occluded,
        )
        return float(self.model.predict(x[None, :])[0])
