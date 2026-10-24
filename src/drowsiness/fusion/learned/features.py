"""The learned model's inputs: 13 features from the last 10-30 s of signals.

Computed causally (only from the past) by ``FeatureTracker``, the same code for
training (from recorded signals) and for live use (from the pipeline).
"""

from __future__ import annotations

import math
from collections import deque

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
