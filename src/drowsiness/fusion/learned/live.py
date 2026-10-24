"""Runs a saved learned model on the live pipeline's output, for comparison only."""

from __future__ import annotations

import math

from drowsiness.fusion.learned.features import FeatureTracker
from drowsiness.fusion.learned.model import LogisticModel


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
