"""End-to-end per-frame processing.

landmarks -> signals -> conditions + indicators -> fusion -> alerts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drowsiness.alerts import AlertEvent, AlertLevel, AlertManager
from drowsiness.calibration import Baseline, Calibrator
from drowsiness.config import AppConfig
from drowsiness.fusion import (
    AdaptiveFusion,
    FusionResult,
    FusionStrategy,
    IndicatorScores,
    StaticFusion,
)
from drowsiness.fusion.adaptive.conditions import ConditionDetector, Conditions, measure
from drowsiness.indicators import (
    EyeClosureIndicator,
    EyeState,
    HeadState,
    HeadTiltIndicator,
    YawnIndicator,
    YawnState,
)
from drowsiness.landmarks import FaceLandmarker, FaceObservation
from drowsiness.signals import FaceSignals, compute_signals

# Blendshapes recorded per frame for evaluating expressions (smiling, talking,
# laughing) that narrow the eyes without drowsiness.
_EXPRESSION_SHAPES = (
    "eyeBlink",
    "eyeSquint",
    "cheekSquint",
    "mouthSmile",
    "jawOpen",
    "mouthStretch",
    "noseSneer",
    "browDown",
)


@dataclass(frozen=True)
class FrameReport:
    timestamp: float
    face: FaceObservation | None
    signals: FaceSignals | None
    calibrating: bool
    calibration_progress: float
    calibration_note: str | None
    baseline: Baseline | None
    eye: EyeState  # as seen by the strategy that drives the alerts
    yawn: YawnState
    head: HeadState
    conditions: Conditions
    fusion: FusionResult  # from the strategy that drives the alerts
    fusion_name: str
    static_score: float  # both strategies on the same scores, for comparison
    adaptive_score: float
    alert_level: AlertLevel
    event: AlertEvent | None
    # The eye cue of each strategy: static counts every closure, adaptive
    # ignores closures while the driver smiles.
    eye_static: EyeState
    eye_adaptive: EyeState
    # The alert level each strategy would raise right now; ``alert_level`` is
    # the one of the strategy that drives the alarm.
    static_level: AlertLevel = AlertLevel.NONE
    adaptive_level: AlertLevel = AlertLevel.NONE

    @property
    def face_detected(self) -> bool:
        return self.face is not None

    @property
    def scores(self) -> IndicatorScores:
        return IndicatorScores(self.eye.score, self.yawn.score, self.head.score)

    def _expression_columns(self) -> dict[str, float]:
        """MediaPipe expression scores (0-1) that can narrow or hide the eyes,
        averaged over left and right; NaN without a face."""
        shapes = self.face.blendshapes if self.face else {}
        out = {}
        for name in _EXPRESSION_SHAPES:
            sides = [shapes[k] for k in (name + "Left", name + "Right", name) if k in shapes]
            out[f"bs_{name}"] = sum(sides) / len(sides) if sides else float("nan")
        return out

    def to_row(self) -> dict[str, Any]:
        """Flat per-frame record, used for recordings and offline evaluation."""
        s, pose = self.signals, self.signals.pose if self.signals else None
        c, w = self.conditions, self.fusion.weights
        nan = float("nan")
        return {
            "timestamp_s": round(self.timestamp, 4),
            "face_detected": int(self.face_detected),
            "calibrating": int(self.calibrating),
            "ear": s.ear if s else nan,
            "mar": s.mar if s else nan,
            "pitch": pose.pitch if pose else nan,
            "yaw": pose.yaw if pose else nan,
            "roll": pose.roll if pose else nan,
            "eyes_closed": int(self.eye.closed),
            "microsleep": int(self.eye.microsleep),
            "perclos": self.eye.perclos,
            "yawning": int(self.yawn.yawning),
            "head_tilted": int(self.head.tilted),
            "eye_score": self.eye.score,
            "yawn_score": self.yawn.score,
            "head_score": self.head.score,
            "frame_brightness": c.brightness if c.brightness is not None else nan,
            "low_light": int(c.low_light),
            "eye_occluded": int(c.eye_occluded),
            "occlusion": c.occlusion,
            "eye_weight": w.eye,
            "yawn_weight": w.yawn,
            "head_weight": w.head,
            "static_score": self.static_score,
            "adaptive_score": self.adaptive_score,
            "smile": c.smile,
            "smiling": int(c.smiling),
            "eye_score_static": self.eye_static.score,
            "eye_score_adaptive": self.eye_adaptive.score,
            "microsleep_static": int(self.eye_static.microsleep),
            "microsleep_adaptive": int(self.eye_adaptive.microsleep),
            **self._expression_columns(),
            "fusion": self.fusion_name,
            "fused_score": self.fusion.score,
            "alert_level": int(self.alert_level),
            "static_level": int(self.static_level),
            "adaptive_level": int(self.adaptive_level),
        }


class DrowsinessPipeline:
    """Processes frames in chronological order and tracks the driver's state."""

    def __init__(
        self,
        config: AppConfig,
        landmarker: FaceLandmarker | None = None,
        fusion: FusionStrategy | None = None,
    ):
        self.config = config
        self._landmarker = landmarker or FaceLandmarker(config.landmarker)
        self._owns_landmarker = landmarker is None
        self._static = StaticFusion(config.fusion)
        self._adaptive = AdaptiveFusion(config.fusion)
        default = self._adaptive if config.fusion.strategy == "adaptive" else self._static
        self.fusion: FusionStrategy = fusion or default
        self._conditions = ConditionDetector(
            config.conditions, config.indicators.yawn_mar_threshold
        )

        ind = config.indicators
        eye_args = (ind.perclos_window_s, ind.perclos_max, ind.microsleep_s, ind.closure_debounce_s)
        self._eye_static = EyeClosureIndicator(*eye_args)  # counts every closure
        self._eye_adaptive = EyeClosureIndicator(*eye_args)  # not while smiling
        self._yawn = YawnIndicator(
            ind.yawn_mar_threshold,
            ind.yawn_min_duration_s,
            ind.yawn_window_s,
            ind.yawns_for_max,
            ind.yawn_debounce_s,
        )
        self._head = HeadTiltIndicator(
            ind.head_pitch_threshold_deg,
            ind.head_roll_threshold_deg,
            ind.head_tilt_max_s,
            ind.head_debounce_s,
        )
        # One alert state machine per strategy, so the dashboard can show what
        # the static baseline would do next to the adaptive strategy. Only the
        # active one (``alerts``) sounds the alarm and is logged.
        self._static_alerts = AlertManager(config.alerts)
        self._adaptive_alerts = AlertManager(config.alerts)
        self.alerts = self._static_alerts if self.fusion is self._static else self._adaptive_alerts
        self.recalibrate()

    @property
    def baseline(self) -> Baseline | None:
        return self._calibrator.result

    def recalibrate(self) -> None:
        """Restart calibration and clear all temporal state."""
        self._calibrator = Calibrator(
            self.config.calibration, self.config.indicators.default_ear_threshold
        )
        self.reset_tracking()

    def reset_tracking(self) -> None:
        """Clear all temporal state but keep the calibration.

        For a new recording of the same driver: nothing from the previous one
        (an eye closure, a yawn count, an alert) carries over.
        """
        for indicator in self._indicators:
            indicator.reset()
        for fusion in (self._static, self._adaptive, self.fusion):
            fusion.reset()
        self._conditions.reset()
        self._static_alerts.reset()
        self._adaptive_alerts.reset()
        self._face_missing_since: float | None = None

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FrameReport:
        face = self._landmarker.detect(frame_bgr, timestamp)
        signals = compute_signals(face) if face is not None else None
        measured = (
            measure(frame_bgr, face, signals, self.config.conditions)
            if face is not None and signals is not None
            else None
        )
        baseline = self._calibrator.result
        conditions = self._conditions.update(
            timestamp, measured, baseline.smile if baseline else None
        )

        calibrating = not self._calibrator.done
        if calibrating:
            self._calibrator.update(timestamp, signals, measured.smile if measured else None)
        elif signals is not None:
            self._face_missing_since = None
            self._update_indicators(timestamp, signals, self._calibrator.result, conditions)
        else:
            self._handle_missing_face(timestamp)

        yawn, head = self._yawn.state.score, self._head.state.score
        static = self._static.fuse(
            IndicatorScores(self._eye_static.state.score, yawn, head), conditions, timestamp
        )
        adaptive_scores = IndicatorScores(self._eye_adaptive.state.score, yawn, head)
        adaptive = self._adaptive.fuse(adaptive_scores, conditions, timestamp)
        if self.fusion is self._static:
            fusion, eye = static, self._eye_static.state
        else:
            eye = self._eye_adaptive.state
            fusion = (
                adaptive
                if self.fusion is self._adaptive
                else self.fusion.fuse(adaptive_scores, conditions, timestamp)
            )

        # Optionally, a microsleep behind an occlusion does not force an alarm on
        # its own (off by default: see ConditionConfig).
        gate = conditions.eye_occluded and self.config.conditions.microsleep_needs_visible_eyes
        event = None
        if not calibrating:
            face_seen = face is not None
            static_event = self._static_alerts.update(
                timestamp, static.score, face_seen, self._eye_static.state.microsleep
            )
            adaptive_event = self._adaptive_alerts.update(
                timestamp,
                fusion.score if self.fusion is not self._static else adaptive.score,
                face_seen,
                self._eye_adaptive.state.microsleep and not gate,
            )
            event = static_event if self.alerts is self._static_alerts else adaptive_event

        return FrameReport(
            timestamp=timestamp,
            face=face,
            signals=signals,
            calibrating=calibrating,
            calibration_progress=self._calibrator.progress,
            calibration_note=self._calibrator.note,
            baseline=self._calibrator.result,
            eye=eye,
            yawn=self._yawn.state,
            head=self._head.state,
            conditions=conditions,
            fusion=fusion,
            fusion_name=self.fusion.name,
            static_score=static.score,
            adaptive_score=adaptive.score,
            alert_level=self.alerts.level,
            event=event,
            eye_static=self._eye_static.state,
            eye_adaptive=self._eye_adaptive.state,
            static_level=self._static_alerts.level,
            adaptive_level=self._adaptive_alerts.level,
        )

    @property
    def _indicators(self) -> tuple:
        return (self._eye_static, self._eye_adaptive, self._yawn, self._head)

    def _handle_missing_face(self, t: float) -> None:
        # A short dropout (one or two lost frames) keeps every timer running, so
        # a real microsleep is not reset by a flicker. A longer loss clears the
        # continuous timers: nothing unobserved may be counted as eyes closed,
        # which would otherwise fire a phantom microsleep when the face returns.
        # Alerting during the loss itself is handled by the alert manager.
        if self._face_missing_since is None:
            self._face_missing_since = t
        if t - self._face_missing_since >= self.config.indicators.face_loss_grace_s:
            for indicator in self._indicators:
                indicator.face_lost(t)

    def _update_indicators(
        self, t: float, signals: FaceSignals, baseline: Baseline, conditions: Conditions
    ) -> None:
        closed = signals.ear < baseline.ear_threshold
        self._eye_static.update(t, closed)
        # A smile narrows the eyes without drowsiness: not an eye closure.
        self._eye_adaptive.update(t, closed and not conditions.smiling)
        self._yawn.update(t, signals.mar)
        if signals.pose is not None:
            self._head.update(
                t, signals.pose.pitch - baseline.pitch, signals.pose.roll - baseline.roll
            )

    def close(self) -> None:
        if self._owns_landmarker:
            self._landmarker.close()

    def __enter__(self) -> DrowsinessPipeline:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
