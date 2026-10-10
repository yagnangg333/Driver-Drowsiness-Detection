"""End-to-end per-frame processing: landmarks -> signals -> indicators -> fusion -> alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drowsiness.alerts import AlertEvent, AlertLevel, AlertManager
from drowsiness.calibration import Baseline, Calibrator
from drowsiness.config import AppConfig
from drowsiness.fusion import FusionResult, FusionStrategy, IndicatorScores, StaticFusion
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


@dataclass(frozen=True)
class FrameReport:
    timestamp: float
    face: FaceObservation | None
    signals: FaceSignals | None
    calibrating: bool
    calibration_progress: float
    calibration_note: str | None
    baseline: Baseline | None
    eye: EyeState
    yawn: YawnState
    head: HeadState
    fusion: FusionResult
    alert_level: AlertLevel
    event: AlertEvent | None

    @property
    def face_detected(self) -> bool:
        return self.face is not None

    @property
    def scores(self) -> IndicatorScores:
        return IndicatorScores(self.eye.score, self.yawn.score, self.head.score)

    def to_row(self) -> dict[str, Any]:
        """Flat per-frame record, used for recordings and offline evaluation."""
        s, pose = self.signals, self.signals.pose if self.signals else None
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
            "fused_score": self.fusion.score,
            "alert_level": int(self.alert_level),
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
        self.fusion = fusion or StaticFusion(config.fusion)

        ind = config.indicators
        self._eye = EyeClosureIndicator(
            ind.perclos_window_s, ind.perclos_max, ind.microsleep_s, ind.closure_debounce_s
        )
        self._yawn = YawnIndicator(
            ind.yawn_mar_threshold, ind.yawn_min_duration_s, ind.yawn_window_s, ind.yawns_for_max
        )
        self._head = HeadTiltIndicator(
            ind.head_pitch_threshold_deg, ind.head_roll_threshold_deg, ind.head_tilt_max_s
        )
        self.alerts = AlertManager(config.alerts)
        self.recalibrate()

    @property
    def baseline(self) -> Baseline | None:
        return self._calibrator.result

    def recalibrate(self) -> None:
        """Restart calibration and clear all temporal state."""
        self._calibrator = Calibrator(
            self.config.calibration, self.config.indicators.default_ear_threshold
        )
        for indicator in (self._eye, self._yawn, self._head):
            indicator.reset()
        self.alerts.reset()

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FrameReport:
        face = self._landmarker.detect(frame_bgr, timestamp)
        signals = compute_signals(face) if face is not None else None

        calibrating = not self._calibrator.done
        if calibrating:
            self._calibrator.update(timestamp, signals)
        elif signals is not None:
            self._update_indicators(timestamp, signals, self._calibrator.result)

        fusion = self.fusion.fuse(
            IndicatorScores(self._eye.state.score, self._yawn.state.score, self._head.state.score)
        )
        event = None
        if not calibrating:
            event = self.alerts.update(
                timestamp, fusion.score, face is not None, self._eye.state.microsleep
            )

        return FrameReport(
            timestamp=timestamp,
            face=face,
            signals=signals,
            calibrating=calibrating,
            calibration_progress=self._calibrator.progress,
            calibration_note=self._calibrator.note,
            baseline=self._calibrator.result,
            eye=self._eye.state,
            yawn=self._yawn.state,
            head=self._head.state,
            fusion=fusion,
            alert_level=self.alerts.level,
            event=event,
        )

    def _update_indicators(self, t: float, signals: FaceSignals, baseline: Baseline) -> None:
        self._eye.update(t, signals.ear < baseline.ear_threshold)
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
