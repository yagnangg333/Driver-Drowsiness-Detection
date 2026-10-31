"""Per-driver calibration of the open-eye EAR and neutral head pose.

For the first few seconds the driver looks at the road with eyes open. The
median signals over that period become the baseline, so thresholds adapt to
each driver's eye shape, glasses and camera placement.

A baseline that is not plausible for open eyes looking ahead (for example the
driver was already drowsy, or looking down at the phone) is rejected and
calibration restarts, so a bad start cannot silently weaken detection.

Drooping eyes that are still plausible pass that check, though. So after
calibration ``OpenEyeTracker`` raises the open-eye EAR whenever the eyes are
seen open wider for a while; it never lowers it.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from collections import deque
from dataclasses import dataclass

import numpy as np

from drowsiness.config import CalibrationConfig
from drowsiness.signals import FaceSignals

log = logging.getLogger(__name__)

# Give up and fall back to defaults if a usable face is not seen for this long.
_TIMEOUT_FACTOR = 3.0
# Fewest face frames in the recent window before the open-eye EAR may rise.
_MIN_RISE_SAMPLES = 20


@dataclass(frozen=True)
class Baseline:
    ear_threshold: float
    ear: float | None = None
    mar: float | None = None
    pitch: float = 0.0
    roll: float = 0.0
    calibrated: bool = False
    # MediaPipe smile score of this driver's resting face: for some faces and
    # camera angles it is well above zero, so smiling is judged relative to it.
    smile: float | None = None


class Calibrator:
    def __init__(self, config: CalibrationConfig, default_ear_threshold: float):
        self._config = config
        self._default = Baseline(ear_threshold=default_ear_threshold)
        self.attempts = 0
        self.note: str | None = None  # why the last attempt was rejected
        self._restart(None)
        self.result: Baseline | None = None if config.enabled else self._default

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def progress(self) -> float:
        if self.done:
            return 1.0
        by_time = self._elapsed / self._config.duration_s
        by_samples = len(self._ears) / self._config.min_samples
        return min(by_time, by_samples, 1.0)

    def update(
        self, timestamp: float, signals: FaceSignals | None, smile: float | None = None
    ) -> Baseline | None:
        """Feed one frame; returns the baseline once calibration finishes."""
        if self.done:
            return self.result
        if self._start is None:
            self._start = timestamp
        self._now = timestamp

        if signals is not None:
            self._ears.append(signals.ear)
            self._mars.append(signals.mar)
            if signals.pose is not None:
                self._pitches.append(signals.pose.pitch)
                self._rolls.append(signals.pose.roll)
            if smile is not None:
                self._smiles.append(smile)

        enough_time = self._elapsed >= self._config.duration_s
        if enough_time and len(self._ears) >= self._config.min_samples:
            self._finish_attempt(timestamp)
        elif self._elapsed >= self._config.duration_s * _TIMEOUT_FACTOR:
            log.warning(
                "Calibration timed out with %d face samples; using default thresholds",
                len(self._ears),
            )
            self.note = "no face seen during calibration; using default thresholds"
            self.result = self._default
        return self.result

    @property
    def _elapsed(self) -> float:
        return 0.0 if self._start is None else self._now - self._start

    def _restart(self, timestamp: float | None) -> None:
        self._start = timestamp
        self._now = timestamp or 0.0
        self._ears: list[float] = []
        self._mars: list[float] = []
        self._pitches: list[float] = []
        self._rolls: list[float] = []
        self._smiles: list[float] = []

    def _finish_attempt(self, timestamp: float) -> None:
        candidate = self._compute()
        problem = self._implausible(candidate)
        self.attempts += 1
        if problem is None:
            self.note = None
            self.result = candidate
            log.info(
                "Calibrated: open-eye EAR %.3f -> threshold %.3f, neutral pitch %.1f, roll %.1f",
                candidate.ear,
                candidate.ear_threshold,
                candidate.pitch,
                candidate.roll,
            )
        elif self.attempts >= self._config.max_attempts:
            self.note = f"calibration failed ({problem}); using default thresholds"
            log.warning(
                "Calibration rejected %d times (%s); using defaults", self.attempts, problem
            )
            self.result = self._default
        else:
            self.note = f"{problem}; please look ahead with eyes open"
            log.warning("Calibration attempt %d rejected: %s", self.attempts, problem)
            self._restart(timestamp)

    def _implausible(self, baseline: Baseline) -> str | None:
        cfg = self._config
        assert baseline.ear is not None
        if baseline.ear < cfg.min_open_ear:
            return f"eyes look closed (EAR {baseline.ear:.2f} < {cfg.min_open_ear:.2f})"
        if baseline.ear > cfg.max_open_ear:
            return f"unusual eye shape reading (EAR {baseline.ear:.2f} > {cfg.max_open_ear:.2f})"
        if abs(baseline.pitch) > cfg.max_neutral_pitch_deg:
            return f"head not facing ahead (pitch {baseline.pitch:+.0f} deg)"
        return None

    def _compute(self) -> Baseline:
        cfg = self._config
        ear = statistics.median(self._ears)
        return Baseline(
            ear_threshold=ear_threshold(ear, cfg),
            ear=ear,
            mar=statistics.median(self._mars),
            pitch=statistics.median(self._pitches) if self._pitches else 0.0,
            roll=statistics.median(self._rolls) if self._rolls else 0.0,
            calibrated=True,
            smile=statistics.median(self._smiles) if self._smiles else None,
        )


def ear_threshold(open_ear: float, config: CalibrationConfig) -> float:
    """The closed-eye EAR threshold for an open-eye EAR."""
    return min(
        max(open_ear * config.ear_closed_ratio, config.min_ear_threshold),
        config.max_ear_threshold,
    )


class OpenEyeTracker:
    """Raises the open-eye EAR after calibration; never lowers it.

    Keeps the EAR of the last ``rise_window_s`` of face frames. Once a full
    window has been seen, its ``rise_percentile`` becomes the open-eye EAR
    whenever it is higher than the current one and plausible for open eyes
    (the calibration limits). If calibration fell back to the defaults, the
    first plausible value replaces the default threshold.

    ``reset`` (a new recording of the same driver) goes back to the
    calibrated baseline.
    """

    def __init__(self, config: CalibrationConfig, baseline: Baseline):
        self._config = config
        self._calibrated = baseline
        self.reset()

    def reset(self) -> None:
        self.baseline = self._calibrated
        self._ears: deque[tuple[float, float]] = deque()
        self._start: float | None = None

    def update(self, t: float, ear: float) -> Baseline:
        """Feed the EAR of a frame with a face; returns the baseline to use for it."""
        cfg = self._config
        if not cfg.open_ear_rises:
            return self.baseline
        if self._start is None:
            self._start = t
        self._ears.append((t, ear))
        while t - self._ears[0][0] > cfg.rise_window_s:
            self._ears.popleft()
        if t - self._start < cfg.rise_window_s or len(self._ears) < _MIN_RISE_SAMPLES:
            return self.baseline
        recent = float(np.percentile([e for _, e in self._ears], cfg.rise_percentile))
        current = self.baseline.ear
        if cfg.min_open_ear <= recent <= cfg.max_open_ear and (current is None or recent > current):
            self.baseline = dataclasses.replace(
                self.baseline, ear=recent, ear_threshold=ear_threshold(recent, cfg)
            )
        return self.baseline
