"""Detection of conditions that make the eye cue unreliable.

Two conditions are tracked:

- **Low light**: the face region is dark, so eye landmarks get noisy.
- **Eye occlusion**: the eye region cannot be seen properly, because of glare on
  glasses, dark glasses or a hand. Clear glasses alone do not count.

Each frame is measured from the image and the landmarks, the measurements are
smoothed over time with an exponential moving average, and each condition has
separate enter and exit thresholds so it does not flicker. The result tells the
adaptive fusion which cues to trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from drowsiness.config import ConditionConfig
from drowsiness.landmarks import FaceObservation
from drowsiness.signals import FaceSignals

# Full eye contours (lids and corners) of MediaPipe's face mesh.
LEFT_EYE_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE_CONTOUR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
# The eye region is the contour's box widened by this fraction of its width on
# every side, so it includes lids, brows and any glasses frame.
_EYE_PAD = 0.25


@dataclass(frozen=True)
class FrameConditions:
    """Raw per-frame measurements."""

    face_brightness: float  # mean grey level of the face region, 0-255
    glare_fraction: float  # share of eye-region pixels that are near white
    eye_contrast_ratio: float  # eye-region contrast / face contrast
    ear_asymmetry: float | None  # |L - R| / max(L, R); None when the head is turned

    def occlusion_reasons(self, config: ConditionConfig) -> tuple[str, ...]:
        reasons = []
        if self.glare_fraction >= config.glare_fraction:
            reasons.append("glare")
        if self.eye_contrast_ratio < config.min_eye_contrast_ratio:
            reasons.append("eyes covered")
        if self.ear_asymmetry is not None and self.ear_asymmetry > config.max_ear_asymmetry:
            reasons.append("one eye hidden")
        return tuple(reasons)


@dataclass(frozen=True)
class Conditions:
    """Smoothed, debounced conditions for the current frame."""

    low_light: bool = False
    eye_occluded: bool = False
    brightness: float | None = None  # smoothed face brightness
    occlusion: float = 0.0  # smoothed share of recent frames that were occluded
    reasons: tuple[str, ...] = ()  # why the eyes count as occluded

    @property
    def normal(self) -> bool:
        return not (self.low_light or self.eye_occluded)

    @property
    def label(self) -> str:
        parts = []
        if self.low_light:
            parts.append("low light")
        if self.eye_occluded:
            parts.append("eyes occluded")
        return ", ".join(parts) or "normal"


NORMAL = Conditions()


def measure(
    frame_bgr: np.ndarray, face: FaceObservation, signals: FaceSignals, config: ConditionConfig
) -> FrameConditions:
    """Measure brightness and eye visibility for one frame."""
    points = face.points2d
    face_gray = _crop_gray(frame_bgr, points)
    face_brightness = float(face_gray.mean()) if face_gray.size else 0.0
    face_contrast = float(face_gray.std()) if face_gray.size else 0.0

    glare, contrast = [], []
    for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
        eye = points[list(contour)]
        pad = _EYE_PAD * float(eye[:, 0].max() - eye[:, 0].min())
        eye_gray = _crop_gray(frame_bgr, eye, pad)
        if eye_gray.size == 0:
            continue
        glare.append(float((eye_gray >= config.glare_level).mean()))
        contrast.append(float(eye_gray.std()))

    # The worst eye decides: glare on one lens or a hand over one eye is enough.
    glare_fraction = max(glare, default=0.0)
    eye_contrast = min(contrast, default=0.0)
    ratio = eye_contrast / face_contrast if face_contrast > 1e-6 else 0.0

    asymmetry = None
    turned = signals.pose is not None and abs(signals.pose.yaw) > config.asymmetry_max_yaw_deg
    widest = max(signals.ear_left, signals.ear_right)
    if not turned and widest > 1e-6:
        asymmetry = abs(signals.ear_left - signals.ear_right) / widest

    return FrameConditions(face_brightness, glare_fraction, ratio, asymmetry)


def _crop_gray(frame_bgr: np.ndarray, points: np.ndarray, pad: float = 0.0) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0, y0 = np.floor(points.min(axis=0) - pad).astype(int)
    x1, y1 = np.ceil(points.max(axis=0) + pad).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0), np.uint8)
    return cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)


class ConditionDetector:
    """Turns per-frame measurements into stable conditions.

    Brightness and the occluded/not-occluded flag are each smoothed with an
    exponential moving average whose weight depends on the time since the last
    frame, so behaviour does not depend on the frame rate. While the face is
    missing nothing can be measured and the last conditions are kept.
    """

    def __init__(self, config: ConditionConfig):
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._last_t: float | None = None
        self._brightness: float | None = None
        self._occlusion = 0.0
        self.state = NORMAL

    def update(self, t: float, frame: FrameConditions | None) -> Conditions:
        if frame is None:
            return self.state
        cfg = self._config
        reasons = frame.occlusion_reasons(cfg)

        if self._last_t is None or self._brightness is None:
            # First observation: start from it instead of averaging with nothing.
            self._brightness = frame.face_brightness
            self._occlusion = 1.0 if reasons else 0.0
        else:
            alpha = 1.0 - math.exp(-max(t - self._last_t, 0.0) / cfg.smoothing_s)
            self._brightness += alpha * (frame.face_brightness - self._brightness)
            self._occlusion += alpha * ((1.0 if reasons else 0.0) - self._occlusion)
        self._last_t = t

        low_light = self.state.low_light
        if low_light and self._brightness > cfg.low_light_exit:
            low_light = False
        elif not low_light and self._brightness < cfg.low_light_enter:
            low_light = True

        occluded = self.state.eye_occluded
        if occluded and self._occlusion < cfg.occlusion_exit:
            occluded = False
        elif not occluded and self._occlusion > cfg.occlusion_enter:
            occluded = True

        # Keep showing the last reason while the condition holds through a
        # frame that happened to look clean.
        if occluded and not reasons:
            reasons = self.state.reasons
        self.state = Conditions(
            low_light=low_light,
            eye_occluded=occluded,
            brightness=self._brightness,
            occlusion=self._occlusion,
            reasons=reasons if occluded else (),
        )
        return self.state
