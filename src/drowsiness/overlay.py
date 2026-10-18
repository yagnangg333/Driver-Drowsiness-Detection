"""Draws landmarks and a status HUD onto BGR frames."""

from __future__ import annotations

import cv2
import numpy as np

from drowsiness.alerts import AlertLevel
from drowsiness.conditions import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR
from drowsiness.pipeline import FrameReport

INNER_LIPS = (
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
)

# BGR
GREEN = (80, 200, 80)
RED = (60, 60, 230)
WHITE = (240, 240, 240)
LEVEL_COLORS = {
    AlertLevel.NONE: (80, 160, 60),
    AlertLevel.WARNING: (0, 190, 240),
    AlertLevel.ALARM: (0, 120, 255),
    AlertLevel.CRITICAL: (40, 40, 220),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(frame: np.ndarray, report: FrameReport, fps: float | None = None) -> np.ndarray:
    """Annotate ``frame`` in place and return it."""
    if report.face is not None:
        points = report.face.points2d.astype(np.int32)
        eye_color = RED if report.eye.closed else GREEN
        for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
            cv2.polylines(frame, [points[list(contour)]], True, eye_color, 1, cv2.LINE_AA)
        mouth_color = RED if report.yawn.yawning else GREEN
        cv2.polylines(frame, [points[list(INNER_LIPS)]], True, mouth_color, 1, cv2.LINE_AA)

    if report.calibrating:
        _draw_calibration(frame, report.calibration_progress, report.calibration_note)
    else:
        _draw_hud(frame, report)
        _draw_banner(frame, report)
    if fps is not None:
        _text(frame, f"{fps:.0f} FPS", (frame.shape[1] - 80, 22), 0.5)
    return frame


def _draw_hud(frame: np.ndarray, report: FrameReport) -> None:
    s = report.signals
    pose = s.pose if s else None
    lines = [
        f"EAR  {s.ear:.3f}" if s else "EAR  --",
        f"MAR  {s.mar:.3f}" if s else "MAR  --",
        f"Pitch {pose.pitch:+.0f}  Roll {pose.roll:+.0f}" if pose else "Pose --",
        f"Eyes closed {report.eye.perclos * 100:.0f}%",
        _conditions_line(report),
    ]
    _panel(frame, (8, 8), (205, 20 + 20 * len(lines) + 3 * 18 + 18))
    y = 28
    for line in lines:
        _text(frame, line, (16, y), 0.5)
        y += 20
    for name, value in (
        ("Eyes", report.eye.score),
        ("Yawn", report.yawn.score),
        ("Head", report.head.score),
    ):
        _bar(frame, name, value, (16, y - 10))
        y += 18
    w = report.fusion.weights
    _text(frame, f"w {w.eye:.2f}/{w.yawn:.2f}/{w.head:.2f}", (16, y + 2), 0.45)


def _conditions_line(report: FrameReport) -> str:
    c = report.conditions
    if c.normal:
        return "Cond: normal"
    return "Cond: " + "+".join(
        name for name, on in (("low light", c.low_light), ("eyes hidden", c.eye_occluded)) if on
    )


def _draw_banner(frame: np.ndarray, report: FrameReport) -> None:
    level = report.alert_level
    h, w = frame.shape[:2]
    text = {
        AlertLevel.NONE: "Awake",
        AlertLevel.WARNING: "Warning: signs of drowsiness",
        AlertLevel.ALARM: "DROWSY - take a break",
        AlertLevel.CRITICAL: "WAKE UP!",
    }[level]
    if not report.face_detected:
        text = "Looking for face..." if level == AlertLevel.NONE else "FACE NOT VISIBLE"
    cv2.rectangle(frame, (0, h - 36), (w, h), LEVEL_COLORS[level], -1)
    _text(frame, f"{text}   score {report.fusion.score:.2f}", (12, h - 12), 0.65, thickness=2)
    if level == AlertLevel.CRITICAL:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), LEVEL_COLORS[level], 6)


def _draw_calibration(frame: np.ndarray, progress: float, note: str | None = None) -> None:
    h, w = frame.shape[:2]
    _panel(frame, (0, h - 60), (w, 60))
    _text(frame, "Calibrating: look at the road with eyes open", (12, h - 36), 0.55)
    if note:
        _panel(frame, (0, h - 90), (w, 30))
        cv2.putText(frame, note[:70], (12, h - 70), _FONT, 0.45, (80, 200, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (12, h - 22), (w - 12, h - 12), WHITE, 1)
    cv2.rectangle(frame, (12, h - 22), (12 + int((w - 24) * progress), h - 12), GREEN, -1)


def _bar(frame: np.ndarray, label: str, value: float, origin: tuple[int, int]) -> None:
    x, y = origin
    _text(frame, label, (x, y + 9), 0.45)
    cv2.rectangle(frame, (x + 45, y), (x + 175, y + 10), WHITE, 1)
    color = GREEN if value < 0.5 else RED
    cv2.rectangle(frame, (x + 45, y), (x + 45 + int(130 * value), y + 10), color, -1)


def _panel(frame: np.ndarray, origin: tuple[int, int], size: tuple[int, int]) -> None:
    x, y = origin
    w, h = size
    roi = frame[y : y + h, x : x + w]
    roi[:] = (roi * 0.45).astype(frame.dtype)


def _text(frame, text, origin, scale, thickness=1) -> None:
    cv2.putText(frame, text, origin, _FONT, scale, WHITE, thickness, cv2.LINE_AA)
