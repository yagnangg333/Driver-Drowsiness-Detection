import cv2
import numpy as np
import pytest

from drowsiness.conditions import (
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    ConditionDetector,
    FrameConditions,
    ear_asymmetry,
    measure,
)
from drowsiness.config import ConditionConfig
from drowsiness.landmarks import FaceLandmarker
from drowsiness.signals import FaceSignals, HeadPose, compute_signals

from .conftest import requires_model

FPS = 15
# FrameConditions(face_brightness, glare_fraction, eye_contrast_ratio, ear_asymmetry)
CLEAR = FrameConditions(80, 0, 0.5, 0)
DARK = FrameConditions(15, 0, 0.5, 0)
GLARE = FrameConditions(80, 0.3, 0.5, 0)


def _feed(detector, frame, seconds, start):
    t, state = start, None
    for _ in range(round(seconds * FPS)):
        state = detector.update(t, frame)
        t += 1 / FPS
    return state, t


@pytest.fixture
def detector():
    return ConditionDetector(ConditionConfig())


@pytest.mark.parametrize(
    "frame, reasons",
    [
        (CLEAR, ()),
        (GLARE, ("glare",)),
        (FrameConditions(80, 0, 0.1, 0), ("eyes covered",)),
        (FrameConditions(80, 0, 0.5, 0.8), ("one eye hidden",)),
        (FrameConditions(80, 0, 0.5, None), ()),  # head turned: asymmetry not checked
        (FrameConditions(80, 0.3, 0.1, 0), ("glare", "eyes covered")),
    ],
)
def test_occlusion_reasons(frame, reasons):
    assert frame.occlusion_reasons(ConditionConfig()) == reasons


def test_starts_normal(detector):
    assert detector.state.normal and detector.state.label == "normal"
    state = detector.update(0.0, CLEAR)
    assert state.normal and state.brightness == 80


def test_low_light_needs_sustained_darkness(detector):
    _, t = _feed(detector, CLEAR, 2.0, 0.0)
    state, t = _feed(detector, DARK, 0.2, t)  # a passing shadow
    assert not state.low_light
    state, t = _feed(detector, DARK, 2.0, t)
    assert state.low_light and state.label == "low light"


def test_low_light_hysteresis(detector):
    state, t = _feed(detector, DARK, 1.0, 0.0)
    assert state.low_light
    between = FrameConditions(45, 0, 0.5, 0)  # above enter (40), below exit (50)
    state, t = _feed(detector, between, 5.0, t)
    assert state.low_light
    state, _ = _feed(detector, CLEAR, 3.0, t)
    assert not state.low_light


def test_occlusion_enters_and_exits(detector):
    _, t = _feed(detector, CLEAR, 2.0, 0.0)
    state, t = _feed(detector, GLARE, 0.3, t)  # a brief reflection
    assert not state.eye_occluded
    state, t = _feed(detector, GLARE, 1.5, t)
    assert state.eye_occluded and state.reasons == ("glare",)
    assert state.label == "eyes occluded"
    state, t = _feed(detector, CLEAR, 0.3, t)  # still held: no flicker
    assert state.eye_occluded and state.reasons == ("glare",)
    state, _ = _feed(detector, CLEAR, 2.0, t)
    assert not state.eye_occluded and state.reasons == ()


def test_face_missing_keeps_conditions(detector):
    state, t = _feed(detector, DARK, 1.0, 0.0)
    assert detector.update(t, None) == state
    assert detector.update(t + 0.5, None).low_light
    # After a long gap the stale average gives way to the first new frame.
    assert not detector.update(t + 5.0, CLEAR).low_light


def test_reset(detector):
    _feed(detector, DARK, 1.0, 0.0)
    detector.reset()
    assert detector.state.normal and detector.state.brightness is None


def test_both_conditions(detector):
    dark_glare = FrameConditions(15, 0.3, 0.5, 0)
    state, _ = _feed(detector, dark_glare, 2.0, 0.0)
    assert state.low_light and state.eye_occluded
    assert state.label == "low light, eyes occluded"


# --- measurements on the real recorded frame (the driver wears clear glasses) ---


def _eye_box(points, contour, pad):
    eye = points[list(contour)]
    width = eye[:, 0].max() - eye[:, 0].min()
    top_left = tuple((eye.min(0) - pad * width).astype(int))
    bottom_right = tuple((eye.max(0) + pad * width).astype(int))
    return top_left, bottom_right


def _simulate(frame, points, kind):
    out = frame.copy()
    if kind == "dim":
        return (frame * 0.2).astype(np.uint8)
    if kind == "sunglasses":
        for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
            cv2.rectangle(out, *_eye_box(points, contour, 0.45), (20, 20, 20), -1)
    elif kind == "glare":
        eye = points[list(LEFT_EYE_CONTOUR)]
        width = eye[:, 0].max() - eye[:, 0].min()
        cx, cy = eye.mean(0).astype(int)
        cv2.circle(out, (int(cx), int(cy)), int(width * 0.3), (255, 255, 255), -1)
    elif kind == "hand":
        cv2.rectangle(out, *_eye_box(points, RIGHT_EYE_CONTOUR, 0.6), (120, 150, 200), -1)
    return out


@requires_model
@pytest.mark.parametrize(
    "kind, dark, occluded",
    [
        ("normal", False, False),  # clear glasses alone are not an occlusion
        ("dim", True, False),
        ("sunglasses", False, True),
        ("glare", False, True),
        ("hand", False, True),
    ],
)
def test_measure_on_real_face(awake_frame, kind, dark, occluded):
    config = ConditionConfig()
    with FaceLandmarker() as landmarker:
        points = landmarker.detect(awake_frame, 0.0).points2d
        frame = _simulate(awake_frame, points, kind)
        face = landmarker.detect(frame, 0.1)
    assert face is not None
    measured = measure(frame, face, compute_signals(face), config)
    assert (measured.face_brightness < config.low_light_enter) == dark
    assert bool(measured.occlusion_reasons(config)) == occluded


@pytest.mark.parametrize(
    "left, right, yaw, expected",
    [
        (0.33, 0.32, 0.0, pytest.approx(0.03, abs=0.01)),  # both open
        (0.30, 0.12, 0.0, pytest.approx(0.6)),  # one eye open, one reads closed
        (0.05, 0.10, 0.0, None),  # both nearly shut: closing the eyes is not occlusion
        (0.03, 0.08, 0.0, None),
        (0.30, 0.12, 40.0, None),  # head turned: the far eye is foreshortened
    ],
)
def test_ear_asymmetry(left, right, yaw, expected):
    signals = FaceSignals(left, right, 0.1, HeadPose(0.0, yaw, 0.0))
    assert ear_asymmetry(signals, ConditionConfig()) == expected
