import dataclasses

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
CFG = ConditionConfig()


def fc(brightness=120.0, glare=0.0, contrast=0.5, asymmetry=0.0) -> FrameConditions:
    """A frame's measurements; defaults describe a bright daytime frame."""
    return FrameConditions(
        face_brightness=brightness,
        frame_brightness=brightness,
        glare_fraction=glare,
        eye_contrast_ratio=contrast,
        ear_asymmetry=asymmetry,
    )


CLEAR = fc()
NIGHT = fc(brightness=40)
GLARE = fc(glare=0.3)


def _feed(detector, frame, seconds, start):
    t, state = start, None
    for _ in range(round(seconds * FPS)):
        state = detector.update(t, frame)
        t += 1 / FPS
    return state, t


@pytest.fixture
def detector():
    return ConditionDetector(CFG)


@pytest.mark.parametrize(
    "frame, reasons",
    [
        (CLEAR, ()),
        (GLARE, ("glare",)),
        (fc(contrast=0.02), ("eyes covered",)),  # below the absolute floor
        (fc(asymmetry=0.8), ("one eye hidden",)),
        (fc(asymmetry=None), ()),  # head turned: asymmetry not checked
        (fc(glare=0.3, contrast=0.02), ("glare", "eyes covered")),
    ],
)
def test_occlusion_reasons(frame, reasons):
    assert frame.occlusion_reasons(CFG) == reasons


def test_starts_normal(detector):
    assert detector.state.normal and detector.state.label == "normal"
    state = detector.update(0.0, CLEAR)
    assert state.normal and state.brightness == 120


def test_low_light_needs_sustained_darkness(detector):
    _, t = _feed(detector, CLEAR, 2.0, 0.0)
    state, t = _feed(detector, NIGHT, 0.2, t)  # a passing shadow
    assert not state.low_light
    state, t = _feed(detector, NIGHT, 3.0, t)
    assert state.low_light and state.label == "low light"


def test_low_light_hysteresis(detector):
    state, t = _feed(detector, NIGHT, 1.0, 0.0)
    assert state.low_light
    # Just above the thresholds but within the margin: still low light.
    between = fc(brightness=CFG.low_light_brightness + CFG.low_light_margin / 2)
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
    state, t = _feed(detector, NIGHT, 1.0, 0.0)
    assert detector.update(t, None) == state
    assert detector.update(t + 0.5, None).low_light
    # After a long gap the stale average gives way to the first new frame.
    assert not detector.update(t + 5.0, CLEAR).low_light


def test_reset(detector):
    _feed(detector, NIGHT, 1.0, 0.0)
    detector.reset()
    assert detector.state.normal and detector.state.brightness is None


def test_both_conditions(detector):
    night_glare = fc(brightness=40, glare=0.3)
    state, _ = _feed(detector, night_glare, 2.0, 0.0)
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
    if kind == "night":
        # Darker and drained of colour, as a camera's night image is.
        grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return (0.35 * (0.5 * frame + 0.5 * grey)).astype(np.uint8)
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
        ("night", True, False),
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
    assert (measured.frame_brightness < config.low_light_brightness) == dark
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


def test_smiling_follows_the_recent_smile_score(detector):
    smile = dataclasses.replace(CLEAR, smile=0.6)
    state, t = _feed(detector, CLEAR, 2.0, 0.0)
    assert not state.smiling and state.label == "normal"
    state, t = _feed(detector, smile, 1.0, t)
    assert state.smiling and state.label == "smiling"
    assert state.normal  # smiling does not change the fusion weights
    state, _ = _feed(detector, CLEAR, 1.2, t)
    assert not state.smiling


def test_a_yawn_is_not_a_smile(detector):
    # MediaPipe partly scores a yawn's stretched mouth as a smile.
    yawn = dataclasses.replace(CLEAR, smile=0.6, mar=0.7)
    state, t = _feed(detector, yawn, 1.5, 0.0)
    assert not state.smiling
    # Still not a smile just after the yawn, while the smile score lingers...
    after = dataclasses.replace(CLEAR, smile=0.6, mar=0.1)
    state, t = _feed(detector, after, 1.0, t)
    assert not state.smiling
    # ...but a real smile once the mouth has been closed for a while.
    state, _ = _feed(detector, after, 2.0, t)
    assert state.smiling


def test_smiling_is_judged_against_the_resting_smile(detector):
    # A resting face that MediaPipe already scores at 0.25: not a smile for
    # this driver, but a clearly bigger smile still is.
    resting = dataclasses.replace(CLEAR, smile=0.25)
    state, t = None, 0.0
    for _ in range(30):
        state = detector.update(t, resting, resting_smile=0.25)
        t += 1 / FPS
    assert not state.smiling
    bigger = dataclasses.replace(CLEAR, smile=0.6)
    for _ in range(30):
        state = detector.update(t, bigger, resting_smile=0.25)
        t += 1 / FPS
    assert state.smiling


def test_no_smile_without_a_face(detector):
    state, t = _feed(detector, dataclasses.replace(CLEAR, smile=0.6), 1.5, 0.0)
    assert state.smiling
    assert not detector.update(t + 5.0, None).smiling
