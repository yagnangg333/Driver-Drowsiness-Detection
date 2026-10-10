import math

import cv2
import numpy as np
import pytest

from drowsiness.signals import (
    FACE_MODEL_3D,
    MOUTH_CORNERS,
    MOUTH_VERTICAL_PAIRS,
    POSE_LANDMARKS,
    estimate_head_pose,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)

IMAGE_SIZE = (640, 480)


def _eye(opening: float) -> np.ndarray:
    # p1..p6: corners at x=0 and x=10, lids at +-opening/2
    h = opening / 2
    return np.array([(0, 0), (3, -h), (7, -h), (10, 0), (7, h), (3, h)], dtype=float)


def test_ear_open_eye():
    assert eye_aspect_ratio(_eye(4.0)) == pytest.approx(0.4)


def test_ear_closed_eye_is_zero():
    assert eye_aspect_ratio(_eye(0.0)) == 0.0


def test_ear_degenerate_eye():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_ear_is_scale_invariant():
    assert eye_aspect_ratio(_eye(3.0) * 5) == pytest.approx(eye_aspect_ratio(_eye(3.0)))


def _mouth(opening: float, width: float = 40.0) -> np.ndarray:
    points = np.zeros((478, 2))
    points[MOUTH_CORNERS[0]] = (0, 0)
    points[MOUTH_CORNERS[1]] = (width, 0)
    for i, (top, bottom) in enumerate(MOUTH_VERTICAL_PAIRS):
        x = width * (i + 1) / 4
        points[top] = (x, -opening / 2)
        points[bottom] = (x, opening / 2)
    return points


def test_mar_closed_and_open():
    assert mouth_aspect_ratio(_mouth(0.0)) == 0.0
    assert mouth_aspect_ratio(_mouth(20.0)) == pytest.approx(0.5)


def _rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    p, y, r = map(math.radians, (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _project(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Landmark array whose pose points are a projected, rotated generic face."""
    width, height = IMAGE_SIZE
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=float)
    rvec, _ = cv2.Rodrigues(_rotation(pitch, yaw, roll))
    tvec = np.array([0.0, 0.0, 2500.0])
    projected, _ = cv2.projectPoints(FACE_MODEL_3D, rvec, tvec, camera, np.zeros(4))
    points = np.zeros((478, 3))
    points[list(POSE_LANDMARKS), :2] = projected.reshape(-1, 2)
    return points


@pytest.mark.parametrize(
    "pitch, yaw, roll", [(0, 0, 0), (20, 0, 0), (-15, 0, 0), (0, 25, 0), (0, 0, -20), (15, -10, 12)]
)
def test_head_pose_round_trip(pitch, yaw, roll):
    pose = estimate_head_pose(_project(pitch, yaw, roll), IMAGE_SIZE)
    assert pose is not None
    assert pose.pitch == pytest.approx(pitch, abs=1.0)
    assert pose.yaw == pytest.approx(yaw, abs=1.0)
    assert pose.roll == pytest.approx(roll, abs=1.0)


def test_positive_pitch_means_nodding_down():
    # Nodding forward foreshortens the nose-to-chin distance in the image.
    nose, chin = POSE_LANDMARKS.index(1), POSE_LANDMARKS.index(152)
    frontal = _project(0, 0, 0)[list(POSE_LANDMARKS)]
    nodded = _project(25, 0, 0)[list(POSE_LANDMARKS)]
    assert nodded[chin, 1] - nodded[nose, 1] < frontal[chin, 1] - frontal[nose, 1]
