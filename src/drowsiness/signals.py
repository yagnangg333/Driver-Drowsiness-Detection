"""Per-frame geometric signals: eye aspect ratio, mouth aspect ratio, head pose.

Landmark indices refer to MediaPipe's 478-point face mesh. "Left" and "right"
mean the side of the image, not the driver's anatomical side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from drowsiness.landmarks import FaceObservation

# Six EAR points per eye, ordered p1..p6: outer/inner corner, two upper-lid
# points and two lower-lid points (Soukupova & Cech, 2016).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

# Inner-lip corners and three vertical upper/lower lip pairs.
MOUTH_CORNERS = (78, 308)
MOUTH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))

# Landmarks used for head pose and their positions on a generic 3D face, in a
# camera-aligned frame: x to image-right, y down, z away from the camera. A face
# looking straight into the camera therefore has zero rotation. The units are
# arbitrary (about 4.5x a real face in mm); scale only affects the translation,
# which is unused, not the rotation.
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
FACE_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, 330.0, 65.0),  # chin
        (-225.0, -170.0, 135.0),  # image-left eye outer corner
        (225.0, -170.0, 135.0),  # image-right eye outer corner
        (-150.0, 150.0, 125.0),  # image-left mouth corner
        (150.0, 150.0, 125.0),  # image-right mouth corner
    ],
    dtype=np.float64,
)
_MODEL_EYE_SPAN = 450.0  # distance between the two eye corners in FACE_MODEL_3D


@dataclass(frozen=True)
class HeadPose:
    """Head rotation in degrees.

    pitch > 0: chin down (nodding); yaw > 0: face turned towards image-right;
    roll > 0: head tilted clockwise as seen in the image.
    """

    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class FaceSignals:
    ear_left: float
    ear_right: float
    mar: float
    pose: HeadPose | None

    @property
    def ear(self) -> float:
        return (self.ear_left + self.ear_right) / 2.0


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR for six (x, y) eye points ordered p1..p6."""
    horizontal = np.linalg.norm(eye[0] - eye[3])
    if horizontal < 1e-6:
        return 0.0
    vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
    return float(vertical / (2.0 * horizontal))


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """Mean inner-lip opening divided by inner mouth width."""
    width = np.linalg.norm(points[MOUTH_CORNERS[0]] - points[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    opening = np.mean([np.linalg.norm(points[a] - points[b]) for a, b in MOUTH_VERTICAL_PAIRS])
    return float(opening / width)


def estimate_head_pose(points: np.ndarray, image_size: tuple[int, int]) -> HeadPose | None:
    """Estimate head rotation from 2D landmarks with a pinhole camera model."""
    width, height = image_size
    image_points = points[list(POSE_LANDMARKS), :2].astype(np.float64)
    focal = float(width)
    camera = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Start from a frontal face at the distance implied by the eye span, so the
    # iterative solver converges to the physically plausible solution.
    eye_span_px = np.linalg.norm(image_points[2] - image_points[3])
    if eye_span_px < 1e-6:
        return None
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [focal * _MODEL_EYE_SPAN / eye_span_px]])
    ok, rvec, _ = cv2.solvePnP(
        FACE_MODEL_3D,
        image_points,
        camera,
        np.zeros(4),
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rotation_to_euler(cv2.Rodrigues(rvec)[0])


def rotation_to_euler(rotation: np.ndarray) -> HeadPose:
    """Decompose R = Rz(roll) @ Ry(yaw) @ Rx(pitch) into degrees."""
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return HeadPose(math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def compute_signals(face: FaceObservation) -> FaceSignals:
    points = face.points2d
    return FaceSignals(
        ear_left=eye_aspect_ratio(points[list(LEFT_EYE)]),
        ear_right=eye_aspect_ratio(points[list(RIGHT_EYE)]),
        mar=mouth_aspect_ratio(points),
        pose=estimate_head_pose(face.landmarks, face.image_size),
    )
