"""Face landmark extraction with the MediaPipe Tasks FaceLandmarker."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drowsiness.config import LandmarkerConfig, resolve_path

# Keep MediaPipe's native (glog/TFLite) start-up messages out of the console.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class FaceObservation:
    """Landmarks of the tracked face in one frame.

    ``landmarks`` is a (478, 3) array: x and y in pixels, z in the same scale
    as x (MediaPipe's relative depth, negative towards the camera).
    """

    landmarks: np.ndarray
    blendshapes: dict[str, float]
    image_size: tuple[int, int]  # (width, height)

    @property
    def points2d(self) -> np.ndarray:
        return self.landmarks[:, :2]


class FaceLandmarker:
    """Stateful video-mode landmarker; feed frames in chronological order."""

    def __init__(self, config: LandmarkerConfig | None = None):
        config = config or LandmarkerConfig()
        model_path = resolve_path(config.model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Face landmarker model not found at '{model_path}'. "
                "Run `drowsiness download-model` to fetch it."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=config.max_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> FaceObservation | None:
        """Return the driver's face (the largest one in view), or None."""
        # MediaPipe requires strictly increasing timestamps in video mode.
        timestamp_ms = max(int(timestamp_s * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        if not result.face_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        faces = [
            np.array([(p.x * width, p.y * height, p.z * width) for p in face], dtype=np.float64)
            for face in result.face_landmarks
        ]
        index = max(range(len(faces)), key=lambda i: _bbox_area(faces[i]))
        blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[index]}
        return FaceObservation(faces[index], blendshapes, (width, height))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _bbox_area(points: np.ndarray) -> float:
    span = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
    return float(span[0] * span[1])


def download_model(destination: str | Path | None = None, force: bool = False) -> Path:
    """Download and verify the FaceLandmarker model. Returns its path."""
    destination = resolve_path(destination or LandmarkerConfig().model_path)
    if destination.is_file() and not force:
        log.info("Model already present at %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    log.info("Downloading %s", MODEL_URL)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as out:
        out.write(response.read())

    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for downloaded model (got {digest})")
    partial.replace(destination)
    log.info("Saved model to %s", destination)
    return destination
