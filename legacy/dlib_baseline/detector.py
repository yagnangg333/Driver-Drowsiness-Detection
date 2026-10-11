"""Core drowsiness detection logic shared by the CLI and the Streamlit app.

Drowsiness is detected with the Eye Aspect Ratio (EAR) from Soukupova & Cech,
"Real-Time Eye Blink Detection using Facial Landmarks" (2016). When the EAR
stays below a threshold for a number of consecutive frames, the eyes are
considered closed for too long and the driver is flagged as drowsy.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = BASE_DIR.parents[1] / "assets" / "alarm.wav"

DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_CONSEC_FRAMES = 20
FRAME_WIDTH = 640

# BGR colours used for annotations.
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 215, 255)

L_START, L_END = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
R_START, R_END = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Return the EAR for the six (x, y) landmarks of one eye."""
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


@lru_cache(maxsize=1)
def _load_models(model_path: str):
    """Load dlib's face detector and landmark predictor once per process."""
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(model_path)


@dataclass
class FrameResult:
    face_found: bool
    ear: Optional[float]
    drowsy: bool


class DrowsinessDetector:
    """Tracks eye closure across frames and annotates each frame."""

    def __init__(
        self,
        threshold: float = DEFAULT_EAR_THRESHOLD,
        consec_frames: int = DEFAULT_CONSEC_FRAMES,
        model_path: Path = MODEL_PATH,
    ):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Landmark model not found at '{model_path}'. Download "
                "shape_predictor_68_face_landmarks.dat (see README) and place "
                "it in the 'models' folder."
            )
        self._face_detector, self._predictor = _load_models(str(model_path))
        self.threshold = threshold
        self.consec_frames = consec_frames
        self._closed_frames = 0

    def reset(self) -> None:
        self._closed_frames = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        """Analyse a BGR frame, draw the overlay on it and return the result."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(gray, 0)

        if not faces:
            _draw_status(frame, "No face detected", YELLOW)
            return FrameResult(face_found=False, ear=None, drowsy=False)

        # Only track the largest face, which is assumed to be the driver.
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = face_utils.shape_to_np(self._predictor(gray, face))
        left_eye = shape[L_START:L_END]
        right_eye = shape[R_START:R_END]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < self.threshold:
            self._closed_frames += 1
        else:
            self._closed_frames = 0
        drowsy = self._closed_frames >= self.consec_frames

        color = RED if drowsy else GREEN
        for eye in (left_eye, right_eye):
            cv2.drawContours(frame, [cv2.convexHull(eye)], -1, color, 1)
        _draw_status(frame, "DROWSY! WAKE UP!" if drowsy else "Awake", color)
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (frame.shape[1] - 130, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return FrameResult(face_found=True, ear=ear, drowsy=drowsy)


def _draw_status(frame: np.ndarray, text: str, color) -> None:
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


class Alarm:
    """Looping alert sound. Degrades to a no-op if audio is unavailable."""

    def __init__(self, sound_path: Path = ALARM_PATH):
        self._mixer = None
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            from pygame import mixer

            mixer.init()
            mixer.music.load(str(sound_path))
            self._mixer = mixer
        except Exception as exc:  # no audio device, missing file, etc.
            print(f"[warning] Alarm sound disabled: {exc}")

    @property
    def available(self) -> bool:
        return self._mixer is not None

    def start(self) -> None:
        if self._mixer and not self._mixer.music.get_busy():
            self._mixer.music.play(loops=-1)

    def stop(self) -> None:
        if self._mixer and self._mixer.music.get_busy():
            self._mixer.music.stop()
