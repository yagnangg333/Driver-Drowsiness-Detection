"""Frame sources: live cameras and video files, with timestamps in seconds."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

# DirectShow opens webcams several times faster than Media Foundation on Windows.
_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class VideoSourceError(RuntimeError):
    pass


def parse_source(value: str | int) -> int | Path:
    """A camera index ("0") or a path to a video file."""
    if isinstance(value, int):
        return value
    return int(value) if value.isdigit() else Path(value)


class VideoSource:
    """Iterates over ``(frame_bgr, timestamp_s)``.

    Live cameras use wall-clock time; files use their own timeline so they
    are processed identically regardless of processing speed.
    """

    def __init__(
        self, source: int | str | Path, width: int = 640, height: int = 480, mirror: bool = True
    ):
        self.source = parse_source(source) if isinstance(source, str) else source
        self.is_live = isinstance(self.source, int)
        if not self.is_live and not Path(self.source).is_file():
            raise VideoSourceError(f"video file not found: {self.source}")

        if self.is_live:
            self._cap = cv2.VideoCapture(self.source, _CAMERA_BACKEND)
            if not self._cap.isOpened():  # fall back to OpenCV's default backend
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(str(self.source))
        if not self._cap.isOpened():
            what = f"camera {self.source}" if self.is_live else f"video {self.source}"
            raise VideoSourceError(f"could not open {what}")
        if self.is_live:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._mirror = mirror and self.is_live
        self._width = width
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        start = time.monotonic()
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self.is_live:
                    raise VideoSourceError("failed to read a frame from the camera")
                return
            if self._mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] > self._width:
                scale = self._width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            timestamp = time.monotonic() - start if self.is_live else index / self.fps
            index += 1
            yield frame, timestamp

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
