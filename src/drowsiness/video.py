"""Frame sources: live cameras and video files, with timestamps in seconds."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# DirectShow opens webcams several times faster than Media Foundation on Windows.
_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

# A live camera may drop frames (USB hiccup, driver reset). Keep retrying, and
# reopen the device once, before declaring the camera lost.
CAMERA_RETRY_S = 2.0
_REOPEN_AFTER_S = 1.0
_RETRY_SLEEP_S = 0.05


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

        self._width = width
        self._height = height
        self._cap = self._open()
        self._mirror = mirror and self.is_live
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _open(self) -> cv2.VideoCapture:
        if self.is_live:
            cap = cv2.VideoCapture(self.source, _CAMERA_BACKEND)
            if not cap.isOpened():  # fall back to OpenCV's default backend
                cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            what = f"camera {self.source}" if self.is_live else f"video {self.source}"
            raise VideoSourceError(f"could not open {what}")
        if self.is_live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return cap

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        start = time.monotonic()
        index = 0
        last_file_t = -1.0
        while True:
            frame = self._read_live() if self.is_live else self._read_file()
            if frame is None:
                return
            if self._mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] > self._width:
                scale = self._width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if self.is_live:
                timestamp = time.monotonic() - start
            else:
                # Use the container's own timestamp of this frame (handles
                # variable frame rate); fall back to the nominal rate.
                timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp <= last_file_t:
                    timestamp = max(index / self.fps, last_file_t + 1e-3)
                last_file_t = timestamp
            index += 1
            yield frame, timestamp

    def _read_file(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def _read_live(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if ok:
            return frame
        failing_since = time.monotonic()
        reopened = False
        while time.monotonic() - failing_since < CAMERA_RETRY_S:
            time.sleep(_RETRY_SLEEP_S)
            if not reopened and time.monotonic() - failing_since >= _REOPEN_AFTER_S:
                reopened = True
                log.warning("Camera stopped delivering frames; reopening it")
                self._cap.release()
                try:
                    self._cap = self._open()
                except VideoSourceError:
                    continue
            ok, frame = self._cap.read()
            if ok:
                log.info("Camera recovered")
                return frame
        raise VideoSourceError(f"camera {self.source} stopped delivering frames")

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
