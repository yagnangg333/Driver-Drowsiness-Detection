from typing import ClassVar

import cv2
import numpy as np
import pytest

from drowsiness import video
from drowsiness.video import VideoSource, VideoSourceError, parse_source

FRAME = np.zeros((48, 64, 3), np.uint8)


class FakeCapture:
    """Stands in for cv2.VideoCapture; ``script`` lists read() outcomes."""

    instances: ClassVar[list] = []

    def __init__(self, script):
        self.script = list(script)
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, prop):
        return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        ok = self.script.pop(0) if self.script else False
        return ok, (FRAME.copy() if ok else None)

    def release(self):
        self.released = True


@pytest.fixture
def fake_camera(monkeypatch):
    scripts: list[list[bool]] = []
    FakeCapture.instances = []

    def factory(*args, **kwargs):
        return FakeCapture(scripts.pop(0) if scripts else [])

    monkeypatch.setattr(video.cv2, "VideoCapture", factory)
    monkeypatch.setattr(video, "CAMERA_RETRY_S", 0.3)
    monkeypatch.setattr(video, "_REOPEN_AFTER_S", 0.1)
    monkeypatch.setattr(video, "_RETRY_SLEEP_S", 0.01)
    return scripts


def test_transient_camera_dropouts_are_retried(fake_camera):
    fake_camera.append([True, False, False, True, True])
    frames = []
    with VideoSource(0) as source:
        for frame, _ in source:
            frames.append(frame)
            if len(frames) == 3:
                break
    assert len(frames) == 3


def test_camera_is_reopened_and_recovers(fake_camera):
    fake_camera.append([True] + [False] * 1000)  # first device handle dies
    fake_camera.append([True, True])  # reopened handle works
    with VideoSource(0) as source:
        it = iter(source)
        next(it)
        next(it)
    assert len(FakeCapture.instances) == 2 and FakeCapture.instances[0].released


def test_lost_camera_raises_after_retrying(fake_camera):
    fake_camera.append([True] + [False] * 1000)
    fake_camera.append([False] * 1000)
    with VideoSource(0) as source, pytest.raises(VideoSourceError, match="stopped delivering"):
        for _ in source:
            pass


def test_video_file_uses_container_timestamps(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    for _ in range(5):
        writer.write(FRAME)
    writer.release()
    with VideoSource(path) as source:
        stamps = [t for _, t in source]
    assert stamps == pytest.approx([0, 1 / 15, 2 / 15, 3 / 15, 4 / 15], abs=1e-3)


def test_missing_video_file(tmp_path):
    with pytest.raises(VideoSourceError, match="not found"):
        VideoSource(tmp_path / "missing.mp4")


@pytest.mark.parametrize("value, expected", [("0", 0), ("2", 2), (3, 3)])
def test_parse_camera_index(value, expected):
    assert parse_source(value) == expected


def test_parse_video_path():
    assert str(parse_source("clips/a.mp4")).endswith("a.mp4")
