"""End-to-end tests with the real MediaPipe model on two recorded webcam frames
(the driver wears glasses in both)."""

import csv
import dataclasses

import cv2
import pytest

from drowsiness.alerts import AlertLevel
from drowsiness.cli import main
from drowsiness.config import AppConfig, CalibrationConfig
from drowsiness.pipeline import DrowsinessPipeline

from .conftest import requires_model

pytestmark = requires_model
FPS = 15


def _feed(pipeline, frame, seconds, start):
    t, report = start, None
    for _ in range(round(seconds * FPS)):
        report = pipeline.process(frame, t)
        t += 1 / FPS
    return report, t


@pytest.fixture
def config(tmp_path):
    config = AppConfig(calibration=CalibrationConfig(duration_s=2.0, min_samples=10))
    return dataclasses.replace(
        config, alerts=dataclasses.replace(config.alerts, log_dir=str(tmp_path))
    )


def test_calibrates_then_detects_closed_eyes(config, awake_frame, drowsy_frame):
    with DrowsinessPipeline(config) as pipeline:
        report, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        assert report.face_detected and not report.calibrating
        assert report.baseline.calibrated
        assert 0.28 < report.baseline.ear < 0.38

        report, t = _feed(pipeline, awake_frame, 1.0, t)
        assert not report.eye.closed and report.alert_level == AlertLevel.NONE

        report, t = _feed(pipeline, drowsy_frame, 2.0, t)
        assert report.eye.closed
        assert report.eye.score == 1.0
        assert report.alert_level >= AlertLevel.ALARM

        report, _ = _feed(pipeline, awake_frame, 0.5, t)
        assert not report.eye.closed


def test_no_face_in_blank_frame(config, awake_frame):
    with DrowsinessPipeline(config) as pipeline:
        report = pipeline.process(awake_frame * 0, 0.0)
    assert not report.face_detected
    assert report.to_row()["face_detected"] == 0


def test_process_command_writes_signals(tmp_path, awake_frame, drowsy_frame):
    video = tmp_path / "clip.avi"
    h, w = awake_frame.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    for frame in [awake_frame] * (2 * FPS) + [drowsy_frame] * (3 * FPS):
        writer.write(frame)
    writer.release()

    out = tmp_path / "signals.csv"
    assert main(["process", str(video), "--out", str(out), "--no-calibration"]) == 0
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 * FPS
    assert rows[0]["eyes_closed"] == "0"
    assert rows[-1]["eyes_closed"] == "1"
    assert int(rows[-1]["alert_level"]) >= AlertLevel.ALARM
