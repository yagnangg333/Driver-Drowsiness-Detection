"""End-to-end tests with the real MediaPipe model on two recorded webcam frames
(the driver wears glasses in both)."""

import csv
import dataclasses
import math

import cv2
import numpy as np
import pytest

from drowsiness import overlay
from drowsiness.alerts import AlertLevel
from drowsiness.cli import main
from drowsiness.config import AppConfig, CalibrationConfig
from drowsiness.landmarks import FaceLandmarker
from drowsiness.pipeline import DrowsinessPipeline
from drowsiness.signals import estimate_head_pose

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
    for column in ("low_light", "eye_occluded", "static_score", "adaptive_score", "eye_weight"):
        assert column in rows[-1]


def test_face_loss_after_calibration_escalates_to_critical(config, awake_frame):
    blank = np.zeros_like(awake_frame)  # head out of view / camera covered
    with DrowsinessPipeline(config) as pipeline:
        report, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        report, t = _feed(pipeline, blank, 6.5, t)
        assert not report.face_detected and report.alert_level == AlertLevel.ALARM
        report, _ = _feed(pipeline, blank, 4.0, t)
        assert report.alert_level == AlertLevel.CRITICAL


def test_calibration_with_closed_eyes_is_rejected(config, drowsy_frame, awake_frame):
    with DrowsinessPipeline(config) as pipeline:
        report, t = _feed(pipeline, drowsy_frame, 2.5, 0.0)
        assert report.calibrating and "eyes look closed" in report.calibration_note
        report, _ = _feed(pipeline, awake_frame, 2.5, t)
        assert not report.calibrating and report.baseline.calibrated
        assert report.baseline.ear > 0.28


def test_overlay_draws_every_state(config, awake_frame, drowsy_frame):
    with DrowsinessPipeline(config) as pipeline:
        states = [pipeline.process(awake_frame, 0.0)]  # calibrating
        _, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        states.append(pipeline.process(awake_frame, t))  # awake
        report, t = _feed(pipeline, drowsy_frame, 2.0, t)
        states.append(report)  # alarm
        states.append(pipeline.process(np.zeros_like(awake_frame), t))  # no face
    for report in states:
        frame = awake_frame.copy()
        out = overlay.draw(frame, report, fps=25.0)
        assert out.shape == awake_frame.shape
        assert not np.array_equal(out, awake_frame)


def _rotation(pitch, yaw):
    p, y = math.radians(pitch), math.radians(yaw)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return ry @ rx


@pytest.mark.parametrize("pitch, yaw", [(20, 0), (-20, 0), (0, 45), (0, -45)])
def test_head_pose_on_real_face_geometry(awake_frame, pitch, yaw):
    """Rotate the driver's real 3D landmarks (not the generic model used by the
    solver) and check the estimate: nodding down must read as positive pitch,
    and turning to check a mirror must not look like a nod."""
    with FaceLandmarker() as landmarker:
        face = landmarker.detect(awake_frame, 0.0)
    points = face.landmarks
    centre = points.mean(axis=0)
    base = estimate_head_pose(points, face.image_size)
    rotated = (points - centre) @ _rotation(pitch, yaw).T + centre
    pose = estimate_head_pose(rotated, face.image_size)
    d_pitch = pose.pitch - base.pitch
    if pitch:
        assert math.copysign(1, d_pitch) == math.copysign(1, pitch)
        assert abs(d_pitch) == pytest.approx(abs(pitch), rel=0.4)
    else:
        assert abs(d_pitch) < 8.0  # well below the 15 deg nod threshold


def test_no_phantom_microsleep_after_face_loss(config, awake_frame, drowsy_frame):
    """Eyes closing, face away for 2 s, back with eyes closed: the 2 s that
    were not observed must not be counted as a closure."""
    blank = np.zeros_like(awake_frame)
    with DrowsinessPipeline(config) as pipeline:
        _, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        _, t = _feed(pipeline, drowsy_frame, 1.0, t)
        _, t = _feed(pipeline, blank, 2.0, t)
        report, _ = _feed(pipeline, drowsy_frame, 0.3, t)
    assert report.face_detected and report.eye.closed
    assert not report.eye.microsleep
    assert report.eye.closed_duration_s < 0.5
    assert report.alert_level < AlertLevel.ALARM


def test_single_dropped_frame_does_not_reset_microsleep(config, awake_frame, drowsy_frame):
    blank = np.zeros_like(awake_frame)
    with DrowsinessPipeline(config) as pipeline:
        _, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        _, t = _feed(pipeline, drowsy_frame, 1.0, t)
        dropped = pipeline.process(blank, t)  # one lost frame (< 0.3 s grace)
        report, _ = _feed(pipeline, drowsy_frame, 0.7, t + 1 / FPS)
    assert not dropped.face_detected
    assert report.eye.microsleep and report.eye.closed_duration_s > 1.5
    assert report.alert_level >= AlertLevel.ALARM


def test_low_light_switches_to_adaptive_weights(config, awake_frame, drowsy_frame):
    with DrowsinessPipeline(config) as pipeline:
        report, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        assert report.conditions.normal
        assert report.fusion.weights.eye == pytest.approx(0.5)

        dark = (drowsy_frame * 0.2).astype(np.uint8)
        report, _ = _feed(pipeline, dark, 3.0, t)
    assert report.conditions.low_light and not report.conditions.eye_occluded
    assert report.fusion.weights.eye == pytest.approx(0.2)
    assert report.fusion.score == pytest.approx(report.adaptive_score)
    # The same indicator scores, weighted the two ways.
    s = report.scores
    assert report.static_score == pytest.approx(0.5 * s.eye + 0.2 * s.yawn + 0.3 * s.head)
    assert report.adaptive_score == pytest.approx(0.2 * s.eye + 0.35 * s.yawn + 0.45 * s.head)
    row = report.to_row()
    assert row["low_light"] == 1 and row["eye_weight"] == pytest.approx(0.2)


def test_static_strategy_ignores_conditions(config, awake_frame):
    config = dataclasses.replace(
        config, fusion=dataclasses.replace(config.fusion, strategy="static")
    )
    with DrowsinessPipeline(config) as pipeline:
        _, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        report, _ = _feed(pipeline, (awake_frame * 0.2).astype(np.uint8), 3.0, t)
    assert report.conditions.low_light
    assert report.fusion.weights.eye == pytest.approx(0.5)
    assert report.fusion.score == pytest.approx(report.static_score)


@pytest.mark.parametrize("gate, expect_alarm", [(True, False), (False, True)])
def test_microsleep_needs_visible_eyes(
    config, awake_frame, drowsy_frame, monkeypatch, gate, expect_alarm
):
    """With the eyes occluded an "eyes closed" reading may be glare, so a
    microsleep alone must not force the alarm (unless the gate is turned off)."""
    from drowsiness import pipeline as pipeline_module
    from drowsiness.conditions import FrameConditions

    config = dataclasses.replace(
        config,
        conditions=dataclasses.replace(config.conditions, microsleep_needs_visible_eyes=gate),
    )
    real_measure = pipeline_module.measure
    occluded = FrameConditions(80, glare_fraction=0.3, eye_contrast_ratio=0.5, ear_asymmetry=0)
    with DrowsinessPipeline(config) as pipeline:
        _, t = _feed(pipeline, awake_frame, 2.5, 0.0)
        monkeypatch.setattr(pipeline_module, "measure", lambda *a: occluded)
        report, _ = _feed(pipeline, drowsy_frame, 2.5, t)
        monkeypatch.setattr(pipeline_module, "measure", real_measure)
    assert report.conditions.eye_occluded
    assert report.eye.microsleep
    assert (report.alert_level >= AlertLevel.ALARM) == expect_alarm
