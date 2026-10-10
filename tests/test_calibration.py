import pytest

from drowsiness.calibration import Calibrator
from drowsiness.config import CalibrationConfig
from drowsiness.signals import FaceSignals, HeadPose


def _signals(ear=0.32, pitch=5.0, roll=-2.0):
    return FaceSignals(ear_left=ear, ear_right=ear, mar=0.05, pose=HeadPose(pitch, 0.0, roll))


def _feed(calibrator, seconds, signals, fps=30, start=0.0):
    t = start
    for _ in range(round(seconds * fps)):
        calibrator.update(t, signals)
        t += 1 / fps
    return t


def test_baseline_from_open_eyes():
    cal = Calibrator(CalibrationConfig(duration_s=5.0), default_ear_threshold=0.21)
    _feed(cal, 4.0, _signals())
    assert not cal.done and 0.7 < cal.progress < 0.9
    _feed(cal, 1.5, _signals(), start=4.0)
    assert cal.done
    base = cal.result
    assert base.calibrated
    assert base.ear_threshold == pytest.approx(0.32 * 0.75)
    assert (base.pitch, base.roll) == (pytest.approx(5.0), pytest.approx(-2.0))


def test_threshold_is_clamped():
    cal = Calibrator(CalibrationConfig(duration_s=1.0), default_ear_threshold=0.21)
    _feed(cal, 1.5, _signals(ear=0.42))  # 0.42 * 0.75 = 0.315 > max 0.30
    assert cal.result.ear_threshold == CalibrationConfig().max_ear_threshold


def test_closed_eyes_are_rejected_and_calibration_restarts():
    cal = Calibrator(CalibrationConfig(duration_s=1.0), default_ear_threshold=0.21)
    t = _feed(cal, 1.2, _signals(ear=0.11))  # driver already drowsy; rejected at ~1.0 s
    assert not cal.done
    assert "eyes look closed" in cal.note
    _feed(cal, 1.5, _signals(ear=0.32), start=t)  # then opens eyes properly
    assert cal.done and cal.result.calibrated
    assert cal.result.ear == pytest.approx(0.32)  # rejected samples were discarded
    assert cal.note is None


def test_looking_down_is_rejected():
    cal = Calibrator(CalibrationConfig(duration_s=1.0), default_ear_threshold=0.21)
    _feed(cal, 1.5, _signals(pitch=35.0))
    assert not cal.done and "head not facing ahead" in cal.note


def test_repeated_rejection_falls_back_to_defaults_with_note():
    cal = Calibrator(CalibrationConfig(duration_s=1.0, max_attempts=2), default_ear_threshold=0.21)
    _feed(cal, 3.0, _signals(ear=0.11))
    assert cal.done and not cal.result.calibrated
    assert cal.result.ear_threshold == 0.21
    assert "calibration failed" in cal.note


def test_no_face_falls_back_to_defaults():
    cal = Calibrator(CalibrationConfig(duration_s=1.0), default_ear_threshold=0.21)
    _feed(cal, 3.5, None)
    assert cal.done and not cal.result.calibrated
    assert cal.result.ear_threshold == 0.21


def test_disabled_calibration_is_done_immediately():
    cal = Calibrator(CalibrationConfig(enabled=False), default_ear_threshold=0.2)
    assert cal.done and cal.result.ear_threshold == 0.2
