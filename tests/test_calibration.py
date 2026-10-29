import pytest

from drowsiness.calibration import Baseline, Calibrator, OpenEyeTracker
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
    assert base.ear_threshold == pytest.approx(0.32 * CalibrationConfig().ear_closed_ratio)
    assert (base.pitch, base.roll) == (pytest.approx(5.0), pytest.approx(-2.0))


def test_default_ratio_tolerates_squinting():
    # 0.60 of the open EAR: a squint or smile at ~70% of open is not "closed".
    cal = Calibrator(CalibrationConfig(duration_s=1.0), default_ear_threshold=0.21)
    _feed(cal, 1.5, _signals(ear=0.36))
    assert cal.result.ear_threshold == pytest.approx(0.216)
    assert cal.result.ear_threshold < 0.36 * 0.70


def test_threshold_is_clamped():
    config = CalibrationConfig(duration_s=1.0, ear_closed_ratio=0.75)
    cal = Calibrator(config, default_ear_threshold=0.21)
    _feed(cal, 1.5, _signals(ear=0.42))  # 0.42 * 0.75 = 0.315 > max 0.30
    assert cal.result.ear_threshold == config.max_ear_threshold


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


def test_calibration_records_the_resting_smile():
    cal = Calibrator(CalibrationConfig(duration_s=1.0, min_samples=5), 0.21)
    signals = FaceSignals(0.3, 0.3, 0.1, HeadPose(0.0, 0.0, 0.0))
    t, result = 0.0, None
    while result is None:
        result = cal.update(t, signals, smile=0.2)
        t += 0.1
    assert result.calibrated and result.smile == 0.2


def _track(tracker, seconds, ear, fps=15, start=0.0):
    t, base = start, tracker.baseline
    for _ in range(round(seconds * fps)):
        base = tracker.update(t, ear)
        t += 1 / fps
    return t, base


def _calibrated(ear=0.25):
    return Baseline(ear_threshold=ear * 0.6, ear=ear, pitch=3.0, calibrated=True)


def test_open_ear_rises_once_eyes_are_seen_wider():
    # Calibrated on drooping eyes (0.25); the eyes later open to 0.30.
    tracker = OpenEyeTracker(CalibrationConfig(rise_window_s=60.0), _calibrated(0.25))
    t, base = _track(tracker, 59.0, 0.30)
    assert base.ear == pytest.approx(0.25)  # not before a full window
    _, base = _track(tracker, 2.0, 0.30, start=t)
    assert base.ear == pytest.approx(0.30)
    assert base.ear_threshold == pytest.approx(0.30 * 0.6)
    assert base.pitch == 3.0 and base.calibrated  # the rest of the baseline is kept


def test_open_ear_never_falls():
    tracker = OpenEyeTracker(CalibrationConfig(rise_window_s=10.0), _calibrated(0.30))
    _, base = _track(tracker, 60.0, 0.18)  # eyes drooping for a minute
    assert base.ear == pytest.approx(0.30)
    assert base.ear_threshold == pytest.approx(0.18)


def test_implausible_open_ear_is_ignored():
    config = CalibrationConfig(rise_window_s=10.0)
    tracker = OpenEyeTracker(config, _calibrated(0.30))
    _, base = _track(tracker, 30.0, config.max_open_ear + 0.05)
    assert base.ear == pytest.approx(0.30)


def test_failed_calibration_is_replaced_by_a_plausible_open_ear():
    tracker = OpenEyeTracker(CalibrationConfig(rise_window_s=10.0), Baseline(ear_threshold=0.21))
    t, base = _track(tracker, 30.0, 0.15)  # eyes still look closed: keep the default
    assert base.ear_threshold == 0.21 and base.ear is None
    _, base = _track(tracker, 15.0, 0.28, start=t)
    assert base.ear == pytest.approx(0.28)
    assert base.ear_threshold == pytest.approx(0.28 * 0.6)


def test_reset_returns_to_the_calibrated_baseline():
    calibrated = _calibrated(0.25)
    tracker = OpenEyeTracker(CalibrationConfig(rise_window_s=10.0), calibrated)
    _track(tracker, 20.0, 0.30)
    assert tracker.baseline.ear == pytest.approx(0.30)
    tracker.reset()
    assert tracker.baseline == calibrated


def test_rising_can_be_switched_off():
    tracker = OpenEyeTracker(
        CalibrationConfig(rise_window_s=10.0, open_ear_rises=False), _calibrated(0.25)
    )
    _, base = _track(tracker, 30.0, 0.32)
    assert base == _calibrated(0.25)
