import pytest

from drowsiness.alerts import AlertLevel, AlertManager, EventLog
from drowsiness.config import AlertConfig, FusionConfig
from drowsiness.fusion import IndicatorScores, StaticFusion


def test_static_fusion_normalizes_weights():
    fusion = StaticFusion(FusionConfig(eye_weight=2, yawn_weight=1, head_weight=1))
    assert fusion.weights.eye == pytest.approx(0.5)
    result = fusion.fuse(IndicatorScores(eye=1.0, yawn=0.0, head=1.0))
    assert result.score == pytest.approx(0.75)


def test_static_fusion_ignores_zero_weight_signal():
    fusion = StaticFusion(FusionConfig(eye_weight=1, yawn_weight=0, head_weight=0))
    assert fusion.fuse(IndicatorScores(0.3, 1.0, 1.0)).score == pytest.approx(0.3)


def run(manager, scores, start=0.0, dt=0.1, face=True):
    t = start
    events = []
    for score in scores:
        event = manager.update(t, score, face)
        if event:
            events.append(event)
        t += dt
    return events, t


@pytest.fixture
def manager():
    return AlertManager(AlertConfig())


def test_levels_follow_score(manager):
    events, _ = run(manager, [0.0, 0.3, 0.5, 0.7])
    assert [e.level for e in events] == [AlertLevel.WARNING, AlertLevel.ALARM, AlertLevel.CRITICAL]
    assert events[0].previous == AlertLevel.NONE


def test_hysteresis_prevents_flicker(manager):
    events, _ = run(manager, [0.30, 0.23, 0.27, 0.22, 0.26])
    assert [e.level for e in events] == [AlertLevel.WARNING]
    events, _ = run(manager, [0.19], start=1.0)
    assert [e.level for e in events] == [AlertLevel.NONE]


def test_sustained_alarm_escalates(manager):
    events, _ = run(manager, [0.5] * 60)  # 6 s at alarm level
    assert [e.level for e in events] == [AlertLevel.ALARM, AlertLevel.CRITICAL]
    assert "sustained" in events[-1].reason


def test_critical_is_held_after_recovery(manager):
    _, t = run(manager, [0.8])
    events, t = run(manager, [0.0] * 40, start=t)  # 4 s < 5 s hold
    assert manager.level == AlertLevel.CRITICAL and not events
    events, _ = run(manager, [0.0] * 20, start=t)
    assert manager.level == AlertLevel.NONE


def test_missing_face_raises_warning(manager):
    events, _ = run(manager, [0.0] * 40, face=False)  # 4 s without a face
    assert [e.level for e in events] == [AlertLevel.WARNING]
    assert "not visible" in events[0].reason


def test_missing_face_escalates_to_audible_alarms(manager):
    # Head slumped out of view / camera blocked: must not stay at a silent warning.
    events, t = run(manager, [0.0] * 105, face=False)  # 10.5 s without a face
    assert [(e.level, round(e.timestamp)) for e in events] == [
        (AlertLevel.WARNING, 3),
        (AlertLevel.ALARM, 6),
        (AlertLevel.CRITICAL, 10),
    ]
    # Face back and awake: critical is held briefly, then clears.
    run(manager, [0.0] * 40, start=t)
    assert manager.level == AlertLevel.CRITICAL
    run(manager, [0.0] * 20, start=t + 4.0)
    assert manager.level == AlertLevel.NONE


def test_microsleep_forces_alarm_regardless_of_weights(manager):
    # Eyes-only score 0.1 would be "normal", but a microsleep is always an alarm.
    event = manager.update(0.0, 0.1, face_detected=True, microsleep=True)
    assert event.level == AlertLevel.ALARM and "microsleep" in event.reason


def test_microsleep_floor_can_be_disabled():
    manager = AlertManager(AlertConfig(microsleep_alarm=False))
    assert manager.update(0.0, 0.1, face_detected=True, microsleep=True) is None
    assert manager.level == AlertLevel.NONE


def test_event_log_writes_csv(tmp_path, manager):
    log = EventLog(tmp_path)
    for event in run(manager, [0.3, 0.5, 0.0, 0.0])[0]:
        log.record(event)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("wall_time,session_time_s")
    assert len(lines) == 1 + len(log.events) == 4


def test_event_log_survives_unwritable_file_and_retries(tmp_path, manager):
    # Simulates the CSV being locked (e.g. open in Excel): writing raises OSError.
    log = EventLog(tmp_path)
    log.path.mkdir()  # a directory where the file should be -> open() fails
    events, _ = run(manager, [0.3, 0.5])
    for event in events:
        log.record(event)  # must not raise
    assert log.write_failed and len(log.events) == 2

    log.path.rmdir()  # lock released
    log.record(run(manager, [0.0], start=5.0)[0][0])
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert not log.write_failed
    assert len(lines) == 1 + 3  # the two missed events were written too
