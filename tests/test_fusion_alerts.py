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


def test_event_log_writes_csv(tmp_path, manager):
    log = EventLog(tmp_path)
    for event in run(manager, [0.3, 0.5, 0.0, 0.0])[0]:
        log.record(event)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("wall_time,session_time_s")
    assert len(lines) == 1 + len(log.events) == 4
