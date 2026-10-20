import pytest

from drowsiness.alerts import AlertLevel, AlertManager, EventLog
from drowsiness.conditions import NORMAL, Conditions
from drowsiness.config import AlertConfig, FusionConfig
from drowsiness.fusion import AdaptiveFusion, IndicatorScores, StaticFusion


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
    # Clearly below: back to normal once the 2 s warning hold has passed.
    events, _ = run(manager, [0.19] * 25, start=1.0)
    assert [e.level for e in events] == [AlertLevel.NONE]
    assert events[0].timestamp >= 0.4 + 2.0  # last warning-level score at 0.4 s


def test_warning_is_held_and_does_not_flicker(manager):
    # A head-tilt score that ends abruptly (as tilt scores do): the warning
    # must not blink on and off every half second.
    events, t = run(manager, [0.30] * 3 + [0.0] * 5 + [0.30] * 3 + [0.0] * 15)
    assert [e.level for e in events] == [AlertLevel.WARNING]
    events, _ = run(manager, [0.0] * 10, start=t)
    assert [e.level for e in events] == [AlertLevel.NONE]


def test_alarm_steps_down_through_warning(manager):
    events, _ = run(manager, [0.5] + [0.0] * 25)
    assert [e.level for e in events] == [AlertLevel.ALARM, AlertLevel.WARNING, AlertLevel.NONE]
    assert [e.reason for e in events[1:]] == ["warning hold", "score back to normal"]


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


def test_alarm_is_held_and_does_not_stutter(manager):
    # Score flickers around the alarm threshold: the sound must stay on, not
    # switch on and off every frame.
    events, t = run(manager, [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0])
    assert [e.level for e in events] == [AlertLevel.ALARM]
    run(manager, [0.0] * 14, start=t)  # last alarm at 0.4 s, so the hold ends at 1.9 s
    assert manager.level == AlertLevel.WARNING  # stepping down, held until 2.4 s


def test_alarm_hold_ends_after_cause_clears(manager):
    _, t = run(manager, [0.5])
    run(manager, [0.0] * 14, start=t)  # 1.4 s < 1.5 s hold
    assert manager.level == AlertLevel.ALARM
    run(manager, [0.0] * 3, start=t + 1.4)
    assert manager.level == AlertLevel.WARNING
    run(manager, [0.0] * 5, start=t + 1.7)  # past the 2 s warning hold
    assert manager.level == AlertLevel.NONE


def test_level_never_drops_while_face_is_missing(manager):
    # Eyes closing (alarm), then the head drops out of view: the scores the
    # pipeline reports may fall, but the alert must not.
    _, t = run(manager, [0.5] * 5)
    events, _ = run(manager, [0.0] * 30, start=t, face=False)  # 3 s, past the 1.5 s hold
    assert manager.level >= AlertLevel.ALARM
    assert all(e.level >= AlertLevel.ALARM for e in events)


def test_single_missing_frame_does_not_change_level(manager):
    _, t = run(manager, [0.5] * 5)
    events, t = run(manager, [0.0], start=t, face=False)
    events += run(manager, [0.5] * 5, start=t)[0]
    assert events == [] and manager.level == AlertLevel.ALARM


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
    # Warning, alarm, back down through a warning, then normal.
    for event in run(manager, [0.3, 0.5] + [0.0] * 25)[0]:
        log.record(event)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("wall_time,session_time_s")
    assert len(lines) == 1 + len(log.events) == 5


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


# --- adaptive fusion ---

LOW_LIGHT = Conditions(low_light=True)
OCCLUDED = Conditions(eye_occluded=True)


def test_adaptive_equals_static_in_normal_conditions():
    config = FusionConfig()
    static, adaptive = StaticFusion(config), AdaptiveFusion(config)
    scores = IndicatorScores(0.8, 0.3, 0.5)
    for t in (0.0, 0.1, 0.2):
        assert adaptive.fuse(scores, NORMAL, t) == static.fuse(scores)


def test_low_light_keeps_normal_weights_by_default():
    # Real night footage showed the eye cue stays reliable in low light.
    fusion = AdaptiveFusion(FusionConfig(transition_s=0))
    result = fusion.fuse(IndicatorScores(1.0, 0.0, 0.0), LOW_LIGHT, 0.0)
    assert result.weights == StaticFusion(FusionConfig()).weights


def test_adaptive_weights_follow_condition_table():
    fusion = AdaptiveFusion(FusionConfig(transition_s=0))
    result = fusion.fuse(IndicatorScores(1.0, 0.0, 0.0), OCCLUDED, 0.0)
    assert result.weights.eye == pytest.approx(0.20)
    assert result.weights.yawn == pytest.approx(0.35)
    assert result.weights.head == pytest.approx(0.45)
    assert result.score == pytest.approx(0.20)


def test_adaptive_both_conditions_take_lowest_weight_per_cue():
    config = FusionConfig(
        adapt_to_low_light=True,
        low_light_eye_weight=0.4,
        low_light_yawn_weight=0.2,
        low_light_head_weight=0.4,
        occluded_eye_weight=0.1,
        occluded_yawn_weight=0.4,
        occluded_head_weight=0.5,
    )
    w = AdaptiveFusion(config).target(Conditions(low_light=True, eye_occluded=True))
    # min per cue: eye 0.1, yawn 0.2, head 0.4 -> normalised by 0.7
    assert (w.eye, w.yawn, w.head) == pytest.approx((0.1 / 0.7, 0.2 / 0.7, 0.4 / 0.7))


def test_adaptive_weights_change_gradually():
    fusion = AdaptiveFusion(FusionConfig(transition_s=1.0))
    scores = IndicatorScores(1.0, 0.0, 0.0)
    assert fusion.fuse(scores, NORMAL, 0.0).weights.eye == pytest.approx(0.5)
    halfway = fusion.fuse(scores, OCCLUDED, 0.15).weights
    assert 0.2 < halfway.eye < 0.5
    assert halfway.eye + halfway.yawn + halfway.head == pytest.approx(1.0)
    assert fusion.fuse(scores, OCCLUDED, 1.2).weights.eye == pytest.approx(0.2)
    fusion.reset()
    assert fusion.fuse(scores, OCCLUDED, 2.0).weights.eye == pytest.approx(0.2)


def test_low_light_follows_changed_normal_weights():
    # Changing the normal weights (e.g. with the dashboard sliders) must not
    # make low light fall back to some other weights.
    config = FusionConfig(eye_weight=0.7, yawn_weight=0.1, head_weight=0.2, transition_s=0)
    fusion = AdaptiveFusion(config)
    assert fusion.target(LOW_LIGHT) == fusion.target(NORMAL) == StaticFusion(config).weights
