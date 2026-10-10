import pytest

from drowsiness.indicators import EyeClosureIndicator, HeadTiltIndicator, YawnIndicator

FPS = 30
DT = 1 / FPS


def _feed(indicator, start: float, seconds: float, *values):
    state = None
    t = start
    for _ in range(round(seconds * FPS)):
        state = indicator.update(t, *values)
        t += DT
    return state, t


def eye_indicator():
    return EyeClosureIndicator(window_s=20.0, perclos_max=0.4, microsleep_s=1.5)


def test_open_eyes_score_zero():
    state, _ = _feed(eye_indicator(), 0.0, 5.0, False)
    assert state.score == 0.0 and state.perclos == 0.0


def test_normal_blink_scores_low():
    ind = eye_indicator()
    _, t = _feed(ind, 0.0, 3.0, False)
    state, _ = _feed(ind, t, 0.2, True)
    assert 0 < state.score < 0.2


def test_microsleep_reaches_full_score():
    ind = eye_indicator()
    _, t = _feed(ind, 0.0, 3.0, False)
    state, _ = _feed(ind, t, 1.6, True)
    assert state.score == 1.0
    assert state.closed_duration_s == pytest.approx(1.6, abs=0.05)


def test_single_open_frame_does_not_reset_microsleep_timer():
    # A noisy EAR frame above the threshold in the middle of a closure.
    ind = EyeClosureIndicator(window_s=20.0, perclos_max=0.4, microsleep_s=1.5, debounce_s=0.15)
    _, t = _feed(ind, 0.0, 1.0, True)
    _, t = _feed(ind, t, DT, False)  # one frame "open"
    state, _ = _feed(ind, t, 0.6, True)
    assert state.closed_duration_s == pytest.approx(1.63, abs=0.05)
    assert state.microsleep and state.score == 1.0


def test_eyes_reopened_beyond_debounce_end_the_closure():
    ind = EyeClosureIndicator(window_s=20.0, perclos_max=0.4, microsleep_s=1.5, debounce_s=0.15)
    _, t = _feed(ind, 0.0, 1.0, True)
    state, _ = _feed(ind, t, 0.3, False)
    assert state.closed_duration_s == 0.0 and not state.microsleep


def test_perclos_over_window():
    ind = eye_indicator()
    t = 0.0
    for _ in range(20):  # 0.5 s closed, 0.5 s open, for 20 s
        _, t = _feed(ind, t, 0.5, True)
        state, t = _feed(ind, t, 0.5, False)
    assert state.perclos == pytest.approx(0.5, abs=0.03)
    assert state.score == 1.0


def test_perclos_forgets_old_closures():
    ind = eye_indicator()
    _, t = _feed(ind, 0.0, 1.0, True)
    state, _ = _feed(ind, t, 25.0, False)
    assert state.perclos == 0.0


def yawn_indicator():
    return YawnIndicator(mar_threshold=0.5, min_duration_s=1.5, window_s=60.0, yawns_for_max=3)


def test_sustained_mouth_opening_counts_one_yawn():
    ind = yawn_indicator()
    state, t = _feed(ind, 0.0, 3.0, 0.8)
    assert state.yawn_count == 1 and state.yawning
    state, _ = _feed(ind, t, 1.0, 0.1)
    assert state.yawn_count == 1 and not state.yawning
    assert state.score == pytest.approx(1 / 3)


def test_short_mouth_opening_is_not_a_yawn():
    state, _ = _feed(yawn_indicator(), 0.0, 1.0, 0.8)
    assert state.yawn_count == 0


def test_yawns_saturate_and_expire():
    ind = yawn_indicator()
    t = 0.0
    for _ in range(4):
        _, t = _feed(ind, t, 2.0, 0.8)
        state, t = _feed(ind, t, 1.0, 0.1)
    assert state.score == 1.0
    state, _ = _feed(ind, t, 61.0, 0.1)
    assert state.yawn_count == 0


def head_indicator():
    return HeadTiltIndicator(pitch_threshold_deg=15, roll_threshold_deg=20, max_duration_s=2.0)


def test_head_nod_score_grows_with_duration():
    ind = head_indicator()
    state, t = _feed(ind, 0.0, 1.0, 25.0, 0.0)
    assert state.tilted and state.score == pytest.approx(0.5, abs=0.05)
    state, _ = _feed(ind, t, 1.5, 25.0, 0.0)
    assert state.score == 1.0


@pytest.mark.parametrize(
    "pitch, roll, tilted", [(10, 0, False), (-30, 0, False), (0, -25, True), (0, 25, True)]
)
def test_head_tilt_directions(pitch, roll, tilted):
    state, _ = _feed(head_indicator(), 0.0, 0.5, pitch, roll)
    assert state.tilted is tilted


def test_head_recovery_resets():
    ind = head_indicator()
    _, t = _feed(ind, 0.0, 3.0, 25.0, 0.0)
    state, _ = _feed(ind, t, 0.1, 0.0, 0.0)
    assert state.score == 0.0
