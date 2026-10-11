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


# --- debounce: one flickering frame must not reset a timer ------------------


def test_head_tilt_survives_one_upright_frame():
    ind = HeadTiltIndicator(15, 20, 2.0, debounce_s=0.20)
    _, t = _feed(ind, 0.0, 1.0, 25.0, 0.0)
    _, t = _feed(ind, t, DT, 0.0, 0.0)  # one frame of landmark flicker
    state, _ = _feed(ind, t, 1.0, 25.0, 0.0)
    assert state.tilt_duration_s == pytest.approx(2.03, abs=0.05)
    assert state.score == 1.0


def test_head_upright_beyond_debounce_ends_the_tilt():
    ind = HeadTiltIndicator(15, 20, 2.0, debounce_s=0.20)
    _, t = _feed(ind, 0.0, 1.0, 25.0, 0.0)
    state, _ = _feed(ind, t, 0.3, 0.0, 0.0)
    assert state.tilt_duration_s == 0.0 and state.score == 0.0


def test_yawn_survives_one_low_mar_frame():
    ind = YawnIndicator(0.5, 1.5, 60.0, 3, debounce_s=0.20)
    _, t = _feed(ind, 0.0, 1.0, 0.8)
    _, t = _feed(ind, t, DT, 0.3)  # one frame where the lips flicker closed
    state, _ = _feed(ind, t, 0.6, 0.8)
    assert state.yawn_count == 1 and state.yawning


def test_mouth_closed_beyond_debounce_ends_the_yawn():
    ind = YawnIndicator(0.5, 1.5, 60.0, 3, debounce_s=0.20)
    _, t = _feed(ind, 0.0, 1.0, 0.8)
    _, t = _feed(ind, t, 0.3, 0.3)
    state, _ = _feed(ind, t, 0.6, 0.8)  # a new opening, too short to be a yawn
    assert state.yawn_count == 0


# --- face lost: no phantom time, no frozen state -----------------------------


def test_face_lost_ends_closure_without_phantom_time():
    ind = EyeClosureIndicator(window_s=20.0, perclos_max=0.4, microsleep_s=1.5, debounce_s=0.15)
    _, t = _feed(ind, 0.0, 3.0, False)
    _, t = _feed(ind, t, 1.0, True)  # eyes closing...
    state = ind.face_lost(t + 2.0)  # ...then 2 s unobserved
    assert state.closed_duration_s == 0.0 and not state.microsleep
    state, _ = _feed(ind, t + 2.0, 0.3, True)  # back, eyes still closed
    assert state.closed_duration_s == pytest.approx(0.27, abs=0.05)
    assert not state.microsleep and state.score < 1.0


def test_face_lost_keeps_the_closed_ratio_until_it_ages_out():
    ind = EyeClosureIndicator(window_s=20.0, perclos_max=0.4, microsleep_s=1.5)
    _, t = _feed(ind, 0.0, 4.0, True)
    assert ind.face_lost(t).perclos == pytest.approx(0.2, abs=0.01)  # not zeroed...
    assert ind.face_lost(t + 25.0).perclos == 0.0  # ...but not frozen either


def test_gap_after_face_loss_is_not_counted_as_closed():
    ind = EyeClosureIndicator(window_s=20.0, perclos_max=0.4, microsleep_s=1.5)
    _, t = _feed(ind, 0.0, 1.0, False)
    before = ind.face_lost(t + 5.0).perclos
    state = ind.update(t + 5.0, True)  # first frame after the gap
    assert state.perclos == before


def test_face_lost_aborts_yawn_and_tilt_in_progress():
    yawn = YawnIndicator(0.5, 1.5, 60.0, 3, debounce_s=0.20)
    _, t = _feed(yawn, 0.0, 1.0, 0.8)
    yawn.face_lost(t + 1.0)
    state, _ = _feed(yawn, t + 1.0, 1.0, 0.8)  # 1 s more after the loss: not a yawn
    assert state.yawn_count == 0

    head = HeadTiltIndicator(15, 20, 2.0, debounce_s=0.20)
    _, t = _feed(head, 0.0, 1.5, 25.0, 0.0)
    assert head.face_lost(t).score == 0.0
    state, _ = _feed(head, t + 1.0, 0.5, 25.0, 0.0)
    assert state.tilt_duration_s == pytest.approx(0.47, abs=0.05)
