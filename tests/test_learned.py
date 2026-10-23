import json

import numpy as np
import pytest

from drowsiness.learned import FEATURES, FeatureTracker, LogisticModel

FPS = 10


def _feed(tracker, seconds, start, **signals):
    base = dict(
        face=True,
        closed=False,
        ear_rel=1.0,
        mar=0.1,
        yawn_score=0.0,
        head_score=0.0,
        pitch_rel=0.0,
        smile=0.0,
        low_light=False,
        eye_occluded=False,
    )
    base.update(signals)
    t, x = start, None
    for _ in range(round(seconds * FPS)):
        x = tracker.update(t, **base)
        t += 1 / FPS
    return dict(zip(FEATURES, x, strict=True)), t


def test_features_open_eyes():
    f, _ = _feed(FeatureTracker(), 5.0, 0.0)
    assert f["closed_frac_10s"] == 0 and f["closure_s"] == 0 and f["blinks_30s"] == 0
    assert f["ear_rel_10s"] == pytest.approx(1.0)


def test_features_track_a_closure():
    tracker = FeatureTracker()
    _, t = _feed(tracker, 5.0, 0.0)
    f, t = _feed(tracker, 2.0, t, closed=True, ear_rel=0.4)
    assert f["closure_s"] == pytest.approx(1.9, abs=0.15)
    assert f["blinks_30s"] == 1 and f["long_closures_30s"] == 1
    assert 0.25 < f["closed_frac_10s"] < 0.35
    f, _ = _feed(tracker, 1.0, t)
    assert f["closure_s"] == 0 and f["blinks_30s"] == 1


def test_closure_is_capped_and_counts_age_out():
    tracker = FeatureTracker()
    f, t = _feed(tracker, 10.0, 0.0, closed=True)
    assert f["closure_s"] == pytest.approx(3.0)
    f, _ = _feed(tracker, 31.0, t)
    assert f["blinks_30s"] == 0 and f["long_closures_30s"] == 0


def test_face_loss_ends_the_closure():
    tracker = FeatureTracker()
    _, t = _feed(tracker, 2.0, 0.0, closed=True)
    f, _ = _feed(tracker, 1.0, t, face=False)
    assert f["closure_s"] == 0


def test_logistic_regression_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, len(FEATURES)))
    true = np.zeros(len(FEATURES))
    true[0], true[2] = 2.0, -1.5
    y = rng.random(2000) < 1 / (1 + np.exp(-(x @ true)))
    model = LogisticModel.fit(x, y, l2=0.1)
    assert model.coef[0] > 1.5 and model.coef[2] < -1.0
    assert np.abs(model.coef[[1, 3, 4]]).max() < 0.3
    p = model.predict(x)
    assert ((p > 0.5) == y).mean() > 0.75

    path = tmp_path / "model.json"
    model.to_json(path)
    again = LogisticModel.from_json(path)
    assert np.allclose(again.predict(x), p)

    data = json.loads(path.read_text())
    data["features"] = data["features"][:-1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different features"):
        LogisticModel.from_json(path)
