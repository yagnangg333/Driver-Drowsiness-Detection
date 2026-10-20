import math

import numpy as np
import pytest

from drowsiness.alerts import AlertLevel, AlertManager
from drowsiness.config import AlertConfig
from drowsiness.evaluation import binary_metrics, detection_delays, replay_alerts, roc_auc


def test_replay_matches_live_alert_manager():
    rng = np.random.default_rng(0)
    t = np.arange(300) / 10
    scores = np.clip(np.cumsum(rng.normal(0, 0.05, 300)), 0, 1)
    face = rng.random(300) > 0.05
    sleep = rng.random(300) > 0.97
    calib = t < 2.0

    live = AlertManager(AlertConfig())
    expected = [
        0 if c else (live.update(ti, s, f, m), int(live.level))[1]
        for ti, s, f, m, c in zip(t, scores, face, sleep, calib, strict=True)
    ]
    levels = replay_alerts(t, scores, face, sleep, calib, AlertConfig())
    assert levels.tolist() == expected
    assert levels.max() >= AlertLevel.WARNING  # the series does raise alerts


def test_replay_ignores_calibration_frames():
    levels = replay_alerts(
        [0, 1, 2], [0.9, 0.9, 0.0], [1, 1, 1], [0, 0, 0], [1, 1, 0], AlertConfig()
    )
    assert levels.tolist() == [0, 0, 0]


def test_binary_metrics():
    m = binary_metrics([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 3, 1)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == pytest.approx(1 / 4)
    assert m["miss_rate"] == pytest.approx(1 / 3)


def test_binary_metrics_undefined_rates_are_nan():
    m = binary_metrics([0, 0], [0, 0])
    assert m["accuracy"] == 1.0
    assert math.isnan(m["precision"]) and math.isnan(m["recall"])


@pytest.mark.parametrize(
    "labels, scores, expected",
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),  # perfect ranking
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0),  # reversed
        ([0, 1], [0.5, 0.5], 0.5),  # tie
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75),
    ],
)
def test_roc_auc(labels, scores, expected):
    assert roc_auc(labels, scores) == pytest.approx(expected)


def test_roc_auc_needs_both_classes():
    assert math.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_detection_delays():
    t = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    positive = [0, 1, 1, 1, 0, 1, 1, 0, 1]
    alerted = [0, 0, 0, 1, 1, 0, 0, 0, 1]
    # first episode starts at 1, alerted at 3; second never alerted; third at once
    assert detection_delays(t, positive, alerted) == [2.0, None, 0.0]
