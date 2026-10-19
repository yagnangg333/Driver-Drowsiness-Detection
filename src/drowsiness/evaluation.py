"""Evaluation helpers: replaying alerts and scoring predictions against labels.

The pipeline computes the static and the adaptive fused score on every frame
from the same indicator scores. ``replay_alerts`` runs the alert state machine
over either score series afterwards, so one pass over a video yields the alert
levels of both strategies, compared on exactly the same input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drowsiness.alerts import AlertManager
from drowsiness.config import AlertConfig


def replay_alerts(
    timestamps: Sequence[float],
    scores: Sequence[float],
    face_detected: Sequence[bool],
    microsleep: Sequence[bool],
    calibrating: Sequence[bool],
    config: AlertConfig,
) -> np.ndarray:
    """Alert level per frame, as the pipeline would have raised it for ``scores``.

    ``calibrating`` frames get level 0 and are not fed to the alert manager,
    exactly as in the live pipeline.
    """
    manager = AlertManager(config)
    levels = np.zeros(len(timestamps), dtype=np.int8)
    for i, (t, score, face, sleep, calib) in enumerate(
        zip(timestamps, scores, face_detected, microsleep, calibrating, strict=True)
    ):
        if calib:
            continue
        manager.update(float(t), float(score), bool(face), bool(sleep))
        levels[i] = int(manager.level)
    return levels


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """Confusion counts and the usual rates; a rate is NaN when undefined."""
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    tn = int(np.sum(~t & ~p))
    fn = int(np.sum(t & ~p))

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "n": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": ratio(fp, fp + tn),
        "miss_rate": ratio(fn, fn + tp),
    }


def roc_auc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the ROC curve: the probability that a random positive scores
    higher than a random negative (ties count half). NaN without both classes."""
    t = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U from average ranks (ties get the mean of their ranks).
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[t].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def detection_delays(
    timestamps: Sequence[float], positive: Sequence[bool], alerted: Sequence[bool]
) -> list[float | None]:
    """For each run of consecutive positive frames, the time from its first
    frame to the first alerted frame within it; None if it was never alerted."""
    t = np.asarray(timestamps, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    alert = np.asarray(alerted, dtype=bool)
    delays: list[float | None] = []
    i = 0
    while i < len(pos):
        if not pos[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(pos) and pos[j + 1]:
            j += 1
        hits = np.flatnonzero(alert[i : j + 1])
        delays.append(float(t[i + hits[0]] - t[i]) if hits.size else None)
        i = j + 1
    return delays
