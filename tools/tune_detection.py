"""Score detection settings on every dataset with the offline simulator.

Each candidate setting (thresholds, extra closure rules, yawn signal) is
simulated with ``tools/replay.py`` and scored for the adaptive strategy:

- NTHU-DDD: per-frame F1, recall and false-alarm rate at warning level,
  leave-one-subject-out when tuning (choose on 3 subjects, test on the 4th)
- YawDD: share of awake clips (normal / talking) with a warning, and of
  yawning clips with a detected yawn and with a false yawn in talking clips
- UTA-RLDD: AUC of the mean score, drowsy (10) vs alert (0) videos
- NITYMED: share of night microsleep videos raising an alarm

    python tools/tune_detection.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import replay

from drowsiness.config import AppConfig
from drowsiness.evaluation import binary_metrics, roc_auc

_CACHE: dict[str, list[pd.DataFrame]] = {}


def recordings(dataset: str) -> list[pd.DataFrame]:
    if dataset not in _CACHE:
        recs = replay.load(dataset)
        if dataset == "nthu":
            replay.with_nthu_baselines(recs, AppConfig())
        _CACHE[dataset] = recs
    return _CACHE[dataset]


def nthu_scores(settings: replay.Settings, subjects: set[str] | None = None) -> dict[str, float]:
    recs = recordings("nthu")
    replay.with_nthu_baselines(recs, settings.config)
    y, p = [], []
    for df in recs:
        if subjects is not None and df["subject"].iloc[0].__str__().zfill(3) not in subjects:
            continue
        sim = replay.simulate(df, settings)
        y.append(df["drowsy"].to_numpy() == 1)
        p.append(sim["adaptive_level"].to_numpy() >= 1)
    m = binary_metrics(np.concatenate(y), np.concatenate(p))
    return {"f1": m["f1"], "recall": m["recall"], "false_alarms": m["false_alarm_rate"]}


def yawdd_scores(settings: replay.Settings) -> dict[str, float]:
    warned_awake, yawn_found, talk_yawn = [], [], []
    for df in recordings("yawdd"):
        sim = replay.simulate(df, settings)
        live = df["calibrating"].to_numpy() == 0
        action = df["action"].iloc[0]
        if action in ("normal", "talking"):
            warned_awake.append(bool((sim["adaptive_level"].to_numpy()[live] >= 1).any()))
        y = sim["yawning"].to_numpy()[live]
        if action == "yawning":
            yawn_found.append(bool(y.any()))
        if action == "talking":
            talk_yawn.append(bool(y.any()))
    return {
        "awake_clips_warned": float(np.mean(warned_awake)),
        "yawn_recall": float(np.mean(yawn_found)),
        "talking_false_yawn": float(np.mean(talk_yawn)),
    }


def uta_scores(settings: replay.Settings) -> dict[str, float]:
    labels, means = [], []
    for df in recordings("uta"):
        label = int(df["label"].iloc[0])
        if label not in (0, 10):
            continue
        sim = replay.simulate(df, settings)
        live = df["calibrating"].to_numpy() == 0
        labels.append(label == 10)
        means.append(float(sim["adaptive_score"].to_numpy()[live].mean()))
    return {"uta_auc": roc_auc(labels, means)}


def nitymed_scores(settings: replay.Settings) -> dict[str, float]:
    alarmed = []
    for df in recordings("nitymed"):
        if df["category"].iloc[0] != "microsleep":
            continue
        sim = replay.simulate(df, settings)
        alarmed.append(bool((sim["adaptive_level"].to_numpy() >= 2).any()))
    return {"night_microsleep_alarmed": float(np.mean(alarmed))}


def everything(settings: replay.Settings) -> dict[str, float]:
    return {
        **nthu_scores(settings),
        **yawdd_scores(settings),
        **uta_scores(settings),
        **nitymed_scores(settings),
    }


@dataclass
class Candidate:
    name: str
    settings: replay.Settings


def leave_one_subject_out(candidates: list[Candidate]) -> pd.DataFrame:
    """For each NTHU subject: pick the best candidate (F1) on the other three,
    report how that pick does on the held-out subject."""
    subjects = sorted({str(df["subject"].iloc[0]).zfill(3) for df in recordings("nthu")})
    rows = []
    for held in subjects:
        train = set(subjects) - {held}
        best = max(candidates, key=lambda c: nthu_scores(c.settings, train)["f1"])
        test = nthu_scores(best.settings, {held})
        base = nthu_scores(candidates[0].settings, {held})
        rows.append(
            {
                "held-out subject": held,
                "chosen": best.name,
                "F1 chosen": test["f1"],
                "F1 default": base["f1"],
                "false alarms chosen": test["false_alarms"],
                "false alarms default": base["false_alarms"],
            }
        )
    return pd.DataFrame(rows)
