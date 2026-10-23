"""Train and evaluate the learned fusion model against static and adaptive fusion.

Features come from the per-frame signals saved by ``tools/run_evaluation.py``
(no video is re-processed), computed by ``drowsiness.learned.FeatureTracker``,
the same code the live system would use.

1. Leave-one-subject-out on NTHU-DDD: train on three subjects, test on the
   fourth; static, adaptive and learned are compared on the same frames.
2. External check: a model trained on all of NTHU-DDD is tested on datasets it
   never saw (UTA-RLDD, YawDD awake clips, NITYMED night videos).

    python tools/learned_fusion.py [--save models/learned_fusion.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import replay

from drowsiness.config import AppConfig
from drowsiness.evaluation import binary_metrics, replay_alerts, roc_auc
from drowsiness.learned import FEATURES, FeatureTracker, LogisticModel

CFG = AppConfig()


def features(df: pd.DataFrame, config: AppConfig = CFG) -> np.ndarray:
    """Feature matrix for one recording, frame by frame, causally."""
    threshold, pitch0, _ = df.attrs.get("baseline") or replay.baseline(df, config)
    open_ear = threshold / config.calibration.ear_closed_ratio
    tracker = FeatureTracker()
    smile = df["bs_mouthSmile"].fillna(0).to_numpy()
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        rows.append(
            tracker.update(
                r.timestamp_s,
                face=bool(r.face_detected),
                closed=bool(r.eyes_closed),
                ear_rel=r.ear / open_ear if r.face_detected else float("nan"),
                mar=r.mar if r.face_detected else float("nan"),
                yawn_score=r.yawn_score,
                head_score=r.head_score,
                pitch_rel=r.pitch - pitch0 if r.face_detected else float("nan"),
                smile=smile[i],
                low_light=bool(r.low_light),
                eye_occluded=bool(r.eye_occluded),
            )
        )
    return np.vstack(rows)


def level_config(threshold: float):
    """Alert thresholds for a probability score: warning at ``threshold``."""
    l2 = min(threshold + 0.2, 0.9)
    return dataclasses.replace(
        CFG.alerts,
        level1_threshold=threshold,
        level2_threshold=l2,
        level3_threshold=min(l2 + 0.08, 0.98),
    )


def alerts(df: pd.DataFrame, score: np.ndarray, sleep_col: str, config=CFG.alerts) -> np.ndarray:
    return replay_alerts(
        df["timestamp_s"],
        score,
        df["face_detected"].astype(bool),
        df[sleep_col].astype(bool),
        df["calibrating"].astype(bool),
        config,
    )


def learned_alerts(df: pd.DataFrame, model: LogisticModel, threshold: float) -> np.ndarray:
    return alerts(df, model.predict(df.attrs["x"]), "microsleep_adaptive", level_config(threshold))


def choose_threshold(model: LogisticModel, recs: list[pd.DataFrame]) -> float:
    """Warning threshold with the best warning-level F1 on these (training) recordings."""
    y = np.concatenate([df["drowsy"].to_numpy() for df in recs])

    def f1(threshold: float) -> float:
        pred = np.concatenate([learned_alerts(df, model, threshold) >= 1 for df in recs])
        return binary_metrics(y, pred)["f1"]

    return float(max(np.linspace(0.2, 0.8, 25), key=f1))


def load(dataset: str) -> list[pd.DataFrame]:
    recs = replay.load(dataset)
    if dataset == "nthu":
        replay.with_nthu_baselines(recs, CFG)
    for df in recs:
        df.attrs["x"] = features(df)
    return recs


def nthu_loso(recs: list[pd.DataFrame], l2: float) -> pd.DataFrame:
    subjects = sorted({str(df["subject"].iloc[0]).zfill(3) for df in recs})
    rows, pooled = [], {"y": [], "static": [], "adaptive": [], "learned": [], "p": []}
    for held in subjects:
        train = [df for df in recs if str(df["subject"].iloc[0]).zfill(3) != held]
        test = [df for df in recs if str(df["subject"].iloc[0]).zfill(3) == held]
        x = np.vstack([df.attrs["x"] for df in train])
        y = np.concatenate([df["drowsy"].to_numpy() for df in train])
        model = LogisticModel.fit(x, y, l2=l2)
        thr = choose_threshold(model, train)  # on the training subjects only
        row = {"held-out subject": held, "threshold": round(float(thr), 3)}
        y_test = np.concatenate([df["drowsy"].to_numpy() for df in test])
        for name in ("static", "adaptive", "learned"):
            if name == "learned":
                pred = [learned_alerts(df, model, thr) >= 1 for df in test]
            else:
                pred = [
                    alerts(df, df[f"{name}_score"].to_numpy(), f"microsleep_{name}") >= 1
                    for df in test
                ]
            pred = np.concatenate(pred)
            m = binary_metrics(y_test == 1, pred)
            row[f"F1 {name}"], row[f"FA {name}"] = m["f1"], m["false_alarm_rate"]
            pooled[name].append(pred)
        p_test = np.concatenate([model.predict(df.attrs["x"]) for df in test])
        row["AUC learned"] = roc_auc(y_test == 1, p_test)
        row["AUC adaptive"] = roc_auc(
            y_test == 1, np.concatenate([df["adaptive_score"] for df in test])
        )
        pooled["y"].append(y_test == 1)
        pooled["p"].append(p_test)
        rows.append(row)
    out = pd.DataFrame(rows)
    y = np.concatenate(pooled["y"])
    summary = {"held-out subject": "pooled", "threshold": np.nan}
    for name in ("static", "adaptive", "learned"):
        m = binary_metrics(y, np.concatenate(pooled[name]))
        summary[f"F1 {name}"], summary[f"FA {name}"] = m["f1"], m["false_alarm_rate"]
        summary[f"recall {name}"] = m["recall"]
    summary["AUC learned"] = roc_auc(y, np.concatenate(pooled["p"]))
    summary["AUC adaptive"] = np.nan
    return pd.concat([out, pd.DataFrame([summary])], ignore_index=True)


def external(model: LogisticModel, thr: float) -> pd.DataFrame:
    rows = []
    # UTA-RLDD: drowsy (10) vs alert (0) videos, mean score
    uta = load("uta")
    lab, means = [], {"static": [], "adaptive": [], "learned": []}
    for df in uta:
        if int(df["label"].iloc[0]) not in (0, 10):
            continue
        live = df["calibrating"].to_numpy() == 0
        lab.append(int(df["label"].iloc[0]) == 10)
        means["static"].append(df["static_score"].to_numpy()[live].mean())
        means["adaptive"].append(df["adaptive_score"].to_numpy()[live].mean())
        means["learned"].append(model.predict(df.attrs["x"])[live].mean())
    rows.append(
        {
            "check": "UTA-RLDD AUC drowsy vs alert (mean score)",
            **{k: round(roc_auc(lab, v), 3) for k, v in means.items()},
        }
    )
    # YawDD awake clips with a warning
    warned = {"static": [], "adaptive": [], "learned": []}
    for df in load("yawdd"):
        if df["action"].iloc[0] not in ("normal", "talking"):
            continue
        live = df["calibrating"].to_numpy() == 0
        for name in ("static", "adaptive"):
            warned[name].append(
                (alerts(df, df[f"{name}_score"].to_numpy(), f"microsleep_{name}")[live] >= 1).any()
            )
        warned["learned"].append(
            (
                alerts(df, model.predict(df.attrs["x"]), "microsleep_adaptive", level_config(thr))[
                    live
                ]
                >= 1
            ).any()
        )
    rows.append(
        {
            "check": "YawDD awake clips warned (lower is better)",
            **{k: round(float(np.mean(v)), 3) for k, v in warned.items()},
        }
    )
    # NITYMED microsleep videos with an alarm
    alarmed = {"static": [], "adaptive": [], "learned": []}
    for df in load("nitymed"):
        if df["category"].iloc[0] != "microsleep":
            continue
        for name in ("static", "adaptive"):
            alarmed[name].append(
                (alerts(df, df[f"{name}_score"].to_numpy(), f"microsleep_{name}") >= 2).any()
            )
        alarmed["learned"].append(
            (
                alerts(df, model.predict(df.attrs["x"]), "microsleep_adaptive", level_config(thr))
                >= 2
            ).any()
        )
    rows.append(
        {
            "check": "NITYMED microsleep videos alarmed",
            **{k: round(float(np.mean(v)), 3) for k, v in alarmed.items()},
        }
    )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--save", type=Path, help="save the model trained on all of NTHU-DDD")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    nthu = load("nthu")
    print("Leave-one-subject-out on NTHU-DDD (warning level):")
    print(nthu_loso(nthu, args.l2).round(3).to_string(index=False), "\n")

    x = np.vstack([df.attrs["x"] for df in nthu])
    y = np.concatenate([df["drowsy"].to_numpy() for df in nthu])
    model = LogisticModel.fit(x, y, l2=args.l2)
    thr = choose_threshold(model, nthu)
    print("Model trained on all of NTHU-DDD; warning threshold", round(thr, 3))
    print(pd.Series(model.coef, index=FEATURES).round(3).sort_values().to_string(), "\n")
    print("External datasets (never used for training):")
    print(external(model, thr).to_string(index=False))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        model.to_json(args.save)
        print("saved", args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
