"""Compare learned-fusion variants with the rule-based adaptive fusion.

Variants differ in regularisation, feature set and training data (NTHU-DDD
alone, or NTHU-DDD plus UTA-RLDD with per-video labels). Every subject is
predicted by a model that never saw them (leave-one-subject-out over 4 NTHU
and 12 UTA subjects). Reported:

- NTHU-DDD: warning-level F1, false-alarm rate and recall per frame
- UTA-RLDD: AUC of drowsy (10) vs alert (0) videos, mean probability
- YawDD awake clips with a warning, NITYMED microsleep videos with an alarm
  (model trained on all of its training data)
- UL-DD (never used for training): AUC of drowsy vs awake sessions (mean
  probability); sessions classified correctly and accuracy and F1 per 4-minute
  window rated KSS >= 7, with thresholds chosen on the other drivers as in
  ``tools/evaluation_report.py``; time warned in awake sessions

    python tools/learned_variants.py
    python tools/learned_variants.py --save models/learned_fusion.json

``--save`` trains the "NTHU only, all features, l2=1" variant on all of
NTHU-DDD and saves it for the dashboard, which shows it next to the rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evaluation_report as er
import learned_fusion as lf

from drowsiness.evaluation import binary_metrics, roc_auc
from drowsiness.fusion.learned import FEATURES, LogisticModel

# Rule-based adaptive fusion, held out the same way (tools/tune_smile.py held_out());
# the video-level columns come from tools/evaluation_report.py.
REFERENCE = {
    "variant": "adaptive rules (current)",
    "NTHU F1": 0.827,
    "NTHU FA": 0.184,
    "NTHU recall": 0.835,
    "UTA AUC": 0.861,
    "YawDD awake warned": 0.127,
    "night microsleep alarmed": 0.947,
    "UL-DD AUC": 0.918,
    "UL-DD sessions": 0.897,
    "UL-DD window acc": 0.782,
    "UL-DD window F1": 0.715,
    "UL-DD awake warned": 0.094,
}
ALL = list(range(len(FEATURES)))
NO_EXPRESSION = [
    i for i, f in enumerate(FEATURES) if f not in ("smile_10s", "low_light", "eye_occluded")
]
INDICATORS = [
    FEATURES.index(f)
    for f in (
        "closed_frac_10s",
        "closed_frac_30s",
        "closure_s",
        "long_closures_30s",
        "yawn_score",
        "head_score",
    )
]


def prepare() -> dict[str, list[pd.DataFrame]]:
    nthu = lf.load("nthu")
    uta = [df for df in lf.load("uta") if int(df["label"].iloc[0]) in (0, 10)]
    for df in nthu:
        df.attrs["subject"] = "N" + str(df["subject"].iloc[0]).zfill(3)
        df.attrs["y"] = df["drowsy"].to_numpy()
    for df in uta:
        df.attrs["subject"] = "U" + str(df["subject"].iloc[0]).zfill(2)
        df.attrs["y"] = np.full(len(df), int(df["label"].iloc[0]) == 10)
    return {
        "nthu": nthu,
        "uta": uta,
        "yawdd": [df for df in lf.load("yawdd") if df["action"].iloc[0] in ("normal", "talking")],
        "nitymed": [df for df in lf.load("nitymed") if df["category"].iloc[0] == "microsleep"],
        "ulldd": with_kss(lf.load("ulldd")),
    }


def with_kss(recs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Each UL-DD frame gets its 4-minute window and that window's KSS rating."""
    kss = er.kss_labels().set_index(["subject", "session", "window"])["kss"]
    for df in recs:
        df["window"] = (df["timestamp_s"] // er.KSS_WINDOW_S).astype(int)
        idx = pd.MultiIndex.from_frame(df[["subject", "session", "window"]])
        df["kss"] = kss.reindex(idx).to_numpy()
    return recs


def ulldd_checks(recs: list[pd.DataFrame], model, cols: list[int], t: float) -> dict:
    drowsy, means, windows, alert_time = [], [], [], []
    for df in recs:
        live = df["calibrating"].to_numpy() == 0
        drowsy.append(df["session"].iloc[0] == "drowsy")
        means.append(model.predict(df.attrs["x"][:, cols])[live].mean())
        # A new frame: df.attrs holds the feature matrix, which concat would compare.
        frames = pd.DataFrame(
            {c: df[c].to_numpy() for c in ("subject", "session", "window", "kss")}
        ).assign(warn=warn(df, model, cols, t) >= 1)[live]
        windows.append(frames.groupby(["subject", "session", "window"]).mean().reset_index())
        alert_time.append(frames["warn"].mean())
    w = pd.concat(windows, ignore_index=True).dropna(subset=["kss"])
    sleepy = w["kss"] >= 7
    pred = er.held_out_predictions(
        w["warn"], sleepy, w["subject"], lambda v: np.quantile(v, np.linspace(0.05, 0.95, 91)), "f1"
    )
    m = binary_metrics(sleepy, pred)
    s = pd.DataFrame(
        {"subject": [df["subject"].iloc[0] for df in recs], "drowsy": drowsy, "t": alert_time}
    )
    sessions = er.held_out_predictions(
        s["t"], s["drowsy"], s["subject"], lambda v: np.linspace(0, 1, 101), "accuracy"
    )
    return {
        "UL-DD AUC": roc_auc(drowsy, means),
        "UL-DD sessions": binary_metrics(s["drowsy"], sessions)["accuracy"],
        "UL-DD window acc": m["accuracy"],
        "UL-DD window F1": m["f1"],
        "UL-DD awake warned": float(s.loc[~s["drowsy"], "t"].mean()),
    }


def fit(train: list[pd.DataFrame], cols: list[int], l2: float) -> LogisticModel:
    """Every subject weighs the same, however many frames it has."""
    frames = {}
    for df in train:
        frames[df.attrs["subject"]] = frames.get(df.attrs["subject"], 0) + len(df)
    x = np.vstack([df.attrs["x"][:, cols] for df in train])
    y = np.concatenate([df.attrs["y"] for df in train])
    w = np.concatenate([np.full(len(df), 1.0 / frames[df.attrs["subject"]]) for df in train])
    w *= len(w) / w.sum()
    return LogisticModel.fit(x, y, l2=l2, weights=w, features=tuple(FEATURES[i] for i in cols))


def warn(df: pd.DataFrame, model: LogisticModel, cols: list[int], threshold: float) -> np.ndarray:
    score = model.predict(df.attrs["x"][:, cols])
    return lf.alerts(df, score, "microsleep_adaptive", lf.level_config(threshold))


def threshold(model: LogisticModel, cols: list[int], recs: list[pd.DataFrame]) -> float:
    y = np.concatenate([df.attrs["y"] for df in recs])

    def f1(t: float) -> float:
        return binary_metrics(y, np.concatenate([warn(df, model, cols, t) >= 1 for df in recs]))[
            "f1"
        ]

    return float(max(np.linspace(0.2, 0.8, 13), key=f1))


def evaluate(data, name: str, cols: list[int], l2: float, use_uta: bool) -> dict:
    nthu, uta = data["nthu"], data["uta"]
    pool = nthu + (uta if use_uta else [])
    y_nthu, pred_nthu, uta_labels, uta_means = [], [], [], []
    for held in sorted({df.attrs["subject"] for df in pool}):
        train = [df for df in pool if df.attrs["subject"] != held]
        model = fit(train, cols, l2)
        if held.startswith("N"):
            t = threshold(model, cols, [df for df in train if df.attrs["subject"].startswith("N")])
            for df in (d for d in nthu if d.attrs["subject"] == held):
                y_nthu.append(df.attrs["y"])
                pred_nthu.append(warn(df, model, cols, t) >= 1)
        else:
            for df in (d for d in uta if d.attrs["subject"] == held):
                live = df["calibrating"].to_numpy() == 0
                uta_labels.append(bool(df.attrs["y"][0]))
                uta_means.append(model.predict(df.attrs["x"][:, cols])[live].mean())
    m = binary_metrics(np.concatenate(y_nthu), np.concatenate(pred_nthu))
    model = fit(pool, cols, l2)
    t = threshold(model, cols, nthu)
    if not use_uta:  # UTA was never seen: test the final model on it
        for df in uta:
            live = df["calibrating"].to_numpy() == 0
            uta_labels.append(bool(df.attrs["y"][0]))
            uta_means.append(model.predict(df.attrs["x"][:, cols])[live].mean())
    awake = [
        (warn(df, model, cols, t)[df["calibrating"].to_numpy() == 0] >= 1).any()
        for df in data["yawdd"]
    ]
    night = [(warn(df, model, cols, t) >= 2).any() for df in data["nitymed"]]
    return {
        "variant": name,
        "NTHU F1": m["f1"],
        "NTHU FA": m["false_alarm_rate"],
        "NTHU recall": m["recall"],
        "UTA AUC": roc_auc(uta_labels, uta_means),
        "YawDD awake warned": float(np.mean(awake)),
        "night microsleep alarmed": float(np.mean(night)),
        **ulldd_checks(data["ulldd"], model, cols, t),
    }


def save(path: Path) -> None:
    """Train the dashboard's model: NTHU-DDD only, all features, l2=1."""
    nthu = lf.load("nthu")
    for df in nthu:
        df.attrs["subject"] = "N" + str(df["subject"].iloc[0]).zfill(3)
        df.attrs["y"] = df["drowsy"].to_numpy()
    model = fit(nthu, ALL, 1.0)
    model.threshold = threshold(model, ALL, nthu)
    y = np.concatenate([df.attrs["y"] for df in nthu])
    subjects = len({df.attrs["subject"] for df in nthu})
    model.trained_on = (
        f"NTHU-DDD, {len(y):,} labelled frames ({int(y.sum()):,} drowsy) of "
        f"{subjects} subjects, each subject weighted equally"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to_json(path)
    print(pd.Series(model.coef, index=FEATURES).round(3).sort_values().to_string())
    print("threshold", round(model.threshold, 3), "- saved", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, help="train the dashboard's model and save it")
    args = parser.parse_args(argv)
    if args.save:
        save(args.save)
        return 0
    data = prepare()
    rows = [REFERENCE]
    for name, cols, l2, use_uta in (
        ("NTHU only, all features, l2=1", ALL, 1, False),
        ("NTHU only, all features, l2=100", ALL, 100, False),
        ("NTHU only, no smile/conditions, l2=10", NO_EXPRESSION, 10, False),
        ("NTHU only, indicator features, l2=10", INDICATORS, 10, False),
        ("NTHU+UTA, all features, l2=10", ALL, 10, True),
        ("NTHU+UTA, no smile/conditions, l2=10", NO_EXPRESSION, 10, True),
        ("NTHU+UTA, indicator features, l2=10", INDICATORS, 10, True),
    ):
        rows.append(evaluate(data, name, cols, l2, use_uta))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
