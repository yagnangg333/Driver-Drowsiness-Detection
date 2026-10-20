"""Choose the smile margin (relative to the driver's resting smile).

Candidates are compared leave-one-subject-out on NTHU-DDD (chosen on three
subjects, scored on the fourth), and checked for the safety problem the
margin addresses: drivers whose resting face already reads as smiling, for
whom eye closures would be ignored most of the time.

    python tools/tune_smile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import replay
import tune_detection as td

from drowsiness.config import AppConfig

MARGINS = (None, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30)  # None: absolute threshold only


def settings(margin: float | None) -> replay.Settings:
    if margin is None:  # the previous rule: absolute threshold, no resting smile
        return replay.Settings(
            smiling_rule=lambda df, c: (
                (replay.smile_mean(df, c.conditions.smile_window_s) > c.conditions.smile_threshold)
                & (
                    df.set_index(pd.to_datetime(df["timestamp_s"], unit="s"))["mar"]
                    .fillna(0)
                    .rolling("2000ms")
                    .max()
                    .to_numpy()
                    <= c.indicators.yawn_mar_threshold
                )
            )
        )
    return replay.Settings(
        config=replay.configure(AppConfig(), conditions={"smile_margin": margin})
    )


def safety(s: replay.Settings) -> dict[str, float]:
    """Awake YawDD clips (and night microsleep videos) mostly read as smiling."""
    shares = []
    for df in td.recordings("yawdd"):
        if df["action"].iloc[0] != "normal":
            continue
        live = (df["calibrating"] == 0).to_numpy() & (df["face_detected"] == 1).to_numpy()
        shares.append(replay.simulate(df, s)["smiling"].to_numpy()[live].mean())
    night = []
    for df in td.recordings("nitymed"):
        if df["category"].iloc[0] != "microsleep":
            continue
        live = (df["calibrating"] == 0).to_numpy() & (df["face_detected"] == 1).to_numpy()
        night.append(replay.simulate(df, s)["smiling"].to_numpy()[live].mean())
    return {
        "YawDD normal clips smiling >50%": int(np.sum(np.array(shares) > 0.5)),
        "night microsleep videos smiling >25%": int(np.sum(np.array(night) > 0.25)),
    }


def main() -> int:
    candidates = [td.Candidate(f"margin {m}", settings(m)) for m in MARGINS]
    print("Leave-one-subject-out on NTHU-DDD:")
    res = td.leave_one_subject_out(candidates)
    print(res.round(3).to_string(index=False))
    rows = []
    for m, c in zip(MARGINS, candidates, strict=True):
        n = td.nthu_scores(c.settings)
        rows.append(
            {
                "margin": "none (absolute 0.10)" if m is None else m,
                "NTHU F1": n["f1"],
                "NTHU false alarms": n["false_alarms"],
                "NTHU recall": n["recall"],
                **td.yawdd_scores(c.settings),
                **td.nitymed_scores(c.settings),
                **safety(c.settings),
            }
        )
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print("\nAll datasets:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nNTHU-DDD held out (smile settings chosen without the scored subject):")
    print(held_out().round(3).to_string(index=False))
    return 0


def held_out(thresholds=(0.05, 0.10, 0.15, 0.20), margins=(0.05, 0.10, 0.15)) -> pd.DataFrame:
    """NTHU-DDD results where the smile settings never saw the scored subject.

    For each subject, the smile threshold and margin with the best F1 on the
    other three subjects are used to score it; the predictions are pooled.
    The static baseline has nothing tuned on NTHU-DDD, so it is scored as is.
    """
    from drowsiness.evaluation import binary_metrics, roc_auc

    recs = td.recordings("nthu")
    replay.with_nthu_baselines(recs, AppConfig())
    grid = [(t, m) for t in thresholds for m in margins]

    def sim(t, m):
        return replay.Settings(
            config=replay.configure(
                AppConfig(), conditions={"smile_threshold": t, "smile_margin": m}
            )
        )

    subjects = sorted({str(df["subject"].iloc[0]).zfill(3) for df in recs})
    y, adaptive, static, a_score, s_score = [], [], [], [], []
    for held in subjects:
        train = set(subjects) - {held}
        t, m = max(grid, key=lambda g: td.nthu_scores(sim(*g), train)["f1"])
        for df in recs:
            if str(df["subject"].iloc[0]).zfill(3) != held:
                continue
            out = replay.simulate(df, sim(t, m))
            y.append(df["drowsy"].to_numpy() == 1)
            adaptive.append(out["adaptive_level"].to_numpy() >= 1)
            static.append(out["static_level"].to_numpy() >= 1)
            a_score.append(out["adaptive_score"].to_numpy())
            s_score.append(out["static_score"].to_numpy())
    y = np.concatenate(y)
    rows = []
    for name, pred, score in (("static", static, s_score), ("adaptive", adaptive, a_score)):
        mm = binary_metrics(y, np.concatenate(pred))
        rows.append(
            {
                "fusion": name,
                "F1": mm["f1"],
                "false alarms": mm["false_alarm_rate"],
                "recall": mm["recall"],
                "precision": mm["precision"],
                "ROC-AUC": roc_auc(y, np.concatenate(score)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
