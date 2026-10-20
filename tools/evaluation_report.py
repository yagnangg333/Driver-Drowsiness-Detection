"""Turn ``tools/run_evaluation.py`` output into the Phase 4 results.

Static and adaptive fusion are compared on exactly the same frames: both
scores were computed in the same pipeline pass, and the alert state machine is
replayed over each. Results go to stdout and ``data/eval/results.md``.

    python tools/evaluation_report.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from drowsiness.config import PROJECT_ROOT, AppConfig
from drowsiness.evaluation import binary_metrics, detection_delays, replay_alerts, roc_auc

EVAL = PROJECT_ROOT / "data" / "eval"
STRATEGIES = ("static", "adaptive")
ALERTS = AppConfig().alerts


def load(dataset: str) -> pd.DataFrame:
    """All recordings of a dataset with alert levels replayed for both strategies."""
    frames = []
    for path in sorted(glob.glob(str(EVAL / dataset / "*.csv"))):
        df = pd.read_csv(path)
        df["recording"] = Path(path).stem
        keys = ["behaviour"] if "behaviour" in df else [None]
        parts = [df] if keys == [None] else [g for _, g in df.groupby("behaviour", sort=False)]
        for part in parts:
            for s in STRATEGIES:
                part[f"{s}_level"] = replay_alerts(
                    part["timestamp_s"],
                    part[f"{s}_score"],
                    part["face_detected"].astype(bool),
                    part[f"microsleep_{s}" if f"microsleep_{s}" in part else "microsleep"].astype(
                        bool
                    ),
                    part["calibrating"].astype(bool),
                    ALERTS,
                )
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Sanity check: the pipeline's own alerts came from the adaptive strategy.
    mismatch = (df["alert_level"] != df["adaptive_level"]).mean()
    if mismatch > 0.001:
        print(f"WARNING: replayed adaptive alerts differ from the live ones on {mismatch:.2%}")
    return df


def yawn_count(yawning: pd.Series) -> int:
    y = yawning.astype(int).to_numpy()
    return int(((y[1:] == 1) & (y[:-1] == 0)).sum() + (y[:1] == 1).sum())


def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.1f}%"


def table(df: pd.DataFrame) -> str:
    """A Markdown table (pandas' own needs the optional tabulate package)."""
    if df.empty:
        return "(no data)"
    cells = [[str(c) for c in df.columns]]
    cells += [[str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    lines.insert(1, "|" + "|".join("---" for _ in df.columns) + "|")
    return "\n".join(lines)


def conditions_line(df: pd.DataFrame) -> str:
    d = df[df["calibrating"] == 0]
    face = d[d["face_detected"] == 1]
    return (
        f"{len(d['recording'].unique())} recordings, {len(d)} frames after calibration; "
        f"face found {pct(d['face_detected'].mean())}; low light {pct(face['low_light'].mean())}; "
        f"eyes occluded {pct(face['eye_occluded'].mean())} of face frames; "
        f"processing {d['process_ms'].median():.1f} ms/frame (median)"
    )


# --- NTHU-DDD: per-frame labels ---------------------------------------------


def nthu(df: pd.DataFrame) -> list[str]:
    out = ["## NTHU-DDD (per-frame drowsy / not drowsy labels)", conditions_line(df), ""]
    rows = []
    for subset, d in [("all", df), *[(s, g) for s, g in df.groupby("scenario")]]:
        for s in STRATEGIES:
            for name, level in (("warning", 1), ("alarm", 2)):
                m = binary_metrics(d["drowsy"] == 1, d[f"{s}_level"] >= level)
                rows.append(
                    {
                        "subset": subset,
                        "fusion": s,
                        "alert at": name,
                        "accuracy": pct(m["accuracy"]),
                        "precision": pct(m["precision"]),
                        "recall": pct(m["recall"]),
                        "F1": pct(m["f1"]),
                        "false alarms": pct(m["false_alarm_rate"]),
                        "misses": pct(m["miss_rate"]),
                    }
                )
    out += [table(pd.DataFrame(rows)), ""]

    rows = []
    for s in STRATEGIES:
        delays = []
        for _, g in df.groupby(["recording", "behaviour"]):
            delays += detection_delays(g["clip_time_s"], g["drowsy"] == 1, g[f"{s}_level"] >= 1)
        hit = [d for d in delays if d is not None]
        rows.append(
            {
                "fusion": s,
                "ROC-AUC (fused score)": f"{roc_auc(df['drowsy'] == 1, df[f'{s}_score']):.3f}",
                "drowsy episodes": len(delays),
                "detected": pct(len(hit) / len(delays)) if delays else "-",
                "median delay": f"{np.median(hit):.1f} s" if hit else "-",
            }
        )
    out += [table(pd.DataFrame(rows)), ""]
    return out


# --- UTA-RLDD: per-video labels --------------------------------------------------


def uta(df: pd.DataFrame) -> list[str]:
    out = [
        "## UTA-RLDD (per-video labels: 0 alert, 5 low vigilant, 10 drowsy)",
        conditions_line(df),
        "",
    ]
    d = df[df["calibrating"] == 0]
    per = d.groupby("recording").agg(subject=("subject", "first"), label=("label", "first"))
    for s in STRATEGIES:
        g = d.groupby("recording")
        per[f"{s}_alert_time"] = g[f"{s}_level"].apply(lambda x: (x >= 1).mean())
        per[f"{s}_mean_score"] = g[f"{s}_score"].mean()
    rows = []
    for label, g in per.groupby("label"):
        row = {"label": label, "videos": len(g)}
        for s in STRATEGIES:
            row[f"{s}: time in alert"] = pct(g[f"{s}_alert_time"].mean())
            row[f"{s}: mean score"] = f"{g[f'{s}_mean_score'].mean():.3f}"
        rows.append(row)
    out += [table(pd.DataFrame(rows)), ""]

    rows = []
    pair = per[per["label"].isin([0, 10])]
    drowsy = pair["label"] == 10
    for s in STRATEGIES:
        wide = pair.pivot(index="subject", columns="label", values=f"{s}_mean_score").dropna()
        higher = int((wide[10] > wide[0]).sum())
        rows.append(
            {
                "fusion": s,
                "AUC (time in alert)": f"{roc_auc(drowsy, pair[f'{s}_alert_time']):.3f}",
                "AUC (mean score)": f"{roc_auc(drowsy, pair[f'{s}_mean_score']):.3f}",
                "drowsy video scores higher": f"{higher}/{len(wide)} subjects",
            }
        )
    out += [table(pd.DataFrame(rows)), ""]
    return out


# --- NITYMED: night driving ------------------------------------------------------


def nitymed(df: pd.DataFrame) -> list[str]:
    out = ["## NITYMED (night driving: yawning and microsleep videos)", conditions_line(df), ""]
    d = df[df["calibrating"] == 0]
    rows = []
    for category, g in d.groupby("category"):
        per = g.groupby("recording")
        row = {"videos": category, "n": per.ngroups}
        if category == "yawning":
            counts = per["yawning"].apply(yawn_count)
            row["yawns detected per video (3 performed)"] = f"{counts.mean():.2f}"
            row["videos with >= 1 yawn"] = pct((counts >= 1).mean())
        for s in STRATEGIES:
            row[f"{s}: videos with a warning"] = pct(per[f"{s}_level"].max().ge(1).mean())
            row[f"{s}: videos with an alarm"] = pct(per[f"{s}_level"].max().ge(2).mean())
            row[f"{s}: time in alert"] = pct((g[f"{s}_level"] >= 1).mean())
        rows.append(row)
    wide = pd.DataFrame(rows).set_index("videos").T.reset_index()
    out += [table(wide.rename(columns={"index": "measure"})), ""]
    return out


# --- YawDD: yawning / talking / normal clips -------------------------------------


def yawdd(df: pd.DataFrame) -> list[str]:
    out = ["## YawDD (day driving: normal, talking, yawning clips)", conditions_line(df), ""]
    d = df[df["calibrating"] == 0]
    per = d.groupby("recording").agg(
        action=("action", "first"),
        glasses=("glasses", "first"),
        yawns=("yawning", yawn_count),
        **{f"{s}_warned": (f"{s}_level", lambda x: (x >= 1).any()) for s in STRATEGIES},
    )
    rows = []
    for (action, glasses), g in per.groupby(["action", "glasses"]):
        row = {
            "clips": action,
            "glasses": glasses,
            "n": len(g),
            "yawn detected": pct((g["yawns"] >= 1).mean()),
        }
        for s in STRATEGIES:
            row[f"{s}: clips with a warning"] = pct(g[f"{s}_warned"].mean())
        rows.append(row)
    out += [table(pd.DataFrame(rows)), ""]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EVAL / "results.md")
    args = parser.parse_args(argv)

    lines = ["# Phase 4 results: static vs adaptive fusion", ""]
    for name, section in (("nthu", nthu), ("uta", uta), ("nitymed", nitymed), ("yawdd", yawdd)):
        df = load(name)
        lines += section(df) if not df.empty else [f"## {name}: no results yet", ""]
    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
