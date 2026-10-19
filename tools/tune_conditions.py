"""Choose the condition thresholds from ``tools/condition_features.py`` output.

Videos are split into a tuning half and a held-out half (alternating, after
sorting by name). Thresholds are picked on the tuning half only, and every
rate is reported on both halves, so the chosen values are not just fitted to
the data they are reported on.

- Low light: the whole-frame brightness that best separates night (NITYMED)
  from day (YawDD) footage, counting a daytime false alarm twice as heavily
  as a missed night frame.
- Eye occlusion: the settings that catch the most sunglasses frames (YawDD)
  while flagging at most ``MAX_FALSE_ALARM`` of the frames in every group of
  footage without sunglasses. A false occlusion lowers the eye weight for a
  driver whose eyes are perfectly visible, so false alarms are capped first.

    python tools/tune_conditions.py data/condition_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAX_FALSE_ALARM = 0.03


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["face"] == 1].copy()
    videos = sorted(df["video"].unique())
    held_out = set(videos[1::2])
    df["split"] = np.where(df["video"].isin(held_out), "held-out", "tuning")
    df["group"] = df["dataset"] + "/" + df["glasses"]
    return df


def choose_low_light(df: pd.DataFrame) -> float:
    tune = df[(df["split"] == "tuning") & df["dataset"].isin(["nitymed", "yawdd"])]
    night = tune["dataset"] == "nitymed"
    best_score, best = -1.0, 0.0
    for b in np.arange(30, 151, 1.0):
        dark = tune["frame_brightness"] < b
        score = dark[night].mean() + 2 * (1 - dark[~night].mean())
        if score > best_score:
            best_score, best = score, float(b)
    return best


def occluded(df: pd.DataFrame, glare: float, contrast: float) -> pd.Series:
    return (df["glare_fraction"] >= glare) | (df["eye_contrast_ratio"] < contrast)


def choose_occlusion(df: pd.DataFrame) -> tuple[float, float]:
    tune = df[df["split"] == "tuning"]
    sun = tune["glasses"] == "sunglasses"
    best_score, best = -1.0, (1.01, 0.0)
    for glare in (0.05, 0.10, 0.15, 0.20, 0.30, 1.01):  # 1.01 = glare check off
        for contrast in np.arange(0.0, 0.31, 0.01):
            flag = occluded(tune, glare, contrast)
            worst_false = flag[~sun].groupby(tune.loc[~sun, "group"]).mean().max()
            if worst_false > MAX_FALSE_ALARM:
                continue
            score = flag[sun].mean()
            if score > best_score:
                best_score, best = score, (glare, round(float(contrast), 2))
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path)
    args = parser.parse_args(argv)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    df = load(args.features)
    print(f"{len(df)} frames with a face, {df['video'].nunique()} videos\n")
    cols = ["face_brightness", "frame_brightness", "eye_contrast_ratio", "glare_fraction"]
    print("Median measurements per group:")
    print(df.groupby("group")[cols].median().round(3), "\n")

    b = choose_low_light(df)
    dark = df["frame_brightness"] < b
    print(f"LOW LIGHT: whole-frame brightness < {b:g}")
    print("share of frames read as low light:")
    print(dark.groupby([df["dataset"], df["split"]]).mean().unstack().round(3), "\n")

    glare, contrast = choose_occlusion(df)
    glare_text = "off" if glare > 1 else f">= {glare:g} near-white eye pixels"
    print(f"EYE OCCLUSION: glare {glare_text}; eyes covered if contrast < {contrast:g}")
    print(f"(chosen so that at most {MAX_FALSE_ALARM:.0%} of frames in any group without")
    print("sunglasses are flagged, on the tuning half)")
    print("share of frames read as occluded:")
    flag = occluded(df, glare, contrast)
    print(flag.groupby([df["group"], df["split"]]).mean().unstack().round(3), "\n")

    print("Each check alone (all frames):")
    checks = {
        "glare": df["glare_fraction"] >= glare,
        "covered": df["eye_contrast_ratio"] < contrast,
        "one eye hidden": df["ear_asymmetry"] > 0.5,
    }
    print(pd.DataFrame({k: v.groupby(df["group"]).mean() for k, v in checks.items()}).round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
