"""Sample frames from the evaluation datasets and record condition measurements.

Used to choose the low-light and eye-occlusion thresholds from real footage
instead of guessing them. For every video it measures:

- the first ``CALIBRATION_S`` seconds densely, like the live calibration, and
- then one frame every ``step`` seconds,

and writes one CSV row per frame with the raw measurements from
``drowsiness.conditions.measure`` plus what is known about the video (dataset,
night or day, type of glasses).

    python tools/condition_features.py --out data/condition_features.csv

Expects the datasets under ``data/`` as described in the README.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2

from drowsiness.conditions import measure
from drowsiness.config import PROJECT_ROOT, ConditionConfig
from drowsiness.landmarks import FaceLandmarker
from drowsiness.signals import compute_signals

DATA = PROJECT_ROOT / "data"
CALIBRATION_S = 5.0
CALIBRATION_STEP_S = 0.5
WIDTH = 640  # the size frames are processed at, as in VideoSource

FIELDS = [
    "dataset",
    "video",
    "night",
    "glasses",
    "t",
    "calibration",
    "face",
    "face_brightness",
    "frame_brightness",
    "glare_fraction",
    "eye_contrast_ratio",
    "ear_asymmetry",
    "ear",
]


def videos() -> list[tuple[str, str, int | None, str, float]]:
    """(dataset, path, night, glasses, sampling step in s) for every video."""
    out = []
    for path in sorted(glob.glob(str(DATA / "DSM_Dataset-HDTV720" / "*" / "*" / "*" / "*.mp4"))):
        out.append(("nitymed", path, 1, "unknown", 2.0))
    for path in sorted(glob.glob(str(DATA / "YawDD dataset" / "Mirror" / "*" / "*.avi"))):
        name = Path(path).name
        kind = "sunglasses" if "SunGlasses" in name else "none" if "NoGlasses" in name else "clear"
        out.append(("yawdd", path, 0, kind, 1.0))
    for path in sorted(glob.glob(str(DATA / "uta-rldd" / "*" / "*" / "*"))):
        out.append(("uta", path, None, "unknown", 10.0))  # lighting not labelled
    return out


def _resize(frame):
    scale = WIDTH / frame.shape[1]
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def process(item: tuple[str, str, int | None, str, float]) -> list[dict]:
    dataset, path, night, glasses, step = item
    config = ConditionConfig()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    times = [i * CALIBRATION_STEP_S for i in range(int(CALIBRATION_S / CALIBRATION_STEP_S))]
    t = CALIBRATION_S
    while t < duration - 0.5:
        times.append(t)
        t += step

    rows = []
    video = str(Path(path).relative_to(DATA))
    with FaceLandmarker() as landmarker:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize(frame)
            row = {
                "dataset": dataset,
                "video": video,
                "night": "" if night is None else night,
                "glasses": glasses,
                "t": round(t, 2),
                "calibration": int(t < CALIBRATION_S),
                "face": 0,
            }
            face = landmarker.detect(frame, t)
            if face is not None:
                signals = compute_signals(face)
                m = measure(frame, face, signals, config)
                row.update(
                    face=1,
                    face_brightness=round(m.face_brightness, 2),
                    frame_brightness=round(m.frame_brightness, 2),
                    glare_fraction=round(m.glare_fraction, 4),
                    eye_contrast_ratio=round(m.eye_contrast_ratio, 4),
                    ear_asymmetry="" if m.ear_asymmetry is None else round(m.ear_asymmetry, 4),
                    ear=round(signals.ear, 4),
                )
            rows.append(row)
    cap.release()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DATA / "condition_features.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)

    items = videos()
    if not items:
        print(f"No videos found under {DATA}", file=sys.stderr)
        return 1
    print(f"{len(items)} videos, {args.workers} workers -> {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with args.out.open("w", newline="", encoding="utf-8") as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rows in pool.imap_unordered(process, items):
            writer.writerows(rows)
            done += 1
            if done % 25 == 0 or done == len(items):
                print(f"  {done}/{len(items)} videos", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
