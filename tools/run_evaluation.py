"""Run the pipeline over the evaluation datasets and save per-frame results.

Each recording becomes one CSV under ``data/eval/<dataset>/`` with every
column of ``FrameReport.to_row()`` (both the static and the adaptive score,
conditions, indicator states) plus what the dataset says about it (labels,
glasses, ...). Finished recordings are skipped, so an interrupted run can be
resumed. ``tools/evaluation_report.py`` turns these files into results.

    python tools/run_evaluation.py                  # all datasets
    python tools/run_evaluation.py --only nthu yawdd

Datasets are expected under ``data/`` as described in the README.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2

from drowsiness.config import PROJECT_ROOT, AppConfig
from drowsiness.pipeline import DrowsinessPipeline
from drowsiness.video import VideoSource

DATA = PROJECT_ROOT / "data"
OUT = DATA / "eval"
NTHU_FPS = 30.0  # NTHU-DDD day scenarios were recorded at 30 fps
UTA_STRIDE = 2  # 10-minute UTA videos: every 2nd frame (>= 7.5 fps) is plenty


@dataclass
class Job:
    dataset: str
    name: str  # output file stem, unique within the dataset
    meta: dict[str, str]
    video: str | None = None  # a video file, or
    frames: list[tuple[str, float, int]] = field(default_factory=list)  # NTHU images
    stride: int = 1

    @property
    def out(self) -> Path:
        return OUT / self.dataset / f"{self.name}.csv"


def yawdd_jobs() -> list[Job]:
    jobs = []
    for path in sorted(glob.glob(str(DATA / "YawDD dataset" / "Mirror" / "*" / "*.avi"))):
        stem = Path(path).stem.replace(" ", "").replace("&", "")
        m = re.match(r"(\d+)-(Female|Male)(\w*?)-(\w+)$", stem)
        if not m:
            continue
        number, gender, look, action = m.groups()
        glasses = (
            "sunglasses" if "SunGlasses" in look else "none" if "NoGlasses" in look else "clear"
        )
        jobs.append(
            Job(
                "yawdd",
                stem,
                {
                    "subject": f"{gender[0]}{number}",
                    "glasses": glasses,
                    "action": action.lower().replace("talkingyawning", "talking_yawning"),
                },
                video=path,
            )
        )
    return jobs


def nitymed_jobs() -> list[Job]:
    jobs = []
    for path in sorted(glob.glob(str(DATA / "DSM_Dataset-HDTV720" / "*" / "*" / "*" / "*.mp4"))):
        parts = Path(path).parts
        category, gender = parts[-4], parts[-3]
        jobs.append(
            Job(
                "nitymed",
                f"{category}_{Path(path).stem}",
                {"category": category.lower(), "gender": gender.lower()},
                video=path,
            )
        )
    return jobs


def uta_jobs() -> list[Job]:
    jobs = []
    for path in sorted(glob.glob(str(DATA / "uta-rldd" / "*" / "*" / "*"))):
        subject = Path(path).parent.name
        label = Path(path).stem  # 0 alert, 5 low vigilant, 10 drowsy
        jobs.append(
            Job(
                "uta",
                f"{subject}_{label}",
                {"subject": subject, "label": label},
                video=path,
                stride=UTA_STRIDE,
            )
        )
    return jobs


def nthu_jobs() -> list[Job]:
    """One job per subject and scenario; frames carry their per-frame labels."""
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(str(DATA / "NTHU-DDD" / "*" / "*.jpg")):
        m = re.match(r"(\d+)_(.+?)_(\w+?)_(\d+)_(drowsy|notdrowsy)$", Path(path).stem)
        if not m:
            continue
        subject, scenario, behaviour, frame, label = m.groups()
        groups[(subject, scenario)][behaviour].append(
            (path, int(frame) / NTHU_FPS, int(label == "drowsy"))
        )
    jobs = []
    for (subject, scenario), behaviours in sorted(groups.items()):
        frames = []
        for behaviour in (
            "nonsleepyCombination",
            "sleepyCombination",
            "slowBlinkWithNodding",
            "yawning",
        ):
            for path, t, label in sorted(behaviours.get(behaviour, []), key=lambda x: x[1]):
                frames.append((path, t, label, behaviour))
        jobs.append(
            Job(
                "nthu",
                f"{subject}_{scenario}",
                {"subject": subject, "scenario": scenario},
                frames=frames,
            )
        )
    return jobs


def _write(job: Job, rows: list[dict]) -> None:
    job.out.parent.mkdir(parents=True, exist_ok=True)
    partial = job.out.with_suffix(".part")
    with partial.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(job.out)


def _row(report, meta: dict, seconds: float, **extra) -> dict:
    return {**meta, **extra, **report.to_row(), "process_ms": round(seconds * 1000, 2)}


def run_video(job: Job, config: AppConfig) -> list[dict]:
    rows = []
    with (
        VideoSource(job.video, config.camera.width, config.camera.height, mirror=False) as video,
        DrowsinessPipeline(config) as pipeline,
    ):
        for i, (frame, t) in enumerate(video):
            if i % job.stride:
                continue
            start = time.perf_counter()
            report = pipeline.process(frame, t)
            rows.append(_row(report, job.meta, time.perf_counter() - start))
    return rows


def run_nthu(job: Job, config: AppConfig) -> list[dict]:
    """Calibrate on the subject's non-sleepy recording, then process every
    recording from its start with fresh temporal state."""
    rows = []
    offset = 0.0  # MediaPipe needs increasing timestamps across recordings
    by_behaviour: dict[str, list] = defaultdict(list)
    for path, t, label, behaviour in job.frames:
        by_behaviour[behaviour].append((path, t, label))
    calibration_source = by_behaviour.get("nonsleepyCombination") or next(
        iter(by_behaviour.values())
    )

    with DrowsinessPipeline(config) as pipeline:
        for path, t, _ in calibration_source:
            if not pipeline.process(cv2.imread(path), offset + t).calibrating:
                break
        offset += calibration_source[-1][1] + 10.0
        for behaviour, frames in by_behaviour.items():
            pipeline.reset_tracking()
            first = frames[0][1]
            for path, t, label in frames:
                frame = cv2.imread(path)
                start = time.perf_counter()
                report = pipeline.process(frame, offset + t - first)
                rows.append(
                    _row(
                        report,
                        job.meta,
                        time.perf_counter() - start,
                        behaviour=behaviour,
                        frame=Path(path).name,
                        drowsy=label,
                        clip_time_s=round(t - first, 3),
                    )
                )
            offset += frames[-1][1] - first + 10.0
    return rows


def process(job: Job) -> tuple[str, str]:
    logging.disable(logging.WARNING)
    config = AppConfig()
    try:
        rows = run_nthu(job, config) if job.frames else run_video(job, config)
    except Exception as exc:  # report and carry on with the other recordings
        return job.name, f"FAILED: {exc}"
    if not rows:
        return job.name, "no frames"
    _write(job, rows)
    return job.name, f"{len(rows)} frames"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sources = {"nthu": nthu_jobs, "yawdd": yawdd_jobs, "nitymed": nitymed_jobs, "uta": uta_jobs}
    parser.add_argument("--only", nargs="+", choices=sources, default=list(sources))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--redo", action="store_true", help="reprocess finished recordings")
    args = parser.parse_args(argv)

    jobs = [job for name in args.only for job in sources[name]()]
    todo = [job for job in jobs if args.redo or not job.out.exists()]
    print(f"{len(jobs)} recordings, {len(todo)} to process, {args.workers} workers", flush=True)
    # Longest first, so a 10-minute video does not start last.
    todo.sort(key=lambda j: (j.dataset != "uta", j.dataset != "nthu"))
    started = time.monotonic()
    with Pool(args.workers) as pool:
        for i, (name, result) in enumerate(pool.imap_unordered(process, todo), start=1):
            elapsed = (time.monotonic() - started) / 60
            print(f"[{i}/{len(todo)} {elapsed:5.1f} min] {name}: {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
