"""Offline simulator: re-run everything after MediaPipe on recorded signals.

``tools/run_evaluation.py`` saves, for every frame, the raw signals (EAR, MAR,
head angles, expression scores, conditions). ``simulate`` feeds them through
the project's own indicator, fusion and alert classes with any settings, so a
change to thresholds or rules can be evaluated on every dataset in minutes
instead of re-running face tracking for hours.

The calibration baseline is not stored per frame; it is rebuilt the way the
pipeline builds it (medians over the calibration frames), so with default
settings the simulation closely reproduces the pipeline (see ``validate``).
"""

from __future__ import annotations

import dataclasses
import glob
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from drowsiness.config import PROJECT_ROOT, AppConfig
from drowsiness.evaluation import replay_alerts
from drowsiness.fusion import AdaptiveFusion, IndicatorScores, StaticFusion
from drowsiness.fusion.adaptive.conditions import Conditions
from drowsiness.indicators import EyeClosureIndicator, HeadTiltIndicator, YawnIndicator

EVAL = PROJECT_ROOT / "data" / "eval"


@dataclass(frozen=True)
class Settings:
    """Everything a simulation may vary; defaults are the shipped config."""

    config: AppConfig = field(default_factory=AppConfig)
    # Extra rule: frames where this returns True do not count as eyes closed
    # (on top of the smile rule), e.g. looking down or sideways.
    ignore_closure: Callable[[pd.DataFrame], np.ndarray] | None = None
    use_smile_rule: bool = True
    # Override how "smiling" is decided per frame (default: the pipeline's rule).
    smiling_rule: Callable[[pd.DataFrame, AppConfig], np.ndarray] | None = None
    # Yawn signal: "mar" (shipped) or MediaPipe's "jawOpen" score.
    yawn_signal: str = "mar"


def load(dataset: str) -> list[pd.DataFrame]:
    """One frame table per recording (NTHU: per recording and behaviour)."""
    out = []
    for path in sorted(glob.glob(str(EVAL / dataset / "*.csv"))):
        df = pd.read_csv(path)
        df["recording"] = path.replace("\\", "/").rsplit("/", 1)[-1][:-4]
        if "behaviour" in df:
            out += [g.reset_index(drop=True) for _, g in df.groupby("behaviour", sort=False)]
        else:
            out.append(df)
    return out


def _calibration_frames(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """The frames calibration would have used, with a face."""
    cal = config.calibration
    if "behaviour" in df:
        # NTHU: calibrated on the start of the subject's non-sleepy recording.
        src = (
            df[df["clip_time_s"] < cal.duration_s]
            if df["behaviour"].iloc[0] == ("nonsleepyCombination")
            else None
        )
        if src is None or len(src) < 5:
            src = df.head(50)
    else:
        src = df[df["calibrating"] == 1]
    return src[src["face_detected"] == 1]


def resting_smile(df: pd.DataFrame, config: AppConfig) -> float | None:
    """The driver's calibrated resting smile score, as the pipeline records it."""
    src = _calibration_frames(df, config)
    if len(src) < 5 or "bs_mouthSmile" not in src or src["bs_mouthSmile"].isna().all():
        return None
    return float(src["bs_mouthSmile"].median())


def baseline(df: pd.DataFrame, config: AppConfig) -> tuple[float, float, float]:
    """(EAR threshold, neutral pitch, neutral roll) as calibration would set them."""
    cal = config.calibration
    src = _calibration_frames(df, config)
    if len(src) < 5:
        return config.indicators.default_ear_threshold, 0.0, 0.0
    ear = float(src["ear"].median())
    if not cal.min_open_ear <= ear <= cal.max_open_ear:
        return config.indicators.default_ear_threshold, 0.0, 0.0
    threshold = min(max(ear * cal.ear_closed_ratio, cal.min_ear_threshold), cal.max_ear_threshold)
    return threshold, float(src["pitch"].median()), float(src["roll"].median())


def smile_mean(df: pd.DataFrame, window_s: float) -> np.ndarray:
    s = df.set_index(pd.to_datetime(df["timestamp_s"], unit="s"))["bs_mouthSmile"]
    return s.fillna(0).rolling(f"{int(window_s * 1000)}ms").mean().to_numpy()


def simulate(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per-frame scores and alert levels for both strategies."""
    cfg = settings.config
    ind = cfg.indicators
    ear_threshold, pitch0, roll0 = baseline(df, cfg)
    if "behaviour" in df and df["behaviour"].iloc[0] != "nonsleepyCombination":
        # NTHU: every recording of a subject uses the non-sleepy calibration;
        # the caller passes it in via df.attrs when available.
        ear_threshold, pitch0, roll0 = df.attrs.get("baseline", (ear_threshold, pitch0, roll0))

    eye_args = (ind.perclos_window_s, ind.perclos_max, ind.microsleep_s, ind.closure_debounce_s)
    eye_s, eye_a = EyeClosureIndicator(*eye_args), EyeClosureIndicator(*eye_args)
    yawn_threshold = ind.yawn_mar_threshold
    yawn = YawnIndicator(
        yawn_threshold,
        ind.yawn_min_duration_s,
        ind.yawn_window_s,
        ind.yawns_for_max,
        ind.yawn_debounce_s,
    )
    head = HeadTiltIndicator(
        ind.head_pitch_threshold_deg,
        ind.head_roll_threshold_deg,
        ind.head_tilt_max_s,
        ind.head_debounce_s,
    )
    static, adaptive = StaticFusion(cfg.fusion), AdaptiveFusion(cfg.fusion)

    has_smile = "bs_mouthSmile" in df and df["bs_mouthSmile"].notna().any()
    if not (has_smile and settings.use_smile_rule):
        smiling = np.zeros(len(df), bool)
    elif settings.smiling_rule is not None:
        smiling = np.asarray(settings.smiling_rule(df, cfg), bool)
    else:
        cond = cfg.conditions
        rest = df.attrs.get("resting_smile", resting_smile(df, cfg))
        threshold = cond.smile_threshold
        if rest is not None:
            threshold = max(threshold, rest + cond.smile_margin)
        smiling = smile_mean(df, cond.smile_window_s) > threshold
        wide = df.set_index(pd.to_datetime(df["timestamp_s"], unit="s"))["mar"].fillna(0)
        recent_max = wide.rolling(f"{int(cond.smile_yawn_guard_s * 1000)}ms").max().to_numpy()
        smiling &= recent_max <= ind.yawn_mar_threshold
    extra = (
        np.asarray(settings.ignore_closure(df), bool)
        if settings.ignore_closure is not None
        else np.zeros(len(df), bool)
    )
    yawn_values = (
        df["bs_jawOpen"].fillna(0).to_numpy() if settings.yawn_signal == "jawOpen" else df["mar"]
    )

    n = len(df)
    s_score, a_score = np.zeros(n), np.zeros(n)
    s_sleep, a_sleep = np.zeros(n, bool), np.zeros(n, bool)
    yawning = np.zeros(n, bool)
    missing_since = None
    cols = zip(
        df["timestamp_s"],
        df["face_detected"],
        df["calibrating"],
        df["ear"],
        yawn_values,
        df["pitch"],
        df["roll"],
        df["low_light"],
        df["eye_occluded"],
        smiling,
        extra,
        strict=True,
    )
    for i, (t, face, calib, ear, y, pitch, roll, dark, occl, smile, ignore) in enumerate(cols):
        conditions = Conditions(low_light=bool(dark), eye_occluded=bool(occl))
        if not calib:
            if face:
                missing_since = None
                closed = ear < ear_threshold and not ignore
                eye_s.update(t, closed)
                eye_a.update(t, closed and not smile)
                yawn.update(t, y)
                if not np.isnan(pitch):
                    head.update(t, pitch - pitch0, roll - roll0)
            else:
                missing_since = t if missing_since is None else missing_since
                if t - missing_since >= ind.face_loss_grace_s:
                    for indicator in (eye_s, eye_a, yawn, head):
                        indicator.face_lost(t)
        ys, hs = yawn.state.score, head.state.score
        s_score[i] = static.fuse(IndicatorScores(eye_s.state.score, ys, hs), conditions, t).score
        a_score[i] = adaptive.fuse(IndicatorScores(eye_a.state.score, ys, hs), conditions, t).score
        s_sleep[i], a_sleep[i] = eye_s.state.microsleep, eye_a.state.microsleep
        yawning[i] = yawn.state.yawning

    out = pd.DataFrame(
        {
            "static_score": s_score,
            "adaptive_score": a_score,
            "yawning": yawning,
            "smiling": smiling,
        },
        index=df.index,
    )
    face = df["face_detected"].astype(bool)
    calib = df["calibrating"].astype(bool)
    for name, score, sleep in (("static", s_score, s_sleep), ("adaptive", a_score, a_sleep)):
        out[f"{name}_level"] = replay_alerts(
            df["timestamp_s"], score, face, sleep, calib, cfg.alerts
        )
    return out


def with_nthu_baselines(recordings: list[pd.DataFrame], config: AppConfig) -> None:
    """Give every NTHU recording its subject's non-sleepy calibration."""
    by_subject = {}
    for df in recordings:
        if "behaviour" in df and df["behaviour"].iloc[0] == "nonsleepyCombination":
            by_subject[df["recording"].iloc[0]] = (baseline(df, config), resting_smile(df, config))
    for df in recordings:
        if "behaviour" in df:
            base, smile = by_subject.get(df["recording"].iloc[0], (None, None))
            df.attrs["baseline"] = base
            df.attrs["resting_smile"] = smile


def configure(config: AppConfig, **sections: dict) -> AppConfig:
    """config with fields replaced, e.g. configure(c, indicators={"microsleep_s": 2.0})."""
    return dataclasses.replace(
        config,
        **{
            name: dataclasses.replace(getattr(config, name), **values)
            for name, values in sections.items()
        },
    )


def validate(dataset: str) -> float:
    """Share of frames where the simulated adaptive alert level equals the pipeline's."""
    recs = load(dataset)
    with_nthu_baselines(recs, AppConfig())
    agree, total = 0, 0
    for df in recs:
        if "microsleep_adaptive" not in df:
            continue
        sim = simulate(df, Settings())
        agree += int((sim["adaptive_level"].to_numpy() == df["alert_level"].to_numpy()).sum())
        total += len(df)
    return agree / total if total else float("nan")


if __name__ == "__main__":
    for name in ("uta", "nthu", "yawdd", "nitymed"):
        print(name, f"simulated alerts match the pipeline on {validate(name):.2%} of frames")
