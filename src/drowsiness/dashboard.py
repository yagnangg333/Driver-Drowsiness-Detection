"""Streamlit dashboard.

Launch with ``drowsiness dashboard`` or ``streamlit run src/drowsiness/dashboard.py``.
The webcam is read on the machine running Streamlit, so run it locally.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import threading
import time
from collections import deque

import cv2
import pandas as pd
import streamlit as st

from drowsiness import overlay
from drowsiness.alerts import AlertEvent, AlertLevel, EventLog
from drowsiness.audio import AlarmPlayer
from drowsiness.config import AppConfig, ConfigError, load_config, resolve_path
from drowsiness.pipeline import DrowsinessPipeline, FrameReport

log = logging.getLogger(__name__)

HISTORY_S = 60.0
PANEL_REFRESH_S = 0.3
CHART_REFRESH_S = 1.0
DISCLAIMER = (
    "Prototype built for a course project, not a safety device. Accuracy has not been "
    "validated; never rely on it while driving."
)


def _base_config() -> AppConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return load_config(args.config)


@st.cache_resource
def _alarm(sound_path: str) -> AlarmPlayer:
    return AlarmPlayer(sound_path)


@st.cache_resource
def _camera_lock() -> threading.Lock:
    """Shared by every browser session, so only one detection loop uses the camera."""
    return threading.Lock()


def _sidebar(config: AppConfig, running: bool) -> tuple[AppConfig, bool]:
    sb = st.sidebar
    sb.header("Setup")
    if running:
        sb.caption("Stop detection to change settings.")
    camera = sb.number_input(
        "Camera index", min_value=0, value=config.camera.index, step=1, disabled=running
    )
    sound = sb.checkbox("Alarm sound", value=config.alerts.sound_enabled, disabled=running)
    calibrate = sb.checkbox(
        "Calibrate at start",
        value=config.calibration.enabled,
        disabled=running,
        help="Learn the driver's open-eye EAR and neutral head pose during the first seconds.",
    )

    with sb.expander("Fusion weights (static baseline)"):
        f = config.fusion
        eye = st.slider("Eye closure", 0.0, 1.0, f.eye_weight, 0.05, disabled=running)
        yawn = st.slider("Yawning", 0.0, 1.0, f.yawn_weight, 0.05, disabled=running)
        head = st.slider("Head tilt", 0.0, 1.0, f.head_weight, 0.05, disabled=running)

    with sb.expander("Alert thresholds"):
        a = config.alerts
        l1 = st.slider("Level 1: warning", 0.05, 1.0, a.level1_threshold, 0.05, disabled=running)
        l2 = st.slider("Level 2: alarm", 0.05, 1.0, a.level2_threshold, 0.05, disabled=running)
        l3 = st.slider("Level 3: critical", 0.05, 1.0, a.level3_threshold, 0.05, disabled=running)

    try:
        config = dataclasses.replace(
            config,
            camera=dataclasses.replace(config.camera, index=int(camera)),
            calibration=dataclasses.replace(config.calibration, enabled=calibrate),
            fusion=dataclasses.replace(
                config.fusion, eye_weight=eye, yawn_weight=yawn, head_weight=head
            ),
            alerts=dataclasses.replace(
                config.alerts,
                sound_enabled=sound,
                level1_threshold=l1,
                level2_threshold=l2,
                level3_threshold=l3,
            ),
        )
        return config, True
    except ConfigError as exc:
        sb.error(str(exc))
        return config, False


def _status(slot, report: FrameReport) -> None:
    level = report.alert_level
    if report.calibrating:
        note = f" - {report.calibration_note}" if report.calibration_note else ""
        slot.info(
            f"Calibrating: look ahead with eyes open ({report.calibration_progress:.0%}){note}"
        )
    elif not report.face_detected:
        message = f"Face not visible (Level {int(level)})"
        (slot.error if level >= AlertLevel.ALARM else slot.warning)(message)
    elif level == AlertLevel.NONE:
        slot.success(f"Awake (score {report.fusion.score:.2f})")
    elif level == AlertLevel.WARNING:
        slot.warning(f"Level 1 warning: signs of drowsiness (score {report.fusion.score:.2f})")
    elif level == AlertLevel.ALARM:
        slot.error(f"Level 2 alarm: driver drowsy (score {report.fusion.score:.2f})")
    else:
        slot.error(f"Level 3 CRITICAL: wake up! (score {report.fusion.score:.2f})")


def _panel(slot, report: FrameReport) -> None:
    s = report.signals
    pose = s.pose if s else None
    with slot.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EAR", f"{s.ear:.3f}" if s else "-")
        c2.metric("MAR", f"{s.mar:.3f}" if s else "-")
        c3.metric("Pitch", f"{pose.pitch:+.0f}°" if pose else "-")
        c4.metric("Roll", f"{pose.roll:+.0f}°" if pose else "-")

        st.progress(
            report.eye.score,
            text=f"Eye closure {report.eye.score:.2f} "
            f"(eyes closed {report.eye.perclos:.0%} of the window)",
        )
        st.progress(
            report.yawn.score,
            text=f"Yawning {report.yawn.score:.2f} ({report.yawn.yawn_count} recent)",
        )
        st.progress(report.head.score, text=f"Head tilt {report.head.score:.2f}")
        st.progress(
            min(report.fusion.score, 1.0),
            text=f"**Fused drowsiness score {report.fusion.score:.2f}**",
        )

        w = report.fusion.weights
        b = report.baseline
        caption = f"Weights: eye {w.eye:.2f} · yawn {w.yawn:.2f} · head {w.head:.2f}"
        if b:
            kind = "calibrated" if b.calibrated else "default"
            caption += f" | EAR threshold {b.ear_threshold:.3f} ({kind})"
        st.caption(caption)
        if b and not b.calibrated and report.calibration_note:
            st.warning(report.calibration_note)


def _event_rows(events: list[AlertEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": e.wall_time.strftime("%H:%M:%S"),
                "from": e.previous.label,
                "to": e.level.label,
                "score": round(e.score, 2),
                "reason": e.reason,
            }
            for e in reversed(events)
        ],
        columns=["time", "from", "to", "score", "reason"],
    )


def _run(config: AppConfig, slots: dict) -> None:
    from drowsiness.video import VideoSource

    alerts = config.alerts
    alarm = _alarm(str(resolve_path(alerts.sound_path)))
    if not alarm.available:
        _alarm.clear()  # retry audio set-up next time (e.g. headphones plugged in)
        if alerts.sound_enabled:
            st.sidebar.warning("No audio device; alarm sound disabled.")
    events = EventLog(resolve_path(alerts.log_dir))
    st.session_state.events = events.events
    st.session_state.log_path = events.path
    history: deque[dict] = deque()
    last_panel = last_chart = 0.0
    started = False

    try:
        cam = config.camera
        with (
            VideoSource(cam.index, cam.width, cam.height, cam.mirror) as video,
            DrowsinessPipeline(config) as pipeline,
        ):
            started = True
            for frame, t in video:
                report = pipeline.process(frame, t)
                if alerts.sound_enabled:
                    alarm.set_level(report.alert_level)
                if report.event:
                    events.record(report.event)
                    slots["events"].dataframe(_event_rows(events.events), hide_index=True)

                overlay.draw(frame, report)
                slots["video"].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if not report.calibrating:
                    history.append(
                        {
                            "time": t,
                            "fused": report.fusion.score,
                            "eye": report.eye.score,
                            "yawn": report.yawn.score,
                            "head": report.head.score,
                        }
                    )
                    while history and t - history[0]["time"] > HISTORY_S:
                        history.popleft()

                now = time.monotonic()
                if now - last_panel >= PANEL_REFRESH_S:
                    _status(slots["status"], report)
                    _panel(slots["panel"], report)
                    last_panel = now
                if history and now - last_chart >= CHART_REFRESH_S:
                    slots["chart"].line_chart(pd.DataFrame(history).set_index("time"), height=220)
                    last_chart = now
    except Exception as exc:
        # Streamlit stops a run (e.g. the toggle was switched off) by raising its
        # own control-flow exceptions; let those through untouched.
        if type(exc).__module__.startswith("streamlit"):
            raise
        if not started:
            st.error(str(exc))
            return
        # Monitoring stopped while the driver relies on it: fail loudly until
        # the user acknowledges by switching detection off.
        log.error("Monitoring stopped: %s", exc)
        alarm.set_level(AlertLevel.CRITICAL)
        slots["status"].error(
            f"MONITORING STOPPED: {exc}. Turn off **Start detection** to silence the alarm."
        )
        while True:
            time.sleep(0.5)
            slots["status"].error(
                f"MONITORING STOPPED: {exc}. Turn off **Start detection** to silence the alarm."
            )
    finally:
        alarm.stop()


def main() -> None:
    st.set_page_config(page_title="Drowsiness Detection", page_icon="😴", layout="wide")
    st.title("Driver Drowsiness Detection")
    st.caption(DISCLAIMER)

    try:
        base = _base_config()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    running_requested = bool(st.session_state.get("running", False))
    config, valid = _sidebar(base, running_requested)

    running = st.toggle("Start detection", key="running", disabled=not valid)
    video_col, info_col = st.columns([3, 2], gap="large")
    slots = {
        "video": video_col.empty(),
        "status": info_col.empty(),
        "panel": info_col.empty(),
    }
    st.subheader("Drowsiness score (last 60 s)")
    slots["chart"] = st.empty()
    st.subheader("Alert history")
    slots["events"] = st.empty()

    if running:
        lock = _camera_lock()
        if not lock.acquire(blocking=False):
            slots["status"].warning(
                "Detection is already running in another browser tab or window. "
                "Stop it there first; two loops on one camera would interfere."
            )
            return
        try:
            _run(config, slots)
        finally:
            lock.release()
        return

    slots["status"].info("Detection stopped. Turn on **Start detection** to begin.")
    events = st.session_state.get("events", [])
    slots["events"].dataframe(_event_rows(events), hide_index=True)
    log_path = st.session_state.get("log_path")
    if events and log_path and log_path.exists():
        st.download_button(
            "Download alert log (CSV)",
            log_path.read_bytes(),
            file_name=log_path.name,
            mime="text/csv",
        )


main()
