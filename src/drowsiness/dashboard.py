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

import altair as alt
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
    "Prototype built for a course project, not a safety device. Evaluated on public "
    "datasets only; never rely on it while driving."
)

# Background colour of each alert level (white text on top, readable in both themes).
_LEVEL_COLOURS = {
    AlertLevel.NONE: "#15803d",
    AlertLevel.WARNING: "#d97706",
    AlertLevel.ALARM: "#dc2626",
    AlertLevel.CRITICAL: "#7f1d1d",
}
STATIC_COLOUR = "#9ca3af"
ADAPTIVE_COLOUR = "#2563eb"
_LEVEL_NAMES = {
    AlertLevel.NONE: "Normal",
    AlertLevel.WARNING: "Warning",
    AlertLevel.ALARM: "Alarm",
    AlertLevel.CRITICAL: "CRITICAL",
}

_CSS = """
<style>
.dd-cards {display: flex; gap: 12px; margin: 4px 0 10px 0;}
.dd-card {flex: 1; border-radius: 10px; padding: 10px 14px; color: #fff;}
.dd-card .dd-label {font-size: 0.78rem; opacity: 0.9; letter-spacing: 0.02em;}
.dd-card .dd-level {font-size: 1.55rem; font-weight: 700; line-height: 1.25;}
.dd-card .dd-score {font-size: 0.85rem; opacity: 0.95;}
.dd-ours {outline: 3px solid rgba(37, 99, 235, 0.85); outline-offset: 2px;}
.dd-tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 6px 0 10px 0;}
.dd-tile {border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 8px; padding: 6px 10px;}
.dd-tile .dd-k {font-size: 0.72rem; opacity: 0.7;}
.dd-tile .dd-v {font-size: 1.15rem; font-weight: 600;}
</style>
"""


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
        help="Learn the driver's open-eye EAR, neutral head pose and resting smile during "
        "the first seconds.",
    )

    strategy = sb.radio(
        "Alarm driven by",
        ["adaptive", "static"],
        index=["adaptive", "static"].index(config.fusion.strategy),
        format_func={"adaptive": "Adaptive (condition-aware)", "static": "Static (baseline)"}.get,
        disabled=running,
        help="Adaptive fusion ignores eye closures while the driver smiles and trusts the "
        "eyes less when they are covered. Both strategies are always shown side by side; "
        "this only chooses which one sounds the alarm and is logged.",
    )

    with sb.expander("Normal-condition weights"):
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
                config.fusion,
                strategy=strategy,
                eye_weight=eye,
                yawn_weight=yawn,
                head_weight=head,
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


def _card(label: str, level: AlertLevel, score: float, ours: bool = False) -> str:
    cls = "dd-card dd-ours" if ours else "dd-card"
    return (
        f'<div class="{cls}" style="background:{_LEVEL_COLOURS[level]}">'
        f'<div class="dd-label">{label}</div>'
        f'<div class="dd-level">{_LEVEL_NAMES[level]}</div>'
        f'<div class="dd-score">score {score:.2f}</div></div>'
    )


def _compare_cards(report: FrameReport) -> str:
    """The two strategies' alert levels side by side: the point of the project."""
    return (
        '<div class="dd-cards">'
        + _card("STATIC · fixed weights", report.static_level, report.static_score)
        + _card("ADAPTIVE · ours", report.adaptive_level, report.adaptive_score, ours=True)
        + "</div>"
    )


def _tiles(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="dd-tile"><div class="dd-k">{k}</div><div class="dd-v">{v}</div></div>'
        for k, v in items
    )
    return f'<div class="dd-tiles">{cells}</div>'


def _degrees(angle: float) -> str:
    rounded = round(angle)
    return f"{rounded:+d}°" if rounded else "0°"  # no "-0°"


def _panel(slot, report: FrameReport) -> None:
    s = report.signals
    pose = s.pose if s else None
    with slot.container():
        if not report.calibrating:
            st.markdown(_compare_cards(report), unsafe_allow_html=True)
            if report.static_level > report.adaptive_level and report.conditions.smiling:
                st.caption("Smiling: the static baseline reads the narrowed eyes as closing.")
        st.markdown(_conditions_badges(report))

        st.markdown(
            _tiles(
                [
                    ("EAR", f"{s.ear:.2f}" if s else "-"),
                    ("MAR", f"{s.mar:.2f}" if s else "-"),
                    ("Pitch", _degrees(pose.pitch) if pose else "-"),
                    ("Roll", _degrees(pose.roll) if pose else "-"),
                ]
            ),
            unsafe_allow_html=True,
        )
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

        w = report.fusion.weights
        b = report.baseline
        caption = f"Weights: eye {w.eye:.2f} · yawn {w.yawn:.2f} · head {w.head:.2f}"
        if b:
            kind = "calibrated" if b.calibrated else "default"
            caption += f" | EAR threshold {b.ear_threshold:.3f} ({kind})"
        st.caption(caption)
        if b and not b.calibrated and report.calibration_note:
            st.warning(report.calibration_note)


_LEVEL_BADGES = {
    AlertLevel.NONE: ":green-badge[Normal]",
    AlertLevel.WARNING: ":orange-badge[Warning]",
    AlertLevel.ALARM: ":red-badge[Alarm]",
    AlertLevel.CRITICAL: ":red-badge[CRITICAL]",
}


def _level_badge(level: AlertLevel) -> str:
    return _LEVEL_BADGES[level]


def _conditions_badges(report: FrameReport) -> str:
    """One line of Streamlit colour badges for the detected conditions."""
    c = report.conditions
    light = "-" if c.brightness is None else f"{c.brightness:.0f}"
    badges = []
    if c.smiling:
        badges.append(":green-badge[😀 Smiling: eye closures ignored]")
    if c.low_light:
        badges.append(f":blue-badge[🌙 Low light ({light})]")
    if c.eye_occluded:
        why = f": {', '.join(c.reasons)}" if c.reasons else ""
        badges.append(f":orange-badge[🕶️ Eyes occluded{why}]")
    if not badges:
        badges.append(f":gray-badge[Conditions normal · brightness {light}]")
    return " ".join(badges)


def _session_tiles(monitored_s: float, fps: float, events: list[AlertEvent]) -> str:
    raised = [e for e in events if e.level > e.previous]
    warnings = sum(e.level == AlertLevel.WARNING for e in raised)
    alarms = sum(e.level >= AlertLevel.ALARM for e in raised)
    minutes, seconds = divmod(int(monitored_s), 60)
    return _tiles(
        [
            ("Monitored", f"{minutes}:{seconds:02d}"),
            ("Frame rate", f"{fps:.0f} fps"),
            ("Warnings", str(warnings)),
            ("Alarms", str(alarms)),
        ]
    )


def _score_chart(history: deque[dict], now: float, thresholds: tuple[float, ...]) -> alt.Chart:
    """Static vs adaptive score over the last minute, on the alert-level bands."""
    df = pd.DataFrame(history)
    df["seconds"] = df["time"] - now
    long = df.melt(
        id_vars="seconds",
        value_vars=["static", "adaptive"],
        var_name="fusion",
        value_name="score",
    )
    l1, l2, l3 = thresholds
    bands = pd.DataFrame(
        {
            "lo": [l1, l2, l3],
            "hi": [l2, l3, 1.0],
            "level": ["Warning", "Alarm", "Critical"],
        }
    )
    band_scale = alt.Scale(
        domain=["Warning", "Alarm", "Critical"],
        range=[_LEVEL_COLOURS[AlertLevel(i)] for i in (1, 2, 3)],
    )
    x = alt.X("seconds:Q", title="seconds ago", scale=alt.Scale(domain=[-HISTORY_S, 0]))
    band_layer = (
        alt.Chart(bands)
        .mark_rect(opacity=0.16)
        .encode(
            y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1])),
            y2="hi:Q",
            color=alt.Color("level:N", scale=band_scale, legend=None),
        )
    )
    band_labels = (
        alt.Chart(bands)
        .mark_text(align="left", baseline="bottom", dx=4, dy=-3, fontSize=11)
        .encode(
            x=alt.datum(-HISTORY_S),
            y="lo:Q",  # just inside the bottom of each band
            text="level:N",
            color=alt.Color("level:N", scale=band_scale, legend=None),
        )
    )
    y = alt.Y("score:Q", title="drowsiness score", scale=alt.Scale(domain=[0, 1]))
    # Colours are set on the marks: Streamlit's chart theme overrides colour scales.
    static = (
        alt.Chart(long[long["fusion"] == "static"])
        .mark_line(strokeWidth=2, strokeDash=[5, 4], color=STATIC_COLOUR)
        .encode(x=x, y=y)
    )
    adaptive = (
        alt.Chart(long[long["fusion"] == "adaptive"])
        .mark_line(strokeWidth=2.5, color=ADAPTIVE_COLOUR)
        .encode(x=x, y=y)
    )
    return (band_layer + band_labels + static + adaptive).properties(height=230)


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


def _guide(slot) -> None:
    """What to try when detection is stopped (also the live-demo script)."""
    with slot.container(border=True):
        st.markdown(
            "**How it works.** The webcam image goes through MediaPipe's face landmarks. "
            "Eye closure, yawning and head tilt are fused into one drowsiness score, and "
            "two strategies run side by side:\n"
            "- **Static** uses fixed weights.\n"
            "- **Adaptive** (ours) ignores eye closures while you smile and trusts the "
            "eyes less when they are covered."
        )
        st.markdown(
            "**Try it.** Turn on *Start detection* and look ahead for 5 s to calibrate. Then:\n"
            "1. **Smile or laugh.** Static warns, adaptive stays *Normal*.\n"
            "2. **Close your eyes for 2 s.** Both raise an alarm (microsleep).\n"
            "3. **Yawn slowly** or **nod forward.** The yawn and head cues rise.\n"
            "4. **Cover the camera.** The alert escalates after 3 s (face lost)."
        )


def _run(config: AppConfig, slots: dict) -> None:
    from drowsiness.video import VideoSource

    alerts = config.alerts
    thresholds = (alerts.level1_threshold, alerts.level2_threshold, alerts.level3_threshold)
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
    first_t = last_t = None
    fps = 0.0
    started = False
    slots["events"].dataframe(_event_rows(events.events), hide_index=True)
    slots["chart"].caption("The chart starts after calibration.")

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

                if first_t is None:
                    first_t = t
                if last_t is not None and t > last_t:
                    fps = 0.9 * fps + 0.1 / (t - last_t) if fps else 1.0 / (t - last_t)
                last_t = t

                if not report.calibrating:
                    history.append(
                        {
                            "time": t,
                            "static": report.static_score,
                            "adaptive": report.adaptive_score,
                        }
                    )
                    while history and t - history[0]["time"] > HISTORY_S:
                        history.popleft()

                now = time.monotonic()
                if now - last_panel >= PANEL_REFRESH_S:
                    _status(slots["status"], report)
                    _panel(slots["panel"], report)
                    slots["session"].markdown(
                        _session_tiles(t - first_t, fps, events.events), unsafe_allow_html=True
                    )
                    last_panel = now
                if history and now - last_chart >= CHART_REFRESH_S:
                    slots["chart"].altair_chart(
                        _score_chart(history, t, thresholds), use_container_width=True
                    )
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
    st.markdown(_CSS, unsafe_allow_html=True)
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
        "session": video_col.empty(),
        "status": info_col.empty(),
        "panel": info_col.empty(),
    }
    st.subheader("Drowsiness score, last 60 s")
    st.markdown(
        f'<span style="color:{ADAPTIVE_COLOUR}">━━</span> **Adaptive** (ours) &nbsp; '
        f'<span style="color:{STATIC_COLOUR}">╍╍</span> **Static** (baseline) &nbsp; '
        "· shaded: warning / alarm / critical",
        unsafe_allow_html=True,
    )
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
    _guide(slots["video"])
    events = st.session_state.get("events", [])
    if events:
        slots["chart"].caption("The chart shows the live session while detection runs.")
    else:
        slots["chart"].caption("Start detection to see the live scores.")
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
