"""Command-line entry point: ``drowsiness <command>`` or ``python -m drowsiness``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

from drowsiness import __version__
from drowsiness.config import AppConfig, ConfigError, load_config, resolve_path

log = logging.getLogger("drowsiness")

_QUIT_KEYS = (ord("q"), 27)  # q, Esc
_RECALIBRATE_KEY = ord("c")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return args.handler(args)
    except (ConfigError, OSError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drowsiness", description="Real-time driver drowsiness detection."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, help="TOML file overriding the defaults")
        p.add_argument("--no-calibration", action="store_true", help="skip per-driver calibration")

    run = sub.add_parser("run", help="live detection in an OpenCV window")
    add_common(run)
    run.add_argument("--source", default=None, help="camera index or video file (default: camera)")
    run.add_argument("--no-sound", action="store_true", help="disable the alarm sound")
    run.add_argument("--record", type=Path, help="write per-frame signals to this CSV")
    run.set_defaults(handler=_cmd_run)

    process = sub.add_parser("process", help="analyse a video file without a window")
    add_common(process)
    process.add_argument("video", type=Path)
    process.add_argument("--out", type=Path, help="per-frame CSV (default: <video>.signals.csv)")
    process.set_defaults(handler=_cmd_process)

    dashboard = sub.add_parser("dashboard", help="open the Streamlit dashboard")
    dashboard.add_argument("--config", type=Path, help="TOML file overriding the defaults")
    dashboard.add_argument(
        "--address",
        default="127.0.0.1",
        help="interface to serve on (default: this computer only). The dashboard streams "
        "the webcam without authentication, so only expose it on a network you trust.",
    )
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=_cmd_dashboard)

    download = sub.add_parser("download-model", help="fetch the MediaPipe face model")
    download.add_argument("--force", action="store_true", help="re-download if present")
    download.set_defaults(handler=_cmd_download)
    return parser


def _load(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    if args.no_calibration:
        config = dataclasses.replace(
            config, calibration=dataclasses.replace(config.calibration, enabled=False)
        )
    return config


def _cmd_run(args: argparse.Namespace) -> int:
    import cv2

    from drowsiness import overlay
    from drowsiness.alerts import EventLog
    from drowsiness.audio import AlarmPlayer
    from drowsiness.pipeline import DrowsinessPipeline
    from drowsiness.recording import SignalRecorder
    from drowsiness.video import VideoSource

    config = _load(args)
    cam = config.camera
    source = args.source if args.source is not None else cam.index
    alarm = AlarmPlayer(
        resolve_path(config.alerts.sound_path), config.alerts.sound_enabled and not args.no_sound
    )
    events = EventLog(resolve_path(config.alerts.log_dir))
    recorder = SignalRecorder(args.record) if args.record else None
    window = "Driver Drowsiness Detection  (q: quit, c: recalibrate)"
    fps = None
    started = False

    log.info("Starting; keep your eyes open and look ahead while calibrating")
    try:
        with (
            VideoSource(source, cam.width, cam.height, cam.mirror) as video,
            DrowsinessPipeline(config) as pipeline,
        ):
            started = True
            last_t = None
            for frame, t in video:
                report = pipeline.process(frame, t)
                if report.event:
                    events.record(report.event)
                alarm.set_level(report.alert_level)
                if recorder:
                    recorder.write(report)

                if last_t is not None and t > last_t:
                    fps = 1 / (t - last_t) if fps is None else 0.9 * fps + 0.1 / (t - last_t)
                last_t = t
                cv2.imshow(window, overlay.draw(frame, report, fps))

                key = cv2.waitKey(1) & 0xFF
                if key in _QUIT_KEYS or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == _RECALIBRATE_KEY:
                    log.info("Recalibrating")
                    pipeline.recalibrate()
    except Exception as exc:
        if not started:
            raise  # failed before monitoring began: a normal error is enough
        # Monitoring stopped while the driver relies on it: fail loudly.
        log.error("Monitoring stopped: %s", exc)
        _fail_loud(alarm, window, f"MONITORING STOPPED: {exc}")
        return 1
    finally:
        alarm.stop()
        if recorder:
            recorder.close()
            log.info("Wrote %d frames to %s", recorder.rows, recorder.path)
        cv2.destroyAllWindows()
    if events.path and events.events:
        log.info("Alert log: %s", events.path)
    return 0


def _fail_loud(alarm, window: str, message: str) -> None:
    """Sound the alarm and show ``message`` until the user presses q or Esc."""
    import cv2
    import numpy as np

    from drowsiness.alerts import AlertLevel

    alarm.set_level(AlertLevel.CRITICAL)
    screen = np.zeros((200, 640, 3), np.uint8)
    screen[:] = (40, 40, 200)
    lines = [message[i : i + 52] for i in range(0, len(message), 52)][:3]
    for i, line in enumerate([*lines, "Press q to acknowledge"]):
        cv2.putText(
            screen, line, (16, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
    cv2.namedWindow(window)
    while True:
        cv2.imshow(window, screen)
        if (cv2.waitKey(100) & 0xFF) in _QUIT_KEYS:
            return
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return


def _cmd_process(args: argparse.Namespace) -> int:
    from drowsiness.alerts import AlertLevel, EventLog
    from drowsiness.pipeline import DrowsinessPipeline
    from drowsiness.recording import SignalRecorder
    from drowsiness.video import VideoSource

    config = _load(args)
    out = args.out or args.video.with_suffix(".signals.csv")
    events = EventLog(log_dir=None)
    level_frames: Counter[AlertLevel] = Counter()
    faces = 0

    with (
        VideoSource(args.video, config.camera.width, config.camera.height) as video,
        DrowsinessPipeline(config) as pipeline,
        SignalRecorder(out) as recorder,
    ):
        for frame, t in video:
            report = pipeline.process(frame, t)
            recorder.write(report)
            faces += report.face_detected
            level_frames[report.alert_level] += 1
            if report.event:
                events.record(report.event)

    total = recorder.rows
    if total == 0:
        log.error("No frames could be read from %s", args.video)
        return 1
    print(f"\nProcessed {total} frames from {args.video}")
    print(f"  face detected: {faces / total:.1%}")
    for level in AlertLevel:
        print(f"  {level.label:<9} {level_frames[level] / total:6.1%} of frames")
    print(f"  alert changes: {len(events.events)}")
    print(f"  per-frame signals: {out}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).with_name("dashboard.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        args.address,
        "--server.port",
        str(args.port),
    ]
    if args.address not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Serving the dashboard on %s: anyone who can reach this address can watch "
            "the webcam and control detection.",
            args.address,
        )
    if args.config:
        command += ["--", "--config", str(args.config.resolve())]
    return subprocess.call(command)


def _cmd_download(args: argparse.Namespace) -> int:
    from drowsiness.landmarks import download_model

    path = download_model(force=args.force)
    print(f"Model ready: {path}")
    return 0
