"""Escalating three-level alert state machine and alert event log."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path

from drowsiness.config import AlertConfig

log = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    NONE = 0
    WARNING = 1  # on-screen warning
    ALARM = 2  # audio alarm
    CRITICAL = 3  # persistent alarm, logged as a critical event

    @property
    def label(self) -> str:
        return {0: "Normal", 1: "Warning", 2: "Alarm", 3: "Critical"}[self.value]


@dataclass(frozen=True)
class AlertEvent:
    timestamp: float  # session time in seconds
    wall_time: datetime
    previous: AlertLevel
    level: AlertLevel
    score: float
    reason: str


class AlertManager:
    """Maps the fused score to an alert level with hysteresis and escalation.

    - A level is entered when the score reaches its threshold and kept until
      the score falls ``hysteresis`` below it, which prevents flicker.
    - Staying at ALARM for ``escalate_after_s`` escalates to CRITICAL.
    - CRITICAL is held for at least ``level3_hold_s`` after its cause clears.
    - Losing the face for ``no_face_warning_s`` raises at least a WARNING.
    """

    def __init__(self, config: AlertConfig):
        self._config = config
        self.reset()

    def reset(self) -> None:
        self.level = AlertLevel.NONE
        self._alarm_since: float | None = None
        self._critical_until = float("-inf")
        self._no_face_since: float | None = None

    def update(self, t: float, score: float, face_detected: bool) -> AlertEvent | None:
        cfg = self._config
        level, reason = self._level_for_score(score)

        if level >= AlertLevel.ALARM:
            if self._alarm_since is None:
                self._alarm_since = t
            if t - self._alarm_since >= cfg.escalate_after_s and level < AlertLevel.CRITICAL:
                level, reason = (
                    AlertLevel.CRITICAL,
                    f"alarm sustained for {cfg.escalate_after_s:g}s",
                )
        else:
            self._alarm_since = None

        if level == AlertLevel.CRITICAL:
            self._critical_until = t + cfg.level3_hold_s
        elif t < self._critical_until:
            level, reason = AlertLevel.CRITICAL, "critical alert hold"

        if face_detected:
            self._no_face_since = None
        else:
            if self._no_face_since is None:
                self._no_face_since = t
            if t - self._no_face_since >= cfg.no_face_warning_s and level < AlertLevel.WARNING:
                level, reason = AlertLevel.WARNING, "driver's face not visible"

        if level == self.level:
            return None
        event = AlertEvent(t, datetime.now(), self.level, level, score, reason)
        self.level = level
        return event

    def _level_for_score(self, score: float) -> tuple[AlertLevel, str]:
        thresholds = self._config.thresholds
        level = AlertLevel.NONE
        for i, threshold in enumerate(thresholds, start=1):
            if score >= threshold:
                level = AlertLevel(i)
        # Hysteresis: hold the current level until the score clearly drops.
        if (
            self.level > AlertLevel.NONE
            and level < self.level
            and score >= thresholds[self.level - 1] - self._config.hysteresis
        ):
            level = self.level
        if level == AlertLevel.NONE:
            return level, "score back to normal"
        return level, f"drowsiness score {score:.2f} >= {thresholds[level - 1]:.2f}"


class EventLog:
    """Keeps alert events in memory and appends them to a per-session CSV."""

    FIELDS = ("wall_time", "session_time_s", "from_level", "to_level", "score", "reason")

    def __init__(self, log_dir: str | Path | None):
        self.events: list[AlertEvent] = []
        self.path: Path | None = None
        if log_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = Path(log_dir) / f"alerts_{stamp}.csv"

    def record(self, event: AlertEvent) -> None:
        self.events.append(event)
        log_fn = log.warning if event.level >= AlertLevel.ALARM else log.info
        log_fn("Alert %s -> %s: %s", event.previous.label, event.level.label, event.reason)
        if self.path is None:
            return
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.FIELDS)
            writer.writerow(
                [
                    event.wall_time.isoformat(timespec="seconds"),
                    f"{event.timestamp:.2f}",
                    event.previous.label,
                    event.level.label,
                    f"{event.score:.3f}",
                    event.reason,
                ]
            )
