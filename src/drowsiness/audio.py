"""Alarm sound for alert levels 2 (alarm) and 3 (critical)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from drowsiness.alerts import AlertLevel

log = logging.getLogger(__name__)

_VOLUME = {AlertLevel.ALARM: 0.5, AlertLevel.CRITICAL: 1.0}


class AlarmPlayer:
    """Loops the alarm while the level is ALARM or above; no-op without audio."""

    def __init__(self, sound_path: str | Path, enabled: bool = True):
        self._mixer = None
        self._playing = False
        if not enabled:
            return
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            from pygame import mixer

            if not mixer.get_init():
                mixer.init()
            mixer.music.load(str(sound_path))
            self._mixer = mixer
        except Exception as exc:  # no audio device, missing file, codec error, ...
            log.warning("Alarm sound disabled: %s", exc)

    @property
    def available(self) -> bool:
        return self._mixer is not None

    def set_level(self, level: AlertLevel) -> None:
        if self._mixer is None:
            return
        if level >= AlertLevel.ALARM:
            self._mixer.music.set_volume(_VOLUME[level])
            if not self._playing:
                self._mixer.music.play(loops=-1)
                self._playing = True
        elif self._playing:
            self._mixer.music.stop()
            self._playing = False

    def stop(self) -> None:
        self.set_level(AlertLevel.NONE)
