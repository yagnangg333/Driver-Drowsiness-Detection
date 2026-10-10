import sys
import types

import pytest

from drowsiness.alerts import AlertLevel
from drowsiness.audio import AlarmPlayer


class FakeMusic:
    def __init__(self):
        self.calls = []

    def load(self, path):
        self.calls.append(("load", path))

    def play(self, loops=0):
        self.calls.append(("play", loops))

    def stop(self):
        self.calls.append(("stop",))

    def set_volume(self, volume):
        self.calls.append(("volume", volume))


@pytest.fixture
def fake_pygame(monkeypatch):
    music = FakeMusic()
    mixer = types.SimpleNamespace(get_init=lambda: True, init=lambda: None, music=music)
    monkeypatch.setitem(sys.modules, "pygame", types.SimpleNamespace(mixer=mixer))
    return music


def test_alarm_follows_alert_level(fake_pygame):
    player = AlarmPlayer("alarm.wav")
    assert player.available
    player.set_level(AlertLevel.WARNING)  # visual only
    player.set_level(AlertLevel.ALARM)
    player.set_level(AlertLevel.CRITICAL)  # louder, not restarted
    player.set_level(AlertLevel.NONE)
    assert fake_pygame.calls == [
        ("load", "alarm.wav"),
        ("volume", 0.5),
        ("play", -1),
        ("volume", 1.0),
        ("stop",),
    ]


def test_disabled_alarm_is_silent(fake_pygame):
    player = AlarmPlayer("alarm.wav", enabled=False)
    player.set_level(AlertLevel.CRITICAL)
    assert not player.available and fake_pygame.calls == []


def test_missing_audio_device_degrades_gracefully(monkeypatch):
    def broken_init():
        raise RuntimeError("no audio device")

    mixer = types.SimpleNamespace(get_init=lambda: False, init=broken_init, music=FakeMusic())
    monkeypatch.setitem(sys.modules, "pygame", types.SimpleNamespace(mixer=mixer))
    player = AlarmPlayer("alarm.wav")
    assert not player.available
    player.set_level(AlertLevel.CRITICAL)  # must not raise
