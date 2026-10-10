import urllib.error

import pytest

from drowsiness import cli, landmarks


@pytest.fixture
def captured_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command: calls.append(command) or 0)
    return calls


def _option(command, name):
    return command[command.index(name) + 1]


def test_dashboard_is_local_only_by_default(captured_command):
    assert cli.main(["dashboard"]) == 0
    assert _option(captured_command[0], "--server.address") == "127.0.0.1"


def test_dashboard_network_exposure_is_explicit_and_warned(captured_command, caplog):
    assert cli.main(["dashboard", "--address", "0.0.0.0", "--port", "9000"]) == 0
    command = captured_command[0]
    assert _option(command, "--server.address") == "0.0.0.0"
    assert _option(command, "--server.port") == "9000"
    assert "anyone who can reach" in caplog.text


def test_invalid_config_is_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[indicators]\nmicrosleep_s = 0.0\n", encoding="utf-8")
    assert cli.main(["process", str(tmp_path / "x.mp4"), "--config", str(bad)]) == 1


def test_download_failure_is_a_clean_error(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(landmarks.urllib.request, "urlopen", offline)
    target = tmp_path / "model.task"
    with pytest.raises(RuntimeError, match="Could not download"):
        landmarks.download_model(target, force=True)
    assert not target.exists() and not target.with_suffix(".task.part").exists()
