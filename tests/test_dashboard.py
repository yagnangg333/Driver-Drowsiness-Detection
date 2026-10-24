"""Smoke test of the Streamlit dashboard in its stopped state (no camera needed)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import drowsiness

APP = Path(drowsiness.__file__).with_name("dashboard.py")


def test_dashboard_renders_with_disclaimer():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Driver Drowsiness Detection"
    assert any("not a safety device" in c.value for c in app.caption)
    assert any("Detection stopped" in i.value for i in app.info)
    assert any("Trained model" in h.value for h in app.subheader)


def test_invalid_settings_disable_start():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    # Level 1 above level 2 is invalid: the start toggle must be disabled.
    app.sidebar.slider[3].set_value(0.9).run()
    assert not app.exception
    assert app.sidebar.error and "alert thresholds" in app.sidebar.error[0].value
    assert app.toggle[0].disabled
