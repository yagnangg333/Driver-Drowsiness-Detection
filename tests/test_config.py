import pytest

from drowsiness.config import (
    PROJECT_ROOT,
    AppConfig,
    ConfigError,
    FusionConfig,
    config_from_dict,
    load_config,
)


def test_default_toml_matches_builtin_defaults():
    assert load_config(PROJECT_ROOT / "configs" / "default.toml") == AppConfig()


def test_partial_override_keeps_other_defaults():
    config = config_from_dict({"fusion": {"eye_weight": 0.7}})
    assert config.fusion.eye_weight == 0.7
    assert config.fusion.head_weight == FusionConfig().head_weight
    assert config.alerts == AppConfig().alerts


def test_int_accepted_for_float():
    assert config_from_dict({"calibration": {"duration_s": 3}}).calibration.duration_s == 3


@pytest.mark.parametrize(
    "data, message",
    [
        ({"camra": {}}, "unknown config section"),
        ({"camera": {"idx": 1}}, "unknown key"),
        ({"camera": {"index": "zero"}}, "must be int"),
        ({"camera": {"mirror": 1}}, "must be bool"),
        ({"alerts": {"level1_threshold": 0.9}}, "thresholds"),
        ({"fusion": {"eye_weight": 0, "yawn_weight": 0, "head_weight": 0}}, "weights"),
        # Values that used to pass validation and then crash or disable detection.
        ({"indicators": {"perclos_window_s": 0.0}}, "perclos_window_s must be > 0"),
        ({"indicators": {"microsleep_s": 0.0}}, "microsleep_s must be > 0"),
        ({"indicators": {"yawns_for_max": 0}}, "yawns_for_max must be > 0"),
        ({"indicators": {"default_ear_threshold": -1.0}}, "default_ear_threshold"),
        ({"calibration": {"min_samples": 0}}, "min_samples must be > 0"),
        ({"calibration": {"duration_s": 0.0}}, "duration_s must be > 0"),
        ({"alerts": {"hysteresis": -1.0}}, "hysteresis"),
        ({"alerts": {"no_face_alarm_s": 2.0}}, "no_face_warning_s < no_face_alarm_s"),
        ({"landmarker": {"max_faces": 0}}, "max_faces"),
        ({"landmarker": {"min_detection_confidence": 1.5}}, "between 0 and 1"),
        ({"camera": {"width": 0}}, "width must be > 0"),
        ({"fusion": {"strategy": "magic"}}, "fusion.strategy"),
        (
            {
                "fusion": {
                    "low_light_eye_weight": 0,
                    "low_light_yawn_weight": 0,
                    "low_light_head_weight": 0,
                }
            },
            "low_light weights",
        ),
        ({"fusion": {"transition_s": -1.0}}, "transition_s"),
        ({"conditions": {"low_light_brightness": 300.0}}, "low_light_brightness must be in"),
        ({"conditions": {"low_light_margin": -1.0}}, "low_light_margin"),
        ({"conditions": {"occlusion_exit": 0.9}}, "occlusion_exit < occlusion_enter"),
        ({"conditions": {"glare_fraction": 2.0}}, "between 0 and 1"),
        ({"conditions": {"smoothing_s": 0.0}}, "smoothing_s must be > 0"),
        ({"conditions": {"asymmetry_min_open_ear": 0.0}}, "asymmetry_min_open_ear"),
    ],
)
def test_invalid_config_rejected(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_microsleep_alarm_not_gated_by_default():
    # Safety default: a microsleep behind glare must still raise the alarm.
    assert AppConfig().conditions.microsleep_needs_visible_eyes is False
