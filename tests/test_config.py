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
    ],
)
def test_invalid_config_rejected(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
