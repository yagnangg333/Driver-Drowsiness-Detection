"""Runtime configuration.

Every tunable value has a default in the dataclasses below. A TOML file
(see ``configs/default.toml``) can override any subset of them, section by
section. Unknown sections or keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path; relative paths are taken from the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True


@dataclass(frozen=True)
class LandmarkerConfig:
    model_path: str = "models/face_landmarker.task"
    max_faces: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    duration_s: float = 5.0
    min_samples: int = 30
    # Eyes count as closed below this fraction of the driver's open-eye EAR.
    ear_closed_ratio: float = 0.75
    min_ear_threshold: float = 0.12
    max_ear_threshold: float = 0.30

    def __post_init__(self) -> None:
        if not 0 < self.ear_closed_ratio < 1:
            raise ConfigError("calibration.ear_closed_ratio must be between 0 and 1")
        if self.min_ear_threshold > self.max_ear_threshold:
            raise ConfigError("calibration.min_ear_threshold must be <= max_ear_threshold")


@dataclass(frozen=True)
class IndicatorConfig:
    # Eye closure
    default_ear_threshold: float = 0.21
    perclos_window_s: float = 20.0
    perclos_max: float = 0.4
    microsleep_s: float = 1.5
    # Yawning
    yawn_mar_threshold: float = 0.5
    yawn_min_duration_s: float = 1.5
    yawn_window_s: float = 60.0
    yawns_for_max: int = 3
    # Head tilt, relative to the calibrated neutral pose
    head_pitch_threshold_deg: float = 15.0
    head_roll_threshold_deg: float = 20.0
    head_tilt_max_s: float = 2.0


@dataclass(frozen=True)
class FusionConfig:
    eye_weight: float = 0.5
    yawn_weight: float = 0.2
    head_weight: float = 0.3

    def __post_init__(self) -> None:
        weights = (self.eye_weight, self.yawn_weight, self.head_weight)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ConfigError("fusion weights must be non-negative and not all zero")


@dataclass(frozen=True)
class AlertConfig:
    level1_threshold: float = 0.25
    level2_threshold: float = 0.45
    level3_threshold: float = 0.65
    hysteresis: float = 0.05
    escalate_after_s: float = 5.0
    level3_hold_s: float = 5.0
    no_face_warning_s: float = 3.0
    sound_enabled: bool = True
    sound_path: str = "assets/alarm.wav"
    log_dir: str = "logs"

    def __post_init__(self) -> None:
        if not 0 < self.level1_threshold < self.level2_threshold < self.level3_threshold <= 1:
            raise ConfigError("alert thresholds must satisfy 0 < level1 < level2 < level3 <= 1")

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return (self.level1_threshold, self.level2_threshold, self.level3_threshold)


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the defaults, overridden by the TOML file at ``path`` if given."""
    if path is None:
        return AppConfig()
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    sections = {f.name: f.default_factory for f in fields(AppConfig)}  # type: ignore[misc]
    unknown = set(data) - set(sections)
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(sorted(unknown))}")

    kwargs = {}
    for name, section_cls in sections.items():
        values = data.get(name, {})
        defaults = {f.name: f.default for f in fields(section_cls)}
        bad = set(values) - set(defaults)
        if bad:
            raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(sorted(bad))}")
        for key, value in values.items():
            _check_type(f"{name}.{key}", value, defaults[key])
        try:
            kwargs[name] = section_cls(**values)
        except TypeError as exc:
            raise ConfigError(f"invalid value in [{name}]: {exc}") from None
    return AppConfig(**kwargs)


def _check_type(key: str, value: Any, default: Any) -> None:
    expected = type(default)
    # TOML integers are fine where a float is expected; bools are not numbers here.
    ok = isinstance(value, expected) and not (expected is not bool and isinstance(value, bool))
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        ok = True
    if not ok:
        raise ConfigError(f"{key} must be {expected.__name__}, got {value!r}")
