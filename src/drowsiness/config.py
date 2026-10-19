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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _require_positive(section: str, obj: object, *names: str) -> None:
    for name in names:
        _require(getattr(obj, name) > 0, f"{section}.{name} must be > 0")


def _require_fraction(section: str, obj: object, *names: str) -> None:
    for name in names:
        _require(0 <= getattr(obj, name) <= 1, f"{section}.{name} must be between 0 and 1")


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True

    def __post_init__(self) -> None:
        _require(self.index >= 0, "camera.index must be >= 0")
        _require_positive("camera", self, "width", "height")


@dataclass(frozen=True)
class LandmarkerConfig:
    model_path: str = "models/face_landmarker.task"
    max_faces: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        _require(self.max_faces >= 1, "landmarker.max_faces must be >= 1")
        _require_fraction(
            "landmarker",
            self,
            "min_detection_confidence",
            "min_presence_confidence",
            "min_tracking_confidence",
        )


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    duration_s: float = 5.0
    min_samples: int = 30
    # Eyes count as closed below this fraction of the driver's open-eye EAR.
    ear_closed_ratio: float = 0.60
    min_ear_threshold: float = 0.12
    max_ear_threshold: float = 0.30
    # Plausibility checks: a baseline outside these limits (eyes already closed,
    # looking far down) is rejected and calibration restarts.
    min_open_ear: float = 0.20
    max_open_ear: float = 0.45
    max_neutral_pitch_deg: float = 25.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        _require_positive("calibration", self, "duration_s", "min_samples", "max_attempts")
        _require(
            0 < self.ear_closed_ratio < 1, "calibration.ear_closed_ratio must be between 0 and 1"
        )
        _require(
            0 < self.min_ear_threshold <= self.max_ear_threshold,
            "calibration: need 0 < min_ear_threshold <= max_ear_threshold",
        )
        _require(
            0 < self.min_open_ear < self.max_open_ear,
            "calibration: need 0 < min_open_ear < max_open_ear",
        )
        _require_positive("calibration", self, "max_neutral_pitch_deg")


@dataclass(frozen=True)
class IndicatorConfig:
    # Eye closure
    default_ear_threshold: float = 0.21
    perclos_window_s: float = 20.0
    perclos_max: float = 0.4
    microsleep_s: float = 1.5
    # A closure only ends once the eyes stay open this long, so a single
    # noisy frame does not reset the microsleep timer.
    closure_debounce_s: float = 0.15
    # Yawning
    yawn_mar_threshold: float = 0.5
    yawn_min_duration_s: float = 1.5
    yawn_window_s: float = 60.0
    yawns_for_max: int = 3
    # A yawn in progress only ends once MAR stays below the threshold this long.
    yawn_debounce_s: float = 0.20
    # Head tilt, relative to the calibrated neutral pose
    head_pitch_threshold_deg: float = 15.0
    head_roll_threshold_deg: float = 20.0
    head_tilt_max_s: float = 2.0
    # A tilt only ends once the head stays upright this long.
    head_debounce_s: float = 0.20
    # Face missing for less than this keeps all indicator state (a dropped
    # frame); longer losses clear the continuous timers so no time passes
    # unobserved into a closure, yawn or tilt.
    face_loss_grace_s: float = 0.30

    def __post_init__(self) -> None:
        _require_positive(
            "indicators",
            self,
            "default_ear_threshold",
            "perclos_window_s",
            "perclos_max",
            "microsleep_s",
            "yawn_mar_threshold",
            "yawn_min_duration_s",
            "yawn_window_s",
            "yawns_for_max",
            "head_pitch_threshold_deg",
            "head_roll_threshold_deg",
            "head_tilt_max_s",
        )
        for name in (
            "closure_debounce_s",
            "yawn_debounce_s",
            "head_debounce_s",
            "face_loss_grace_s",
        ):
            _require(getattr(self, name) >= 0, f"indicators.{name} must be >= 0")


@dataclass(frozen=True)
class ConditionConfig:
    """Detection of conditions that make the eye cue unreliable (Phase 3)."""

    # Low light. Cameras brighten dark scenes automatically, so the face itself
    # rarely looks dark at night, but the frame as a whole does (the dark car,
    # the road outside). Low light starts when the smoothed whole-frame
    # brightness (0-255) falls below ``low_light_brightness`` and ends once it
    # rises ``low_light_margin`` above it. Tuned on real night (NITYMED) vs day
    # (YawDD, UTA-RLDD) footage with tools/tune_conditions.py.
    low_light_brightness: float = 76.0
    low_light_margin: float = 8.0
    # Eye-region occlusion: the eyes cannot be seen at all. Each frame counts as
    # "occluded" if any of these hold. Thresholds tuned on real footage
    # (tools/tune_conditions.py) to flag at most 3% of frames of uncovered eyes
    # in any dataset, day or night: a false occlusion would lower the eye weight
    # for a driver whose eyes are perfectly visible. Lightly tinted sunglasses
    # are deliberately not flagged; landmarks read the eyes through them as
    # well as through clear glasses.
    # glare: this fraction of eye-region pixels at or above ``glare_level``
    glare_level: int = 245
    glare_fraction: float = 0.15
    # flat eye region (dark lenses, a hand): eye contrast below this fraction
    # of the face's contrast
    min_eye_contrast_ratio: float = 0.10
    # one eye reads very differently from the other (|L - R| / max(L, R)),
    # checked only while the head faces roughly forward and at least one eye is
    # clearly open: with both eyes nearly shut the ratio of two tiny values is
    # meaningless, and closing the eyes must never look like an occlusion.
    max_ear_asymmetry: float = 0.5
    asymmetry_max_yaw_deg: float = 25.0
    asymmetry_min_open_ear: float = 0.20
    # The share of recent frames that were occluded is smoothed over
    # ``smoothing_s``; the condition starts above ``occlusion_enter`` and ends
    # below ``occlusion_exit``.
    occlusion_enter: float = 0.6
    occlusion_exit: float = 0.3
    smoothing_s: float = 1.0
    # If true, a microsleep does not force an alarm on its own while the eyes are
    # occluded (glare can make open eyes read as closed). Off by default: the eye
    # weight under occlusion (0.2) is below the warning threshold, so with it on
    # a real microsleep behind glare would raise no alert at all.
    microsleep_needs_visible_eyes: bool = False

    def __post_init__(self) -> None:
        _require(
            0 <= self.low_light_brightness <= 255,
            "conditions.low_light_brightness must be in 0..255",
        )
        _require(self.low_light_margin >= 0, "conditions.low_light_margin must be >= 0")
        _require(0 < self.glare_level <= 255, "conditions.glare_level must be in 1..255")
        _require_fraction("conditions", self, "glare_fraction", "min_eye_contrast_ratio")
        _require_positive(
            "conditions",
            self,
            "max_ear_asymmetry",
            "asymmetry_max_yaw_deg",
            "asymmetry_min_open_ear",
            "smoothing_s",
        )
        _require(
            0 <= self.occlusion_exit < self.occlusion_enter <= 1,
            "conditions: need 0 <= occlusion_exit < occlusion_enter <= 1",
        )


FUSION_STRATEGIES = ("adaptive", "static")


def _check_weights(name: str, weights: tuple[float, float, float]) -> None:
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ConfigError(f"fusion {name} weights must be non-negative and not all zero")


@dataclass(frozen=True)
class FusionConfig:
    # "adaptive" changes the weights with the detected conditions; "static"
    # always uses the normal-condition weights (the baseline).
    strategy: str = "adaptive"
    # Normal conditions; also the static baseline.
    eye_weight: float = 0.5
    yawn_weight: float = 0.2
    head_weight: float = 0.3
    # Low light: eye landmarks get noisy, so trust mouth and head more.
    low_light_eye_weight: float = 0.2
    low_light_yawn_weight: float = 0.35
    low_light_head_weight: float = 0.45
    # Eye region occluded (glare, dark glasses, hand).
    occluded_eye_weight: float = 0.2
    occluded_yawn_weight: float = 0.35
    occluded_head_weight: float = 0.45
    # Weights move to a new condition's values over this long instead of jumping.
    transition_s: float = 1.0

    def __post_init__(self) -> None:
        _require(
            self.strategy in FUSION_STRATEGIES,
            f"fusion.strategy must be one of {', '.join(FUSION_STRATEGIES)}",
        )
        _check_weights("normal", (self.eye_weight, self.yawn_weight, self.head_weight))
        _check_weights(
            "low_light",
            (self.low_light_eye_weight, self.low_light_yawn_weight, self.low_light_head_weight),
        )
        _check_weights(
            "occluded",
            (self.occluded_eye_weight, self.occluded_yawn_weight, self.occluded_head_weight),
        )
        _require(self.transition_s >= 0, "fusion.transition_s must be >= 0")


@dataclass(frozen=True)
class AlertConfig:
    level1_threshold: float = 0.25
    level2_threshold: float = 0.45
    level3_threshold: float = 0.65
    hysteresis: float = 0.05
    escalate_after_s: float = 5.0
    # Once raised, an alarm (or critical) sounds for at least this long, so it
    # does not stutter on and off.
    level2_hold_s: float = 1.5
    level3_hold_s: float = 5.0
    # A microsleep (eyes closed for indicators.microsleep_s) always raises at
    # least an alarm, whatever the fusion weights.
    microsleep_alarm: bool = True
    # Losing the driver's face escalates too: the head may have slumped out of
    # view or the camera may be blocked.
    no_face_warning_s: float = 3.0
    no_face_alarm_s: float = 6.0
    no_face_critical_s: float = 10.0
    sound_enabled: bool = True
    sound_path: str = "assets/alarm.wav"
    log_dir: str = "logs"

    def __post_init__(self) -> None:
        if not 0 < self.level1_threshold < self.level2_threshold < self.level3_threshold <= 1:
            raise ConfigError("alert thresholds must satisfy 0 < level1 < level2 < level3 <= 1")
        _require(self.hysteresis >= 0, "alerts.hysteresis must be >= 0")
        _require_positive("alerts", self, "escalate_after_s", "no_face_warning_s")
        _require(self.level2_hold_s >= 0, "alerts.level2_hold_s must be >= 0")
        _require(self.level3_hold_s >= 0, "alerts.level3_hold_s must be >= 0")
        _require(
            self.no_face_warning_s < self.no_face_alarm_s < self.no_face_critical_s,
            "alerts: need no_face_warning_s < no_face_alarm_s < no_face_critical_s",
        )

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return (self.level1_threshold, self.level2_threshold, self.level3_threshold)


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    conditions: ConditionConfig = field(default_factory=ConditionConfig)
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
