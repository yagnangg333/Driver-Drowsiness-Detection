"""Fusion of indicator scores into a single drowsiness score.

``StaticFusion`` is the fixed-weight baseline. ``AdaptiveFusion`` changes the
weights with the detected conditions (low light, eye occlusion) and, under
normal conditions, uses exactly the static weights. Every strategy receives the
same indicator scores, so comparisons isolate the effect of the weighting alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from drowsiness.conditions import NORMAL, Conditions
from drowsiness.config import FusionConfig


@dataclass(frozen=True)
class IndicatorScores:
    eye: float
    yawn: float
    head: float


@dataclass(frozen=True)
class Weights:
    eye: float
    yawn: float
    head: float

    def normalized(self) -> Weights:
        total = self.eye + self.yawn + self.head
        return Weights(self.eye / total, self.yawn / total, self.head / total)


@dataclass(frozen=True)
class FusionResult:
    score: float
    weights: Weights


class FusionStrategy(Protocol):
    name: str

    def fuse(
        self, scores: IndicatorScores, conditions: Conditions = NORMAL, t: float = 0.0
    ) -> FusionResult: ...

    def reset(self) -> None: ...


def _weighted(w: Weights, scores: IndicatorScores) -> float:
    return w.eye * scores.eye + w.yawn * scores.yawn + w.head * scores.head


class StaticFusion:
    """Weighted average with fixed, condition-independent weights."""

    name = "static"

    def __init__(self, config: FusionConfig):
        self.weights = Weights(
            config.eye_weight, config.yawn_weight, config.head_weight
        ).normalized()

    def fuse(
        self, scores: IndicatorScores, conditions: Conditions = NORMAL, t: float = 0.0
    ) -> FusionResult:
        return FusionResult(_weighted(self.weights, scores), self.weights)

    def reset(self) -> None:
        pass


class AdaptiveFusion:
    """Condition-aware weights: w_i(C) = R_i(C) / sum_j R_j(C).

    Each condition has its own weight row. When several conditions hold at once
    every cue gets its lowest weight among them (it is trusted only as much as
    the worst condition allows) before normalising. The weights move towards
    their new values over ``transition_s`` rather than jumping, so a condition
    change does not make the score jump.
    """

    name = "adaptive"

    def __init__(self, config: FusionConfig):
        self._normal = Weights(config.eye_weight, config.yawn_weight, config.head_weight)
        self._low_light = Weights(
            config.low_light_eye_weight, config.low_light_yawn_weight, config.low_light_head_weight
        )
        self._occluded = Weights(
            config.occluded_eye_weight, config.occluded_yawn_weight, config.occluded_head_weight
        )
        self._transition_s = config.transition_s
        self.reset()

    def reset(self) -> None:
        self._current = self._normal.normalized()
        self._last_t: float | None = None

    def target(self, conditions: Conditions) -> Weights:
        rows = []
        if conditions.low_light:
            rows.append(self._low_light)
        if conditions.eye_occluded:
            rows.append(self._occluded)
        if not rows:
            return self._normal.normalized()
        return Weights(
            min(r.eye for r in rows), min(r.yawn for r in rows), min(r.head for r in rows)
        ).normalized()

    def fuse(
        self, scores: IndicatorScores, conditions: Conditions = NORMAL, t: float = 0.0
    ) -> FusionResult:
        target = self.target(conditions)
        if self._last_t is None or self._transition_s == 0:
            self._current = target
        else:
            # Move each weight at most dt / transition_s per frame; all three move
            # by the same fraction of the way, so they keep summing to 1.
            step = max(t - self._last_t, 0.0) / self._transition_s
            gap = max(
                abs(target.eye - self._current.eye),
                abs(target.yawn - self._current.yawn),
                abs(target.head - self._current.head),
            )
            f = 1.0 if gap <= step else step / gap
            c = self._current
            self._current = Weights(
                c.eye + f * (target.eye - c.eye),
                c.yawn + f * (target.yawn - c.yawn),
                c.head + f * (target.head - c.head),
            )
        self._last_t = t
        return FusionResult(_weighted(self._current, scores), self._current)


def make_fusion(config: FusionConfig, strategy: str | None = None) -> FusionStrategy:
    """Build the fusion strategy named in the config (or ``strategy``)."""
    strategy = strategy or config.strategy
    if strategy == "adaptive":
        return AdaptiveFusion(config)
    if strategy == "static":
        return StaticFusion(config)
    raise ValueError(f"unknown fusion strategy: {strategy}")
