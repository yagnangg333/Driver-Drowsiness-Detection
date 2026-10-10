"""Fusion of indicator scores into a single drowsiness score.

``StaticFusion`` is the fixed-weight baseline the adaptive, condition-aware
fusion will be compared against. Every strategy receives the same indicator
scores, so comparisons isolate the effect of the weighting alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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

    def fuse(self, scores: IndicatorScores) -> FusionResult: ...


class StaticFusion:
    """Weighted average with fixed, condition-independent weights."""

    name = "static"

    def __init__(self, config: FusionConfig):
        self.weights = Weights(
            config.eye_weight, config.yawn_weight, config.head_weight
        ).normalized()

    def fuse(self, scores: IndicatorScores) -> FusionResult:
        w = self.weights
        score = w.eye * scores.eye + w.yawn * scores.yawn + w.head * scores.head
        return FusionResult(score, w)
