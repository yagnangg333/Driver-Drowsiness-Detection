"""The interface shared by every fusion strategy.

A strategy turns the three indicator scores (eye, yawn, head, each 0-1) into
one drowsiness score with weights that sum to 1. Every strategy receives the
same scores and conditions, so comparisons isolate the effect of the weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from drowsiness.fusion.adaptive.conditions import NORMAL, Conditions


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


def weighted(w: Weights, scores: IndicatorScores) -> float:
    return w.eye * scores.eye + w.yawn * scores.yawn + w.head * scores.head
