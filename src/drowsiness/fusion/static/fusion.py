"""Static fusion: a weighted average with fixed, condition-independent weights.

The baseline. Every eye closure counts, whatever the conditions. Weights come
from ``fusion.eye_weight``, ``yawn_weight`` and ``head_weight`` in the config.
"""

from __future__ import annotations

from drowsiness.config import FusionConfig
from drowsiness.fusion.adaptive.conditions import NORMAL, Conditions
from drowsiness.fusion.base import FusionResult, IndicatorScores, Weights, weighted


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
        return FusionResult(weighted(self.weights, scores), self.weights)

    def reset(self) -> None:
        pass
