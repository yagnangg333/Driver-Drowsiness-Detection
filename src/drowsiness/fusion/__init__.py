"""Fusion: how the three cue scores become one drowsiness score.

Three strategies, one folder each, all fed the same indicator scores:

- ``static/``   fixed weights (eyes 0.5, yawn 0.2, head 0.3): the baseline.
- ``adaptive/`` detects conditions that mislead the eye cue (smiling, covered
  eyes, low light) and adapts to them: the project's method.
- ``learned/``  a logistic regression trained on labelled video; shown live for
  comparison, never drives the alarm.

``base.py`` holds the interface every strategy implements. The safety rules (a
microsleep or a lost face forcing an alarm) live in ``alerts.py``, outside every
strategy.
"""

from drowsiness.fusion.adaptive.fusion import AdaptiveFusion
from drowsiness.fusion.base import (
    FusionResult,
    FusionStrategy,
    IndicatorScores,
    Weights,
    weighted,
)
from drowsiness.fusion.static.fusion import StaticFusion

__all__ = [
    "AdaptiveFusion",
    "FusionResult",
    "FusionStrategy",
    "IndicatorScores",
    "StaticFusion",
    "Weights",
    "weighted",
]
