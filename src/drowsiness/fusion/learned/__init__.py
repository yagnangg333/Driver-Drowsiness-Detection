"""Learned fusion: a logistic regression trained on labelled video.

Instead of hand-set weights, a small, explainable model learns how to combine
13 windowed features. It was evaluated against the rules (see EVALUATION.md)
and not adopted; the dashboard runs it live for comparison only. The safety
rules stay in the alert manager whatever the model says.

- ``features.py`` the 13 features, computed causally (only from the past), the
  same code for training and for live use.
- ``model.py``    the model: L2-regularised logistic regression trained by
  iteratively reweighted least squares in NumPy, saved as a short JSON file.
- ``live.py``     runs a saved model on the live pipeline's output.

Training scripts: ``tools/learned_fusion.py`` and ``tools/learned_variants.py``.
The trained model: ``models/learned_fusion.json``.
"""

from drowsiness.fusion.learned.features import FEATURE_LABELS, FEATURES, FeatureTracker
from drowsiness.fusion.learned.live import LiveLearnedFusion
from drowsiness.fusion.learned.model import LogisticModel

__all__ = ["FEATURES", "FEATURE_LABELS", "FeatureTracker", "LiveLearnedFusion", "LogisticModel"]
