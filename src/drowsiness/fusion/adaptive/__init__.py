"""Adaptive (condition-aware) fusion: the project's method.

- ``conditions.py`` detects, on every frame, conditions that mislead the eye
  cue: smiling, eyes occluded (glare, dark lenses, a hand) and low light.
- ``fusion.py`` adapts the weights to them. The pipeline also stops counting
  eye closures while the driver smiles (its adaptive eye indicator).
"""
