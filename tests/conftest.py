from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness.config import LandmarkerConfig, resolve_path

FIXTURES = Path(__file__).parent / "fixtures"

requires_model = pytest.mark.skipif(
    not resolve_path(LandmarkerConfig().model_path).is_file(),
    reason="face landmarker model not downloaded (run `drowsiness download-model`)",
)


@pytest.fixture(scope="session")
def awake_frame() -> np.ndarray:
    return cv2.imread(str(FIXTURES / "awake.png"))


@pytest.fixture(scope="session")
def drowsy_frame() -> np.ndarray:
    return cv2.imread(str(FIXTURES / "drowsy.png"))
