"""Binarização através do algoritmo
de limiarização global.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Binarizer


class SimpleBinarizer(Binarizer):
    """Binarização através do método
    de Otsu.
    """
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        _, img = cv2.threshold(image,
                               self.threshold,
                               255,
                               cv2.THRESH_BINARY)
        return img