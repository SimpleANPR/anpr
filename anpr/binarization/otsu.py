"""Binarização através do algoritmo
de Otsu.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Binarizer


class OtsuBinarizer(Binarizer):
    """Binarização através do método
    de Otsu.
    """

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        _, img = cv2.threshold(image,
                               0,
                               255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return img
