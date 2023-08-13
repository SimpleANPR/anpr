"""Filtragem através do filtro
bilateral.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class Contrast(ImageProcessor):
    def __init__(self, factor: float) -> None:
        self._factor = float(factor)

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=self._factor)