"""Filtragem através do filtro
bilateral.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class Brightness(ImageProcessor):
    def __init__(self, offset: float) -> None:
        self._offset = float(offset)

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, beta=self._offset)