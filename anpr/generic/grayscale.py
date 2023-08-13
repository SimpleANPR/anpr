"""Mudança de RGB para Escala de 
Cinza.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class GrayScale(ImageProcessor):
    def __init__(self, conversion: int = cv2.COLOR_BGR2GRAY):
        self._conversion = conversion

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, self._conversion)