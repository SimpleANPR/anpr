"""Operação morfológica de 
fechamento.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class MorphologicalClosing(ImageProcessor):
    def __init__(self, kernel: np.ndarray):
        self._kernel = kernel

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(src=image, 
                                op=cv2.MORPH_CLOSE, 
                                kernel=self._kernel)