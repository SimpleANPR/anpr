"""Operação de negação de uma imagem.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class Negative(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(image)