"""Binarização local através do algoritmo
de Adaptative Gaussian.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Binarizer


class AdaptativeGaussianBinarizer(Binarizer):
    """Binarização local através do método
    Adaptative Gaussian.
    """

    def __init__(self,
                 block_size: int,
                 C: int) -> None:
        self.block_size = block_size
        self.C = C

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(image,
                                     255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     self.block_size,
                                     self.C)
