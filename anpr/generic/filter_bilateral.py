"""Filtragem através do filtro
bilateral.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class FilterBilateral(ImageProcessor):
    def __init__(self, kernel_size: int,
                 sigma_space: int,
                 sigma_color: int):
        self._kernel = kernel_size
        self._space = sigma_space
        self._color = sigma_color

    def process(self, image: np.ndarray, ) -> np.ndarray:
        return cv2.bilateralFilter(src=image,
                                   d=self._kernel,
                                   sigmaColor=self._color,
                                   sigmaSpace=self._space)
