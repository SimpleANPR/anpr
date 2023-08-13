"""Filtragem através do filtro
bilateral.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class FilterBilateral(ImageProcessor):
    def __init__(self, kernel_size: int):
        self._kernel = kernel_size

    def process(self, image: np.ndarray, sigma_space: int, sigma_color: int) -> np.ndarray:
        return cv2.bilateralFilter(src=image, 
                                   d=self._kernel, 
                                   sigmaColor=sigma_color, 
                                   sigmaSpace=sigma_space)