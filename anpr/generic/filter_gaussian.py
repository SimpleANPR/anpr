"""Filtragem através do filtro
gaussiano.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class FilterGaussian(ImageProcessor):
    def __init__(self, kernel_size: int):
        self._kernel = kernel_size

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(src=image, 
                                ksize=(self._kernel,self._kernel), 
                                sigmaX=0,
                                sigmaY=0)