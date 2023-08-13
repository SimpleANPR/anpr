"""Filtragem através do filtro
bilateral.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class HistogramNormalization(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            h, s, v = cv2.split(cv2.cvtColor(image, 
                                             cv2.COLOR_RGB2HSV))
            equalized_hsv = cv2.merge((h, s, cv2.equalizeHist(v)))
            equalized_hsv = cv2.cvtColor(equalized_hsv, 
                                         cv2.COLOR_HSV2RGB)
            return equalized_hsv

        return cv2.equalizeHist(image)