"""Difference of Gaussians (DoG).
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class DifferenceOfGaussians(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        # Apply 3x3 and 7x7 Gaussian blur
        low_sigma = cv2.GaussianBlur(image, (3, 3), 0)
        high_sigma = cv2.GaussianBlur(image, (5, 5), 0)

        # Calculate the DoG by subtracting
        dog = low_sigma - high_sigma

        return dog
