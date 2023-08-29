"""sobel filter.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class Sobel(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        abs_grad_x = cv2.convertScaleAbs(sobelx)
        abs_grad_y = cv2.convertScaleAbs(sobely)

        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
