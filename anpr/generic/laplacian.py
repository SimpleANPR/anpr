"""laplacian filter.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor



class Laplacian(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(cv2.Laplacian(image,cv2.CV_8U), alpha=0.25)
