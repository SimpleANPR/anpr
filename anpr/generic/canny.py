"""Canny edges.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class Canny(ImageProcessor):
    def __init__(self,
                 th1: float = 30,
                 th2: float = 200):
        """Construtor. Recebe como argumento o tamanho
        N do Kernel NxN que deve ser utilizado.

        Args:
            th1 (float), th2 (float): valores de limiares.
        """
        self.th1 = th1
        self.th2 = th2

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, threshold1=self.th1, threshold2=self.th2)
