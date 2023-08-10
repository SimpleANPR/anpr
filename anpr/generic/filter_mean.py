"""Filtragem através do filtro
média.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor


class FilterMean(ImageProcessor):
    def __init__(self, kernel_size: int):
        """Construtor. Recebe como argumento o tamanho
        N do Kernel NxN que deve ser utilizado.

        Args:
            kernel_size (int): tamanho do kernel.
        """
        self._kernel = np.ones((kernel_size, kernel_size),
                               dtype=np.float32) / (kernel_size ** 2)


    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self._kernel)