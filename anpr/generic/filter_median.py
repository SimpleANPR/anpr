"""Filtragem através do filtro
mediana.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor



class FilterMedian(ImageProcessor):
    def __init__(self, kernel_size: int):
        """Construtor. Recebe como argumento o tamanho
        N do Kernel NxN que deve ser utilizado.

        Args:
            kernel_size (int): tamanho do kernel.
        """
        self._kernel_size = kernel_size

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, self._kernel_size)