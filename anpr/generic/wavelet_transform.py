"""Filtragem através do filtro
de transformada wavelet.
"""
from __future__ import annotations

import numpy as np
import pywt

from anpr.core import ImageProcessor


class WaveletTransform(ImageProcessor):
    def __init__(self, wavelet: str = 'haar'):
        self.wavelet = wavelet

    def process(self, image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray]]:
        return pywt.dwt2(image, self.wavelet)