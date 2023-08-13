"""Interpolação usando o algoritmo
do Vizinho Mais Próximo.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Resizer


class Nearest(Resizer):
    def __init__(self, target_resolution: tuple[int, int]) -> None:
        self._res = target_resolution

    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, dsize=self._res, interpolation = cv2.INTER_NEAREST)