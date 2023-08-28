"""Interpolação usando o algoritmo
do Vizinho Mais Próximo.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Resizer


class Nearest(Resizer):
    def __init__(self,
                 target_resolution: tuple[int, int],
                 factor: float = None) -> None:
        self._res = target_resolution
        self._f = factor

    def resize(self, image: np.ndarray) -> np.ndarray:
        if self._f is not None:
            return cv2.resize(src=image,
                              dsize=None,
                              fx=self._f,
                              fy=self._f,
                              interpolation=cv2.INTER_NEAREST)

        return cv2.resize(image,
                          dsize=self._res,
                          interpolation=cv2.INTER_NEAREST)
