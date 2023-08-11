from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Resizer

class Bilinear(Resizer):
    def __init__(self, target_resolution: tuple[int, int]) -> None:
        self._res = target_resolution

    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(src=image, dsize=self._res, interpolation=cv2.INTER_LINEAR)

    def process(self, image: np.ndarray) -> np.ndarray:
        return self.resize(image)