"""Esse módulo contém um
extrator de características
de cor.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core.feature_extractor import FeatureExtractor


class ColorFeatures(FeatureExtractor):
    def __init__(self,
                 color_model: str = 'RGB') -> None:
        self._color_model = color_model

    def extract(self, image: np.ndarray) -> np.ndarray:
        grayscale = len(image.shape) < 3
        features = [self._color_statistics(image,
                                           grayscale,
                                           m)
                    for m in ['RGB', 'HSV', 'YCrCb',
                              'Lab', 'Luv', 'YUV']]

        return np.array(features).flatten()

    def _color_statistics(self,
                          image: np.ndarray,
                          grayscale: bool,
                          model: str = 'RGB') -> list[float]:
        if not grayscale:
            if model != self._color_model:
                cvt = f'COLOR_{self._color_model}2{model}'
                cvt_code = getattr(cv2, cvt)
                image = cv2.cvtColor(image, cvt_code)

        return [image.max(), image.min(),
                image.mean(), image.std()]
