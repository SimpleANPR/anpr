"""Esse módulo contém um
extrator de características
de cor.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_central, moments_hu, moments_normalized

from anpr.core.feature_extractor import FeatureExtractor


class TextureFeatures(FeatureExtractor):
    def __init__(self,
                 color_model: str = 'RGB',
                 moments_order: int = 5) -> None:
        self._color_model = color_model
        self._order = moments_order
        self._distances = [1]
        self._angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = image

        if len(image.shape) > 2:
            cvt = f'COLOR_{self._color_model}2GRAY'
            cvt_code = getattr(cv2, cvt)
            gray = cv2.cvtColor(image, cvt_code)

        return np.concatenate([self._glcm_features(gray),
                               self._hu_moments(gray)])

    def _glcm_features(self, image: np.ndarray) -> np.ndarray:
        gray_com = graycomatrix(image,
                                self._distances,
                                self._angles,
                                levels=256)
        contrast = graycoprops(gray_com, 'contrast').flatten()
        dissimilarity = graycoprops(gray_com, 'dissimilarity').flatten()
        homogeneity = graycoprops(gray_com, 'homogeneity').flatten()
        energy = graycoprops(gray_com, 'energy').flatten()
        correlation = graycoprops(gray_com, 'correlation').flatten()
        ASM = graycoprops(gray_com, 'ASM').flatten()

        return np.concatenate([contrast, dissimilarity, homogeneity,
                               energy, correlation, ASM])

    def _hu_moments(self, image: np.ndarray) -> np.ndarray:
        mu = moments_central(image, order=self._order)
        nu = moments_normalized(mu, order=self._order)
        return moments_hu(nu)
