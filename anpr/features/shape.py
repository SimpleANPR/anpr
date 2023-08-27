"""Esse módulo contém um
extrator de características
de formato.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core.feature_extractor import FeatureExtractor
from anpr.binarization.kittler import KittlerBinarizer


class ShapeFeatures(FeatureExtractor):
    def __init__(self,
                 color_model: str = 'RGB') -> None:
        self._color_model = color_model

    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            cvt = f'COLOR_{self._color_model}2GRAY'
            cvt_code = getattr(cv2, cvt)
            image = cv2.cvtColor(image, cvt_code)

        # Obtendo imagem binária
        bin = KittlerBinarizer().binarize(image)

        # Obtendo contornos
        contours, _ = cv2.findContours(bin,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Obtendo estatísticas dos polígonos
        n_polygons = self._polygons(contours)

        # Obtendo estatísticas sobre as áreas dos contornos
        areas_statistics = self._contours_area(contours)

        # Obtendo estatísticas sobre convexidade dos contornos
        convexity_statistics = self._contours_convexity(contours)

        return np.concatenate([n_polygons, areas_statistics, convexity_statistics])

    def _polygons(self,
                  contours: list[np.ndarray]) -> np.ndarray:
        sides = np.array([len(cv2.approxPolyDP(c,
                                               epsilon=0.1 *
                                               cv2.arcLength(c, True),
                                               closed=True))
                          for c in contours])

        if sides.shape[0] <= 0:
            sides = np.zeros((4,))

        return np.array([sides.max(), sides.min(),
                         sides.mean(), sides.std()])

    def _contours_area(self, contours: list[np.ndarray]) -> np.ndarray:
        areas = np.array([cv2.contourArea(c) for c in contours])

        if areas.shape[0] <= 0:
            areas = np.zeros((4,))

        return np.array([areas.max(), areas.min(),
                         areas.mean(), areas.std()])

    def _contours_convexity(self, contours: list[np.ndarray]) -> np.ndarray:
        convexity = np.array([1 if cv2.isContourConvex(c) else 0
                              for c in contours])

        if convexity.shape[0] <= 0:
            convexity = np.zeros((4,))

        return np.array([convexity.mean(), convexity.std()])
