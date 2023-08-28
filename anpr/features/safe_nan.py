"""Esse módulo contém um agregador de
características.
"""
from __future__ import annotations

import numpy as np

from anpr.core.feature_extractor import FeatureExtractor


class SafeNanExtractor(FeatureExtractor):
    """Agregador de etapas de processamento
    sequencial. Aplica cada um dos processamentos
    de forma sequencial (i.e., saída de um é entrada
    para o próximo).
    """

    def __init__(self, 
                 extractor: FeatureExtractor,
                 nan_value=0.0) -> None:
        self._extractor = extractor
        self._nan_value = nan_value

    def extract(self, image: np.ndarray) -> np.ndarray:
        feature = self._extractor.extract(image)
        feature = np.nan_to_num(feature, 
                                nan=self._nan_value)
        return feature
