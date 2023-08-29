"""Esse módulo contém um agregador de
características.
"""
from __future__ import annotations

import numpy as np

from anpr.core.feature_extractor import FeatureExtractor


class AggregateExtractor(FeatureExtractor):
    """Agregador de etapas de processamento
    sequencial. Aplica cada um dos processamentos
    de forma sequencial (i.e., saída de um é entrada
    para o próximo).
    """

    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        self._extractors = extractors

    def extract(self, image: np.ndarray) -> np.ndarray:
        features = []

        for p in self._extractors:
            features.append(p.extract(image))

        return np.concatenate(features)
