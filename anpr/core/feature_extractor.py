"""Esse módulo contém a interface
de um extrator de características.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Recebe uma imagem e retorna
        uma array de características.

        Args:
            image (np.ndarray): imagem.

        Returns:
            np.ndarray: vetor de características.
        """
