"""Esse módulo contém a interface
de um processador de imagem genérico.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Recebe uma imagem e retorna uma nova
        imagem.

        Args:
            image (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
