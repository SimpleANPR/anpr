"""Esse módulo contém a interface
de um binarizador.
"""
from __future__ import annotations

from abc import abstractmethod
import numpy as np


from . import ImageProcessor


class Binarizer(ImageProcessor):
    @abstractmethod
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Recebe uma imagem em escala de cinza
        e retorna uma imagem binarizada.

        Args:
            image (np.ndarray): imagem.

        Returns:
            np.ndarray: imagem binarizada.
        """

    def process(self, image: np.ndarray) -> np.ndarray:
        return self.binarize(image)
