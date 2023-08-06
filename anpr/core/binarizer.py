"""Esse módulo contém a interface
de um binarizador.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .image_processor import ImageProcessor


class Binarizer(ImageProcessor):
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Recebe uma imagem em escala de cinza
        e retorna uma imagem binarizada.

        Args:
            image (np.ndarray): imagem.

        Returns:
            np.ndarray: imagem binarizada.
        """
        assert len(image.shape) == 2, "Imagem tem que ser single-channel"
        return self._binarize(image)

    @abstractmethod
    def _binarize(self, image: np.ndarray) -> np.ndarray: ...

    def process(self, image: np.ndarray) -> np.ndarray:
        return self.binarize(image)
