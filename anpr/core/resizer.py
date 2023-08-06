"""Esse módulo contém a interface
de um redimensionador de imagens.
"""
from __future__ import annotations

from abc import abstractmethod
import numpy as np

from . import ImageProcessor


class Resizer(ImageProcessor):
    def __init__(self, target_resolution: tuple[int, int]) -> None:
        """Construtor.

        Args:
            target_resolution (tuple[int, int]): resolução destino.
        """
        self._res = target_resolution

    @property
    def resolution(self) -> tuple[int, int]:
        """Retorna a resolução destino desse Resizer.

        Returns:
            tuple[int, int]: resolução destino.
        """
        return self._res

    @abstractmethod
    def resize(image: np.ndarray) -> np.ndarray:
        """Redimensionada a imagem passada como argumento
        para a nova resolução.

        Args:
            image (np.ndarray): imagem original.

        Returns:
            np.ndarray: nova imagem.
        """

    def process(self, image: np.ndarray) -> np.ndarray:
        return self.resize(image)
