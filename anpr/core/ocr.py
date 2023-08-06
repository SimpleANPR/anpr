"""Esse módulo contém a interface
de um reconhecedor de caracteres.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class OCR(ABC):
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> str:
        """Recebe uma imagem e retorna o conteúdo
        textual.

        Args:
            image (np.ndarray): imagem.

        Returns:
            str: texto extraído da imagem.
        """
