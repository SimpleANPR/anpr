"""Filtragem através do filtro
non-local-means.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor



class FilterNLM(ImageProcessor):
    def __init__(self, h: float, hForColorComponents: float = None, template_size: int = 7, search_size: int = 21):
        """Construtor. Recebe como argumentos os parâmetros para o filtro Non-Local Means.

        Args:
            h (float): Parâmetro de filtragem para componentes de luminância.
            hForColorComponents (float): Parâmetro de filtragem para componentes de cor (opcional).
            template_size (int): Tamanho da janela de template.
            search_size (int): Tamanho da janela de busca.
        """
        self._h = h
        self._hForColorComponents = hForColorComponents if hForColorComponents is not None else h
        self._template_size = template_size
        self._search_size = search_size

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:  # Imagem colorida (3 canais)
            return cv2.fastNlMeansDenoisingColored(image, None, h=self._h, hColor=self._hForColorComponents, templateWindowSize=self._template_size, searchWindowSize=self._search_size)
        else:  # Imagem em escala de cinza (1 canal)
            return cv2.fastNlMeansDenoising(image, None, h=self._h, templateWindowSize=self._template_size, searchWindowSize=self._search_size)