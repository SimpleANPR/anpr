"""Esse módulo contém a interface
de um detector de placas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Classe que representa o resultado
    de uma detecção de placa.

    found_plate: se uma placa foi encontrada.
    n_candidates: quantidade de regiões candidatas.
    plate_polygon: retângulo da placa (lx, uy, w, h)
    plate_image: imagem que contém apenas o retângulo da placa.
    """
    found_plate: bool
    n_candidates: int
    plate_polygon: tuple[int, int, int, int]
    plate_image: np.ndarray
    extras: object | None = None


class PlateDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Recebe uma imagem e retorna um resultado
        de detecção por uma placa veicular.

        Args:
            image (np.ndarray): imagem.

        Returns:
            DetectionResult: resultado da detecção.
        """
