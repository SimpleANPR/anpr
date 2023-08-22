"""Esse módulo contém um detector de
placas baseado em classificação e
segmentação (ROI).

Dada uma imagem f(x,y) de tamanho MxN,
o objetivo é que, usando um estimador G, 
sejam definidas as probabilidade P(xi, yi) de 
cada  ponto (xi, yi) fazer parte da região
da placa.


Depois, um algoritmo de segmentação é utilizado
para encontrar as regiões similares. A região
que possui a maior probabilidade (pode ser calculado
de diferentes formas) é escolhida como placa.

Referências:
1. Color Texture-Based Object Detection: An Application to License Plate Localization
2. Car Plate Detection Using Cascaded Tree-Style Learner Based on Hybrid Object Features
3. License Plate-Location using Adaboost Algorithm
"""
from __future__ import annotations

import numpy as np

from anpr.core.plate_detector import DetectionResult, PlateDetector


class RegressorDetector(PlateDetector):
    def __init__(self, 
                 regressor,
                 cluster) -> None:
        ...

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Recebe uma imagem e retorna um resultado
        de detecção por uma placa veicular.

        Args:
            image (np.ndarray): imagem.

        Returns:
            DetectionResult: resultado da detecção.
        """
        return None
