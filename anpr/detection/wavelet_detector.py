from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from anpr.binarization.simple import SimpleBinarizer
from anpr.core.binarizer import Binarizer
from anpr.core.plate_detector import DetectionResult, PlateDetector
from anpr.generic.grayscale import GrayScale
from anpr.generic.wavelet_transform import WaveletTransform


@dataclass(frozen=True)
class DetectionExtras:
    candidates: list[tuple[int, int, int, int]]
    candidates_img: list[np.ndarray]


class WaveletDetector(PlateDetector):
    def __init__(self,
                 binarizer: Binarizer = None,
                 dilation_kernel: tuple[int, int] = None,
                 dilation_iterations: int = 2) -> None:
        if binarizer is None:
            binarizer = SimpleBinarizer(45)

        if dilation_kernel is None:
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                        (7, 7))

        self._binarizer = binarizer
        self._dilate_kernel = dilation_kernel
        self._dilation_iterations = dilation_iterations

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Recebe uma imagem e retorna um resultado
        de detecção por uma placa veicular.

        Args:
            image (np.ndarray): imagem.

        Returns:
            DetectionResult: resultado da detecção.
        """
        original_image = image

        if len(image.shape) == 3:
            image = GrayScale().process(image)

        LL, (LH, HL, _) = WaveletTransform().process(image)

        # horizontal = self._binarizer.binarize(LH)
        vertical = self._binarizer.binarize(HL)

        # soma dos números de pixels em cada linha
        # soma_linha = np.sum(horizontal, axis=1)/255
        # soma_linha_max = np.argmax(soma_linha)

        # limita área de busca para região abaixo da linha de
        #   referência calculada acima. pode apresentar
        #   erros caso a linha de referencia não esteja abaixo da placa
        # vertical[soma_linha_max:] = 0
        vertical_dilated = cv2.dilate(vertical,
                                      self._dilate_kernel,
                                      iterations=self._dilation_iterations)
        vertical_dilated = vertical_dilated.astype(np.uint8)
        contours, _ = cv2.findContours(vertical_dilated,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        license_plate_list = []
        polygon_list = []

        i = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            not_ret = (w) <= (h)
            proportion = w/h

            if (not_ret or proportion < 2):
                pass
            else:
                # cv2.drawContours(LL, contours, i, (0,255,0), 3)
                cv2.rectangle(LL, (x, y), (x+w, y+h), (36, 255, 12), 1)

                # Aplicando um "trick" para retornar às coordenadas
                #   originais através do up-sample.
                # Mains informações em: 
                # https://www.mathworks.com/help/wavelet/ug/fast-wavelet-transform-fwt-algorithm.html
                x, y, w, h = 2*x, 2*y, 2*w, 2*h
                license_plate_list.append(original_image[y:(y+h), x:(x+w)])
                polygon_list.append((x, y, w, h))

            i += 1

        n_candidates = len(license_plate_list)
        found_plate = n_candidates > 0

        return DetectionResult(
            found_plate=found_plate,
            n_candidates=n_candidates,
            plate_polygon=polygon_list[0] if found_plate else None,
            plate_image=license_plate_list[0] if found_plate else None,
            extras=DetectionExtras(
                candidates=polygon_list,
                candidates_img=license_plate_list))
