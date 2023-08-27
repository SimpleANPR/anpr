from __future__ import annotations

from dataclasses import dataclass

from anpr.core.plate_detector import DetectionResult, PlateDetector
from anpr.generic.wavelet_transform import WaveletTransform
from anpr.binarization.simple import SimpleBinarizer

import numpy as np
import cv2

@dataclass(frozen=True)
class DetectionExtras:
    candidates: list[np.ndarray]

class WaveletDetector(PlateDetector):
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Recebe uma imagem e retorna um resultado
        de detecção por uma placa veicular.

        Args:
            image (np.ndarray): imagem.

        Returns:
            DetectionResult: resultado da detecção.
        """
        LL, (LH, HL, _)= WaveletTransform().process(image)

        horizontal = SimpleBinarizer(45).binarize(LH)
        vertical = SimpleBinarizer(45).binarize(HL)

        soma_linha = np.sum(horizontal, axis=1)/255 #soma dos números de pixels em cada linha
        #soma_linha_max = np.argmax(soma_linha)

        #limita área de busca para região abaixo da linha de referência calculada acima. pode apresentar erros caso a linha de referencia não esteja abaixo da placa
        #vertical[soma_linha_max:] = 0
        
        kernel = np.ones((7,7),np.uint8)
        vertical_dilated = cv2.dilate(vertical, kernel, iterations=2)

        contours,_ = cv2.findContours((vertical_dilated.astype(np.uint8)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        license_plate_list = []
        polygon_list = []

        i = 0
        for c in contours: 
            x,y,w,h = cv2.boundingRect(c)

            not_ret = (w) <= (h)
            proportion = w/h

            if(not_ret or proportion < 2):
                pass

            else:
                #cv2.drawContours(LL, contours, i, (0,255,0), 3)
                cv2.rectangle(LL, (x,y),(x+w,y+h),(36,255,12), 1)
                license_plate_list.append(LL[y:(y+h), x:(x+w)])
                polygon_list.append((x,y,w,h))

            i += 1
        return DetectionResult(
            found_plate=(len(license_plate_list)>0),
            n_candidates=len(license_plate_list),
            plate_polygon=polygon_list[0],
            plate_image=license_plate_list[0],
            extras=DetectionExtras(
                candidates=license_plate_list))