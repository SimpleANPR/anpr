"""Esse módulo contém uma implementação
de leitor de caracteres usando o EasyOCR.
"""
from __future__ import annotations

import string
import easyocr
import numpy as np

from anpr.core.ocr import OCR


class EasyOCR(OCR):
    def __init__(self, language: str = 'pt') -> None:
        self._reader = easyocr.Reader([language],
                                      detector=False)

        upper_cases = list(string.ascii_uppercase)
        numbers = [str(i) for i in range(0, 10)]
        self._characters = upper_cases + numbers

    def extract_text(self, image: np.ndarray) -> str:
        texts = self._reader.recognize(image,
                                       allowlist=self._characters,
                                       detail=0,
                                       paragraph=True)

        if len(texts) > 1:
            raise ValueError('More than one text found.')

        return texts[0]
