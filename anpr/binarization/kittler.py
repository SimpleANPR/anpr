"""Binarização através do algoritmo
de Kittler.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import Binarizer


class KittlerBinarizer(Binarizer):
    """Binarização através do método
    de Kittler.

    Essa classe utiliza a seguinte implementação
    como base:
    https://github.com/manuelaguadomtz/pythreshold/blob/master/pythreshold/global_th/min_err.py
    """

    def __init__(self, n_bins: int = 256) -> None:
        self.n_bins = n_bins

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        # Obtemos o valor de threshold
        th = self._get_threshold(image)

        # Calculamos a imagem binarizada
        _, bin_img = cv2.threshold(image,
                                   thresh=th,
                                   maxval=255,
                                   type=cv2.THRESH_BINARY)

        return bin_img

    def _get_threshold(self, image: np.ndarray) -> int:
        # Input image histogram
        hist = np.histogram(image,
                            bins=range(self.n_bins))[0].astype(np.float32)

        # The number of background pixels for each threshold
        w_backg = hist.cumsum()
        w_backg[w_backg == 0] = 1  # to avoid divisions by zero

        # The number of foreground pixels for each threshold
        w_foreg = w_backg[-1] - w_backg
        w_foreg[w_foreg == 0] = 1  # to avoid divisions by zero

        # Cumulative distribution function
        cdf = np.cumsum(hist * np.arange(len(hist)))

        # Means (Last term is to avoid divisions by zero)
        b_mean = cdf / w_backg
        f_mean = (cdf[-1] - cdf) / w_foreg

        # Standard deviations
        b_std = ((np.arange(len(hist)) - b_mean)**2 * hist).cumsum() / w_backg
        f_std = ((np.arange(len(hist)) - f_mean) ** 2 * hist).cumsum()
        f_std = (f_std[-1] - f_std) / w_foreg

        # To avoid log of 0 invalid calculations
        b_std[b_std == 0] = 1
        f_std[f_std == 0] = 1

        # Estimating error
        error_a = w_backg * np.log(b_std) + w_foreg * np.log(f_std)
        error_b = w_backg * np.log(w_backg) + w_foreg * np.log(w_foreg)
        error = 1 + 2 * error_a - 2 * error_b

        return np.argmin(error)
