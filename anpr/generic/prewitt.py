"""laplacian filter.
"""
from __future__ import annotations

import cv2
import numpy as np

from anpr.core import ImageProcessor



class Prewitt(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(image, -1, kernelx)
        img_prewitty = cv2.filter2D(image, -1, kernely)
        return  cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
