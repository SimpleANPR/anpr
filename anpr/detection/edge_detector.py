from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from anpr.core.binarizer import Binarizer
from anpr.core.plate_detector import DetectionResult, PlateDetector
from anpr.generic.grayscale import GrayScale
from anpr.generic.filter_bilateral import FilterBilateral
from anpr.generic.canny import Canny
from anpr.core.image_processor import ImageProcessor

@dataclass(frozen=True)
class DetectionExtras:
    candidates: list[tuple[int, int, int, int]]
    candidates_img: list[np.ndarray]

class EdgeDetector(PlateDetector):
    def __init__(self, low_pass: ImageProcessor = None, high_pass: ImageProcessor = None) -> None:

        if low_pass is None:
            low_pass = FilterBilateral(11, 17, 17)

        if high_pass is None:
            high_pass = Canny(30, 200)

        self.low_pass = low_pass
        self.high_pass = high_pass

    def detect(self, image: np.ndarray) -> DetectionResult:
        original_image = image
    
        if len(image.shape) == 3:
            image = GrayScale().process(image)
        
        smoothed = self.low_pass.process(image)
        edges = self.high_pass.process(smoothed)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(image.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0,255, -1)
            new_image = cv2.bitwise_and(image, image, mask=mask)

            (x,y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = image[x1:x2+1, y1:y2+1]

            return DetectionResult(
                found_plate=True,
                n_candidates=len(contours),
                plate_polygon=location,
                plate_image=cropped_image)
        else:
            return DetectionResult(
                found_plate=False,
                n_candidates=len(contours), 
                plate_polygon=None,
                plate_image=None)
