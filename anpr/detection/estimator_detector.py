"""Esse módulo contém um detector de
placas baseado em classificação e
segmentação (ROI).

Dada uma imagem f(x,y) de tamanho MxN,
o objetivo é que, usando um classificador C
todos os pontos (xi, yi) são atribuídos
a uma das 2 classes região de Placa (RP)
e região de Não Placa (NP). As características
utilizadas são relativas às informações globais
de uma dada região da imagem.

Em outras palavras, produzimos uma nova imagem
g(x, y) binária, onde 1 indica regiões que
podem pertencer a uma placa e 0 indica
regiões que podem não pertencer a placas.

A partir da imagem g(x,y), aplicamos um algoritmo
para identificação de retângulos e selecionamos
uma ou mais regiões candidatas.

Referências:
1. Color Texture-Based Object Detection: An Application to License Plate Localization
2. Car Plate Detection Using Cascaded Tree-Style Learner Based on Hybrid Object Features
3. License Plate-Location using Adaboost Algorithm
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import sklearn.preprocessing
from sklearn.svm import SVC

from anpr.core.image_processor import ImageProcessor
from anpr.core.feature_extractor import FeatureExtractor
from anpr.core.plate_detector import DetectionResult, PlateDetector
from anpr.datasets.open_alpr import OpenALPRDataset
from anpr.features.color import ColorFeatures


@dataclass(frozen=True)
class DetectionExtras:
    candidates: list[tuple[int, int, int, int]]
    contours: tuple[np.ndarray]


class EstimatorDetector(PlateDetector):
    def __init__(self,
                 estimator_algorithm: str = 'svm',
                 features: list[str] = ['color'],
                 scaler: str = 'MinMax',
                 preprocessing: ImageProcessor = None,
                 pp_in_predict: bool = False,
                 seed: int = None) -> None:
        if seed is None:
            seed = np.random.randint(0, 99999)

        if preprocessing is None:
            class _Identity(ImageProcessor):
                def process(self, image: np.ndarray) -> np.ndarray:
                    return image
            preprocessing = _Identity()

        if scaler is not None:
            scaler_name = f'{scaler}Scaler'
            scaler = getattr(sklearn.preprocessing, scaler_name)()

        self._estimator = SVC()
        self._scaler = scaler
        self._extractor = ColorFeatures()
        self._seed = seed

        self._preprocessing = preprocessing
        self._apply_predict = pp_in_predict

        self._rng = np.random.default_rng(self._seed)

    def fit(self,
            indices: list[int],
            ds: OpenALPRDataset) -> None:
        # Obtendo o dataset para treinamento
        X, y = self._prepare_dataset(indices, ds)

        # Treinando scaler se existir
        if self._scaler is not None:
            self._scaler.fit(X)
            X = self._scaler.transform(X)

        # Treinando o estimador
        self._estimator.fit(X, y)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Recebe uma imagem e retorna um resultado
        de detecção por uma placa veicular.

        Args:
            image (np.ndarray): imagem.

        Returns:
            DetectionResult: resultado da detecção.
        """
        # TODO: estratégias para melhoria de
        # performance:
        #  - Paralelismo
        #  - Não calcular pixel a pixel (pular alguns e depois interpolar)
        #    - Podemos pular alguns pixels;
        #    - Podemos usar a classificação para a área completa;
        #  - Não adicionar bordas pretas (ignorar pixels das bordas: NP)
        #  - Reduzir tamanho da imagem (resize para tamanho menor mantendo proporções)
        if self._apply_predict:
            image = self._preprocessing.process(image)

        # Criando array destino
        if len(image.shape) == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
        bin = np.zeros((h, w), dtype=np.uint8)

        # Classificando regiões da imagem
        step = self._window_size
        offset = step // 2
        for i in range(offset, h - offset, step):
            for j in range(offset, w - offset, step):
                # Obtendo slices da região da imagem
                i_slice = slice(i-offset, i+offset)
                j_slice = slice(j-offset, j+offset)

                # Obtendo características da imagem
                sub_img = image[i_slice, j_slice]
                features = self._extractor.extract(sub_img)
                features = np.expand_dims(features, axis=0)

                # Passando pelo scaler se existir
                if self._scaler is not None:
                    features = self._scaler.transform(features)

                # Obtendo a classe dessa região
                cls = self._estimator.predict(features)

                # Salvando na imagem binarizada
                bin[i_slice, j_slice] = cls

        # Extraindo contornos
        contours, _ = cv2.findContours(bin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Aproximando polígonos
        polygons = [cv2.approxPolyDP(c,
                                     epsilon=self._dp_eps,
                                     closed=True)
                    for c in contours]

        # Filtrando polígonos com quase 4 lados
        polys_4_sided = [p for p in polygons
                         if abs(len(p) - 4) <= 1]

        if len(polys_4_sided) <= 0:
            return DetectionResult(
                found_plate=False,
                n_candidates=0,
                plate_polygon=None,
                plate_image=None)

        # Rankeando polígonos
        if len(polys_4_sided) > 1:
            ...

        # Obtendo polígono mais provável
        polygon = polys_4_sided[0]

        # Extraindo bounding rect
        bounding_rect = cv2.boundingRect(polygon)

        # Extraindo sub-imagem
        x, y, w, h = bounding_rect
        plate = image[y:y+h, x:x+w].copy()

        return DetectionResult(
            found_plate=True,
            n_candidates=len(polys_4_sided),
            plate_polygon=bounding_rect,
            plate_image=plate,
            extras=DetectionExtras(
                all_rectangles=[cv2.boundingRect(p)
                                for p in polys_4_sided],
                contours=contours))

    def _prepare_dataset(self,
                         indices: list[int],
                         ds: OpenALPRDataset) -> tuple[np.ndarray,
                                                       np.ndarray]:
        """Retorna o dataset para treinamento.

        Args:
            indices (list[int]): índices das imagens que 
                devem ser utilizadas.
            ds (OpenALPRDataset): dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: X, y.
        """
        X = []
        y = []

        for idx in indices:
            # Obtendo imagem do dataset
            ds_img = ds.image_at(idx)
            full_img = self._preprocessing.process(ds_img.image)

            _, _, w, h = ds_img.plate_rect
            if self._window_size > w or self._window_size > h:
                continue

            # Obtendo exemplos positivos
            xs_p, ys_p = self._create_positives_from_plate(full_img,
                                                           ds_img.plate_rect)

            # Obtendo exemplos negativos
            xs_n, ys_n = self._create_n_negatives(full_img,
                                                  ds_img.plate_rect,
                                                  len(ys_p),
                                                  self._rng.integers(0, 9999))
            X.extend(xs_p + xs_n)
            y.extend(ys_p + ys_n)

        return np.array(X), np.array(y)

    def _create_positives_from_plate(self,
                                     image: np.ndarray,
                                     plate_info: tuple[int, int,
                                                       int, int]) -> tuple[list[np.ndarray],
                                                                           list[np.ndarray]]:
        x, y, w, h = plate_info
        plate = image[y: y + h, x: x + w]

        assert self._window_size <= h, h
        assert self._window_size <= w, w

        n_h = math.floor(h / self._window_size)
        n_w = math.floor(w / self._window_size)

        # Inicializando com região completa da placa
        X = []

        # Criando amostras do tamanho da janela
        for i in range(n_h):
            for j in range(n_w):
                # Obtendo região de placa
                start_y = i * self._window_size
                end_y = start_y + self._window_size

                start_x = j * self._window_size
                end_x = start_x + self._window_size

                # Obtendo placa
                plate_img = plate[start_y:end_y, start_x:end_x]
                if plate_img.shape[:2] != (self._window_size,
                                           self._window_size):
                    continue

                # Calculando características dessa placa
                features = self._extractor.extract(plate_img)

                # Salvando na array
                X.append(features)

        y = [1 for _ in range(len(X))]
        return X, y

    def _create_n_negatives(self,
                            image: np.ndarray,
                            plate_info: tuple[int, int, int, int],
                            n: int,
                            seed: int) -> tuple[list[np.ndarray],
                                                list[np.ndarray]]:
        rng = np.random.default_rng(seed)
        h, w, _ = image.shape
        px, py, pw, ph = plate_info

        def _is_in_plate(pos: int,
                         lower: int,
                         upper: int):
            if pos < lower or pos > upper:
                return False

            return True

        def _is_valid_idx(pos: int,
                          plate_lower: int,
                          plate_upper: int,
                          img_max: int):
            # Se a posição estiver fora dos limites,
            #   ela é inválida.
            out_of_bounds = pos > img_max or pos < 0

            # Se fizer parte da placa, também
            #   é inválida.
            inside_plate = _is_in_plate(pos,
                                        plate_lower,
                                        plate_upper)

            # Se algum dos limites da janela
            #   fizer parte da placa ou estiver
            #   fora da imagem, ela é inválida.
            min_pos_w = pos - self._window_size
            min_w_out_of_bounds = min_pos_w > img_max or min_pos_w < 0
            min_w_inside_plate = _is_in_plate(min_pos_w,
                                              plate_lower,
                                              plate_upper)

            max_pos_w = pos + self._window_size
            max_w_out_of_bounds = min_pos_w > max_pos_w or max_pos_w < 0
            max_w_inside_plate = _is_in_plate(max_pos_w,
                                              plate_lower,
                                              plate_upper)

            any_out_bounds = out_of_bounds or min_w_out_of_bounds or \
                max_w_out_of_bounds
            any_inside = inside_plate or min_w_inside_plate or \
                max_w_inside_plate
            return not (any_out_bounds or any_inside)

        # Obtendo regiões aleatórias que não possuem
        #   parte da placa.
        valid_xs = [i for i in range(w)
                    if _is_valid_idx(i, px, px + pw, w)]
        valid_ys = [i for i in range(h)
                    if _is_valid_idx(i, py, py + ph, h)]
        n = min(n, min(len(valid_xs), len(valid_ys)))

        xs = rng.choice(valid_xs,
                        size=n,
                        replace=False)
        ys = rng.choice([i for i in range(h)
                         if _is_valid_idx(i, py, py + ph, h)],
                        size=n,
                        replace=False)
        points = np.array(list(zip(ys, xs)),
                          dtype=np.int32)

        def _extract_feature(p: np.ndarray) -> np.ndarray:
            w = self._window_size
            lp = p - w
            y, x = lp[0], lp[1]
            sub_img = image[y:y+w, x:x+w]
            return self._extractor.extract(sub_img)

        # Extraindo características
        X = [_extract_feature(p) for p in points]
        y = [0 for _ in range(len(X))]

        return X, y

    def _flatten_extractor(self) -> FeatureExtractor:
        class _Flatten(FeatureExtractor):
            def extract(self, image: np.ndarray) -> np.ndarray:
                if len(image.shape) == 3:
                    image = image.sum(axis=-1)

                assert len(image.shape) == 2
                return image.ravel()

        return _Flatten()
