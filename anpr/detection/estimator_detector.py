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
from typing import ClassVar

import cv2
import numpy as np
import shapely
import sklearn.preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from anpr.core.feature_extractor import FeatureExtractor
from anpr.core.image_processor import ImageProcessor
from anpr.core.plate_detector import DetectionResult, PlateDetector
from anpr.datasets.open_alpr import OpenALPRDataset
from anpr.features.color import ColorFeatures
from anpr.generic.brightness import Brightness
from anpr.generic.contrast import Contrast


@dataclass(frozen=True)
class DetectionExtras:
    candidates: list[tuple[int, int, int, int]]
    candidates_probs: np.ndarray
    contours: tuple[np.ndarray]


class EstimatorDetector(PlateDetector):
    _CLF: ClassVar[dict] = {
        'logistic_regression': LogisticRegression,
        'gaussian_process': GaussianProcessClassifier,
        'decision_tree': DecisionTreeClassifier,
        'extra_trees': ExtraTreesClassifier,
        'random_forest': RandomForestClassifier
    }

    def __init__(self,
                 estimator_algorithm: str = 'logistic_regression',
                 estimator_params: dict = dict(),
                 features: list[str] = ['color'],
                 scaler: str = 'MinMax',
                 preprocessing: ImageProcessor = None,
                 pp_in_predict: bool = True,
                 train_n_variations_for_sample: int = 20,
                 seed: int = None) -> None:
        assert estimator_algorithm in self._CLF

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

        # Armazenando a seed e criando um gerador
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # Gerando uma seed para o estimador
        if 'random_state' not in estimator_params:
            estimator_params['random_state'] = self._rng.integers(0, 99999)

        # Instanciando estimador e scaler
        self._estimator = self._CLF[estimator_algorithm](**estimator_params)
        self._scaler = scaler

        # Instanciando extrator de características
        self._extractor = ColorFeatures()

        # Variáveis de controle
        self._preprocessing = preprocessing
        self._apply_predict = pp_in_predict
        self._n_variations = train_n_variations_for_sample

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
        original_image = image

        if self._apply_predict:
            image = self._preprocessing.process(image)

        # Obtendo contornos externos
        img_h, img_w = image.shape
        contours, _ = cv2.findContours(image,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Obtendo os retângulos
        rectangles = [cv2.boundingRect(c) for c in contours]
        print(f'Encontrados {len(rectangles)} retângulos...')

        # Filtrando retângulos "grandes"
        max_ratio = 0.4
        rectangles = [r
                      for r in rectangles
                      if (r[2] <= max_ratio * img_w)
                      and (r[3] <= max_ratio * img_h)]
        print(f'Restaram {len(rectangles)} retângulos após '
              'remoção de retângulos grandes...')

        # Filtrando retângulos "pequenos"
        min_ratio = 0.005
        rectangles = [r
                      for r in rectangles
                      if (r[2] >= min_ratio * img_w)
                      and (r[3] >= min_ratio * img_h)]
        print(f'Restaram {len(rectangles)} retângulos após '
              'remoção de retângulos pequenos...')

        # Removendo retângulos com proporção inviável para
        #   caractere
        max_ch_prop = 0.75
        rectangles = [r
                      for r in rectangles
                      if (r[2] / r[3] <= max_ch_prop)]
        print(f'Restaram {len(rectangles)} retângulos após '
              'remoção de retângulos com proporção inviável...')

        # Removendo retângulos que possuem overlap
        overlap_th: float = 0.6
        to_remove = sorted(self._indices_to_remove_overlap(rectangles,
                                                           overlap_th),
                           reverse=True)
        for idx in to_remove:
            del rectangles[idx]

        print(f'Restaram {len(rectangles)} após remoção '
              'de overlap...')

        # Removendo retângulos que não possuem vizinhos
        diagonal_size = self._euclidean_distance(np.array([0, 0]),
                                                 np.array([img_w, img_h]))
        neighbor_th = 0.05
        rectangles = [r for i, r in enumerate(rectangles)
                      if self._has_neighbor(i,
                                            rectangles,
                                            diagonal_size * neighbor_th)]
        print(f'Restaram {len(rectangles)} retângulos após '
              'remoção dos sem vizinhos próximos...')

        # Buscando regiões com maior densidade de retângulos através
        #   da dilatação dos retângulos.
        blobs = np.zeros((img_h, img_w),
                         dtype=np.uint8)

        # Colocamos os retângulos encontrados na cor branca
        for r in rectangles:
            x, y, w, h = r
            blobs[y:y+h, x:x+w] = 1

        # Dilatamos os retângulos para formar um retângulo maior
        blobs = cv2.dilate(blobs,
                           cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (20, 20)),
                           iterations=2)

        # Obtemos os novos contornos
        blobs_contours, _ = cv2.findContours(blobs,
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

        # Calculamos os últimos candidatos
        candidates = [cv2.boundingRect(c) for c in blobs_contours]
        candidates = [c for c in candidates
                      if c[2] / c[3] > 0.7]
        n_candidates = len(candidates)
        print('Quantidade de candidatos finais:', n_candidates)

        if n_candidates <= 0:
            return DetectionResult(
                found_plate=False,
                n_candidates=0,
                plate_polygon=None,
                plate_image=None)

        # Realizando classificação nas regiões candidatas
        features = [self._extractor.extract(original_image[c[1]:c[1] + c[3],
                                                           c[0]:c[0] + c[2]])
                    for c in candidates]
        predictions = self._estimator.predict_proba(features)

        # Selecionamos o candidato que o classificador teve
        #   maior confiança em ser uma região de placa.
        best_candidate = predictions[:, 0].argmax()
        plate_rect = candidates[best_candidate]
        plate_img = original_image[plate_rect[1]:plate_rect[1] + plate_rect[3],
                                   plate_rect[0]:plate_rect[0] + plate_rect[2]].copy()

        return DetectionResult(
            found_plate=True,
            n_candidates=n_candidates,
            plate_polygon=plate_rect,
            plate_image=plate_img,
            extras=DetectionExtras(
                candidates=candidates,
                candidates_probs=predictions,
                contours=blobs_contours))

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
            full_img = ds_img.image

            _, _, w, h = ds_img.plate_rect

            # Obtendo exemplos positivos
            xs_p, ys_p = self._create_positives_from_plate(full_img,
                                                           ds_img.plate_rect)

            # Obtendo exemplos negativos
            xs_n, ys_n = self._create_n_negatives(full_img,
                                                  ds_img.plate_rect)
            X.extend(xs_p + xs_n)
            y.extend(ys_p + ys_n)

        return np.array(X), np.array(y)

    def _create_positives_from_plate(self,
                                     image: np.ndarray,
                                     plate_info: tuple[int, int,
                                                       int, int]) -> tuple[list[np.ndarray],
                                                                           list[np.ndarray]]:
        img_h, img_w, _ = image.shape
        x, y, w, h = plate_info
        strategies = ['smaller', 'greater']

        # Inicializando com região completa da placa
        X = [self._extractor.extract(image[y: y + h, x: x + w])]

        def _rand_slices(k_lower: float,
                         k_upper: float,
                         a_lower: float,
                         a_upper: float) -> tuple[slice, slice]:
            # Gerando um fator
            k = self._rng.uniform(k_lower, k_upper)

            # Gerando um offset
            offset_y = self._rng.integers(a_lower, a_upper)
            offset_x = self._rng.integers(a_lower, a_upper)

            # Calculando novos tamanhos
            new_w = max(int(w * k), 3)
            new_h = max(int(h * k), 3)

            # Calculando limietes
            start_y = max(y+offset_y, y)
            end_y = min(start_y+new_h, img_h)
            start_x = max(x+offset_x, x)
            end_x = min(start_x+new_w, img_w)

            # Calculando slices
            i = slice(start_y, end_y)
            j = slice(start_x, end_x)

            return i, j

        def _rand_luminosity(img: np.ndarray) -> np.ndarray:
            contrast = self._rng.uniform(0.75, 1.25)
            brightness = self._rng.uniform(-5, 5)
            brightness_updated = Brightness(brightness).process(img)
            contrast_updated = Contrast(contrast).process(brightness_updated)
            return contrast_updated

        # Gerando exemplos da mesma placa
        for _ in range(self._n_variations):
            strategy = self._rng.choice(strategies)
            i, j = None, None

            # Obtendo slices aleatórios
            if strategy == 'smaller':
                i, j = _rand_slices(0, 1, -7, 7)
            elif strategy == 'greater':
                i, j = _rand_slices(1, 1.75, -7, 7)

            # Obtendo imagem
            img = image[i, j]

            # Talvez aplicar uma mudança na luminosidade
            if self._rng.random() > 0.75:
                img = _rand_luminosity(img)

            # Adicionando amostra
            X.append(self._extractor.extract(img))

        # Adicionando classes
        y = [1 for _ in range(len(X))]

        return X, y

    def _create_n_negatives(self,
                            image: np.ndarray,
                            plate_info: tuple[int, int, int, int]) -> tuple[list[np.ndarray],
                                                                            list[np.ndarray]]:
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

            return not (out_of_bounds or inside_plate)

        # Obtendo regiões aleatórias que não possuem
        #   parte da placa.
        valid_xs = [i for i in range(w)
                    if _is_valid_idx(i, px, px + pw, w)]
        valid_ys = [i for i in range(h)
                    if _is_valid_idx(i, py, py + ph, h)]

        # Obtendo os pontos para a quantidade de variações
        xs = self._rng.choice(valid_xs,
                              size=self._n_variations + 1,
                              replace=False)
        ys = self._rng.choice(valid_ys,
                              size=self._n_variations + 1,
                              replace=False)
        points = np.array(list(zip(ys, xs)),
                          dtype=np.int32)

        def _extract_feature(p: np.ndarray) -> np.ndarray:
            y, x = p[0], p[1]
            rand_w = self._rng.integers(4, pw + 4)
            rand_h = self._rng.integers(4, ph + 4)
            sub_img = image[y:y+rand_h, x:x+rand_w]
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

    def _indices_to_remove_overlap(self,
                                   all_rectangles: list[tuple],
                                   threshold: float) -> list[int]:
        polygons = sorted([(i, shapely.box(xmin=x, xmax=x + w,
                                           ymin=y, ymax=y + h))
                           for i, (x, y, w, h) in enumerate(all_rectangles)],
                          key=lambda e: e[1].area)
        indices = []
        for i, (idx, p) in enumerate(polygons):
            for j, p_ in polygons[i+1:]:
                if p.intersects(p_):
                    int_area = shapely.intersection(p_, p).area
                    min_area = p.area

                    assert int_area >= 0.0
                    assert min_area <= p_.area, f'{min_area} > {p_.area}'

                    if int_area >= threshold * min_area:
                        indices.append(idx)
                        break

        return indices

    def _distance(self, rect_a, rect_b) -> float:
        # Obtendo dados dos retângulos
        xa, ya, wa, ha = rect_a
        xb, yb, wb, hb = rect_b

        # Calculamos o centro
        ca = np.array([xa + wa // 2,
                       ya + ha // 2])
        cb = np.array([xb + wb // 2,
                       yb + hb // 2])

        # Calculando a normal do vetor distância
        return self._euclidean_distance(ca, cb)

    def _euclidean_distance(self,
                            arr_a: np.ndarray,
                            arr_b: np.ndarray) -> float:
        return np.sqrt(np.sum((arr_a - arr_b) ** 2))

    def _has_neighbor(self,
                      idx: int,
                      all_rectangles: list[tuple],
                      proximity_th: float) -> bool:
        # Obtendo retângulo
        rect = all_rectangles[idx]

        # Calculando distância dos demais retângulos
        others = all_rectangles[:idx] + all_rectangles[idx+1:]
        distances = list(map(lambda r: self._distance(rect, r),
                             others))
        others = sorted(zip(distances, others),
                        key=lambda t: t[0])
        
        if len(others) <= 0:
            return True

        # Checando se possui vizinho
        nearest_dist = others[0][0]
        return nearest_dist <= proximity_th
