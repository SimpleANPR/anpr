"""Esse módulo contém as imagens
de benchmark utilizadas pelo OpenALPR.

O dataset original pode ser encontrado em:
https://github.com/openalpr/benchmarks
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import gdown
import numpy as np


@dataclass
class OpenALPRImage:
    """Classe que representa uma imagem
    anotada do dataset OpenALPR.

    name: nome da imagem
    image: imagem como np array (RGB)
    plate_position: retângulo da placa (lx, uy, w, h)
    plate_text: conteúdo do texto
    image_path: caminho para a imagem
    """
    name: str
    image: np.ndarray
    plate_rect: tuple[int, int, int, int]
    plate_text: str
    image_path: Path

    def load_image(self):
        if self.image is None:
            self.image = cv2.imread(str(self.image_path))
            self.image = cv2.cvtColor(self.image,
                                      cv2.COLOR_BGR2RGB)

    def free(self):
        self.image = None


class OpenALPRDataset:
    _URL = ('https://drive.google.com/uc?id='
            '1pOCgNkgLu8xCskxiFHGSzDIp8ltdHtuG')
    _DIR = Path('.').joinpath('img').joinpath('open_alpr')
    _MAX_LOADED = 20

    def __init__(self):
        if not self._DIR.exists():
            self._download()

        self._imgs: dict[str, OpenALPRImage] = dict()
        self._n_loaded = 0

        files: list[Path] = sorted(list(self._DIR.rglob('*')),
                                   key=lambda f: f.name)

        for i in range(0, len(files), 2):
            img_path = files[i]
            txt_path = files[i+1]
            assert any(img_path.name.endswith(t)
                       for t in ['png', 'jpg', 'jpeg']), img_path
            assert txt_path.name.endswith('txt'), txt_path

            annotation = txt_path.read_text()
            parts = annotation.split()
            assert len(parts) == 6

            plate_position = tuple(map(lambda p: max(int(p), 0),
                                       parts[1:-1]))
            plate = parts[-1]
            img_name = img_path.name
            self._imgs[img_name] = OpenALPRImage(name=img_name,
                                                 image=None,
                                                 plate_rect=plate_position,
                                                 plate_text=plate,
                                                 image_path=img_path)

        self._indices = {i: k for i, k in enumerate(self._imgs)}

    def total_images(self):
        return len(self._imgs)

    def image(self, name: str) -> OpenALPRImage:
        img = self._imgs[name]

        if img.image is None:
            img.load_image()
            self._n_loaded += 1

        # De tempos em tempos, realizamos
        #   uma limpeza das imagens carregadas.
        if self._n_loaded >= self._MAX_LOADED:
            for k in self._imgs:
                if k == name:
                    continue

                self._imgs[k].free()

        return img

    def image_at(self, index: int) -> OpenALPRImage:
        return self.image(self._indices[index])

    @classmethod
    def _download(cls):
        cls._DIR.mkdir(parents=True,
                       exist_ok=False)
        zip_path = cls._DIR.joinpath('data.zip')

        # Fazendo download do zip
        gdown.cached_download(cls._URL,
                              str(zip_path),
                              postprocess=gdown.extractall)

        # Apagando o zip após download e extraçaõ
        zip_path.unlink()
