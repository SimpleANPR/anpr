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

        self._br_plates = {"AYO9034.jpg",
                           "AZJ6991.jpg",
                           "FZB9581.jpg",
                           "GWT2180.jpg",
                           "HPM9362.jpg",
                           "JGZ3298.jpg",
                           "JIT7463.jpg",
                           "JIY4434.jpg",
                           "JOG9221.jpg",
                           "JPQ9870.jpg",
                           "JQS5683.jpg",
                           "JQV5526.jpg",
                           "JRD2238.jpg",
                           "JRK5336.jpg",
                           "JRV1942.jpg",
                           "JSC7486.jpg",
                           "JSG9648.jpg",
                           "JSK5419.jpg",
                           "JSP7678.jpg",
                           "JSQ1413.jpg",
                           "JST2699.jpg",
                           "MTW5608.jpg",
                           "MXQ1601.jpg",
                           "MYX3152.jpg",
                           "NTD9247.jpg",
                           "NTH0518.jpg",
                           "NTK5785.jpg",
                           "NTO1053.jpg",
                           "NTV0498.jpg",
                           "NYI3834.jpg",
                           "NYL3614.jpg",
                           "NYM7544.jpg",
                           "NYY1710.jpg",
                           "NYZ0897.jpg",
                           "NZF0384.jpg",
                           "NZF7823.jpg",
                           "NZJ6581.jpg",
                           "NZM5430.jpg",
                           "NZO6276.jpg",
                           "NZP8292.jpg",
                           "NZU2673.jpg",
                           "NZW2197.jpg",
                           "OCX4764.jpg",
                           "ODC9387.jpg",
                           "ODJ1599.jpg",
                           "OEL1145.jpg",
                           "OKK7448.jpg",
                           "OKL0817.jpg",
                           "OKL1235.jpg",
                           "OKM0944.jpg",
                           "OKM2371.jpg",
                           "OKP7250.jpg",
                           "OKS0078.jpg",
                           "OKT3896.jpg",
                           "OKV8004.jpg",
                           "OLA1208.jpg",
                           "OLB4809.jpg",
                           "OLC4728.jpg",
                           "OLC7676.jpg",
                           "OLE5095.jpg",
                           "OUG6157.jpg",
                           "OUH9191.jpg",
                           "OUM7311.jpg",
                           "OUN4297.jpg",
                           "OUP1442.jpg",
                           "OUP9563.jpg",
                           "OUQ6314.jpg",
                           "OUW5216.jpg",
                           "OUZ2198.jpg",
                           "OVA1319.jpg",
                           "OVK3653.jpg",
                           "OYJ9557.jpg",
                           "OZC8400.jpg",
                           "OZG3580.jpg",
                           "OZK4620.jpg",
                           "OZK6717.jpg",
                           "OZL2318.jpg",
                           "OZP8126.jpg",
                           "OZR2224.jpg",
                           "OZS6477.jpg",
                           "OZU2104.jpg",
                           "OZU5764.jpg",
                           "OZV6697.jpg",
                           "PAG5219.jpg",
                           "PJB2414.jpg",
                           "PJB7392.jpg",
                           "PJC4903.jpg",
                           "PJD2685.jpg",
                           "PJF4224.jpg",
                           "PJF5797.jpg",
                           "PJG0783.jpg",
                           "PJH0957.jpg",
                           "PJI5396.jpg",
                           "PJI5921.jpg",
                           "PJI7589.jpg",
                           "PJJ1406.jpg",
                           "PJJ4955.jpg",
                           "PJK4867.jpg",
                           "PJP2783.jpg",
                           "PJP8208.jpg",
                           "PJT2905.jpg",
                           "PJT4884.jpg",
                           "PJU2207.jpg",
                           "PJU2853.jpg",
                           "PJV9741.jpg",
                           "PJX5721.jpg",
                           "PJY5472.jpg",
                           "PJY8509.jpg",
                           "PUT6858.jpg",
                           "PVX0095.jpg",
                           "PWC0633.jpg",
                           "PWF3266.jpg",
                           "PXP8172.jpg",
                           "PYB6477.jpg"}
        self._us_plates = {"wts-lg-000191.jpg",
                           "316b64c0-55bf-4079-a1c0-d93f461a576f.jpg",
                           "car1.jpg",
                           "car9.jpg",
                           "wts-lg-000131.jpg",
                           "wts-lg-000174.jpg",
                           "wts-lg-000162.jpg",
                           "3850ba91-3c64-4c64-acba-0c46b61ec0da.jpg",
                           "wts-lg-000098.jpg",
                           "wts-lg-000123.jpg",
                           "wts-lg-000192.jpg",
                           "wts-lg-000039.jpg",
                           "wts-lg-000064.jpg",
                           "wts-lg-000053.jpg",
                           "car12.jpg",
                           "wts-lg-000145.jpg",
                           "wts-lg-000199.jpg",
                           "wts-lg-000149.jpg",
                           "us2.jpg",
                           "wts-lg-000059.jpg",
                           "wts-lg-000099.jpg",
                           "wts-lg-000197.jpg",
                           "wts-lg-000115.jpg",
                           "wts-lg-000169.jpg",
                           "wts-lg-000148.jpg",
                           "wts-lg-000127.jpg",
                           "wts-lg-000101.jpg",
                           "wts-lg-000134.jpg",
                           "wts-lg-000124.jpg",
                           "wts-lg-000014.jpg",
                           "f0a3b8c0-198a-471b-9ca9-345c3dd28073.jpg",
                           "wts-lg-000069.jpg",
                           "wts-lg-000128.jpg",
                           "wts-lg-000147.jpg",
                           "wts-lg-000080.jpg",
                           "wts-lg-000050.jpg",
                           "wts-lg-000041.jpg",
                           "wts-lg-000038.jpg",
                           "wts-lg-000164.jpg",
                           "car9-9.jpg",
                           "wts-lg-000113.jpg",
                           "wts-lg-000190.jpg",
                           "wts-lg-000030.jpg",
                           "wts-lg-000172.jpg",
                           "wts-lg-000144.jpg",
                           "wts-lg-000067.jpg",
                           "car14.jpg",
                           "car7.jpg",
                           "wts-lg-000091.jpg",
                           "wts-lg-000018.jpg",
                           "wts-lg-000065.jpg",
                           "37170dd1-2802-4e38-b982-c5d07c64ff67.jpg",
                           "car9-2.jpg",
                           "wts-lg-000171.jpg",
                           "wts-lg-000136.jpg",
                           "wts-lg-000057.jpg",
                           "wts-lg-000075.jpg",
                           "wts-lg-000055.jpg",
                           "wts-lg-000049.jpg",
                           "wts-lg-000100.jpg",
                           "wts-lg-000083.jpg",
                           "wts-lg-000189.jpg",
                           "car18.jpg",
                           "wts-lg-000186.jpg",
                           "wts-lg-000046.jpg",
                           "wts-lg-000119.jpg",
                           "a03ced3f-5a97-4e75-8106-fabfd2b8b76e.jpg",
                           "wts-lg-000095.jpg",
                           "car17.jpg",
                           "0b86cecf-67d1-4fc0-87c9-b36b0ee228bb.jpg",
                           "wts-lg-000181.jpg",
                           "4be2025c-09f7-4bb0-b1bd-8e8633e6dec1.jpg",
                           "wts-lg-000126.jpg",
                           "car8.jpg",
                           "car9-1.jpg",
                           "wts-lg-000087.jpg",
                           "wts-lg-000168.jpg",
                           "wts-lg-000158.jpg",
                           "wts-lg-000016.jpg",
                           "car11.jpg",
                           "wts-lg-000152.jpg",
                           "wts-lg-000177.jpg",
                           "wts-lg-000194.jpg",
                           "wts-lg-000173.jpg",
                           "wts-lg-000133.jpg",
                           "d4f79480-366a-40b6-ab2c-328bcba705b2.jpg",
                           "wts-lg-000106.jpg",
                           "cfaa9dd2-a388-4e92-bb3a-ae65c28d8139.jpg",
                           "wts-lg-000040.jpg",
                           "wts-lg-000155.jpg",
                           "car2.jpg",
                           "wts-lg-000017.jpg",
                           "us8.jpg",
                           "car3.jpg",
                           "wts-lg-000035.jpg",
                           "wts-lg-000071.jpg",
                           "wts-lg-000175.jpg",
                           "car22.jpg",
                           "us7.jpg",
                           "wts-lg-000179.jpg",
                           "wts-lg-000060.jpg",
                           "wts-lg-000066.jpg",
                           "wts-lg-000077.jpg",
                           "wts-lg-000045.jpg",
                           "wts-lg-000063.jpg",
                           "car13.jpg",
                           "car9-0.jpg",
                           "wts-lg-000167.jpg",
                           "us6.jpg",
                           "wts-lg-000042.jpg",
                           "wts-lg-000140.jpg",
                           "wts-lg-000058.jpg",
                           "wts-lg-000020.jpg",
                           "wts-lg-000090.jpg",
                           "wts-lg-000120.jpg",
                           "wts-lg-000047.jpg",
                           "wts-lg-000188.jpg",
                           "us1.jpg",
                           "wts-lg-000056.jpg",
                           "wts-lg-000027.jpg",
                           "wts-lg-000139.jpg",
                           "wts-lg-000023.jpg",
                           "wts-lg-000061.jpg",
                           "wts-lg-000021.jpg",
                           "us4.jpg",
                           "e73fd200-7ba4-4725-9d1d-2ec710864df6.jpg",
                           "wts-lg-000141.jpg",
                           "wts-lg-000118.jpg",
                           "wts-lg-000187.jpg",
                           "car21.jpg",
                           "wts-lg-000024.jpg",
                           "wts-lg-000121.jpg",
                           "wts-lg-000178.jpg",
                           "wts-lg-000062.jpg",
                           "car19.jpg",
                           "car6.jpg",
                           "wts-lg-000010.jpg",
                           "wts-lg-000165.jpg",
                           "wts-lg-000078.jpg",
                           "wts-lg-000143.jpg",
                           "wts-lg-000028.jpg",
                           "wts-lg-000043.jpg",
                           "33fa5185-0286-4e8f-b775-46162eba39d4.jpg",
                           "wts-lg-000163.jpg",
                           "wts-lg-000161.jpg",
                           "wts-lg-000176.jpg",
                           "car16.jpg",
                           "5b562a61-34ad-4f00-9164-d34abb7a38e4.jpg",
                           "wts-lg-000160.jpg",
                           "car9-4.jpg",
                           "wts-lg-000097.jpg",
                           "wts-lg-000073.jpg",
                           "wts-lg-000129.jpg",
                           "wts-lg-000142.jpg",
                           "wts-lg-000156.jpg",
                           "wts-lg-000044.jpg",
                           "wts-lg-000034.jpg",
                           "wts-lg-000076.jpg",
                           "12c6cb72-3ea3-49e7-b381-e0cdfc5e8960.jpg",
                           "wts-lg-000082.jpg",
                           "wts-lg-000196.jpg",
                           "wts-lg-000138.jpg",
                           "wts-lg-000011.jpg",
                           "us5.jpg",
                           "wts-lg-000183.jpg",
                           "f8fc5e59-9083-466b-ae3f-6b869a0b257b.jpg",
                           "wts-lg-000029.jpg",
                           "wts-lg-000125.jpg",
                           "1e241dc8-8f18-4955-8988-03a0ab49f813.jpg",
                           "wts-lg-000102.jpg",
                           "wts-lg-000037.jpg",
                           "wts-lg-000193.jpg",
                           "car15.jpg",
                           "7fbfbe28-aecb-45be-bd05-7cf26acb3c5c.jpg",
                           "wts-lg-000157.jpg",
                           "wts-lg-000068.jpg",
                           "wts-lg-000089.jpg",
                           "wts-lg-000022.jpg",
                           "wts-lg-000013.jpg",
                           "wts-lg-000084.jpg",
                           "wts-lg-000079.jpg",
                           "wts-lg-000025.jpg",
                           "wts-lg-000070.jpg",
                           "wts-lg-000072.jpg",
                           "wts-lg-000170.jpg",
                           "wts-lg-000086.jpg",
                           "wts-lg-000166.jpg",
                           "wts-lg-000132.jpg",
                           "wts-lg-000154.jpg",
                           "car9-7.jpg",
                           "wts-lg-000085.jpg",
                           "wts-lg-000195.jpg",
                           "wts-lg-000159.jpg",
                           "wts-lg-000094.jpg",
                           "wts-lg-000026.jpg",
                           "wts-lg-000117.jpg",
                           "wts-lg-000033.jpg",
                           "wts-lg-000088.jpg",
                           "us10.jpg",
                           "wts-lg-000036.jpg",
                           "wts-lg-000122.jpg",
                           "wts-lg-000031.jpg",
                           "wts-lg-000012.jpg",
                           "wts-lg-000130.jpg",
                           "car9-5.jpg",
                           "wts-lg-000052.jpg",
                           "21d8c31d-3deb-494b-9c63-c0223306fd82.jpg",
                           "wts-lg-000151.jpg",
                           "c9368c55-210d-456c-a5ef-c310e60039ec.jpg",
                           "wts-lg-000135.jpg",
                           "us3.jpg",
                           "wts-lg-000074.jpg",
                           "wts-lg-000048.jpg",
                           "wts-lg-000096.jpg",
                           "wts-lg-000137.jpg",
                           "22e54a62-57a8-4a0a-88c1-4b9758f67651.jpg",
                           "wts-lg-000032.jpg",
                           "wts-lg-000051.jpg",
                           "car5.jpg",
                           "car20.jpg",
                           "wts-lg-000146.jpg",
                           "wts-lg-000150.jpg"}
        self._eu_plates = {"test_027.jpg",
                           "test_088.jpg",
                           "test_073.jpg",
                           "test_074.jpg",
                           "test_019.jpg",
                           "test_086.jpg",
                           "test_091.jpg",
                           "test_071.jpg",
                           "test_072.jpg",
                           "test_041.jpg",
                           "test_006.jpg",
                           "test_053.jpg",
                           "test_079.jpg",
                           "test_005.jpg",
                           "test_036.jpg",
                           "test_047.jpg",
                           "test_064.jpg",
                           "test_063.jpg",
                           "test_012.jpg",
                           "test_008.jpg",
                           "test_050.jpg",
                           "test_018.jpg",
                           "test_016.jpg",
                           "test_085.jpg",
                           "eu6.jpg",
                           "test_083.jpg",
                           "test_046.jpg",
                           "eu3.jpg",
                           "test_044.jpg",
                           "eu8.jpg",
                           "test_049.jpg",
                           "eu7.jpg",
                           "test_058.jpg",
                           "test_097.jpg",
                           "test_052.jpg",
                           "test_020.jpg",
                           "test_078.jpg",
                           "test_081.jpg",
                           "test_090.jpg",
                           "test_076.jpg",
                           "test_087.jpg",
                           "test_054.jpg",
                           "test_092.jpg",
                           "test_055.jpg",
                           "test_094.jpg",
                           "test_015.jpg",
                           "test_032.jpg",
                           "eu5.jpg",
                           "test_031.jpg",
                           "test_068.jpg",
                           "eu11.jpg",
                           "test_017.jpg",
                           "test_028.jpg",
                           "test_089.jpg",
                           "eu2.jpg",
                           "test_002.jpg",
                           "test_024.jpg",
                           "test_034.jpg",
                           "test_096.jpg",
                           "test_082.jpg",
                           "test_013.jpg",
                           "test_042.jpg",
                           "test_045.jpg",
                           "test_069.jpg",
                           "test_080.jpg",
                           "test_026.jpg",
                           "test_048.jpg",
                           "test_043.jpg",
                           "test_066.jpg",
                           "eu10.jpg",
                           "test_075.jpg",
                           "test_035.jpg",
                           "test_023.jpg",
                           "test_033.jpg",
                           "test_025.jpg",
                           "test_095.jpg",
                           "test_070.jpg",
                           "eu1.jpg",
                           "test_022.jpg",
                           "eu9.jpg",
                           "test_010.jpg",
                           "test_029.jpg",
                           "test_056.jpg",
                           "test_021.jpg",
                           "test_037.jpg",
                           "test_067.jpg",
                           "test_039.jpg",
                           "test_065.jpg",
                           "test_061.jpg",
                           "test_007.jpg",
                           "test_084.jpg",
                           "test_001.jpg",
                           "test_093.jpg",
                           "test_062.jpg",
                           "test_009.jpg",
                           "test_077.jpg",
                           "test_040.jpg",
                           "test_011.jpg",
                           "test_057.jpg",
                           "test_014.jpg",
                           "test_003.jpg",
                           "test_060.jpg",
                           "eu4.jpg",
                           "test_051.jpg",
                           "test_059.jpg",
                           "test_004.jpg",
                           "test_038.jpg",
                           "test_030.jpg"}
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

    def br_plates(self) -> set[str]:
        return self._br_plates

    def us_plates(self) -> set[str]:
        return self._us_plates

    def eu_plates(self) -> set[str]:
        return self._eu_plates

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
