"""
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def assert_single_channel(fn: Callable[[np.ndarray], np.ndarray]) -> Callable:
    """Decorador que adiciona um assert para garantir
    que a imagem só possui um canal.

    Args:
        fn (Callable[[np.ndarray], np.ndarray]): função a ser aplicada.

    Returns:
        Callable: função a ser aplicada com assert prévio de canal único.
    """
    def wrapper(image: np.ndarray) -> np.ndarray:
        assert len(image.shape) == 1

        return fn(image)

    return wrapper
