from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any
import math
import numbers
import numpy as np

ACTION_NAMES = ("steer", "acceleration", "brake", "nitro", "drift", "fire", "rescue")
BINARY_ACTIONS = ACTION_NAMES[2:]

@dataclass(frozen=True)
class Action:
    """Controles deseados. No se recortan ni corrigen silenciosamente."""
    steer: float = 0.0
    acceleration: float = 0.0
    brake: bool = False
    nitro: bool = False
    drift: bool = False
    fire: bool = False
    rescue: bool = False

    def __post_init__(self):
        for key, low, high in (("steer", -1.0, 1.0), ("acceleration", 0.0, 1.0)):
            v = getattr(self, key)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
                raise TypeError(f"{key} debe ser un escalar real, no {type(v).__name__}")
            if not math.isfinite(float(v)) or not low <= float(v) <= high:
                raise ValueError(f"{key}={v!r} fuera de [{low}, {high}]")
            object.__setattr__(self, key, float(v))
        for key in BINARY_ACTIONS:
            v = getattr(self, key)
            if not isinstance(v, (bool, np.bool_, numbers.Integral)) or int(v) not in (0, 1):
                raise TypeError(f"{key} debe ser bool o entero 0/1")
            object.__setattr__(self, key, bool(v))

    @classmethod
    def from_any(cls, value: Any) -> "Action":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("act(rgb) debe devolver Action o dict con los siete campos")
        if set(value) != set(ACTION_NAMES):
            raise ValueError(f"Campos requeridos exactamente: {ACTION_NAMES}")
        return cls(**value)

    def as_dict(self) -> dict:
        return asdict(self)

    def as_array(self) -> np.ndarray:
        return np.asarray([getattr(self, k) for k in ACTION_NAMES], dtype=np.float32)


def check_rgb(rgb: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"RGB debe ser uint8 [H,W,3], recibido {a.dtype} {a.shape}")
    if width is not None and a.shape != (height, width, 3):
        raise ValueError(f"Resolución inesperada: {a.shape}; esperado {(height, width, 3)}")
    return a
