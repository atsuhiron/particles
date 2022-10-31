from typing import Union
from typing import Tuple
import importlib

import numpy as np
import cupy as cp

# type alias
xparray = Union[np.ndarray, cp.ndarray]

# Note: CUDA 環境がある場合、これを True にする。
_TRY_USE_CUPY = False

_SHOW_XP_ENV = False
_IS_CUPY_ENV = False


def is_cupy_env() -> bool:
    return _IS_CUPY_ENV


def import_numpy_or_cupy():
    global _IS_CUPY_ENV, _SHOW_XP_ENV

    if not _TRY_USE_CUPY:
        if not _SHOW_XP_ENV:
            print("using numpy")
            _IS_CUPY_ENV = False
        _SHOW_XP_ENV = True
        return importlib.import_module("numpy")
    try:
        if not _SHOW_XP_ENV:
            print("using cupy")
            _IS_CUPY_ENV = True
        _SHOW_XP_ENV = True
        return importlib.import_module("cupy")
    except Exception:
        if not _SHOW_XP_ENV:
            print("using numpy")
            _IS_CUPY_ENV = False
        _SHOW_XP_ENV = True
        return importlib.import_module("numpy")


def to_numpy(*args: xparray) -> Tuple[np.ndarray, ...]:
    new_arrays = []
    for v in args:
        if isinstance(v, cp.ndarray):
            new_arrays.append(v.get())
        else:
            new_arrays.append(v)
    return tuple(new_arrays)
