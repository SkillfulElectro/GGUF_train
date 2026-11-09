"""Apple Metal / MPS backend queries compatible with torch.mps."""

from __future__ import annotations

import ctypes

from . import _C, _device as _dev


def is_available() -> bool:
    return int(_C.lib.gt_device_count(_dev.DEV_MPS)) > 0


def is_built() -> bool:
    return True


def device_count() -> int:
    return int(_C.lib.gt_device_count(_dev.DEV_MPS))


def synchronize() -> None:
    if is_available():
        _C.lib.gt_synchronize(_dev.DEV_MPS, 0)


def empty_cache() -> None:
    pass


def manual_seed(seed: int) -> None:
    _C.lib.gt_manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


def current_allocated_memory() -> int:
    if not is_available():
        return 0
    f = ctypes.c_size_t(0)
    t = ctypes.c_size_t(0)
    _C.lib.gt_device_memory(_dev.DEV_MPS, 0, ctypes.byref(f), ctypes.byref(t))
    return max(0, int(t.value) - int(f.value))
