"""CUDA / HIP / GPU backend queries compatible with torch.cuda."""

from __future__ import annotations

import ctypes

from . import _C, _device as _dev
from .amp import GradScaler, autocast

_CURRENT_DEVICE = 0


def is_available() -> bool:
    return int(_C.lib.gt_device_count(_dev.DEV_CUDA)) > 0


def is_initialized() -> bool:
    return True


def device_count() -> int:
    return int(_C.lib.gt_device_count(_dev.DEV_CUDA))


def current_device() -> int:
    return _CURRENT_DEVICE


def set_device(device) -> None:
    global _CURRENT_DEVICE
    _CURRENT_DEVICE = device.index if isinstance(device, _dev.device) else int(device)


def get_device_name(device=None) -> str:
    idx = _CURRENT_DEVICE if device is None else (device.index if isinstance(device, _dev.device) else int(device))
    buf = ctypes.create_string_buffer(256)
    _C.lib.gt_device_name(_dev.DEV_CUDA, idx, buf, 256)
    return buf.value.decode("utf-8", errors="replace")


def get_device_capability(device=None) -> tuple[int, int]:
    (device,)
    return (8, 0) if is_available() else (0, 0)


class _DeviceProperties:
    def __init__(self, idx: int):
        self.name = get_device_name(idx)
        free_b, total_b = mem_get_info(idx)
        self.total_memory = total_b
        self.major, self.minor = get_device_capability(idx)
        self.multi_processor_count = 1


def get_device_properties(device=None) -> _DeviceProperties:
    idx = _CURRENT_DEVICE if device is None else (device.index if isinstance(device, _dev.device) else int(device))
    return _DeviceProperties(idx)


def mem_get_info(device=None) -> tuple[int, int]:
    idx = _CURRENT_DEVICE if device is None else (device.index if isinstance(device, _dev.device) else int(device))
    f = ctypes.c_size_t(0)
    t = ctypes.c_size_t(0)
    _C.lib.gt_device_memory(_dev.DEV_CUDA, idx, ctypes.byref(f), ctypes.byref(t))
    return int(f.value), int(t.value)


def synchronize(device=None) -> None:
    idx = _CURRENT_DEVICE if device is None else (device.index if isinstance(device, _dev.device) else int(device))
    if is_available():
        _C.lib.gt_synchronize(_dev.DEV_CUDA, idx)


def empty_cache() -> None:
    pass


def manual_seed(seed: int) -> None:
    _C.lib.gt_manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


def manual_seed_all(seed: int) -> None:
    manual_seed(seed)


def memory_allocated(device=None) -> int:
    free_b, total_b = mem_get_info(device)
    return max(0, total_b - free_b)


def max_memory_allocated(device=None) -> int:
    return memory_allocated(device)


def reset_peak_memory_stats(device=None) -> None:
    (device,)


class _CudaAutocast(autocast):
    """torch.cuda.amp.autocast(enabled=True, dtype=torch.float16): autocast for CUDA tensors."""

    def __init__(self, enabled: bool = True, dtype=None, cache_enabled=None):
        super().__init__("cuda", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


class _CudaAmp:
    autocast = _CudaAutocast
    GradScaler = GradScaler


amp = _CudaAmp()
