"""Backend capability inspection compatible with torch.backends."""

from __future__ import annotations

import ctypes

from . import _C, _device as _dev


class _CPU:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def features() -> str:
        buf = ctypes.create_string_buffer(512)
        _C.lib.gt_cpu_features(buf, 512)
        return buf.value.decode("utf-8", errors="replace")


class _CUDA:
    @staticmethod
    def is_built() -> bool:
        return int(_C.lib.gt_device_count(_dev.DEV_CUDA)) > 0

    class matmul:
        allow_tf32 = False


class _MPS:
    @staticmethod
    def is_available() -> bool:
        return int(_C.lib.gt_device_count(_dev.DEV_MPS)) > 0

    @staticmethod
    def is_built() -> bool:
        return _MPS.is_available()


class _Vulkan:
    @staticmethod
    def is_available() -> bool:
        return int(_C.lib.gt_device_count(_dev.DEV_VULKAN)) > 0


class _CuDNN:
    enabled = False
    benchmark = False
    deterministic = False
    allow_tf32 = False

    @staticmethod
    def is_available() -> bool:
        return False


class _MKLDNN:
    enabled = False

    @staticmethod
    def is_available() -> bool:
        return False


class _OpenMP:
    @staticmethod
    def is_available() -> bool:
        return "OpenMP" in _CPU.features()


class _Quantized:
    engine = "ggml"
    supported_engines = ["ggml"]


cpu = _CPU()
cuda = _CUDA()
mps = _MPS()
vulkan = _Vulkan()
cudnn = _CuDNN()
mkldnn = _MKLDNN()
openmp = _OpenMP()
quantized = _Quantized()


def summary() -> str:
    buf = ctypes.create_string_buffer(512)
    _C.lib.gt_backend_summary(buf, 512)
    return buf.value.decode("utf-8", errors="replace")
