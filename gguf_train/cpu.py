"""CPU backend utilities compatible with torch.cpu."""

from __future__ import annotations

from .amp import autocast as _amp_autocast
from .backends import cpu as _cpu_backend


def is_available() -> bool:
    return True


def device_count() -> int:
    return 1


def current_device() -> int:
    return 0


def synchronize(device=None) -> None:
    (device,)


def features() -> str:
    return _cpu_backend.features()


def autocast(enabled: bool = True, dtype=None, cache_enabled: bool = True):
    return _amp_autocast("cpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


class Stream:
    def __init__(self, device=None) -> None:
        self.device = device

    def synchronize(self) -> None:
        pass


__all__ = ["is_available", "device_count", "current_device", "synchronize", "features", "autocast", "Stream"]
