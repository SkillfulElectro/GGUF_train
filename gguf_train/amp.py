"""Automatic Mixed Precision (AMP) autocast and GradScaler compatible with torch.amp."""

from __future__ import annotations

import threading

from . import _dtype as _dt

_AMP_LOCAL = threading.local()


def _get_amp_state():
    if not hasattr(_AMP_LOCAL, "enabled"):
        _AMP_LOCAL.enabled = False
        _AMP_LOCAL.dtype = _dt.bfloat16
        _AMP_LOCAL.device_type = "cpu"
    return _AMP_LOCAL


class autocast:
    """Context manager / decorator for mixed-precision training."""

    def __init__(self, device_type: str = "cpu", dtype=None, enabled: bool = True, cache_enabled: bool | None = None):
        (cache_enabled,)
        self.device_type = device_type
        self.dtype = _dt.as_dtype(dtype) if dtype is not None else (_dt.float16 if device_type == "cuda" else _dt.bfloat16)
        self.enabled = bool(enabled)
        self._prev = None

    def __enter__(self):
        st = _get_amp_state()
        self._prev = (st.enabled, st.dtype, st.device_type)
        if self.enabled:
            st.enabled = True
            st.dtype = self.dtype
            st.device_type = self.device_type
        else:
            st.enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        st = _get_amp_state()
        st.enabled, st.dtype, st.device_type = self._prev
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def is_autocast_enabled(device_type: str = "cpu") -> bool:
    (device_type,)
    return bool(_get_amp_state().enabled)


def get_autocast_dtype(device_type: str = "cpu"):
    (device_type,)
    return _get_amp_state().dtype


class GradScaler:
    """Gradient scaler for float16/bfloat16 mixed-precision training."""

    def __init__(
        self,
        device: str = "cpu",
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self.device = device
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._enabled = bool(enabled)
        self._growth_tracker = 0
        self._found_inf = False

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale if self._enabled else 1.0

    def scale(self, outputs):
        if not self._enabled:
            return outputs
        if isinstance(outputs, (list, tuple)):
            return type(outputs)(o * self._scale for o in outputs)
        return outputs * self._scale

    def unscale_(self, optimizer) -> None:
        if not self._enabled:
            return
        inv = 1.0 / self._scale
        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad
                    if bool((~g.isfinite()).any().item()):
                        self._found_inf = True
                    g.mul_(inv)

    def step(self, optimizer, *args, **kwargs):
        if not self._enabled:
            return optimizer.step(*args, **kwargs)
        self.unscale_(optimizer)
        if not self._found_inf:
            return optimizer.step(*args, **kwargs)
        return None

    def update(self, new_scale: float | None = None) -> None:
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
            return
        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def state_dict(self) -> dict:
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._scale = float(state_dict.get("scale", self._scale))
        self._growth_factor = float(state_dict.get("growth_factor", self._growth_factor))
        self._backoff_factor = float(state_dict.get("backoff_factor", self._backoff_factor))
        self._growth_interval = int(state_dict.get("growth_interval", self._growth_interval))
        self._growth_tracker = int(state_dict.get("_growth_tracker", 0))
