"""Thread-local autograd enable/disable, inference_mode, and anomaly detection state."""

from __future__ import annotations

import threading

_tls = threading.local()


def _state():
    if not hasattr(_tls, "grad_enabled"):
        _tls.grad_enabled = True
        _tls.inference_mode = False
        _tls.detect_anomaly = False
        _tls.recorder = None        # see _autograd_engine._Recorder
    return _tls


def is_grad_enabled() -> bool:
    st = _state()
    return bool(st.grad_enabled and not st.inference_mode)


def is_inference_mode_enabled() -> bool:
    return bool(_state().inference_mode)


def is_anomaly_enabled() -> bool:
    return bool(_state().detect_anomaly)


class set_grad_enabled:
    def __init__(self, mode: bool):
        self.mode = bool(mode)
        self.prev = True

    def __enter__(self):
        st = _state()
        self.prev = st.grad_enabled
        st.grad_enabled = self.mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state().grad_enabled = self.prev

    def __call__(self, func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__(self.mode):
                return func(*args, **kwargs)

        return wrapper


class no_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(False)


class enable_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(True)


class inference_mode:
    def __init__(self, mode: bool = True):
        self.mode = bool(mode)
        self.prev_inf = False
        self.prev_grad = True

    def __enter__(self):
        st = _state()
        self.prev_inf = st.inference_mode
        self.prev_grad = st.grad_enabled
        st.inference_mode = self.mode
        if self.mode:
            st.grad_enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        st = _state()
        st.inference_mode = self.prev_inf
        st.grad_enabled = self.prev_grad

    def __call__(self, func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__(self.mode):
                return func(*args, **kwargs)

        return wrapper


class detect_anomaly:
    def __init__(self, check_nan: bool = True):
        self.check_nan = check_nan
        self.prev = False

    def __enter__(self):
        st = _state()
        self.prev = st.detect_anomaly
        st.detect_anomaly = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state().detect_anomaly = self.prev


class set_multithreading_enabled:
    def __init__(self, mode: bool):
        self.mode = bool(mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

