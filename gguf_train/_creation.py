"""Tensor creation and random sampling routines."""

from __future__ import annotations

import ctypes
import math
import time

from . import _C, _device as _dev, _dtype as _dt
from ._functions import _c_i64_arr, _wrap
from ._tensor import Tensor, _from_python

_CURRENT_SEED = 0


def _norm_size(*size) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(int(s) for s in size[0])
    return tuple(int(s) for s in size)


def tensor(data, *, dtype=None, device=None, requires_grad: bool = False, pin_memory: bool = False) -> Tensor:
    (pin_memory,)
    return _from_python(data, dtype=dtype, device=device, requires_grad=requires_grad)


def as_tensor(data, dtype=None, device=None) -> Tensor:
    dev = _dev.as_device(device)
    if isinstance(data, Tensor):
        out = data
        if dev is not None and out.device != dev:
            out = out.to(dev)
        if dtype is not None:
            dt = _dt.as_dtype(dtype)
            if out.dtype != dt:
                out = out.to(dt)
        return out
    return _from_python(data, dtype=dtype, device=device)


def from_numpy(ndarray) -> Tensor:
    return _from_python(ndarray)


def scalar_tensor(s, *, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    return _from_python(s, dtype=dtype, device=device, requires_grad=requires_grad)


def empty(*size, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False, pin_memory: bool = False) -> Tensor:
    (layout, pin_memory)
    sh = _norm_size(*size)
    dt = _dt.as_dtype(dtype) if dtype is not None else _dt.get_default_dtype()
    dev = _dev.as_device(device)
    ptr = _C.check_ptr(
        _C.lib.gt_tensor_empty(len(sh), _c_i64_arr(sh), dt.code, dev.dev_type, dev.index),
        "empty",
    )
    t = Tensor(_ptr=ptr, requires_grad=requires_grad)
    if out is not None:
        out.copy_(t)
        return out
    return t


def empty_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return empty(
        input.shape,
        dtype=dtype or input.dtype,
        layout=layout,
        device=device or input.device,
        requires_grad=requires_grad,
    )


def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    t = empty(*size, dtype=dtype, layout=layout, device=device, requires_grad=False)
    _C.check_rc(_C.lib.gt_fill_scalar_i64(t._ptr, 0), "zeros")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def zeros_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return zeros(input.shape, dtype=dtype or input.dtype, layout=layout, device=device or input.device, requires_grad=requires_grad)


def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    t = empty(*size, dtype=dtype, layout=layout, device=device, requires_grad=False)
    dt = t.dtype
    if dt.is_floating_point or dt.is_complex:
        _C.check_rc(_C.lib.gt_fill_scalar_f64(t._ptr, 1.0), "ones")
    else:
        _C.check_rc(_C.lib.gt_fill_scalar_i64(t._ptr, 1), "ones")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def ones_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return ones(input.shape, dtype=dtype or input.dtype, layout=layout, device=device or input.device, requires_grad=requires_grad)


def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    sh = (size,) if isinstance(size, int) else tuple(size)
    if dtype is None:
        if isinstance(fill_value, bool):
            dt = _dt.bool
        elif isinstance(fill_value, int):
            dt = _dt.int64
        else:
            dt = _dt.get_default_dtype()
    else:
        dt = _dt.as_dtype(dtype)
    t = empty(sh, dtype=dt, layout=layout, device=device, requires_grad=False)
    if dt.is_floating_point or dt.is_complex:
        _C.check_rc(_C.lib.gt_fill_scalar_f64(t._ptr, float(fill_value)), "full")
    else:
        _C.check_rc(_C.lib.gt_fill_scalar_i64(t._ptr, int(fill_value)), "full")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def full_like(input: Tensor, fill_value, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return full(input.shape, fill_value, dtype=dtype or input.dtype, layout=layout, device=device or input.device, requires_grad=requires_grad)


def arange(start=0, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    if end is None:
        end = start
        start = 0
    if step == 0:
        raise ValueError("step must be nonzero")
    n = max(0, int(math.ceil((end - start) / step)))
    if dtype is None:
        if isinstance(start, float) or isinstance(end, float) or isinstance(step, float):
            dt = _dt.get_default_dtype()
        else:
            dt = _dt.int64
    else:
        dt = _dt.as_dtype(dtype)
    t = empty((n,), dtype=dt, layout=layout, device=device, requires_grad=False)
    if n > 0:
        _C.check_rc(_C.lib.gt_fill_arange(t._ptr, float(start), float(step)), "arange")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def range(start, end, step=1, *, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return arange(start, end + (step * 1e-9 if isinstance(step, float) else 1), step, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


def linspace(start: float, end: float, steps: int, *, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    dt = _dt.as_dtype(dtype) if dtype is not None else _dt.get_default_dtype()
    t = empty((int(steps),), dtype=dt, layout=layout, device=device, requires_grad=False)
    if steps > 0:
        _C.check_rc(_C.lib.gt_fill_linspace(t._ptr, float(start), float(end), int(steps)), "linspace")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def logspace(start: float, end: float, steps: int, base: float = 10.0, *, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    lin = linspace(start, end, steps, dtype=dtype, layout=layout, device=device, requires_grad=False)
    res = base ** lin
    if requires_grad:
        res.requires_grad_(True)
    if out is not None:
        out.copy_(res)
        return out
    return res


def eye(n: int, m: int | None = None, *, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    m = n if m is None else m
    dt = _dt.as_dtype(dtype) if dtype is not None else _dt.get_default_dtype()
    t = empty((int(n), int(m)), dtype=dt, layout=layout, device=device, requires_grad=False)
    _C.check_rc(_C.lib.gt_fill_eye(t._ptr, int(n), int(m)), "eye")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def rand(*size, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    (generator,)
    t = empty(*size, dtype=dtype, layout=layout, device=device, requires_grad=False)
    _C.check_rc(_C.lib.gt_fill_uniform(t._ptr, 0.0, 1.0), "rand")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def rand_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return rand(input.shape, dtype=dtype or input.dtype, layout=layout, device=device or input.device, requires_grad=requires_grad)


def randn(*size, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    (generator,)
    t = empty(*size, dtype=dtype, layout=layout, device=device, requires_grad=False)
    _C.check_rc(_C.lib.gt_fill_normal(t._ptr, 0.0, 1.0), "randn")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def randn_like(input: Tensor, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    return randn(input.shape, dtype=dtype or input.dtype, layout=layout, device=device or input.device, requires_grad=requires_grad)


def randint(low: int, high: int | None = None, size: tuple[int, ...] | None = None, *, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    (generator,)
    if size is None and isinstance(high, (tuple, list)):
        size = tuple(high)
        high = low
        low = 0
    elif high is None and size is not None:
        high = low
        low = 0
    elif high is None:
        raise TypeError("randint missing size")
    dt = _dt.as_dtype(dtype) if dtype is not None else _dt.int64
    t = empty(size, dtype=dt, layout=layout, device=device, requires_grad=False)
    _C.check_rc(_C.lib.gt_fill_randint(t._ptr, int(low), int(high)), "randint")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def randint_like(input: Tensor, low: int, high: int | None = None, *, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    if high is None:
        high = low
        low = 0
    return randint(low, high, input.shape, dtype=dtype or input.dtype, layout=layout, device=device or input.device, requires_grad=requires_grad)


def randperm(n: int, *, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    (generator,)
    dt = _dt.as_dtype(dtype) if dtype is not None else _dt.int64
    t = empty((int(n),), dtype=dt, layout=layout, device=device, requires_grad=False)
    _C.check_rc(_C.lib.gt_fill_randperm(t._ptr, int(n)), "randperm")
    if requires_grad:
        t.requires_grad_(True)
    if out is not None:
        out.copy_(t)
        return out
    return t


def normal(mean, std=1.0, size=None, *, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad: bool = False) -> Tensor:
    (generator,)
    if size is not None and not isinstance(mean, Tensor) and not isinstance(std, Tensor):
        t = empty(size, dtype=dtype, layout=layout, device=device, requires_grad=False)
        _C.check_rc(_C.lib.gt_fill_normal(t._ptr, float(mean), float(std)), "normal")
        if requires_grad:
            t.requires_grad_(True)
        if out is not None:
            out.copy_(t)
            return out
        return t
    if isinstance(mean, Tensor):
        noise = randn_like(mean)
        if isinstance(std, Tensor):
            res = mean + std * noise
        else:
            res = mean + float(std) * noise
    elif isinstance(std, Tensor):
        noise = randn_like(std)
        res = float(mean) + std * noise
    else:
        raise TypeError("normal expects Tensor mean/std or explicit size")
    if out is not None:
        out.copy_(res)
        return out
    return res


def bernoulli(input: Tensor, *, generator=None, out=None) -> Tensor:
    (generator,)
    res = input.bernoulli()
    if out is not None:
        out.copy_(res)
        return out
    return res


def multinomial(input: Tensor, num_samples: int, replacement: bool = False, *, generator=None, out=None) -> Tensor:
    (generator,)
    res = input.multinomial(num_samples, replacement=replacement)
    if out is not None:
        out.copy_(res)
        return out
    return res


def manual_seed(seed: int) -> _dev.Generator:
    global _CURRENT_SEED
    _CURRENT_SEED = int(seed) & 0xFFFFFFFFFFFFFFFF
    _C.lib.gt_manual_seed(_CURRENT_SEED)
    return _dev.default_generator.manual_seed(_CURRENT_SEED)


def seed() -> int:
    s = int(time.time_ns()) & 0xFFFFFFFFFFFFFFFF
    manual_seed(s)
    return s


def initial_seed() -> int:
    return _CURRENT_SEED


def get_rng_state() -> Tensor:
    import ctypes
    need = _C.lib.gt_rng_get_state(None, 0)
    buf = ctypes.create_string_buffer(need + 16)
    _C.lib.gt_rng_get_state(buf, need + 16)
    raw = buf.value
    return _from_python(list(raw), dtype=_dt.uint8)


def set_rng_state(new_state: Tensor | bytes | bytearray) -> None:
    if isinstance(new_state, (bytes, bytearray)):
        raw = bytes(new_state)
    elif isinstance(new_state, Tensor):
        raw = bytes(int(v) & 0xFF for v in new_state.reshape((-1,)).tolist())
    else:
        raise TypeError("set_rng_state expects a uint8 Tensor or bytes")
    if len(raw) == 8:
        manual_seed(int.from_bytes(raw, "little"))
        return
    _C.check_rc(_C.lib.gt_rng_set_state(raw), "set_rng_state")
