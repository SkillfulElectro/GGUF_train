"""Testing utilities compatible with torch.testing."""

from __future__ import annotations

import math

from . import _dtype as _dt
from ._creation import empty, rand, randint
from ._tensor import Tensor


def make_tensor(*shape, dtype=None, device=None, low=None, high=None, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dt = _dt.as_dtype(dtype) if dtype is not None else _dt.float32
    if dt.is_floating_point or dt.is_complex:
        lo = -9.0 if low is None else float(low)
        hi = 9.0 if high is None else float(high)
        t = rand(shape, dtype=dt, device=device) * (hi - lo) + lo
    elif dt == _dt.bool:
        t = randint(0, 2, shape, dtype=_dt.int64, device=device).bool()
    else:
        lo = -9 if low is None else int(low)
        hi = 9 if high is None else int(high)
        t = randint(lo, hi + 1, shape, dtype=dt, device=device)
    if requires_grad and (dt.is_floating_point or dt.is_complex):
        t.requires_grad_(True)
    return t


def assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    check_dtype: bool = True,
    check_device: bool = True,
    msg: str | None = None,
) -> None:
    if check_device and actual.device != expected.device:
        raise AssertionError(f"Device mismatch: {actual.device} != {expected.device}")
    if check_dtype and actual.dtype != expected.dtype:
        raise AssertionError(f"Dtype mismatch: {actual.dtype} != {expected.dtype}")
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")

    if rtol is None or atol is None:
        defaults = {
            _dt.float16: (1e-3, 1e-5),
            _dt.bfloat16: (1.6e-2, 1e-5),
            _dt.float32: (1.3e-6, 1e-5),
            _dt.float64: (1e-7, 1e-7),
        }
        dr, da = defaults.get(actual.dtype, (0.0, 0.0))
        rtol = dr if rtol is None else rtol
        atol = da if atol is None else atol

    if not actual.allclose(expected, rtol=rtol, atol=atol, equal_nan=equal_nan):
        diff = (actual.float() - expected.float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
        prefix = f"{msg}\n" if msg else ""
        raise AssertionError(
            f"{prefix}Tensor-likes are not close! Mismatch max abs={max_abs} (rtol={rtol}, atol={atol})"
        )


def assert_allclose(actual: Tensor, expected: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    assert_close(actual, expected, rtol=rtol, atol=atol, check_dtype=False, check_device=False)
