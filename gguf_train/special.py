"""Special mathematical functions compatible with torch.special."""

from __future__ import annotations

import math

from ._ops import (
    digamma,
    erf,
    erfc,
    erfinv,
    exp2,
    expm1,
    lgamma,
    log1p,
    log_softmax,
    logsumexp,
    sigmoid as expit,
    softmax,
    where,
)
from ._creation import ones_like, zeros_like
from ._tensor import Tensor, _AUTOCAST_KEEP, _autocast_op

gammaln = lgamma


# The functions written here in terms of log, exp and sums keep the dtype of their input under
# autocast (torch.special has them as single operations, outside the autocast lists).
@_autocast_op(_AUTOCAST_KEEP)
def logit(input: Tensor, eps: float | None = None, *, out=None) -> Tensor:
    x = input if eps is None else input.clamp(min=eps, max=1.0 - eps)
    res = (x / (1.0 - x)).log()
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def sinc(input: Tensor, *, out=None) -> Tensor:
    px = input * math.pi
    res = where(input == 0, ones_like(input), px.sin() / px)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def entr(input: Tensor, *, out=None) -> Tensor:
    pos = -input * input.clamp(min=1e-38).log()
    res = where(input > 0, pos, where(input == 0, zeros_like(input), zeros_like(input) - float("inf")))
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def xlogy(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    res = where(input == 0, zeros_like(input), input * other.log())
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def ndtr(input: Tensor, *, out=None) -> Tensor:
    res = 0.5 * (1.0 + (input / math.sqrt(2.0)).erf())
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def ndtri(input: Tensor, *, out=None) -> Tensor:
    res = math.sqrt(2.0) * (2.0 * input - 1.0).erfinv()
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def i0(input: Tensor, *, out=None) -> Tensor:
    # Polynomial/series expansion for modified Bessel function I_0
    ax = input.abs()
    res = ones_like(ax)
    term = ones_like(ax)
    x2 = (ax * 0.5) ** 2
    for k in range(1, 20):
        term = term * x2 / (k * k)
        res = res + term
    if out is not None:
        out.copy_(res)
        return out
    return res
