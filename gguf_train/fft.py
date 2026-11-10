"""Fast Fourier Transform (FFT) submodule compatible with torch.fft."""

from __future__ import annotations

from . import _dtype as _dt
from ._creation import arange
from ._functions import FFT1DFn
from ._tensor import Tensor


def _norm_to_code(norm: str | None) -> int:
    if norm in (None, "backward"):
        return 0
    if norm == "forward":
        return 1
    if norm == "ortho":
        return 2
    raise ValueError(f"Invalid FFT norm: {norm!r}")


def fft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out=None) -> Tensor:
    d = dim if dim >= 0 else dim + input.ndim
    length = input.shape[d] if n is None else int(n)
    res = FFT1DFn.apply(input, length, d, 0, _norm_to_code(norm), 0)
    if out is not None:
        out.copy_(res)
        return out
    return res


def ifft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out=None) -> Tensor:
    d = dim if dim >= 0 else dim + input.ndim
    length = input.shape[d] if n is None else int(n)
    res = FFT1DFn.apply(input, length, d, 1, _norm_to_code(norm), 0)
    if out is not None:
        out.copy_(res)
        return out
    return res


def rfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out=None) -> Tensor:
    d = dim if dim >= 0 else dim + input.ndim
    length = input.shape[d] if n is None else int(n)
    full = FFT1DFn.apply(input, length, d, 0, _norm_to_code(norm), 1)
    res = full.narrow(d, 0, length // 2 + 1)
    if out is not None:
        out.copy_(res)
        return out
    return res


def irfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out=None) -> Tensor:
    d = dim if dim >= 0 else dim + input.ndim
    m = input.shape[d]
    length = (2 * (m - 1)) if n is None else int(n)
    # Reconstruct Hermitian conjugate symmetry for full length
    from ._ops import cat
    if length % 2 == 0:
        neg_part = input.narrow(d, 1, max(0, m - 2)).flip((d,)).conj()
    else:
        neg_part = input.narrow(d, 1, max(0, m - 1)).flip((d,)).conj()
    full_spec = cat([input, neg_part], dim=d) if neg_part.shape[d] > 0 else input
    res = FFT1DFn.apply(full_spec, length, d, 1, _norm_to_code(norm), 0).real
    if out is not None:
        out.copy_(res)
        return out
    return res


def fftn(input: Tensor, s=None, dim=None, norm: str | None = None, *, out=None) -> Tensor:
    dims = tuple(range(input.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    if s is None:
        sizes = [None] * len(dims)
    else:
        sizes = list(s)
    x = input
    for d, sz in zip(dims, sizes):
        x = fft(x, n=sz, dim=d, norm=norm)
    if out is not None:
        out.copy_(x)
        return out
    return x


def ifftn(input: Tensor, s=None, dim=None, norm: str | None = None, *, out=None) -> Tensor:
    dims = tuple(range(input.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    sizes = [None] * len(dims) if s is None else list(s)
    x = input
    for d, sz in zip(dims, sizes):
        x = ifft(x, n=sz, dim=d, norm=norm)
    if out is not None:
        out.copy_(x)
        return out
    return x


def fft2(input: Tensor, s=None, dim: tuple[int, int] = (-2, -1), norm: str | None = None, *, out=None) -> Tensor:
    return fftn(input, s=s, dim=dim, norm=norm, out=out)


def ifft2(input: Tensor, s=None, dim: tuple[int, int] = (-2, -1), norm: str | None = None, *, out=None) -> Tensor:
    return ifftn(input, s=s, dim=dim, norm=norm, out=out)


def rfftn(input: Tensor, s=None, dim=None, norm: str | None = None, *, out=None) -> Tensor:
    dims = tuple(range(input.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    sizes = [None] * len(dims) if s is None else list(s)
    x = input
    for d, sz in zip(dims[:-1], sizes[:-1]):
        x = fft(x, n=sz, dim=d, norm=norm)
    x = rfft(x, n=sizes[-1], dim=dims[-1], norm=norm)
    if out is not None:
        out.copy_(x)
        return out
    return x


def irfftn(input: Tensor, s=None, dim=None, norm: str | None = None, *, out=None) -> Tensor:
    dims = tuple(range(input.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    sizes = [None] * len(dims) if s is None else list(s)
    x = irfft(input, n=sizes[-1], dim=dims[-1], norm=norm)
    for d, sz in zip(reversed(dims[:-1]), reversed(sizes[:-1])):
        x = ifft(x, n=sz, dim=d, norm=norm)
    res = x.real
    if out is not None:
        out.copy_(res)
        return out
    return res


def rfft2(input: Tensor, s=None, dim: tuple[int, int] = (-2, -1), norm: str | None = None, *, out=None) -> Tensor:
    return rfftn(input, s=s, dim=dim, norm=norm, out=out)


def irfft2(input: Tensor, s=None, dim: tuple[int, int] = (-2, -1), norm: str | None = None, *, out=None) -> Tensor:
    return irfftn(input, s=s, dim=dim, norm=norm, out=out)


def fftfreq(n: int, d: float = 1.0, *, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    dt = dtype or _dt.float32
    val = 1.0 / (n * d)
    half = (n - 1) // 2 + 1
    from ._ops import cat
    pos = arange(0, half, dtype=dt, device=device)
    neg = arange(-(n // 2), 0, dtype=dt, device=device)
    res = cat([pos, neg], dim=0) * val
    if requires_grad:
        res.requires_grad_(True)
    return res


def rfftfreq(n: int, d: float = 1.0, *, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    dt = dtype or _dt.float32
    val = 1.0 / (n * d)
    res = arange(0, n // 2 + 1, dtype=dt, device=device) * val
    if requires_grad:
        res.requires_grad_(True)
    return res


def fftshift(input: Tensor, dim=None) -> Tensor:
    dims = tuple(range(input.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    shifts = tuple(input.shape[d] // 2 for d in dims)
    return input.roll(shifts, dims=dims)


def ifftshift(input: Tensor, dim=None) -> Tensor:
    dims = tuple(range(input.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    shifts = tuple(-(input.shape[d] // 2) for d in dims)
    return input.roll(shifts, dims=dims)
