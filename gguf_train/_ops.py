"""Top-level PyTorch-compatible tensor operations and math functions."""

from __future__ import annotations

import builtins
import math
import string

from . import _C, _dtype as _dt
from ._creation import arange, zeros
from ._functions import CatFn, _wrap
from ._tensor import Tensor, _ensure_tensor


def _out_handle(res: Tensor, out: Tensor | None) -> Tensor:
    if out is not None:
        out.copy_(res)
        return out
    return res


def is_tensor(obj) -> bool:
    return isinstance(obj, Tensor)


def is_floating_point(input: Tensor) -> bool:
    return input.dtype.is_floating_point


def is_complex(input: Tensor) -> bool:
    return input.dtype.is_complex


def numel(input: Tensor) -> int:
    return input.numel()


def clone(input: Tensor, *, memory_format=None) -> Tensor:
    return input.clone(memory_format=memory_format)


# ---------- Unary Ops ----------

def neg(input: Tensor, *, out=None):
    return _out_handle(input.neg(), out)

def abs(input: Tensor, *, out=None):
    return _out_handle(input.abs(), out)

def sign(input: Tensor, *, out=None):
    return _out_handle(input.sign(), out)

def sqrt(input: Tensor, *, out=None):
    return _out_handle(input.sqrt(), out)

def rsqrt(input: Tensor, *, out=None):
    return _out_handle(input.rsqrt(), out)

def square(input: Tensor, *, out=None):
    return _out_handle(input.square(), out)

def exp(input: Tensor, *, out=None):
    return _out_handle(input.exp(), out)

def expm1(input: Tensor, *, out=None):
    return _out_handle(input.expm1(), out)

def exp2(input: Tensor, *, out=None):
    return _out_handle(input.exp2(), out)

def log(input: Tensor, *, out=None):
    return _out_handle(input.log(), out)

def log2(input: Tensor, *, out=None):
    return _out_handle(input.log2(), out)

def log10(input: Tensor, *, out=None):
    return _out_handle(input.log10(), out)

def log1p(input: Tensor, *, out=None):
    return _out_handle(input.log1p(), out)

def sin(input: Tensor, *, out=None):
    return _out_handle(input.sin(), out)

def cos(input: Tensor, *, out=None):
    return _out_handle(input.cos(), out)

def tan(input: Tensor, *, out=None):
    return _out_handle(input.tan(), out)

def asin(input: Tensor, *, out=None):
    return _out_handle(input.asin(), out)

def acos(input: Tensor, *, out=None):
    return _out_handle(input.acos(), out)

def atan(input: Tensor, *, out=None):
    return _out_handle(input.atan(), out)

def atan2(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(input.atan2(other), out)

def sinh(input: Tensor, *, out=None):
    return _out_handle(input.sinh(), out)

def cosh(input: Tensor, *, out=None):
    return _out_handle(input.cosh(), out)

def tanh(input: Tensor, *, out=None):
    return _out_handle(input.tanh(), out)

def asinh(input: Tensor, *, out=None):
    return _out_handle(input.asinh(), out)

def acosh(input: Tensor, *, out=None):
    return _out_handle(input.acosh(), out)

def atanh(input: Tensor, *, out=None):
    return _out_handle(input.atanh(), out)

def erf(input: Tensor, *, out=None):
    return _out_handle(input.erf(), out)

def erfc(input: Tensor, *, out=None):
    return _out_handle(input.erfc(), out)

def erfinv(input: Tensor, *, out=None):
    return _out_handle(input.erfinv(), out)

def lgamma(input: Tensor, *, out=None):
    return _out_handle(input.lgamma(), out)

def digamma(input: Tensor, *, out=None):
    return _out_handle(input.digamma(), out)

def floor(input: Tensor, *, out=None):
    return _out_handle(input.floor(), out)

def ceil(input: Tensor, *, out=None):
    return _out_handle(input.ceil(), out)

def round(input: Tensor, *, out=None):
    return _out_handle(input.round(), out)

def trunc(input: Tensor, *, out=None):
    return _out_handle(input.trunc(), out)

def frac(input: Tensor, *, out=None):
    return _out_handle(input.frac(), out)

def reciprocal(input: Tensor, *, out=None):
    return _out_handle(input.reciprocal(), out)

def relu(input: Tensor):
    return input.relu()

def sigmoid(input: Tensor, *, out=None):
    return _out_handle(input.sigmoid(), out)

def silu(input: Tensor):
    return input.silu()

def gelu(input: Tensor):
    return input.gelu()

def clamp(input: Tensor, min=None, max=None, *, out=None):
    return _out_handle(input.clamp(min=min, max=max), out)

def clip(input: Tensor, min=None, max=None, *, out=None):
    return clamp(input, min=min, max=max, out=out)

def isnan(input: Tensor):
    return input.isnan()

def isinf(input: Tensor):
    return input.isinf()

def isfinite(input: Tensor):
    return input.isfinite()

def nan_to_num(input: Tensor, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None, *, out=None):
    return _out_handle(input.nan_to_num(nan=nan, posinf=posinf, neginf=neginf), out)

def real(input: Tensor):
    return input.real

def imag(input: Tensor):
    return input.imag

def conj(input: Tensor):
    return input.conj()

def angle(input: Tensor):
    return input.angle()


# ---------- Binary / Comparison / Ternary ----------

def add(input: Tensor, other, *, alpha: float = 1.0, out=None):
    return _out_handle(input.add(other, alpha=alpha), out)

def sub(input: Tensor, other, *, alpha: float = 1.0, out=None):
    return _out_handle(input.sub(other, alpha=alpha), out)

def mul(input: Tensor, other, *, out=None):
    return _out_handle(input.mul(other), out)

def div(input: Tensor, other, *, rounding_mode: str | None = None, out=None):
    return _out_handle(input.div(other, rounding_mode=rounding_mode), out)

def true_divide(input: Tensor, other, *, out=None):
    return div(input, other, out=out)

def floor_divide(input: Tensor, other, *, out=None):
    return _out_handle(input // other, out)

def pow(input: Tensor, exponent, *, out=None):
    return _out_handle(input.pow(exponent), out)

def maximum(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(input.maximum(other), out)

def minimum(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(input.minimum(other), out)

def where(condition: Tensor, input=None, other=None):
    if input is None and other is None:
        return condition.nonzero(as_tuple=True)
    x = input if isinstance(input, Tensor) else _ensure_tensor(input, other)
    y = other if isinstance(other, Tensor) else _ensure_tensor(other, x)
    return x.where(condition, y)

def lerp(input: Tensor, end: Tensor, weight, *, out=None):
    return _out_handle(input.lerp(end, weight), out)

def eq(input: Tensor, other, *, out=None):
    return _out_handle(input == other, out)

def ne(input: Tensor, other, *, out=None):
    return _out_handle(input != other, out)

def lt(input: Tensor, other, *, out=None):
    return _out_handle(input < other, out)

def le(input: Tensor, other, *, out=None):
    return _out_handle(input <= other, out)

def gt(input: Tensor, other, *, out=None):
    return _out_handle(input > other, out)

def ge(input: Tensor, other, *, out=None):
    return _out_handle(input >= other, out)

def logical_not(input: Tensor, *, out=None):
    return _out_handle(input.logical_not(), out)

def logical_and(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(_wrap(_C.lib.gt_binary_op(input._ptr, _ensure_tensor(other, input)._ptr, _C.BIN_LOGICAL_AND, 1.0)), out)

def logical_or(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(_wrap(_C.lib.gt_binary_op(input._ptr, _ensure_tensor(other, input)._ptr, _C.BIN_LOGICAL_OR, 1.0)), out)

def logical_xor(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(_wrap(_C.lib.gt_binary_op(input._ptr, _ensure_tensor(other, input)._ptr, _C.BIN_LOGICAL_XOR, 1.0)), out)

def bitwise_and(input: Tensor, other, *, out=None):
    return _out_handle(input & other, out)

def bitwise_or(input: Tensor, other, *, out=None):
    return _out_handle(input | other, out)

def bitwise_xor(input: Tensor, other, *, out=None):
    return _out_handle(input ^ other, out)

def equal(input: Tensor, other: Tensor) -> bool:
    return input.equal(other)

def allclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
    return input.allclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan)

def isclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> Tensor:
    ok = (input - other).abs() <= (atol + rtol * other.abs())
    if equal_nan:
        ok = ok | (input.isnan() & other.isnan())
    return ok


# ---------- Shape / Concatenation / Indexing ----------

def cat(tensors, dim: int = 0, *, out=None) -> Tensor:
    ts = list(tensors)
    if not ts:
        raise RuntimeError("cat expects a non-empty list of Tensors")
    if len(ts) == 1:
        res = ts[0].clone()
    else:
        res = CatFn.apply(dim, *ts)
    return _out_handle(res, out)

def concat(tensors, dim: int = 0, *, out=None) -> Tensor:
    return cat(tensors, dim=dim, out=out)

def concatenate(tensors, dim: int = 0, *, out=None) -> Tensor:
    return cat(tensors, dim=dim, out=out)

def stack(tensors, dim: int = 0, *, out=None) -> Tensor:
    ts = list(tensors)
    if not ts:
        raise RuntimeError("stack expects a non-empty list of Tensors")
    expanded = [t.unsqueeze(dim) for t in ts]
    return cat(expanded, dim=dim, out=out)

def hstack(tensors, *, out=None) -> Tensor:
    ts = list(tensors)
    dim = 0 if ts[0].ndim == 1 else 1
    return cat(ts, dim=dim, out=out)

def vstack(tensors, *, out=None) -> Tensor:
    ts = [t.unsqueeze(0) if t.ndim == 1 else t for t in tensors]
    return cat(ts, dim=0, out=out)

def dstack(tensors, *, out=None) -> Tensor:
    ts = []
    for t in tensors:
        if t.ndim == 1:
            ts.append(t.unsqueeze(0).unsqueeze(-1))
        elif t.ndim == 2:
            ts.append(t.unsqueeze(-1))
        else:
            ts.append(t)
    return cat(ts, dim=2, out=out)

def column_stack(tensors, *, out=None) -> Tensor:
    ts = [t.unsqueeze(1) if t.ndim == 1 else t for t in tensors]
    return cat(ts, dim=1, out=out)

def row_stack(tensors, *, out=None) -> Tensor:
    return vstack(tensors, out=out)

def split(tensor: Tensor, split_size_or_sections, dim: int = 0):
    return tensor.split(split_size_or_sections, dim=dim)

def chunk(input: Tensor, chunks: int, dim: int = 0):
    return input.chunk(chunks, dim=dim)

def tensor_split(input: Tensor, indices_or_sections, dim: int = 0):
    return input.tensor_split(indices_or_sections, dim=dim)

def unbind(input: Tensor, dim: int = 0):
    return input.unbind(dim=dim)

def reshape(input: Tensor, shape) -> Tensor:
    return input.reshape(shape)

def squeeze(input: Tensor, dim=None) -> Tensor:
    return input.squeeze(dim=dim)

def unsqueeze(input: Tensor, dim: int) -> Tensor:
    return input.unsqueeze(dim)

def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    return input.transpose(dim0, dim1)

def permute(input: Tensor, dims) -> Tensor:
    return input.permute(dims)

def movedim(input: Tensor, source, destination) -> Tensor:
    return input.movedim(source, destination)

def moveaxis(input: Tensor, source, destination) -> Tensor:
    return input.moveaxis(source, destination)

def swapaxes(input: Tensor, axis0: int, axis1: int) -> Tensor:
    return input.swapaxes(axis0, axis1)

def swapdims(input: Tensor, dim0: int, dim1: int) -> Tensor:
    return input.swapdims(dim0, dim1)

def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    return input.flatten(start_dim=start_dim, end_dim=end_dim)

def narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor:
    return input.narrow(dim, start, length)

def select(input: Tensor, dim: int, index: int) -> Tensor:
    return input.select(dim, index)

def index_select(input: Tensor, dim: int, index: Tensor, *, out=None) -> Tensor:
    return _out_handle(input.index_select(dim, index), out)

def gather(input: Tensor, dim: int, index: Tensor, *, out=None) -> Tensor:
    return _out_handle(input.gather(dim, index), out)

def scatter(input: Tensor, dim: int, index: Tensor, src) -> Tensor:
    return input.scatter(dim, index, src)

def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    return input.scatter_add(dim, index, src)

def masked_select(input: Tensor, mask: Tensor, *, out=None) -> Tensor:
    return _out_handle(input.masked_select(mask), out)

def flip(input: Tensor, dims) -> Tensor:
    return input.flip(dims)

def roll(input: Tensor, shifts, dims=None) -> Tensor:
    return input.roll(shifts, dims=dims)

def rot90(input: Tensor, k: int = 1, dims=(0, 1)) -> Tensor:
    return input.rot90(k=k, dims=dims)

def tril(input: Tensor, diagonal: int = 0, *, out=None) -> Tensor:
    return _out_handle(input.tril(diagonal=diagonal), out)

def triu(input: Tensor, diagonal: int = 0, *, out=None) -> Tensor:
    return _out_handle(input.triu(diagonal=diagonal), out)

def tril_indices(row: int, col: int, offset: int = 0, *, dtype=_dt.int64, device=None, layout=None):
    (layout,)
    pairs_r, pairs_c = [], []
    for r in range(row):
        for c in range(builtins.min(col, r + offset + 1)):
            pairs_r.append(r)
            pairs_c.append(c)
    from ._tensor import _from_python
    return _from_python([pairs_r, pairs_c], dtype=dtype, device=device)

def triu_indices(row: int, col: int, offset: int = 0, *, dtype=_dt.int64, device=None, layout=None):
    (layout,)
    pairs_r, pairs_c = [], []
    for r in range(row):
        for c in range(builtins.max(0, r + offset), col):
            pairs_r.append(r)
            pairs_c.append(c)
    from ._tensor import _from_python
    return _from_python([pairs_r, pairs_c], dtype=dtype, device=device)

def diag(input: Tensor, diagonal: int = 0, *, out=None) -> Tensor:
    return _out_handle(input.diag(diagonal=diagonal), out)

def diagonal(input: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
    return input.diagonal(offset=offset, dim1=dim1, dim2=dim2)

def trace(input: Tensor) -> Tensor:
    return input.trace()

def nonzero(input: Tensor, *, out=None, as_tuple: bool = False):
    res = input.nonzero(as_tuple=as_tuple)
    if out is not None and not as_tuple:
        out.copy_(res)
        return out
    return res

def bincount(input: Tensor, weights: Tensor | None = None, minlength: int = 0) -> Tensor:
    return input.bincount(weights=weights, minlength=minlength)

def unique(input: Tensor, sorted: bool = True, return_inverse: bool = False, return_counts: bool = False, dim: int | None = None):
    return input.unique(sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

def meshgrid(*tensors, indexing: str | None = "ij"):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    ts = [t.flatten() for t in tensors]
    if indexing == "xy" and len(ts) >= 2:
        ts[0], ts[1] = ts[1], ts[0]
    shape = tuple(t.numel() for t in ts)
    grids = []
    for i, t in enumerate(ts):
        vsh = [1] * len(ts)
        vsh[i] = t.numel()
        grids.append(t.reshape(vsh).expand(shape).clone())
    if indexing == "xy" and len(grids) >= 2:
        grids[0], grids[1] = grids[1], grids[0]
    return tuple(grids)

def cartesian_prod(*tensors):
    grids = meshgrid(*tensors, indexing="ij")
    return stack([g.flatten() for g in grids], dim=-1)


# ---------- Reductions & Sorting ----------

def sum(input: Tensor, dim=None, keepdim: bool = False, *, dtype=None, out=None):
    return _out_handle(input.sum(dim=dim, keepdim=keepdim, dtype=dtype), out)

def mean(input: Tensor, dim=None, keepdim: bool = False, *, dtype=None, out=None):
    return _out_handle(input.mean(dim=dim, keepdim=keepdim, dtype=dtype), out)

def prod(input: Tensor, dim=None, keepdim: bool = False, *, dtype=None, out=None):
    return _out_handle(input.prod(dim=dim, keepdim=keepdim, dtype=dtype), out)

def logsumexp(input: Tensor, dim, keepdim: bool = False, *, out=None):
    return _out_handle(input.logsumexp(dim=dim, keepdim=keepdim), out)

def max(input: Tensor, dim_or_other=None, keepdim: bool = False):
    if isinstance(dim_or_other, Tensor):
        return maximum(input, dim_or_other)
    return input.max(dim=dim_or_other, keepdim=keepdim)

def min(input: Tensor, dim_or_other=None, keepdim: bool = False):
    if isinstance(dim_or_other, Tensor):
        return minimum(input, dim_or_other)
    return input.min(dim=dim_or_other, keepdim=keepdim)

def amax(input: Tensor, dim=None, keepdim: bool = False, *, out=None):
    return _out_handle(input.amax(dim=dim, keepdim=keepdim), out)

def amin(input: Tensor, dim=None, keepdim: bool = False, *, out=None):
    return _out_handle(input.amin(dim=dim, keepdim=keepdim), out)

def argmax(input: Tensor, dim: int | None = None, keepdim: bool = False):
    return input.argmax(dim=dim, keepdim=keepdim)

def argmin(input: Tensor, dim: int | None = None, keepdim: bool = False):
    return input.argmin(dim=dim, keepdim=keepdim)

def all(input: Tensor, dim=None, keepdim: bool = False, *, out=None):
    return _out_handle(input.all(dim=dim, keepdim=keepdim), out)

def any(input: Tensor, dim=None, keepdim: bool = False, *, out=None):
    return _out_handle(input.any(dim=dim, keepdim=keepdim), out)

def var(input: Tensor, dim=None, unbiased: bool = True, keepdim: bool = False, *, correction: int | None = None, out=None):
    return _out_handle(input.var(dim=dim, unbiased=unbiased, keepdim=keepdim, correction=correction), out)

def std(input: Tensor, dim=None, unbiased: bool = True, keepdim: bool = False, *, correction: int | None = None, out=None):
    return _out_handle(input.std(dim=dim, unbiased=unbiased, keepdim=keepdim, correction=correction), out)

def var_mean(input: Tensor, dim=None, *, correction: int | None = 1, keepdim: bool = False):
    return (
        input.var(dim=dim, keepdim=keepdim, correction=correction),
        input.mean(dim=dim, keepdim=keepdim),
    )

def std_mean(input: Tensor, dim=None, *, correction: int | None = 1, keepdim: bool = False):
    return (
        input.std(dim=dim, keepdim=keepdim, correction=correction),
        input.mean(dim=dim, keepdim=keepdim),
    )

def norm(input: Tensor, p: float | str = "fro", dim=None, keepdim: bool = False, dtype=None, *, out=None):
    return _out_handle(input.norm(p=p, dim=dim, keepdim=keepdim, dtype=dtype), out)

def cumsum(input: Tensor, dim: int, *, dtype=None, out=None):
    x = input if dtype is None else input.to(dtype)
    return _out_handle(x.cumsum(dim), out)

def cumprod(input: Tensor, dim: int, *, dtype=None, out=None):
    x = input if dtype is None else input.to(dtype)
    return _out_handle(x.cumprod(dim), out)

def diff(input: Tensor, n: int = 1, dim: int = -1, prepend=None, append=None):
    parts = []
    if prepend is not None:
        parts.append(prepend)
    parts.append(input)
    if append is not None:
        parts.append(append)
    x = parts[0] if len(parts) == 1 else cat(parts, dim=dim)
    d = dim if dim >= 0 else dim + x.ndim
    for _ in range(n):
        sz = x.shape[d]
        x = x.narrow(d, 1, sz - 1) - x.narrow(d, 0, sz - 1)
    return x

def sort(input: Tensor, dim: int = -1, descending: bool = False):
    return input.sort(dim=dim, descending=descending)

def argsort(input: Tensor, dim: int = -1, descending: bool = False):
    return input.argsort(dim=dim, descending=descending)

def topk(input: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    return input.topk(k, dim=dim, largest=largest, sorted=sorted)

def kthvalue(input: Tensor, k: int, dim: int = -1, keepdim: bool = False):
    return input.kthvalue(k, dim=dim, keepdim=keepdim)

def softmax(input: Tensor, dim: int, dtype=None):
    return input.softmax(dim=dim, dtype=dtype)

def log_softmax(input: Tensor, dim: int, dtype=None):
    return input.log_softmax(dim=dim, dtype=dtype)


# ---------- Matrix & Tensor Contractions ----------

def matmul(input: Tensor, other: Tensor, *, out=None):
    return _out_handle(input.matmul(other), out)

def mm(input: Tensor, mat2: Tensor, *, out=None):
    return _out_handle(input.mm(mat2), out)

def bmm(input: Tensor, mat2: Tensor, *, out=None):
    return _out_handle(input.bmm(mat2), out)

def mv(input: Tensor, vec: Tensor, *, out=None):
    return _out_handle(input.mv(vec), out)

def dot(input: Tensor, tensor: Tensor, *, out=None):
    return _out_handle(input.dot(tensor), out)

def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, *, beta: float = 1.0, alpha: float = 1.0, out=None):
    return _out_handle(input.addmm(mat1, mat2, beta=beta, alpha=alpha), out)

def baddbmm(input: Tensor, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0, alpha: float = 1.0, out=None):
    return _out_handle(input.baddbmm(batch1, batch2, beta=beta, alpha=alpha), out)

def outer(input: Tensor, vec2: Tensor, *, out=None):
    return _out_handle(input.outer(vec2), out)

def ger(input: Tensor, vec2: Tensor, *, out=None):
    return outer(input, vec2, out=out)

def tensordot(a: Tensor, b: Tensor, dims=2, *, out=None) -> Tensor:
    if isinstance(dims, int):
        dims_a = list(range(a.ndim - dims, a.ndim))
        dims_b = list(range(0, dims))
    else:
        dims_a = [d if d >= 0 else d + a.ndim for d in dims[0]]
        dims_b = [d if d >= 0 else d + b.ndim for d in dims[1]]
    free_a = [i for i in range(a.ndim) if i not in dims_a]
    free_b = [i for i in range(b.ndim) if i not in dims_b]
    perm_a = free_a + dims_a
    perm_b = dims_b + free_b
    a_p = a.permute(perm_a)
    b_p = b.permute(perm_b)
    k = 1
    for d in dims_a:
        k *= a.shape[d]
    free_a_sh = tuple(a.shape[d] for d in free_a)
    free_b_sh = tuple(b.shape[d] for d in free_b)
    m = math.prod(free_a_sh) if free_a_sh else 1
    n = math.prod(free_b_sh) if free_b_sh else 1
    res = a_p.reshape((m, k)).matmul(b_p.reshape((k, n))).reshape(free_a_sh + free_b_sh)
    return _out_handle(res, out)

def kron(input: Tensor, other: Tensor) -> Tensor:
    a, b = input, other
    nd = builtins.max(a.ndim, b.ndim)
    while a.ndim < nd:
        a = a.unsqueeze(0)
    while b.ndim < nd:
        b = b.unsqueeze(0)
    sh_a = a.shape
    sh_b = b.shape
    exp_a = []
    exp_b = []
    out_sh = []
    for sa, sb in zip(sh_a, sh_b):
        exp_a.extend([sa, 1])
        exp_b.extend([1, sb])
        out_sh.append(sa * sb)
    prod = a.reshape(exp_a) * b.reshape(exp_b)
    return prod.reshape(out_sh)


def pad(input: Tensor, pad: tuple[int, ...] | list[int], mode: str = "constant", value: float | None = None) -> Tensor:
    from .nn.functional import pad as _f_pad
    return _f_pad(input, pad, mode=mode, value=value)

def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    num = (x1 * x2).sum(dim=dim)
    n1 = x1.norm(p=2, dim=dim).clamp(min=eps)
    n2 = x2.norm(p=2, dim=dim).clamp(min=eps)
    return num / (n1 * n2)

def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    diff_t = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    if p == 2.0:
        return (diff_t * diff_t).sum(dim=-1).sqrt()
    if p == 1.0:
        return diff_t.abs().sum(dim=-1)
    if math.isinf(p):
        return diff_t.abs().amax(dim=-1)
    return (diff_t.abs() ** p).sum(dim=-1) ** (1.0 / p)

def pdist(input: Tensor, p: float = 2.0) -> Tensor:
    n = input.shape[0]
    if n < 2:
        return zeros((0,), dtype=input.dtype, device=input.device)
    dmat = cdist(input, input, p=p)
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(i)
            cols.append(j)
    from ._tensor import _from_python
    r_t = _from_python(rows, dtype=_dt.int64, device=input.device)
    c_t = _from_python(cols, dtype=_dt.int64, device=input.device)
    return dmat[r_t, c_t]

def _expand_einsum_sub(sub: str, ndim: int, ell_labels: list[str]) -> str:
    if "..." not in sub:
        return sub
    idx = sub.index("...")
    before = sub[:idx]
    after = sub[idx + 3 :]
    n_ell = ndim - len(before) - len(after)
    return before + "".join(ell_labels[-n_ell:] if n_ell > 0 else []) + after

def _reduce_single_einsum(t: Tensor, sub: str) -> tuple[Tensor, str]:
    # Handle repeated indices on a single operand (trace / diagonal)
    counts = {}
    for ch in sub:
        counts[ch] = counts.get(ch, 0) + 1
    while builtins.any(v > 1 for v in counts.values()):
        for ch, cnt in counts.items():
            if cnt > 1:
                i1 = sub.index(ch)
                i2 = sub.index(ch, i1 + 1)
                t = t.diagonal(offset=0, dim1=i1, dim2=i2)
                # diagonal appends the shared axis at the end
                sub = "".join(c for k, c in enumerate(sub) if k not in (i1, i2)) + ch
                break
        counts = {}
        for ch in sub:
            counts[ch] = counts.get(ch, 0) + 1
    return t, sub

def einsum(equation: str, *operands) -> Tensor:
    """Differentiable general einsum supporting arbitrary contractions and '...'."""
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    eq = equation.replace(" ", "")
    lhs, rhs = (eq.split("->") if "->" in eq else (eq, None))
    in_subs = lhs.split(",")
    if len(in_subs) != len(operands):
        raise ValueError(f"einsum: expected {len(in_subs)} operands, got {len(operands)}")

    used = set(eq.replace(".", "").replace(",", "").replace("->", ""))
    free_chars = [c for c in (string.ascii_uppercase + string.ascii_lowercase) if c not in used]

    max_ell = 0
    for sub, op in zip(in_subs, operands):
        if "..." in sub:
            n_ell = op.ndim - (len(sub) - 3)
            max_ell = builtins.max(max_ell, n_ell)
    ell_labels = free_chars[:max_ell]

    exp_subs = [_expand_einsum_sub(sub, op.ndim, ell_labels) for sub, op in zip(in_subs, operands)]
    if rhs is None:
        all_chars = "".join(exp_subs)
        cnt = {}
        for c in all_chars:
            cnt[c] = cnt.get(c, 0) + 1
        rhs_exp = "".join(sorted(c for c, v in cnt.items() if v == 1))
    else:
        if "..." in rhs:
            rhs_exp = rhs.replace("...", "".join(ell_labels))
        else:
            rhs_exp = rhs

    # Simplify single operands (diagonal/trace)
    ops_clean = []
    subs_clean = []
    for op, sub in zip(operands, exp_subs):
        op2, sub2 = _reduce_single_einsum(op, sub)
        ops_clean.append(op2)
        subs_clean.append(sub2)

    # Pairwise contraction
    cur_t, cur_s = ops_clean[0], subs_clean[0]
    for idx in range(1, len(ops_clean)):
        nxt_t, nxt_s = ops_clean[idx], subs_clean[idx]
        # Sum out labels not needed in rhs_exp or later operands
        later = set(rhs_exp)
        for j in range(idx + 1, len(subs_clean)):
            later.update(subs_clean[j])
        # Contract shared labels between cur_s and nxt_s that are NOT in `later`
        common = [c for c in cur_s if c in nxt_s]
        contract = [c for c in common if c not in later]
        batch = [c for c in common if c in later]
        free_a = [c for c in cur_s if c not in common]
        free_b = [c for c in nxt_s if c not in common]

        # Pre-sum any axes unique to cur_s or nxt_s that are NOT in `later`
        sum_a = [i for i, c in enumerate(cur_s) if c not in nxt_s and c not in later]
        if sum_a:
            for d in sorted(sum_a, reverse=True):
                cur_t = cur_t.sum(dim=d)
            cur_s = "".join(c for i, c in enumerate(cur_s) if i not in sum_a)
            free_a = [c for c in cur_s if c not in common]

        sum_b = [i for i, c in enumerate(nxt_s) if c not in cur_s and c not in later]
        if sum_b:
            for d in sorted(sum_b, reverse=True):
                nxt_t = nxt_t.sum(dim=d)
            nxt_s = "".join(c for i, c in enumerate(nxt_s) if i not in sum_b)
            free_b = [c for c in nxt_s if c not in common]

        perm_a = [cur_s.index(c) for c in (batch + free_a + contract)]
        perm_b = [nxt_s.index(c) for c in (batch + contract + free_b)]
        a_p = cur_t.permute(perm_a)
        b_p = nxt_t.permute(perm_b)

        b_sh = tuple(cur_t.shape[cur_s.index(c)] for c in batch)
        fa_sh = tuple(cur_t.shape[cur_s.index(c)] for c in free_a)
        fb_sh = tuple(nxt_t.shape[nxt_s.index(c)] for c in free_b)
        m = math.prod(fa_sh) if fa_sh else 1
        n = math.prod(fb_sh) if fb_sh else 1
        k = math.prod(cur_t.shape[cur_s.index(c)] for c in contract) if contract else 1

        res = a_p.reshape(b_sh + (m, k)).matmul(b_p.reshape(b_sh + (k, n))).reshape(b_sh + fa_sh + fb_sh)
        cur_t = res
        cur_s = "".join(batch + free_a + free_b)

    # Final sum-out for single-operand or remaining non-rhs indices
    rem_sum = [i for i, c in enumerate(cur_s) if c not in rhs_exp]
    if rem_sum:
        for d in sorted(rem_sum, reverse=True):
            cur_t = cur_t.sum(dim=d)
        cur_s = "".join(c for i, c in enumerate(cur_s) if i not in rem_sum)

    if cur_s != rhs_exp:
        perm = [cur_s.index(c) for c in rhs_exp]
        cur_t = cur_t.permute(perm)
    return cur_t
