"""Linear algebra submodule compatible with torch.linalg."""

from __future__ import annotations

from collections import namedtuple
import ctypes
import math

from . import _C, _dtype as _dt
from ._creation import eye, zeros
from ._functions import LinalgDetFn, LinalgInvFn, LinalgSolveFn, _wrap
from ._ops import matmul
from ._tensor import Tensor, _AUTOCAST_FLOAT32, _AUTOCAST_KEEP, _AUTOCAST_LOWER, _autocast_op

QRResult = namedtuple("QRResult", ["Q", "R"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
LstsqResult = namedtuple("LstsqResult", ["solution", "residuals", "rank", "singular_values"])


# Decompositions and solvers are not autocast: they run in the dtype of their input, and the
# products and sums inside them are not cast either.
@_autocast_op(_AUTOCAST_KEEP)
def inv(A: Tensor, *, out=None) -> Tensor:
    res = LinalgInvFn.apply(A)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def solve(A: Tensor, B: Tensor, *, left: bool = True, out=None) -> Tensor:
    if not left:
        # X A = B -> A^T X^T = B^T
        res = solve(A.mT, B.mT, left=True).mT
    else:
        res = LinalgSolveFn.apply(A, B)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def cholesky(A: Tensor, *, upper: bool = False, out=None) -> Tensor:
    res = _wrap(_C.lib.gt_linalg_cholesky(A._ptr, 1 if upper else 0), "linalg.cholesky")
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def qr(A: Tensor, mode: str = "reduced", *, out=None):
    full = 1 if mode == "complete" else 0
    q_p, r_p = ctypes.c_void_p(), ctypes.c_void_p()
    _C.check_rc(_C.lib.gt_linalg_qr(A._ptr, ctypes.byref(q_p), ctypes.byref(r_p), full), "linalg.qr")
    q, r = _wrap(q_p.value), _wrap(r_p.value)
    if mode == "r":
        return QRResult(zeros((0,), dtype=A.dtype, device=A.device), r)
    if out is not None:
        out[0].copy_(q)
        out[1].copy_(r)
        return QRResult(out[0], out[1])
    return QRResult(q, r)


@_autocast_op(_AUTOCAST_KEEP)
def svd(A: Tensor, full_matrices: bool = True, *, driver=None, out=None):
    (driver,)
    u_p, s_p, vh_p = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    _C.check_rc(
        _C.lib.gt_linalg_svd(A._ptr, 1 if full_matrices else 0, ctypes.byref(u_p), ctypes.byref(s_p), ctypes.byref(vh_p)),
        "linalg.svd",
    )
    U, S, Vh = _wrap(u_p.value), _wrap(s_p.value), _wrap(vh_p.value)
    if out is not None:
        out[0].copy_(U)
        out[1].copy_(S)
        out[2].copy_(Vh)
        return SVDResult(out[0], out[1], out[2])
    return SVDResult(U, S, Vh)


@_autocast_op(_AUTOCAST_KEEP)
def svdvals(A: Tensor, *, driver=None, out=None) -> Tensor:
    _, S, _ = svd(A, full_matrices=False, driver=driver)
    if out is not None:
        out.copy_(S)
        return out
    return S


@_autocast_op(_AUTOCAST_KEEP)
def eigh(A: Tensor, UPLO: str = "L", *, out=None):
    (UPLO,)
    l_p, v_p = ctypes.c_void_p(), ctypes.c_void_p()
    _C.check_rc(_C.lib.gt_linalg_eigh(A._ptr, 1, ctypes.byref(l_p), ctypes.byref(v_p)), "linalg.eigh")
    L, V = _wrap(l_p.value), _wrap(v_p.value)
    if out is not None:
        out[0].copy_(L)
        out[1].copy_(V)
        return EighResult(out[0], out[1])
    return EighResult(L, V)


@_autocast_op(_AUTOCAST_KEEP)
def eigvalsh(A: Tensor, UPLO: str = "L", *, out=None) -> Tensor:
    (UPLO,)
    l_p, v_p = ctypes.c_void_p(), ctypes.c_void_p()
    _C.check_rc(_C.lib.gt_linalg_eigh(A._ptr, 0, ctypes.byref(l_p), ctypes.byref(v_p)), "linalg.eigvalsh")
    L = _wrap(l_p.value)
    if out is not None:
        out.copy_(L)
        return out
    return L


@_autocast_op(_AUTOCAST_KEEP)
def det(A: Tensor, *, out=None) -> Tensor:
    res = LinalgDetFn.apply(A)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def slogdet(A: Tensor, *, out=None):
    s_p, ld_p = ctypes.c_void_p(), ctypes.c_void_p()
    _C.check_rc(_C.lib.gt_linalg_slogdet(A._ptr, ctypes.byref(s_p), ctypes.byref(ld_p)), "linalg.slogdet")
    sign_t, logabsdet_t = _wrap(s_p.value), _wrap(ld_p.value)
    if out is not None:
        out[0].copy_(sign_t)
        out[1].copy_(logabsdet_t)
        return SlogdetResult(out[0], out[1])
    return SlogdetResult(sign_t, logabsdet_t)


@_autocast_op(_AUTOCAST_FLOAT32)
def vector_norm(x: Tensor, ord: float | int = 2, dim=None, keepdim: bool = False, *, dtype=None, out=None) -> Tensor:
    t = x if dtype is None else x.to(dtype)
    if ord == 0:
        res = (t != 0).to(_dt.float32 if not t.is_floating_point() else t.dtype).sum(dim=dim, keepdim=keepdim)
    else:
        res = t.norm(p=ord, dim=dim, keepdim=keepdim)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_FLOAT32)
def matrix_norm(A: Tensor, ord: float | str = "fro", dim: tuple[int, int] = (-2, -1), keepdim: bool = False, *, dtype=None, out=None) -> Tensor:
    t = A if dtype is None else A.to(dtype)
    d0, d1 = dim
    if ord == "fro":
        res = (t * t).sum(dim=dim, keepdim=keepdim).sqrt()
    elif ord == "nuc":
        a_m = t.movedim((d0, d1), (-2, -1))
        s = svdvals(a_m)
        res = s.sum(dim=-1, keepdim=keepdim)
        if keepdim:
            res = res.unsqueeze(-1)
    elif ord == 1:
        res = t.abs().sum(dim=d0, keepdim=keepdim).amax(dim=d1 if keepdim else (d1 - (1 if d1 > d0 else 0)), keepdim=keepdim)
    elif ord == -1:
        res = t.abs().sum(dim=d0, keepdim=keepdim).amin(dim=d1 if keepdim else (d1 - (1 if d1 > d0 else 0)), keepdim=keepdim)
    elif ord == float("inf"):
        res = t.abs().sum(dim=d1, keepdim=keepdim).amax(dim=d0 if keepdim else (d0 - (1 if d0 > d1 else 0)), keepdim=keepdim)
    elif ord == float("-inf"):
        res = t.abs().sum(dim=d1, keepdim=keepdim).amin(dim=d0 if keepdim else (d0 - (1 if d0 > d1 else 0)), keepdim=keepdim)
    elif ord == 2:
        a_m = t.movedim((d0, d1), (-2, -1))
        s = svdvals(a_m)
        res = s[..., 0]
    elif ord == -2:
        a_m = t.movedim((d0, d1), (-2, -1))
        s = svdvals(a_m)
        res = s[..., -1]
    else:
        raise ValueError(f"Unsupported matrix norm ord={ord}")
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_FLOAT32)
def norm(A: Tensor, ord=None, dim=None, keepdim: bool = False, *, dtype=None, out=None) -> Tensor:
    if dim is None and ord is None:
        return vector_norm(A, ord=2, dim=None, keepdim=keepdim, dtype=dtype, out=out)
    if (isinstance(dim, tuple) and len(dim) == 2) or (dim is None and A.ndim == 2):
        return matrix_norm(A, ord="fro" if ord is None else ord, dim=(-2, -1) if dim is None else dim, keepdim=keepdim, dtype=dtype, out=out)
    return vector_norm(A, ord=2 if ord is None else ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


@_autocast_op(_AUTOCAST_KEEP)
def pinv(A: Tensor, *, rcond: float | None = None, hermitian: bool = False, out=None) -> Tensor:
    (hermitian,)
    U, S, Vh = svd(A, full_matrices=False)
    if rcond is None:
        eps = _dt.finfo(A.dtype).eps
        cutoff = eps * max(A.shape[-2], A.shape[-1]) * S.amax(dim=-1, keepdim=True)
    else:
        cutoff = float(rcond) * S.amax(dim=-1, keepdim=True)
    S_inv = (S > cutoff).to(S.dtype) / S.clamp(min=1e-30)
    res = Vh.mT.matmul(S_inv.unsqueeze(-1) * U.mT)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def lstsq(A: Tensor, B: Tensor, rcond=None, *, driver=None):
    (driver,)
    X = pinv(A, rcond=rcond).matmul(B)
    resids = zeros((0,), dtype=A.dtype, device=A.device)
    rank = matrix_rank(A)
    sv = svdvals(A)
    return LstsqResult(X, resids, rank, sv)


@_autocast_op(_AUTOCAST_KEEP)
def matrix_rank(A: Tensor, *, tol=None, hermitian: bool = False, out=None) -> Tensor:
    (hermitian,)
    S = svdvals(A)
    if tol is None:
        eps = _dt.finfo(A.dtype).eps
        cutoff = eps * max(A.shape[-2], A.shape[-1]) * S.amax(dim=-1, keepdim=True)
    else:
        cutoff = float(tol)
    res = (S > cutoff).to(_dt.int64).sum(dim=-1)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_KEEP)
def cond(A: Tensor, p=None, *, out=None) -> Tensor:
    if p in (None, 2):
        S = svdvals(A)
        res = S[..., 0] / S[..., -1]
    else:
        res = matrix_norm(A, ord=p) * matrix_norm(inv(A), ord=p)
    if out is not None:
        out.copy_(res)
        return out
    return res


def matrix_power(A: Tensor, n: int, *, out=None) -> Tensor:
    if n == 0:
        res = eye(A.shape[-1], dtype=A.dtype, device=A.device).expand(A.shape).clone()
    else:
        base = A if n > 0 else inv(A)
        k = abs(n)
        res = eye(A.shape[-1], dtype=A.dtype, device=A.device).expand(A.shape).clone()
        cur = base
        while k > 0:
            if k & 1:
                res = matmul(res, cur)
            k >>= 1
            if k > 0:
                cur = matmul(cur, cur)
    if out is not None:
        out.copy_(res)
        return out
    return res


def cross(input: Tensor, other: Tensor, *, dim: int = -1, out=None) -> Tensor:
    d = dim if dim >= 0 else dim + input.ndim
    a0, a1, a2 = input.unbind(dim=d)
    b0, b1, b2 = other.unbind(dim=d)
    from ._ops import stack
    res = stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], dim=d)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_LOWER)
def vecdot(x: Tensor, y: Tensor, *, dim: int = -1, out=None) -> Tensor:
    res = (x * y).sum(dim=dim)
    if out is not None:
        out.copy_(res)
        return out
    return res


@_autocast_op(_AUTOCAST_LOWER)
def multi_dot(tensors, *, out=None) -> Tensor:
    ts = list(tensors)
    cur = ts[0]
    for nxt in ts[1:]:
        cur = cur.matmul(nxt)
    if out is not None:
        out.copy_(cur)
        return out
    return cur
