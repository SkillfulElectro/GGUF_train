"""Functional autograd operators (jacobian, hessian, vjp, jvp, vhp, hvp)."""

from __future__ import annotations

from .._autograd_engine import grad
from .._creation import zeros_like
from .._ops import cat, stack
from .._tensor import Tensor


def _as_tuple(x):
    return (x,) if isinstance(x, Tensor) else tuple(x)


def vjp(func, inputs, v=None, create_graph: bool = False, strict: bool = False):
    (strict,)
    single = isinstance(inputs, Tensor)
    in_tup = tuple(x.detach().clone().requires_grad_(True) for x in _as_tuple(inputs))
    out = func(*in_tup) if not single else func(in_tup[0])
    out_tup = _as_tuple(out)
    if v is None:
        if len(out_tup) == 1 and out_tup[0].numel() == 1:
            v_tup = None
        else:
            raise RuntimeError("v must be provided for non-scalar outputs")
    else:
        v_tup = _as_tuple(v)
    gs = grad(out_tup, in_tup, grad_outputs=v_tup, create_graph=create_graph, allow_unused=True)
    gs = tuple(zeros_like(x) if g is None else g for x, g in zip(in_tup, gs))
    return out, (gs[0] if single else gs)


def jvp(func, inputs, v=None, create_graph: bool = False, strict: bool = False):
    """Forward-mode JVP via double-backward trick (requires create_graph=True support)."""
    (strict,)
    single = isinstance(inputs, Tensor)
    in_tup = tuple(x.detach().clone().requires_grad_(True) for x in _as_tuple(inputs))
    v_tup = _as_tuple(v) if v is not None else tuple(zeros_like(x) + 1.0 for x in in_tup)
    out = func(*in_tup) if not single else func(in_tup[0])
    out_tup = _as_tuple(out)
    # Double-backward trick: compute dummy u, vjp = grad(out, x, u, create_graph=True), jvp = grad(vjp, u, v)
    us = tuple(zeros_like(o, requires_grad=True) for o in out_tup)
    vjps = grad(out_tup, in_tup, grad_outputs=us, create_graph=True, allow_unused=True)
    valid_vjps = [g for g in vjps if g is not None]
    valid_vs = [vs for g, vs in zip(vjps, v_tup) if g is not None]
    if not valid_vjps:
        jvps = tuple(zeros_like(o) for o in out_tup)
    else:
        jvps = grad(valid_vjps, us, grad_outputs=valid_vs, create_graph=create_graph, allow_unused=True)
        jvps = tuple(zeros_like(o) if j is None else j for o, j in zip(out_tup, jvps))
    return out, (jvps[0] if isinstance(out, Tensor) else jvps)


def jacobian(func, inputs, create_graph: bool = False, strict: bool = False, vectorize: bool = False):
    (strict, vectorize)
    single_in = isinstance(inputs, Tensor)
    in_tup = _as_tuple(inputs)
    in_vars = tuple(x.detach().clone().requires_grad_(True) for x in in_tup)
    out = func(in_vars[0]) if single_in else func(*in_vars)
    single_out = isinstance(out, Tensor)
    out_tup = _as_tuple(out)

    jacs_per_out = []
    for o in out_tup:
        n_out = o.numel()
        o_flat = o.reshape((-1,))
        rows_per_in = [[] for _ in in_vars]
        for i in range(n_out):
            gs = grad(o_flat[i], in_vars, retain_graph=True, create_graph=create_graph, allow_unused=True)
            for j, (x, g) in enumerate(zip(in_vars, gs)):
                rows_per_in[j].append((zeros_like(x) if g is None else g).reshape((1, *x.shape)))
        in_jacs = tuple(cat(rows, dim=0).reshape((*o.shape, *x.shape)) for x, rows in zip(in_vars, rows_per_in))
        jacs_per_out.append(in_jacs[0] if single_in else in_jacs)
    return jacs_per_out[0] if single_out else tuple(jacs_per_out)


def hessian(func, inputs, create_graph: bool = False, strict: bool = False, vectorize: bool = False):
    (strict, vectorize)
    single = isinstance(inputs, Tensor)

    def _grad_fn(*xs):
        in_vars = tuple(x for x in xs)
        y = func(in_vars[0]) if single else func(*in_vars)
        gs = grad(y, in_vars, create_graph=True, allow_unused=True)
        gs = tuple(zeros_like(x) if g is None else g for x, g in zip(in_vars, gs))
        return gs[0] if single else gs

    return jacobian(_grad_fn, inputs, create_graph=create_graph)


def hvp(func, inputs, v=None, create_graph: bool = False, strict: bool = False):
    return vhp(func, inputs, v=v, create_graph=create_graph, strict=strict)


def vhp(func, inputs, v=None, create_graph: bool = False, strict: bool = False):
    (strict,)
    single = isinstance(inputs, Tensor)
    in_tup = tuple(x.detach().clone().requires_grad_(True) for x in _as_tuple(inputs))
    v_tup = _as_tuple(v) if v is not None else tuple(zeros_like(x) + 1.0 for x in in_tup)
    y = func(in_tup[0]) if single else func(*in_tup)
    gs = grad(y, in_tup, create_graph=True, allow_unused=True)
    valid = [(g, vi) for g, vi in zip(gs, v_tup) if g is not None]
    if not valid:
        hs = tuple(zeros_like(x) for x in in_tup)
    else:
        hs = grad([g for g, _ in valid], in_tup, grad_outputs=[vi for _, vi in valid], create_graph=create_graph, allow_unused=True)
        hs = tuple(zeros_like(x) if h is None else h for x, h in zip(in_tup, hs))
    return y, (hs[0] if single else hs)
