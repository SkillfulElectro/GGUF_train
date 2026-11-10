"""Numerical gradient & second-order gradient checkers matching torch.autograd.gradcheck."""

from __future__ import annotations

from .. import _autograd_state
from .._autograd_engine import grad
from .._creation import zeros_like
from .._tensor import Tensor


def _as_tuple(x):
    if isinstance(x, Tensor):
        return (x,)
    return tuple(x)


def _numerical_vjp(func, inputs_tup, grad_outputs, eps: float):
    """Compute central finite-difference Jacobian-vector product sum(g_out * dF/dx_i)."""
    res = []
    for idx, inp in enumerate(inputs_tup):
        if not isinstance(inp, Tensor) or not inp.requires_grad:
            res.append(None)
            continue
        flat_orig = inp.detach().reshape((-1,)).tolist()
        num_grad_flat = [0.0] * len(flat_orig)
        inp_work = inp.detach().clone().requires_grad_(True)
        for j in range(len(flat_orig)):
            old_val = flat_orig[j]

            with _autograd_state.no_grad():
                inp_work_flat = inp_work.reshape((-1,))
                inp_work_flat[j] = old_val + eps
            in_plus = list(inputs_tup)
            in_plus[idx] = inp_work
            out_plus = _as_tuple(func(*in_plus))

            with _autograd_state.no_grad():
                inp_work_flat = inp_work.reshape((-1,))
                inp_work_flat[j] = old_val - eps
            in_minus = list(inputs_tup)
            in_minus[idx] = inp_work
            out_minus = _as_tuple(func(*in_minus))

            with _autograd_state.no_grad():
                inp_work_flat = inp_work.reshape((-1,))
                inp_work_flat[j] = old_val

                acc = 0.0
                for op, om, go in zip(out_plus, out_minus, grad_outputs):
                    if isinstance(op, Tensor) and (op.dtype.is_floating_point or op.dtype.is_complex):
                        diff = (op.detach() - om.detach()) / (2.0 * eps)
                        acc += float((diff * go.detach()).sum().item())
            num_grad_flat[j] = acc

        from .._tensor import _from_python
        ng = _from_python(num_grad_flat, dtype=inp.dtype, device=inp.device).reshape(inp.shape)
        res.append(ng)
    return tuple(res)


def gradcheck(
    func,
    inputs,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
) -> bool:
    (nondet_tol,)
    in_tup = _as_tuple(inputs)
    diff_inps = [x for x in in_tup if isinstance(x, Tensor) and x.requires_grad]
    if not diff_inps:
        return True

    outputs = _as_tuple(func(*in_tup))
    diff_outs = [o for o in outputs if isinstance(o, Tensor) and (o.dtype.is_floating_point or o.dtype.is_complex)]
    grad_outs = []
    from .._creation import randn_like
    with _autograd_state.no_grad():
        for o in diff_outs:
            grad_outs.append(randn_like(o))

    ana_grads = grad(diff_outs, diff_inps, grad_outputs=grad_outs, allow_unused=True)
    num_grads_all = _numerical_vjp(func, in_tup, grad_outs, eps)
    num_grads = [ng for x, ng in zip(in_tup, num_grads_all) if isinstance(x, Tensor) and x.requires_grad]

    for i, (ag, ng, inp) in enumerate(zip(ana_grads, num_grads, diff_inps)):
        if ag is None:
            ag = zeros_like(inp)
        if not ag.allclose(ng, rtol=rtol, atol=atol):
            if raise_exception:
                max_err = float((ag - ng).abs().max().item()) if ag.numel() > 0 else 0.0
                raise RuntimeError(
                    f"Jacobian mismatch for output 0 with respect to input {i}: max abs diff {max_err} "
                    f"(analytical={ag.tolist()}, numerical={ng.tolist()})"
                )
            return False
    return True


def gradgradcheck(
    func,
    inputs,
    grad_outputs=None,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
) -> bool:
    (gen_non_contig_grad_outputs,)
    in_tup = _as_tuple(inputs)
    outputs = _as_tuple(func(*in_tup))
    diff_outs = [o for o in outputs if isinstance(o, Tensor) and (o.dtype.is_floating_point or o.dtype.is_complex)]
    from .._creation import randn_like

    if grad_outputs is None:
        g_outs = tuple(randn_like(o).requires_grad_(True) for o in diff_outs)
    else:
        g_outs = _as_tuple(grad_outputs)

    diff_inps = [x for x in in_tup if isinstance(x, Tensor) and x.requires_grad]

    def _first_order(*args):
        ins = args[: len(in_tup)]
        gos = args[len(in_tup) :]
        outs = _as_tuple(func(*ins))
        d_outs = [o for o in outs if isinstance(o, Tensor) and (o.dtype.is_floating_point or o.dtype.is_complex)]
        d_ins = [x for x in ins if isinstance(x, Tensor) and x.requires_grad]
        gs = grad(d_outs, d_ins, grad_outputs=gos, create_graph=True, allow_unused=True)
        return tuple(g for g in gs if g is not None)

    return gradcheck(_first_order, in_tup + g_outs, eps=eps, atol=atol, rtol=rtol, raise_exception=raise_exception)
