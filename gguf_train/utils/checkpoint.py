"""Activation checkpointing compatible with torch.utils.checkpoint."""

from __future__ import annotations

from .. import _autograd_state
from .._autograd_engine import Context, Function, grad
from .._tensor import Tensor


def _collect_captured_tensors(fn, explicit_ids: set[int]) -> list[Tensor]:
    """Find requires_grad Tensors in bound Module/self or function closures for non-reentrant checkpointing."""
    found: list[Tensor] = []
    seen_objs: set[int] = set()
    seen_tensors = set(explicit_ids)

    def _visit(obj, depth: int = 0):
        if obj is None or depth > 6 or id(obj) in seen_objs:
            return
        seen_objs.add(id(obj))
        if isinstance(obj, Tensor):
            if obj.requires_grad and id(obj) not in seen_tensors:
                seen_tensors.add(id(obj))
                found.append(obj)
            return
        if isinstance(obj, (list, tuple, set)):
            for el in obj:
                _visit(el, depth + 1)
            return
        if isinstance(obj, dict):
            for el in obj.values():
                _visit(el, depth + 1)
            return
        if hasattr(obj, "parameters") and callable(obj.parameters):
            for p in obj.parameters():
                if isinstance(p, Tensor) and p.requires_grad and id(p) not in seen_tensors:
                    seen_tensors.add(id(p))
                    found.append(p)
        if callable(obj):
            if getattr(obj, "__self__", None) is not None:
                _visit(obj.__self__, depth + 1)
            closure = getattr(obj, "__closure__", None)
            if closure:
                for cell in closure:
                    try:
                        _visit(cell.cell_contents, depth + 1)
                    except ValueError:
                        pass

    _visit(fn)
    return found


class _CheckpointFn(Function):
    @staticmethod
    def forward(ctx: Context, run_fn, n_explicit: int, *all_tensors_and_args):
        ctx.run_fn = run_fn
        ctx.n_explicit = n_explicit
        args = all_tensors_and_args[:n_explicit]
        captured = all_tensors_and_args[n_explicit:]
        ctx.inputs = args
        ctx.captured = captured
        with _autograd_state.no_grad():
            outputs = run_fn(*args)
        return outputs

    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        detached_inputs = []
        for x in ctx.inputs:
            if isinstance(x, Tensor) and (x.dtype.is_floating_point or x.dtype.is_complex):
                dx = x.detach().requires_grad_(x.requires_grad)
            else:
                dx = x
            detached_inputs.append(dx)

        with _autograd_state.enable_grad():
            outputs = ctx.run_fn(*detached_inputs)
        out_tup = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
        diff_outs = [o for o in out_tup if isinstance(o, Tensor) and o.requires_grad]
        diff_gos = [g for o, g in zip(out_tup, grad_outputs) if isinstance(o, Tensor) and o.requires_grad]
        diff_ins = [x for x in detached_inputs if isinstance(x, Tensor) and x.requires_grad]
        all_targets = diff_ins + list(ctx.captured)
        if not diff_outs or not all_targets:
            return (None, None) + (None,) * (len(ctx.inputs) + len(ctx.captured))
        grads = grad(diff_outs, all_targets, grad_outputs=diff_gos, allow_unused=True)
        g_iter = iter(grads[: len(diff_ins)])
        res = [None, None]
        for x in detached_inputs:
            if isinstance(x, Tensor) and x.requires_grad:
                res.append(next(g_iter))
            else:
                res.append(None)
        res.extend(grads[len(diff_ins) :])
        return tuple(res)


def checkpoint(function, *args, use_reentrant: bool = False, **kwargs):
    (use_reentrant,)
    run_fn = (lambda *a: function(*a, **kwargs)) if kwargs else function
    explicit_ids = {id(a) for a in args if isinstance(a, Tensor)}
    captured = _collect_captured_tensors(function, explicit_ids)
    if kwargs:
        for v in kwargs.values():
            if isinstance(v, Tensor) and v.requires_grad and id(v) not in explicit_ids:
                captured.append(v)
                explicit_ids.add(id(v))
    return _CheckpointFn.apply(run_fn, len(args), *args, *captured)


def checkpoint_sequential(functions, segments: int, input: Tensor, **kwargs):
    mods = list(functions)
    seg_sz = max(1, len(mods) // segments)

    def _run(start: int, end: int):
        def fn(x):
            for m in mods[start:end]:
                x = m(x)
            return x
        return fn

    x = input
    for s in range(0, len(mods), seg_sz):
        x = checkpoint(_run(s, min(s + seg_sz, len(mods))), x, **kwargs)
    return x
