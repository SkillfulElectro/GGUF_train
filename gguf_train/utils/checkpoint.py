"""Activation checkpointing compatible with torch.utils.checkpoint.

``checkpoint(function, *args)`` runs `function` without recording a graph, so the tensors it
creates on the way are freed as soon as it returns. When a gradient is asked for, the function
is run a second time - this time recording - and differentiated. Memory is traded for the
second run.

How it works here:

* The forward run happens under ``no_grad`` with a recorder installed that notes every tensor
  requiring grad that an operation of the function takes in from outside: the arguments, the
  parameters of the modules it calls, tensors it finds in a closure, a global, an attribute.
  Those tensors become the inputs of one graph node standing for the whole function, so a
  gradient reaches all of them however the function got hold of them, with ``backward()`` and
  with ``torch.autograd.grad()`` alike. No input has to require grad.
* The random number generator is put back, for the second run, into the state the first run
  started from (``preserve_rng_state=True``, the default), so dropout draws the same mask
  twice; afterwards it continues where the code around the backward pass left it. Autocast is
  restored in the same way.
* The second run is differentiated on its own: the pass stops at the tensors the function was
  given and never enters the graph around it.

What the function must not do: change one of its inputs in place, or behave differently the
second time (other than through the random numbers, which are replayed). A tensor it computes
and stores somewhere other than in its return value carries no gradient.

``use_reentrant`` is accepted and makes no difference - there is one implementation, with the
abilities of PyTorch's ``use_reentrant=False``. Keyword arguments other than the documented
ones are passed to `function`.
"""

from __future__ import annotations

import contextlib

from .. import _autograd_state, amp
from .._autograd_engine import Context, Function, _Recorder, grad
from .._creation import _rng_state_raw, set_rng_state
from .._tensor import Tensor

__all__ = ["checkpoint", "checkpoint_sequential", "noop_context_fn"]


def noop_context_fn():
    """The default `context_fn`: nothing special around the forward run or the recomputation."""
    return contextlib.nullcontext(), contextlib.nullcontext()


def _map_tensors(obj, fn):
    """`obj` with every tensor inside its tuples, lists and dicts replaced by ``fn(tensor)``,
    visited in a fixed order."""
    if isinstance(obj, Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        items = [_map_tensors(v, fn) for v in obj]
        return type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)     # namedtuple
    if isinstance(obj, list):
        return [_map_tensors(v, fn) for v in obj]
    if isinstance(obj, dict):
        items = [(k, _map_tensors(v, fn)) for k, v in obj.items()]
        try:
            return type(obj)(items)
        except TypeError:
            return dict(items)
    return obj


def _tensors_of(obj) -> list[Tensor]:
    found: list[Tensor] = []

    def keep(t):
        found.append(t)
        return t

    _map_tensors(obj, keep)
    return found


class _Call:
    """One checkpoint() call: what is needed to run the function again."""

    __slots__ = ("function", "args", "kwargs", "rng_state", "autocast", "recompute_context", "check", "count", "positions", "meta", "detached")


class _CheckpointFn(Function):
    """The graph node standing for a checkpointed function. Its inputs are the tensors the
    function took in (see _Recorder), its outputs the tensors the function returned."""

    @staticmethod
    def forward(ctx: Context, call: _Call, *taken_in):
        ctx.call = call
        ctx.taken_in = taken_in
        return call.detached

    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        call: _Call = ctx.call
        taken_in = ctx.taken_in
        # the engine has grad mode on exactly when it is asked for a differentiable gradient
        create_graph = _autograd_state.is_grad_enabled()

        with contextlib.ExitStack() as stack:
            if call.rng_state is not None:
                # replay the random numbers of the forward run, then give the generator back
                # in the state the surrounding code left it in
                stack.callback(set_rng_state, _rng_state_raw())
                set_rng_state(call.rng_state)
            stack.enter_context(_autograd_state.enable_grad())
            stack.enter_context(amp._restored(call.autocast))
            recorder = stack.enter_context(_Recorder())
            stack.enter_context(call.recompute_context)
            result = call.function(*call.args, **call.kwargs)

        mark = recorder.mark
        declared = {id(t) for t in taken_in}
        for t in recorder.found.values():
            # a leaf it did not use before simply gets no gradient from here; a computed tensor
            # must not be passed over, the gradient belongs to the graph it comes from
            if t._grad_fn is not None and id(t) not in declared:
                raise RuntimeError(
                    "torch.utils.checkpoint: when the function was run again for the backward pass it used a "
                    "tensor it had not used in the forward pass. A checkpointed function has to do the same "
                    "thing both times (with preserve_rng_state=False its random numbers differ)."
                )
        tensors = _tensors_of(result)
        if len(tensors) != call.count:
            raise RuntimeError(
                f"torch.utils.checkpoint: the function returned {call.count} tensors in the forward pass and "
                f"{len(tensors)} when it was run again for the backward pass."
            )

        outputs, grads_in = [], []
        for position, meta, g in zip(call.positions, call.meta, grad_outputs):
            out = tensors[position]
            if call.check and (tuple(out.shape), out.dtype) != meta:
                raise RuntimeError(
                    "torch.utils.checkpoint: the function returned a tensor of shape "
                    f"{list(meta[0])} and dtype {meta[1]} in the forward pass and one of shape {list(out.shape)} and "
                    f"dtype {out.dtype} when it was run again for the backward pass."
                )
            if g is None or not out.requires_grad or out._grad_fn is None or out._grad_fn.seq < mark:
                continue            # nothing flows back through this output
            outputs.append(out)
            grads_in.append(g)
        if not outputs:
            return (None,) * (1 + len(taken_in))
        grads = grad(
            outputs,
            list(taken_in),
            grad_outputs=grads_in,
            create_graph=create_graph,
            allow_unused=True,
            _stop_before=mark,
            _leaf_hooks=False,
        )
        return (None,) + tuple(grads)


def checkpoint(function, *args, use_reentrant=None, context_fn=noop_context_fn, determinism_check: str = "default", debug: bool = False, **kwargs):
    """Run ``function(*args, **kwargs)`` without keeping its intermediate tensors; run it again
    when the backward pass gets there.

    Args:
        function: any callable. What it returns may be a tensor or tuples, lists and dicts of
            tensors and other values.
        preserve_rng_state (keyword, default True): replay the random numbers of the forward
            run when the function is recomputed.
        context_fn: a callable returning two context managers, for the forward run and for
            the recomputation.
        determinism_check: "default" compares shape and dtype of what the recomputation
            returns with the forward run; "none" does not.
        use_reentrant, debug: accepted for compatibility; they change nothing.
    """
    (use_reentrant, debug)
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    if determinism_check not in ("default", "none"):
        raise ValueError(f"determinism_check should be one of ['default', 'none'], but got {determinism_check}")
    forward_context, recompute_context = context_fn()

    if not _autograd_state.is_grad_enabled():
        with forward_context:
            return function(*args, **kwargs)

    call = _Call()
    call.function = function
    call.args = args
    call.kwargs = kwargs
    call.rng_state = _rng_state_raw() if preserve_rng_state else None
    call.autocast = amp._snapshot()
    call.recompute_context = recompute_context
    call.check = determinism_check == "default"

    with _autograd_state.no_grad(), _Recorder() as recorder, forward_context:
        result = function(*args, **kwargs)
    taken_in = list(recorder.found.values())
    if not taken_in:
        return result               # nothing a gradient could flow back to

    # The node stands for the tensors the function made. Under no_grad those do not require
    # grad; a returned tensor that does existed before (or is a new leaf) and keeps its own
    # history, like an argument handed back unchanged.
    given = {id(a) for a in args if isinstance(a, Tensor)}
    given.update(id(v) for v in kwargs.values() if isinstance(v, Tensor))
    tensors = _tensors_of(result)
    positions = tuple(
        i
        for i, t in enumerate(tensors)
        if not t.requires_grad and id(t) not in given and (t.dtype.is_floating_point or t.dtype.is_complex)
    )
    if not positions:
        return result
    call.count = len(tensors)
    call.positions = positions
    call.meta = tuple((tuple(tensors[i].shape), tensors[i].dtype) for i in positions)
    call.detached = tuple(tensors[i].detach() for i in positions)
    outputs = _CheckpointFn.apply(call, *taken_in)
    call.detached = None            # the node must not keep its own outputs alive

    replacement = dict(zip(positions, outputs))
    counter = iter(range(len(tensors)))
    return _map_tensors(result, lambda t: replacement.get(next(counter), t))


def checkpoint_sequential(functions, segments: int, input, use_reentrant=None, **kwargs):
    """Run `functions` one after the other on `input`, in `segments` chunks: the first
    ``segments - 1`` chunks hold ``len(functions) // segments`` functions each and are
    checkpointed; the functions that are left form the last chunk, which runs normally.

    With more segments than functions every function is its own chunk.
    """
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(kwargs))
    from ..nn import Sequential

    functions = list(functions.children()) if isinstance(functions, Sequential) else list(functions)
    segments = int(segments)
    if segments < 1:
        raise ValueError(f"segments must be at least 1, got {segments}")

    def run(start: int, end: int):
        def forward(x):
            for fn in functions[start:end]:
                x = fn(x)
            return x

        return forward

    if not functions:
        return input
    segments = min(segments, len(functions))
    size = len(functions) // segments
    end = 0
    for start in range(0, size * (segments - 1), size):
        end = start + size
        input = checkpoint(run(start, end), input, use_reentrant=use_reentrant, preserve_rng_state=preserve_rng_state)
    return run(end, len(functions))(input)
