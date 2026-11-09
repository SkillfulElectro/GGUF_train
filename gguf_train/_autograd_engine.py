"""Dynamic reverse-mode autograd engine with higher-order (create_graph=True) support."""

from __future__ import annotations

import weakref

from . import _autograd_state


class Context:
    """Context object passed to Function.forward and Function.backward."""

    def __init__(self):
        self._saved_tensors: tuple = ()
        self.needs_input_grad: tuple[bool, ...] = ()
        self._dirty: set[int] = set()
        self._non_diff: set[int] = set()

    def save_for_backward(self, *tensors) -> None:
        self._saved_tensors = tuple(tensors)

    @property
    def saved_tensors(self) -> tuple:
        return self._saved_tensors

    def mark_dirty(self, *args) -> None:
        for a in args:
            self._dirty.add(id(a))

    def mark_non_differentiable(self, *args) -> None:
        for a in args:
            self._non_diff.add(id(a))


class GradNode:
    """Node in the dynamic backward graph."""

    __slots__ = ("fn_cls", "ctx", "next_edges", "input_requires_grad", "num_outputs", "output_refs", "name", "__weakref__")

    def __init__(
        self,
        fn_cls,
        ctx: Context,
        next_edges: list[tuple[object, int]],
        input_requires_grad: list[bool],
        num_outputs: int = 1,
    ):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.next_edges = next_edges  # list of (Tensor_or_None, output_nr)
        self.input_requires_grad = input_requires_grad
        self.num_outputs = num_outputs
        self.output_refs: list = [None] * num_outputs
        self.name = f"{fn_cls.__name__}Backward0"

    @property
    def next_functions(self):
        return tuple((t.grad_fn if t is not None else None, slot) for t, slot in self.next_edges)

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x}>"


class Function:
    """Base class for custom and built-in differentiable operations (matches torch.autograd.Function)."""

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        from ._tensor import Tensor

        ctx = Context()
        raw_inputs_flags: list[bool] = []
        for a in args:
            if isinstance(a, Tensor):
                raw_inputs_flags.append(bool(a.requires_grad))
            else:
                raw_inputs_flags.append(False)

        ctx.needs_input_grad = tuple(raw_inputs_flags)
        any_requires_grad = _autograd_state.is_grad_enabled() and any(raw_inputs_flags)

        with _autograd_state.no_grad():
            outputs = cls.forward(ctx, *args, **kwargs)

        single = not isinstance(outputs, tuple)
        out_tuple = (outputs,) if single else outputs

        if any_requires_grad:
            next_edges = []
            for a in args:
                if isinstance(a, Tensor) and a.requires_grad:
                    next_edges.append((a, getattr(a, "_output_nr", 0)))
                else:
                    next_edges.append((None, 0))
            node = GradNode(cls, ctx, next_edges, raw_inputs_flags, num_outputs=len(out_tuple))
            for idx, out in enumerate(out_tuple):
                if isinstance(out, Tensor) and id(out) not in ctx._non_diff:
                    if out.dtype.is_floating_point or out.dtype.is_complex:
                        out._requires_grad = True
                        out._grad_fn = node
                        out._output_nr = idx
                        out._is_leaf = False
                        node.output_refs[idx] = weakref.ref(out)

        return out_tuple[0] if single else out_tuple


def _unbroadcast(grad, target_shape: tuple[int, ...]):
    """Sum out broadcasted dimensions differentiably (supports create_graph=True)."""
    if grad is None:
        return None
    if grad.shape == target_shape:
        return grad
    g = grad
    while len(g.shape) > len(target_shape):
        g = g.sum(dim=0)
    reduce_dims = [i for i, (gd, td) in enumerate(zip(g.shape, target_shape)) if td == 1 and gd != 1]
    if reduce_dims:
        for d in sorted(reduce_dims, reverse=True):
            g = g.sum(dim=d, keepdim=True)
    if g.shape != target_shape:
        g = g.reshape(target_shape)
    return g


def _accumulate(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _run_autograd(
    outputs,
    grad_outputs,
    inputs,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool,
    materialize_grads: bool,
):
    from ._tensor import Tensor
    from ._creation import ones_like, zeros_like

    out_list = [outputs] if isinstance(outputs, Tensor) else list(outputs)
    if grad_outputs is None:
        g_list = []
        for o in out_list:
            if o.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            g_list.append(ones_like(o, requires_grad=create_graph))
    else:
        g_list = [grad_outputs] if isinstance(grad_outputs, Tensor) else list(grad_outputs)
        if len(g_list) != len(out_list):
            raise RuntimeError("Mismatch in number of outputs and grad_outputs")
        g_list = [
            (ones_like(o, requires_grad=create_graph) if g is None else g)
            for o, g in zip(out_list, g_list)
        ]

    visited: set[int] = set()
    topo: list[GradNode] = []

    def _dfs(node: GradNode | None):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        for inp_t, _ in node.next_edges:
            if inp_t is not None and inp_t._grad_fn is not None:
                _dfs(inp_t._grad_fn)
        topo.append(node)

    for o in out_list:
        if o._grad_fn is not None:
            _dfs(o._grad_fn)
        elif not o.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")

    node_grads: dict[int, list] = {}
    leaf_grads: dict[int, Tensor] = {}
    tensor_by_id: dict[int, Tensor] = {}

    for o, g in zip(out_list, g_list):
        if o._grad_fn is not None:
            nid = id(o._grad_fn)
            if nid not in node_grads:
                node_grads[nid] = [None] * o._grad_fn.num_outputs
            slot = getattr(o, "_output_nr", 0)
            node_grads[nid][slot] = _accumulate(node_grads[nid][slot], g)
        else:
            tid = id(o)
            tensor_by_id[tid] = o
            leaf_grads[tid] = _accumulate(leaf_grads.get(tid), g)

    grad_ctx = _autograd_state.enable_grad() if create_graph else _autograd_state.no_grad()
    with grad_ctx:
        for node in reversed(topo):
            nid = id(node)
            slots = node_grads.get(nid)
            if slots is None or all(s is None for s in slots):
                continue

            for idx, ref in enumerate(node.output_refs):
                if ref is not None and slots[idx] is not None:
                    out_t = ref()
                    if out_t is not None and out_t._hooks:
                        g_slot = slots[idx]
                        for hk in list(out_t._hooks.values()):
                            res = hk(g_slot)
                            if res is not None:
                                g_slot = res
                        slots[idx] = g_slot

            raw_grads = node.fn_cls.backward(node.ctx, *slots)
            if not isinstance(raw_grads, tuple):
                raw_grads = (raw_grads,)

            if len(raw_grads) < len(node.next_edges):
                raw_grads = raw_grads + (None,) * (len(node.next_edges) - len(raw_grads))

            for (inp_t, out_nr), g_in in zip(node.next_edges, raw_grads):
                if inp_t is None or g_in is None:
                    continue
                if inp_t.dtype != g_in.dtype and (inp_t.dtype.is_floating_point or inp_t.dtype.is_complex):
                    g_in = g_in.to(inp_t.dtype)

                if inp_t._grad_fn is not None:
                    pnid = id(inp_t._grad_fn)
                    if pnid not in node_grads:
                        node_grads[pnid] = [None] * inp_t._grad_fn.num_outputs
                    node_grads[pnid][out_nr] = _accumulate(node_grads[pnid][out_nr], g_in)
                    if inp_t._retains_grad:
                        tid = id(inp_t)
                        tensor_by_id[tid] = inp_t
                        leaf_grads[tid] = _accumulate(leaf_grads.get(tid), g_in)
                elif inp_t.requires_grad:
                    tid = id(inp_t)
                    tensor_by_id[tid] = inp_t
                    leaf_grads[tid] = _accumulate(leaf_grads.get(tid), g_in)

            if not retain_graph and not create_graph:
                node.ctx._saved_tensors = ()

        # Apply leaf tensor hooks once on the fully-accumulated gradient
        for tid, g in list(leaf_grads.items()):
            t = tensor_by_id[tid]
            if t._hooks and t._grad_fn is None:
                for hk in list(t._hooks.values()):
                    res = hk(g)
                    if res is not None:
                        g = res
                leaf_grads[tid] = g

    if inputs is not None:
        in_list = [inputs] if isinstance(inputs, Tensor) else list(inputs)
        results = []
        for inp in in_list:
            g = leaf_grads.get(id(inp))
            if g is None:
                if not allow_unused:
                    raise RuntimeError(
                        "One of the differentiated Tensors appears to not have been used in the graph. "
                        "Set allow_unused=True if this is the desired behavior."
                    )
                if materialize_grads:
                    g = zeros_like(inp)
            results.append(g)
        return tuple(results)

    for tid, g in leaf_grads.items():
        t = tensor_by_id[tid]
        if t._is_leaf or t._retains_grad:
            g_store = g if create_graph else g.detach()
            if t.grad is None:
                t.grad = g_store
            else:
                with _autograd_state.no_grad():
                    t.grad = t.grad + g_store
    return None


def backward(
    tensors,
    grad_tensors=None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    inputs=None,
) -> None:
    if retain_graph is None:
        retain_graph = create_graph
    _run_autograd(
        outputs=tensors,
        grad_outputs=grad_tensors,
        inputs=inputs,
        retain_graph=bool(retain_graph),
        create_graph=bool(create_graph),
        allow_unused=True,
        materialize_grads=False,
    )


def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
    materialize_grads: bool = False,
) -> tuple:
    (only_inputs,)
    if retain_graph is None:
        retain_graph = create_graph
    return _run_autograd(
        outputs=outputs,
        grad_outputs=grad_outputs,
        inputs=inputs,
        retain_graph=bool(retain_graph),
        create_graph=bool(create_graph),
        allow_unused=bool(allow_unused),
        materialize_grads=bool(materialize_grads),
    )
