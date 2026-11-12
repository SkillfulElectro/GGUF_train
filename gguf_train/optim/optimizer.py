"""Base Optimizer class compatible with torch.optim.Optimizer."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy

from .. import _autograd_state
from .._tensor import Tensor


def _snapshot(value):
    """A copy of a piece of optimizer state for a state_dict: tensors are cloned, containers
    are rebuilt as dicts, lists and tuples, which torch.save and pickle can store."""
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, deque)):
        return [_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return deepcopy(value)


def _restored(value, param: Tensor):
    """A piece of loaded state as the optimizer's own: copied, tensors on the parameter's device."""
    if isinstance(value, Tensor):
        return value.detach().clone().to(param.device)
    if isinstance(value, dict):
        return {k: _restored(v, param) for k, v in value.items()}
    if isinstance(value, (list, deque)):
        return [_restored(v, param) for v in value]
    if isinstance(value, tuple):
        return tuple(_restored(v, param) for v in value)
    return deepcopy(value)


class Optimizer:
    def __init__(self, params, defaults: dict) -> None:
        self.defaults = dict(defaults)
        self.state: defaultdict[Tensor, dict] = defaultdict(dict)
        self.param_groups: list[dict] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for pg in param_groups:
            self.add_param_group(pg)

    def add_param_group(self, param_group: dict) -> None:
        pg = dict(param_group)
        params = list(pg["params"])
        for p in params:
            if not isinstance(p, Tensor):
                raise TypeError("optimizer can only optimize Tensors")
        pg["params"] = params
        for k, v in self.defaults.items():
            pg.setdefault(k, v)
        self.param_groups.append(pg)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure=None):
        raise NotImplementedError

    @staticmethod
    def _evaluate(closure):
        """The loss a closure returns, or None without a closure.

        A closure calls backward(); it is run with gradient recording switched on, whatever the
        mode step() was called in.
        """
        if closure is None:
            return None
        with _autograd_state.enable_grad():
            return closure()

    def state_dict(self) -> dict:
        param_to_idx = {}
        packed_groups = []
        idx = 0
        for group in self.param_groups:
            packed = {k: v for k, v in group.items() if k != "params"}
            p_idxs = []
            for p in group["params"]:
                if p not in param_to_idx:
                    param_to_idx[p] = idx
                    idx += 1
                p_idxs.append(param_to_idx[p])
            packed["params"] = p_idxs
            packed_groups.append(packed)

        packed_state = {}
        for p, st in self.state.items():
            if isinstance(p, Tensor) and p in param_to_idx:
                packed_state[param_to_idx[p]] = _snapshot(st)
        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict) -> None:
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state dict has {len(saved_groups)} parameter groups, the optimizer has {len(self.param_groups)}"
            )
        # the indices state_dict() hands out: one per distinct parameter, in order of appearance
        idx_to_param = {}
        seen = {}
        for group, saved_g in zip(self.param_groups, saved_groups):
            if "params" in saved_g and len(saved_g["params"]) != len(group["params"]):
                raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")
            for p in group["params"]:
                if id(p) not in seen:
                    seen[id(p)] = len(seen)
                    idx_to_param[seen[id(p)]] = p

        for g, saved_g in zip(self.param_groups, saved_groups):
            for k, v in saved_g.items():
                if k != "params":
                    g[k] = deepcopy(v)

        self.state.clear()
        for pid, st in state_dict.get("state", {}).items():
            p = idx_to_param.get(pid)
            if p is not None:
                self.state[p] = _restored(st, p)
