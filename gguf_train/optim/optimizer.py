"""Base Optimizer class compatible with torch.optim.Optimizer."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from .._tensor import Tensor


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
            if p in param_to_idx:
                packed_state[param_to_idx[p]] = {
                    k: (v.clone().detach() if isinstance(v, Tensor) else deepcopy(v)) for k, v in st.items()
                }
        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict) -> None:
        idx_to_param = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                idx_to_param[idx] = p
                idx += 1

        for g, saved_g in zip(self.param_groups, state_dict.get("param_groups", [])):
            for k, v in saved_g.items():
                if k != "params":
                    g[k] = v

        self.state.clear()
        for pid, st in state_dict.get("state", {}).items():
            p = idx_to_param.get(pid)
            if p is not None:
                self.state[p] = {
                    k: (v.to(p.device) if isinstance(v, Tensor) else deepcopy(v)) for k, v in st.items()
                }
