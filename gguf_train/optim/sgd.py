"""SGD optimizer backed by native ggml_opt_step_sgd kernels."""

from __future__ import annotations

from .. import _C, _autograd_state
from .._creation import zeros_like
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            dampening=float(dampening),
            weight_decay=float(weight_decay),
            nesterov=bool(nesterov),
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with _autograd_state.enable_grad():
                loss = closure()

        with _autograd_state.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                mom = group["momentum"]
                damp = group["dampening"]
                wd = group["weight_decay"]
                nest = group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    first_step = 0
                    if mom != 0.0 and "momentum_buffer" not in st:
                        st["momentum_buffer"] = zeros_like(p)
                        first_step = 1
                    buf = st.get("momentum_buffer", None)
                    _C.check_rc(
                        _C.lib.gt_sgd_step(
                            p._ptr,
                            g._ptr,
                            buf._ptr if buf is not None else 0,
                            lr,
                            mom,
                            damp,
                            wd,
                            1 if nest else 0,
                            first_step,
                        ),
                        "sgd_step",
                    )
        return loss
