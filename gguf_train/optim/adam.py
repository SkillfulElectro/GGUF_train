"""Adam and AdamW optimizers backed by native ggml_opt_step_adamw kernels."""

from __future__ import annotations

from .. import _C, _autograd_state
from .._creation import zeros_like
from .optimizer import Optimizer


class _AdamBase(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        decoupled_wd: bool = False,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            amsgrad=bool(amsgrad),
        )
        super().__init__(params, defaults)
        self._decoupled_wd = decoupled_wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with _autograd_state.enable_grad():
                loss = closure()

        with _autograd_state.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                amsgrad = group["amsgrad"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if len(st) == 0:
                        st["step"] = 0
                        st["exp_avg"] = zeros_like(p)
                        st["exp_avg_sq"] = zeros_like(p)
                        if amsgrad:
                            st["max_exp_avg_sq"] = zeros_like(p)
                    st["step"] += 1
                    _C.check_rc(
                        _C.lib.gt_adamw_step(
                            p._ptr,
                            g._ptr,
                            st["exp_avg"]._ptr,
                            st["exp_avg_sq"]._ptr,
                            st["max_exp_avg_sq"]._ptr if amsgrad else 0,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            wd,
                            int(st["step"]),
                            1 if amsgrad else 0,
                            1 if self._decoupled_wd else 0,
                        ),
                        "adam_step",
                    )
        return loss


class Adam(_AdamBase):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, decoupled_wd=False)


class AdamW(_AdamBase):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2, amsgrad: bool = False) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, decoupled_wd=True)
