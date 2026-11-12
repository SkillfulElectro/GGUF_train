"""Additional optimizers (RMSprop, Adagrad, Adadelta, Adamax, NAdam, RAdam)."""

from __future__ import annotations

import math

from .. import _autograd_state
from .._creation import zeros_like
from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False) -> None:
        super().__init__(params, dict(lr=float(lr), alpha=float(alpha), eps=float(eps), weight_decay=float(weight_decay), momentum=float(momentum), centered=bool(centered)))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with _autograd_state.no_grad():
            for group in self.param_groups:
                lr, alpha, eps, wd, mom, centered = group["lr"], group["alpha"], group["eps"], group["weight_decay"], group["momentum"], group["centered"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if wd != 0.0:
                        g = g + wd * p
                    st = self.state[p]
                    if len(st) == 0:
                        st["square_avg"] = zeros_like(p)
                        if mom > 0.0:
                            st["momentum_buffer"] = zeros_like(p)
                        if centered:
                            st["grad_avg"] = zeros_like(p)
                    sq = st["square_avg"]
                    sq.mul_(alpha).add_(g * g, alpha=1.0 - alpha)
                    if centered:
                        ga = st["grad_avg"]
                        ga.mul_(alpha).add_(g, alpha=1.0 - alpha)
                        avg = (sq - ga * ga).clamp(min=0.0).sqrt().add_(eps)
                    else:
                        avg = sq.sqrt().add_(eps)
                    if mom > 0.0:
                        buf = st["momentum_buffer"]
                        buf.mul_(mom).add_(g / avg)
                        p.add_(buf, alpha=-lr)
                    else:
                        p.add_(g / avg, alpha=-lr)
        return loss


class Adagrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0.0, weight_decay: float = 0.0, eps: float = 1e-10) -> None:
        super().__init__(params, dict(lr=float(lr), lr_decay=float(lr_decay), weight_decay=float(weight_decay), eps=float(eps)))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with _autograd_state.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if group["weight_decay"] != 0.0:
                        g = g + group["weight_decay"] * p
                    st = self.state[p]
                    if len(st) == 0:
                        st["step"] = 0
                        st["sum"] = zeros_like(p)
                    st["step"] += 1
                    clr = group["lr"] / (1.0 + (st["step"] - 1) * group["lr_decay"])
                    st["sum"].add_(g * g)
                    p.add_(g / (st["sum"].sqrt() + group["eps"]), alpha=-clr)
        return loss


class Adadelta(Optimizer):
    def __init__(self, params, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=float(lr), rho=float(rho), eps=float(eps), weight_decay=float(weight_decay)))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with _autograd_state.no_grad():
            for group in self.param_groups:
                rho, eps, lr, wd = group["rho"], group["eps"], group["lr"], group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad + wd * p if wd != 0.0 else p.grad
                    st = self.state[p]
                    if len(st) == 0:
                        st["square_avg"] = zeros_like(p)
                        st["acc_delta"] = zeros_like(p)
                    sq, acc = st["square_avg"], st["acc_delta"]
                    sq.mul_(rho).add_(g * g, alpha=1.0 - rho)
                    std = (acc + eps).sqrt()
                    delta = (std / (sq + eps).sqrt()) * g
                    acc.mul_(rho).add_(delta * delta, alpha=1.0 - rho)
                    p.add_(delta, alpha=-lr)
        return loss


class Adamax(Optimizer):
    def __init__(self, params, lr: float = 2e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=float(lr), betas=betas, eps=float(eps), weight_decay=float(weight_decay)))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with _autograd_state.no_grad():
            for group in self.param_groups:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad + group["weight_decay"] * p if group["weight_decay"] != 0.0 else p.grad
                    st = self.state[p]
                    if len(st) == 0:
                        st["step"] = 0
                        st["exp_avg"] = zeros_like(p)
                        st["exp_inf"] = zeros_like(p)
                    st["step"] += 1
                    st["exp_avg"].mul_(b1).add_(g, alpha=1.0 - b1)
                    st["exp_inf"].copy_((st["exp_inf"] * b2).maximum(g.abs()))
                    bc = 1.0 - b1 ** st["step"]
                    p.add_((st["exp_avg"] / (st["exp_inf"] + group["eps"])), alpha=-group["lr"] / bc)
        return loss


class NAdam(Optimizer):
    def __init__(self, params, lr: float = 2e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=float(lr), betas=betas, eps=float(eps), weight_decay=float(weight_decay)))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with _autograd_state.no_grad():
            for group in self.param_groups:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad + group["weight_decay"] * p if group["weight_decay"] != 0.0 else p.grad
                    st = self.state[p]
                    if len(st) == 0:
                        st["step"] = 0
                        st["exp_avg"] = zeros_like(p)
                        st["exp_avg_sq"] = zeros_like(p)
                    st["step"] += 1
                    t = st["step"]
                    st["exp_avg"].mul_(b1).add_(g, alpha=1.0 - b1)
                    st["exp_avg_sq"].mul_(b2).add_(g * g, alpha=1.0 - b2)
                    m_hat = st["exp_avg"] / (1.0 - b1 ** t)
                    v_hat = st["exp_avg_sq"] / (1.0 - b2 ** t)
                    m_nesterov = b1 * m_hat + (1.0 - b1) * g / (1.0 - b1 ** t)
                    p.add_(m_nesterov / (v_hat.sqrt() + group["eps"]), alpha=-group["lr"])
        return loss


class RAdam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=float(lr), betas=betas, eps=float(eps), weight_decay=float(weight_decay)))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with _autograd_state.no_grad():
            for group in self.param_groups:
                b1, b2 = group["betas"]
                rho_inf = 2.0 / (1.0 - b2) - 1.0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad + group["weight_decay"] * p if group["weight_decay"] != 0.0 else p.grad
                    st = self.state[p]
                    if len(st) == 0:
                        st["step"] = 0
                        st["exp_avg"] = zeros_like(p)
                        st["exp_avg_sq"] = zeros_like(p)
                    st["step"] += 1
                    t = st["step"]
                    st["exp_avg"].mul_(b1).add_(g, alpha=1.0 - b1)
                    st["exp_avg_sq"].mul_(b2).add_(g * g, alpha=1.0 - b2)
                    m_hat = st["exp_avg"] / (1.0 - b1 ** t)
                    b2_t = b2 ** t
                    rho_t = rho_inf - 2.0 * t * b2_t / (1.0 - b2_t)
                    if rho_t > 5.0:
                        r_t = math.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                        v_hat = (st["exp_avg_sq"] / (1.0 - b2_t)).sqrt() + group["eps"]
                        p.add_(m_hat / v_hat, alpha=-group["lr"] * r_t)
                    else:
                        p.add_(m_hat, alpha=-group["lr"])
        return loss
