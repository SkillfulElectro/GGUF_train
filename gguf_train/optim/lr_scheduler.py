"""Learning rate schedulers compatible with torch.optim.lr_scheduler."""

from __future__ import annotations

import math

from .optimizer import Optimizer


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        if last_epoch == -1:
            for g in optimizer.param_groups:
                g.setdefault("initial_lr", g["lr"])
        self.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._last_lr = [g["lr"] for g in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        values = self.get_lr()
        for g, lr in zip(self.optimizer.param_groups, values):
            g["lr"] = lr
        self._last_lr = list(values)

    def state_dict(self) -> dict:
        return {"base_lrs": list(self.base_lrs), "last_epoch": self.last_epoch, "_last_lr": list(self._last_lr)}

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))
        self._last_lr = list(state_dict.get("_last_lr", self._last_lr))


_LRScheduler = LRScheduler


class LambdaLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1) -> None:
        self.lr_lambdas = lr_lambda if isinstance(lr_lambda, (list, tuple)) else [lr_lambda] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [base * fn(self.last_epoch) for base, fn in zip(self.base_lrs, self.lr_lambdas)]


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        exp = self.last_epoch // self.step_size
        return [base * (self.gamma ** exp) for base in self.base_lrs]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: list[int], gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.milestones = sorted(milestones)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        cnt = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [base * (self.gamma ** cnt) for base in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [base * (self.gamma ** self.last_epoch) for base in self.base_lrs]


class LinearLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, start_factor: float = 1.0 / 3.0, end_factor: float = 1.0, total_iters: int = 5, last_epoch: int = -1) -> None:
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        self.total_iters = int(total_iters)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        t = min(self.last_epoch, self.total_iters)
        f = self.start_factor + (self.end_factor - self.start_factor) * (t / max(self.total_iters, 1))
        return [base * f for base in self.base_lrs]


class ConstantLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, factor: float = 1.0 / 3.0, total_iters: int = 5, last_epoch: int = -1) -> None:
        self.factor = float(factor)
        self.total_iters = int(total_iters)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        f = self.factor if self.last_epoch < self.total_iters else 1.0
        return [base * f for base in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1) -> None:
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        t = min(self.last_epoch, self.T_max)
        return [
            self.eta_min + (base - self.eta_min) * (1.0 + math.cos(math.pi * t / max(self.T_max, 1))) * 0.5
            for base in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0.0, last_epoch: int = -1) -> None:
        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        T_cur = epoch
        T_i = self.T_0
        if self.T_mult == 1:
            T_cur = epoch % T_i
        else:
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= self.T_mult
        return [
            self.eta_min + (base - self.eta_min) * (1.0 + math.cos(math.pi * T_cur / max(T_i, 1))) * 0.5
            for base in self.base_lrs
        ]


class OneCycleLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_lr: float, total_steps: int, pct_start: float = 0.3, div_factor: float = 25.0, final_div_factor: float = 1e4, last_epoch: int = -1) -> None:
        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = min(self.last_epoch, self.total_steps)
        up_steps = max(1, int(self.total_steps * self.pct_start))
        init_lr = self.max_lr / self.div_factor
        min_lr = init_lr / self.final_div_factor
        if step <= up_steps:
            pct = step / up_steps
            lr = init_lr + (self.max_lr - init_lr) * 0.5 * (1.0 - math.cos(math.pi * pct))
        else:
            pct = (step - up_steps) / max(1, self.total_steps - up_steps)
            lr = min_lr + (self.max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * pct))
        return [lr for _ in self.base_lrs]


class ReduceLROnPlateau:
    def __init__(self, optimizer: Optimizer, mode: str = "min", factor: float = 0.1, patience: int = 10, threshold: float = 1e-4, min_lr: float = 0.0) -> None:
        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0

    def step(self, metrics: float) -> None:
        cur = float(metrics)
        improved = (cur < self.best - self.threshold) if self.mode == "min" else (cur > self.best + self.threshold)
        if improved:
            self.best = cur
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                for g in self.optimizer.param_groups:
                    g["lr"] = max(self.min_lr, g["lr"] * self.factor)
                self.num_bad_epochs = 0

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


class SequentialLR:
    def __init__(self, optimizer: Optimizer, schedulers: list, milestones: list[int]) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1
        idx = 0
        for m in self.milestones:
            if self.last_epoch >= m:
                idx += 1
        idx = min(idx, len(self.schedulers) - 1)
        self.schedulers[idx].step()

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]
