"""Probability distributions compatible with torch.distributions."""

from __future__ import annotations

import math

from .. import _dtype as _dt
from .._creation import as_tensor, rand, randn
from .._ops import cat, softmax
from .._tensor import Tensor


def _as_dist_tensor(val) -> Tensor:
    return val if isinstance(val, Tensor) else as_tensor(val, dtype=_dt.get_default_dtype())


class Distribution:
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        raise NotImplementedError

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def entropy(self) -> Tensor:
        raise NotImplementedError


class Normal(Distribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc if isinstance(loc, Tensor) else float(loc)
        self.scale = scale if isinstance(scale, Tensor) else float(scale)

    def _shape(self) -> tuple[int, ...]:
        if isinstance(self.loc, Tensor):
            return self.loc.shape
        if isinstance(self.scale, Tensor):
            return self.scale.shape
        return ()

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        base_sh = self._shape()
        sh = tuple(sample_shape) + base_sh
        dt = self.loc.dtype if isinstance(self.loc, Tensor) else (self.scale.dtype if isinstance(self.scale, Tensor) else _dt.float32)
        dev = self.loc.device if isinstance(self.loc, Tensor) else None
        eps = randn(sh, dtype=dt, device=dev)
        return self.loc + eps * self.scale

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self.rsample(sample_shape).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        var = self.scale * self.scale if isinstance(self.scale, Tensor) else (self.scale ** 2)
        log_scale = self.scale.log() if isinstance(self.scale, Tensor) else math.log(self.scale)
        return -((value - self.loc) ** 2) / (2.0 * var) - log_scale - 0.5 * math.log(2.0 * math.pi)

    def entropy(self):
        log_scale = self.scale.log() if isinstance(self.scale, Tensor) else math.log(self.scale)
        return 0.5 + 0.5 * math.log(2.0 * math.pi) + log_scale


class Uniform(Distribution):
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        base_sh = self.low.shape if isinstance(self.low, Tensor) else (self.high.shape if isinstance(self.high, Tensor) else ())
        sh = tuple(sample_shape) + base_sh
        u = rand(sh)
        return self.low + u * (self.high - self.low)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self.rsample(sample_shape).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        lb = (value >= self.low) & (value < self.high)
        span = self.high - self.low
        lp = -(span.log() if isinstance(span, Tensor) else math.log(span))
        from .._ops import where
        return where(lb, value * 0.0 + lp, value * 0.0 - float("inf"))

    def entropy(self):
        span = self.high - self.low
        return span.log() if isinstance(span, Tensor) else math.log(span)


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None) -> None:
        if probs is not None:
            probs = _as_dist_tensor(probs)
            self.probs = probs
            p = probs.clamp(min=1e-7, max=1.0 - 1e-7)
            self.logits = (p / (1.0 - p)).log()
        elif logits is not None:
            logits = _as_dist_tensor(logits)
            self.logits = logits
            self.probs = logits.sigmoid()
        else:
            raise ValueError("Either probs or logits must be specified")

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        sh = tuple(sample_shape) + self.probs.shape
        return (rand(sh, dtype=self.probs.dtype, device=self.probs.device) < self.probs).to(self.probs.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        p = self.probs.clamp(min=1e-12, max=1.0 - 1e-12)
        return value * p.log() + (1.0 - value) * (1.0 - p).log()

    def entropy(self) -> Tensor:
        p = self.probs.clamp(min=1e-12, max=1.0 - 1e-12)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


class Categorical(Distribution):
    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None) -> None:
        if probs is not None:
            self.probs = probs / probs.sum(dim=-1, keepdim=True)
            self.logits = self.probs.clamp(min=1e-12).log()
        elif logits is not None:
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            self.probs = self.logits.exp()
        else:
            raise ValueError("Either probs or logits must be specified")

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        p = self.probs
        flat_p = p.reshape((-1, p.shape[-1]))
        n = 1
        for s in sample_shape:
            n *= s
        idx = flat_p.multinomial(n, replacement=True)
        out_sh = tuple(sample_shape) + p.shape[:-1]
        if n == 1:
            return idx.squeeze(-1).reshape(p.shape[:-1])
        return idx.t().reshape(out_sh)

    def log_prob(self, value: Tensor) -> Tensor:
        idx = value.long().unsqueeze(-1)
        return self.logits.gather(-1, idx).squeeze(-1)

    def entropy(self) -> Tensor:
        return -(self.probs * self.logits).sum(dim=-1)


class Exponential(Distribution):
    def __init__(self, rate) -> None:
        self.rate = rate

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        base_sh = self.rate.shape if isinstance(self.rate, Tensor) else ()
        u = rand(tuple(sample_shape) + base_sh).clamp(min=1e-12)
        return -u.log() / self.rate

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self.rsample(sample_shape).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        lr = self.rate.log() if isinstance(self.rate, Tensor) else math.log(self.rate)
        return lr - self.rate * value


class Laplace(Distribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = scale

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        base_sh = self.loc.shape if isinstance(self.loc, Tensor) else ()
        u = rand(tuple(sample_shape) + base_sh) - 0.5
        return self.loc - self.scale * u.sign() * (1.0 - 2.0 * u.abs()).clamp(min=1e-12).log()

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self.rsample(sample_shape).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        ls = self.scale.log() if isinstance(self.scale, Tensor) else math.log(self.scale)
        return -math.log(2.0) - ls - (value - self.loc).abs() / self.scale


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return 0.5 * (var_ratio + t1 - 1.0 - var_ratio.log())
    if isinstance(p, Categorical) and isinstance(q, Categorical):
        return (p.probs * (p.logits - q.logits)).sum(dim=-1)
    if isinstance(p, Bernoulli) and isinstance(q, Bernoulli):
        return p.probs * (p.probs.log() - q.probs.log()) + (1.0 - p.probs) * ((1.0 - p.probs).log() - (1.0 - q.probs).log())
    raise NotImplementedError(f"kl_divergence not implemented for {type(p)} and {type(q)}")
