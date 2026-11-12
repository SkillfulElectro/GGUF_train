"""Probability distributions compatible with torch.distributions.

Every distribution holds its parameters as tensors broadcast against each other; their common
shape is `batch_shape`. The distributions here are over single numbers, so `event_shape` is
empty, and a sample has the shape ``sample_shape + batch_shape``.

The formulas are PyTorch's, including its treatment of probabilities at the ends of [0, 1]:
logits are taken from probabilities clamped to [eps, 1 - eps] (eps of the dtype), so every log
is finite, and `kl_divergence` then applies 0 log 0 = 0 and x log(x / 0) = inf where a
probability is exactly 0 or 1.

`validate_args` is accepted for compatibility and ignored: parameters and values are not
checked (PyTorch checks them unless it is told not to).
"""

from __future__ import annotations

import math

from .. import _dtype as _dt
from .._autograd_state import no_grad
from .._creation import as_tensor, rand, randn
from .._device import Size
from .._ops import broadcast_tensors, where
from .._tensor import Tensor, _is_int

__all__ = [
    "Bernoulli",
    "Categorical",
    "Distribution",
    "Exponential",
    "Laplace",
    "Normal",
    "Uniform",
    "kl_divergence",
]


def _broadcast_all(*values) -> tuple[Tensor, ...]:
    """The parameters as tensors of one shape. Numbers (and lists, arrays) take the dtype and
    device of the first tensor among the parameters - the default dtype when there is none."""
    like = next((v for v in values if isinstance(v, Tensor)), None)
    if like is None:
        dtype, device = _dt.get_default_dtype(), None
    else:
        dtype, device = like.dtype, like.device
        if not dtype.is_floating_point:
            dtype = _dt.get_default_dtype()
    tensors = []
    for v in values:
        if not isinstance(v, Tensor):
            v = as_tensor(v, dtype=dtype, device=device)
        elif not v.dtype.is_floating_point:
            v = v.to(dtype)
        tensors.append(v)
    return broadcast_tensors(*tensors)


def _sample_shape(sample_shape) -> tuple[int, ...]:
    return (int(sample_shape),) if _is_int(sample_shape) else tuple(int(s) for s in sample_shape)


def _clamp_probs(probs: Tensor) -> Tensor:
    eps = _dt.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1.0 - eps)


def _softplus(x: Tensor) -> Tensor:
    """log(1 + exp(x)) without overflow."""
    return x.clamp(min=0.0) + (-x.abs()).exp().log1p()


def _bce_with_logits(logits: Tensor, target: Tensor) -> Tensor:
    """-[target * log(sigmoid(logits)) + (1 - target) * log(1 - sigmoid(logits))], elementwise."""
    return logits.clamp(min=0.0) - logits * target + (-logits.abs()).exp().log1p()


class Distribution:
    """Base class: `batch_shape`, `event_shape`, sample() built on rsample()."""

    has_rsample = False
    _parameters: tuple[str, ...] = ()
    # defaults for subclasses that set up their state without calling __init__
    _batch_shape = Size(())
    _event_shape = Size(())
    _validate_args = None

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None) -> None:
        self._batch_shape = Size(batch_shape)
        self._event_shape = Size(event_shape)
        self._validate_args = validate_args

    @property
    def batch_shape(self) -> Size:
        """The shape the parameters are broadcast to: one distribution per entry."""
        return self._batch_shape

    @property
    def event_shape(self) -> Size:
        """The shape of one draw from one of the distributions; () for a number."""
        return self._event_shape

    def _extended_shape(self, sample_shape=()) -> tuple[int, ...]:
        return _sample_shape(sample_shape) + tuple(self._batch_shape) + tuple(self._event_shape)

    def sample(self, sample_shape=()) -> Tensor:
        """Draws of shape sample_shape + batch_shape + event_shape, without a gradient."""
        with no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=()) -> Tensor:
        """Like sample(), but differentiable in the parameters (reparameterized)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement rsample()")

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement log_prob()")

    def entropy(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement entropy()")

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement mean")

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement variance")

    @property
    def stddev(self) -> Tensor:
        return self.variance.sqrt()

    def _value(self, value) -> Tensor:
        like = getattr(self, self._parameters[0])
        return value if isinstance(value, Tensor) else as_tensor(value, dtype=like.dtype, device=like.device)

    def __repr__(self) -> str:
        parts = []
        for name in self._parameters:
            value = getattr(self, name)
            parts.append(f"{name}: {value if value.numel() == 1 else value.shape}")
        return f"{type(self).__name__}({', '.join(parts)})"


class Normal(Distribution):
    has_rsample = True
    _parameters = ("loc", "scale")

    def __init__(self, loc, scale, validate_args=None) -> None:
        self.loc, self.scale = _broadcast_all(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        return self.scale * self.scale

    def rsample(self, sample_shape=()) -> Tensor:
        # noise for every entry of the batch: parameters that broadcast do not share their noise
        eps = randn(self._extended_shape(sample_shape), dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value) -> Tensor:
        value = self._value(value)
        diff = value - self.loc
        return -(diff * diff) / (2.0 * self.variance) - self.scale.log() - 0.5 * math.log(2.0 * math.pi)

    def entropy(self) -> Tensor:
        return self.scale.log() + (0.5 + 0.5 * math.log(2.0 * math.pi))


class Uniform(Distribution):
    """Uniform on the half-open interval [low, high)."""

    has_rsample = True
    _parameters = ("low", "high")

    def __init__(self, low, high, validate_args=None) -> None:
        self.low, self.high = _broadcast_all(low, high)
        super().__init__(self.low.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return (self.high + self.low) / 2.0

    @property
    def variance(self) -> Tensor:
        span = self.high - self.low
        return span * span / 12.0

    def rsample(self, sample_shape=()) -> Tensor:
        u = rand(self._extended_shape(sample_shape), dtype=self.low.dtype, device=self.low.device)
        return self.low + u * (self.high - self.low)

    def log_prob(self, value) -> Tensor:
        value = self._value(value)
        log_density = -(self.high - self.low).log()
        inside = (value >= self.low) & (value < self.high)
        return where(inside, log_density, log_density - math.inf)

    def entropy(self) -> Tensor:
        return (self.high - self.low).log()


class Bernoulli(Distribution):
    _parameters = ("probs",)

    def __init__(self, probs=None, logits=None, validate_args=None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            (self.probs,) = _broadcast_all(probs)
            clamped = _clamp_probs(self.probs)
            self.logits = clamped.log() - (-clamped).log1p()
        else:
            (self.logits,) = _broadcast_all(logits)
            self.probs = self.logits.sigmoid()
        super().__init__(self.probs.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return self.probs

    @property
    def variance(self) -> Tensor:
        return self.probs * (1.0 - self.probs)

    def sample(self, sample_shape=()) -> Tensor:
        with no_grad():
            u = rand(self._extended_shape(sample_shape), dtype=self.probs.dtype, device=self.probs.device)
            return (u < self.probs).to(self.probs.dtype)

    def log_prob(self, value) -> Tensor:
        return -_bce_with_logits(self.logits, self._value(value))

    def entropy(self) -> Tensor:
        return _bce_with_logits(self.logits, self.probs)


class Categorical(Distribution):
    """Over the indices 0 ... K-1; the last dimension of `probs` / `logits` runs over them."""

    _parameters = ("probs",)

    def __init__(self, probs=None, logits=None, validate_args=None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            (probs,) = _broadcast_all(probs)
            if probs.ndim < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(dim=-1, keepdim=True)
            self.logits = _clamp_probs(self.probs).log()
        else:
            (logits,) = _broadcast_all(logits)
            if logits.ndim < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            self.probs = self.logits.exp()
        self._num_events = self.probs.shape[-1]
        super().__init__(self.probs.shape[:-1], validate_args=validate_args)

    def sample(self, sample_shape=()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        count = math.prod(_sample_shape(sample_shape))
        with no_grad():
            if count == 0 or 0 in shape:
                return as_tensor([], dtype=_dt.int64, device=self.probs.device).reshape(shape)
            flat = self.probs.reshape(-1, self._num_events)
            draws = flat.multinomial(count, replacement=True)        # (batch entries, count)
            return draws.t().reshape(shape)

    def log_prob(self, value) -> Tensor:
        value = value if isinstance(value, Tensor) else as_tensor(value, device=self.logits.device)
        index = value.long().unsqueeze(-1)
        index, log_pmf = broadcast_tensors(index, self.logits)
        return log_pmf.gather(-1, index[..., :1]).squeeze(-1)

    def entropy(self) -> Tensor:
        logits = self.logits.clamp(min=_dt.finfo(self.logits.dtype).min)
        return -(logits * self.probs).sum(dim=-1)


class Exponential(Distribution):
    has_rsample = True
    _parameters = ("rate",)

    def __init__(self, rate, validate_args=None) -> None:
        (self.rate,) = _broadcast_all(rate)
        super().__init__(self.rate.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def stddev(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def variance(self) -> Tensor:
        return (self.rate * self.rate).reciprocal()

    def rsample(self, sample_shape=()) -> Tensor:
        u = rand(self._extended_shape(sample_shape), dtype=self.rate.dtype, device=self.rate.device)
        u = u.clamp(max=1.0 - _dt.finfo(u.dtype).eps)            # -log(1 - u) stays finite
        return -(-u).log1p() / self.rate

    def log_prob(self, value) -> Tensor:
        return self.rate.log() - self.rate * self._value(value)

    def entropy(self) -> Tensor:
        return 1.0 - self.rate.log()


class Laplace(Distribution):
    has_rsample = True
    _parameters = ("loc", "scale")

    def __init__(self, loc, scale, validate_args=None) -> None:
        self.loc, self.scale = _broadcast_all(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return 2.0 * self.scale * self.scale

    @property
    def stddev(self) -> Tensor:
        return math.sqrt(2.0) * self.scale

    def rsample(self, sample_shape=()) -> Tensor:
        finfo = _dt.finfo(self.loc.dtype)
        u = rand(self._extended_shape(sample_shape), dtype=self.loc.dtype, device=self.loc.device)
        u = (u * 2.0 - 1.0).clamp(min=finfo.eps - 1.0, max=1.0 - finfo.eps)      # inside (-1, 1): log(1 - |u|) stays finite
        return self.loc - self.scale * u.sign() * (-u.abs()).log1p()

    def log_prob(self, value) -> Tensor:
        return -(2.0 * self.scale).log() - (self._value(value) - self.loc).abs() / self.scale

    def entropy(self) -> Tensor:
        return 1.0 + (2.0 * self.scale).log()


def _kl_normal_normal(p: Normal, q: Normal) -> Tensor:
    ratio = p.scale / q.scale
    var_ratio = ratio * ratio
    t1 = (p.loc - q.loc) / q.scale
    return 0.5 * (var_ratio + t1 * t1 - 1.0 - var_ratio.log())


def _kl_bernoulli_bernoulli(p: Bernoulli, q: Bernoulli) -> Tensor:
    # p log(p / q) + (1 - p) log((1 - p) / (1 - q)), with 0 log 0 = 0 and x log(x / 0) = inf.
    # Written with logits, the logs of probabilities near 0 and 1 stay finite; the two rules are
    # applied where a probability is exactly 0 or 1. (Clamping the probabilities instead, as
    # the unused `eps` of the old code suggested, turns the infinite cases into finite numbers
    # - and 1 - 1e-12 is 1 in float32, so it would not even remove the nan.)
    p_probs, q_probs, p_logits, q_logits = broadcast_tensors(p.probs, q.probs, p.logits, q.logits)
    t1 = p_probs * (_softplus(-q_logits) - _softplus(-p_logits))
    t1 = where(q_probs == 0, p_probs * 0.0 + math.inf, t1)
    t1 = where(p_probs == 0, p_probs * 0.0, t1)
    t2 = (1.0 - p_probs) * (_softplus(q_logits) - _softplus(p_logits))
    t2 = where(q_probs == 1, p_probs * 0.0 + math.inf, t2)
    t2 = where(p_probs == 1, p_probs * 0.0, t2)
    return t1 + t2


def _kl_categorical_categorical(p: Categorical, q: Categorical) -> Tensor:
    p_probs, q_probs, p_logits, q_logits = broadcast_tensors(p.probs, q.probs, p.logits, q.logits)
    t = p_probs * (p_logits - q_logits)
    t = where(q_probs == 0, p_probs * 0.0 + math.inf, t)
    t = where(p_probs == 0, p_probs * 0.0, t)
    return t.sum(dim=-1)


def _kl_uniform_uniform(p: Uniform, q: Uniform) -> Tensor:
    result = ((q.high - q.low) / (p.high - p.low)).log()
    outside = (q.low > p.low) | (q.high < p.high)          # p puts mass where q has none
    return where(outside, result * 0.0 + math.inf, result)


def _kl_exponential_exponential(p: Exponential, q: Exponential) -> Tensor:
    ratio = q.rate / p.rate
    return ratio - 1.0 - ratio.log()


def _kl_laplace_laplace(p: Laplace, q: Laplace) -> Tensor:
    scale_ratio = p.scale / q.scale
    distance = (p.loc - q.loc).abs()
    return -scale_ratio.log() + distance / q.scale + scale_ratio * (-distance / p.scale).exp() - 1.0


_KL = {
    (Normal, Normal): _kl_normal_normal,
    (Bernoulli, Bernoulli): _kl_bernoulli_bernoulli,
    (Categorical, Categorical): _kl_categorical_categorical,
    (Uniform, Uniform): _kl_uniform_uniform,
    (Exponential, Exponential): _kl_exponential_exponential,
    (Laplace, Laplace): _kl_laplace_laplace,
}


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """KL(p || q) for every entry of the (broadcast) batch."""
    for (p_type, q_type), fn in _KL.items():
        if isinstance(p, p_type) and isinstance(q, q_type):
            return fn(p, q)
    raise NotImplementedError(f"No KL(p || q) is implemented for p type {type(p).__name__} and q type {type(q).__name__}")
