"""Normalization layers compatible with torch.nn."""

from __future__ import annotations

from ... import _dtype as _dt
from ..._creation import ones, zeros
from ..._tensor import Tensor
from .. import functional as F
from ..parameter import Parameter
from .module import Module


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        dt = dtype or _dt.get_default_dtype()
        if self.elementwise_affine:
            self.weight = Parameter(ones(self.normalized_shape, dtype=dt, device=device))
            if bias:
                self.bias = Parameter(zeros(self.normalized_shape, dtype=dt, device=device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps: float | None = None, elementwise_affine: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = 1e-6 if eps is None else float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        dt = dtype or _dt.get_default_dtype()
        if self.elementwise_affine:
            self.weight = Parameter(ones(self.normalized_shape, dtype=dt, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        dt = dtype or _dt.get_default_dtype()
        if self.affine:
            self.weight = Parameter(ones((self.num_channels,), dtype=dt, device=device))
            self.bias = Parameter(zeros((self.num_channels,), dtype=dt, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)


class _BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        dt = dtype or _dt.get_default_dtype()
        if self.affine:
            self.weight = Parameter(ones((self.num_features,), dtype=dt, device=device))
            self.bias = Parameter(zeros((self.num_features,), dtype=dt, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", zeros((self.num_features,), dtype=dt, device=device))
            self.register_buffer("running_var", ones((self.num_features,), dtype=dt, device=device))
            self.register_buffer("num_batches_tracked", zeros((), dtype=_dt.int64, device=device))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, input: Tensor) -> Tensor:
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=self.training or not self.track_running_stats,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"


class BatchNorm1d(_BatchNorm):
    pass


class BatchNorm2d(_BatchNorm):
    pass


class BatchNorm3d(_BatchNorm):
    pass


class InstanceNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        dt = dtype or _dt.get_default_dtype()
        if self.affine:
            self.weight = Parameter(ones((self.num_features,), dtype=dt, device=device))
            self.bias = Parameter(zeros((self.num_features,), dtype=dt, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.instance_norm(input, weight=self.weight, bias=self.bias, eps=self.eps)


class InstanceNorm2d(InstanceNorm1d):
    pass
