"""Flatten, Unflatten, and Padding modules compatible with torch.nn."""

from __future__ import annotations

from ..._tensor import Tensor
from .. import functional as F
from .module import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)
        self.end_dim = int(end_dim)

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    def __init__(self, dim: int, unflattened_size) -> None:
        super().__init__()
        self.dim = int(dim)
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, input: Tensor) -> Tensor:
        return input.unflatten(self.dim, self.unflattened_size)


class _PadNd(Module):
    def __init__(self, padding, mode: str, value: float = 0.0) -> None:
        super().__init__()
        self.padding = (padding,) * 4 if isinstance(padding, int) else tuple(padding)
        self.mode = mode
        self.value = float(value)

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, mode=self.mode, value=self.value)


class ZeroPad2d(_PadNd):
    def __init__(self, padding) -> None:
        super().__init__(padding, mode="constant", value=0.0)


class ConstantPad2d(_PadNd):
    def __init__(self, padding, value: float) -> None:
        super().__init__(padding, mode="constant", value=value)


class ReflectionPad2d(_PadNd):
    def __init__(self, padding) -> None:
        super().__init__(padding, mode="reflect")


class ReplicationPad2d(_PadNd):
    def __init__(self, padding) -> None:
        super().__init__(padding, mode="replicate")
