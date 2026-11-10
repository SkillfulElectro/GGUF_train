"""Dropout layers compatible with torch.nn."""

from __future__ import annotations

from ..._tensor import Tensor
from .. import functional as F
from .module import Module


class _DropoutNd(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, got {p}")
        self.p = float(p)
        self.inplace = bool(inplace)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, p=self.p, training=self.training, inplace=self.inplace)


class Dropout1d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout1d(input, p=self.p, training=self.training, inplace=self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout2d(input, p=self.p, training=self.training, inplace=self.inplace)


class Dropout3d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, p=self.p, training=self.training, inplace=self.inplace)
