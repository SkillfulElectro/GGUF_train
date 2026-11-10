"""Pooling & Upsampling modules compatible with torch.nn."""

from __future__ import annotations

from ..._tensor import Tensor
from .. import functional as F
from .module import Module


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, self.output_size)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = int(output_size)

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size, return_indices: bool = False) -> None:
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input: Tensor):
        return F.adaptive_max_pool2d(input, self.output_size, return_indices=self.return_indices)


class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode: str = "nearest", align_corners: bool | None = None) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
