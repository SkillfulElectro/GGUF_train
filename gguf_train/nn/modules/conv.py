"""Convolution modules compatible with torch.nn."""

from __future__ import annotations

import math

from ... import _dtype as _dt
from ..._creation import empty
from ..._tensor import Tensor
from .. import functional as F, init
from ..parameter import Parameter
from .module import Module


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding if isinstance(padding, int) else padding[0]
        self.dilation = dilation if isinstance(dilation, int) else dilation[0]
        self.groups = int(groups)
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.out_channels, self.in_channels // self.groups, self.kernel_size[0]), dtype=dt, device=device))
        if bias:
            self.bias = Parameter(empty((self.out_channels,), dtype=dt, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = int(groups)
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.out_channels, self.in_channels // self.groups, *self.kernel_size), dtype=dt, device=device))
        if bias:
            self.bias = Parameter(empty((self.out_channels,), dtype=dt, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else tuple(output_padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = int(groups)
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.in_channels, self.out_channels // self.groups, *self.kernel_size), dtype=dt, device=device))
        if bias:
            self.bias = Parameter(empty((self.out_channels,), dtype=dt, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        self.padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)
        self.groups = int(groups)
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.out_channels, self.in_channels // self.groups, *self.kernel_size), dtype=dt, device=device))
        if bias:
            self.bias = Parameter(empty((self.out_channels,), dtype=dt, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
