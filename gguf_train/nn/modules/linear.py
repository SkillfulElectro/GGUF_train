"""Linear layers compatible with torch.nn."""

from __future__ import annotations

import math

from ... import _dtype as _dt
from ..._creation import empty
from ..._tensor import Tensor
from .. import functional as F, init
from ..parameter import Parameter
from .module import Module


class Identity(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.out_features, self.in_features), dtype=dt, device=device))
        if bias:
            self.bias = Parameter(empty((self.out_features,), dtype=dt, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.weight.dtype.is_quantized:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                init.uniform_(self.bias, -bound, bound)

    def quantize_weight_(self, qtype) -> None:
        """Convert weight to a ggml quantized format (e.g. q4_0, q8_0) for inference/LoRA."""
        qdt = _dt.as_dtype(qtype)
        qw = self.weight.detach().quantize(qdt)
        self._parameters["weight"] = Parameter(qw, requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Bilinear(Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.in1_features = int(in1_features)
        self.in2_features = int(in2_features)
        self.out_features = int(out_features)
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.out_features, self.in1_features, self.in2_features), dtype=dt, device=device))
        if bias:
            self.bias = Parameter(empty((self.out_features,), dtype=dt, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(self.in1_features, 1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in1_features={self.in1_features}, in2_features={self.in2_features}, out_features={self.out_features}, bias={self.bias is not None}"


class LazyLinear(Module):
    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = 0
        self.out_features = int(out_features)
        self._has_bias = bias
        self._dev = device
        self._dt = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight is None:
            self.in_features = int(input.shape[-1])
            dt = self._dt or input.dtype
            dev = self._dev or input.device
            self.weight = Parameter(empty((self.out_features, self.in_features), dtype=dt, device=dev))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self._has_bias:
                self.bias = Parameter(empty((self.out_features,), dtype=dt, device=dev))
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                init.uniform_(self.bias, -bound, bound)
        return F.linear(input, self.weight, self.bias)
