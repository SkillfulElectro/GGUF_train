"""Activation modules compatible with torch.nn."""

from __future__ import annotations

from ..._creation import full
from ..._tensor import Tensor
from .. import functional as F
from ..parameter import Parameter
from .module import Module


class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class ReLU6(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu6(input, inplace=self.inplace)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None) -> None:
        super().__init__()
        self.num_parameters = int(num_parameters)
        self.weight = Parameter(full((self.num_parameters,), float(init), dtype=dtype, device=device))

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)


class ELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, alpha=self.alpha, inplace=self.inplace)


class CELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, alpha=self.alpha, inplace=self.inplace)


class SELU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, inplace=self.inplace)


class GELU(Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class SiLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)


class Mish(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, inplace=self.inplace)


class Hardswish(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, inplace=self.inplace)


class Hardsigmoid(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, inplace=self.inplace)


class Hardtanh(Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, min_val=self.min_val, max_val=self.max_val, inplace=self.inplace)


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class LogSigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.logsigmoid(input)


class Softplus(Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, beta=self.beta, threshold=self.threshold)


class Softsign(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.softsign(input)


class GLU(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, input: Tensor) -> Tensor:
        return F.glu(input, dim=self.dim)


class Softmax(Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Softmax2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=1)
