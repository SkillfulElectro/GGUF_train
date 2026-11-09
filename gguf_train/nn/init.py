"""Weight initialization functions compatible with torch.nn.init."""

from __future__ import annotations

import math

from .. import _autograd_state
from .._tensor import Tensor


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    linear_fns = {
        "linear", "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid",
    }
    if nonlinearity in linear_fns:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        neg_slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + neg_slope ** 2))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    if nonlinearity in ("gelu", "silu", "swish"):
        return math.sqrt(2.0)
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    dims = tensor.dim()
    if dims < 2:
        raise ValueError("Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if dims > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    return num_input_fmaps * receptive_field_size, num_output_fmaps * receptive_field_size


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    mode = mode.lower()
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode {mode} not supported, use 'fan_in' or 'fan_out'")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    with _autograd_state.no_grad():
        return tensor.uniform_(a, b)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    with _autograd_state.no_grad():
        return tensor.normal_(mean, std)


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Tensor:
    with _autograd_state.no_grad():
        return tensor.trunc_normal_(mean, std, a, b)


def constant_(tensor: Tensor, val: float) -> Tensor:
    with _autograd_state.no_grad():
        return tensor.fill_(val)


def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)


def zeros_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 0.0)


def eye_(tensor: Tensor) -> Tensor:
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    from .._creation import eye
    with _autograd_state.no_grad():
        return tensor.copy_(eye(tensor.shape[0], tensor.shape[1], dtype=tensor.dtype, device=tensor.device))


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    if tensor.ndim not in (3, 4, 5):
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")
    with _autograd_state.no_grad():
        tensor.zero_()
        out_ch, in_ch = tensor.shape[0], tensor.shape[1]
        out_g = out_ch // groups
        min_d = min(out_g, in_ch)
        mid = tuple(s // 2 for s in tensor.shape[2:])
        for g in range(groups):
            for d in range(min_d):
                idx = (g * out_g + d, d) + mid
                tensor[idx] = 1.0
    return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(float(fan))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(float(fan))
    return normal_(tensor, 0.0, std)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    from .._creation import randn
    with _autograd_state.no_grad():
        flattened = randn((rows, cols), dtype=tensor.dtype, device=tensor.device)
        if rows < cols:
            flattened = flattened.t()
        q, r = flattened.qr()
        d = r.diag()
        ph = d.sign()
        q = q * ph.unsqueeze(0)
        if rows < cols:
            q = q.t()
        tensor.copy_(q.reshape(tensor.shape) * gain)
    return tensor


def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01) -> Tensor:
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))
    from .._creation import randperm
    with _autograd_state.no_grad():
        tensor.normal_(0.0, std)
        for col_idx in range(cols):
            row_indices = randperm(rows)[:num_zeros]
            tensor[row_indices, col_idx] = 0.0
    return tensor
