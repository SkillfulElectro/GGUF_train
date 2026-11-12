"""Neural network functional interface compatible with torch.nn.functional."""

from __future__ import annotations

import math

from .. import _C, _autograd_state, _dtype as _dt
from .._creation import ones, tensor as _tensor, zeros, zeros_like
from .._functions import (
    AvgPool2dFn,
    BatchNormFn,
    Conv1dFn,
    Conv2dFn,
    Conv3dFn,
    ConvTranspose2dFn,
    CrossEntropyFn,
    EmbeddingFn,
    FlashAttentionFn,
    GroupNormFn,
    LayerNormFn,
    LinearFn,
    MaxPool2dFn,
    PadFn,
    RMSNormFn,
    RoPEFn,
    SoftmaxFn,
    SwiGLUFn,
    UnarySimpleFn,
    Upsample2dFn,
)
from .._ops import cat, stack, where
from .._tensor import (
    Tensor,
    _AUTOCAST_FLOAT32,
    _AUTOCAST_KEEP,
    _AUTOCAST_LOWER,
    _AUTOCAST_PROMOTE,
    _autocast_float32,
    _autocast_op,
    _is_int,
    _is_scalar,
)


def _pair(v) -> tuple[int, int]:
    return (int(v), int(v)) if _is_int(v) else (int(v[0]), int(v[1]))


def _triple(v) -> tuple[int, int, int]:
    return (int(v), int(v), int(v)) if _is_int(v) else (int(v[0]), int(v[1]), int(v[2]))


def _reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction mode: {reduction!r}")


# ---------- Activations ----------

def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return input.relu_()
    return UnarySimpleFn.apply(input, _C.UNARY_RELU)


def relu_(input: Tensor) -> Tensor:
    return input.relu_()


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_RELU6)
    if inplace:
        input.copy_(res)
        return input
    return res


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_LEAKY_RELU, float(negative_slope))
    if inplace:
        input.copy_(res)
        return input
    return res


def leaky_relu_(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    return leaky_relu(input, negative_slope=negative_slope, inplace=True)


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_ELU, float(alpha))
    if inplace:
        input.copy_(res)
        return input
    return res


def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_CELU, float(alpha))
    if inplace:
        input.copy_(res)
        return input
    return res


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_SELU)
    if inplace:
        input.copy_(res)
        return input
    return res


@_autocast_op(_AUTOCAST_LOWER)
def prelu(input: Tensor, weight: Tensor) -> Tensor:
    w = weight
    if w.ndim == 1 and w.numel() > 1 and input.ndim >= 2:
        sh = [1] * input.ndim
        sh[1] = w.numel()
        w = w.reshape(tuple(sh))
    return where(input > 0, input, input * w)


def glu(input: Tensor, dim: int = -1) -> Tensor:
    a, b = input.chunk(2, dim=dim)
    return a * b.sigmoid()


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    if approximate == "tanh":
        return UnarySimpleFn.apply(input, _C.UNARY_GELU_TANH)
    if approximate == "quick":
        return UnarySimpleFn.apply(input, _C.UNARY_GELU_QUICK)
    return UnarySimpleFn.apply(input, _C.UNARY_GELU)


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_SILU)
    if inplace:
        input.copy_(res)
        return input
    return res


swish = silu


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_MISH)
    if inplace:
        input.copy_(res)
        return input
    return res


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_HARDSWISH)
    if inplace:
        input.copy_(res)
        return input
    return res


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_HARDSIGMOID)
    if inplace:
        input.copy_(res)
        return input
    return res


def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    res = UnarySimpleFn.apply(input, _C.UNARY_HARDTANH, float(min_val), float(max_val))
    if inplace:
        input.copy_(res)
        return input
    return res


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def logsigmoid(input: Tensor) -> Tensor:
    return UnarySimpleFn.apply(input, _C.UNARY_LOG_SIGMOID)


@_autocast_op(_AUTOCAST_FLOAT32)
def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    return UnarySimpleFn.apply(input, _C.UNARY_SOFTPLUS, float(beta), float(threshold))


def softsign(input: Tensor) -> Tensor:
    return input / (1.0 + input.abs())


def tanhshrink(input: Tensor) -> Tensor:
    return input - input.tanh()


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    pos = (input - lambd).clamp(min=0.0)
    neg = (input + lambd).clamp(max=0.0)
    return pos + neg


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    mask = (input > lambd) | (input < -lambd)
    return where(mask, input, zeros_like(input))


def threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    res = where(input > threshold, input, zeros_like(input) + value)
    if inplace:
        input.copy_(res)
        return input
    return res


def softmax(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    (_stacklevel,)
    d = -1 if dim is None else dim
    (x,) = _autocast_float32(input) if dtype is None else (input.to(dtype),)
    return SoftmaxFn.apply(x, d, 0)


def softmin(input: Tensor, dim: int | None = None, dtype=None) -> Tensor:
    return softmax(-input, dim=dim, dtype=dtype)


def log_softmax(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    (_stacklevel,)
    d = -1 if dim is None else dim
    (x,) = _autocast_float32(input) if dtype is None else (input.to(dtype),)
    return SoftmaxFn.apply(x, d, 1)


def gumbel_softmax(logits: Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    from .._creation import rand_like
    u = rand_like(logits).clamp(min=eps, max=1.0 - eps)
    g = -(-(u.log())).log()
    y_soft = softmax((logits + g) / tau, dim=dim)
    if hard:
        idx = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = zeros_like(y_soft).scatter(dim, idx, ones(idx.shape, dtype=y_soft.dtype, device=y_soft.device))
        return y_hard - y_soft.detach() + y_soft
    return y_soft


# ---------- Linear / Dropout / Embedding ----------

@_autocast_op(_AUTOCAST_LOWER)
def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return LinearFn.apply(input, weight, bias)


@_autocast_op(_AUTOCAST_PROMOTE)
def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    # weight: (out_features, in1_features, in2_features)
    out_f = weight.shape[0]
    res_list = []
    for k in range(out_f):
        # (..., in1) @ (in1, in2) -> (..., in2) * input2 -> sum(-1)
        proj = input1.matmul(weight[k]) * input2
        res_list.append(proj.sum(dim=-1, keepdim=True))
    out = cat(res_list, dim=-1)
    if bias is not None:
        out = out + bias
    return out


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if not training or p == 0.0:
        return input
    if p == 1.0:
        out = zeros_like(input)
        if inplace:
            input.copy_(out)
            return input
        return out
    from .._creation import rand_like
    keep = 1.0 - float(p)
    mask = (rand_like(input) < keep).to(input.dtype) / keep
    out = input * mask
    if inplace:
        input.copy_(out)
        return input
    return out


def dropout1d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    return dropout2d(input, p=p, training=training, inplace=inplace)


def dropout2d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if not training or p == 0.0:
        return input
    from .._creation import rand
    keep = 1.0 - float(p)
    m_sh = (input.shape[0], input.shape[1]) + (1,) * (input.ndim - 2)
    mask = (rand(m_sh, dtype=input.dtype, device=input.device) < keep).to(input.dtype) / keep
    out = input * mask
    if inplace:
        input.copy_(out)
        return input
    return out


def dropout3d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    return dropout2d(input, p=p, training=training, inplace=inplace)


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    (max_norm, norm_type, scale_grad_by_freq, sparse)
    pidx = -1 if padding_idx is None else int(padding_idx)
    return EmbeddingFn.apply(weight, input, pidx)


@_autocast_op(_AUTOCAST_KEEP)
def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Tensor | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> Tensor:
    """Sums, means or maxima over bags of embeddings.

    input 2-D (B, L): every row is a bag. input 1-D with offsets: bag i is
    input[offsets[i]:offsets[i + 1]], the last one runs to the end of the input (or to the last
    offset with include_last_offset). An empty bag gives zeros. Entries equal to padding_idx
    are left out of the reduction; per_sample_weights (mode "sum") scales each entry.
    """
    (max_norm, norm_type, scale_grad_by_freq, sparse)
    if mode not in ("sum", "mean", "max"):
        raise ValueError(f"embedding_bag: mode has to be one of 'sum', 'mean' or 'max', got {mode!r}")
    if weight.ndim != 2:
        raise ValueError(f"embedding_bag: weight has to be a 2D tensor, got {weight.ndim}D")
    if per_sample_weights is not None:
        if mode != "sum":
            raise NotImplementedError(
                f"embedding_bag: per_sample_weights was not None. per_sample_weights is only supported for mode='sum' (got mode='{mode}')"
            )
        if per_sample_weights.shape != input.shape:
            raise ValueError(
                f"embedding_bag: per_sample_weights of shape {list(per_sample_weights.shape)} must have the shape of the input "
                f"{list(input.shape)}"
            )
    n_rows, emb_dim = weight.shape
    bounds = None                    # 1-D input: bag i covers [bounds[i], bounds[i + 1])
    if input.ndim == 2:
        if offsets is not None:
            raise ValueError("embedding_bag: if input is 2D, then offsets has to be None, as every row of the input is a bag")
        n_bags, bag_len = input.shape
        flat = input.reshape((-1,))
    elif input.ndim == 1:
        if offsets is None:
            raise ValueError("embedding_bag: offsets has to be a 1D tensor for a 1D input, got None")
        if offsets.ndim != 1:
            raise ValueError(f"embedding_bag: offsets has to be a 1D tensor, got {offsets.ndim}D")
        if offsets.is_floating_point() or offsets.is_complex() or offsets.dtype == _dt.bool:
            raise TypeError(f"embedding_bag: offsets has to be an integer tensor, got {offsets.dtype}")
        bounds = [int(o) for o in offsets.tolist()]
        total = input.shape[0]
        if include_last_offset:
            if not bounds:
                raise ValueError("embedding_bag: include_last_offset needs at least one offset, the end of the last bag")
        else:
            bounds.append(total)
        if len(bounds) > 1 and bounds[0] != 0:
            raise ValueError(f"embedding_bag: offsets[0] has to be 0, the first bag starts at the first index, got {bounds[0]}")
        if any(b2 < b1 for b1, b2 in zip(bounds, bounds[1:])):
            raise ValueError("embedding_bag: offsets have to be in ascending order")
        if bounds[-1] > total or bounds[0] < 0:
            raise ValueError(f"embedding_bag: offsets cannot go beyond the length of the input ({total}), got {bounds[-1]}")
        n_bags = len(bounds) - 1
        flat = input if bounds[-1] == total else input[: bounds[-1]]
        if per_sample_weights is not None and bounds[-1] != total:
            per_sample_weights = per_sample_weights[: bounds[-1]]
    else:
        raise ValueError(f"embedding_bag: input has to be a 1D or 2D tensor, got {input.ndim}D")

    pidx = None
    if padding_idx is not None:
        pidx = int(padding_idx) + n_rows if int(padding_idx) < 0 else int(padding_idx)
        if not 0 <= pidx < n_rows:
            raise ValueError(f"embedding_bag: padding_idx {padding_idx} is out of range for {n_rows} embeddings")

    # (N, D); the row of padding_idx comes back as zeros and takes no gradient
    emb = EmbeddingFn.apply(weight, flat, -1 if pidx is None else pidx)
    keep = None if pidx is None else (flat != pidx)          # (N,) entries that count
    if per_sample_weights is not None:
        emb = emb * per_sample_weights.reshape((-1, 1)).to(emb.dtype)
    if n_bags == 0:
        return zeros((0, emb_dim), dtype=weight.dtype, device=weight.device)

    if bounds is None:
        # equal bags: one reduction over the bag dimension
        bags = emb.reshape((n_bags, bag_len, emb_dim))
        counts = None if keep is None else keep.reshape((n_bags, bag_len)).to(weight.dtype).sum(dim=1)
        if bag_len == 0:
            return zeros((n_bags, emb_dim), dtype=weight.dtype, device=weight.device)
        if mode == "max":
            if keep is None:
                return bags.amax(dim=1)
            hidden = (~keep).reshape((n_bags, bag_len, 1))
            best = bags.masked_fill(hidden, float("-inf")).amax(dim=1)
            return where((counts > 0).unsqueeze(-1), best, zeros_like(best))
        total_emb = bags.sum(dim=1)
        if mode == "sum":
            return total_emb
        if counts is None:
            return total_emb / float(bag_len)
        return total_emb / where(counts > 0, counts, zeros_like(counts) + 1.0).unsqueeze(-1)

    sizes = [b2 - b1 for b1, b2 in zip(bounds, bounds[1:])]
    if mode == "max":
        rows = []
        for i, size in enumerate(sizes):
            seg = emb[bounds[i] : bounds[i + 1]]
            if keep is not None and size > 0:
                seg_keep = keep[bounds[i] : bounds[i + 1]]
                if int(seg_keep.sum().item()) == 0:
                    size = 0
                else:
                    seg = seg.masked_fill((~seg_keep).unsqueeze(-1), float("-inf"))
            rows.append(seg.amax(dim=0) if size > 0 else zeros((emb_dim,), dtype=weight.dtype, device=weight.device))
        return stack(rows, dim=0)

    # ragged bags: add every row to the row of its bag
    bag_of = [i for i, size in enumerate(sizes) for _ in range(size)]
    out = zeros((n_bags, emb_dim), dtype=emb.dtype, device=emb.device)
    if not bag_of:
        return out
    bag_ids = _tensor(bag_of, dtype=_dt.int64, device=emb.device)
    out = out.scatter_add(0, bag_ids.unsqueeze(-1).expand(len(bag_of), emb_dim), emb)
    if mode == "sum":
        return out
    if keep is None:
        counts = _tensor([float(max(size, 1)) for size in sizes], dtype=emb.dtype, device=emb.device)
    else:
        counts = zeros((n_bags,), dtype=emb.dtype, device=emb.device).scatter_add(0, bag_ids, keep.to(emb.dtype))
        counts = where(counts > 0, counts, zeros_like(counts) + 1.0)
    return out / counts.unsqueeze(-1)


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    if tensor.dtype.is_floating_point or tensor.dtype.is_complex or tensor.dtype == _dt.bool:
        raise RuntimeError(f"one_hot is only applicable to integer class indices, got {tensor.dtype}")
    largest = -1
    if tensor.numel() > 0:
        largest = int(tensor.max().item())
        if int(tensor.min().item()) < 0:
            # scatter would count a negative index from the end and mark the wrong class
            raise RuntimeError("one_hot: class values must be non-negative")
    if num_classes < 0:
        num_classes = largest + 1 if tensor.numel() > 0 else 1
    elif largest >= num_classes:
        raise RuntimeError(f"one_hot: class value {largest} is out of range for num_classes={num_classes}")
    out_sh = tensor.shape + (num_classes,)
    idx = tensor.long().unsqueeze(-1)
    out = zeros(out_sh, dtype=_dt.int64, device=tensor.device)
    src = ones(idx.shape, dtype=_dt.int64, device=tensor.device)
    return out.scatter(-1, idx, src)


# ---------- Normalization ----------

def rms_norm(input: Tensor, normalized_shape, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    (normalized_shape,)
    return RMSNormFn.apply(input, weight, float(eps))


@_autocast_op(_AUTOCAST_FLOAT32)
def layer_norm(
    input: Tensor,
    normalized_shape,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    norm_ndim = 1 if _is_int(normalized_shape) else len(normalized_shape)
    return LayerNormFn.apply(input, norm_ndim, weight, bias, float(eps))


@_autocast_op(_AUTOCAST_FLOAT32)
def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    return GroupNormFn.apply(input, int(num_groups), weight, bias, float(eps))


def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    return BatchNormFn.apply(input, running_mean, running_var, weight, bias, bool(training), float(momentum), float(eps))


@_autocast_op(_AUTOCAST_KEEP)
def instance_norm(
    input: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    (running_mean, running_var, use_input_stats, momentum)
    B, C = input.shape[0], input.shape[1]
    return group_norm(input, C, weight=weight, bias=bias, eps=eps)


@_autocast_op(_AUTOCAST_FLOAT32)
def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out=None) -> Tensor:
    denom = input.norm(p=p, dim=dim, keepdim=True).clamp(min=eps)
    res = input / denom
    if out is not None:
        out.copy_(res)
        return out
    return res


# ---------- Convolutions & Pooling ----------

@_autocast_op(_AUTOCAST_LOWER)
def conv1d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Tensor:
    s = stride[0] if isinstance(stride, (tuple, list)) else int(stride)
    p = padding[0] if isinstance(padding, (tuple, list)) else int(padding)
    d = dilation[0] if isinstance(dilation, (tuple, list)) else int(dilation)
    return Conv1dFn.apply(input, weight, bias, s, p, d, int(groups))


@_autocast_op(_AUTOCAST_LOWER)
def conv2d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups: int = 1) -> Tensor:
    return Conv2dFn.apply(input, weight, bias, _pair(stride), _pair(padding), _pair(dilation), int(groups))


@_autocast_op(_AUTOCAST_LOWER)
def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups: int = 1,
    dilation=1,
) -> Tensor:
    return ConvTranspose2dFn.apply(
        input, weight, bias, _pair(stride), _pair(padding), _pair(output_padding), _pair(dilation), int(groups)
    )


@_autocast_op(_AUTOCAST_LOWER)
def conv3d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups: int = 1) -> Tensor:
    return Conv3dFn.apply(input, weight, bias, _triple(stride), _triple(padding), _triple(dilation), int(groups))


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False, return_indices: bool = False):
    (ceil_mode,)
    k = _pair(kernel_size)
    s = k if stride is None else _pair(stride)
    p = _pair(padding)
    d = _pair(dilation)
    out, idx = MaxPool2dFn.apply(input, k, s, p, d)
    if return_indices:
        return out, idx
    return out


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override=None) -> Tensor:
    (ceil_mode, divisor_override)
    k = _pair(kernel_size)
    s = k if stride is None else _pair(stride)
    p = _pair(padding)
    return AvgPool2dFn.apply(input, k, s, p, 1 if count_include_pad else 0)


def max_pool1d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False, return_indices: bool = False):
    k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
    s = k if stride is None else (stride[0] if isinstance(stride, (tuple, list)) else int(stride))
    p = padding[0] if isinstance(padding, (tuple, list)) else int(padding)
    d = dilation[0] if isinstance(dilation, (tuple, list)) else int(dilation)
    res = max_pool2d(input.unsqueeze(-2), (1, k), (1, s), (0, p), (1, d), ceil_mode=ceil_mode, return_indices=return_indices)
    if return_indices:
        return res[0].squeeze(-2), res[1].squeeze(-2)
    return res.squeeze(-2)


def avg_pool1d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True) -> Tensor:
    k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
    s = k if stride is None else (stride[0] if isinstance(stride, (tuple, list)) else int(stride))
    p = padding[0] if isinstance(padding, (tuple, list)) else int(padding)
    return avg_pool2d(input.unsqueeze(-2), (1, k), (1, s), (0, p), ceil_mode=ceil_mode, count_include_pad=count_include_pad).squeeze(-2)


def _adaptive_sizes(name: str, input: Tensor, output_size) -> tuple[int, int]:
    """(out_h, out_w) of an adaptive 2-D pooling; None keeps the input size of that dimension."""
    if input.ndim not in (3, 4):
        raise RuntimeError(f"{name}: expected a 3D (C, H, W) or 4D (N, C, H, W) input, got {input.ndim}D")
    sizes = tuple(output_size) if isinstance(output_size, (tuple, list)) else (output_size, output_size)
    if len(sizes) != 2:
        raise RuntimeError(f"{name}: output_size must be an int or a pair, got {output_size!r}")
    out = []
    for o, i in zip(sizes, input.shape[-2:]):
        o = i if o is None else int(o)
        if o < 0:
            raise RuntimeError(f"{name}: output_size must be non-negative, got {output_size!r}")
        if i == 0 and o > 0:
            raise RuntimeError(f"{name}: the input has an empty spatial dimension: {list(input.shape)}")
        out.append(o)
    return out[0], out[1]


def _adaptive_windows(in_size: int, out_size: int) -> list[tuple[int, int]]:
    """The windows of adaptive pooling: output i covers [floor(i*in/out), ceil((i+1)*in/out))."""
    return [((i * in_size) // out_size, -((-(i + 1) * in_size) // out_size)) for i in range(out_size)]


def _adaptive_avg_matrix(in_size: int, out_size: int, like: Tensor) -> Tensor:
    """(out_size, in_size): row i holds 1/len(window i) on the inputs of window i."""
    flat = []
    for start, end in _adaptive_windows(in_size, out_size):
        w = 1.0 / (end - start)
        flat.extend(w if start <= k < end else 0.0 for k in range(in_size))
    return _tensor(flat, dtype=like.dtype, device=like.device).reshape((out_size, in_size))


def adaptive_avg_pool2d(input: Tensor, output_size) -> Tensor:
    oh, ow = _adaptive_sizes("adaptive_avg_pool2d", input, output_size)
    ih, iw = input.shape[-2], input.shape[-1]
    if oh == 1 and ow == 1:
        return input.mean(dim=(-2, -1), keepdim=True)
    if input.ndim == 4 and oh > 0 and ow > 0 and ih % oh == 0 and iw % ow == 0:
        sh, sw = ih // oh, iw // ow
        return avg_pool2d(input, (sh, sw), stride=(sh, sw))
    if not input.is_floating_point():
        raise RuntimeError(f"adaptive_avg_pool2d: expected a floating point input, got {input.dtype}")
    # Windows of unequal size. The mean over a rectangle is the mean over its rows, then over
    # its columns: one matrix product per dimension, with the window averages as the matrix.
    out = input
    if oh != ih:
        out = _adaptive_avg_matrix(ih, oh, input).matmul(out)
    if ow != iw:
        out = out.matmul(_adaptive_avg_matrix(iw, ow, input).transpose(0, 1))
    return out


def adaptive_avg_pool1d(input: Tensor, output_size: int) -> Tensor:
    if input.ndim not in (2, 3):
        raise RuntimeError(f"adaptive_avg_pool1d: expected a 2D (C, L) or 3D (N, C, L) input, got {input.ndim}D")
    return adaptive_avg_pool2d(input.unsqueeze(-2), (1, int(output_size))).squeeze(-2)


def adaptive_max_pool2d(input: Tensor, output_size, return_indices: bool = False):
    oh, ow = _adaptive_sizes("adaptive_max_pool2d", input, output_size)
    ih, iw = input.shape[-2], input.shape[-1]
    if oh > 0 and ow > 0 and ih % oh == 0 and iw % ow == 0:
        sh, sw = ih // oh, iw // ow
        batched = input if input.ndim == 4 else input.unsqueeze(0)
        res = max_pool2d(batched, (sh, sw), stride=(sh, sw), return_indices=return_indices)
        if input.ndim == 4:
            return res
        return (res[0].squeeze(0), res[1].squeeze(0)) if return_indices else res.squeeze(0)
    if oh == 0 or ow == 0:
        empty_out = input[..., :oh, :ow]
        return (empty_out, empty_out.long()) if return_indices else empty_out
    # Windows of unequal size, [floor(i*in/out), ceil((i+1)*in/out)): the maximum of a rectangle
    # is the maximum over its columns, then over its rows.
    col_vals, col_idx = [], []
    for start, end in _adaptive_windows(iw, ow):
        v, i = input[..., :, start:end].max(dim=-1)
        col_vals.append(v)
        col_idx.append(i + start)
    cols = stack(col_vals, dim=-1)              # (..., ih, ow)
    cols_i = stack(col_idx, dim=-1)
    out_vals, out_idx = [], []
    for start, end in _adaptive_windows(ih, oh):
        v, i = cols[..., start:end, :].max(dim=-2)
        out_vals.append(v)
        if return_indices:
            col = cols_i[..., start:end, :].gather(-2, i.unsqueeze(-2)).squeeze(-2)
            out_idx.append((i + start) * iw + col)   # position in the flattened (H, W) plane
    values = stack(out_vals, dim=-2)
    return (values, stack(out_idx, dim=-2)) if return_indices else values


@_autocast_op(_AUTOCAST_FLOAT32)
def interpolate(input: Tensor, size=None, scale_factor=None, mode: str = "nearest", align_corners: bool | None = None, recompute_scale_factor=None, antialias: bool = False) -> Tensor:
    (recompute_scale_factor, antialias)
    if size is not None:
        oh, ow = _pair(size)
    elif scale_factor is not None:
        sf_h, sf_w = (float(scale_factor), float(scale_factor)) if _is_scalar(scale_factor) else (float(scale_factor[0]), float(scale_factor[1]))
        oh = int(input.shape[-2] * sf_h)
        ow = int(input.shape[-1] * sf_w)
    else:
        raise ValueError("either size or scale_factor should be defined")
    mode_map = {"nearest": 0, "bilinear": 1, "bicubic": 2, "linear": 1}
    mcode = mode_map.get(mode, 1)
    return Upsample2dFn.apply(input, (oh, ow), mcode, 1 if align_corners else 0)


upsample = interpolate


def pad(input: Tensor, pad, mode: str = "constant", value: float = 0.0) -> Tensor:
    mode_map = {"constant": 0, "reflect": 1, "replicate": 2, "circular": 3}
    if mode not in mode_map:
        raise ValueError(f"Unsupported pad mode: {mode!r}")
    return PadFn.apply(input, tuple(int(v) for v in pad), mode_map[mode], float(value if value is not None else 0.0))


# ---------- Attention / RoPE / SwiGLU ----------

@_autocast_op(_AUTOCAST_LOWER)
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    (enable_gqa,)
    sc = (1.0 / math.sqrt(query.shape[-1])) if scale is None else float(scale)
    if attn_mask is not None:
        # bool: True = may attend; float: added to the scores. Broadcast against (B, H, L, S).
        if query.ndim == 4 and key.ndim == 4:
            want = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])
            if attn_mask.ndim > 4 or any(m != 1 and m != w for m, w in zip(reversed(attn_mask.shape), reversed(want))):
                raise RuntimeError(
                    f"scaled_dot_product_attention: attn_mask of shape {list(attn_mask.shape)} cannot be broadcast "
                    f"to the attention weights {list(want)}"
                )
            if attn_mask.ndim != 4:
                attn_mask = attn_mask.reshape((1,) * (4 - attn_mask.ndim) + tuple(attn_mask.shape))
    if dropout_p == 0.0:
        return FlashAttentionFn.apply(query, key, value, attn_mask, sc, is_causal)
    # Dropout path uses differentiable operations directly
    B, H, L, D = query.shape
    H_kv = key.shape[1]
    k_u = key if H == H_kv else key.repeat_interleave(H // H_kv, dim=1)
    v_u = value if H == H_kv else value.repeat_interleave(H // H_kv, dim=1)
    scores = query.matmul(k_u.transpose(-2, -1)) * sc
    if is_causal:
        S = key.shape[-2]
        cm = ones((L, S), dtype=query.dtype, device=query.device).triu(diagonal=1) * (-1e9)
        scores = scores + cm
    if attn_mask is not None:
        if attn_mask.dtype == _dt.bool:
            scores = scores.masked_fill(~attn_mask, -1e9)
        else:
            scores = scores + attn_mask
    probs = softmax(scores, dim=-1)
    probs = dropout(probs, p=dropout_p, training=True)
    return probs.matmul(v_u)


def rope(input: Tensor, positions: Tensor, n_dims: int | None = None, mode: int = 0, n_ctx_orig: int = 0, freq_base: float = 10000.0, freq_scale: float = 1.0) -> Tensor:
    nd = input.shape[-1] if n_dims is None else int(n_dims)
    return RoPEFn.apply(input, positions, nd, int(mode), int(n_ctx_orig), float(freq_base), float(freq_scale))


def swiglu(x: Tensor, up: Tensor) -> Tensor:
    return SwiGLUFn.apply(x, up)


# ---------- Loss Functions ----------

@_autocast_op(_AUTOCAST_FLOAT32)
def mse_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    diff = input - target
    return _reduce_loss(diff * diff, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def l1_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    return _reduce_loss((input - target).abs(), reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def smooth_l1_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0) -> Tensor:
    (size_average, reduce)
    diff = (input - target).abs()
    loss = where(diff < beta, 0.5 * (diff * diff) / beta, diff - 0.5 * beta)
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0) -> Tensor:
    diff = (input - target).abs()
    loss = where(diff < delta, 0.5 * (diff * diff), delta * (diff - 0.5 * delta))
    return _reduce_loss(loss, reduction)


def _class_loss_layout(name: str, input: Tensor, weight: Tensor | None):
    """What cross_entropy and nll_loss agree on: the class dimension, the class count, and the
    shape a tensor of class indices has to have."""
    if input.ndim < 1:
        raise RuntimeError(f"{name}: input needs a class dimension: (C), (N, C) or (N, C, d1, ...)")
    class_dim = 1 if input.ndim >= 2 else 0
    n_classes = input.shape[class_dim]
    if weight is not None and tuple(weight.shape) != (n_classes,):
        raise RuntimeError(
            f"{name}: weight tensor should be defined either for all {n_classes} classes or no classes "
            f"but got weight tensor of shape: {list(weight.shape)}"
        )
    index_shape = input.shape[:1] + input.shape[2:] if input.ndim >= 2 else ()
    return class_dim, n_classes, index_shape


def _class_index_target(name: str, input: Tensor, target: Tensor, index_shape: tuple) -> Tensor:
    """`target` as class indices of exactly `index_shape`. The same number of elements in another
    arrangement would pair every target with the wrong position."""
    if target.is_floating_point() or target.is_complex() or target.dtype == _dt.bool:
        raise RuntimeError(f"{name}: class indices must be an integer tensor, got {target.dtype}")
    if target.shape == index_shape:
        return target
    if input.ndim == 1 and target.numel() == 1:
        return target.reshape(())      # one item: (1,) says the same as ()
    raise RuntimeError(
        f"{name}: expected a target of shape {list(index_shape)} for an input of shape {list(input.shape)}, "
        f"got {list(target.shape)}"
    )


@_autocast_op(_AUTOCAST_FLOAT32)
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average=None,
    ignore_index: int = -100,
    reduce=None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    (size_average, reduce)
    red_map = {"none": 0, "mean": 1, "sum": 2}
    if reduction not in red_map:
        raise ValueError(f"Invalid reduction: {reduction!r}")
    if not 0.0 <= label_smoothing <= 1.0:
        raise RuntimeError(f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}")
    class_dim, n_classes, index_shape = _class_loss_layout("cross_entropy", input, weight)

    if target.is_floating_point():
        # class probabilities
        if target.shape != input.shape:
            raise RuntimeError(
                f"cross_entropy: a floating point target holds class probabilities and needs the shape of the input "
                f"{list(input.shape)}, got {list(target.shape)}; class indices must be an integer tensor"
            )
        log_probs = log_softmax(input, dim=class_dim)
        tgt = target
        if label_smoothing > 0.0:
            tgt = tgt * (1.0 - label_smoothing) + label_smoothing / float(n_classes)
        weighted = tgt * log_probs
        if weight is not None:
            w_sh = [1] * input.ndim
            w_sh[class_dim] = n_classes
            weighted = weighted * weight.reshape(tuple(w_sh))
        return _reduce_loss(-weighted.sum(dim=class_dim), reduction)

    target = _class_index_target("cross_entropy", input, target, index_shape)
    if input.ndim > 2:
        # Flatten (N, C, d1, ...) -> (N*d1*..., C)
        inp_flat = input.movedim(1, -1).reshape((-1, n_classes))
        tgt_flat = target.reshape((-1,))
        loss = CrossEntropyFn.apply(inp_flat, tgt_flat, weight, int(ignore_index), float(label_smoothing), red_map[reduction])
        if reduction == "none":
            return loss.reshape(target.shape)
        return loss
    return CrossEntropyFn.apply(input, target, weight, int(ignore_index), float(label_smoothing), red_map[reduction])


@_autocast_op(_AUTOCAST_FLOAT32)
def nll_loss(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Invalid reduction: {reduction!r}")
    _, n_classes, index_shape = _class_loss_layout("nll_loss", input, weight)
    target = _class_index_target("nll_loss", input, target, index_shape)
    # Gather -input[i, target[i]]
    if input.ndim > 2:
        inp = input.movedim(1, -1).reshape((-1, n_classes))
        tgt = target.reshape((-1,))
    elif input.ndim == 1:
        inp = input.unsqueeze(0)
        tgt = target.reshape((1,))
    else:
        inp = input
        tgt = target
    valid = tgt != ignore_index
    safe_tgt = where(valid, tgt, zeros_like(tgt))
    if safe_tgt.numel() > 0:
        lowest, highest = int(safe_tgt.min().item()), int(safe_tgt.max().item())
        if lowest < 0 or highest >= n_classes:
            # gather would count a negative class from the end
            raise RuntimeError(f"nll_loss: target {lowest if lowest < 0 else highest} is out of bounds for {n_classes} classes")
    safe_tgt = safe_tgt.unsqueeze(-1)
    picked = -inp.gather(1, safe_tgt).squeeze(-1)
    if weight is not None:
        w = weight.index_select(0, safe_tgt.squeeze(-1))
        picked = picked * w
        denom = where(valid, w, zeros_like(w)).sum()
    else:
        denom = valid.to(inp.dtype).sum()
    picked = where(valid, picked, zeros_like(picked))
    if reduction == "none":
        return picked.reshape(target.shape)
    if reduction == "sum":
        return picked.sum()
    # the weights of the targets that count; when none counts the sum above is 0 and so is the loss
    return picked.sum() / where(denom > 0, denom, zeros_like(denom) + 1.0)


@_autocast_op(_AUTOCAST_FLOAT32)
def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    eps = 1e-12
    x = input.clamp(min=eps, max=1.0 - eps)
    loss = -(target * x.log() + (1.0 - target) * (1.0 - x).log())
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    pos_weight: Tensor | None = None,
) -> Tensor:
    (size_average, reduce)
    # Numerically stable log-sum-exp formulation
    max_val = (-input).clamp(min=0.0)
    if pos_weight is None:
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_w = 1.0 + (pos_weight - 1.0) * target
        loss = (1.0 - target) * input + log_w * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def kl_div(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean", log_target: bool = False) -> Tensor:
    (size_average, reduce)
    if log_target:
        loss = target.exp() * (target - input)
    else:
        loss = where(target > 0, target * (target.clamp(min=1e-38).log() - input), zeros_like(input))
    if reduction == "batchmean":
        return loss.sum() / float(input.shape[0])
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean") -> Tensor:
    from .._ops import cosine_similarity
    cos = cosine_similarity(input1, input2, dim=1 if input1.ndim >= 2 else 0)
    loss = where(target == 1, 1.0 - cos, (cos - margin).clamp(min=0.0))
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean") -> Tensor:
    loss = (-target * (input1 - input2) + margin).clamp(min=0.0)
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, p: float = 2.0, eps: float = 1e-6, swap: bool = False, reduction: str = "mean") -> Tensor:
    d_ap = ((anchor - positive) ** 2 + eps).sum(dim=-1).sqrt() if p == 2.0 else (anchor - positive).norm(p=p, dim=-1)
    d_an = ((anchor - negative) ** 2 + eps).sum(dim=-1).sqrt() if p == 2.0 else (anchor - negative).norm(p=p, dim=-1)
    if swap:
        d_pn = ((positive - negative) ** 2 + eps).sum(dim=-1).sqrt() if p == 2.0 else (positive - negative).norm(p=p, dim=-1)
        d_an = d_an.minimum(d_pn)
    loss = (d_ap - d_an + margin).clamp(min=0.0)
    return _reduce_loss(loss, reduction)


@_autocast_op(_AUTOCAST_FLOAT32)
def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    from .._ops import cosine_similarity as _cs
    return _cs(x1, x2, dim=dim, eps=eps)


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> Tensor:
    return ((x1 - x2) + eps).norm(p=p, dim=-1, keepdim=keepdim)
