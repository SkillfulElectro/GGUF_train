"""Neural network functional interface compatible with torch.nn.functional."""

from __future__ import annotations

import math

from .. import _C, _autograd_state, _dtype as _dt
from .._creation import ones, zeros, zeros_like
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
from .._tensor import Tensor


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _triple(v) -> tuple[int, int, int]:
    return (v, v, v) if isinstance(v, int) else (int(v[0]), int(v[1]), int(v[2]))


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
    x = input if dtype is None else input.to(dtype)
    return SoftmaxFn.apply(x, d, 0)


def softmin(input: Tensor, dim: int | None = None, dtype=None) -> Tensor:
    return softmax(-input, dim=dim, dtype=dtype)


def log_softmax(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    (_stacklevel,)
    d = -1 if dim is None else dim
    x = input if dtype is None else input.to(dtype)
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

def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return LinearFn.apply(input, weight, bias)


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


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    if num_classes < 0:
        num_classes = int(tensor.max().item()) + 1 if tensor.numel() > 0 else 1
    out_sh = tensor.shape + (num_classes,)
    idx = tensor.long().unsqueeze(-1)
    out = zeros(out_sh, dtype=_dt.int64, device=tensor.device)
    src = ones(idx.shape, dtype=_dt.int64, device=tensor.device)
    return out.scatter(-1, idx, src)


# ---------- Normalization ----------

def rms_norm(input: Tensor, normalized_shape, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    (normalized_shape,)
    return RMSNormFn.apply(input, weight, float(eps))


def layer_norm(
    input: Tensor,
    normalized_shape,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    norm_ndim = 1 if isinstance(normalized_shape, int) else len(normalized_shape)
    return LayerNormFn.apply(input, norm_ndim, weight, bias, float(eps))


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


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out=None) -> Tensor:
    denom = input.norm(p=p, dim=dim, keepdim=True).clamp(min=eps)
    res = input / denom
    if out is not None:
        out.copy_(res)
        return out
    return res


# ---------- Convolutions & Pooling ----------

def conv1d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Tensor:
    s = stride[0] if isinstance(stride, (tuple, list)) else int(stride)
    p = padding[0] if isinstance(padding, (tuple, list)) else int(padding)
    d = dilation[0] if isinstance(dilation, (tuple, list)) else int(dilation)
    return Conv1dFn.apply(input, weight, bias, s, p, d, int(groups))


def conv2d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups: int = 1) -> Tensor:
    return Conv2dFn.apply(input, weight, bias, _pair(stride), _pair(padding), _pair(dilation), int(groups))


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


def adaptive_avg_pool2d(input: Tensor, output_size) -> Tensor:
    oh, ow = _pair(output_size)
    ih, iw = input.shape[-2], input.shape[-1]
    if oh == 1 and ow == 1:
        return input.mean(dim=(-2, -1), keepdim=True)
    if ih % oh == 0 and iw % ow == 0:
        sh, sw = ih // oh, iw // ow
        return avg_pool2d(input, (sh, sw), stride=(sh, sw))
    return Upsample2dFn.apply(input, (oh, ow), 1, 0)


def adaptive_avg_pool1d(input: Tensor, output_size: int) -> Tensor:
    return adaptive_avg_pool2d(input.unsqueeze(-2), (1, int(output_size))).squeeze(-2)


def adaptive_max_pool2d(input: Tensor, output_size, return_indices: bool = False):
    oh, ow = _pair(output_size)
    ih, iw = input.shape[-2], input.shape[-1]
    sh, sw = max(1, ih // oh), max(1, iw // ow)
    kh = ih - (oh - 1) * sh
    kw = iw - (ow - 1) * sw
    return max_pool2d(input, (kh, kw), stride=(sh, sw), return_indices=return_indices)


def interpolate(input: Tensor, size=None, scale_factor=None, mode: str = "nearest", align_corners: bool | None = None, recompute_scale_factor=None, antialias: bool = False) -> Tensor:
    (recompute_scale_factor, antialias)
    if size is not None:
        oh, ow = _pair(size)
    elif scale_factor is not None:
        sf_h, sf_w = (float(scale_factor), float(scale_factor)) if isinstance(scale_factor, (int, float)) else (float(scale_factor[0]), float(scale_factor[1]))
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

def mse_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    diff = input - target
    return _reduce_loss(diff * diff, reduction)


def l1_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    return _reduce_loss((input - target).abs(), reduction)


def smooth_l1_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0) -> Tensor:
    (size_average, reduce)
    diff = (input - target).abs()
    loss = where(diff < beta, 0.5 * (diff * diff) / beta, diff - 0.5 * beta)
    return _reduce_loss(loss, reduction)


def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0) -> Tensor:
    diff = (input - target).abs()
    loss = where(diff < delta, 0.5 * (diff * diff), delta * (diff - 0.5 * delta))
    return _reduce_loss(loss, reduction)


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
    if target.is_floating_point() and target.shape == input.shape:
        log_probs = log_softmax(input, dim=1 if input.ndim >= 2 else 0)
        tgt = target
        if label_smoothing > 0.0:
            c = input.shape[1 if input.ndim >= 2 else 0]
            tgt = tgt * (1.0 - label_smoothing) + label_smoothing / float(c)
        if weight is not None and input.ndim >= 2:
            w_sh = [1] * input.ndim
            w_sh[1] = weight.shape[0]
            per_sample = -(tgt * log_probs * weight.reshape(tuple(w_sh))).sum(dim=1)
        else:
            per_sample = -(tgt * log_probs).sum(dim=1 if input.ndim >= 2 else 0)
        return _reduce_loss(per_sample, reduction)

    red_map = {"none": 0, "mean": 1, "sum": 2}
    if reduction not in red_map:
        raise ValueError(f"Invalid reduction: {reduction!r}")
    if input.ndim > 2:
        # Flatten (N, C, d1, ...) -> (N*d1*..., C)
        C = input.shape[1]
        inp_flat = input.movedim(1, -1).reshape((-1, C))
        tgt_flat = target.reshape((-1,))
        loss = CrossEntropyFn.apply(inp_flat, tgt_flat, weight, int(ignore_index), float(label_smoothing), red_map[reduction])
        if reduction == "none":
            return loss.reshape(target.shape)
        return loss
    return CrossEntropyFn.apply(input, target, weight, int(ignore_index), float(label_smoothing), red_map[reduction])


def nll_loss(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    # Gather -input[i, target[i]]
    if input.ndim > 2:
        C = input.shape[1]
        inp = input.movedim(1, -1).reshape((-1, C))
        tgt = target.reshape((-1,))
    else:
        inp = input
        tgt = target
    valid = tgt != ignore_index
    safe_tgt = where(valid, tgt, zeros_like(tgt)).unsqueeze(-1)
    picked = -inp.gather(1, safe_tgt).squeeze(-1)
    if weight is not None:
        w = weight.index_select(0, safe_tgt.squeeze(-1))
        picked = picked * w
        denom = where(valid, w, zeros_like(w)).sum()
    else:
        denom = valid.to(inp.dtype).sum()
    picked = where(valid, picked, zeros_like(picked))
    if reduction == "none":
        return picked.reshape(target.shape) if input.ndim > 2 else picked
    if reduction == "sum":
        return picked.sum()
    return picked.sum() / denom.clamp(min=1.0)


def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor | None = None, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    (size_average, reduce)
    eps = 1e-12
    x = input.clamp(min=eps, max=1.0 - eps)
    loss = -(target * x.log() + (1.0 - target) * (1.0 - x).log())
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


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


def kl_div(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean", log_target: bool = False) -> Tensor:
    (size_average, reduce)
    if log_target:
        loss = target.exp() * (target - input)
    else:
        loss = where(target > 0, target * (target.clamp(min=1e-38).log() - input), zeros_like(input))
    if reduction == "batchmean":
        return loss.sum() / float(input.shape[0])
    return _reduce_loss(loss, reduction)


def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean") -> Tensor:
    from .._ops import cosine_similarity
    cos = cosine_similarity(input1, input2, dim=1 if input1.ndim >= 2 else 0)
    loss = where(target == 1, 1.0 - cos, (cos - margin).clamp(min=0.0))
    return _reduce_loss(loss, reduction)


def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean") -> Tensor:
    loss = (-target * (input1 - input2) + margin).clamp(min=0.0)
    return _reduce_loss(loss, reduction)


def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, p: float = 2.0, eps: float = 1e-6, swap: bool = False, reduction: str = "mean") -> Tensor:
    d_ap = ((anchor - positive) ** 2 + eps).sum(dim=-1).sqrt() if p == 2.0 else (anchor - positive).norm(p=p, dim=-1)
    d_an = ((anchor - negative) ** 2 + eps).sum(dim=-1).sqrt() if p == 2.0 else (anchor - negative).norm(p=p, dim=-1)
    if swap:
        d_pn = ((positive - negative) ** 2 + eps).sum(dim=-1).sqrt() if p == 2.0 else (positive - negative).norm(p=p, dim=-1)
        d_an = d_an.minimum(d_pn)
    loss = (d_ap - d_an + margin).clamp(min=0.0)
    return _reduce_loss(loss, reduction)


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    from .._ops import cosine_similarity as _cs
    return _cs(x1, x2, dim=dim, eps=eps)


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> Tensor:
    return ((x1 - x2) + eps).norm(p=p, dim=-1, keepdim=keepdim)
