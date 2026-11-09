"""Built-in differentiable autograd Functions backed by native ggml / C++ kernels."""

from __future__ import annotations

import ctypes
import math

from . import _C, _autograd_state, _dtype as _dt
from ._autograd_engine import Context, Function, _unbroadcast


def _c_i64_arr(vals):
    n = len(vals)
    return (ctypes.c_int64 * max(n, 1))(*vals) if n else None


def _c_i32_arr(vals):
    n = len(vals)
    return (ctypes.c_int * max(n, 1))(*vals) if n else None


def _wrap(ptr: int, ctx: str = ""):
    from ._tensor import Tensor

    return Tensor(_ptr=_C.check_ptr(ptr, ctx), requires_grad=False)


def _compute_view_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * max(shape[i + 1], 1)
    return tuple(strides)


def _can_view_reshape(old_shape: tuple[int, ...], old_strides: tuple[int, ...], new_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if not old_shape:
        return _compute_view_strides(new_shape)
    # Collapse contiguous chunks (ignoring size-1 dims)
    chunks: list[tuple[int, int]] = []  # (size, stride)
    for s, st in zip(old_shape, old_strides):
        if s == 1:
            continue
        if not chunks:
            chunks.append((s, st))
        else:
            prev_s, prev_st = chunks[-1]
            if prev_st == s * st:
                chunks[-1] = (prev_s * s, st)
            else:
                chunks.append((s, st))

    new_strides = [1] * len(new_shape)
    if not chunks:
        return _compute_view_strides(new_shape)

    chunk_idx = 0
    cur_size, cur_stride = chunks[0]
    i = 0
    while i < len(new_shape):
        ns = new_shape[i]
        if ns == 1:
            i += 1
            continue
        if chunk_idx >= len(chunks):
            return None
        # Collect dims in new_shape multiplying up to cur_size
        prod = 1
        j = i
        while j < len(new_shape) and prod < cur_size:
            prod *= new_shape[j]
            j += 1
        if prod != cur_size:
            return None
        st = cur_stride
        for k in range(j - 1, i - 1, -1):
            new_strides[k] = st
            st *= new_shape[k]
        i = j
        chunk_idx += 1
        if chunk_idx < len(chunks):
            cur_size, cur_stride = chunks[chunk_idx]

    # Set strides for trailing size-1 dims
    for k in range(len(new_shape) - 1, -1, -1):
        if new_shape[k] == 1:
            new_strides[k] = new_strides[k + 1] * max(new_shape[k + 1], 1) if k + 1 < len(new_shape) else 1
    return tuple(new_strides)


# ==================== View / Shape Functions ====================


class ReshapeFn(Function):
    @staticmethod
    def forward(ctx: Context, x, new_shape: tuple[int, ...]):
        ctx.orig_shape = x.shape
        new_strides = _can_view_reshape(x.shape, x.stride(), new_shape)
        if new_strides is not None:
            nd = len(new_shape)
            ptr = _C.lib.gt_tensor_view(
                x._ptr, nd, _c_i64_arr(new_shape), _c_i64_arr(new_strides), x.storage_offset()
            )
            return _wrap(ptr, "reshape view")
        xc = x.contiguous()
        nd = len(new_shape)
        ns = _compute_view_strides(new_shape)
        ptr = _C.lib.gt_tensor_view(xc._ptr, nd, _c_i64_arr(new_shape), _c_i64_arr(ns), 0)
        return _wrap(ptr, "reshape")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        return (grad_out.reshape(ctx.orig_shape),)


class TransposeFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim0: int, dim1: int):
        nd = x.ndim
        d0 = dim0 if dim0 >= 0 else dim0 + nd
        d1 = dim1 if dim1 >= 0 else dim1 + nd
        if d0 < 0 or d0 >= nd or d1 < 0 or d1 >= nd:
            raise IndexError(f"Dimension out of range (expected to be in [{-nd}, {nd - 1}], got {dim0}, {dim1})")
        ctx.dims = (d0, d1)
        sh = list(x.shape)
        st = list(x.stride())
        sh[d0], sh[d1] = sh[d1], sh[d0]
        st[d0], st[d1] = st[d1], st[d0]
        ptr = _C.lib.gt_tensor_view(x._ptr, nd, _c_i64_arr(sh), _c_i64_arr(st), x.storage_offset())
        return _wrap(ptr, "transpose")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        d0, d1 = ctx.dims
        return (grad_out.transpose(d0, d1),)


class PermuteFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dims: tuple[int, ...]):
        nd = x.ndim
        norm = tuple(d if d >= 0 else d + nd for d in dims)
        if len(norm) != nd or sorted(norm) != list(range(nd)):
            raise RuntimeError(f"Invalid permute dims {dims} for shape {x.shape}")
        inv = [0] * nd
        for i, d in enumerate(norm):
            inv[d] = i
        ctx.inv_dims = tuple(inv)
        old_sh = x.shape
        old_st = x.stride()
        sh = [old_sh[d] for d in norm]
        st = [old_st[d] for d in norm]
        ptr = _C.lib.gt_tensor_view(x._ptr, nd, _c_i64_arr(sh), _c_i64_arr(st), x.storage_offset())
        return _wrap(ptr, "permute")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        return (grad_out.permute(ctx.inv_dims),)


class ExpandFn(Function):
    @staticmethod
    def forward(ctx: Context, x, new_shape: tuple[int, ...]):
        ctx.orig_shape = x.shape
        nd_old = x.ndim
        nd_new = len(new_shape)
        if nd_new < nd_old:
            raise RuntimeError(f"Number of dims in expand {nd_new} must be >= {nd_old}")
        old_sh = (1,) * (nd_new - nd_old) + x.shape
        old_st = (0,) * (nd_new - nd_old) + x.stride()
        final_sh = []
        final_st = []
        for idx, (os, ost, ns) in enumerate(zip(old_sh, old_st, new_shape)):
            target = os if ns == -1 else ns
            if os == target:
                final_sh.append(target)
                final_st.append(ost)
            elif os == 1:
                final_sh.append(target)
                final_st.append(0)
            else:
                raise RuntimeError(
                    f"The expanded size of the tensor ({target}) must match the existing size ({os}) at non-singleton dimension {idx}."
                )
        ptr = _C.lib.gt_tensor_view(x._ptr, nd_new, _c_i64_arr(final_sh), _c_i64_arr(final_st), x.storage_offset())
        return _wrap(ptr, "expand")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        return (_unbroadcast(grad_out, ctx.orig_shape),)


class SliceScatterFn(Function):
    @staticmethod
    def forward(ctx: Context, grad_out, orig_shape: tuple[int, ...], orig_dtype, orig_device, index_spec):
        from ._creation import zeros

        ctx.index_spec = index_spec
        g_in = zeros(orig_shape, dtype=orig_dtype, device=orig_device)
        g_view = g_in[index_spec]
        g_src = grad_out.expand(g_view.shape) if grad_out.shape != g_view.shape else grad_out
        _C.check_rc(_C.lib.gt_tensor_copy_inplace(g_view._ptr, g_src._ptr))
        return g_in

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None, None)
        return (grad_out[ctx.index_spec], None, None, None, None)


class SliceViewFn(Function):
    @staticmethod
    def forward(ctx: Context, x, new_shape: tuple[int, ...], new_strides: tuple[int, ...], new_offset: int, index_spec):
        ctx.orig_shape = x.shape
        ctx.index_spec = index_spec
        ctx.orig_dtype = x.dtype
        ctx.orig_device = x.device
        ptr = _C.lib.gt_tensor_view(x._ptr, len(new_shape), _c_i64_arr(new_shape), _c_i64_arr(new_strides), new_offset)
        return _wrap(ptr, "slice view")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None, None)
        return (SliceScatterFn.apply(grad_out, ctx.orig_shape, ctx.orig_dtype, ctx.orig_device, ctx.index_spec), None, None, None, None)


class CastFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dtype: _dt.dtype):
        ctx.orig_dtype = x.dtype
        if x.dtype == dtype:
            ptr = _C.lib.gt_tensor_view(x._ptr, x.ndim, _c_i64_arr(x.shape), _c_i64_arr(x.stride()), x.storage_offset())
            return _wrap(ptr, "cast noop")
        ptr = _C.lib.gt_tensor_to_dtype(x._ptr, dtype.code)
        return _wrap(ptr, "to dtype")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        if ctx.orig_dtype.is_floating_point or ctx.orig_dtype.is_complex:
            return (grad_out.to(ctx.orig_dtype),)
        return (None,)


class ToDeviceFn(Function):
    @staticmethod
    def forward(ctx: Context, x, device):
        ctx.orig_device = x.device
        if x.device == device:
            ptr = _C.lib.gt_tensor_view(x._ptr, x.ndim, _c_i64_arr(x.shape), _c_i64_arr(x.stride()), x.storage_offset())
            return _wrap(ptr, "to device noop")
        ptr = _C.lib.gt_tensor_to_device(x._ptr, device.dev_type, device.index)
        return _wrap(ptr, "to device")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        return (grad_out.to(ctx.orig_device),)


class CloneFn(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ptr = _C.lib.gt_tensor_clone(x._ptr)
        return _wrap(ptr, "clone")

    @staticmethod
    def backward(ctx: Context, grad_out):
        return (grad_out,)


class ContiguousFn(Function):
    @staticmethod
    def forward(ctx: Context, x):
        if x.is_contiguous() and x.storage_offset() == 0:
            ptr = _C.lib.gt_tensor_view(x._ptr, x.ndim, _c_i64_arr(x.shape), _c_i64_arr(x.stride()), x.storage_offset())
            return _wrap(ptr, "contiguous view")
        ptr = _C.lib.gt_tensor_contiguous(x._ptr)
        return _wrap(ptr, "contiguous")

    @staticmethod
    def backward(ctx: Context, grad_out):
        return (grad_out,)


class CatFn(Function):
    @staticmethod
    def forward(ctx: Context, dim: int, *tensors):
        nd = tensors[0].ndim
        d = dim if dim >= 0 else dim + nd
        ctx.dim = d
        ctx.sizes = [t.shape[d] for t in tensors]
        ptrs = (ctypes.c_void_p * len(tensors))(*[t._ptr for t in tensors])
        out_ptr = _C.lib.gt_cat(len(tensors), ptrs, d)
        return _wrap(out_ptr, "cat")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * (1 + len(ctx.sizes))
        chunks = grad_out.split(ctx.sizes, dim=ctx.dim)
        return (None,) + tuple(chunks)


# ==================== Unary Functions ====================


class UnarySimpleFn(Function):
    """Differentiable unary ops dispatching via gt_unary_op."""

    @staticmethod
    def forward(ctx: Context, x, op_code: int, alpha: float = 0.0, beta: float = 0.0):
        ctx.op_code = op_code
        ctx.alpha = alpha
        ctx.beta = beta
        out = _wrap(_C.lib.gt_unary_op(x._ptr, op_code, alpha, beta), "unary")
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        x, y = ctx.saved_tensors
        op = ctx.op_code
        a, b = ctx.alpha, ctx.beta
        from ._tensor import Tensor

        if op == _C.UNARY_NEG:
            gx = -grad_out
        elif op == _C.UNARY_ABS:
            gx = grad_out * x.sign()
        elif op == _C.UNARY_SQRT:
            gx = grad_out / (2.0 * y)
        elif op == _C.UNARY_RSQRT:
            gx = grad_out * (-0.5) * (y * y * y)
        elif op == _C.UNARY_SQUARE:
            gx = grad_out * (2.0 * x)
        elif op == _C.UNARY_EXP:
            gx = grad_out * y
        elif op == _C.UNARY_EXPM1:
            gx = grad_out * (y + 1.0)
        elif op == _C.UNARY_EXP2:
            gx = grad_out * (y * math.log(2.0))
        elif op == _C.UNARY_LOG:
            gx = grad_out / x
        elif op == _C.UNARY_LOG2:
            gx = grad_out / (x * math.log(2.0))
        elif op == _C.UNARY_LOG10:
            gx = grad_out / (x * math.log(10.0))
        elif op == _C.UNARY_LOG1P:
            gx = grad_out / (x + 1.0)
        elif op == _C.UNARY_SIN:
            gx = grad_out * x.cos()
        elif op == _C.UNARY_COS:
            gx = -grad_out * x.sin()
        elif op == _C.UNARY_TAN:
            c = x.cos()
            gx = grad_out / (c * c)
        elif op == _C.UNARY_ASIN:
            gx = grad_out / (1.0 - x * x).sqrt()
        elif op == _C.UNARY_ACOS:
            gx = -grad_out / (1.0 - x * x).sqrt()
        elif op == _C.UNARY_ATAN:
            gx = grad_out / (1.0 + x * x)
        elif op == _C.UNARY_SINH:
            gx = grad_out * x.cosh()
        elif op == _C.UNARY_COSH:
            gx = grad_out * x.sinh()
        elif op == _C.UNARY_TANH:
            gx = grad_out * (1.0 - y * y)
        elif op == _C.UNARY_ASINH:
            gx = grad_out / (x * x + 1.0).sqrt()
        elif op == _C.UNARY_ACOSH:
            gx = grad_out / (x * x - 1.0).sqrt()
        elif op == _C.UNARY_ATANH:
            gx = grad_out / (1.0 - x * x)
        elif op == _C.UNARY_ERF:
            gx = grad_out * ((2.0 / math.sqrt(math.pi)) * (-(x * x)).exp())
        elif op == _C.UNARY_ERFC:
            gx = grad_out * ((-2.0 / math.sqrt(math.pi)) * (-(x * x)).exp())
        elif op == _C.UNARY_RECIPROCAL:
            gx = -grad_out / (x * x)
        elif op == _C.UNARY_RELU:
            gx = grad_out * (x > 0).to(grad_out.dtype)
        elif op == _C.UNARY_RELU6:
            mask = ((x > 0) & (x < 6)).to(grad_out.dtype)
            gx = grad_out * mask
        elif op == _C.UNARY_SIGMOID:
            gx = grad_out * y * (1.0 - y)
        elif op == _C.UNARY_SILU:
            if _autograd_state.is_grad_enabled():
                s = x.sigmoid()
                gx = grad_out * (s * (1.0 + x * (1.0 - s)))
            else:
                deriv = _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_SILU_DERIV, 0.0, 0.0), "silu_deriv")
                gx = grad_out * deriv
        elif op == _C.UNARY_GELU:
            if _autograd_state.is_grad_enabled():
                cdf = 0.5 * (1.0 + (x * (1.0 / math.sqrt(2.0))).erf())
                pdf = (1.0 / math.sqrt(2.0 * math.pi)) * (-0.5 * x * x).exp()
                gx = grad_out * (cdf + x * pdf)
            else:
                deriv = _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_GELU_DERIV, 0.0, 0.0), "gelu_deriv")
                gx = grad_out * deriv
        elif op == _C.UNARY_GELU_TANH:
            deriv = _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_GELU_TANH_DERIV, 0.0, 0.0), "gelu_tanh_deriv")
            gx = grad_out * deriv
        elif op == _C.UNARY_GELU_QUICK:
            s = (1.702 * x).sigmoid()
            gx = grad_out * (s + 1.702 * x * s * (1.0 - s))
        elif op == _C.UNARY_LEAKY_RELU:
            mask = (x > 0).to(grad_out.dtype)
            slope = mask + (1.0 - mask) * a
            gx = grad_out * slope
        elif op == _C.UNARY_ELU:
            mask = (x > 0).to(grad_out.dtype)
            gx = grad_out * (mask + (1.0 - mask) * (y + a))
        elif op == _C.UNARY_CELU:
            mask = (x > 0).to(grad_out.dtype)
            gx = grad_out * (mask + (1.0 - mask) * (x / a).exp())
        elif op == _C.UNARY_SELU:
            selu_alpha = 1.6732632423543772848170429916717
            selu_scale = 1.0507009873554804934193349852946
            mask = (x > 0).to(grad_out.dtype)
            gx = grad_out * (mask * selu_scale + (1.0 - mask) * (selu_scale * selu_alpha * x.exp()))
        elif op == _C.UNARY_SOFTPLUS:
            beta_v = a if a > 0 else 1.0
            thresh = b if b > 0 else 20.0
            bx = x * beta_v
            s = bx.sigmoid()
            mask = (bx > thresh).to(grad_out.dtype)
            gx = grad_out * (mask + (1.0 - mask) * s)
        elif op == _C.UNARY_MISH:
            if _autograd_state.is_grad_enabled():
                sp = (x.exp() + 1.0).log()
                th = sp.tanh()
                sig = x.sigmoid()
                gx = grad_out * (th + x * sig * (1.0 - th * th))
            else:
                deriv = _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_MISH_DERIV, 0.0, 0.0), "mish_deriv")
                gx = grad_out * deriv
        elif op == _C.UNARY_HARDSWISH:
            m1 = ((x > -3.0) & (x < 3.0)).to(grad_out.dtype)
            m2 = (x >= 3.0).to(grad_out.dtype)
            gx = grad_out * (m2 + m1 * ((2.0 * x + 3.0) / 6.0))
        elif op == _C.UNARY_HARDSIGMOID:
            m = ((x > -3.0) & (x < 3.0)).to(grad_out.dtype)
            gx = grad_out * (m * (1.0 / 6.0))
        elif op == _C.UNARY_HARDTANH:
            m = ((x > a) & (x < b)).to(grad_out.dtype)
            gx = grad_out * m
        elif op == _C.UNARY_LOG_SIGMOID:
            gx = grad_out * (1.0 - x.sigmoid())
        elif op == _C.UNARY_CLAMP:
            has_lo = not math.isnan(a)
            has_hi = not math.isnan(b)
            m = None
            if has_lo:
                m = x >= a
            if has_hi:
                m2 = x <= b
                m = m2 if m is None else (m & m2)
            gx = grad_out if m is None else (grad_out * m.to(grad_out.dtype))
        elif op in (_C.UNARY_FLOOR, _C.UNARY_CEIL, _C.UNARY_ROUND, _C.UNARY_TRUNC, _C.UNARY_SIGN):
            gx = x * 0.0
        elif op == _C.UNARY_FRAC:
            gx = grad_out
        else:
            raise NotImplementedError(f"Backward not implemented for unary op {op}")
        return (gx, None, None, None)


# ==================== Binary Functions ====================


class AddFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b, alpha: float = 1.0):
        ctx.shapes = (a.shape, b.shape)
        ctx.alpha = alpha
        return _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_ADD, alpha), "add")

    @staticmethod
    def backward(ctx: Context, grad_out):
        sa, sb = ctx.shapes
        ga = _unbroadcast(grad_out, sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast(grad_out * ctx.alpha if ctx.alpha != 1.0 else grad_out, sb) if ctx.needs_input_grad[1] else None
        return (ga, gb, None)


class SubFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b, alpha: float = 1.0):
        ctx.shapes = (a.shape, b.shape)
        ctx.alpha = alpha
        return _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_SUB, alpha), "sub")

    @staticmethod
    def backward(ctx: Context, grad_out):
        sa, sb = ctx.shapes
        ga = _unbroadcast(grad_out, sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast((-grad_out) * ctx.alpha if ctx.alpha != 1.0 else -grad_out, sb) if ctx.needs_input_grad[1] else None
        return (ga, gb, None)


class MulFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.shapes = (a.shape, b.shape)
        return _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_MUL, 1.0), "mul")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_tensors
        sa, sb = ctx.shapes
        ga = _unbroadcast(grad_out * b, sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast(grad_out * a, sb) if ctx.needs_input_grad[1] else None
        return (ga, gb)


class DivFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.shapes = (a.shape, b.shape)
        return _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_DIV, 1.0), "div")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_tensors
        sa, sb = ctx.shapes
        ga = _unbroadcast(grad_out / b, sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast(-grad_out * a / (b * b), sb) if ctx.needs_input_grad[1] else None
        return (ga, gb)


class PowFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        out = _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_POW, 1.0), "pow")
        ctx.save_for_backward(a, b, out)
        ctx.shapes = (a.shape, b.shape)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b, out = ctx.saved_tensors
        sa, sb = ctx.shapes
        ga = _unbroadcast(grad_out * b * (a ** (b - 1.0)), sa) if ctx.needs_input_grad[0] else None
        gb = None
        if ctx.needs_input_grad[1]:
            from ._ops import where
            safe_a = where(a > 0, a, a.new_ones(a.shape))
            gb = _unbroadcast(grad_out * out * safe_a.log(), sb)
        return (ga, gb)


class BinaryScalarFn(Function):
    @staticmethod
    def forward(ctx: Context, a, scalar: float, op_code: int, reverse: int = 0):
        out = _wrap(_C.lib.gt_binary_scalar_op(a._ptr, float(scalar), int(reverse), int(op_code)), "binary_scalar")
        ctx.save_for_backward(a, out)
        ctx.scalar = float(scalar)
        ctx.op_code = op_code
        ctx.reverse = reverse
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        a, out = ctx.saved_tensors
        s = ctx.scalar
        op = ctx.op_code
        rev = ctx.reverse
        if op == _C.BIN_ADD:
            ga = grad_out
        elif op == _C.BIN_SUB:
            ga = -grad_out if rev else grad_out
        elif op == _C.BIN_MUL:
            ga = grad_out * s
        elif op == _C.BIN_DIV:
            if rev:
                ga = -grad_out * s / (a * a)
            else:
                ga = grad_out / s
        elif op == _C.BIN_POW:
            if rev:
                ga = grad_out * out * math.log(max(s, 1e-38))
            else:
                if s == 0.0:
                    ga = a * 0.0
                else:
                    ga = grad_out * s * (a ** (s - 1.0))
        else:
            ga = None
        return (ga, None, None, None)


class Atan2Fn(Function):
    @staticmethod
    def forward(ctx: Context, y, x):
        ctx.save_for_backward(y, x)
        ctx.shapes = (y.shape, x.shape)
        return _wrap(_C.lib.gt_binary_op(y._ptr, x._ptr, _C.BIN_ATAN2, 1.0), "atan2")

    @staticmethod
    def backward(ctx: Context, grad_out):
        y, x = ctx.saved_tensors
        sy, sx = ctx.shapes
        denom = x * x + y * y
        gy = _unbroadcast(grad_out * x / denom, sy) if ctx.needs_input_grad[0] else None
        gx = _unbroadcast(-grad_out * y / denom, sx) if ctx.needs_input_grad[1] else None
        return (gy, gx)


class MaximumFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.shapes = (a.shape, b.shape)
        return _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_MAXIMUM, 1.0), "maximum")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_tensors
        sa, sb = ctx.shapes
        gt_m = (a > b).to(grad_out.dtype)
        eq_m = (a == b).to(grad_out.dtype) * 0.5
        wa = gt_m + eq_m
        wb = 1.0 - wa
        ga = _unbroadcast(grad_out * wa, sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast(grad_out * wb, sb) if ctx.needs_input_grad[1] else None
        return (ga, gb)


class MinimumFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.shapes = (a.shape, b.shape)
        return _wrap(_C.lib.gt_binary_op(a._ptr, b._ptr, _C.BIN_MINIMUM, 1.0), "minimum")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_tensors
        sa, sb = ctx.shapes
        lt_m = (a < b).to(grad_out.dtype)
        eq_m = (a == b).to(grad_out.dtype) * 0.5
        wa = lt_m + eq_m
        wb = 1.0 - wa
        ga = _unbroadcast(grad_out * wa, sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast(grad_out * wb, sb) if ctx.needs_input_grad[1] else None
        return (ga, gb)


class WhereFn(Function):
    @staticmethod
    def forward(ctx: Context, cond, x, y):
        ctx.save_for_backward(cond)
        ctx.shapes = (x.shape, y.shape)
        return _wrap(_C.lib.gt_where(cond._ptr, x._ptr, y._ptr), "where")

    @staticmethod
    def backward(ctx: Context, grad_out):
        (cond,) = ctx.saved_tensors
        sx, sy = ctx.shapes
        c = cond.to(grad_out.dtype)
        gx = _unbroadcast(grad_out * c, sx) if ctx.needs_input_grad[1] else None
        gy = _unbroadcast(grad_out * (1.0 - c), sy) if ctx.needs_input_grad[2] else None
        return (None, gx, gy)


class LerpFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b, w):
        ctx.save_for_backward(a, b, w)
        ctx.shapes = (a.shape, b.shape, w.shape)
        return _wrap(_C.lib.gt_lerp(a._ptr, b._ptr, w._ptr), "lerp")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b, w = ctx.saved_tensors
        sa, sb, sw = ctx.shapes
        ga = _unbroadcast(grad_out * (1.0 - w), sa) if ctx.needs_input_grad[0] else None
        gb = _unbroadcast(grad_out * w, sb) if ctx.needs_input_grad[1] else None
        gw = _unbroadcast(grad_out * (b - a), sw) if ctx.needs_input_grad[2] else None
        return (ga, gb, gw)


class ClampTensorFn(Function):
    @staticmethod
    def forward(ctx: Context, x, min_t, max_t, min_f: float | None, max_f: float | None):
        ctx.save_for_backward(x, min_t, max_t)
        ctx.min_f = min_f
        ctx.max_f = max_f
        ptr = _C.lib.gt_clamp_tensor(
            x._ptr,
            min_t._ptr if min_t is not None else 0,
            max_t._ptr if max_t is not None else 0,
            1 if min_f is not None else 0,
            1 if max_f is not None else 0,
            float(min_f) if min_f is not None else 0.0,
            float(max_f) if max_f is not None else 0.0,
        )
        return _wrap(ptr, "clamp")

    @staticmethod
    def backward(ctx: Context, grad_out):
        x, min_t, max_t = ctx.saved_tensors
        m_lo = None
        m_hi = None
        if min_t is not None:
            m_lo = x >= min_t
        elif ctx.min_f is not None:
            m_lo = x >= ctx.min_f
        if max_t is not None:
            m_hi = x <= max_t
        elif ctx.max_f is not None:
            m_hi = x <= ctx.max_f
        mask = None
        if m_lo is not None and m_hi is not None:
            mask = m_lo & m_hi
        elif m_lo is not None:
            mask = m_lo
        elif m_hi is not None:
            mask = m_hi
        gx = grad_out if mask is None else grad_out * mask.to(grad_out.dtype)
        return (gx, None, None, None, None)


# ==================== MatMul / Linear / Addmm / Outer ====================


class MatMulFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return _wrap(_C.lib.gt_matmul(a._ptr, b._ptr), "matmul")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_tensors
        ga = None
        gb = None
        if a.ndim == 1 and b.ndim == 1:
            if ctx.needs_input_grad[0]:
                ga = grad_out * b
            if ctx.needs_input_grad[1]:
                gb = grad_out * a
            return (ga, gb)
        if a.ndim == 1 and b.ndim >= 2:
            a2 = a.unsqueeze(0)
            g2 = grad_out.unsqueeze(-2)
            if ctx.needs_input_grad[0]:
                ga_raw = g2.matmul(b.transpose(-2, -1))
                ga = _unbroadcast(ga_raw.squeeze(-2), a.shape)
            if ctx.needs_input_grad[1]:
                gb = _unbroadcast(a2.transpose(-2, -1).matmul(g2), b.shape)
            return (ga, gb)
        if a.ndim >= 2 and b.ndim == 1:
            b2 = b.unsqueeze(-1)
            g2 = grad_out.unsqueeze(-1)
            if ctx.needs_input_grad[0]:
                ga = _unbroadcast(g2.matmul(b2.transpose(-2, -1)), a.shape)
            if ctx.needs_input_grad[1]:
                gb_raw = a.transpose(-2, -1).matmul(g2)
                gb = _unbroadcast(gb_raw.squeeze(-1), b.shape)
            return (ga, gb)
        if ctx.needs_input_grad[0]:
            ga = _unbroadcast(grad_out.matmul(b.transpose(-2, -1)), a.shape)
        if ctx.needs_input_grad[1]:
            gb = _unbroadcast(a.transpose(-2, -1).matmul(grad_out), b.shape)
        return (ga, gb)


class LinearFn(Function):
    @staticmethod
    def forward(ctx: Context, input_t, weight, bias=None):
        ctx.save_for_backward(input_t, weight, bias)
        out = _wrap(_C.lib.gt_linear(input_t._ptr, weight._ptr, bias._ptr if bias is not None else 0), "linear")
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        inp, weight, bias = ctx.saved_tensors
        gi = gw = gb = None
        if ctx.needs_input_grad[0]:
            # grad_out: (..., out_features), weight: (out_features, in_features)
            if weight.dtype.is_quantized:
                gi = grad_out.matmul(weight.dequantize(_dt.float32))
            else:
                gi = grad_out.matmul(weight)
        if ctx.needs_input_grad[1] and not weight.dtype.is_quantized:
            in_feat = inp.shape[-1]
            out_feat = weight.shape[0]
            g2 = grad_out.reshape((-1, out_feat))
            i2 = inp.reshape((-1, in_feat))
            gw = g2.t().matmul(i2)
        if bias is not None and ctx.needs_input_grad[2]:
            dims = tuple(range(grad_out.ndim - 1))
            gb = grad_out.sum(dim=dims) if dims else grad_out
        return (gi, gw, gb)


class AddmmFn(Function):
    @staticmethod
    def forward(ctx: Context, self_t, mat1, mat2, beta: float = 1.0, alpha: float = 1.0):
        ctx.save_for_backward(mat1, mat2)
        ctx.shapes = (self_t.shape, mat1.shape, mat2.shape)
        ctx.beta = beta
        ctx.alpha = alpha
        return _wrap(_C.lib.gt_addmm(self_t._ptr, mat1._ptr, mat2._ptr, beta, alpha), "addmm")

    @staticmethod
    def backward(ctx: Context, grad_out):
        m1, m2 = ctx.saved_tensors
        s_self, _, _ = ctx.shapes
        gs = _unbroadcast(grad_out * ctx.beta, s_self) if ctx.needs_input_grad[0] else None
        g1 = (grad_out.matmul(m2.t()) * ctx.alpha) if ctx.needs_input_grad[1] else None
        g2 = (m1.t().matmul(grad_out) * ctx.alpha) if ctx.needs_input_grad[2] else None
        return (gs, g1, g2, None, None)


class OuterFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return _wrap(_C.lib.gt_outer(a._ptr, b._ptr), "outer")

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_tensors
        ga = grad_out.matmul(b) if ctx.needs_input_grad[0] else None
        gb = grad_out.t().matmul(a) if ctx.needs_input_grad[1] else None
        return (ga, gb)


# ==================== Reductions / Cumsum / Sort / TopK ====================


class ReduceFn(Function):
    @staticmethod
    def forward(ctx: Context, x, op_code: int, dims: tuple[int, ...], keepdim: bool):
        nd = x.ndim
        norm_dims = tuple(d if d >= 0 else d + nd for d in dims)
        ctx.orig_shape = x.shape
        ctx.dims = norm_dims
        ctx.keepdim = keepdim
        ctx.op_code = op_code
        d_arr = _c_i32_arr(norm_dims) if norm_dims else None
        out = _wrap(_C.lib.gt_reduce(x._ptr, op_code, len(norm_dims), d_arr, 1 if keepdim else 0, -1), "reduce")
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        x, y = ctx.saved_tensors
        op = ctx.op_code
        g = grad_out
        # Re-insert squeezed reduced dims so broadcast works cleanly
        if not ctx.keepdim and ctx.orig_shape:
            red_set = set(range(len(ctx.orig_shape))) if not ctx.dims else set(ctx.dims)
            new_sh = tuple(1 if i in red_set else s for i, s in enumerate(ctx.orig_shape))
            g = g.reshape(new_sh)
            y_exp = y.reshape(new_sh)
        else:
            y_exp = y

        if op == _C.RED_SUM:
            gx = g.expand(ctx.orig_shape)
        elif op == _C.RED_MEAN:
            red_set = set(range(len(ctx.orig_shape))) if not ctx.dims else set(ctx.dims)
            count = 1
            for i in red_set:
                count *= max(ctx.orig_shape[i], 1)
            gx = (g / float(max(count, 1))).expand(ctx.orig_shape)
        elif op == _C.RED_PROD:
            # Differentiable safe product gradient even when some entries are zero
            from ._ops import where
            is_zero = x == 0
            n_zeros = is_zero.to(_dt.int64).sum(dim=ctx.dims if ctx.dims else None, keepdim=True)
            safe_x = where(is_zero, x.new_ones(x.shape), x)
            prod_nz = safe_x.prod(dim=ctx.dims if ctx.dims else None, keepdim=True) if ctx.dims else safe_x.prod()
            if not ctx.keepdim and ctx.orig_shape and ctx.dims:
                red_set = set(ctx.dims)
                new_sh = tuple(1 if i in red_set else s for i, s in enumerate(ctx.orig_shape))
                n_zeros = n_zeros.reshape(new_sh)
                prod_nz = prod_nz.reshape(new_sh)
            zero_case = where(is_zero, prod_nz, x.new_zeros(x.shape))
            nonzero_case = y_exp / safe_x
            local_grad = where(n_zeros == 0, nonzero_case, where(n_zeros == 1, zero_case, x.new_zeros(x.shape)))
            gx = g.expand(ctx.orig_shape) * local_grad
        elif op in (_C.RED_MAX, _C.RED_MIN):
            mask = (x == y_exp).to(g.dtype)
            if ctx.dims:
                for d in sorted(ctx.dims, reverse=True):
                    cnt = mask.sum(dim=d, keepdim=True)
                    mask = mask / cnt
            else:
                cnt = mask.sum()
                mask = mask / cnt
            gx = g.expand(ctx.orig_shape) * mask
        elif op == _C.RED_LOGSUMEXP:
            gx = g.expand(ctx.orig_shape) * (x - y_exp).exp()
        elif op == _C.RED_NORM_L2:
            safe_y = y_exp + (y_exp == 0).to(y_exp.dtype) * 1e-12
            gx = g.expand(ctx.orig_shape) * (x / safe_y)
        elif op == _C.RED_NORM_L1:
            gx = g.expand(ctx.orig_shape) * x.sign()
        elif op in (_C.RED_VAR_POP, _C.RED_VAR_SAMPLE):
            red_set = set(range(len(ctx.orig_shape))) if not ctx.dims else set(ctx.dims)
            count = 1
            for i in red_set:
                count *= max(ctx.orig_shape[i], 1)
            denom = float(max(count - (1 if op == _C.RED_VAR_SAMPLE else 0), 1))
            mean_x = x.mean(dim=ctx.dims if ctx.dims else None, keepdim=True)
            gx = g.expand(ctx.orig_shape) * (2.0 / denom) * (x - mean_x)
        elif op in (_C.RED_STD_POP, _C.RED_STD_SAMPLE):
            red_set = set(range(len(ctx.orig_shape))) if not ctx.dims else set(ctx.dims)
            count = 1
            for i in red_set:
                count *= max(ctx.orig_shape[i], 1)
            denom = float(max(count - (1 if op == _C.RED_STD_SAMPLE else 0), 1))
            mean_x = x.mean(dim=ctx.dims if ctx.dims else None, keepdim=True)
            safe_y = y_exp + (y_exp == 0).to(y_exp.dtype) * 1e-12
            gx = g.expand(ctx.orig_shape) * (x - mean_x) / (denom * safe_y)
        else:
            gx = None
        return (gx, None, None, None)


class CumsumFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim: int):
        d = dim if dim >= 0 else dim + x.ndim
        ctx.dim = d
        return _wrap(_C.lib.gt_cumsum(x._ptr, d), "cumsum")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,)
        # Reverse cumsum along dim
        rev = grad_out.flip(dims=(ctx.dim,))
        cs = rev.cumsum(dim=ctx.dim)
        return (cs.flip(dims=(ctx.dim,)), None)


class SortFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim: int, descending: bool):
        d = dim if dim >= 0 else dim + x.ndim
        ctx.dim = d
        ctx.orig_shape = x.shape
        val_p = ctypes.c_void_p()
        idx_p = ctypes.c_void_p()
        _C.check_rc(_C.lib.gt_sort(x._ptr, d, 1 if descending else 0, ctypes.byref(val_p), ctypes.byref(idx_p)), "sort")
        vals = _wrap(val_p.value, "sort values")
        idxs = _wrap(idx_p.value, "sort indices")
        ctx.mark_non_differentiable(idxs)
        ctx.save_for_backward(idxs)
        return vals, idxs

    @staticmethod
    def backward(ctx: Context, grad_val, grad_idx=None):
        if grad_val is None:
            return (None, None, None)
        (idxs,) = ctx.saved_tensors
        from ._creation import zeros_like
        g_in = zeros_like(grad_val)
        return (g_in.scatter(ctx.dim, idxs, grad_val), None, None)


class TopKFn(Function):
    @staticmethod
    def forward(ctx: Context, x, k: int, dim: int, largest: bool, sorted_flag: bool):
        d = dim if dim >= 0 else dim + x.ndim
        ctx.dim = d
        ctx.orig_shape = x.shape
        val_p = ctypes.c_void_p()
        idx_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_topk(x._ptr, int(k), d, 1 if largest else 0, 1 if sorted_flag else 0, ctypes.byref(val_p), ctypes.byref(idx_p)),
            "topk",
        )
        vals = _wrap(val_p.value, "topk values")
        idxs = _wrap(idx_p.value, "topk indices")
        ctx.mark_non_differentiable(idxs)
        ctx.save_for_backward(idxs)
        return vals, idxs

    @staticmethod
    def backward(ctx: Context, grad_val, grad_idx=None):
        if grad_val is None:
            return (None, None, None, None, None)
        (idxs,) = ctx.saved_tensors
        from ._creation import zeros
        g_in = zeros(ctx.orig_shape, dtype=grad_val.dtype, device=grad_val.device)
        return (g_in.scatter(ctx.dim, idxs, grad_val), None, None, None, None)


# ==================== Indexing / Pad / Triangular / Flip / Roll ====================


class IndexSelectFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim: int, index):
        d = dim if dim >= 0 else dim + x.ndim
        ctx.dim = d
        ctx.dim_size = x.shape[d]
        ctx.save_for_backward(index)
        return _wrap(_C.lib.gt_index_select(x._ptr, d, index._ptr), "index_select")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        (index,) = ctx.saved_tensors
        gx = _wrap(_C.lib.gt_index_select_add(grad_out._ptr, ctx.dim, index._ptr, ctx.dim_size), "index_select_backward")
        return (gx, None, None)


class GatherFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim: int, index):
        d = dim if dim >= 0 else dim + x.ndim
        ctx.dim = d
        ctx.orig_shape = x.shape
        ctx.save_for_backward(index)
        return _wrap(_C.lib.gt_gather(x._ptr, d, index._ptr), "gather")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        (index,) = ctx.saved_tensors
        from ._creation import zeros
        g_in = zeros(ctx.orig_shape, dtype=grad_out.dtype, device=grad_out.device)
        return (g_in.scatter_add(ctx.dim, index, grad_out), None, None)


class ScatterFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim: int, index, src, accumulate: int = 0):
        d = dim if dim >= 0 else dim + x.ndim
        ctx.dim = d
        ctx.accumulate = accumulate
        ctx.save_for_backward(index)
        return _wrap(_C.lib.gt_scatter(x._ptr, d, index._ptr, src._ptr, accumulate), "scatter")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None, None)
        (index,) = ctx.saved_tensors
        g_src = grad_out.gather(ctx.dim, index) if ctx.needs_input_grad[3] else None
        if ctx.accumulate:
            g_x = grad_out if ctx.needs_input_grad[0] else None
        else:
            from ._creation import zeros_like
            g_x = grad_out.scatter(ctx.dim, index, zeros_like(g_src if g_src is not None else grad_out.gather(ctx.dim, index))) if ctx.needs_input_grad[0] else None
        return (g_x, None, None, g_src, None)


class PadFn(Function):
    @staticmethod
    def forward(ctx: Context, x, pad: tuple[int, ...], mode_code: int, value: float):
        ctx.orig_shape = x.shape
        ctx.pad = pad
        p_arr = _c_i64_arr(pad)
        return _wrap(_C.lib.gt_pad(x._ptr, len(pad) // 2, p_arr, mode_code, float(value)), "pad")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None)
        nd = len(ctx.orig_shape)
        pad = ctx.pad
        slices = [slice(None)] * nd
        n_pairs = len(pad) // 2
        for i in range(n_pairs):
            dim = nd - 1 - i
            left = pad[2 * i]
            right = pad[2 * i + 1]
            start = max(left, 0)
            stop = grad_out.shape[dim] - max(right, 0)
            slices[dim] = slice(start, stop)
        return (grad_out[tuple(slices)], None, None, None)


class TrilTriuFn(Function):
    @staticmethod
    def forward(ctx: Context, x, diagonal: int, upper: int):
        ctx.diagonal = diagonal
        ctx.upper = upper
        return _wrap(_C.lib.gt_tril_triu(x._ptr, int(diagonal), int(upper)), "tril_triu")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        return (_wrap(_C.lib.gt_tril_triu(grad_out._ptr, int(ctx.diagonal), int(ctx.upper)), "tril_triu_bw"), None, None)


class FlipFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dims: tuple[int, ...]):
        nd = x.ndim
        norm = tuple(d if d >= 0 else d + nd for d in dims)
        ctx.dims = norm
        return _wrap(_C.lib.gt_flip(x._ptr, len(norm), _c_i32_arr(norm)), "flip")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None)
        return (_wrap(_C.lib.gt_flip(grad_out._ptr, len(ctx.dims), _c_i32_arr(ctx.dims)), "flip_bw"), None)


class RollFn(Function):
    @staticmethod
    def forward(ctx: Context, x, shifts: tuple[int, ...], dims: tuple[int, ...]):
        nd = x.ndim
        norm = tuple(d if d >= 0 else d + nd for d in dims)
        ctx.shifts = shifts
        ctx.dims = norm
        return _wrap(_C.lib.gt_roll(x._ptr, len(norm), _c_i64_arr(shifts), _c_i32_arr(norm)), "roll")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        neg_shifts = tuple(-s for s in ctx.shifts)
        return (_wrap(_C.lib.gt_roll(grad_out._ptr, len(ctx.dims), _c_i64_arr(neg_shifts), _c_i32_arr(ctx.dims)), "roll_bw"), None, None)


# ==================== Neural Network Functions ====================


class SoftmaxFn(Function):
    @staticmethod
    def forward(ctx: Context, x, dim: int, log_softmax: int):
        nd = x.ndim
        d = dim if dim >= 0 else dim + nd
        ctx.dim = d
        ctx.log_softmax = log_softmax
        out = _wrap(_C.lib.gt_softmax(x._ptr, d, log_softmax), "softmax")
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        (out,) = ctx.saved_tensors
        if _autograd_state.is_grad_enabled():
            if ctx.log_softmax:
                gx = grad_out - out.exp() * grad_out.sum(dim=ctx.dim, keepdim=True)
            else:
                dot = (grad_out * out).sum(dim=ctx.dim, keepdim=True)
                gx = out * (grad_out - dot)
            return (gx, None, None)
        gx = _wrap(_C.lib.gt_softmax_backward(grad_out._ptr, out._ptr, ctx.dim, ctx.log_softmax), "softmax_bw")
        return (gx, None, None)


class RMSNormFn(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, eps: float):
        ctx.eps = float(eps)
        ctx.save_for_backward(x, weight)
        return _wrap(_C.lib.gt_rms_norm(x._ptr, weight._ptr if weight is not None else 0, float(eps)), "rms_norm")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        x, weight = ctx.saved_tensors
        if _autograd_state.is_grad_enabled():
            var = (x * x).mean(dim=-1, keepdim=True)
            inv = (var + ctx.eps).rsqrt()
            x_norm = x * inv
            g = grad_out * weight if weight is not None else grad_out
            d = float(x.shape[-1])
            gx = inv * (g - x_norm * (g * x_norm).sum(dim=-1, keepdim=True) / d)
            gw = None
            if weight is not None and ctx.needs_input_grad[1]:
                dims = tuple(range(x.ndim - 1))
                gw = (grad_out * x_norm).sum(dim=dims) if dims else (grad_out * x_norm)
            return (gx, gw, None)
        gw_p = ctypes.c_void_p()
        gx = _wrap(
            _C.lib.gt_rms_norm_backward(
                grad_out._ptr, x._ptr, weight._ptr if weight is not None else 0, ctx.eps, ctypes.byref(gw_p)
            ),
            "rms_norm_bw",
        )
        gw = _wrap(gw_p.value, "rms_norm_gw") if gw_p.value and ctx.needs_input_grad[1] else None
        if gw_p.value and gw is None:
            _C.lib.gt_tensor_free(gw_p.value)
        return (gx, gw, None)


class LayerNormFn(Function):
    @staticmethod
    def forward(ctx: Context, x, norm_ndim: int, weight, bias, eps: float):
        ctx.norm_ndim = norm_ndim
        ctx.eps = float(eps)
        ctx.save_for_backward(x, weight, bias)
        return _wrap(
            _C.lib.gt_layer_norm(
                x._ptr, norm_ndim, weight._ptr if weight is not None else 0, bias._ptr if bias is not None else 0, float(eps)
            ),
            "layer_norm",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None, None)
        x, weight, bias = ctx.saved_tensors
        if _autograd_state.is_grad_enabled():
            norm_dims = tuple(range(x.ndim - ctx.norm_ndim, x.ndim))
            mean = x.mean(dim=norm_dims, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=norm_dims, keepdim=True)
            inv = (var + ctx.eps).rsqrt()
            x_hat = (x - mean) * inv
            g = grad_out * weight if weight is not None else grad_out
            n_el = 1
            for d in norm_dims:
                n_el *= x.shape[d]
            gx = inv * (g - g.mean(dim=norm_dims, keepdim=True) - x_hat * (g * x_hat).mean(dim=norm_dims, keepdim=True))
            batch_dims = tuple(range(x.ndim - ctx.norm_ndim))
            gw = ((grad_out * x_hat).sum(dim=batch_dims) if batch_dims else (grad_out * x_hat)) if (weight is not None and ctx.needs_input_grad[2]) else None
            gb = (grad_out.sum(dim=batch_dims) if batch_dims else grad_out) if (bias is not None and ctx.needs_input_grad[3]) else None
            return (gx, None, gw, gb, None)

        gx_p = ctypes.c_void_p()
        gw_p = ctypes.c_void_p()
        gb_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_layer_norm_backward(
                grad_out._ptr,
                x._ptr,
                ctx.norm_ndim,
                weight._ptr if weight is not None else 0,
                ctx.eps,
                ctypes.byref(gx_p),
                ctypes.byref(gw_p),
                ctypes.byref(gb_p),
            ),
            "layer_norm_bw",
        )
        gx = _wrap(gx_p.value) if ctx.needs_input_grad[0] and gx_p.value else None
        gw = _wrap(gw_p.value) if weight is not None and ctx.needs_input_grad[2] and gw_p.value else None
        gb = _wrap(gb_p.value) if bias is not None and ctx.needs_input_grad[3] and gb_p.value else None
        if gx_p.value and gx is None:
            _C.lib.gt_tensor_free(gx_p.value)
        if gw_p.value and gw is None:
            _C.lib.gt_tensor_free(gw_p.value)
        if gb_p.value and gb is None:
            _C.lib.gt_tensor_free(gb_p.value)
        return (gx, None, gw, gb, None)


class GroupNormFn(Function):
    @staticmethod
    def forward(ctx: Context, x, num_groups: int, weight, bias, eps: float):
        ctx.num_groups = num_groups
        ctx.eps = float(eps)
        ctx.save_for_backward(x, weight, bias)
        return _wrap(
            _C.lib.gt_group_norm(
                x._ptr, num_groups, weight._ptr if weight is not None else 0, bias._ptr if bias is not None else 0, float(eps)
            ),
            "group_norm",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None, None)
        x, weight, bias = ctx.saved_tensors
        B, C = x.shape[0], x.shape[1]
        spatial = x.shape[2:]
        G = ctx.num_groups
        xr = x.reshape((B, G, C // G, *spatial))
        mean = xr.mean(dim=tuple(range(2, xr.ndim)), keepdim=True)
        var = ((xr - mean) ** 2).mean(dim=tuple(range(2, xr.ndim)), keepdim=True)
        inv = (var + ctx.eps).rsqrt()
        x_hat_g = (xr - mean) * inv
        x_hat = x_hat_g.reshape(x.shape)
        w_view = weight.reshape((1, C) + (1,) * len(spatial)) if weight is not None else None
        g = grad_out * w_view if w_view is not None else grad_out
        gr = g.reshape((B, G, C // G, *spatial))
        rdims = tuple(range(2, gr.ndim))
        gx = (inv * (gr - gr.mean(dim=rdims, keepdim=True) - x_hat_g * (gr * x_hat_g).mean(dim=rdims, keepdim=True))).reshape(x.shape)
        sum_dims = (0,) + tuple(range(2, x.ndim))
        gw = (grad_out * x_hat).sum(dim=sum_dims) if (weight is not None and ctx.needs_input_grad[2]) else None
        gb = grad_out.sum(dim=sum_dims) if (bias is not None and ctx.needs_input_grad[3]) else None
        return (gx, None, gw, gb, None)


class BatchNormFn(Function):
    @staticmethod
    def forward(ctx: Context, x, running_mean, running_var, weight, bias, training: bool, momentum: float, eps: float):
        ctx.training = bool(training)
        ctx.eps = float(eps)
        sm_p = ctypes.c_void_p()
        sv_p = ctypes.c_void_p()
        out = _wrap(
            _C.lib.gt_batch_norm(
                x._ptr,
                weight._ptr if weight is not None else 0,
                bias._ptr if bias is not None else 0,
                running_mean._ptr if running_mean is not None else 0,
                running_var._ptr if running_var is not None else 0,
                1 if training else 0,
                float(momentum),
                float(eps),
                ctypes.byref(sm_p),
                ctypes.byref(sv_p),
            ),
            "batch_norm",
        )
        sm = _wrap(sm_p.value) if sm_p.value else None
        sv = _wrap(sv_p.value) if sv_p.value else None
        ctx.save_for_backward(x, running_mean, running_var, weight, sm, sv)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 8
        x, rm, rv, w, sm, sv = ctx.saved_tensors
        gx_p = ctypes.c_void_p()
        gw_p = ctypes.c_void_p()
        gb_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_batch_norm_backward(
                grad_out._ptr,
                x._ptr,
                w._ptr if w is not None else 0,
                sm._ptr if sm is not None else 0,
                sv._ptr if sv is not None else 0,
                rm._ptr if rm is not None else 0,
                rv._ptr if rv is not None else 0,
                1 if ctx.training else 0,
                ctx.eps,
                ctypes.byref(gx_p),
                ctypes.byref(gw_p),
                ctypes.byref(gb_p),
            ),
            "batch_norm_bw",
        )
        gx = _wrap(gx_p.value) if ctx.needs_input_grad[0] and gx_p.value else None
        gw = _wrap(gw_p.value) if w is not None and ctx.needs_input_grad[3] and gw_p.value else None
        gb = _wrap(gb_p.value) if ctx.needs_input_grad[4] and gb_p.value else None
        if gx_p.value and gx is None:
            _C.lib.gt_tensor_free(gx_p.value)
        if gw_p.value and gw is None:
            _C.lib.gt_tensor_free(gw_p.value)
        if gb_p.value and gb is None:
            _C.lib.gt_tensor_free(gb_p.value)
        return (gx, None, None, gw, gb, None, None, None)


class EmbeddingFn(Function):
    @staticmethod
    def forward(ctx: Context, weight, indices, padding_idx: int):
        ctx.num_embeddings = weight.shape[0]
        ctx.padding_idx = int(padding_idx)
        ctx.save_for_backward(indices)
        return _wrap(_C.lib.gt_embedding(weight._ptr, indices._ptr, int(padding_idx)), "embedding")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None)
        (indices,) = ctx.saved_tensors
        gw = _wrap(
            _C.lib.gt_embedding_backward(grad_out._ptr, indices._ptr, ctx.num_embeddings, ctx.padding_idx),
            "embedding_bw",
        )
        return (gw, None, None)


class Conv2dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int], groups: int):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        return _wrap(
            _C.lib.gt_conv2d(
                x._ptr,
                weight._ptr,
                bias._ptr if bias is not None else 0,
                stride[0], stride[1],
                padding[0], padding[1],
                dilation[0], dilation[1],
                groups,
            ),
            "conv2d",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 7
        x, weight = ctx.saved_tensors
        gx_p = ctypes.c_void_p()
        gw_p = ctypes.c_void_p()
        gb_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_conv2d_backward(
                grad_out._ptr,
                x._ptr,
                weight._ptr,
                ctx.stride[0], ctx.stride[1],
                ctx.padding[0], ctx.padding[1],
                ctx.dilation[0], ctx.dilation[1],
                ctx.groups,
                1 if ctx.needs_input_grad[0] else 0,
                1 if ctx.needs_input_grad[1] else 0,
                1 if (ctx.has_bias and ctx.needs_input_grad[2]) else 0,
                ctypes.byref(gx_p),
                ctypes.byref(gw_p),
                ctypes.byref(gb_p),
            ),
            "conv2d_bw",
        )
        gx = _wrap(gx_p.value) if gx_p.value else None
        gw = _wrap(gw_p.value) if gw_p.value else None
        gb = _wrap(gb_p.value) if gb_p.value else None
        return (gx, gw, gb, None, None, None, None)


class Conv1dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride: int, padding: int, dilation: int, groups: int):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        return _wrap(
            _C.lib.gt_conv1d(x._ptr, weight._ptr, bias._ptr if bias is not None else 0, stride, padding, dilation, groups),
            "conv1d",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 7
        x, weight = ctx.saved_tensors
        gx_p = ctypes.c_void_p()
        gw_p = ctypes.c_void_p()
        gb_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_conv1d_backward(
                grad_out._ptr, x._ptr, weight._ptr,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
                1 if ctx.needs_input_grad[0] else 0,
                1 if ctx.needs_input_grad[1] else 0,
                1 if (ctx.has_bias and ctx.needs_input_grad[2]) else 0,
                ctypes.byref(gx_p), ctypes.byref(gw_p), ctypes.byref(gb_p),
            ),
            "conv1d_bw",
        )
        gx = _wrap(gx_p.value) if gx_p.value else None
        gw = _wrap(gw_p.value) if gw_p.value else None
        gb = _wrap(gb_p.value) if gb_p.value else None
        return (gx, gw, gb, None, None, None, None)


class ConvTranspose2dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride, padding, output_padding, dilation, groups: int):
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        return _wrap(
            _C.lib.gt_conv_transpose2d(
                x._ptr, weight._ptr, bias._ptr if bias is not None else 0,
                stride[0], stride[1], padding[0], padding[1],
                output_padding[0], output_padding[1], dilation[0], dilation[1], groups,
            ),
            "conv_transpose2d",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 8
        x, weight = ctx.saved_tensors
        gx_p = ctypes.c_void_p()
        gw_p = ctypes.c_void_p()
        gb_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_conv_transpose2d_backward(
                grad_out._ptr, x._ptr, weight._ptr,
                ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1],
                ctx.output_padding[0], ctx.output_padding[1], ctx.dilation[0], ctx.dilation[1], ctx.groups,
                1 if ctx.needs_input_grad[0] else 0,
                1 if ctx.needs_input_grad[1] else 0,
                1 if (ctx.has_bias and ctx.needs_input_grad[2]) else 0,
                ctypes.byref(gx_p), ctypes.byref(gw_p), ctypes.byref(gb_p),
            ),
            "conv_transpose2d_bw",
        )
        gx = _wrap(gx_p.value) if gx_p.value else None
        gw = _wrap(gw_p.value) if gw_p.value else None
        gb = _wrap(gb_p.value) if gb_p.value else None
        return (gx, gw, gb, None, None, None, None, None)


class Conv3dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride, padding, dilation, groups: int):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        return _wrap(
            _C.lib.gt_conv3d(
                x._ptr, weight._ptr, bias._ptr if bias is not None else 0,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
                groups,
            ),
            "conv3d",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 7
        x, weight = ctx.saved_tensors
        gx_p = ctypes.c_void_p()
        gw_p = ctypes.c_void_p()
        gb_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_conv3d_backward(
                grad_out._ptr, x._ptr, weight._ptr,
                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                ctx.groups,
                1 if ctx.needs_input_grad[0] else 0,
                1 if ctx.needs_input_grad[1] else 0,
                1 if (ctx.has_bias and ctx.needs_input_grad[2]) else 0,
                ctypes.byref(gx_p), ctypes.byref(gw_p), ctypes.byref(gb_p),
            ),
            "conv3d_bw",
        )
        gx = _wrap(gx_p.value) if gx_p.value else None
        gw = _wrap(gw_p.value) if gw_p.value else None
        gb = _wrap(gb_p.value) if gb_p.value else None
        return (gx, gw, gb, None, None, None, None)


class MaxPool2dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, k, s, p, d):
        out_p = ctypes.c_void_p()
        idx_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_max_pool2d(x._ptr, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1], ctypes.byref(out_p), ctypes.byref(idx_p)),
            "max_pool2d",
        )
        out = _wrap(out_p.value)
        idx = _wrap(idx_p.value)
        ctx.mark_non_differentiable(idx)
        ctx.save_for_backward(x, idx)
        return out, idx

    @staticmethod
    def backward(ctx: Context, grad_out, grad_idx=None):
        if grad_out is None:
            return (None, None, None, None, None)
        x, idx = ctx.saved_tensors
        return (_wrap(_C.lib.gt_max_pool2d_backward(grad_out._ptr, idx._ptr, x._ptr), "max_pool2d_bw"), None, None, None, None)


class AvgPool2dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, k, s, p, count_include_pad: int):
        ctx.orig_shape = x.shape
        ctx.k = k
        ctx.s = s
        ctx.p = p
        ctx.cip = count_include_pad
        ctx.save_for_backward(x)
        return _wrap(_C.lib.gt_avg_pool2d(x._ptr, k[0], k[1], s[0], s[1], p[0], p[1], count_include_pad), "avg_pool2d")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None, None)
        (x,) = ctx.saved_tensors
        gx = _wrap(
            _C.lib.gt_avg_pool2d_backward(
                grad_out._ptr, x._ptr, ctx.k[0], ctx.k[1], ctx.s[0], ctx.s[1], ctx.p[0], ctx.p[1], ctx.cip
            ),
            "avg_pool2d_bw",
        )
        return (gx, None, None, None, None)


class Upsample2dFn(Function):
    @staticmethod
    def forward(ctx: Context, x, size: tuple[int, int], mode_code: int, align_corners: int):
        ctx.in_h = x.shape[-2]
        ctx.in_w = x.shape[-1]
        ctx.mode_code = mode_code
        ctx.align_corners = align_corners
        return _wrap(_C.lib.gt_upsample_2d(x._ptr, size[0], size[1], mode_code, align_corners), "upsample2d")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None, None, None, None)
        return (
            _wrap(
                _C.lib.gt_upsample_2d_backward(grad_out._ptr, ctx.in_h, ctx.in_w, ctx.mode_code, ctx.align_corners),
                "upsample2d_bw",
            ),
            None,
            None,
            None,
        )


class CrossEntropyFn(Function):
    @staticmethod
    def forward(ctx: Context, logits, target, weight, ignore_index: int, label_smoothing: float, reduction_code: int):
        ctx.ignore_index = int(ignore_index)
        ctx.label_smoothing = float(label_smoothing)
        ctx.reduction_code = int(reduction_code)
        ctx.save_for_backward(logits, target, weight)
        return _wrap(
            _C.lib.gt_cross_entropy_loss(
                logits._ptr,
                target._ptr,
                weight._ptr if weight is not None else 0,
                int(ignore_index),
                float(label_smoothing),
                int(reduction_code),
            ),
            "cross_entropy",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 6
        logits, target, weight = ctx.saved_tensors
        gx = _wrap(
            _C.lib.gt_cross_entropy_loss_backward(
                grad_out._ptr,
                logits._ptr,
                target._ptr,
                weight._ptr if weight is not None else 0,
                ctx.ignore_index,
                ctx.label_smoothing,
                ctx.reduction_code,
            ),
            "cross_entropy_bw",
        )
        return (gx, None, None, None, None, None)


class RoPEFn(Function):
    @staticmethod
    def forward(ctx: Context, x, pos, n_dims: int, mode: int, n_ctx_orig: int, freq_base: float, freq_scale: float):
        ctx.args = (n_dims, mode, n_ctx_orig, float(freq_base), float(freq_scale))
        ctx.save_for_backward(pos)
        return _wrap(_C.lib.gt_rope(x._ptr, pos._ptr, n_dims, mode, n_ctx_orig, float(freq_base), float(freq_scale)), "rope")

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 7
        (pos,) = ctx.saved_tensors
        gx = _wrap(_C.lib.gt_rope_backward(grad_out._ptr, pos._ptr, *ctx.args), "rope_bw")
        return (gx, None, None, None, None, None, None)


class FlashAttentionFn(Function):
    @staticmethod
    def forward(ctx: Context, q, k, v, mask, scale: float, is_causal: bool):
        ctx.scale = float(scale) if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
        ctx.is_causal = bool(is_causal)
        ctx.save_for_backward(q, k, v, mask)
        return _wrap(
            _C.lib.gt_flash_attention(q._ptr, k._ptr, v._ptr, mask._ptr if mask is not None else 0, ctx.scale, 1 if is_causal else 0),
            "flash_attention",
        )

    @staticmethod
    def backward(ctx: Context, grad_out):
        if grad_out is None:
            return (None,) * 6
        q, k, v, mask = ctx.saved_tensors
        B, H, L, D = q.shape
        _, H_kv, S, Dv = v.shape
        k_use = k if H == H_kv else k.repeat_interleave(H // H_kv, dim=1)
        v_use = v if H == H_kv else v.repeat_interleave(H // H_kv, dim=1)
        scores = q.matmul(k_use.transpose(-2, -1)) * ctx.scale
        if ctx.is_causal:
            from ._creation import ones
            cm = ones((L, S), dtype=q.dtype, device=q.device).triu(diagonal=1) * (-1e9)
            scores = scores + cm
        if mask is not None:
            if mask.dtype == _dt.bool:
                scores = scores.masked_fill(~mask, -1e9)
            else:
                scores = scores + mask
        probs = scores.softmax(dim=-1)
        # Recompute gradients analytically
        dv_u = probs.transpose(-2, -1).matmul(grad_out)
        dp = grad_out.matmul(v_use.transpose(-2, -1))
        ds = probs * (dp - (dp * probs).sum(dim=-1, keepdim=True))
        dq = ds.matmul(k_use) * ctx.scale
        dk_u = ds.transpose(-2, -1).matmul(q) * ctx.scale
        if H != H_kv:
            g = H // H_kv
            dk = dk_u.reshape((B, H_kv, g, S, D)).sum(dim=2)
            dv = dv_u.reshape((B, H_kv, g, S, Dv)).sum(dim=2)
        else:
            dk, dv = dk_u, dv_u
        return (dq, dk, dv, None, None, None)


class SwiGLUFn(Function):
    @staticmethod
    def forward(ctx: Context, x, up):
        ctx.save_for_backward(x, up)
        return _wrap(_C.lib.gt_swiglu(x._ptr, up._ptr), "swiglu")

    @staticmethod
    def backward(ctx: Context, grad_out):
        x, up = ctx.saved_tensors
        s = x.sigmoid()
        silu_x = x * s
        gup = grad_out * silu_x if ctx.needs_input_grad[1] else None
        if ctx.needs_input_grad[0]:
            silu_d = s * (1.0 + x * (1.0 - s))
            gx = grad_out * up * silu_d
        else:
            gx = None
        return (gx, gup)


# ==================== Linear Algebra & FFT Functions ====================


class LinalgInvFn(Function):
    @staticmethod
    def forward(ctx: Context, a):
        inv = _wrap(_C.lib.gt_linalg_inv(a._ptr), "linalg.inv")
        ctx.save_for_backward(inv)
        return inv

    @staticmethod
    def backward(ctx: Context, grad_out):
        (inv,) = ctx.saved_tensors
        inv_t = inv.transpose(-2, -1)
        return (-inv_t.matmul(grad_out).matmul(inv_t),)


class LinalgSolveFn(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        x = _wrap(_C.lib.gt_linalg_solve(a._ptr, b._ptr), "linalg.solve")
        ctx.save_for_backward(a, x)
        return x

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, x = ctx.saved_tensors
        a_t = a.transpose(-2, -1)
        b_is_vec = b_was_1d = (x.ndim == a.ndim - 1)
        g_rhs = grad_out.unsqueeze(-1) if b_is_vec else grad_out
        gb = _wrap(_C.lib.gt_linalg_solve(a_t._ptr, g_rhs._ptr), "solve_bw")
        if b_was_1d:
            gb = gb.squeeze(-1)
            x2 = x.unsqueeze(-1)
            gb2 = gb.unsqueeze(-1)
            ga = -gb2.matmul(x2.transpose(-2, -1))
        else:
            ga = -gb.matmul(x.transpose(-2, -1))
        return (ga if ctx.needs_input_grad[0] else None, gb if ctx.needs_input_grad[1] else None)


class LinalgDetFn(Function):
    @staticmethod
    def forward(ctx: Context, a):
        det = _wrap(_C.lib.gt_linalg_det(a._ptr), "linalg.det")
        ctx.save_for_backward(a, det)
        return det

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, det = ctx.saved_tensors
        inv_t = _wrap(_C.lib.gt_linalg_inv(a._ptr), "det_inv").transpose(-2, -1)
        scale = (grad_out * det).unsqueeze(-1).unsqueeze(-1)
        return (scale * inv_t,)


class FFT1DFn(Function):
    @staticmethod
    def forward(ctx: Context, x, n: int, dim: int, inverse: int, norm_mode: int, real_input: int):
        nd = x.ndim
        d = dim if dim >= 0 else dim + nd
        ctx.args = (n, d, inverse, norm_mode, real_input)
        return _wrap(_C.lib.gt_fft_1d(x._ptr, int(n), d, inverse, real_input, norm_mode), "fft")

    @staticmethod
    def backward(ctx: Context, grad_out):
        n, d, inverse, norm_mode, real_input = ctx.args
        inv_bw = 0 if inverse else 1
        # Adjoint of norm_mode: ortho->ortho, forward(0)->backward(1), backward(1)->forward(0)
        bw_norm = 2 if norm_mode == 2 else (1 - norm_mode)
        gx = _wrap(_C.lib.gt_fft_1d(grad_out._ptr, int(n), d, inv_bw, 0, bw_norm), "fft_bw")
        if real_input and gx.dtype.is_complex:
            gx = _wrap(_C.lib.gt_unary_op(gx._ptr, _C.UNARY_REAL, 0.0, 0.0), "fft_bw_real")
        return (gx, None, None, None, None, None)


class RealFn(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ctx.args = (x.dtype, x.dtype.is_complex)
        return _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_REAL, 0.0, 0.0), "real")

    @staticmethod
    def backward(ctx: Context, grad_out):
        in_dtype, is_complex = ctx.args
        if is_complex:
            return (grad_out.to(in_dtype),)
        return (grad_out.to(in_dtype),)


class ImagFn(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ctx.args = (x.dtype, x.dtype.is_complex)
        return _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_IMAG, 0.0, 0.0), "imag")

    @staticmethod
    def backward(ctx: Context, grad_out):
        in_dtype, is_complex = ctx.args
        if is_complex:
            return (grad_out.to(in_dtype) * 1j,)
        return (zeros_like(grad_out, dtype=in_dtype),)


class ConjFn(Function):
    @staticmethod
    def forward(ctx: Context, x):
        return _wrap(_C.lib.gt_unary_op(x._ptr, _C.UNARY_CONJ, 0.0, 0.0), "conj")

    @staticmethod
    def backward(ctx: Context, grad_out):
        return (ConjFn.apply(grad_out),)
