"""Core Tensor class with GGML native storage, slicing/indexing, and autograd."""

from __future__ import annotations

import array
import ctypes
import functools
import itertools
import math
import numbers
import operator
import struct
import sys
import warnings

from . import _C, _autograd_state, _device as _dev, _dtype as _dt
from .amp import _active as _autocast_active, _suspended as _autocast_suspended
from ._autograd_engine import backward as _autograd_backward
from ._functions import (
    AddFn,
    AddmmFn,
    Atan2Fn,
    BinaryScalarFn,
    CastFn,
    CatFn,
    ClampTensorFn,
    CloneFn,
    ConjFn,
    ContiguousFn,
    CumsumFn,
    DiagonalFn,
    DiagonalScatterFn,
    DivFn,
    ExpandFn,
    FlipFn,
    GatherFn,
    ImagFn,
    IndexSelectFn,
    IndexTensorFn,
    LerpFn,
    LinalgDetFn,
    LinalgInvFn,
    LinalgSolveFn,
    MaskedSelectFn,
    MatMulFn,
    MaximumFn,
    MinimumFn,
    MulFn,
    OuterFn,
    PadFn,
    PermuteFn,
    PowFn,
    RealFn,
    ReduceFn,
    ReshapeFn,
    RollFn,
    ScatterFn,
    SliceViewFn,
    SoftmaxFn,
    SortFn,
    SubFn,
    ToDeviceFn,
    TopKFn,
    TransposeFn,
    TrilTriuFn,
    UnarySimpleFn,
    WhereFn,
    _c_i32_arr,
    _c_i64_arr,
    _can_view_reshape,
    _wrap,
)


# ---------- NumPy interoperability ----------
# NumPy is optional. It is never imported on behalf of the caller: an ndarray or a NumPy scalar
# can only get here after the caller has imported NumPy, and sys.modules is where it is found.

_NUMPY_DTYPE_NAMES = {
    "float32": _dt.float32,
    "float64": _dt.float64,
    "float16": _dt.float16,
    "int64": _dt.int64,
    "int32": _dt.int32,
    "int16": _dt.int16,
    "int8": _dt.int8,
    "uint8": _dt.uint8,
    "bool": _dt.bool,
    "complex64": _dt.complex64,
    "complex128": _dt.complex128,
}
_NUMPY_NAME_OF_DTYPE = {dt: name for name, dt in _NUMPY_DTYPE_NAMES.items()}
_NUMPY_SUPPORTED = "float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool"


class _ArrayLayoutError(TypeError, ValueError):
    """An ndarray whose memory a tensor cannot use as it is (negative or fractional strides,
    foreign byte order, unaligned data).

    PyTorch raises ValueError for these. Every other input from_numpy() rejects is a TypeError,
    and callers that catch that for "this cannot become a tensor" should get these too, so the
    exception is both."""


def _numpy():
    return sys.modules.get("numpy")


def _dtype_from_numpy(np_dtype) -> _dt.dtype:
    """The tensor dtype with the layout of a NumPy dtype; TypeError when there is none."""
    dt = _NUMPY_DTYPE_NAMES.get(np_dtype.name)
    if dt is None:
        kind = getattr(np_dtype.type, "__name__", str(np_dtype))
        raise TypeError(f"can't convert np.ndarray of type numpy.{kind}. The only supported types are: {_NUMPY_SUPPORTED}.")
    return dt


def _unwrap_numpy_scalar(x):
    """np.float32(2.0), np.int64(3), np.bool_(True), ... as the Python scalar of the same value.

    In arithmetic and as fill values NumPy scalars behave like Python numbers: they do not
    decide the dtype of the result."""
    np = _numpy()
    if np is None or not isinstance(x, np.generic):
        return x
    value = x.item()
    if isinstance(value, (bool, int, float, complex)):
        return value
    # np.longdouble and np.clongdouble have no Python equivalent and return themselves
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.complexfloating):
        return complex(x)
    return x


def _is_real_number(x) -> bool:
    """A real scalar other than a bool: Python int and float, NumPy integer and floating scalars."""
    t = type(x)
    if t is float or t is int:
        return True
    if t is bool or isinstance(x, Tensor):
        return False
    # NumPy registers its integer and floating scalars with the numbers ABCs (np.bool_ is not one)
    return isinstance(x, numbers.Real)


def _is_int(x) -> bool:
    """A Python int or a NumPy integer scalar - not a bool."""
    t = type(x)
    if t is int:
        return True
    if t is bool or isinstance(x, Tensor):
        return False
    return isinstance(x, numbers.Integral)


def _dims_tuple(dim) -> tuple:
    """dim=1, dim=np.int64(1), dim=(0, 1) or dim=[0, np.int64(1)] as a tuple of ints."""
    try:
        return (operator.index(dim),)
    except TypeError:
        return tuple(operator.index(d) for d in dim)


def _numpy_style_reduction(fn):
    """Lets a reduction be called the NumPy way: ``axis=`` for ``dim=``, ``keepdims=`` for
    ``keepdim=``, and ``out=None``. PyTorch accepts these spellings, and they are how
    np.sum(t), np.mean(t), np.max(t), ... call the method of an object that is not an array."""

    @functools.wraps(fn)
    def reduction(self, *args, **kwargs):
        if kwargs:
            if "axis" in kwargs:
                if "dim" in kwargs:
                    raise TypeError(f"{fn.__name__}() got both 'dim' and 'axis'")
                kwargs["dim"] = kwargs.pop("axis")
            if "keepdims" in kwargs:
                if "keepdim" in kwargs:
                    raise TypeError(f"{fn.__name__}() got both 'keepdim' and 'keepdims'")
                kwargs["keepdim"] = kwargs.pop("keepdims")
            if "out" in kwargs and kwargs.pop("out") is not None:
                raise TypeError(f"{fn.__name__}(): out= is not supported")
        return fn(self, *args, **kwargs)

    return reduction


def _check_expands_to(t, shape: tuple, what: str) -> None:
    """`t` must broadcast to exactly `shape` (the `input` of addmm, baddbmm)."""
    have = tuple(t.shape)
    if len(have) > len(shape) or any(h != 1 and h != w for h, w in zip(reversed(have), reversed(shape))):
        raise RuntimeError(
            f"{what}: the expanded size of the tensor must match the existing size: "
            f"input of shape {list(have)} cannot be broadcast to {list(shape)}"
        )


# ---------- autocast (torch.amp) ----------
#
# The policy - which operation is cast to what - is described in amp.py. Operations that are a
# single native call cast their operands in place (`_autocast_lower`, `_autocast_float32`);
# operations written in terms of other operations carry `@_autocast_op(...)`, which casts the
# arguments and then runs the body with autocast suspended, so that nothing inside is cast a
# second time, to something else.

_AUTOCAST_LOWER = "lower"        # the autocast dtype: float16 / bfloat16
_AUTOCAST_FLOAT32 = "float32"
_AUTOCAST_PROMOTE = "promote"    # float32 as soon as one operand is float32, else the autocast dtype
_AUTOCAST_KEEP = "keep"          # the dtypes the operands have


def _autocast_cast(value, active: dict, to_float32: bool):
    """`value` as autocast hands it to an operation. A floating point tensor on a device type
    with autocast enabled is cast to that device type's autocast dtype, or to float32; float64,
    integer and other-device tensors are not touched. Lists and tuples of tensors are followed."""
    if isinstance(value, Tensor):
        dt = value.dtype
        if not dt.is_floating_point or dt == _dt.float64:
            return value
        target = active.get(value.device.type)
        if target is None:
            return value
        if to_float32:
            target = _dt.float32
        return value if dt == target else value.to(target)
    if type(value) in (list, tuple):
        if any(isinstance(v, Tensor) for v in value):
            return type(value)(_autocast_cast(v, active, to_float32) for v in value)
    return value


def _autocast_has_float32(values, active: dict) -> bool:
    for v in values:
        if isinstance(v, Tensor):
            if v.dtype == _dt.float32 and v.device.type in active:
                return True
        elif type(v) in (list, tuple) and _autocast_has_float32(v, active):
            return True
    return False


def _autocast_lower(*tensors) -> tuple:
    """The operands of an operation autocast runs in the autocast dtype (matmul, linear, conv)."""
    active = _autocast_active()
    if not active:
        return tensors
    return tuple(_autocast_cast(x, active, False) for x in tensors)


def _autocast_float32(*tensors) -> tuple:
    """The operands of an operation autocast runs in float32 (softmax, norms, losses, sums)."""
    active = _autocast_active()
    if not active:
        return tensors
    return tuple(_autocast_cast(x, active, True) for x in tensors)


def _autocast_op(policy: str):
    """Decorator for an operation that takes tensors as arguments: inside an autocast block the
    tensor arguments are cast as `policy` says and the body runs with autocast suspended.
    A call that names its result (`out=`) or its type (`dtype=`) is not autocast."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            active = _autocast_active()
            if not active:
                return fn(*args, **kwargs)
            if policy != _AUTOCAST_KEEP and kwargs.get("out") is None and kwargs.get("dtype") is None:
                to_float32 = policy == _AUTOCAST_FLOAT32 or (
                    policy == _AUTOCAST_PROMOTE and _autocast_has_float32(args + tuple(kwargs.values()), active)
                )
                args = tuple(_autocast_cast(a, active, to_float32) for a in args)
                kwargs = {k: _autocast_cast(v, active, to_float32) for k, v in kwargs.items()}
            with _autocast_suspended():
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def _autocast_float32_method(method):
    """A one-tensor method autocast runs in float32 (exp, log, reciprocal, ...)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        active = _autocast_active()
        if active:
            self = _autocast_cast(self, active, True)
        return method(self, *args, **kwargs)

    return wrapper


def _is_bool_scalar(x) -> bool:
    if type(x) is bool:
        return True
    np = _numpy()
    return np is not None and isinstance(x, np.bool_)


def _is_scalar(x) -> bool:
    """A real number or a bool, Python or NumPy."""
    return _is_real_number(x) or _is_bool_scalar(x)


def _tensor_from_ndarray(arr, device=None) -> Tensor:
    """A tensor holding a copy of `arr` - any strides, any byte order - with the same dtype."""
    dt = _dtype_from_numpy(arr.dtype)
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="))
    dev = _dev.as_device(device)
    shape = tuple(int(n) for n in arr.shape)
    if arr.flags.c_contiguous:
        data, nbytes = ctypes.c_void_p(arr.ctypes.data), arr.nbytes
    else:
        data = arr.tobytes()  # row-major, whatever the strides are
        nbytes = len(data)
    ptr = _C.check_ptr(
        _C.lib.gt_tensor_from_data(len(shape), _c_i64_arr(shape), dt.code, dev.dev_type, dev.index, data, nbytes),
        "tensor from ndarray",
    )
    return Tensor(_ptr=ptr, requires_grad=False)


# Arrays whose memory a tensor storage is using (from_numpy), by the token handed to the native
# side. The native storage can outlive the tensor it was created for - views, saved tensors of
# the autograd graph - so an array is released when its *storage* is reported gone.
_SHARED_ARRAYS: dict = {}
_shared_array_tokens = itertools.count(1)


def _release_shared_arrays() -> None:
    tokens = (ctypes.c_uint64 * 32)()
    while True:
        n = int(_C.lib.gt_external_released(tokens, 32))
        for i in range(n):
            _SHARED_ARRAYS.pop(int(tokens[i]), None)
        if n < 32:
            return


def _from_numpy_shared(arr) -> Tensor:
    """A tensor on the memory of `arr`: nothing is copied and writes show on both sides."""
    dt = _dtype_from_numpy(arr.dtype)
    if not arr.dtype.isnative:
        raise _ArrayLayoutError(
            "given numpy array has byte order different from the native byte order. "
            "Conversion between byte orders is currently not supported."
        )
    shape = tuple(int(n) for n in arr.shape)
    if arr.size == 0:
        # no element to share
        return _wrap(_C.lib.gt_tensor_empty(len(shape), _c_i64_arr(shape), dt.code, _dev.DEV_CPU, 0), "from_numpy")
    itemsize = arr.itemsize
    strides = []
    for step in arr.strides:
        if step < 0:
            raise _ArrayLayoutError(
                "At least one stride in the given numpy array is negative, and tensors with negative strides "
                "are not currently supported. (You can probably work around this by making a copy of your "
                "array with array.copy().)"
            )
        if step % itemsize != 0:
            raise _ArrayLayoutError(
                "given numpy array strides not a multiple of the element byte size. "
                "Copy the numpy array to reallocate the memory."
            )
        strides.append(step // itemsize)
    if not arr.flags.aligned:
        raise _ArrayLayoutError(
            "given numpy array is not aligned for its element type. Copy the numpy array to reallocate the memory."
        )
    if not arr.flags.writeable:
        warnings.warn(
            "The given NumPy array is not writable, and tensors do not support non-writable memory. Writing to "
            "the tensor writes to the underlying (supposedly non-writable) array. Copy the array first "
            "(array.copy()) to protect its data or to make it writable; the tensor itself is otherwise usable.",
            UserWarning,
            stacklevel=3,
        )
    if _SHARED_ARRAYS:
        _release_shared_arrays()
    token = next(_shared_array_tokens)
    _SHARED_ARRAYS[token] = arr  # before the native call: from then on the storage relies on the memory
    ptr = _C.lib.gt_tensor_from_external(len(shape), _c_i64_arr(shape), _c_i64_arr(strides), dt.code, ctypes.c_void_p(arr.ctypes.data), token)
    if not ptr:
        del _SHARED_ARRAYS[token]
        _C.check_ptr(ptr, "from_numpy")
    return Tensor(_ptr=ptr, requires_grad=False)


def _infer_nested_shape_and_flat(obj):
    """Shape, values in row-major order and dtype of (nested) Python data.

    The leaves of the nesting are Python numbers, tensors, NumPy scalars and arrays, and
    anything else that offers __array__. Python numbers have the default types - bool, int64,
    the default floating type - while tensors and NumPy values keep theirs; different leaf
    types are promoted. Leaves need not be scalars: a list of arrays of one shape stacks them.
    """
    default_float = _dt.get_default_dtype()
    default_complex = _dt.complex128 if default_float == _dt.float64 else _dt.complex64
    np = _numpy()
    array_types = (np.ndarray, np.generic) if np is not None else ()
    flat: list = []
    found = None  # the promoted dtype of the leaves seen so far

    def note(dt) -> None:
        nonlocal found
        if found is None:
            found = dt
        elif found is not dt:
            found = _dt.promote_types(found, dt)

    def array_leaf(arr) -> tuple:
        note(_dtype_from_numpy(arr.dtype))
        if arr.ndim == 0:
            flat.append(arr.item())
        else:
            flat.extend(arr.ravel().tolist())
        return tuple(int(n) for n in arr.shape)

    def leaf(x) -> tuple:
        t = type(x)
        if t is float:
            flat.append(x)
            note(default_float)
            return ()
        if t is int:
            flat.append(x)
            note(_dt.int64)
            return ()
        if t is bool:
            flat.append(x)
            note(_dt.bool)
            return ()
        if isinstance(x, Tensor):
            note(x.dtype)
            if x.ndim == 0:
                flat.append(x.item())
            elif x.numel() > 0:
                flat.extend(x.reshape(-1).tolist())
            return tuple(x.shape)
        # NumPy values before the Python types: np.float64 is a float, yet keeps its own dtype
        if array_types and isinstance(x, array_types):
            return array_leaf(np.asarray(x))
        if isinstance(x, bool):
            flat.append(x)
            note(_dt.bool)
            return ()
        if isinstance(x, int):
            flat.append(int(x))
            note(_dt.int64)
            return ()
        if isinstance(x, float):
            flat.append(float(x))
            note(default_float)
            return ()
        if isinstance(x, complex):
            flat.append(complex(x))
            note(default_complex)
            return ()
        if hasattr(x, "__array__") and not isinstance(x, type):
            import numpy

            arr = numpy.asarray(x)
            if isinstance(arr, numpy.ndarray):
                return array_leaf(arr)
        raise TypeError(f"Unsupported scalar/sequence element type: {type(x)}")

    def walk(x) -> tuple:
        if isinstance(x, (list, tuple, range)):
            if len(x) == 0:
                return (0,)
            sub = None
            for el in x:
                sh = walk(el)
                if sub is None:
                    sub = sh
                elif sub != sh:
                    raise ValueError("Expected all sub-lists to have identical shape")
            return (len(x),) + sub
        return leaf(x)

    shape = walk(obj)
    return shape, flat, found if found is not None else default_float


def _pack_flat_to_bytes(flat: list, dtype: _dt.dtype) -> bytes:
    if not flat:
        return b""
    if dtype == _dt.float32:
        return struct.pack(f"<{len(flat)}f", *(float(v.real if isinstance(v, complex) else v) for v in flat))
    if dtype == _dt.float64:
        return struct.pack(f"<{len(flat)}d", *(float(v.real if isinstance(v, complex) else v) for v in flat))
    if dtype == _dt.int64:
        return struct.pack(f"<{len(flat)}q", *(int(v.real if isinstance(v, complex) else v) for v in flat))
    if dtype == _dt.int32:
        return struct.pack(f"<{len(flat)}i", *(int(v.real if isinstance(v, complex) else v) for v in flat))
    if dtype == _dt.int16:
        return struct.pack(f"<{len(flat)}h", *(int(v.real if isinstance(v, complex) else v) for v in flat))
    if dtype == _dt.int8:
        return struct.pack(f"<{len(flat)}b", *(int(v.real if isinstance(v, complex) else v) for v in flat))
    if dtype == _dt.uint8:
        return struct.pack(f"<{len(flat)}B", *(int(v.real if isinstance(v, complex) else v) & 0xFF for v in flat))
    if dtype == _dt.bool:
        return struct.pack(f"<{len(flat)}B", *(1 if v else 0 for v in flat))
    if dtype == _dt.complex64:
        pairs = []
        for v in flat:
            c = complex(v)
            pairs.append(float(c.real))
            pairs.append(float(c.imag))
        return struct.pack(f"<{2 * len(flat)}f", *pairs)
    if dtype == _dt.complex128:
        pairs = []
        for v in flat:
            c = complex(v)
            pairs.append(float(c.real))
            pairs.append(float(c.imag))
        return struct.pack(f"<{2 * len(flat)}d", *pairs)
    # float16 / bfloat16 -> pack as f32 first and let caller cast
    return struct.pack(f"<{len(flat)}f", *(float(v.real if isinstance(v, complex) else v) for v in flat))


def _unpack_bytes_to_flat(buf: bytes, numel: int, dtype: _dt.dtype) -> list:
    if numel == 0:
        return []
    if dtype == _dt.float32:
        return list(struct.unpack(f"<{numel}f", buf[: 4 * numel]))
    if dtype == _dt.float64:
        return list(struct.unpack(f"<{numel}d", buf[: 8 * numel]))
    if dtype == _dt.int64:
        return list(struct.unpack(f"<{numel}q", buf[: 8 * numel]))
    if dtype == _dt.int32:
        return list(struct.unpack(f"<{numel}i", buf[: 4 * numel]))
    if dtype == _dt.int16:
        return list(struct.unpack(f"<{numel}h", buf[: 2 * numel]))
    if dtype == _dt.int8:
        return list(struct.unpack(f"<{numel}b", buf[:numel]))
    if dtype == _dt.uint8:
        return list(struct.unpack(f"<{numel}B", buf[:numel]))
    if dtype == _dt.bool:
        return [bool(b) for b in struct.unpack(f"<{numel}B", buf[:numel])]
    if dtype == _dt.complex64:
        vals = struct.unpack(f"<{2 * numel}f", buf[: 8 * numel])
        return [complex(vals[2 * i], vals[2 * i + 1]) for i in range(numel)]
    if dtype == _dt.complex128:
        vals = struct.unpack(f"<{2 * numel}d", buf[: 16 * numel])
        return [complex(vals[2 * i], vals[2 * i + 1]) for i in range(numel)]
    raise TypeError(f"Cannot directly unpack dtype {dtype}")


def _flat_to_nested(flat: list, shape: tuple[int, ...]):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat[: shape[0]]
    stride = 1
    for s in shape[1:]:
        stride *= s
    return [_flat_to_nested(flat[i * stride : (i + 1) * stride], shape[1:]) for i in range(shape[0])]


def _from_python(data, dtype: _dt.dtype | None = None, device=None, requires_grad: bool = False) -> Tensor:
    dev = _dev.as_device(device)
    if isinstance(data, Tensor):
        t = data
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        else:
            t = t.clone()
        if t.device != dev:
            t = t.to(dev)
        if requires_grad:
            t.requires_grad_(True)
        return t

    # A NumPy array or scalar (never importing numpy for it): a copy, with the array's dtype.
    np_mod = _numpy()
    if np_mod is not None and isinstance(data, (np_mod.ndarray, np_mod.generic)):
        t = _tensor_from_ndarray(np_mod.asarray(data), dev)
        if dtype is not None and t.dtype != _dt.as_dtype(dtype):
            t = t.to(_dt.as_dtype(dtype))
        if requires_grad:
            t.requires_grad_(True)
        return t

    shape, flat, inferred_dt = _infer_nested_shape_and_flat(data)
    target_dt = _dt.as_dtype(dtype) if dtype is not None else inferred_dt

    if target_dt in (_dt.float16, _dt.bfloat16) or target_dt.is_quantized:
        raw = _pack_flat_to_bytes(flat, _dt.float32)
        ptr = _C.check_ptr(
            _C.lib.gt_tensor_from_data(len(shape), _c_i64_arr(shape), _dt.float32.code, dev.dev_type, dev.index, raw, len(raw)),
            "tensor_from_data",
        )
        t = Tensor(_ptr=ptr, requires_grad=False).to(target_dt)
    else:
        raw = _pack_flat_to_bytes(flat, target_dt)
        ptr = _C.check_ptr(
            _C.lib.gt_tensor_from_data(len(shape), _c_i64_arr(shape), target_dt.code, dev.dev_type, dev.index, raw, len(raw)),
            "tensor_from_data",
        )
        t = Tensor(_ptr=ptr, requires_grad=False)

    if requires_grad:
        t.requires_grad_(True)
    return t


def _ensure_tensor(x, ref: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x
    x = _unwrap_numpy_scalar(x)  # a NumPy scalar is an operand like a Python number
    return _from_python(x, dtype=ref.dtype if isinstance(x, float) else None, device=ref.device)


def _index_list_to_tensor(items: list, device) -> Tensor:
    """A list used as an index: all bools -> a boolean mask, otherwise integer indices."""
    flat = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        else:
            flat.append(x)

    walk(items)
    if flat and all(isinstance(x, bool) for x in flat):
        return _from_python(items, dtype=_dt.bool, device=device)
    if any(isinstance(x, float) for x in flat):
        raise IndexError("a list used as an index must hold integers or booleans, not floats")
    return _from_python(items, dtype=_dt.int64, device=device)


def _normalize_index(idx, device):
    """`idx` with every entry in the form the indexing code works on: ints, slices, None,
    Ellipsis, lists and tensors. NumPy integers - anything with __index__ - become ints,
    NumPy arrays become index tensors, a tuple inside the index is a list of indices."""
    kind = type(idx)
    if kind is int or kind is slice or idx is None or idx is Ellipsis:
        return idx

    def entry(it):
        t = type(it)
        if it is None or it is Ellipsis or t is int or t is slice or t is list or t is bool or isinstance(it, Tensor):
            return it
        np = _numpy()
        if np is not None:
            if isinstance(it, np.ndarray):
                if it.dtype.kind in "iu" and it.dtype.name not in _NUMPY_DTYPE_NAMES:
                    it = it.astype(np.int64)  # an index of any integer type is an index
                elif it.dtype.kind not in "iub":
                    raise IndexError("arrays used as indices must be of integer or boolean type")
                return _tensor_from_ndarray(it, device)
            if isinstance(it, np.bool_):
                return bool(it)
        if t is tuple:
            return list(it)
        if isinstance(it, (int, slice, list)):
            return it
        try:
            return operator.index(it)
        except TypeError:
            return it  # reported by the indexing code, with the list of what is valid

    if kind is tuple:
        return tuple(entry(it) for it in idx)
    return entry(idx)


def _advanced_result_shape(shape: tuple[int, ...], dims: list[int], index_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Shape of x[..., i0, i1, ...]: the broadcast index shape replaces the indexed dimensions
    when they are next to each other and comes first when they are not (numpy/torch rule)."""
    idx_shape: tuple[int, ...] = ()
    for sh in index_shapes:
        out = []
        for i in range(1, max(len(idx_shape), len(sh)) + 1):
            da = idx_shape[-i] if i <= len(idx_shape) else 1
            db = sh[-i] if i <= len(sh) else 1
            if da != db and da != 1 and db != 1:
                raise IndexError("shape mismatch: indexing tensors could not be broadcast together")
            out.append(db if da == 1 else da)
        idx_shape = tuple(reversed(out))
    rest = [shape[d] for d in range(len(shape)) if d not in dims]
    if dims[-1] - dims[0] + 1 == len(dims):
        return tuple(shape[: dims[0]]) + idx_shape + tuple(shape[dims[-1] + 1 :])
    return idx_shape + tuple(rest)


def _broadcast_value(value: Tensor, shape: tuple[int, ...]) -> Tensor:
    """`value` expanded to the shape an assignment writes, or a clear error."""
    vs = value.shape
    ok = len(vs) <= len(shape) and all(v == 1 or v == s for v, s in zip(reversed(vs), reversed(shape)))
    if not ok:
        raise RuntimeError(
            f"shape mismatch: value tensor of shape {list(vs)} cannot be broadcast to indexing result of shape {list(shape)}"
        )
    return value.expand(shape)


def _format_tensor_str(t: Tensor) -> str:
    dt_suffix = ""
    if t.dtype != _dt.get_default_dtype():
        dt_suffix = f", dtype={t.dtype}"
    dev_suffix = f", device='{t.device}'" if t.device.type != "cpu" else ""
    grad_suffix = ""
    if t.grad_fn is not None:
        grad_suffix = f", grad_fn=<{t.grad_fn.name}>"
    elif t.requires_grad:
        grad_suffix = ", requires_grad=True"

    if t.device.type == "meta":
        return f"tensor(..., size={tuple(t.shape)}{dt_suffix}{dev_suffix}{grad_suffix})"

    if t.numel() > 1000:
        # Summarize large tensors without dumping millions of elements
        return f"tensor(<shape={tuple(t.shape)}>{dt_suffix}{dev_suffix}{grad_suffix})"

    data = t.tolist()

    def _fmt_num(v):
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, complex):
            return f"{v.real:.4f}{v.imag:+.4f}j"
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        s = f"{v:.4f}".rstrip("0")
        return s if not s.endswith(".") else s + "0" if len(s) == 2 and s[0] == "-" else s

    def _fmt_rec(x, indent: int) -> str:
        if not isinstance(x, list):
            return _fmt_num(x)
        if not x:
            return "[]"
        if not isinstance(x[0], list):
            items = x if len(x) <= 8 else (x[:3] + ["..."] + x[-3:])
            parts = [el if el == "..." else _fmt_num(el) for el in items]
            return "[" + ", ".join(parts) + "]"
        items = x if len(x) <= 6 else (x[:3] + ["..."] + x[-3:])
        sep = ",\n" + (" " * (indent + 1))
        inner = []
        for el in items:
            if el == "...":
                inner.append("...")
            else:
                inner.append(_fmt_rec(el, indent + 1))
        return "[" + sep.join(inner) + "]"

    body = _fmt_rec(data, 7)
    return f"tensor({body}{dt_suffix}{dev_suffix}{grad_suffix})"


class Tensor:
    """PyTorch-compatible N-dimensional Tensor backed by ggml / C++ storage."""

    def __init__(self, *args, _ptr: int | None = None, requires_grad: bool = False):
        self._ptr: int = 0
        self._requires_grad: bool = False
        self._grad: Tensor | None = None
        self._grad_fn = None
        self._output_nr: int = 0
        self._is_leaf: bool = True
        self._retains_grad: bool = False
        self._hooks: dict[int, object] = {}
        self._next_hook_id: int = 0
        self._version: int = 0

        if _ptr is not None:
            self._ptr = int(_ptr)
            self._requires_grad = bool(requires_grad)
            return

        if len(args) == 0:
            sh = (0,)
            dt = _dt.get_default_dtype()
            ptr = _C.check_ptr(_C.lib.gt_tensor_empty(1, _c_i64_arr(sh), dt.code, _dev.DEV_CPU, 0))
            self._ptr = ptr
        elif len(args) == 1 and isinstance(args[0], Tensor):
            ptr = _C.check_ptr(_C.lib.gt_tensor_view(args[0]._ptr, args[0].ndim, _c_i64_arr(args[0].shape), _c_i64_arr(args[0].stride()), args[0].storage_offset()))
            self._ptr = ptr
        elif all(isinstance(a, int) for a in args):
            sh = tuple(args)
            dt = _dt.get_default_dtype()
            ptr = _C.check_ptr(_C.lib.gt_tensor_empty(len(sh), _c_i64_arr(sh), dt.code, _dev.DEV_CPU, 0))
            self._ptr = ptr
        elif len(args) == 1:
            t = _from_python(args[0])
            self._ptr = t._ptr
            t._ptr = 0
        else:
            raise TypeError(f"Tensor() received invalid arguments: {args}")

        if requires_grad:
            self.requires_grad_(True)

    # ---------- copy / pickle ----------
    # A Tensor owns a native handle. The default copy and pickle protocols duplicate the Python
    # attributes, handle included: two objects then free the same native tensor (a double free,
    # or a crash when one of them is used after the other is gone).

    def _shell(self) -> Tensor:
        """A new object of the same class with the Python-side attributes of this one and no data."""
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._ptr = 0
        new._grad = None
        new._grad_fn = None
        new._hooks = {}
        new._next_hook_id = 0
        return new

    def __copy__(self) -> Tensor:
        """Shallow copy: another tensor object on the same data."""
        new = self._shell()
        new._ptr = _C.check_ptr(
            _C.lib.gt_tensor_view(self._ptr, self.ndim, _c_i64_arr(self.shape), _c_i64_arr(self.stride()), self.storage_offset()),
            "copy",
        )
        new._grad = self._grad
        new._grad_fn = self._grad_fn
        new._hooks = dict(self._hooks)
        new._next_hook_id = self._next_hook_id
        return new

    def __deepcopy__(self, memo) -> Tensor:
        hit = memo.get(id(self))
        if hit is not None:
            return hit
        if self._grad_fn is not None:
            raise RuntimeError(
                "Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment"
            )
        import copy as _copy

        with _autograd_state.no_grad():
            data = self.clone()
        new = self._shell()
        new._ptr, data._ptr = data._ptr, 0
        memo[id(self)] = new
        if self._grad is not None:
            new._grad = _copy.deepcopy(self._grad, memo)
        return new

    def __reduce_ex__(self, protocol):
        from .serialization import _restore_pickled, _to_serializable

        return (_restore_pickled, (_to_serializable(self),))

    def __del__(self):
        ptr = getattr(self, "_ptr", 0)
        if ptr and _C is not None and getattr(_C, "lib", None) is not None:
            _C.lib.gt_tensor_free(ptr)
            self._ptr = 0
            # that may have been the last user of memory borrowed from a NumPy array
            if _SHARED_ARRAYS and _release_shared_arrays is not None:
                _release_shared_arrays()

    # ---------- Metadata Properties ----------

    @property
    def shape(self) -> _dev.Size:
        nd = _C.lib.gt_tensor_ndim(self._ptr)
        if nd <= 0:
            return _dev.Size(())
        arr = (ctypes.c_int64 * nd)()
        _C.lib.gt_tensor_shape(self._ptr, arr)
        return _dev.Size(int(arr[i]) for i in range(nd))

    def size(self, dim: int | None = None):
        sh = self.shape
        if dim is None:
            return sh
        return sh[dim]

    def stride(self, dim: int | None = None):
        nd = _C.lib.gt_tensor_ndim(self._ptr)
        if nd <= 0:
            st = ()
        else:
            arr = (ctypes.c_int64 * nd)()
            _C.lib.gt_tensor_strides(self._ptr, arr)
            st = tuple(int(arr[i]) for i in range(nd))
        if dim is None:
            return st
        return st[dim]

    @property
    def ndim(self) -> int:
        return int(_C.lib.gt_tensor_ndim(self._ptr))

    def dim(self) -> int:
        return self.ndim

    def ndimension(self) -> int:
        return self.ndim

    def numel(self) -> int:
        return int(_C.lib.gt_tensor_numel(self._ptr))

    def nelement(self) -> int:
        return self.numel()

    def element_size(self) -> int:
        return int(_C.lib.gt_tensor_elem_size(self.dtype.code))

    @property
    def nbytes(self) -> int:
        return int(_C.lib.gt_tensor_nbytes(self._ptr))

    @property
    def dtype(self) -> _dt.dtype:
        return _dt.from_code(int(_C.lib.gt_tensor_dtype(self._ptr)))

    @property
    def device(self) -> _dev.device:
        dt = int(_C.lib.gt_tensor_device_type(self._ptr))
        idx = int(_C.lib.gt_tensor_device_index(self._ptr))
        rev = {0: "cpu", 1: "cuda", 2: "mps", 3: "vulkan", 4: "meta"}
        return _dev.device(rev.get(dt, "cpu"), idx if dt != 0 and dt != 4 else None)

    @property
    def layout(self) -> _dev.layout:
        return _dev.strided

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    @property
    def is_meta(self) -> bool:
        return self.device.type == "meta"

    def is_quantized_tensor(self) -> bool:
        return self.dtype.is_quantized

    def is_floating_point(self) -> bool:
        return self.dtype.is_floating_point

    def is_complex(self) -> bool:
        return self.dtype.is_complex

    def is_contiguous(self, memory_format=None) -> bool:
        (memory_format,)
        return bool(_C.lib.gt_tensor_is_contiguous(self._ptr))

    def data_ptr(self) -> int:
        return int(_C.lib.gt_tensor_data_ptr(self._ptr))

    def storage_offset(self) -> int:
        return int(_C.lib.gt_tensor_offset(self._ptr))

    # ---------- Autograd Properties & Methods ----------

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool) -> None:
        self.requires_grad_(val)

    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        val = bool(requires_grad)
        if val and not (self.dtype.is_floating_point or self.dtype.is_complex):
            raise RuntimeError("only Tensors of floating point and complex dtype can require gradients")
        if not val and not self._is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables.")
        self._requires_grad = val
        return self

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, g) -> None:
        self._grad = g

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def retains_grad(self) -> bool:
        return self._retains_grad

    def retain_grad(self) -> None:
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        self._retains_grad = True

    def register_hook(self, hook):
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
        hid = self._next_hook_id
        self._next_hook_id += 1
        self._hooks[hid] = hook

        class _Handle:
            def __init__(self, tensor_ref, h_id):
                self._t = tensor_ref
                self._id = h_id

            def remove(self):
                self._t._hooks.pop(self._id, None)

        return _Handle(self, hid)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none:
            self._grad = None
        elif self._grad is not None:
            self._grad.zero_()

    def detach(self) -> Tensor:
        ptr = _C.check_ptr(
            _C.lib.gt_tensor_view(self._ptr, self.ndim, _c_i64_arr(self.shape), _c_i64_arr(self.stride()), self.storage_offset()),
            "detach",
        )
        return Tensor(_ptr=ptr, requires_grad=False)

    def detach_(self) -> Tensor:
        self._requires_grad = False
        self._grad_fn = None
        self._is_leaf = True
        return self

    @property
    def data(self) -> Tensor:
        return self.detach()

    @data.setter
    def data(self, other: Tensor) -> None:
        with _autograd_state.no_grad():
            self.copy_(other)

    def _rebind(self, other: Tensor) -> None:
        """Make this object refer to the data of `other`: its storage, shape, dtype and device.

        The Python object stays the one it was, with its requires_grad flag, its .grad and its
        hooks - which is what lets nn.Module.to()/half()/cuda() convert a Parameter without
        replacing it: modules sharing the Parameter keep sharing it, and an optimizer created
        before the conversion keeps updating the parameter the model uses.
        """
        ptr = _C.check_ptr(
            _C.lib.gt_tensor_view(other._ptr, other.ndim, _c_i64_arr(other.shape), _c_i64_arr(other.stride()), other.storage_offset()),
            "rebind",
        )
        old, self._ptr = self._ptr, ptr
        if old:
            _C.lib.gt_tensor_free(old)
            if _SHARED_ARRAYS:
                _release_shared_arrays()
        self._version += 1
        if self._requires_grad and not (self.dtype.is_floating_point or self.dtype.is_complex):
            self._requires_grad = False

    def backward(self, gradient=None, retain_graph: bool | None = None, create_graph: bool = False, inputs=None) -> None:
        _autograd_backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph, inputs=inputs)

    # ---------- Conversion / Extraction ----------

    def clone(self, *, memory_format=None) -> Tensor:
        (memory_format,)
        return CloneFn.apply(self)

    def contiguous(self, memory_format=None) -> Tensor:
        (memory_format,)
        return ContiguousFn.apply(self)

    def to(self, *args, **kwargs) -> Tensor:
        target_dtype = kwargs.get("dtype", None)
        target_device = kwargs.get("device", None)
        copy = kwargs.get("copy", False)
        for a in args:
            if isinstance(a, _dt.dtype):
                target_dtype = a
            elif isinstance(a, _dev.device):
                target_device = a
            elif isinstance(a, Tensor):
                target_dtype = a.dtype
                target_device = a.device
            elif isinstance(a, str):
                target_device = _dev.device(a)
            else:
                raise TypeError(f"Unsupported argument to Tensor.to: {a}")

        out = self
        if target_device is not None:
            td = _dev.as_device(target_device)
            if out.device != td:
                out = ToDeviceFn.apply(out, td)
        if target_dtype is not None:
            tdt = _dt.as_dtype(target_dtype)
            if out.dtype != tdt:
                out = CastFn.apply(out, tdt)
        if copy and out is self:
            out = out.clone()
        return out

    def float(self) -> Tensor:
        return self.to(_dt.float32)

    def double(self) -> Tensor:
        return self.to(_dt.float64)

    def half(self) -> Tensor:
        return self.to(_dt.float16)

    def bfloat16(self) -> Tensor:
        return self.to(_dt.bfloat16)

    def int(self) -> Tensor:
        return self.to(_dt.int32)

    def long(self) -> Tensor:
        return self.to(_dt.int64)

    def short(self) -> Tensor:
        return self.to(_dt.int16)

    def char(self) -> Tensor:
        return self.to(_dt.int8)

    def byte(self) -> Tensor:
        return self.to(_dt.uint8)

    def bool(self) -> Tensor:
        return self.to(_dt.bool)

    def cpu(self) -> Tensor:
        return self.to(_dev.device("cpu"))

    def cuda(self, device=None) -> Tensor:
        idx = 0 if device is None else (device.index if isinstance(device, _dev.device) else int(device))
        return self.to(_dev.device("cuda", idx))

    def mps(self) -> Tensor:
        return self.to(_dev.device("mps"))

    def type(self, dtype=None):
        if dtype is None:
            return f"torch.{self.dtype.name.capitalize()}Tensor"
        if isinstance(dtype, str):
            mapping = {
                "torch.FloatTensor": _dt.float32,
                "torch.DoubleTensor": _dt.float64,
                "torch.HalfTensor": _dt.float16,
                "torch.BFloat16Tensor": _dt.bfloat16,
                "torch.LongTensor": _dt.int64,
                "torch.IntTensor": _dt.int32,
                "torch.ShortTensor": _dt.int16,
                "torch.CharTensor": _dt.int8,
                "torch.ByteTensor": _dt.uint8,
                "torch.BoolTensor": _dt.bool,
            }
            if dtype in mapping:
                return self.to(mapping[dtype])
        return self.to(dtype)

    def type_as(self, other: Tensor) -> Tensor:
        return self.to(other.dtype)

    def quantize(self, qtype) -> Tensor:
        qdt = _dt.as_dtype(qtype)
        if not qdt.is_quantized:
            raise ValueError(f"Expected quantized dtype, got {qdt}")
        return _wrap(_C.lib.gt_quantize(self._ptr, qdt.code), "quantize")

    def dequantize(self, dtype=_dt.float32) -> Tensor:
        tdt = _dt.as_dtype(dtype)
        return _wrap(_C.lib.gt_dequantize(self._ptr, tdt.code), "dequantize")

    def _host_f32_if_needed(self) -> Tensor:
        src = self
        if src.device.type != "cpu":
            src = _wrap(_C.lib.gt_tensor_to_device(src._ptr, _dev.DEV_CPU, 0))
        if src.dtype in (_dt.float16, _dt.bfloat16) or src.dtype.is_quantized:
            src = _wrap(_C.lib.gt_tensor_to_dtype(src._ptr, _dt.float32.code))
        return src

    def item(self):
        if self.numel() != 1:
            raise ValueError(f"only one element tensors can be converted to Python scalars (got {self.numel()} elements)")
        dt = self.dtype
        if dt.is_complex:
            return self.tolist() if self.ndim == 0 else self.reshape(()).tolist()
        if dt == _dt.bool:
            return bool(_C.lib.gt_tensor_item_i64(self._ptr))
        if not dt.is_floating_point and not dt.is_quantized:
            return int(_C.lib.gt_tensor_item_i64(self._ptr))
        return float(_C.lib.gt_tensor_item_f64(self._ptr))

    def tolist(self):
        n = self.numel()
        sh = tuple(self.shape)
        if n == 0:
            return _flat_to_nested([], sh)
        src = self._host_f32_if_needed()
        dt = src.dtype
        es = src.element_size()
        buf = ctypes.create_string_buffer(n * es)
        _C.check_rc(_C.lib.gt_tensor_copy_to_host(src._ptr, buf, n * es), "tolist")
        flat = _unpack_bytes_to_flat(buf.raw, n, dt)
        return _flat_to_nested(flat, sh)

    def numpy(self, *, force: bool = False):
        """This tensor as a NumPy array on the same memory: a write through either shows in both.

        Nothing is copied, whatever the strides of the tensor, and the array keeps the memory
        alive for as long as it or one of its views exists. ``force=True`` gives the sharing up
        where it has to: it detaches a tensor that requires grad, copies one from another
        device, and converts bfloat16 and quantized data - which NumPy cannot hold - to float32.
        """
        import numpy as np

        dt = self.dtype
        foreign = dt == _dt.bfloat16 or dt.is_quantized
        if not force:
            if self.requires_grad:
                raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
            if self.device.type != "cpu":
                raise TypeError(
                    f"can't convert {self.device} device type tensor to numpy. "
                    "Use Tensor.cpu() to copy the tensor to host memory first."
                )
            if foreign:
                raise TypeError(
                    f"Got unsupported ScalarType {dt.name}: NumPy has no such type. Convert the tensor first "
                    "(for example with .float()), or pass force=True for a float32 copy."
                )
        # A handle of its own on the storage: the array stays valid whatever happens to `self`.
        src = self.detach()
        if src.device.type != "cpu":
            src = _wrap(_C.lib.gt_tensor_to_device(src._ptr, _dev.DEV_CPU, 0), "numpy")
        if foreign:
            src = _wrap(_C.lib.gt_tensor_to_dtype(src._ptr, _dt.float32.code), "numpy")

        np_dtype = np.dtype(_NUMPY_NAME_OF_DTYPE[src.dtype])
        shape = tuple(src.shape)
        if src.numel() == 0:
            return np.empty(shape, dtype=np_dtype)
        base = int(_C.lib.gt_tensor_storage_host_ptr(src._ptr))
        nbytes = int(_C.lib.gt_tensor_storage_nbytes(src._ptr))
        if not base or not nbytes:
            raise RuntimeError(f"numpy(): a tensor on device {src.device} has no host memory")
        itemsize = np_dtype.itemsize
        # The whole storage as a buffer; NumPy checks that the view stays inside it. The buffer
        # is the array's base object, and what it holds on to is what the array keeps alive.
        memory = (ctypes.c_char * nbytes).from_address(base)
        memory._storage = src
        return np.ndarray(
            shape,
            dtype=np_dtype,
            buffer=memory,
            offset=src.storage_offset() * itemsize,
            strides=tuple(step * itemsize for step in src.stride()),
        )

    def __array__(self, dtype=None, copy=None):
        """np.asarray(t), np.array(t): the same array as numpy() - shared memory - converted
        or copied only when `dtype` or `copy` ask for it."""
        arr = self.numpy()
        if dtype is not None:
            np = sys.modules["numpy"]
            if arr.dtype != np.dtype(dtype):
                if copy is False:
                    raise ValueError(
                        f"Unable to avoid copy while creating an array as requested: the tensor holds {arr.dtype}, not {np.dtype(dtype)}."
                    )
                return arr.astype(dtype)
        return arr.copy() if copy else arr

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """NumPy ufuncs with a tensor among their operands: np.sin(t), ndarray + t, ndarray @ t.

        Where the ufunc has a tensor counterpart, that computes the result - so it follows the
        dtype rules of tensors, stays on the tensor's device and, for a tensor that requires
        grad, in the autograd graph. Every other call (a ufunc without counterpart, ``out=``,
        ``where=``, ``reduce`` and friends) is computed by NumPy on the tensors' memory, which
        has the limits of numpy(): CPU tensors that do not require grad. Results are tensors,
        except for arrays passed as ``out``.
        """
        if method == "__call__" and out is None and not kwargs:
            fn = _UFUNC_TENSOR_OPS.get(ufunc.__name__)
            if fn is not None:
                operands = [_ufunc_operand(x, self) for x in inputs]
                if not any(x is NotImplemented for x in operands):
                    return fn(*operands)

        def host(x):
            return x.numpy() if isinstance(x, Tensor) else x

        args = [host(x) for x in inputs]
        outs = None
        if out is not None:
            outs = out if isinstance(out, tuple) else (out,)
            for o in outs:
                if isinstance(o, Tensor):
                    o._check_inplace()
            kwargs["out"] = tuple(host(o) for o in outs)
        if method == "at" and isinstance(inputs[0], Tensor):
            inputs[0]._check_inplace()  # ufunc.at writes into its first operand
        result = getattr(ufunc, method)(*args, **kwargs)
        if outs is not None:
            return outs if len(outs) > 1 else outs[0]

        np = sys.modules["numpy"]

        def wrap(r):
            if isinstance(r, (np.ndarray, np.generic)) and r.dtype.name in _NUMPY_DTYPE_NAMES:
                return _tensor_from_ndarray(np.asarray(r), self.device)
            return r

        return tuple(wrap(r) for r in result) if isinstance(result, tuple) else wrap(result)

    # ---------- Python Protocols ----------

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self.shape[0]):
            yield self[i]

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    def __index__(self) -> int:
        if self.dtype.is_floating_point or self.dtype.is_complex:
            raise TypeError("only integer tensors of a single element can be converted to an index")
        return int(self.item())

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __repr__(self) -> str:
        return _format_tensor_str(self)

    def __str__(self) -> str:
        return _format_tensor_str(self)

    # ---------- Indexing / Slicing ----------

    def _parse_view_index(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        # Check for advanced indexing (Tensor / list)
        for item in idx:
            if isinstance(item, (Tensor, list)):
                return None

        # Expand Ellipsis
        n_ell = sum(1 for item in idx if item is Ellipsis)
        if n_ell > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        # None, True and False add a dimension; everything else uses one up
        n_consuming = sum(1 for item in idx if item is not None and item is not Ellipsis and type(item) is not bool)
        if n_consuming > self.ndim:
            raise IndexError(f"too many indices for tensor of dimension {self.ndim}")

        expanded = []
        for item in idx:
            if item is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - n_consuming))
            else:
                expanded.append(item)
        while sum(1 for item in expanded if item is not None and type(item) is not bool) < self.ndim:
            expanded.append(slice(None))

        cur_shape = self.shape
        cur_strides = self.stride()
        offset = self.storage_offset()
        new_shape = []
        new_strides = []
        dim_i = 0

        for item in expanded:
            if item is None:
                new_shape.append(1)
                new_strides.append(0)
            elif type(item) is bool:
                # a 0-d boolean mask (numpy, torch): True keeps everything in a new dimension of
                # size 1, False selects nothing. Not the integers 1 and 0.
                new_shape.append(1 if item else 0)
                new_strides.append(0)
            elif isinstance(item, int):
                sz = cur_shape[dim_i]
                i = item if item >= 0 else item + sz
                if i < 0 or i >= sz:
                    raise IndexError(f"index {item} is out of bounds for dimension {dim_i} with size {sz}")
                offset += i * cur_strides[dim_i]
                dim_i += 1
            elif isinstance(item, slice):
                sz = cur_shape[dim_i]
                start, stop, step = item.indices(sz)
                if step < 0:
                    # Handle negative step via SliceViewFn on positive flip, fallback None
                    return None
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                length = max(0, (stop - start + step - 1) // step)
                offset += start * cur_strides[dim_i]
                new_shape.append(length)
                new_strides.append(cur_strides[dim_i] * step)
                dim_i += 1
            else:
                return None

        return tuple(new_shape), tuple(new_strides), int(offset)

    def _split_advanced_index(self, idx):
        """Split an index that contains tensors or lists into a basic part and index tensors.

        Returns ``(basic, dims, tensors)``. ``basic`` (ints, slices, None; every advanced
        entry replaced by full slices) is applied first and yields a view; ``tensors[i]``
        then indexes dimension ``dims[i]`` of that view. ``basic`` is None when it would
        select everything. A boolean mask stands for the coordinates of its True elements,
        one integer tensor per dimension it covers, and must match those dimensions exactly.
        """
        items = list(idx) if isinstance(idx, tuple) else [idx]
        if sum(1 for it in items if it is Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        for i, it in enumerate(items):
            if isinstance(it, bool):
                raise IndexError("a bare True/False is not supported as an index; use a boolean tensor")
            if isinstance(it, list):
                items[i] = _index_list_to_tensor(it, self.device)
            elif isinstance(it, Tensor):
                if it.dtype != _dt.bool and (it.dtype.is_floating_point or it.dtype.is_complex):
                    raise IndexError("tensors used as indices must be integer or boolean tensors")
                if it.dtype == _dt.bool and it.ndim == 0:
                    raise IndexError("0-dimensional boolean index tensors are not supported")
            elif not (it is None or it is Ellipsis or isinstance(it, (int, slice))):
                raise IndexError(
                    f"only integers, slices, ellipsis, None and integer or boolean tensors are valid indices (got {type(it).__name__})"
                )

        def consumed(it) -> int:
            if it is None or it is Ellipsis:
                return 0
            return it.ndim if isinstance(it, Tensor) and it.dtype == _dt.bool else 1

        n_consumed = sum(consumed(it) for it in items)
        if n_consumed > self.ndim:
            raise IndexError(f"too many indices for tensor of dimension {self.ndim}")
        expanded = []
        for it in items:
            if it is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - n_consumed))
            else:
                expanded.append(it)

        # NumPy/PyTorch rule: once an index holds an array, its integers are array indices as
        # well (0-d ones). That changes the result only when the array indices are not all
        # next to each other - the indexed dimensions then come first - so only then are the
        # integers turned into index tensors. Otherwise taking them first, as part of the view,
        # gives the same result for less work.
        advanced = [k for k, it in enumerate(expanded) if isinstance(it, (int, Tensor))]
        if (
            advanced
            and advanced[-1] - advanced[0] + 1 != len(advanced)
            and any(isinstance(expanded[k], int) for k in advanced)
            and any(isinstance(expanded[k], Tensor) for k in advanced)
        ):
            expanded = [
                _from_python(it, dtype=_dt.int64, device=self.device) if isinstance(it, int) else it for it in expanded
            ]

        basic, dims, tensors = [], [], []
        src_dim = 0   # dimension of self the next entry applies to
        view_dim = 0  # the same position in the view that `basic` produces
        for it in expanded:
            if it is None:
                basic.append(None)
                view_dim += 1
            elif isinstance(it, int):
                basic.append(it)
                src_dim += 1
            elif isinstance(it, slice):
                basic.append(it)
                src_dim += 1
                view_dim += 1
            elif it.dtype == _dt.bool:
                for k in range(it.ndim):
                    if it.shape[k] != self.shape[src_dim + k]:
                        raise IndexError(
                            f"The shape of the mask {list(it.shape)} at index {k} does not match the shape of the "
                            f"indexed tensor {list(self.shape)} at index {src_dim + k}"
                        )
                for k, coord in enumerate(it.nonzero(as_tuple=True)):
                    basic.append(slice(None))
                    dims.append(view_dim + k)
                    tensors.append(coord)
                src_dim += it.ndim
                view_dim += it.ndim
            else:
                basic.append(slice(None))
                dims.append(view_dim)
                tensors.append(it)
                src_dim += 1
                view_dim += 1
        trivial = all(isinstance(b, slice) and b == slice(None) for b in basic)
        return (None if trivial else tuple(basic)), dims, tensors

    def __getitem__(self, idx):
        idx = _normalize_index(idx, self.device)
        # A boolean mask of the tensor's own shape: one pass in C
        if isinstance(idx, Tensor) and idx.dtype == _dt.bool and idx.ndim > 0 and idx.shape == self.shape:
            return MaskedSelectFn.apply(self, idx)

        parsed = self._parse_view_index(idx)
        if parsed is not None:
            new_shape, new_strides, offset = parsed
            return SliceViewFn.apply(self, new_shape, new_strides, offset, idx)

        # Negative slice step handling
        idx_tup = idx if isinstance(idx, tuple) else (idx,)
        if any(isinstance(it, slice) and it.step is not None and it.step < 0 for it in idx_tup):
            # Convert negative step slices by flipping those dims first
            res = self
            new_idx = []
            dim_i = 0
            for it in idx_tup:
                if it is None:
                    new_idx.append(None)
                elif isinstance(it, slice) and it.step is not None and it.step < 0:
                    sz = res.shape[dim_i]
                    res = res.flip((dim_i,))
                    st = it.start
                    sp = it.stop
                    # Map [start:stop:step] on reversed axis
                    start_rev = 0 if st is None else (sz - 1 - (st if st >= 0 else st + sz))
                    stop_rev = sz if sp is None else (sz - 1 - (sp if sp >= 0 else sp + sz))
                    new_idx.append(slice(max(start_rev, 0), max(stop_rev, 0), -it.step))
                    dim_i += 1
                elif it is Ellipsis:
                    new_idx.append(Ellipsis)
                    dim_i += self.ndim - sum(1 for x in idx_tup if x is not None and x is not Ellipsis)
                else:
                    new_idx.append(it)
                    dim_i += 1
            return res[tuple(new_idx)]

        # Advanced indexing with tensors, lists and boolean masks: slices and ints first, as a
        # view, then the index tensors - through a Function, so gradients flow back.
        basic, dims, tensors = self._split_advanced_index(idx)
        base = self[basic] if basic is not None else self
        if not tensors:
            return base
        return IndexTensorFn.apply(base, tuple(dims), *tensors)

    def __setitem__(self, idx, value):
        # Writes into the tensor's storage: refused for a leaf that requires grad (as every
        # other in-place operation is), and the version counter moves on.
        self._check_inplace()
        idx = _normalize_index(idx, self.device)
        val_t = _ensure_tensor(value, self)
        if val_t.dtype != self.dtype:
            val_t = val_t.to(self.dtype)

        if isinstance(idx, Tensor) and idx.dtype == _dt.bool and idx.ndim > 0 and idx.shape == self.shape and val_t.numel() == 1:
            _C.check_rc(_C.lib.gt_masked_fill_inplace(self._ptr, idx._ptr, float(val_t.item())), "setitem mask")
            return

        parsed = self._parse_view_index(idx)
        if parsed is not None:
            new_shape, new_strides, offset = parsed
            v = Tensor(_ptr=_C.check_ptr(_C.lib.gt_tensor_view(self._ptr, len(new_shape), _c_i64_arr(new_shape), _c_i64_arr(new_strides), offset)))
            if val_t.shape != v.shape:
                val_t = _broadcast_value(val_t, v.shape)
            _C.check_rc(_C.lib.gt_tensor_copy_inplace(v._ptr, val_t._ptr), "setitem copy")
            return

        basic, dims, tensors = self._split_advanced_index(idx)
        if not tensors:
            raise IndexError(f"Unsupported __setitem__ index: {idx}")
        target = self
        if basic is not None:
            # ints and slices select a view of this tensor; the index tensors then write into it
            view = self._parse_view_index(basic)
            if view is None:
                raise IndexError("a slice with a negative step cannot be combined with tensor indices in an assignment")
            new_shape, new_strides, offset = view
            target = Tensor(_ptr=_C.check_ptr(_C.lib.gt_tensor_view(self._ptr, len(new_shape), _c_i64_arr(new_shape), _c_i64_arr(new_strides), offset)))
        result_shape = _advanced_result_shape(target.shape, dims, [ti.shape for ti in tensors])
        if val_t.shape != result_shape:
            val_t = _broadcast_value(val_t, result_shape)
        p_arr = (ctypes.c_void_p * len(tensors))(*[ti._ptr for ti in tensors])
        _C.check_rc(_C.lib.gt_index_put(target._ptr, len(tensors), p_arr, _c_i32_arr(dims), val_t._ptr, 0), "index_put")

    # ---------- Shape / View / Splitting ----------

    def view(self, *shape) -> Tensor:
        """The same elements seen in another shape - always sharing memory with `self`, never a
        copy. When the strides of `self` do not allow that (after a transpose, say), view()
        raises; reshape() is the method that copies when it has to."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        sh = self._resolve_neg1_shape(shape)
        if _can_view_reshape(self.shape, self.stride(), sh) is None:
            raise RuntimeError(
                "view size is not compatible with input tensor's size and stride (at least one dimension "
                "spans across two contiguous subspaces). Use .reshape(...) instead."
            )
        return ReshapeFn.apply(self, sh)

    def view_as(self, other: Tensor) -> Tensor:
        return self.view(other.shape)

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        sh = self._resolve_neg1_shape(shape)
        return ReshapeFn.apply(self, sh)

    def reshape_as(self, other: Tensor) -> Tensor:
        return self.reshape(other.shape)

    def _resolve_neg1_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        n = self.numel()
        neg_idx = -1
        known = 1
        out = list(shape)
        for i, s in enumerate(out):
            s = int(s)
            out[i] = s
            if s == -1:
                if neg_idx != -1:
                    raise RuntimeError("only one dimension can be inferred")
                neg_idx = i
            elif s < -1:
                raise RuntimeError(f"invalid shape dimension {s}")
            else:
                known *= s
        if neg_idx != -1:
            if known == 0:
                if n != 0:
                    raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {n}")
                raise RuntimeError(
                    f"cannot reshape tensor of 0 elements into shape {list(shape)} because the unspecified "
                    "dimension size -1 can be any value and is ambiguous"
                )
            if n % known != 0:
                raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {n}")
            out[neg_idx] = n // known
        elif known != n:
            # without this a view of the wrong size was handed out: fewer elements than the
            # tensor has, or more - reaching past the end of its memory
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {n}")
        return tuple(out)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        nd = self.ndim
        if nd == 0:
            return self.reshape((1,))
        s = start_dim if start_dim >= 0 else start_dim + nd
        e = end_dim if end_dim >= 0 else end_dim + nd
        flat_size = 1
        for d in range(s, e + 1):
            flat_size *= self.shape[d]
        new_sh = self.shape[:s] + (flat_size,) + self.shape[e + 1 :]
        return self.reshape(new_sh)

    def unflatten(self, dim: int, sizes: tuple[int, ...]) -> Tensor:
        nd = self.ndim
        d = dim if dim >= 0 else dim + nd
        new_sh = self.shape[:d] + tuple(sizes) + self.shape[d + 1 :]
        return self.reshape(new_sh)

    def squeeze(self, dim: int | tuple[int, ...] | None = None) -> Tensor:
        sh = self.shape
        if dim is None:
            new_sh = tuple(s for s in sh if s != 1)
            return self.reshape(new_sh) if new_sh != sh else self.view(sh)
        dims = _dims_tuple(dim)
        norm = {d if d >= 0 else d + self.ndim for d in dims}
        new_sh = tuple(s for i, s in enumerate(sh) if not (i in norm and s == 1))
        return self.reshape(new_sh)

    def unsqueeze(self, dim: int) -> Tensor:
        nd = self.ndim
        d = dim if dim >= 0 else dim + nd + 1
        sh = list(self.shape)
        sh.insert(d, 1)
        return self.reshape(tuple(sh))

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return TransposeFn.apply(self, dim0, dim1)

    def swapdims(self, dim0: int, dim1: int) -> Tensor:
        return self.transpose(dim0, dim1)

    def swapaxes(self, axis0: int, axis1: int) -> Tensor:
        return self.transpose(axis0, axis1)

    def t(self) -> Tensor:
        if self.ndim > 2:
            raise RuntimeError("t() expects a tensor with <= 2 dimensions")
        if self.ndim < 2:
            return self.view(self.shape)
        return self.transpose(0, 1)

    @property
    def T(self) -> Tensor:
        if self.ndim == 0:
            return self
        return self.permute(*reversed(range(self.ndim)))

    @property
    def mT(self) -> Tensor:
        if self.ndim < 2:
            raise RuntimeError("mT expects at least 2 dimensions")
        return self.transpose(-2, -1)

    def permute(self, *dims) -> Tensor:
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return PermuteFn.apply(self, tuple(dims))

    def movedim(self, source, destination) -> Tensor:
        if _is_int(source):
            source = (source,)
        if _is_int(destination):
            destination = (destination,)
        nd = self.ndim
        src = [s if s >= 0 else s + nd for s in source]
        dst = [d if d >= 0 else d + nd for d in destination]
        order = [n for n in range(nd) if n not in src]
        for d, s in sorted(zip(dst, src)):
            order.insert(d, s)
        return self.permute(*order)

    def moveaxis(self, source, destination) -> Tensor:
        return self.movedim(source, destination)

    def expand(self, *sizes) -> Tensor:
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        return ExpandFn.apply(self, tuple(sizes))

    def expand_as(self, other: Tensor) -> Tensor:
        return self.expand(other.shape)

    def repeat(self, *repeats) -> Tensor:
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = tuple(repeats[0])
        x = self
        while x.ndim < len(repeats):
            x = x.unsqueeze(0)
        for d, r in enumerate(repeats):
            if r != 1:
                x = CatFn.apply(d, *([x] * r))
        return x

    def repeat_interleave(self, repeats: int, dim: int | None = None) -> Tensor:
        if dim is None:
            flat = self.flatten()
            return flat.repeat_interleave(repeats, dim=0)
        d = dim if dim >= 0 else dim + self.ndim
        sh = self.shape
        x = self.unsqueeze(d + 1)
        exp_sh = sh[: d + 1] + (repeats,) + sh[d + 1 :]
        x = x.expand(exp_sh)
        out_sh = sh[:d] + (sh[d] * repeats,) + sh[d + 1 :]
        return x.reshape(out_sh)

    def narrow(self, dim: int, start: int, length: int) -> Tensor:
        d = dim if dim >= 0 else dim + self.ndim
        slices = [slice(None)] * self.ndim
        slices[d] = slice(start, start + length)
        return self[tuple(slices)]

    def select(self, dim: int, index: int) -> Tensor:
        d = dim if dim >= 0 else dim + self.ndim
        slices = [slice(None)] * self.ndim
        slices[d] = index
        return self[tuple(slices)]

    def split(self, split_size_or_sections, dim: int = 0):
        d = dim if dim >= 0 else dim + self.ndim
        total = self.shape[d]
        if _is_int(split_size_or_sections):
            step = split_size_or_sections
            sizes = [step] * (total // step)
            if total % step:
                sizes.append(total % step)
        else:
            sizes = list(split_size_or_sections)
        out = []
        offset = 0
        for sz in sizes:
            out.append(self.narrow(d, offset, sz))
            offset += sz
        return tuple(out)

    def chunk(self, chunks: int, dim: int = 0):
        d = dim if dim >= 0 else dim + self.ndim
        total = self.shape[d]
        chunk_sz = max(1, (total + chunks - 1) // chunks)
        return self.split(chunk_sz, dim=d)

    def tensor_split(self, indices_or_sections, dim: int = 0):
        d = dim if dim >= 0 else dim + self.ndim
        total = self.shape[d]
        if _is_int(indices_or_sections):
            n = indices_or_sections
            q, r = divmod(total, n)
            sizes = [q + (1 if i < r else 0) for i in range(n)]
            return self.split(sizes, dim=d)
        idx = [0] + list(indices_or_sections) + [total]
        return tuple(self.narrow(d, idx[i], max(0, idx[i + 1] - idx[i])) for i in range(len(idx) - 1))

    def unbind(self, dim: int = 0):
        d = dim if dim >= 0 else dim + self.ndim
        return tuple(self.select(d, i) for i in range(self.shape[d]))

    # ---------- Operators & Math ----------

    def add(self, other, *, alpha: float = 1.0) -> Tensor:
        if _is_real_number(other):
            if alpha == 1.0:
                return BinaryScalarFn.apply(self, float(other), _C.BIN_ADD, 0)
            return BinaryScalarFn.apply(self, float(other) * alpha, _C.BIN_ADD, 0)
        return AddFn.apply(self, _ensure_tensor(other, self), float(alpha))

    def sub(self, other, *, alpha: float = 1.0) -> Tensor:
        if _is_real_number(other):
            return BinaryScalarFn.apply(self, float(other) * alpha, _C.BIN_SUB, 0)
        return SubFn.apply(self, _ensure_tensor(other, self), float(alpha))

    def mul(self, other) -> Tensor:
        if _is_real_number(other):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_MUL, 0)
        return MulFn.apply(self, _ensure_tensor(other, self))

    def div(self, other, *, rounding_mode: str | None = None) -> Tensor:
        if rounding_mode == "floor":
            return self // other
        if rounding_mode == "trunc":
            q = self / other
            return q.trunc()
        if _is_real_number(other):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_DIV, 0)
        return DivFn.apply(self, _ensure_tensor(other, self))

    def true_divide(self, other) -> Tensor:
        return self.div(other)

    def floor_divide(self, other) -> Tensor:
        return self // other

    def pow(self, exponent) -> Tensor:
        (self,) = _autocast_float32(self)
        if _is_real_number(exponent):
            return BinaryScalarFn.apply(self, float(exponent), _C.BIN_POW, 0)
        (exponent,) = _autocast_float32(_ensure_tensor(exponent, self))
        return PowFn.apply(self, exponent)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        if _is_real_number(other):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_SUB, 1)
        return SubFn.apply(_ensure_tensor(other, self), self, 1.0)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        if _is_real_number(other):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_DIV, 1)
        return DivFn.apply(_ensure_tensor(other, self), self)

    def __floordiv__(self, other):
        if _is_real_number(other):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 0, _C.BIN_FLOOR_DIV))
        return _wrap(_C.lib.gt_binary_op(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_FLOOR_DIV, 1.0))

    def __rfloordiv__(self, other):
        if _is_real_number(other):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 1, _C.BIN_FLOOR_DIV))
        return _wrap(_C.lib.gt_binary_op(_ensure_tensor(other, self)._ptr, self._ptr, _C.BIN_FLOOR_DIV, 1.0))

    def __mod__(self, other):
        if _is_real_number(other):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 0, _C.BIN_REMAINDER))
        return _wrap(_C.lib.gt_binary_op(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_REMAINDER, 1.0))

    def __rmod__(self, other):
        if _is_real_number(other):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 1, _C.BIN_REMAINDER))
        return _wrap(_C.lib.gt_binary_op(_ensure_tensor(other, self)._ptr, self._ptr, _C.BIN_REMAINDER, 1.0))

    def __pow__(self, other):
        return self.pow(other)

    def __rpow__(self, other):
        (self,) = _autocast_float32(self)
        if _is_real_number(other):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_POW, 1)
        (other,) = _autocast_float32(_ensure_tensor(other, self))
        return PowFn.apply(other, self)

    def __neg__(self):
        return self.neg()

    def __pos__(self):
        return self

    def __abs__(self):
        return self.abs()

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        return _ensure_tensor(other, self).matmul(self)

    def __invert__(self):
        op = _C.UNARY_LOGICAL_NOT if self.dtype == _dt.bool else _C.UNARY_BITWISE_NOT
        return _wrap(_C.lib.gt_unary_op(self._ptr, op, 0.0, 0.0))

    def _cmp_op(self, other, code: int) -> Tensor:
        if _is_scalar(other):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 0, code))
        return _wrap(_C.lib.gt_binary_op(self._ptr, _ensure_tensor(other, self)._ptr, code, 1.0))

    def __eq__(self, other):
        return self._cmp_op(other, _C.BIN_EQ)

    def __ne__(self, other):
        return self._cmp_op(other, _C.BIN_NE)

    def __lt__(self, other):
        return self._cmp_op(other, _C.BIN_LT)

    def __le__(self, other):
        return self._cmp_op(other, _C.BIN_LE)

    def __gt__(self, other):
        return self._cmp_op(other, _C.BIN_GT)

    def __ge__(self, other):
        return self._cmp_op(other, _C.BIN_GE)

    def __and__(self, other):
        code = _C.BIN_LOGICAL_AND if self.dtype == _dt.bool else _C.BIN_BITWISE_AND
        return self._cmp_op(other, code)

    def __or__(self, other):
        code = _C.BIN_LOGICAL_OR if self.dtype == _dt.bool else _C.BIN_BITWISE_OR
        return self._cmp_op(other, code)

    def __xor__(self, other):
        code = _C.BIN_LOGICAL_XOR if self.dtype == _dt.bool else _C.BIN_BITWISE_XOR
        return self._cmp_op(other, code)

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def eq(self, other):
        return self == other

    def ne(self, other):
        return self != other

    def lt(self, other):
        return self < other

    def le(self, other):
        return self <= other

    def gt(self, other):
        return self > other

    def ge(self, other):
        return self >= other

    def equal(self, other: Tensor) -> bool:
        if not isinstance(other, Tensor) or self.shape != other.shape or self.dtype != other.dtype:
            return False
        if self.numel() == 0:
            return True
        return bool((self == other).all().item())

    def allclose(self, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
        diff = (self - other).abs()
        tol = atol + rtol * other.abs()
        ok = diff <= tol
        if equal_nan:
            both_nan = self.isnan() & other.isnan()
            ok = ok | both_nan
        return bool(ok.all().item())

    # ---------- In-place Mutations ----------

    def _check_inplace(self):
        if _autograd_state.is_grad_enabled() and self.requires_grad and self._is_leaf:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
        self._version += 1

    def copy_(self, src: Tensor, non_blocking: bool = False) -> Tensor:
        (non_blocking,)
        self._check_inplace()
        s = _ensure_tensor(src, self)
        if s.shape != self.shape:
            s = s.expand(self.shape)
        _C.check_rc(_C.lib.gt_tensor_copy_inplace(self._ptr, s._ptr), "copy_")
        return self

    def fill_(self, value) -> Tensor:
        self._check_inplace()
        value = _unwrap_numpy_scalar(value)
        if isinstance(value, (int, bool)) and not self.dtype.is_floating_point:
            _C.check_rc(_C.lib.gt_fill_scalar_i64(self._ptr, int(value)), "fill_")
        else:
            _C.check_rc(_C.lib.gt_fill_scalar_f64(self._ptr, float(value)), "fill_")
        return self

    def zero_(self) -> Tensor:
        return self.fill_(0)

    def uniform_(self, from_: float = 0.0, to: float = 1.0) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_fill_uniform(self._ptr, float(from_), float(to)), "uniform_")
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_fill_normal(self._ptr, float(mean), float(std)), "normal_")
        return self

    def trunc_normal_(self, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_fill_trunc_normal(self._ptr, float(mean), float(std), float(a), float(b)), "trunc_normal_")
        return self

    def bernoulli_(self, p: float = 0.5) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_fill_bernoulli_prob(self._ptr, float(p)), "bernoulli_")
        return self

    def bernoulli(self) -> Tensor:
        return _wrap(_C.lib.gt_bernoulli_tensor(self._ptr), "bernoulli")

    def multinomial(self, num_samples: int, replacement: bool = False) -> Tensor:
        return _wrap(_C.lib.gt_multinomial(self._ptr, int(num_samples), 1 if replacement else 0), "multinomial")

    def add_(self, other, *, alpha: float = 1.0) -> Tensor:
        self._check_inplace()
        if _is_scalar(other):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_ADD, float(alpha)), "add_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_ADD, float(alpha)), "add_")
        return self

    def sub_(self, other, *, alpha: float = 1.0) -> Tensor:
        self._check_inplace()
        if _is_scalar(other):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_SUB, float(alpha)), "sub_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_SUB, float(alpha)), "sub_")
        return self

    def mul_(self, other) -> Tensor:
        self._check_inplace()
        if _is_scalar(other):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_MUL, 1.0), "mul_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_MUL, 1.0), "mul_")
        return self

    def div_(self, other) -> Tensor:
        self._check_inplace()
        if _is_scalar(other):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_DIV, 1.0), "div_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_DIV, 1.0), "div_")
        return self

    def __iadd__(self, other):
        return self.add_(other)

    def __isub__(self, other):
        return self.sub_(other)

    def __imul__(self, other):
        return self.mul_(other)

    def __itruediv__(self, other):
        return self.div_(other)

    def clamp_(self, min=None, max=None) -> Tensor:
        self._check_inplace()
        a = float(min) if min is not None else float("nan")
        b = float(max) if max is not None else float("nan")
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_CLAMP, a, b), "clamp_")
        return self

    def clip_(self, min=None, max=None) -> Tensor:
        return self.clamp_(min=min, max=max)

    def relu_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_RELU, 0.0, 0.0), "relu_")
        return self

    def sigmoid_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_SIGMOID, 0.0, 0.0), "sigmoid_")
        return self

    def tanh_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_TANH, 0.0, 0.0), "tanh_")
        return self

    def neg_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_NEG, 0.0, 0.0), "neg_")
        return self

    def abs_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_ABS, 0.0, 0.0), "abs_")
        return self

    def sqrt_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_SQRT, 0.0, 0.0), "sqrt_")
        return self

    def exp_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_EXP, 0.0, 0.0), "exp_")
        return self

    def log_(self) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_unary_inplace(self._ptr, _C.UNARY_LOG, 0.0, 0.0), "log_")
        return self

    def masked_fill_(self, mask: Tensor, value: float) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_masked_fill_inplace(self._ptr, mask._ptr, float(value)), "masked_fill_")
        return self

    def masked_fill(self, mask: Tensor, value: float) -> Tensor:
        return WhereFn.apply(mask, _from_python(float(value), dtype=self.dtype, device=self.device), self)

    def masked_select(self, mask: Tensor) -> Tensor:
        if not isinstance(mask, Tensor) or mask.dtype != _dt.bool:
            raise RuntimeError("masked_select: expected a boolean tensor for mask")
        return MaskedSelectFn.apply(self, mask)

    def scatter_(self, dim: int, index: Tensor, src) -> Tensor:
        self._check_inplace()
        src_t = _ensure_tensor(src, self)
        _C.check_rc(_C.lib.gt_scatter_inplace(self._ptr, dim if dim >= 0 else dim + self.ndim, index._ptr, src_t._ptr, 0), "scatter_")
        return self

    def scatter_add_(self, dim: int, index: Tensor, src: Tensor) -> Tensor:
        self._check_inplace()
        _C.check_rc(_C.lib.gt_scatter_inplace(self._ptr, dim if dim >= 0 else dim + self.ndim, index._ptr, src._ptr, 1), "scatter_add_")
        return self

    # ---------- Differentiable Unary Methods ----------

    def neg(self):
        return UnarySimpleFn.apply(self, _C.UNARY_NEG)

    def abs(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ABS)

    def sign(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SIGN)

    def sqrt(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SQRT)

    def rsqrt(self):
        return UnarySimpleFn.apply(self, _C.UNARY_RSQRT)

    def square(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SQUARE)

    def exp(self):
        return UnarySimpleFn.apply(self, _C.UNARY_EXP)

    def expm1(self):
        return UnarySimpleFn.apply(self, _C.UNARY_EXPM1)

    def exp2(self):
        return UnarySimpleFn.apply(self, _C.UNARY_EXP2)

    def log(self):
        return UnarySimpleFn.apply(self, _C.UNARY_LOG)

    def log2(self):
        return UnarySimpleFn.apply(self, _C.UNARY_LOG2)

    def log10(self):
        return UnarySimpleFn.apply(self, _C.UNARY_LOG10)

    def log1p(self):
        return UnarySimpleFn.apply(self, _C.UNARY_LOG1P)

    def sin(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SIN)

    def cos(self):
        return UnarySimpleFn.apply(self, _C.UNARY_COS)

    def tan(self):
        return UnarySimpleFn.apply(self, _C.UNARY_TAN)

    def asin(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ASIN)

    def acos(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ACOS)

    def atan(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ATAN)

    def atan2(self, other):
        return Atan2Fn.apply(self, _ensure_tensor(other, self))

    def sinh(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SINH)

    def cosh(self):
        return UnarySimpleFn.apply(self, _C.UNARY_COSH)

    def tanh(self):
        return UnarySimpleFn.apply(self, _C.UNARY_TANH)

    def asinh(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ASINH)

    def acosh(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ACOSH)

    def atanh(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ATANH)

    def erf(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ERF)

    def erfc(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ERFC)

    def erfinv(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_ERFINV, 0.0, 0.0))

    def lgamma(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_LGAMMA, 0.0, 0.0))

    def digamma(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_DIGAMMA, 0.0, 0.0))

    def floor(self):
        return UnarySimpleFn.apply(self, _C.UNARY_FLOOR)

    def ceil(self):
        return UnarySimpleFn.apply(self, _C.UNARY_CEIL)

    def round(self):
        return UnarySimpleFn.apply(self, _C.UNARY_ROUND)

    def trunc(self):
        return UnarySimpleFn.apply(self, _C.UNARY_TRUNC)

    def frac(self):
        return UnarySimpleFn.apply(self, _C.UNARY_FRAC)

    def reciprocal(self):
        return UnarySimpleFn.apply(self, _C.UNARY_RECIPROCAL)

    def relu(self):
        return UnarySimpleFn.apply(self, _C.UNARY_RELU)

    def sigmoid(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SIGMOID)

    def silu(self):
        return UnarySimpleFn.apply(self, _C.UNARY_SILU)

    def gelu(self):
        return UnarySimpleFn.apply(self, _C.UNARY_GELU)

    def clamp(self, min=None, max=None):
        if isinstance(min, Tensor) or isinstance(max, Tensor):
            return ClampTensorFn.apply(
                self,
                min if isinstance(min, Tensor) else None,
                max if isinstance(max, Tensor) else None,
                None if (min is None or isinstance(min, Tensor)) else float(min),
                None if (max is None or isinstance(max, Tensor)) else float(max),
            )
        a = float(min) if min is not None else float("nan")
        b = float(max) if max is not None else float("nan")
        return UnarySimpleFn.apply(self, _C.UNARY_CLAMP, a, b)

    def clip(self, min=None, max=None):
        return self.clamp(min=min, max=max)

    def isnan(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_ISNAN, 0.0, 0.0))

    def isinf(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_ISINF, 0.0, 0.0))

    def isfinite(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_ISFINITE, 0.0, 0.0))

    def logical_not(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_LOGICAL_NOT, 0.0, 0.0))

    def nan_to_num(self, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None):
        if posinf is None or neginf is None:
            if self.dtype.is_floating_point:
                info = _dt.finfo(self.dtype)
                pos_val = info.max if posinf is None else float(posinf)
                neg_val = info.min if neginf is None else float(neginf)
            else:
                pos_val = 0.0 if posinf is None else float(posinf)
                neg_val = 0.0 if neginf is None else float(neginf)
        else:
            pos_val = float(posinf)
            neg_val = float(neginf)
        from ._ops import where
        out = where(self.isnan(), float(nan), self)
        out = where(self == math.inf, pos_val, out)
        out = where(self == -math.inf, neg_val, out)
        return out

    @property
    def real(self):
        return RealFn.apply(self)

    @property
    def imag(self):
        return ImagFn.apply(self)

    def conj(self):
        return ConjFn.apply(self)

    def angle(self):
        return _wrap(_C.lib.gt_unary_op(self._ptr, _C.UNARY_ANGLE, 0.0, 0.0))

    def maximum(self, other):
        return MaximumFn.apply(self, _ensure_tensor(other, self))

    def minimum(self, other):
        return MinimumFn.apply(self, _ensure_tensor(other, self))

    def where(self, condition: Tensor, other):
        return WhereFn.apply(condition, self, _ensure_tensor(other, self))

    def lerp(self, end: Tensor, weight):
        w = weight if isinstance(weight, Tensor) else _from_python(float(weight), dtype=self.dtype, device=self.device)
        return LerpFn.apply(self, end, w)

    # ---------- Reductions & Linear Algebra ----------

    def _norm_dims(self, dim):
        if dim is None:
            return ()
        return _dims_tuple(dim)

    @_numpy_style_reduction
    def sum(self, dim=None, keepdim: bool = False, *, dtype=None):
        (x,) = _autocast_float32(self) if dtype is None else (self.to(dtype),)
        return ReduceFn.apply(x, _C.RED_SUM, self._norm_dims(dim), keepdim)

    @_numpy_style_reduction
    def mean(self, dim=None, keepdim: bool = False, *, dtype=None):
        x = self if dtype is None else self.to(dtype)
        return ReduceFn.apply(x, _C.RED_MEAN, self._norm_dims(dim), keepdim)

    @_numpy_style_reduction
    def prod(self, dim=None, keepdim: bool = False, *, dtype=None):
        (x,) = _autocast_float32(self) if dtype is None else (self.to(dtype),)
        return ReduceFn.apply(x, _C.RED_PROD, self._norm_dims(dim), keepdim)

    def logsumexp(self, dim, keepdim: bool = False):
        (x,) = _autocast_float32(self)
        return ReduceFn.apply(x, _C.RED_LOGSUMEXP, self._norm_dims(dim), keepdim)

    @_numpy_style_reduction
    def max(self, dim: int | None = None, keepdim: bool = False):
        if dim is None:
            return ReduceFn.apply(self, _C.RED_MAX, (), False)
        d = dim if dim >= 0 else dim + self.ndim
        vals = ReduceFn.apply(self, _C.RED_MAX, (d,), keepdim)
        idxs = _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMAX, 1, _c_i32_arr((d,)), 1 if keepdim else 0, -1))
        return vals, idxs

    @_numpy_style_reduction
    def min(self, dim: int | None = None, keepdim: bool = False):
        if dim is None:
            return ReduceFn.apply(self, _C.RED_MIN, (), False)
        d = dim if dim >= 0 else dim + self.ndim
        vals = ReduceFn.apply(self, _C.RED_MIN, (d,), keepdim)
        idxs = _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMIN, 1, _c_i32_arr((d,)), 1 if keepdim else 0, -1))
        return vals, idxs

    @_numpy_style_reduction
    def argmax(self, dim: int | None = None, keepdim: bool = False):
        dims = () if dim is None else (dim if dim >= 0 else dim + self.ndim,)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMAX, len(dims), arr, 1 if keepdim else 0, -1))

    @_numpy_style_reduction
    def argmin(self, dim: int | None = None, keepdim: bool = False):
        dims = () if dim is None else (dim if dim >= 0 else dim + self.ndim,)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMIN, len(dims), arr, 1 if keepdim else 0, -1))

    @_numpy_style_reduction
    def all(self, dim=None, keepdim: bool = False):
        dims = self._norm_dims(dim)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ALL, len(dims), arr, 1 if keepdim else 0, -1))

    @_numpy_style_reduction
    def any(self, dim=None, keepdim: bool = False):
        dims = self._norm_dims(dim)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ANY, len(dims), arr, 1 if keepdim else 0, -1))

    @_numpy_style_reduction
    def var(self, dim=None, unbiased: bool = True, keepdim: bool = False, *, correction=None, ddof=None, dtype=None):
        """Variance: sum((x - mean)**2) / max(0, N - correction) over `dim` (all elements when
        None). `correction` is any number - 1, the default, is the sample variance, 0 the
        population variance - and `unbiased=False` is the old spelling of `correction=0`."""
        return self._variance(dim, unbiased, keepdim, correction, ddof, dtype, False)

    @_numpy_style_reduction
    def std(self, dim=None, unbiased: bool = True, keepdim: bool = False, *, correction=None, ddof=None, dtype=None):
        """Standard deviation: the square root of var() with the same arguments."""
        return self._variance(dim, unbiased, keepdim, correction, ddof, dtype, True)

    def _variance(self, dim, unbiased, keepdim, correction, ddof, dtype, take_sqrt: bool):
        name = "std" if take_sqrt else "var"
        if type(dim) is bool:
            # var(unbiased) / std(unbiased): a bool in first place is not a dimension
            dim, unbiased = None, dim
        if ddof is not None:                       # NumPy's name; np.var(t) and np.std(t) pass it
            if correction is not None:
                raise TypeError(f"{name}() got both 'correction' and 'ddof'")
            correction = ddof
        if correction is None:
            correction = 1 if unbiased else 0
        else:
            correction = _unwrap_numpy_scalar(correction)
            if isinstance(correction, Tensor) and correction.numel() == 1:
                correction = correction.item()
            if not _is_real_number(correction):
                raise TypeError(f"{name}(): argument 'correction' must be Number, not {type(correction).__name__}")
        x = self if dtype is None else self.to(dtype)
        dims = self._norm_dims(dim)
        count = x.numel()
        if dims:
            count = 1
            for d in set(d if d >= 0 else d + x.ndim for d in dims):
                if not 0 <= d < x.ndim:
                    raise IndexError(f"Dimension out of range (expected to be in range of [{-x.ndim}, {x.ndim - 1}], but got {d})")
                count *= x.shape[d]
        if correction == 1 and count > 0:
            return ReduceFn.apply(x, _C.RED_STD_SAMPLE if take_sqrt else _C.RED_VAR_SAMPLE, dims, keepdim)
        # the population statistic, rescaled from 1/N to 1/max(0, N - correction)
        out = ReduceFn.apply(x, _C.RED_STD_POP if take_sqrt else _C.RED_VAR_POP, dims, keepdim)
        if correction == 0 and count > 0:
            return out
        if count == 0:
            return out * math.nan                  # no elements: 0 / 0
        degrees = max(0.0, count - correction)
        if degrees == 0.0:
            factor = math.inf                      # sum / 0: inf, or nan where the sum is 0 too
        elif take_sqrt:
            factor = math.sqrt(count / degrees)
        else:
            factor = count / degrees
        return out * factor

    def norm(self, p: float | str = 2.0, dim=None, keepdim: bool = False, dtype=None):
        (x,) = _autocast_float32(self) if dtype is None else (self.to(dtype),)
        dims = self._norm_dims(dim)
        if p == "fro":
            p = 2.0
        pv = float(p)
        if pv == 2.0:
            return ReduceFn.apply(x, _C.RED_NORM_L2, dims, keepdim)
        if pv == 1.0:
            return ReduceFn.apply(x, _C.RED_NORM_L1, dims, keepdim)
        if math.isinf(pv):
            return x.abs().amax(dim=dims if dims else None, keepdim=keepdim) if hasattr(x, "amax") else ReduceFn.apply(x.abs(), _C.RED_MAX, dims, keepdim)
        # General Lp differentiable formula
        return (x.abs() ** pv).sum(dim=dims if dims else None, keepdim=keepdim) ** (1.0 / pv)

    @_numpy_style_reduction
    def amax(self, dim=None, keepdim: bool = False):
        return ReduceFn.apply(self, _C.RED_MAX, self._norm_dims(dim), keepdim)

    @_numpy_style_reduction
    def amin(self, dim=None, keepdim: bool = False):
        return ReduceFn.apply(self, _C.RED_MIN, self._norm_dims(dim), keepdim)

    def cumsum(self, dim: int) -> Tensor:
        (x,) = _autocast_float32(self)
        return CumsumFn.apply(x, dim)

    def cumprod(self, dim: int) -> Tensor:
        (x,) = _autocast_float32(self)
        d = dim if dim >= 0 else dim + x.ndim
        return _wrap(_C.lib.gt_cumprod(x._ptr, d))

    def sort(self, dim: int = -1, descending: bool = False):
        return SortFn.apply(self, dim, descending)

    def argsort(self, dim: int = -1, descending: bool = False) -> Tensor:
        _, idx = self.sort(dim=dim, descending=descending)
        return idx

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        return TopKFn.apply(self, k, dim, largest, sorted)

    def kthvalue(self, k: int, dim: int = -1, keepdim: bool = False):
        vals, idxs = self.sort(dim=dim, descending=False)
        d = dim if dim >= 0 else dim + self.ndim
        v = vals.select(d, k - 1)
        i = idxs.select(d, k - 1)
        if keepdim:
            v = v.unsqueeze(d)
            i = i.unsqueeze(d)
        return v, i

    def softmax(self, dim: int, dtype=None) -> Tensor:
        (x,) = _autocast_float32(self) if dtype is None else (self.to(dtype),)
        return SoftmaxFn.apply(x, dim, 0)

    def log_softmax(self, dim: int, dtype=None) -> Tensor:
        (x,) = _autocast_float32(self) if dtype is None else (self.to(dtype),)
        return SoftmaxFn.apply(x, dim, 1)

    def index_select(self, dim: int, index: Tensor) -> Tensor:
        return IndexSelectFn.apply(self, dim, index)

    def gather(self, dim: int, index: Tensor) -> Tensor:
        return GatherFn.apply(self, dim, index)

    def scatter(self, dim: int, index: Tensor, src) -> Tensor:
        return ScatterFn.apply(self, dim, index, _ensure_tensor(src, self), 0)

    def scatter_add(self, dim: int, index: Tensor, src: Tensor) -> Tensor:
        return ScatterFn.apply(self, dim, index, src, 1)

    def flip(self, dims) -> Tensor:
        if _is_int(dims):
            dims = (dims,)
        return FlipFn.apply(self, tuple(dims))

    def roll(self, shifts, dims=None) -> Tensor:
        if dims is None:
            return self.flatten().roll(shifts, dims=0).reshape(self.shape)
        sh_tup = _dims_tuple(shifts)
        dm_tup = _dims_tuple(dims)
        return RollFn.apply(self, sh_tup, dm_tup)

    def rot90(self, k: int = 1, dims: tuple[int, int] = (0, 1)) -> Tensor:
        k = k % 4
        if k == 0:
            return self
        if k == 1:
            return self.transpose(dims[0], dims[1]).flip((dims[0],))
        if k == 2:
            return self.flip(dims)
        return self.transpose(dims[0], dims[1]).flip((dims[1],))

    def tril(self, diagonal: int = 0) -> Tensor:
        return TrilTriuFn.apply(self, diagonal, 0)

    def triu(self, diagonal: int = 0) -> Tensor:
        return TrilTriuFn.apply(self, diagonal, 1)

    def diag(self, diagonal: int = 0) -> Tensor:
        k = int(diagonal)
        if self.ndim == 1:
            n = self.shape[0] + abs(k)
            return DiagonalScatterFn.apply(self, (n, n), k)
        if self.ndim == 2:
            return DiagonalFn.apply(self, k).clone()
        raise RuntimeError(f"diag expects a 1-D or 2-D tensor, got {self.ndim}-D")

    def diagonal(self, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
        if self.ndim < 2:
            raise RuntimeError("diagonal expects a tensor with at least 2 dimensions")
        x = self.movedim((dim1, dim2), (-2, -1))
        return DiagonalFn.apply(x, int(offset))

    def trace(self) -> Tensor:
        diagonal = self.diagonal()
        return ReduceFn.apply(diagonal, _C.RED_SUM, diagonal._norm_dims(None), False)   # not sum(): trace is not autocast

    def nonzero(self, *, as_tuple: bool = False):
        nz = _wrap(_C.lib.gt_nonzero(self._ptr), "nonzero")
        if as_tuple:
            return tuple(nz[:, d] for d in range(nz.shape[1]))
        return nz

    def bincount(self, weights: Tensor | None = None, minlength: int = 0) -> Tensor:
        return _wrap(_C.lib.gt_bincount(self._ptr, weights._ptr if weights is not None else 0, int(minlength)), "bincount")

    def unique(self, sorted: bool = True, return_inverse: bool = False, return_counts: bool = False, dim: int | None = None):
        (dim,)
        u_p = ctypes.c_void_p()
        inv_p = ctypes.c_void_p()
        cnt_p = ctypes.c_void_p()
        _C.check_rc(
            _C.lib.gt_unique(
                self._ptr, 1 if sorted else 0, 1 if return_inverse else 0, 1 if return_counts else 0,
                ctypes.byref(u_p), ctypes.byref(inv_p), ctypes.byref(cnt_p),
            ),
            "unique",
        )
        u = _wrap(u_p.value)
        if not return_inverse and not return_counts:
            return u
        res = [u]
        if return_inverse:
            res.append(_wrap(inv_p.value))
        if return_counts:
            res.append(_wrap(cnt_p.value))
        return tuple(res)

    def matmul(self, other: Tensor) -> Tensor:
        a, b = _autocast_lower(self, other)
        return MatMulFn.apply(a, b)

    # mm, bmm, mv and dot are matmul for operands of one particular rank each. matmul itself
    # broadcasts and accepts any ranks, so the rank is checked here: torch.dot of two matrices
    # is an error, not a matrix product.

    def mm(self, mat2: Tensor) -> Tensor:
        if not isinstance(mat2, Tensor):
            raise TypeError(f"mm(): argument 'mat2' must be Tensor, not {type(mat2).__name__}")
        if self.ndim != 2 or mat2.ndim != 2:
            raise RuntimeError(f"mm: expected 2D tensors, got {self.ndim}D and {mat2.ndim}D")
        if self.shape[1] != mat2.shape[0]:
            raise RuntimeError(
                f"mat1 and mat2 shapes cannot be multiplied ({self.shape[0]}x{self.shape[1]} and {mat2.shape[0]}x{mat2.shape[1]})"
            )
        return self.matmul(mat2)

    def bmm(self, mat2: Tensor) -> Tensor:
        if not isinstance(mat2, Tensor):
            raise TypeError(f"bmm(): argument 'mat2' must be Tensor, not {type(mat2).__name__}")
        if self.ndim != 3 or mat2.ndim != 3:
            raise RuntimeError(f"bmm: expected 3D tensors, got {self.ndim}D and {mat2.ndim}D")
        if self.shape[0] != mat2.shape[0] or self.shape[2] != mat2.shape[1]:
            raise RuntimeError(
                f"Expected size for first two dimensions of batch2 tensor to be: [{self.shape[0]}, {self.shape[2]}] "
                f"but got: [{mat2.shape[0]}, {mat2.shape[1]}]."
            )
        return self.matmul(mat2)

    def mv(self, vec: Tensor) -> Tensor:
        if not isinstance(vec, Tensor):
            raise TypeError(f"mv(): argument 'vec' must be Tensor, not {type(vec).__name__}")
        if self.ndim != 2 or vec.ndim != 1:
            raise RuntimeError(f"mv: expected 2D matrix and 1D vector, got {self.ndim}D and {vec.ndim}D")
        if self.shape[1] != vec.shape[0]:
            raise RuntimeError(f"size mismatch, got input ({self.shape[0]}), mat ({self.shape[0]}x{self.shape[1]}), vec ({vec.shape[0]})")
        return self.matmul(vec)

    def dot(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"dot(): argument 'tensor' must be Tensor, not {type(other).__name__}")
        if self.ndim != 1 or other.ndim != 1:
            raise RuntimeError(f"1D tensors expected, but got {self.ndim}D and {other.ndim}D tensors")
        if self.shape[0] != other.shape[0]:
            raise RuntimeError(
                f"inconsistent tensor size, expected tensor [{self.shape[0]}] and src [{other.shape[0]}] to have the same "
                f"number of elements, but got {self.shape[0]} and {other.shape[0]} elements respectively"
            )
        # the inner product through MatMulFn directly: it keeps the dtype of the vectors (a
        # sum would widen integers to int64), and under autocast dot is not a matmul - it runs
        # in the widest type of its operands, which is what MatMulFn's own promotion gives
        return MatMulFn.apply(self, other)

    def addmm(self, mat1: Tensor, mat2: Tensor, *, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
        """beta * self + alpha * (mat1 @ mat2), for matrices; self broadcasts to the product."""
        if mat1.ndim != 2 or mat2.ndim != 2:
            raise RuntimeError(f"addmm: expected 2D matrices mat1 and mat2, got {mat1.ndim}D and {mat2.ndim}D")
        if mat1.shape[1] != mat2.shape[0]:
            raise RuntimeError(
                f"mat1 and mat2 shapes cannot be multiplied ({mat1.shape[0]}x{mat1.shape[1]} and {mat2.shape[0]}x{mat2.shape[1]})"
            )
        _check_expands_to(self, (mat1.shape[0], mat2.shape[1]), "addmm")
        x, mat1, mat2 = _autocast_lower(self, mat1, mat2)
        return AddmmFn.apply(x, mat1, mat2, float(beta), float(alpha))

    def baddbmm(self, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
        """beta * self + alpha * (batch1 @ batch2), for batches of matrices."""
        if batch1.ndim != 3 or batch2.ndim != 3:
            raise RuntimeError(f"baddbmm: expected 3D tensors batch1 and batch2, got {batch1.ndim}D and {batch2.ndim}D")
        if batch1.shape[0] != batch2.shape[0] or batch1.shape[2] != batch2.shape[1]:
            raise RuntimeError(
                f"Expected size for first two dimensions of batch2 tensor to be: [{batch1.shape[0]}, {batch1.shape[2]}] "
                f"but got: [{batch2.shape[0]}, {batch2.shape[1]}]."
            )
        _check_expands_to(self, (batch1.shape[0], batch1.shape[1], batch2.shape[2]), "baddbmm")
        x, batch1, batch2 = _autocast_lower(self, batch1, batch2)
        out = MatMulFn.apply(batch1, batch2)
        if alpha != 1:
            out = out * alpha
        if beta == 0:
            return out          # self is ignored: its nan and inf do not reach the result
        return out + (x if beta == 1 else x * beta)

    def outer(self, vec2: Tensor) -> Tensor:
        return OuterFn.apply(self, vec2)

    def ger(self, vec2: Tensor) -> Tensor:
        return self.outer(vec2)

    def inverse(self) -> Tensor:
        return LinalgInvFn.apply(self)

    def det(self) -> Tensor:
        return LinalgDetFn.apply(self)

    def cholesky(self, upper: bool = False) -> Tensor:
        return _wrap(_C.lib.gt_linalg_cholesky(self._ptr, 1 if upper else 0))

    def qr(self, some: bool = True):
        q_p, r_p = ctypes.c_void_p(), ctypes.c_void_p()
        _C.check_rc(_C.lib.gt_linalg_qr(self._ptr, ctypes.byref(q_p), ctypes.byref(r_p), 0 if some else 1), "qr")
        return _wrap(q_p.value), _wrap(r_p.value)

    def svd(self, some: bool = True):
        u_p, s_p, v_p = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        _C.check_rc(_C.lib.gt_linalg_svd(self._ptr, 0 if some else 1, ctypes.byref(u_p), ctypes.byref(s_p), ctypes.byref(v_p)), "svd")
        return _wrap(u_p.value), _wrap(s_p.value), _wrap(v_p.value).transpose(-2, -1)

    # ---------- new_* Factory Methods ----------

    def new_empty(self, *size, dtype=None, device=None, requires_grad: bool = False):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        from ._creation import empty
        return empty(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_zeros(self, *size, dtype=None, device=None, requires_grad: bool = False):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        from ._creation import zeros
        return zeros(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_ones(self, *size, dtype=None, device=None, requires_grad: bool = False):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        from ._creation import ones
        return ones(size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad: bool = False):
        from ._creation import full
        return full(size, fill_value, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

    def new_tensor(self, data, *, dtype=None, device=None, requires_grad: bool = False):
        return _from_python(data, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)


# ---------- autocast: the elementwise functions that run in float32 ----------

for _name in ("acos", "asin", "cosh", "erfinv", "exp", "expm1", "log", "log10", "log2", "log1p", "reciprocal", "rsqrt", "sinh", "tan"):
    setattr(Tensor, _name, _autocast_float32_method(getattr(Tensor, _name)))
del _name


# ---------- NumPy ufuncs computed by tensor operations (Tensor.__array_ufunc__) ----------


def _ufunc_operand(x, ref: Tensor):
    """An operand of a ufunc call in the form tensor operators take it: tensors and scalars as
    they are, arrays and nested lists as tensors on the device of `ref`."""
    if isinstance(x, Tensor):
        return x
    np = _numpy()
    if np is not None and isinstance(x, np.ndarray):
        return _tensor_from_ndarray(x, ref.device)
    x = _unwrap_numpy_scalar(x)
    if isinstance(x, (bool, int, float, complex)):
        return x
    if isinstance(x, (list, tuple)):
        return _from_python(x, device=ref.device)
    return NotImplemented


def _ufunc_table() -> dict:
    def method(name: str):
        fn = getattr(Tensor, name)

        def call(*operands):
            # the named operations want tensors on both sides; a scalar joins the other operand
            ref = next(x for x in operands if isinstance(x, Tensor))
            return fn(*(_ensure_tensor(x, ref) for x in operands))

        return call

    table = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "true_divide": operator.truediv,
        "floor_divide": operator.floordiv,
        "remainder": operator.mod,
        "power": operator.pow,
        "matmul": operator.matmul,
        "negative": operator.neg,
        "positive": operator.pos,
        "absolute": operator.abs,
        "invert": operator.invert,
        "equal": operator.eq,
        "not_equal": operator.ne,
        "less": operator.lt,
        "less_equal": operator.le,
        "greater": operator.gt,
        "greater_equal": operator.ge,
        "bitwise_and": operator.and_,
        "bitwise_or": operator.or_,
        "bitwise_xor": operator.xor,
    }
    # NumPy name -> Tensor method
    named = {
        "sin": "sin", "cos": "cos", "tan": "tan",
        "arcsin": "asin", "arccos": "acos", "arctan": "atan", "arctan2": "atan2",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
        "arcsinh": "asinh", "arccosh": "acosh", "arctanh": "atanh",
        "exp": "exp", "expm1": "expm1", "exp2": "exp2",
        "log": "log", "log2": "log2", "log10": "log10", "log1p": "log1p",
        "sqrt": "sqrt", "square": "square", "sign": "sign", "fabs": "abs",
        "floor": "floor", "ceil": "ceil", "trunc": "trunc", "rint": "round",
        "isnan": "isnan", "isinf": "isinf", "isfinite": "isfinite", "logical_not": "logical_not",
        "maximum": "maximum", "minimum": "minimum",
    }  # fmt: skip
    for np_name, tensor_name in named.items():
        table[np_name] = method(tensor_name)
    return table


_UFUNC_TENSOR_OPS = _ufunc_table()
