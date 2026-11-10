"""Core Tensor class with GGML native storage, slicing/indexing, and autograd."""

from __future__ import annotations

import array
import ctypes
import math
import struct
import sys

from . import _C, _autograd_state, _device as _dev, _dtype as _dt
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
    DivFn,
    ExpandFn,
    FlipFn,
    GatherFn,
    ImagFn,
    IndexSelectFn,
    LerpFn,
    LinalgDetFn,
    LinalgInvFn,
    LinalgSolveFn,
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
    _wrap,
)


def _infer_nested_shape_and_flat(obj):
    if isinstance(obj, Tensor):
        return obj.shape, obj.tolist(), obj.dtype
    if not isinstance(obj, (list, tuple)):
        if isinstance(obj, bool):
            return (), [obj], _dt.bool
        if isinstance(obj, int):
            return (), [obj], _dt.int64
        if isinstance(obj, complex):
            return (), [obj], _dt.complex64
        if isinstance(obj, float):
            return (), [obj], _dt.get_default_dtype()
        raise TypeError(f"Unsupported scalar/sequence element type: {type(obj)}")

    if len(obj) == 0:
        return (0,), [], _dt.get_default_dtype()

    # Check first element recursively and gather all leaves
    flat = []
    sub_shape = None
    inferred = None

    def _walk(x, depth: int):
        nonlocal sub_shape, inferred
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return (0,), []
            sub_res = None
            out_flat = []
            for el in x:
                sh, fl = _walk(el, depth + 1)
                if sub_res is None:
                    sub_res = sh
                elif sub_res != sh:
                    raise ValueError("Expected all sub-lists to have identical shape")
                out_flat.extend(fl)
            return (len(x),) + sub_res, out_flat
        else:
            _, fl, dt = _infer_nested_shape_and_flat(x)
            if inferred is None:
                inferred = dt
            else:
                inferred = _dt.promote_types(inferred, dt)
            return (), fl

    shape, flat = _walk(obj, 0)
    if inferred is None:
        inferred = _dt.get_default_dtype()
    elif inferred.is_floating_point:
        inferred = _dt.get_default_dtype()
    return shape, flat, inferred


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

    # Check numpy ndarray without importing numpy
    np_mod = sys.modules.get("numpy")
    if np_mod is not None and isinstance(data, np_mod.ndarray):
        arr = np_mod.ascontiguousarray(data)
        np_dt = str(arr.dtype)
        np_map = {
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
        src_dt = np_map.get(np_dt)
        if src_dt is not None:
            shape = tuple(int(s) for s in arr.shape)
            raw = arr.tobytes()
            ptr = _C.check_ptr(
                _C.lib.gt_tensor_from_data(len(shape), _c_i64_arr(shape), src_dt.code, dev.dev_type, dev.index, raw, len(raw)),
                "from_numpy",
            )
            t = Tensor(_ptr=ptr, requires_grad=False)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
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
    return _from_python(x, dtype=ref.dtype if isinstance(x, float) else None, device=ref.device)


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

    def __del__(self):
        ptr = getattr(self, "_ptr", 0)
        if ptr and _C is not None and getattr(_C, "lib", None) is not None:
            _C.lib.gt_tensor_free(ptr)
            self._ptr = 0

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
        import numpy as np

        if not force and self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        if not force and self.device.type != "cpu":
            raise TypeError("can't convert non-CPU tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")
        src = self
        if src.device.type != "cpu":
            src = src.cpu()
        if src.dtype == _dt.bfloat16 or src.dtype.is_quantized:
            src = src.to(_dt.float32)
        if not src.is_contiguous() or src.storage_offset() != 0:
            src = src.contiguous()
        np_map = {
            _dt.float32: np.float32,
            _dt.float64: np.float64,
            _dt.float16: np.float16,
            _dt.int64: np.int64,
            _dt.int32: np.int32,
            _dt.int16: np.int16,
            _dt.int8: np.int8,
            _dt.uint8: np.uint8,
            _dt.bool: np.bool_,
            _dt.complex64: np.complex64,
            _dt.complex128: np.complex128,
        }
        arr = np.empty(tuple(src.shape), dtype=np_map[src.dtype])
        if src.numel() > 0:
            _C.check_rc(_C.lib.gt_tensor_copy_to_host(src._ptr, arr.ctypes.data, arr.nbytes), "numpy")
        return arr

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
        n_consuming = sum(1 for item in idx if item is not None and item is not Ellipsis)
        if n_consuming > self.ndim:
            raise IndexError(f"too many indices for tensor of dimension {self.ndim}")

        expanded = []
        for item in idx:
            if item is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - n_consuming))
            else:
                expanded.append(item)
        while sum(1 for item in expanded if item is not None) < self.ndim:
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

    def __getitem__(self, idx):
        # Single boolean mask tensor
        if isinstance(idx, Tensor) and idx.dtype == _dt.bool:
            return _wrap(_C.lib.gt_masked_select(self._ptr, idx._ptr), "masked_select")

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

        # Advanced indexing with Tensor / list
        n_ell = sum(1 for it in idx_tup if it is Ellipsis)
        n_cons = sum(1 for it in idx_tup if it is not None and it is not Ellipsis)
        expanded = []
        for it in idx_tup:
            if it is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - n_cons))
            else:
                expanded.append(it)
        while sum(1 for it in expanded if it is not None) < self.ndim:
            expanded.append(slice(None))

        # First handle pure slices/ints via view, then gt_index_tensor on remaining tensor indices
        t_ptrs = []
        t_dims = []
        pre_slices = []
        dim_i = 0
        for it in expanded:
            if isinstance(it, slice):
                pre_slices.append(it)
                dim_i += 1
            elif isinstance(it, int):
                pre_slices.append(it)
            elif isinstance(it, (Tensor, list)):
                pre_slices.append(slice(None))
                t_idx = it if isinstance(it, Tensor) else _from_python(it, dtype=_dt.int64, device=self.device)
                t_ptrs.append(t_idx)
                t_dims.append(dim_i)
                dim_i += 1
            elif it is None:
                pre_slices.append(None)
                dim_i += 1

        base = self[tuple(pre_slices)] if any(not (isinstance(x, slice) and x == slice(None)) for x in pre_slices) else self
        if not t_ptrs:
            return base
        ptrs_arr = (ctypes.c_void_p * len(t_ptrs))(*[t._ptr for t in t_ptrs])
        dims_arr = _c_i32_arr(t_dims)
        return _wrap(_C.lib.gt_index_tensor(base._ptr, len(t_ptrs), ptrs_arr, dims_arr), "index")

    def __setitem__(self, idx, value):
        val_t = _ensure_tensor(value, self)
        if val_t.dtype != self.dtype:
            val_t = val_t.to(self.dtype)

        if isinstance(idx, Tensor) and idx.dtype == _dt.bool:
            if val_t.numel() == 1:
                _C.check_rc(_C.lib.gt_masked_fill_inplace(self._ptr, idx._ptr, float(val_t.item())), "setitem mask")
            else:
                # Scatter/assign masked elements
                indices = idx.nonzero()
                ptrs = [indices[:, d] for d in range(indices.shape[1])]
                dims = list(range(indices.shape[1]))
                p_arr = (ctypes.c_void_p * len(ptrs))(*[t._ptr for t in ptrs])
                _C.check_rc(_C.lib.gt_index_put(self._ptr, len(ptrs), p_arr, _c_i32_arr(dims), val_t._ptr, 0), "setitem mask tensor")
            self._version += 1
            return

        parsed = self._parse_view_index(idx)
        if parsed is not None:
            new_shape, new_strides, offset = parsed
            v = Tensor(_ptr=_C.check_ptr(_C.lib.gt_tensor_view(self._ptr, len(new_shape), _c_i64_arr(new_shape), _c_i64_arr(new_strides), offset)))
            if val_t.shape != v.shape:
                val_t = val_t.expand(v.shape)
            _C.check_rc(_C.lib.gt_tensor_copy_inplace(v._ptr, val_t._ptr), "setitem copy")
            self._version += 1
            return

        idx_tup = idx if isinstance(idx, tuple) else (idx,)
        t_ptrs = []
        t_dims = []
        for d, it in enumerate(idx_tup):
            if isinstance(it, (Tensor, list)):
                t_idx = it if isinstance(it, Tensor) else _from_python(it, dtype=_dt.int64, device=self.device)
                t_ptrs.append(t_idx)
                t_dims.append(d)
        if t_ptrs:
            p_arr = (ctypes.c_void_p * len(t_ptrs))(*[t._ptr for t in t_ptrs])
            _C.check_rc(_C.lib.gt_index_put(self._ptr, len(t_ptrs), p_arr, _c_i32_arr(t_dims), val_t._ptr, 0), "index_put")
            self._version += 1
            return
        raise IndexError(f"Unsupported __setitem__ index: {idx}")

    # ---------- Shape / View / Splitting ----------

    def view(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        sh = self._resolve_neg1_shape(shape)
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
            if known == 0 or n % known != 0:
                raise RuntimeError(f"shape '{shape}' is invalid for input of size {n}")
            out[neg_idx] = n // known
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
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
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
        if isinstance(source, int):
            source = (source,)
        if isinstance(destination, int):
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
        if isinstance(split_size_or_sections, int):
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
        if isinstance(indices_or_sections, int):
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
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if alpha == 1.0:
                return BinaryScalarFn.apply(self, float(other), _C.BIN_ADD, 0)
            return BinaryScalarFn.apply(self, float(other) * alpha, _C.BIN_ADD, 0)
        return AddFn.apply(self, _ensure_tensor(other, self), float(alpha))

    def sub(self, other, *, alpha: float = 1.0) -> Tensor:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return BinaryScalarFn.apply(self, float(other) * alpha, _C.BIN_SUB, 0)
        return SubFn.apply(self, _ensure_tensor(other, self), float(alpha))

    def mul(self, other) -> Tensor:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_MUL, 0)
        return MulFn.apply(self, _ensure_tensor(other, self))

    def div(self, other, *, rounding_mode: str | None = None) -> Tensor:
        if rounding_mode == "floor":
            return self // other
        if rounding_mode == "trunc":
            q = self / other
            return q.trunc()
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_DIV, 0)
        return DivFn.apply(self, _ensure_tensor(other, self))

    def true_divide(self, other) -> Tensor:
        return self.div(other)

    def floor_divide(self, other) -> Tensor:
        return self // other

    def pow(self, exponent) -> Tensor:
        if isinstance(exponent, (int, float)) and not isinstance(exponent, bool):
            return BinaryScalarFn.apply(self, float(exponent), _C.BIN_POW, 0)
        return PowFn.apply(self, _ensure_tensor(exponent, self))

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_SUB, 1)
        return SubFn.apply(_ensure_tensor(other, self), self, 1.0)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_DIV, 1)
        return DivFn.apply(_ensure_tensor(other, self), self)

    def __floordiv__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 0, _C.BIN_FLOOR_DIV))
        return _wrap(_C.lib.gt_binary_op(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_FLOOR_DIV, 1.0))

    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 1, _C.BIN_FLOOR_DIV))
        return _wrap(_C.lib.gt_binary_op(_ensure_tensor(other, self)._ptr, self._ptr, _C.BIN_FLOOR_DIV, 1.0))

    def __mod__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 0, _C.BIN_REMAINDER))
        return _wrap(_C.lib.gt_binary_op(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_REMAINDER, 1.0))

    def __rmod__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _wrap(_C.lib.gt_binary_scalar_op(self._ptr, float(other), 1, _C.BIN_REMAINDER))
        return _wrap(_C.lib.gt_binary_op(_ensure_tensor(other, self)._ptr, self._ptr, _C.BIN_REMAINDER, 1.0))

    def __pow__(self, other):
        return self.pow(other)

    def __rpow__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return BinaryScalarFn.apply(self, float(other), _C.BIN_POW, 1)
        return PowFn.apply(_ensure_tensor(other, self), self)

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
        if isinstance(other, (int, float, bool)):
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
        if isinstance(other, (int, float)):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_ADD, float(alpha)), "add_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_ADD, float(alpha)), "add_")
        return self

    def sub_(self, other, *, alpha: float = 1.0) -> Tensor:
        self._check_inplace()
        if isinstance(other, (int, float)):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_SUB, float(alpha)), "sub_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_SUB, float(alpha)), "sub_")
        return self

    def mul_(self, other) -> Tensor:
        self._check_inplace()
        if isinstance(other, (int, float)):
            _C.check_rc(_C.lib.gt_binary_scalar_inplace(self._ptr, float(other), _C.BIN_MUL, 1.0), "mul_")
        else:
            _C.check_rc(_C.lib.gt_binary_inplace(self._ptr, _ensure_tensor(other, self)._ptr, _C.BIN_MUL, 1.0), "mul_")
        return self

    def div_(self, other) -> Tensor:
        self._check_inplace()
        if isinstance(other, (int, float)):
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
        return _wrap(_C.lib.gt_masked_select(self._ptr, mask._ptr), "masked_select")

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
        if isinstance(dim, int):
            return (dim,)
        return tuple(dim)

    def sum(self, dim=None, keepdim: bool = False, *, dtype=None):
        x = self if dtype is None else self.to(dtype)
        return ReduceFn.apply(x, _C.RED_SUM, self._norm_dims(dim), keepdim)

    def mean(self, dim=None, keepdim: bool = False, *, dtype=None):
        x = self if dtype is None else self.to(dtype)
        return ReduceFn.apply(x, _C.RED_MEAN, self._norm_dims(dim), keepdim)

    def prod(self, dim=None, keepdim: bool = False, *, dtype=None):
        x = self if dtype is None else self.to(dtype)
        return ReduceFn.apply(x, _C.RED_PROD, self._norm_dims(dim), keepdim)

    def logsumexp(self, dim, keepdim: bool = False):
        return ReduceFn.apply(self, _C.RED_LOGSUMEXP, self._norm_dims(dim), keepdim)

    def max(self, dim: int | None = None, keepdim: bool = False):
        if dim is None:
            return ReduceFn.apply(self, _C.RED_MAX, (), False)
        d = dim if dim >= 0 else dim + self.ndim
        vals = ReduceFn.apply(self, _C.RED_MAX, (d,), keepdim)
        idxs = _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMAX, 1, _c_i32_arr((d,)), 1 if keepdim else 0, -1))
        return vals, idxs

    def min(self, dim: int | None = None, keepdim: bool = False):
        if dim is None:
            return ReduceFn.apply(self, _C.RED_MIN, (), False)
        d = dim if dim >= 0 else dim + self.ndim
        vals = ReduceFn.apply(self, _C.RED_MIN, (d,), keepdim)
        idxs = _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMIN, 1, _c_i32_arr((d,)), 1 if keepdim else 0, -1))
        return vals, idxs

    def argmax(self, dim: int | None = None, keepdim: bool = False):
        dims = () if dim is None else (dim if dim >= 0 else dim + self.ndim,)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMAX, len(dims), arr, 1 if keepdim else 0, -1))

    def argmin(self, dim: int | None = None, keepdim: bool = False):
        dims = () if dim is None else (dim if dim >= 0 else dim + self.ndim,)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ARGMIN, len(dims), arr, 1 if keepdim else 0, -1))

    def all(self, dim=None, keepdim: bool = False):
        dims = self._norm_dims(dim)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ALL, len(dims), arr, 1 if keepdim else 0, -1))

    def any(self, dim=None, keepdim: bool = False):
        dims = self._norm_dims(dim)
        arr = _c_i32_arr(dims) if dims else None
        return _wrap(_C.lib.gt_reduce(self._ptr, _C.RED_ANY, len(dims), arr, 1 if keepdim else 0, -1))

    def var(self, dim=None, unbiased: bool = True, keepdim: bool = False, *, correction: int | None = None):
        unb = (correction != 0) if correction is not None else unbiased
        op = _C.RED_VAR_SAMPLE if unb else _C.RED_VAR_POP
        return ReduceFn.apply(self, op, self._norm_dims(dim), keepdim)

    def std(self, dim=None, unbiased: bool = True, keepdim: bool = False, *, correction: int | None = None):
        unb = (correction != 0) if correction is not None else unbiased
        op = _C.RED_STD_SAMPLE if unb else _C.RED_STD_POP
        return ReduceFn.apply(self, op, self._norm_dims(dim), keepdim)

    def norm(self, p: float | str = 2.0, dim=None, keepdim: bool = False, dtype=None):
        x = self if dtype is None else self.to(dtype)
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

    def amax(self, dim=None, keepdim: bool = False):
        return ReduceFn.apply(self, _C.RED_MAX, self._norm_dims(dim), keepdim)

    def amin(self, dim=None, keepdim: bool = False):
        return ReduceFn.apply(self, _C.RED_MIN, self._norm_dims(dim), keepdim)

    def cumsum(self, dim: int) -> Tensor:
        return CumsumFn.apply(self, dim)

    def cumprod(self, dim: int) -> Tensor:
        d = dim if dim >= 0 else dim + self.ndim
        return _wrap(_C.lib.gt_cumprod(self._ptr, d))

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
        x = self if dtype is None else self.to(dtype)
        return SoftmaxFn.apply(x, dim, 0)

    def log_softmax(self, dim: int, dtype=None) -> Tensor:
        x = self if dtype is None else self.to(dtype)
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
        if isinstance(dims, int):
            dims = (dims,)
        return FlipFn.apply(self, tuple(dims))

    def roll(self, shifts, dims=None) -> Tensor:
        if dims is None:
            return self.flatten().roll(shifts, dims=0).reshape(self.shape)
        sh_tup = (shifts,) if isinstance(shifts, int) else tuple(shifts)
        dm_tup = (dims,) if isinstance(dims, int) else tuple(dims)
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
        return _wrap(_C.lib.gt_diag(self._ptr, int(diagonal)))

    def diagonal(self, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
        x = self.movedim((dim1, dim2), (-2, -1))
        if x.ndim == 2:
            return x.diag(offset)
        # Collect diagonals across batch
        m, n = x.shape[-2], x.shape[-1]
        if offset >= 0:
            length = max(0, min(m, n - offset))
            r0, c0 = 0, offset
        else:
            length = max(0, min(m + offset, n))
            r0, c0 = -offset, 0
        sh = x.shape[:-2] + (length,)
        st = x.stride()[:-2] + (x.stride()[-2] + x.stride()[-1],)
        off = x.storage_offset() + r0 * x.stride()[-2] + c0 * x.stride()[-1]
        return SliceViewFn.apply(x, sh, st, off, ())

    def trace(self) -> Tensor:
        return self.diagonal().sum()

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
        return MatMulFn.apply(self, other)

    def mm(self, mat2: Tensor) -> Tensor:
        return self.matmul(mat2)

    def bmm(self, mat2: Tensor) -> Tensor:
        return self.matmul(mat2)

    def mv(self, vec: Tensor) -> Tensor:
        return self.matmul(vec)

    def dot(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def addmm(self, mat1: Tensor, mat2: Tensor, *, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
        return AddmmFn.apply(self, mat1, mat2, float(beta), float(alpha))

    def baddbmm(self, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
        return _wrap(_C.lib.gt_baddbmm(self._ptr, batch1._ptr, batch2._ptr, float(beta), float(alpha)))

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
