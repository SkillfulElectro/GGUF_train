"""Dtype definitions, promotion rules, finfo, iinfo, and GGML/GGUF type mappings."""

from __future__ import annotations


class dtype:
    """PyTorch-compatible data type descriptor mapped to ggml/gguf_train native types."""

    def __init__(
        self,
        name: str,
        gt_id: int,
        itemsize: int,
        is_floating_point: bool = False,
        is_complex: bool = False,
        is_signed: bool = True,
        is_quantized: bool = False,
        short_name: str | None = None,
    ):
        self.name = name
        self.gt_id = gt_id
        self.itemsize = itemsize
        self.is_floating_point = is_floating_point
        self.is_complex = is_complex
        self.is_signed = is_signed
        self.is_quantized = is_quantized
        self.short_name = short_name or name

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dtype):
            return self.gt_id == other.gt_id
        return False

    def __hash__(self) -> int:
        return hash(self.gt_id)

    def __reduce__(self):
        return (_dtype_from_name, (self.name,))

    @property
    def code(self) -> int:
        return self.gt_id


def _dtype_from_name(name: str) -> dtype:
    return _NAME_TO_DTYPE[name]


# Standard PyTorch dtypes (IDs match csrc/gguf_train.h enum gt_dtype)
float32 = dtype("float32", 0, 4, is_floating_point=True)
float = float32
float16 = dtype("float16", 1, 2, is_floating_point=True)
half = float16
bfloat16 = dtype("bfloat16", 2, 2, is_floating_point=True)
float64 = dtype("float64", 3, 8, is_floating_point=True)
double = float64

int8 = dtype("int8", 4, 1, is_signed=True)
uint8 = dtype("uint8", 5, 1, is_signed=False)
int16 = dtype("int16", 6, 2, is_signed=True)
short = int16
int32 = dtype("int32", 7, 4, is_signed=True)
int = int32
int64 = dtype("int64", 8, 8, is_signed=True)
long = int64
bool = dtype("bool", 9, 1, is_signed=False)

complex64 = dtype("complex64", 10, 8, is_complex=True)
cfloat = complex64
complex128 = dtype("complex128", 11, 16, is_complex=True)
cdouble = complex128

# Quantized GGUF / ggml dtypes (GT_Q_BASE + ggml_type)
# Values match ggml_type enum in third_party/llama.cpp/ggml/include/ggml.h
q4_0 = dtype("q4_0", 100 + 2, 1, is_quantized=True)
q4_1 = dtype("q4_1", 100 + 3, 1, is_quantized=True)
q5_0 = dtype("q5_0", 100 + 6, 1, is_quantized=True)
q5_1 = dtype("q5_1", 100 + 7, 1, is_quantized=True)
q8_0 = dtype("q8_0", 100 + 8, 1, is_quantized=True)
q8_1 = dtype("q8_1", 100 + 9, 1, is_quantized=True)
q2_k = dtype("q2_k", 100 + 10, 1, is_quantized=True)
q3_k = dtype("q3_k", 100 + 11, 1, is_quantized=True)
q4_k = dtype("q4_k", 100 + 12, 1, is_quantized=True)
q5_k = dtype("q5_k", 100 + 13, 1, is_quantized=True)
q6_k = dtype("q6_k", 100 + 14, 1, is_quantized=True)
q8_k = dtype("q8_k", 100 + 15, 1, is_quantized=True)
iq2_xxs = dtype("iq2_xxs", 100 + 16, 1, is_quantized=True)
iq2_xs = dtype("iq2_xs", 100 + 17, 1, is_quantized=True)
iq3_xxs = dtype("iq3_xxs", 100 + 18, 1, is_quantized=True)
iq1_s = dtype("iq1_s", 100 + 19, 1, is_quantized=True)
iq4_nl = dtype("iq4_nl", 100 + 20, 1, is_quantized=True)
iq3_s = dtype("iq3_s", 100 + 21, 1, is_quantized=True)
iq2_s = dtype("iq2_s", 100 + 22, 1, is_quantized=True)
iq4_xs = dtype("iq4_xs", 100 + 23, 1, is_quantized=True)
iq1_m = dtype("iq1_m", 100 + 29, 1, is_quantized=True)
tq1_0 = dtype("tq1_0", 100 + 34, 1, is_quantized=True)
tq2_0 = dtype("tq2_0", 100 + 35, 1, is_quantized=True)
mxfp4 = dtype("mxfp4", 100 + 39, 1, is_quantized=True)

# PyTorch compatibility aliases
qint8 = int8
quint8 = uint8
qint32 = int32

_ALL_DTYPES = [
    float32, float16, bfloat16, float64,
    int8, uint8, int16, int32, int64, bool,
    complex64, complex128,
    q4_0, q4_1, q5_0, q5_1, q8_0, q8_1,
    q2_k, q3_k, q4_k, q5_k, q6_k, q8_k,
    iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq4_nl, iq3_s, iq2_s, iq4_xs, iq1_m,
    tq1_0, tq2_0, mxfp4,
]

_ID_TO_DTYPE: dict[int, dtype] = {d.gt_id: d for d in _ALL_DTYPES}
_NAME_TO_DTYPE: dict[str, dtype] = {d.name: d for d in _ALL_DTYPES}
_NAME_TO_DTYPE.update({
    "float": float32,
    "torch.float32": float32,
    "torch.float": float32,
    "half": float16,
    "torch.float16": float16,
    "torch.half": float16,
    "torch.bfloat16": bfloat16,
    "double": float64,
    "torch.float64": float64,
    "torch.double": float64,
    "torch.int8": int8,
    "torch.uint8": uint8,
    "short": int16,
    "torch.int16": int16,
    "int": int32,
    "torch.int32": int32,
    "long": int64,
    "torch.int64": int64,
    "torch.bool": bool,
    "cfloat": complex64,
    "torch.complex64": complex64,
    "cdouble": complex128,
    "torch.complex128": complex128,
    "f32": float32,
    "f16": float16,
    "bf16": bfloat16,
    "f64": float64,
    "i8": int8,
    "u8": uint8,
    "i16": int16,
    "i32": int32,
    "i64": int64,
})

_DEFAULT_DTYPE: dtype = float32


def get_default_dtype() -> dtype:
    return _DEFAULT_DTYPE


def set_default_dtype(d: dtype) -> None:
    global _DEFAULT_DTYPE
    d = resolve_dtype(d)
    if not d.is_floating_point:
        raise TypeError("only floating-point types are supported as the default type")
    _DEFAULT_DTYPE = d


def set_default_tensor_type(t) -> None:
    if isinstance(t, str):
        if "Double" in t:
            set_default_dtype(float64)
        elif "Half" in t:
            set_default_dtype(float16)
        elif "BFloat16" in t:
            set_default_dtype(bfloat16)
        else:
            set_default_dtype(float32)
    elif isinstance(t, dtype):
        set_default_dtype(t)


def resolve_dtype(d: dtype | str | type | None, default: dtype | None = None) -> dtype:
    if d is None:
        return default if default is not None else _DEFAULT_DTYPE
    if isinstance(d, dtype):
        return d
    if isinstance(d, __import__("builtins").int) and not isinstance(d, __import__("builtins").bool):
        if d in _ID_TO_DTYPE:
            return _ID_TO_DTYPE[d]
        return dtype(f"q_{d}", d, 1, is_quantized=True)
    if isinstance(d, str):
        key = d.strip().lower()
        if key in _NAME_TO_DTYPE:
            return _NAME_TO_DTYPE[key]
        raise ValueError(f"Unknown dtype: {d!r}")
    if d is __builtins__ or hasattr(d, "__name__"):
        name = getattr(d, "__name__", "")
        if name == "float":
            return _DEFAULT_DTYPE
        if name == "int":
            return int64
        if name == "bool":
            return bool
        if name == "complex":
            return complex64
    raise TypeError(f"Cannot interpret {d!r} as a dtype")


def dtype_from_gt_id(gt_id: int) -> dtype:
    if gt_id in _ID_TO_DTYPE:
        return _ID_TO_DTYPE[gt_id]
    return dtype(f"ggml_type_{gt_id}", gt_id, 1, is_quantized=(gt_id >= 100))


as_dtype = resolve_dtype
from_code = dtype_from_gt_id


_PROMO_ORDER = [
    bool, uint8, int8, int16, int32, int64,
    float16, bfloat16, float32, float64,
    complex64, complex128,
]


def promote_types(type1: dtype, type2: dtype) -> dtype:
    t1 = resolve_dtype(type1)
    t2 = resolve_dtype(type2)
    if t1 == t2:
        return t1
    if t1.is_quantized:
        t1 = float32
    if t2.is_quantized:
        t2 = float32
    if t1.is_complex or t2.is_complex:
        if t1 == float64 or t1 == complex128 or t2 == float64 or t2 == complex128:
            return complex128
        return complex64
    if t1.is_floating_point or t2.is_floating_point:
        if t1 == float64 or t2 == float64:
            return float64
        if (t1 == float16 and t2 == bfloat16) or (t1 == bfloat16 and t2 == float16):
            return float32
        if t1 == float32 or t2 == float32:
            return float32
        return t1 if t1.is_floating_point else t2
    # Both integer / bool
    if t1 == bool:
        return t2
    if t2 == bool:
        return t1
    if t1 == int64 or t2 == int64:
        return int64
    if t1 == int32 or t2 == int32:
        return int32
    if t1 == int16 or t2 == int16:
        return int16
    if (t1 == int8 and t2 == uint8) or (t1 == uint8 and t2 == int8):
        return int16
    return t1 if _PROMO_ORDER.index(t1) >= _PROMO_ORDER.index(t2) else t2


def result_type(tensor_or_scalar1, tensor_or_scalar2=None) -> dtype:
    def _kind(x):
        if hasattr(x, "dtype") and isinstance(x.dtype, dtype):
            return x.dtype
        if isinstance(x, __builtins__["bool"] if isinstance(__builtins__, dict) else __builtins__.bool):
            return bool
        if isinstance(x, int if False else __import__("builtins").int):
            return int64
        if isinstance(x, __import__("builtins").float):
            return _DEFAULT_DTYPE
        if isinstance(x, complex):
            return complex64
        if isinstance(x, dtype):
            return x
        return _DEFAULT_DTYPE

    d1 = _kind(tensor_or_scalar1)
    if tensor_or_scalar2 is None:
        return d1
    d2 = _kind(tensor_or_scalar2)
    return promote_types(d1, d2)


def can_cast(from_type: dtype, to_type: dtype) -> builtins.bool:
    f = resolve_dtype(from_type)
    t = resolve_dtype(to_type)
    if f == t:
        return True
    if t.is_complex:
        return True
    if f.is_complex and not t.is_complex:
        return False
    if t.is_floating_point:
        return not f.is_floating_point or t.itemsize >= f.itemsize
    if f.is_floating_point:
        return False
    if f == bool:
        return True
    if t == bool:
        return False
    return _PROMO_ORDER.index(t) >= _PROMO_ORDER.index(f)


class finfo:
    def __init__(self, d: dtype):
        d = resolve_dtype(d)
        if not d.is_floating_point and not d.is_complex:
            raise TypeError(f"data type {d} not inexact")
        self.dtype = d
        if d in (float32, complex64):
            self.bits = 32
            self.eps = 1.1920928955078125e-07
            self.min = -3.4028234663852886e+38
            self.max = 3.4028234663852886e+38
            self.tiny = 1.1754943508222875e-38
            self.smallest_normal = self.tiny
            self.resolution = 1e-6
        elif d in (float64, complex128):
            self.bits = 64
            self.eps = 2.220446049250313e-16
            self.min = -1.7976931348623157e+308
            self.max = 1.7976931348623157e+308
            self.tiny = 2.2250738585072014e-308
            self.smallest_normal = self.tiny
            self.resolution = 1e-15
        elif d == float16:
            self.bits = 16
            self.eps = 0.0009765625
            self.min = -65504.0
            self.max = 65504.0
            self.tiny = 6.103515625e-05
            self.smallest_normal = self.tiny
            self.resolution = 1e-3
        elif d == bfloat16:
            self.bits = 16
            self.eps = 0.0078125
            self.min = -3.3895313892515355e+38
            self.max = 3.3895313892515355e+38
            self.tiny = 1.1754943508222875e-38
            self.smallest_normal = self.tiny
            self.resolution = 1e-2

    def __repr__(self) -> str:
        return f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, eps={self.eps}, tiny={self.tiny}, dtype={self.dtype.name})"


class iinfo:
    def __init__(self, d: dtype):
        d = resolve_dtype(d)
        if d.is_floating_point or d.is_complex:
            raise TypeError(f"iinfo expects integer dtype, got {d}")
        self.dtype = d
        if d == int8:
            self.bits, self.min, self.max = 8, -128, 127
        elif d == uint8:
            self.bits, self.min, self.max = 8, 0, 255
        elif d == int16:
            self.bits, self.min, self.max = 16, -32768, 32767
        elif d == int32:
            self.bits, self.min, self.max = 32, -2147483648, 2147483647
        elif d == int64:
            self.bits, self.min, self.max = 64, -9223372036854775808, 9223372036854775807
        elif d == bool:
            self.bits, self.min, self.max = 8, 0, 1
        else:
            self.bits, self.min, self.max = 8, -128, 127

    def __repr__(self) -> str:
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype.name})"
