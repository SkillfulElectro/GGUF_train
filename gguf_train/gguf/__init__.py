"""Native GGUF file reader/writer, quantization, and metadata inspection."""

from __future__ import annotations

import ctypes
import numbers
from pathlib import Path

from .. import _C, _dtype as _dt
from .._tensor import Tensor

# enum gguf_type
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)
_INT_TYPES = frozenset((_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _UINT64, _INT64))
_FLOAT_TYPES = frozenset((_FLOAT32, _FLOAT64))
_ARRAY_CTYPES = {
    _UINT8: ctypes.c_uint8,
    _INT8: ctypes.c_int8,
    _UINT16: ctypes.c_uint16,
    _INT16: ctypes.c_int16,
    _UINT32: ctypes.c_uint32,
    _INT32: ctypes.c_int32,
    _FLOAT32: ctypes.c_float,
    _BOOL: ctypes.c_int8,
    _UINT64: ctypes.c_uint64,
    _INT64: ctypes.c_int64,
    _FLOAT64: ctypes.c_double,
}
_INT64_MIN, _INT64_MAX, _UINT64_MAX = -(2**63), 2**63 - 1, 2**64 - 1

# Keys under these prefixes record the real shape and dtype of tensors GGUF cannot describe
# (see csrc/gguf_train.h). They are applied when the tensors are read; they are not metadata.
_RESERVED_PREFIXES = ("gguf_train.shape.", "gguf_train.dtype.")


def _kv_scalar(handle, i: int, ktype: int):
    lib = _C.lib
    n = int(lib.gt_gguf_kv_val_str(handle, i, None, 0))
    if n < 0:
        raise RuntimeError(f"GGUF metadata entry {i} cannot be read")
    buf = ctypes.create_string_buffer(n + 1)
    lib.gt_gguf_kv_val_str(handle, i, buf, n + 1)
    text = buf.value.decode("utf-8", errors="replace")
    if ktype in _INT_TYPES:
        return int(text)
    if ktype in _FLOAT_TYPES:
        return float(text)
    if ktype == _BOOL:
        return text == "true"
    return text


def _kv_array(handle, i: int) -> list:
    lib = _C.lib
    etype = int(lib.gt_gguf_kv_arr_type(handle, i))
    n = int(lib.gt_gguf_kv_arr_n(handle, i))
    if n <= 0:
        return []
    if etype == _STRING:
        return [(lib.gt_gguf_kv_arr_str(handle, i, j) or b"").decode("utf-8", errors="replace") for j in range(n)]
    ctype = _ARRAY_CTYPES.get(etype)
    data = lib.gt_gguf_kv_arr_data(handle, i)
    if ctype is None or not data:
        raise RuntimeError(f"GGUF metadata entry {i} is an array of an unknown element type ({etype})")
    values = ctypes.cast(data, ctypes.POINTER(ctype * n)).contents[:]
    return [bool(v) for v in values] if etype == _BOOL else values


class GGUFFile:
    """Opened GGUF archive exposing metadata KV pairs and tensors.

    ``metadata`` maps keys to ``int``, ``float``, ``bool`` or ``str`` values, and to lists of
    those for GGUF arrays.
    """

    def __init__(self, path: str | Path, dequantize: bool = False) -> None:
        self.path = str(path)
        self._handle = _C.check_ptr(_C.lib.gt_gguf_open(self.path.encode("utf-8")), f"open GGUF {self.path}")
        try:
            self.metadata: dict[str, object] = {}
            n_kv = int(_C.lib.gt_gguf_n_kv(self._handle))
            for i in range(n_kv):
                k_raw = _C.lib.gt_gguf_kv_key(self._handle, i)
                key = k_raw.decode("utf-8", errors="replace") if k_raw else f"kv_{i}"
                if key.startswith(_RESERVED_PREFIXES):
                    continue
                ktype = int(_C.lib.gt_gguf_kv_type(self._handle, i))
                if ktype == _ARRAY:
                    self.metadata[key] = _kv_array(self._handle, i)
                else:
                    self.metadata[key] = _kv_scalar(self._handle, i, ktype)

            self.tensors: dict[str, Tensor] = {}
            n_t = int(_C.lib.gt_gguf_n_tensors(self._handle))
            for i in range(n_t):
                t_raw = _C.lib.gt_gguf_tensor_name(self._handle, i)
                name = t_raw.decode("utf-8", errors="replace")
                ptr = _C.check_ptr(_C.lib.gt_gguf_get_tensor(self._handle, t_raw, 1 if dequantize else 0), f"read tensor {name}")
                self.tensors[name] = Tensor(_ptr=ptr, requires_grad=False)
        finally:
            self.close()

    def __getitem__(self, key: str) -> Tensor:
        return self.tensors[key]

    def __contains__(self, key: object) -> bool:
        return key in self.tensors

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    def keys(self):
        return self.tensors.keys()

    def values(self):
        return self.tensors.values()

    def items(self):
        return self.tensors.items()

    def get(self, key: str, default=None):
        return self.tensors.get(key, default)

    def close(self) -> None:
        if getattr(self, "_handle", 0):
            _C.lib.gt_gguf_close(self._handle)
            self._handle = 0


def load_gguf(path: str | Path, dequantize: bool = False, return_metadata: bool = False):
    """Load tensors and metadata from a .gguf file."""
    gf = GGUFFile(path, dequantize=dequantize)
    if return_metadata:
        return gf.tensors, gf.metadata
    return gf


def _encode_text(key: str, value) -> bytes:
    raw = bytes(value) if isinstance(value, (bytes, bytearray)) else str(value).encode("utf-8")
    if b"\0" in raw:
        raise ValueError(f"save_gguf: metadata {key!r} contains a NUL byte, which a GGUF string written here cannot hold")
    return raw


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _int_type(key: str, lo: int, hi: int) -> int:
    if _INT64_MIN <= lo and hi <= _INT64_MAX:
        return _INT64
    if 0 <= lo and hi <= _UINT64_MAX:
        return _UINT64
    raise ValueError(f"save_gguf: metadata {key!r} does not fit a 64-bit integer")


def _encode_array(key: str, values: list):
    """Element gguf_type and the ctypes array holding the elements (kept alive by the caller)."""
    n = len(values)
    if n == 0:
        return _INT64, None, ()
    if all(isinstance(v, bool) for v in values):
        return _BOOL, (ctypes.c_int8 * n)(*[1 if v else 0 for v in values]), ()
    if all(_is_int(v) for v in values):
        ints = [int(v) for v in values]
        if _int_type(key, min(ints), max(ints)) == _INT64:
            return _INT64, (ctypes.c_int64 * n)(*ints), ()
        return _UINT64, (ctypes.c_uint64 * n)(*ints), ()
    if all(_is_real(v) for v in values):
        return _FLOAT64, (ctypes.c_double * n)(*[float(v) for v in values]), ()
    if all(isinstance(v, (str, bytes, bytearray)) for v in values):
        encoded = [_encode_text(key, v) for v in values]
        return _STRING, (ctypes.c_char_p * n)(*encoded), encoded
    raise TypeError(
        f"save_gguf: metadata {key!r} is a list that is neither all bool, all numbers nor all str; "
        "GGUF arrays hold a single element type"
    )


def save_gguf(
    path: str | Path,
    tensors: dict[str, Tensor],
    metadata: dict[str, object] | None = None,
    *,
    copy_metadata_from: str | Path | None = None,
) -> None:
    """Write tensors and metadata KV pairs to a .gguf file.

    Tensors keep their shape and dtype, including the ones GGUF has no way to describe
    (scalars, more than four dimensions, uint8, bool, complex): those are recorded under
    the reserved ``gguf_train.shape.*`` / ``gguf_train.dtype.*`` keys.

    Metadata values are stored as bool, int64 (uint64 above its range), float64 or string;
    lists and tuples of one such kind become GGUF arrays; anything else is stored as its
    ``str()``. ``copy_metadata_from`` names a GGUF file whose metadata is copied first with
    every value's exact type; ``metadata`` is applied on top of it.
    """
    names = list(tensors.keys())
    for name in names:
        if not isinstance(tensors[name], Tensor):
            raise TypeError(f"save_gguf: {name!r} is a {type(tensors[name]).__name__}, not a Tensor")
    c_names = (ctypes.c_char_p * max(len(names), 1))(*[str(n).encode("utf-8") for n in names])
    c_ptrs = (ctypes.c_void_p * max(len(names), 1))(*[tensors[n]._ptr for n in names])

    meta_items = list((metadata or {}).items())
    n_kv = len(meta_items)
    size = max(n_kv, 1)
    c_keys = (ctypes.c_char_p * size)()
    c_types = (ctypes.c_int * size)()
    c_vals = (ctypes.c_char_p * size)()
    c_arr_types = (ctypes.c_int * size)()
    c_arr_n = (ctypes.c_int64 * size)()
    c_arr_data = (ctypes.c_void_p * size)()
    keep_alive = []
    for i, (k, v) in enumerate(meta_items):
        key = str(k)
        c_keys[i] = _encode_text(key, key)
        if isinstance(v, (list, tuple)):
            etype, arr, extra = _encode_array(key, list(v))
            c_types[i] = _ARRAY
            c_arr_types[i] = etype
            c_arr_n[i] = len(v)
            c_arr_data[i] = ctypes.cast(arr, ctypes.c_void_p) if arr is not None else None
            keep_alive.append((arr, extra))
            continue
        if isinstance(v, bool):
            c_types[i] = _BOOL
            s = b"1" if v else b"0"
        elif _is_int(v):
            c_types[i] = _int_type(key, int(v), int(v))
            s = str(int(v)).encode("ascii")
        elif _is_real(v):
            c_types[i] = _FLOAT64
            s = repr(float(v)).encode("ascii")
        else:
            c_types[i] = _STRING
            s = _encode_text(key, v)
        keep_alive.append(s)
        c_vals[i] = s

    source = str(copy_metadata_from).encode("utf-8") if copy_metadata_from is not None else None
    _C.check_rc(
        _C.lib.gt_gguf_save_ex(
            str(path).encode("utf-8"), len(names), c_names, c_ptrs,
            n_kv, c_keys, c_types, c_vals, c_arr_types, c_arr_n, c_arr_data, source,
        ),
        f"save_gguf {path}",
    )


def inspect_gguf(path: str | Path) -> dict:
    """Return metadata and tensor shapes/dtypes of a .gguf file without full dequantization."""
    gf = load_gguf(path, dequantize=False)
    return {
        "path": str(path),
        "n_tensors": len(gf.tensors),
        "metadata": gf.metadata,
        "tensors": {k: {"shape": tuple(v.shape), "dtype": str(v.dtype)} for k, v in gf.tensors.items()},
    }


def quantize_tensor(t: Tensor, qtype="q4_0") -> Tensor:
    return t.quantize(qtype)


def dequantize_tensor(t: Tensor, dtype=_dt.float32) -> Tensor:
    return t.dequantize(dtype)


def quantize_state_dict(state_dict: dict[str, Tensor], qtype="q4_0", min_numel: int = 256) -> dict[str, Tensor]:
    qdt = _dt.as_dtype(qtype)
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, Tensor) and v.ndim >= 2 and v.numel() >= min_numel and v.shape[-1] % 32 == 0 and v.is_floating_point():
            out[k] = v.quantize(qdt)
        else:
            out[k] = v
    return out


def dequantize_state_dict(state_dict: dict[str, Tensor], dtype=_dt.float32) -> dict[str, Tensor]:
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, Tensor) and v.dtype.is_quantized:
            out[k] = v.dequantize(dtype)
        else:
            out[k] = v
    return out


def quantize_gguf(src_path: str | Path, dst_path: str | Path, qtype="q4_0", min_numel: int = 256) -> None:
    gf = load_gguf(src_path, dequantize=True)
    qt = quantize_state_dict(gf.tensors, qtype=qtype, min_numel=min_numel)
    # The metadata is copied file to file, so every value keeps its exact GGUF type.
    save_gguf(dst_path, qt, copy_metadata_from=src_path)


__all__ = [
    "GGUFFile",
    "load_gguf",
    "save_gguf",
    "inspect_gguf",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_state_dict",
    "dequantize_state_dict",
    "quantize_gguf",
]
