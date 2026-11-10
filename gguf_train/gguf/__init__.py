"""Native GGUF file reader/writer, quantization, and metadata inspection."""

from __future__ import annotations

import ctypes
from pathlib import Path

from .. import _C, _dtype as _dt
from .._tensor import Tensor


class GGUFFile:
    """Opened GGUF archive exposing metadata KV pairs and tensors."""

    def __init__(self, path: str | Path, dequantize: bool = False) -> None:
        self.path = str(path)
        self._handle = _C.check_ptr(_C.lib.gt_gguf_open(self.path.encode("utf-8")), f"open GGUF {self.path}")
        try:
            self.metadata: dict[str, object] = {}
            n_kv = int(_C.lib.gt_gguf_n_kv(self._handle))
            buf = ctypes.create_string_buffer(4096)
            for i in range(n_kv):
                k_raw = _C.lib.gt_gguf_kv_key(self._handle, i)
                key = k_raw.decode("utf-8", errors="replace") if k_raw else f"kv_{i}"
                ktype = int(_C.lib.gt_gguf_kv_type(self._handle, i))
                _C.lib.gt_gguf_kv_val_str(self._handle, i, buf, 4096)
                sval = buf.value.decode("utf-8", errors="replace")
                if ktype in (0, 1, 2, 3, 4, 5, 10, 11):
                    try:
                        self.metadata[key] = int(sval)
                    except ValueError:
                        self.metadata[key] = sval
                elif ktype in (6, 12):
                    try:
                        self.metadata[key] = float(sval)
                    except ValueError:
                        self.metadata[key] = sval
                elif ktype == 7:
                    self.metadata[key] = sval.lower() in ("true", "1")
                else:
                    self.metadata[key] = sval

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


def save_gguf(path: str | Path, tensors: dict[str, Tensor], metadata: dict[str, object] | None = None) -> None:
    """Write tensors and metadata KV pairs to a .gguf file."""
    names = list(tensors.keys())
    c_names = (ctypes.c_char_p * max(len(names), 1))(*[n.encode("utf-8") for n in names])
    c_ptrs = (ctypes.c_void_p * max(len(names), 1))(*[tensors[n]._ptr for n in names])

    meta_items = list((metadata or {}).items())
    n_kv = len(meta_items)
    if n_kv > 0:
        c_keys = (ctypes.c_char_p * n_kv)(*[str(k).encode("utf-8") for k, _ in meta_items])
        c_types = (ctypes.c_int * n_kv)()
        c_vals = (ctypes.c_char_p * n_kv)()
        encoded_vals = []
        for i, (_, v) in enumerate(meta_items):
            if isinstance(v, bool):
                c_types[i] = 7  # GGUF_TYPE_BOOL
                s = b"1" if v else b"0"
            elif isinstance(v, int):
                c_types[i] = 11  # GGUF_TYPE_INT64
                s = str(v).encode("utf-8")
            elif isinstance(v, float):
                c_types[i] = 12  # GGUF_TYPE_FLOAT64
                s = repr(v).encode("utf-8")
            else:
                c_types[i] = 8  # GGUF_TYPE_STRING
                s = str(v).encode("utf-8")
            encoded_vals.append(s)
            c_vals[i] = s
    else:
        c_keys = None
        c_types = None
        c_vals = None

    _C.check_rc(
        _C.lib.gt_gguf_save(str(path).encode("utf-8"), len(names), c_names, c_ptrs, n_kv, c_keys, c_types, c_vals),
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
    save_gguf(dst_path, qt, metadata=gf.metadata)


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
