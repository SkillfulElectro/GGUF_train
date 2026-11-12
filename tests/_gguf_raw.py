"""A GGUF reader and writer made of nothing but ``struct``.

Tests use it to build files the package under test did not write, and to look at what
the package wrote without going through the package's own reader.
"""

from __future__ import annotations

import struct
from collections import namedtuple
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types (enum gguf_type)
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
_FORMATS = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i",
    FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

# ggml tensor types (enum ggml_type)
F32, F16, Q4_0, Q8_0, I8, I16, I32, I64, F64 = 0, 1, 2, 8, 24, 25, 26, 27, 28

#: ``type`` is a GGUF value type; ``elem_type`` is set for arrays only.
KV = namedtuple("KV", "type elem_type value")
#: ``dims`` are as stored, innermost dimension first; ``type`` is a ggml tensor type.
TensorInfo = namedtuple("TensorInfo", "dims type offset")


def _string(value) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _scalar(vtype: int, value) -> bytes:
    if vtype == STRING:
        return _string(value)
    return struct.pack(_FORMATS[vtype], value)


def kv(key: str, vtype: int, value, elem_type: int | None = None) -> bytes:
    """One encoded metadata entry; pass ``elem_type`` with ``ARRAY``."""
    out = _string(key) + struct.pack("<I", vtype)
    if vtype == ARRAY:
        out += struct.pack("<IQ", elem_type, len(value))
        out += b"".join(_scalar(elem_type, v) for v in value)
    else:
        out += _scalar(vtype, value)
    return out


def _pad(n: int) -> int:
    return (-n) % GGUF_ALIGNMENT


def write_gguf(path, kvs, tensors) -> Path:
    """Write a GGUF file from encoded ``kv()`` entries and ``(name, dims, ggml_type, data)`` tensors.

    ``dims`` are written as given, innermost dimension first: a (rows, cols) matrix is
    ``[cols, rows]``. ``data`` is the raw little-endian contents.
    """
    infos = []
    blobs = []
    offset = 0
    for name, dims, ttype, data in tensors:
        infos.append(
            _string(name)
            + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<IQ", ttype, offset)
        )
        blob = bytes(data) + b"\x00" * _pad(len(data))
        blobs.append(blob)
        offset += len(blob)

    header = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(infos), len(kvs)) + b"".join(kvs) + b"".join(infos)
    header += b"\x00" * _pad(len(header))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        for blob in blobs:
            f.write(blob)
    return path


class _Cursor:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.pos = 0

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, self.raw, self.pos)
        self.pos += size
        return values[0] if len(values) == 1 else values

    def string(self) -> str:
        n = self.unpack("<Q")
        value = self.raw[self.pos:self.pos + n]
        self.pos += n
        return value.decode("utf-8")

    def scalar(self, vtype: int):
        return self.string() if vtype == STRING else self.unpack(_FORMATS[vtype])


def read_gguf(path):
    """Return ``(metadata, tensors)``: ``{key: KV}`` and ``{name: TensorInfo}``, in file order."""
    cur = _Cursor(Path(path).read_bytes())
    if cur.raw[:4] != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    cur.pos = 4
    _version, n_tensors, n_kv = cur.unpack("<IQQ")
    metadata = {}
    for _ in range(n_kv):
        key = cur.string()
        vtype = cur.unpack("<I")
        if vtype == ARRAY:
            elem_type, n = cur.unpack("<IQ")
            metadata[key] = KV(ARRAY, elem_type, [cur.scalar(elem_type) for _ in range(n)])
        else:
            metadata[key] = KV(vtype, None, cur.scalar(vtype))
    tensors = {}
    for _ in range(n_tensors):
        name = cur.string()
        n_dims = cur.unpack("<I")
        dims = [cur.unpack("<Q") for _ in range(n_dims)]
        ttype, offset = cur.unpack("<IQ")
        tensors[name] = TensorInfo(dims, ttype, offset)
    return metadata, tensors
