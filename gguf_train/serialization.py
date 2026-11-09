"""Checkpoint save & load supporting both .pt/.pth pickle files and native .gguf files."""

from __future__ import annotations

import ctypes
import io
import pickle
from pathlib import Path

from . import _C, _device as _dev, _dtype as _dt
from ._functions import _c_i64_arr
from ._tensor import Tensor


class _SerializedTensor:
    """Picklable representation of a Tensor."""

    def __init__(self, shape: tuple[int, ...], dtype_code: int, raw: bytes, requires_grad: bool, is_param: bool = False):
        self.shape = shape
        self.dtype_code = dtype_code
        self.raw = raw
        self.requires_grad = requires_grad
        self.is_param = is_param

    def restore(self, map_device: _dev.device | None = None) -> Tensor:
        dev = map_device or _dev.device("cpu")
        ptr = _C.check_ptr(
            _C.lib.gt_tensor_from_data(
                len(self.shape),
                _c_i64_arr(self.shape),
                self.dtype_code,
                dev.dev_type,
                dev.index,
                self.raw if self.raw else None,
                len(self.raw),
            ),
            "load tensor",
        )
        t = Tensor(_ptr=ptr, requires_grad=False)
        if self.requires_grad and (t.dtype.is_floating_point or t.dtype.is_complex):
            t.requires_grad_(True)
        if self.is_param:
            from .nn.parameter import Parameter
            return Parameter(t, requires_grad=self.requires_grad)
        return t


def _to_serializable(obj):
    from .nn.parameter import Parameter

    if isinstance(obj, Tensor):
        src = obj
        if src.device.type != "cpu":
            src = src.cpu()
        if not src.is_contiguous() or src.storage_offset() != 0:
            src = src.contiguous()
        nb = src.nbytes
        raw = b""
        if nb > 0:
            buf = ctypes.create_string_buffer(nb)
            _C.check_rc(_C.lib.gt_tensor_copy_to_host(src._ptr, buf, nb), "save tensor")
            raw = buf.raw
        return _SerializedTensor(
            tuple(src.shape),
            src.dtype.code,
            raw,
            bool(obj.requires_grad),
            is_param=isinstance(obj, Parameter),
        )
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_serializable(v) for v in obj)
    return obj


def _from_serializable(obj, map_dev: _dev.device | None = None):
    if isinstance(obj, _SerializedTensor):
        return obj.restore(map_dev)
    if isinstance(obj, dict):
        return {k: _from_serializable(v, map_dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v, map_dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_from_serializable(v, map_dev) for v in obj)
    return obj


def _resolve_map_device(map_location) -> _dev.device | None:
    if map_location is None:
        return None
    if isinstance(map_location, (_dev.device, str)):
        return _dev.as_device(map_location)
    if callable(map_location):
        return None
    return None


def save(obj, f, pickle_protocol: int = pickle.HIGHEST_PROTOCOL, **kwargs) -> None:
    (kwargs,)
    if isinstance(f, (str, Path)) and str(f).lower().endswith(".gguf"):
        from . import gguf as _gguf
        if isinstance(obj, Tensor):
            _gguf.save_gguf(str(f), {"tensor": obj})
        elif isinstance(obj, dict):
            tensors = {str(k): v for k, v in obj.items() if isinstance(v, Tensor)}
            meta = {str(k): v for k, v in obj.items() if not isinstance(v, Tensor)}
            _gguf.save_gguf(str(f), tensors, metadata=meta or None)
        else:
            raise TypeError("Saving as .gguf requires a Tensor or state_dict")
        return

    payload = _to_serializable(obj)
    if isinstance(f, (str, Path)):
        with open(f, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle_protocol)
    else:
        pickle.dump(payload, f, protocol=pickle_protocol)


def load(f, map_location=None, weights_only: bool = False, **kwargs):
    (weights_only, kwargs)
    if isinstance(f, (str, Path)):
        p = Path(f)
        with open(p, "rb") as fp:
            magic = fp.read(4)
        if magic == b"GGUF" or str(f).lower().endswith(".gguf"):
            from . import gguf as _gguf
            gf = _gguf.load_gguf(str(p))
            dev = _resolve_map_device(map_location)
            if dev is not None:
                return {k: v.to(dev) for k, v in gf.tensors.items()}
            return gf.tensors
        with open(p, "rb") as fp:
            raw_obj = pickle.load(fp)
    else:
        raw_obj = pickle.load(f)

    return _from_serializable(raw_obj, _resolve_map_device(map_location))
