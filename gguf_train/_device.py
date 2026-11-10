"""Device, Size, layout, memory_format, and Generator abstractions."""

from __future__ import annotations

import builtins

# Match csrc/gguf_train.h enum gt_device_type
DEV_CPU = 0
DEV_CUDA = 1
DEV_MPS = 2
DEV_VULKAN = 3
DEV_META = 4

_TYPE_TO_ID = {
    "cpu": DEV_CPU,
    "cuda": DEV_CUDA,
    "mps": DEV_MPS,
    "vulkan": DEV_VULKAN,
    "meta": DEV_META,
}
_ID_TO_TYPE = {v: k for k, v in _TYPE_TO_ID.items()}


class device:
    """PyTorch-compatible device descriptor backed by ggml backends."""

    __slots__ = ("type", "index", "gt_type")

    def __init__(self, dev: str | int | device | None = None, index: int | None = None):
        if isinstance(dev, device):
            self.type = dev.type
            self.index = dev.index if index is None else int(index)
        elif dev is None:
            self.type = _DEFAULT_DEVICE.type
            self.index = _DEFAULT_DEVICE.index if index is None else int(index)
        elif isinstance(dev, int):
            self.type = "cuda"
            self.index = int(dev)
        elif isinstance(dev, str):
            s = dev.strip().lower()
            if ":" in s:
                t, idx_s = s.split(":", 1)
                self.type = t
                self.index = int(idx_s)
            else:
                self.type = s
                self.index = 0 if index is None else int(index)
        else:
            raise TypeError(f"Invalid device specification: {dev!r}")

        if self.type not in _TYPE_TO_ID:
            raise RuntimeError(f"Expected one of cpu, cuda, mps, vulkan, meta device type; got {self.type!r}")
        self.gt_type = _TYPE_TO_ID[self.type]

    def __repr__(self) -> str:
        if self.type == "cpu":
            return "device(type='cpu')"
        return f"device(type='{self.type}', index={self.index})"

    def __str__(self) -> str:
        if self.type == "cpu" or self.index == 0:
            return self.type if self.type == "cpu" else f"{self.type}:{self.index}"
        return f"{self.type}:{self.index}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = device(other)
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        return False

    def __hash__(self) -> int:
        return hash((self.type, self.index))

    @property
    def dev_type(self) -> int:
        return self.gt_type


_DEFAULT_DEVICE = device("cpu", 0)


def get_default_device() -> device:
    return _DEFAULT_DEVICE


def set_default_device(dev: str | device | None) -> None:
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device("cpu") if dev is None else device(dev)


def resolve_device(dev: str | int | device | None) -> device:
    if dev is None:
        return _DEFAULT_DEVICE
    return device(dev)


as_device = resolve_device


def device_from_gt(dev_type: int, dev_idx: int) -> device:
    return device(_ID_TO_TYPE.get(dev_type, "cpu"), dev_idx)


class Size(tuple):
    """PyTorch-compatible Size tuple with numel()."""

    def __new__(cls, iterable=()):
        return super().__new__(cls, (builtins.int(x) for x in iterable))

    def numel(self) -> int:
        n = 1
        for d in self:
            n *= d
        return n

    def __repr__(self) -> str:
        return f"torch.Size([{', '.join(str(x) for x in self)}])"


class memory_format:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


contiguous_format = memory_format("contiguous_format")
channels_last = memory_format("channels_last")
channels_last_3d = memory_format("channels_last_3d")
preserve_format = memory_format("preserve_format")


class layout:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


strided = layout("strided")
sparse_coo = layout("sparse_coo")
sparse_csr = layout("sparse_csr")


class Generator:
    """PyTorch-compatible PRNG generator wrapping gguf_train RNG state."""

    def __init__(self, device: str | device | None = None):
        self.device = resolve_device(device)
        self._seed = 0x5A17_C0DE_1234_5678

    def manual_seed(self, seed: int) -> "Generator":
        from . import _C
        self._seed = builtins.int(seed) & 0xFFFFFFFFFFFFFFFF
        _C.lib.gt_manual_seed(self._seed)
        return self

    def seed(self) -> int:
        import os
        s = builtins.int.from_bytes(os.urandom(8), "little")
        self.manual_seed(s)
        return s

    def initial_seed(self) -> int:
        return self._seed

    def get_state(self):
        from ._creation import get_rng_state
        return get_rng_state()

    def set_state(self, state):
        from ._creation import set_rng_state
        set_rng_state(state)


default_generator = Generator()

