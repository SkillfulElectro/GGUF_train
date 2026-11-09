"""Drop-in PyTorch-compatible alias package backed by gguf_train (llama.cpp / ggml)."""

from __future__ import annotations

import sys

import gguf_train as _gt

# Register submodule aliases in sys.modules so both `import torch.X` and `from torch.X import Y` resolve to the exact gguf_train modules.
_SUBMODULE_ALIASES = {
    "torch.nn": _gt.nn,
    "torch.nn.functional": _gt.nn.functional,
    "torch.nn.init": _gt.nn.init,
    "torch.nn.parameter": _gt.nn.parameter,
    "torch.nn.utils": _gt.nn.utils,
    "torch.nn.modules": _gt.nn.modules,
    "torch.optim": _gt.optim,
    "torch.optim.lr_scheduler": _gt.optim.lr_scheduler,
    "torch.autograd": _gt.autograd,
    "torch.autograd.gradcheck": _gt.autograd.gradcheck,
    "torch.autograd.functional": _gt.autograd.functional,
    "torch.utils": _gt.utils,
    "torch.utils.data": _gt.utils.data,
    "torch.utils.checkpoint": _gt.utils.checkpoint,
    "torch.linalg": _gt.linalg,
    "torch.fft": _gt.fft,
    "torch.special": _gt.special,
    "torch.amp": _gt.amp,
    "torch.cpu": _gt.cpu,
    "torch.cuda": _gt.cuda,
    "torch.mps": _gt.mps,
    "torch.backends": _gt.backends,
    "torch.random": _gt.random,
    "torch.jit": _gt.jit,
    "torch.fx": _gt.fx,
    "torch.onnx": _gt.onnx,
    "torch.distributions": _gt.distributions,
    "torch.testing": _gt.testing,
    "torch.profiler": _gt.profiler,
    "torch.gguf": _gt.gguf,
    "torch.llama": _gt.llama,
}
for _alias, _mod in _SUBMODULE_ALIASES.items():
    sys.modules[_alias] = _mod

for _k, _v in _gt.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v

__version__ = _gt.__version__
