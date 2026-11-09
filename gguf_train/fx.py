"""Compatibility torch.fx module backed by gguf_train._compat."""

from ._compat import _FxModule

wrap = _FxModule.wrap
symbolic_trace = _FxModule.symbolic_trace

__all__ = ["wrap", "symbolic_trace"]
