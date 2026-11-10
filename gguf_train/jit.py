"""Compatibility torch.jit module backed by gguf_train._compat."""

from ._compat import _JitModule

ScriptModule = _JitModule.ScriptModule
script = _JitModule.script
trace = _JitModule.trace
ignore = _JitModule.ignore
unused = _JitModule.unused
export = _JitModule.export

__all__ = ["ScriptModule", "script", "trace", "ignore", "unused", "export"]
