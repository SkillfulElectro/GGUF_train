"""Compatibility torch.onnx module backed by gguf_train._compat."""

from ._compat import _OnnxModule

export = _OnnxModule.export

__all__ = ["export"]
