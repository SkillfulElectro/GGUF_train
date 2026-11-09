"""Compatibility shims for torch.jit, torch.fx, torch.onnx, torch.compile, and version metadata."""

from __future__ import annotations

import functools


def compile(model=None, *, fullgraph: bool = False, dynamic=None, backend: str = "ggml", mode=None, options=None, disable: bool = False):
    (fullgraph, dynamic, backend, mode, options, disable)
    if model is None:
        return lambda fn: fn
    return model


class _JitModule:
    ScriptModule = object

    @staticmethod
    def script(obj, optimize=None, _frames_up: int = 0, _rcb=None):
        (optimize, _frames_up, _rcb)
        return obj

    @staticmethod
    def trace(func, example_inputs=None, **kwargs):
        (example_inputs, kwargs)
        return func

    @staticmethod
    def ignore(drop=None, **kwargs):
        (kwargs,)
        if drop is not None and callable(drop):
            return drop
        return lambda fn: fn

    @staticmethod
    def unused(fn):
        return fn

    @staticmethod
    def export(fn):
        return fn


class _FxModule:
    @staticmethod
    def wrap(fn_or_name):
        if callable(fn_or_name):
            return fn_or_name
        return lambda fn: fn

    @staticmethod
    def symbolic_trace(root, concrete_args=None):
        (concrete_args,)
        return root


class _OnnxModule:
    @staticmethod
    def export(model, args, f, **kwargs):
        raise NotImplementedError("gguf_train uses native GGUF export via torch.save(..., 'model.gguf') or gguf_train.gguf.save_gguf()")


jit = _JitModule()
fx = _FxModule()
onnx = _OnnxModule()
