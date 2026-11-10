"""Autograd API compatible with torch.autograd."""

from __future__ import annotations

from .._autograd_engine import Function, backward, grad
from .._autograd_state import (
    detect_anomaly,
    enable_grad,
    inference_mode,
    is_grad_enabled,
    no_grad,
    set_grad_enabled,
    set_multithreading_enabled,
)
from .._tensor import Tensor as Variable
from . import functional
from .gradcheck import gradcheck, gradgradcheck

__all__ = [
    "Function",
    "Variable",
    "backward",
    "grad",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "inference_mode",
    "detect_anomaly",
    "set_multithreading_enabled",
    "gradcheck",
    "gradgradcheck",
    "functional",
]
