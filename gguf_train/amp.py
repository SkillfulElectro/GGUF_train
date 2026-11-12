"""Automatic Mixed Precision (AMP): autocast and GradScaler, compatible with torch.amp.

Inside ``with autocast(device_type, dtype=...)`` floating point tensors on devices of that type
are cast on their way into certain operations. Which operations, and to what, is one table for
every device type - PyTorch's CUDA autocast lists (v2.5):

autocast dtype (float16 / bfloat16)
    ``@``/matmul, mm, bmm, mv, addmm, baddbmm, einsum, linalg.multi_dot, linalg.vecdot,
    F.linear, F.conv1d/2d/3d, F.conv_transpose2d, F.prelu, F.scaled_dot_product_attention -
    and so the layers built on them (Linear, Conv*, MultiheadAttention, ...).

float32
    softmax, log_softmax, softmin, F.layer_norm, F.group_norm, F.normalize, cosine_similarity,
    cdist, pdist, F.interpolate, F.softplus, every loss function in torch.nn.functional,
    the reductions sum, prod, cumsum, cumprod, logsumexp and norm (also linalg.norm,
    vector_norm, matrix_norm), and the elementwise acos, asin, cosh, erfinv, exp, expm1, log,
    log10, log2, log1p, pow, reciprocal, rsqrt, sinh and tan.

widest operand type
    dot, tensordot, F.bilinear (and cat, stack, atan2, ..., which promote on their own).

Everything else runs in the dtype of its inputs. Not autocast at all: float64 and integer
tensors, tensors on other device types, in-place operations, calls that pass ``out=`` or an
explicit ``dtype=``, and backward passes - a backward pass runs in the types its forward pass
used, wherever it is called. The casts are ordinary differentiable operations, so gradients
arrive at the parameters in the parameters' own dtype.

Two differences from PyTorch, both on purpose:

* PyTorch keeps a second, shorter table for CPU autocast, which leaves softmax, layer_norm,
  sum, exp and the like in the dtype of their input. Here CPU tensors get the table above
  too, so those run in float32 on every device.
* F.binary_cross_entropy runs in float32 (as on PyTorch CPU); PyTorch CUDA refuses it.
"""

from __future__ import annotations

import functools
import threading
import warnings

from . import _dtype as _dt

_AMP_LOCAL = threading.local()

_SUPPORTED_DTYPES = (_dt.bfloat16, _dt.float16)


def _active() -> dict:
    """Device type -> autocast dtype, for the autocast regions the calling thread is inside."""
    try:
        return _AMP_LOCAL.active
    except AttributeError:
        _AMP_LOCAL.active = {}
        _AMP_LOCAL.entered = []
        return _AMP_LOCAL.active


def _default_dtype(device_type: str):
    return _dt.float16 if device_type == "cuda" else _dt.bfloat16


class _suspended:
    """``with _suspended():`` - nothing inside is autocast. For the body of an operation whose
    operands have been cast already, and for backward passes."""

    __slots__ = ("_saved",)

    def __enter__(self):
        active = _active()
        self._saved = dict(active) if active else None
        active.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._saved is not None:
            active = _active()
            active.clear()
            active.update(self._saved)
        return False


def _snapshot() -> dict:
    """What autocast is doing on this thread right now, for `_restored`."""
    return dict(_active())


class _restored:
    """``with _restored(state):`` - autocast as it was when ``state = _snapshot()`` was taken.
    torch.utils.checkpoint recomputes a function during the backward pass, where autocast is
    suspended; the recomputation has to make the tensors the forward pass made."""

    __slots__ = ("_state", "_saved")

    def __init__(self, state: dict):
        self._state = state

    def __enter__(self):
        active = _active()
        self._saved = dict(active)
        active.clear()
        active.update(self._state)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        active = _active()
        active.clear()
        active.update(self._saved)
        return False


class autocast:
    """Context manager / decorator for mixed-precision regions; see the module documentation
    for what is cast. ``dtype`` defaults to bfloat16 on CPU and float16 on CUDA."""

    def __init__(self, device_type: str = "cpu", dtype=None, enabled: bool = True, cache_enabled: bool | None = None):
        (cache_enabled,)
        if not isinstance(device_type, str):
            kind = getattr(device_type, "type", None)   # a torch.device
            if not isinstance(kind, str):
                raise ValueError(f"Expected `device_type` of type `str`, got: `{type(device_type)}`")
            device_type = kind
        self.device_type = device_type
        self.dtype = _dt.as_dtype(dtype) if dtype is not None else _default_dtype(device_type)
        self.enabled = bool(enabled)
        if self.enabled and self.dtype not in _SUPPORTED_DTYPES:
            warnings.warn(
                f"In {device_type} autocast, but the target dtype {self.dtype} is not supported. Disabling autocast.\n"
                "Autocast supports torch.bfloat16 and torch.float16.",
                UserWarning,
                stacklevel=2,
            )
            self.enabled = False

    # What the block replaces is remembered per thread, not on the object: one autocast object
    # may be entered again while it is open, and by several threads at once.

    def __enter__(self):
        active = _active()
        _AMP_LOCAL.entered.append((self, active.get(self.device_type)))
        if self.enabled:
            active[self.device_type] = self.dtype
        else:
            active.pop(self.device_type, None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        active = _active()
        entered = _AMP_LOCAL.entered
        for i in range(len(entered) - 1, -1, -1):
            if entered[i][0] is self:
                previous = entered.pop(i)[1]
                break
        else:
            return False        # left without having been entered on this thread
        if previous is None:
            active.pop(self.device_type, None)
        else:
            active[self.device_type] = previous
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def is_autocast_enabled(device_type: str = "cpu") -> bool:
    return device_type in _active()


def get_autocast_dtype(device_type: str = "cpu"):
    return _active().get(device_type, _default_dtype(device_type))


class GradScaler:
    """Gradient scaler for float16/bfloat16 mixed-precision training."""

    def __init__(
        self,
        device: str = "cpu",
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self.device = device
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._enabled = bool(enabled)
        self._growth_tracker = 0
        self._found_inf = False
        # id(optimizer) -> "unscaled" | "stepped", since the last update()
        self._optimizer_stage: dict[int, str] = {}

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale if self._enabled else 1.0

    def scale(self, outputs):
        if not self._enabled:
            return outputs
        if isinstance(outputs, (list, tuple)):
            return type(outputs)(o * self._scale for o in outputs)
        return outputs * self._scale

    def unscale_(self, optimizer) -> None:
        """Divides the gradients of `optimizer`'s parameters by the scale, in place - once per
        optimizer and step. Call it before clipping gradients; step() then leaves them alone."""
        if not self._enabled:
            return
        stage = self._optimizer_stage.get(id(optimizer))
        if stage == "unscaled":
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if stage == "stepped":
            raise RuntimeError("unscale_() is being called after step().")
        inv = 1.0 / self._scale
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad
                    if bool((~g.isfinite()).any().item()):
                        self._found_inf = True      # stays set until update(), whichever optimizer saw it
                    g.mul_(inv)
        self._optimizer_stage[id(optimizer)] = "unscaled"

    def step(self, optimizer, *args, **kwargs):
        """optimizer.step() on unscaled gradients; skipped when a gradient is inf or NaN."""
        if not self._enabled:
            return optimizer.step(*args, **kwargs)
        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")
        stage = self._optimizer_stage.get(id(optimizer))
        if stage == "stepped":
            raise RuntimeError("step() has already been called since the last update().")
        if stage is None:
            self.unscale_(optimizer)
        self._optimizer_stage[id(optimizer)] = "stepped"
        if not self._found_inf:
            return optimizer.step(*args, **kwargs)
        return None

    def update(self, new_scale: float | None = None) -> None:
        if not self._enabled:
            return
        self._optimizer_stage.clear()
        if new_scale is not None:
            self._scale = float(new_scale)
            self._found_inf = False
            return
        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def state_dict(self) -> dict:
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._scale = float(state_dict.get("scale", self._scale))
        self._growth_factor = float(state_dict.get("growth_factor", self._growth_factor))
        self._backoff_factor = float(state_dict.get("backoff_factor", self._backoff_factor))
        self._growth_interval = int(state_dict.get("growth_interval", self._growth_interval))
        self._growth_tracker = int(state_dict.get("_growth_tracker", 0))
