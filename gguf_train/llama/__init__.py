"""Native llama.cpp GGUF model inference/fine-tuning, static ggml_opt_graph, and LoRA."""

from __future__ import annotations

import ctypes
import math
import warnings
from pathlib import Path

from .. import _C, _autograd_state, _device as _dev, _dtype as _dt
from .._creation import zeros
from .._functions import _c_i64_arr
from .._tensor import Tensor, _from_python
from ..nn import functional as F, init
from ..nn.modules.module import Module
from ..nn.parameter import Parameter


# What the native tokenizer calls return when they fail outright (rather than when the
# output buffer is merely too small, which is reported as the negated size needed).
_INT32_MIN = -(2**31)

# llama.cpp sizes a context in multiples of this many tokens, see LlamaContext.finetune.
_TRAIN_WINDOW_MULTIPLE = 256


class LlamaModel:
    """Wrapper over a GGUF model loaded by llama.cpp."""

    def __init__(self, path: str | Path, n_gpu_layers: int = 0, vocab_only: bool = False, use_mmap: bool = True) -> None:
        ll = _C.llama_lib()
        self.path = str(path)
        self._ptr = _C.check_ptr(
            ll.gt_llama_load_model(self.path.encode("utf-8"), int(n_gpu_layers), 1 if vocab_only else 0, 1 if use_mmap else 0),
            f"llama load {self.path}",
        )

    def __del__(self):
        ptr = getattr(self, "_ptr", 0)
        if ptr and _C is not None:
            _C.llama_lib().gt_llama_free_model(ptr)
            self._ptr = 0

    def save_model(self, path: str | Path) -> None:
        _C.check_rc(_C.llama_lib().gt_llama_save_model(self._ptr, str(path).encode("utf-8")), "llama save_model")

    @property
    def n_vocab(self) -> int:
        return int(_C.llama_lib().gt_llama_n_vocab(self._ptr))

    @property
    def n_ctx_train(self) -> int:
        return int(_C.llama_lib().gt_llama_n_ctx_train(self._ptr))

    @property
    def n_embd(self) -> int:
        return int(_C.llama_lib().gt_llama_n_embd(self._ptr))

    @property
    def n_layer(self) -> int:
        return int(_C.llama_lib().gt_llama_n_layer(self._ptr))

    @property
    def n_head(self) -> int:
        return int(_C.llama_lib().gt_llama_n_head(self._ptr))

    @property
    def n_head_kv(self) -> int:
        return int(_C.llama_lib().gt_llama_n_head_kv(self._ptr))

    @property
    def n_params(self) -> int:
        return int(_C.llama_lib().gt_llama_model_n_params(self._ptr))

    @property
    def model_size(self) -> int:
        return int(_C.llama_lib().gt_llama_model_size(self._ptr))

    @property
    def bos_token_id(self) -> int:
        return int(_C.llama_lib().gt_llama_vocab_bos(self._ptr))

    @property
    def eos_token_id(self) -> int:
        return int(_C.llama_lib().gt_llama_vocab_eos(self._ptr))

    @property
    def pad_token_id(self) -> int:
        return int(_C.llama_lib().gt_llama_vocab_pad(self._ptr))

    @property
    def description(self) -> str:
        buf = ctypes.create_string_buffer(256)
        _C.llama_lib().gt_llama_model_desc(self._ptr, buf, 256)
        return buf.value.decode("utf-8", errors="replace")

    @property
    def metadata(self) -> dict[str, str]:
        ll = _C.llama_lib()
        n = int(ll.gt_llama_model_meta_count(self._ptr))

        def text(read) -> bytes:
            # llama.cpp reports the full length, so a value that did not fit is read again
            # into a buffer that holds it (chat templates run to many kilobytes).
            size = 512
            while True:
                buf = ctypes.create_string_buffer(size)
                needed = int(read(buf, size))
                if needed < size:
                    return buf.value
                size = needed + 1

        out = {}
        for i in range(n):
            key = text(lambda buf, size: ll.gt_llama_model_meta_key(self._ptr, i, buf, size))
            value = text(lambda buf, size: ll.gt_llama_model_meta_val(self._ptr, key, buf, size))
            out[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
        return out

    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
        ll = _C.llama_lib()
        b = text.encode("utf-8")
        cap = max(len(b) + 16, 64)
        arr = (ctypes.c_int32 * cap)()
        n = int(ll.gt_llama_tokenize(self._ptr, b, len(b), arr, cap, 1 if add_special else 0, 1 if parse_special else 0))
        if _INT32_MIN < n < 0:
            # The buffer was too small; -n is the number of tokens needed.
            cap = -n + 4
            arr = (ctypes.c_int32 * cap)()
            n = int(ll.gt_llama_tokenize(self._ptr, b, len(b), arr, cap, 1 if add_special else 0, 1 if parse_special else 0))
        if n < 0:
            raise RuntimeError(_C.last_error() or "llama tokenize failed")
        return [int(arr[i]) for i in range(n)]

    def detokenize(self, tokens: list[int], remove_special: bool = False, unparse_special: bool = True) -> str:
        ll = _C.llama_lib()
        n = len(tokens)
        if n == 0:
            return ""
        arr = (ctypes.c_int32 * n)(*[int(t) for t in tokens])
        cap = max(n * 16, 256)
        buf = ctypes.create_string_buffer(cap)
        rc = int(ll.gt_llama_detokenize(self._ptr, arr, n, buf, cap, 1 if remove_special else 0, 1 if unparse_special else 0))
        if _INT32_MIN < rc < 0:
            # The buffer was too small; -rc is the number of bytes needed.
            cap = -rc + 16
            buf = ctypes.create_string_buffer(cap)
            rc = int(ll.gt_llama_detokenize(self._ptr, arr, n, buf, cap, 1 if remove_special else 0, 1 if unparse_special else 0))
        if rc < 0:
            raise RuntimeError(_C.last_error() or "llama detokenize failed")
        return buf.raw[:rc].decode("utf-8", errors="replace")

    def new_context(self, n_ctx: int = 512, n_batch: int = 512, n_threads: int = 4, embeddings: bool = False) -> LlamaContext:
        return LlamaContext(self, n_ctx=n_ctx, n_batch=n_batch, n_threads=n_threads, embeddings=embeddings)


class LlamaContext:
    """Execution & native training context over a LlamaModel."""

    def __init__(self, model: LlamaModel, n_ctx: int = 512, n_batch: int = 512, n_threads: int = 4, embeddings: bool = False) -> None:
        ll = _C.llama_lib()
        self.model = model
        self._ptr = _C.check_ptr(
            ll.gt_llama_new_context(model._ptr, int(n_ctx), int(n_batch), int(n_threads), 1 if embeddings else 0),
            "llama new_context",
        )

    def __del__(self):
        ptr = getattr(self, "_ptr", 0)
        if ptr and _C is not None:
            _C.llama_lib().gt_llama_free_context(ptr)
            self._ptr = 0

    def clear_kv(self) -> None:
        _C.llama_lib().gt_llama_clear_kv(self._ptr)

    @property
    def n_ctx(self) -> int:
        return int(_C.llama_lib().gt_llama_n_ctx(self._ptr))

    def forward_logits(self, tokens: list[int] | Tensor, pos_offset: int = 0, only_last: bool = False) -> Tensor:
        tok_list = tokens.tolist() if isinstance(tokens, Tensor) else list(tokens)
        arr = (ctypes.c_int32 * len(tok_list))(*[int(t) for t in tok_list])
        ptr = _C.check_ptr(
            _C.llama_lib().gt_llama_forward_logits(self._ptr, arr, len(tok_list), int(pos_offset), 1 if only_last else 0),
            "llama forward_logits",
        )
        return Tensor(_ptr=ptr, requires_grad=False)

    def forward_embeddings(self, tokens: list[int] | Tensor) -> Tensor:
        tok_list = tokens.tolist() if isinstance(tokens, Tensor) else list(tokens)
        arr = (ctypes.c_int32 * len(tok_list))(*[int(t) for t in tok_list])
        ptr = _C.check_ptr(
            _C.llama_lib().gt_llama_forward_embeddings(self._ptr, arr, len(tok_list)),
            "llama forward_embeddings",
        )
        return Tensor(_ptr=ptr, requires_grad=False)

    def generate(self, prompt: str, max_new_tokens: int = 32, temperature: float = 0.0, top_k: int = 40) -> str:
        self.clear_kv()
        tokens = self.model.tokenize(prompt, add_special=True)
        out_tokens = []
        pos = 0
        logits = self.forward_logits(tokens, pos_offset=pos, only_last=True)[0]
        pos += len(tokens)
        eos = self.model.eos_token_id
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                nxt = int(logits.argmax().item())
            else:
                scaled = logits / float(temperature)
                if top_k > 0 and top_k < scaled.shape[0]:
                    vals, idxs = scaled.topk(top_k)
                    p = F.softmax(vals, dim=-1)
                    pick = int(p.multinomial(1).item())
                    nxt = int(idxs[pick].item())
                else:
                    p = F.softmax(scaled, dim=-1)
                    nxt = int(p.multinomial(1).item())
            if nxt == eos:
                break
            out_tokens.append(nxt)
            logits = self.forward_logits([nxt], pos_offset=pos, only_last=True)[0]
            pos += 1
        return self.model.detokenize(out_tokens)

    def finetune(
        self,
        tokens: list[list[int]] | Tensor,
        epochs: int = 1,
        batch_size: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        val_split: float = 0.0,
        optimizer: str = "adamw",
    ) -> None:
        """Fine-tune the model in place on next-token prediction with llama.cpp's native trainer.

        ``tokens`` holds equal-length token sequences. Every sequence is trained on as an
        independent sample: token ``k`` is predicted from tokens ``0..k-1`` of the same
        sequence, and nothing beyond the sequence (no padding) enters the loss. Sequences
        longer than this context's ``n_ctx + 1`` tokens are truncated to that length.

        Training runs in a temporary llama.cpp context, so this context stays usable for
        inference afterwards (its KV cache is cleared, since the weights changed) and
        ``finetune`` can be called repeatedly.

        Limits inherited from llama.cpp's trainer:

        * The model must be loaded with ``use_mmap=False``; memory-mapped weights are read-only.
        * One optimizer step is taken per sequence, so ``batch_size`` must be 1.
        * Only F32 tensors are trained (quantized tensors and the token embedding stay frozen).
        * The attention key/value projections only receive gradients when a sequence fills the
          whole training window, i.e. when it has ``256 * k + 1`` tokens. Other lengths train
          the remaining weights and emit a warning.
        """
        if isinstance(tokens, Tensor):
            rows = tokens.tolist()
        else:
            rows = [list(row) for row in tokens]
        if not rows or not rows[0] or len(rows[0]) < 2:
            raise ValueError("tokens must contain at least one sequence of two or more tokens")
        n_samples = len(rows)
        seq_len = len(rows[0])
        if any(len(row) != seq_len for row in rows):
            raise ValueError("token sequences must have equal lengths")
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1)")
        if optimizer.lower() not in ("adamw", "sgd"):
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if int(batch_size) != 1:
            raise ValueError(
                f"batch_size={batch_size} is not supported: llama.cpp's trainer takes one optimizer step per "
                "sequence and cannot combine several sequences into one step; pass batch_size=1"
            )
        if not (math.isfinite(lr) and lr > 0.0):
            raise ValueError("lr must be a positive finite number")
        if not 0.0 <= weight_decay <= 1.0:
            raise ValueError("weight_decay must be in [0, 1]")
        if int(epochs) < 0:
            raise ValueError("epochs must be non-negative")

        usable = min(seq_len - 1, self.n_ctx)
        if int(epochs) > 0 and usable % _TRAIN_WINDOW_MULTIPLE != 0:
            warnings.warn(
                f"finetune: {usable} trained positions per sequence do not fill a llama.cpp training window "
                f"(a multiple of {_TRAIN_WINDOW_MULTIPLE} tokens), so the attention key/value projection weights "
                f"receive no gradient in this run. Use sequences of {_TRAIN_WINDOW_MULTIPLE}*k + 1 tokens to train "
                "all weights.",
                stacklevel=2,
            )
        flat = [int(tok) for row in rows for tok in row]
        arr = (ctypes.c_int32 * len(flat))(*flat)
        opt_code = 1 if optimizer.lower() == "sgd" else 0
        _C.check_rc(
            _C.llama_lib().gt_llama_finetune(
                self._ptr, arr, n_samples, seq_len, int(epochs), int(batch_size), float(lr), float(weight_decay), float(val_split), opt_code
            ),
            "llama finetune",
        )


class OptGraph:
    """Static graph trained natively by ggml_opt (no Python in the training loop).

    Build the graph with :meth:`input`, :meth:`param` and the operation methods, each of
    which returns a node id, then call :meth:`fit`. The parameters are float32 tensors and
    are updated in place.

    ``input`` must be declared as ``(nbatch_physical, features)``; ``fit`` feeds it one
    batch of dataset rows at a time. A graph can be fitted any number of times.
    """

    _LOSSES = {"mean": 0, "sum": 1, "cross_entropy": 2, "mse": 3}
    _OPTIMIZERS = {"adamw": 0, "sgd": 1}
    # Only operations ggml can differentiate. ("gelu" is not among them: ggml has no
    # backward pass for it and no exact way to spell it with operations that have one.)
    _UNARY = {
        "relu": _C.UNARY_RELU,
        "silu": _C.UNARY_SILU,
        "sigmoid": _C.UNARY_SIGMOID,
        "tanh": _C.UNARY_TANH,
        "softplus": _C.UNARY_SOFTPLUS,
        "exp": _C.UNARY_EXP,
        "log": _C.UNARY_LOG,
        "square": _C.UNARY_SQUARE,
        "sqrt": _C.UNARY_SQRT,
        "neg": _C.UNARY_NEG,
        "abs": _C.UNARY_ABS,
    }

    def __init__(self, device: str | _dev.device = "cpu", loss_type: str = "mse", optimizer: str = "adamw", nbatch_physical: int = 32, nbatch_logical: int = 32) -> None:
        _dev.as_device(device)
        loss_code = self._LOSSES.get(str(loss_type).lower())
        if loss_code is None:
            raise ValueError(f"OptGraph: unknown loss_type {loss_type!r}; expected one of {sorted(self._LOSSES)}")
        opt_code = self._OPTIMIZERS.get(str(optimizer).lower())
        if opt_code is None:
            raise ValueError(f"OptGraph: unknown optimizer {optimizer!r}; expected one of {sorted(self._OPTIMIZERS)}")
        if int(nbatch_physical) < 1:
            raise ValueError(f"OptGraph: nbatch_physical must be at least 1, got {nbatch_physical}")
        self.loss_type = str(loss_type).lower()
        self._ptr = _C.check_ptr(
            _C.lib.gt_opt_graph_create(loss_code, opt_code, int(nbatch_physical), int(nbatch_logical)),
            "opt_graph_create",
        )

    def __del__(self):
        ptr = getattr(self, "_ptr", 0)
        if ptr and _C is not None and getattr(_C, "lib", None) is not None:
            _C.lib.gt_opt_graph_free(ptr)
            self._ptr = 0

    @staticmethod
    def _node(node_id: int) -> int:
        """Node ids are non-negative; the native builders return -1 after recording an error."""
        if node_id < 0:
            raise RuntimeError(_C.last_error() or "OptGraph: the native graph builder failed")
        return int(node_id)

    def input(self, shape: tuple[int, ...]) -> int:
        shape = tuple(int(d) for d in shape)
        return self._node(_C.lib.gt_opt_graph_input(self._ptr, len(shape), _c_i64_arr(shape)))

    def param(self, tensor: Tensor) -> int:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"OptGraph.param expects a Tensor, got {type(tensor).__name__}")
        return self._node(_C.lib.gt_opt_graph_param(self._ptr, tensor._ptr))

    def linear(self, x_id: int, w_id: int, b_id: int = -1) -> int:
        return self._node(_C.lib.gt_opt_graph_linear(self._ptr, int(x_id), int(w_id), int(b_id)))

    def unary(self, x_id: int, op: str) -> int:
        code = self._UNARY.get(str(op).lower())
        if code is None:
            raise ValueError(f"OptGraph.unary: {op!r} cannot be trained through (ggml has no backward pass for it); supported: {sorted(self._UNARY)}")
        return self._node(_C.lib.gt_opt_graph_unary(self._ptr, int(x_id), code))

    def add(self, a_id: int, b_id: int) -> int:
        return self._node(_C.lib.gt_opt_graph_add(self._ptr, int(a_id), int(b_id)))

    def mul(self, a_id: int, b_id: int) -> int:
        return self._node(_C.lib.gt_opt_graph_mul(self._ptr, int(a_id), int(b_id)))

    def rms_norm(self, x_id: int, w_id: int = -1, eps: float = 1e-6) -> int:
        return self._node(_C.lib.gt_opt_graph_rms_norm(self._ptr, int(x_id), int(w_id), float(eps)))

    def softmax(self, x_id: int) -> int:
        return self._node(_C.lib.gt_opt_graph_softmax(self._ptr, int(x_id)))

    def fit(self, output_id: int, dataset_x: Tensor, dataset_y: Tensor | None = None, epochs: int = 10, lr: float = 1e-3, wd: float = 0.0, val_split: float = 0.0, silent: bool = True) -> tuple[float, float]:
        """Train the parameters ``output_id`` depends on; returns ``(train_loss, val_loss)`` of the last epoch.

        ``dataset_x`` has one row per sample; rows beyond a multiple of ``nbatch_physical``
        are dropped. ``dataset_y`` holds one row of targets per sample for the ``"mse"`` and
        ``"cross_entropy"`` losses (one-hot or probability rows for the latter) and must be
        left out for ``"mean"`` and ``"sum"``, which are computed from the output alone.
        ``val_loss`` is 0.0 when no batches are held out.
        """
        if not isinstance(dataset_x, Tensor):
            raise TypeError(f"OptGraph.fit expects dataset_x to be a Tensor, got {type(dataset_x).__name__}")
        if dataset_y is not None and not isinstance(dataset_y, Tensor):
            raise TypeError(f"OptGraph.fit expects dataset_y to be a Tensor or None, got {type(dataset_y).__name__}")
        if int(epochs) < 0:
            raise ValueError(f"OptGraph.fit: epochs must be non-negative, got {epochs}")
        t_loss = ctypes.c_double(0.0)
        v_loss = ctypes.c_double(0.0)
        rc = _C.lib.gt_opt_graph_fit(
            self._ptr,
            int(output_id),
            dataset_x._ptr,
            dataset_y._ptr if dataset_y is not None else None,
            int(epochs),
            float(lr),
            float(wd),
            float(val_split),
            1 if silent else 0,
            ctypes.byref(t_loss),
            ctypes.byref(v_loss),
        )
        if rc != 0:
            raise RuntimeError(_C.last_error() or f"OptGraph.fit failed with code {rc}")
        return float(t_loss.value), float(v_loss.value)


class LoRALinear(Module):
    """Low-Rank Adaptation wrapper around nn.Linear (works even with quantized GGUF base weights!)."""

    def __init__(self, base_linear: Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        dev = base_linear.weight.device
        dt = base_linear.weight.dtype if base_linear.weight.is_floating_point() else _dt.float32
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = float(dropout)
        self.lora_A = Parameter(zeros((self.rank, in_f), dtype=dt, device=dev))
        self.lora_B = Parameter(zeros((out_f, self.rank), dtype=dt, device=dev))
        init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        dx = F.dropout(x, p=self.dropout, training=self.training) if self.dropout > 0.0 else x
        lora_out = F.linear(F.linear(dx, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> Module:
        from ..nn.modules.linear import Linear
        w = self.base.weight
        if w.dtype.is_quantized:
            w = w.dequantize(_dt.float32)
        merged_w = w.detach() + (self.lora_B.matmul(self.lora_A) * self.scaling).detach()
        lin = Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None, device=merged_w.device, dtype=merged_w.dtype)
        with _autograd_state.no_grad():
            lin.weight.copy_(merged_w)
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias.detach())
        return lin


def inject_lora(module: Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0, target_names: set[str] | None = None) -> Module:
    """Freeze base model and replace Linear layers (optionally filtered by name) with LoRALinear."""
    from ..nn.modules.linear import Linear
    for p in module.parameters():
        p.requires_grad_(False)
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        if isinstance(child, Linear) and (target_names is None or name in target_names):
            module._modules[name] = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
        else:
            inject_lora(child, rank=rank, alpha=alpha, dropout=dropout, target_names=target_names)
    return module


def merge_lora(module: Module) -> Module:
    for name, child in list(module._modules.items()):
        if isinstance(child, LoRALinear):
            module._modules[name] = child.merge_weights()
        elif child is not None:
            merge_lora(child)
    return module


__all__ = [
    "LlamaModel",
    "LlamaContext",
    "OptGraph",
    "LoRALinear",
    "inject_lora",
    "merge_lora",
]
