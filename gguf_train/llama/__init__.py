"""Native llama.cpp GGUF model inference/fine-tuning, static ggml_opt_graph, and LoRA."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from .. import _C, _autograd_state, _device as _dev, _dtype as _dt
from .._creation import zeros
from .._functions import _c_i64_arr
from .._tensor import Tensor, _from_python
from ..nn import functional as F, init
from ..nn.modules.module import Module
from ..nn.parameter import Parameter


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
        kbuf = ctypes.create_string_buffer(512)
        vbuf = ctypes.create_string_buffer(2048)
        out = {}
        for i in range(n):
            ll.gt_llama_model_meta_key(self._ptr, i, kbuf, 512)
            key = kbuf.value.decode("utf-8", errors="replace")
            ll.gt_llama_model_meta_val(self._ptr, kbuf.value, vbuf, 2048)
            out[key] = vbuf.value.decode("utf-8", errors="replace")
        return out

    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
        ll = _C.llama_lib()
        b = text.encode("utf-8")
        cap = max(len(b) + 16, 64)
        arr = (ctypes.c_int32 * cap)()
        n = int(ll.gt_llama_tokenize(self._ptr, b, len(b), arr, cap, 1 if add_special else 0, 1 if parse_special else 0))
        if n < 0:
            cap = -n + 4
            arr = (ctypes.c_int32 * cap)()
            n = int(ll.gt_llama_tokenize(self._ptr, b, len(b), arr, cap, 1 if add_special else 0, 1 if parse_special else 0))
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
        if rc < 0:
            cap = -rc + 16
            buf = ctypes.create_string_buffer(cap)
            rc = int(ll.gt_llama_detokenize(self._ptr, arr, n, buf, cap, 1 if remove_special else 0, 1 if unparse_special else 0))
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
        if isinstance(tokens, Tensor):
            rows = tokens.tolist()
        else:
            rows = list(tokens)
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
    """High-speed native ggml_opt_graph builder and trainer (zero Python loop overhead)."""

    def __init__(self, device: str | _dev.device = "cpu", loss_type: str = "mse", optimizer: str = "adamw", nbatch_physical: int = 32, nbatch_logical: int = 32) -> None:
        _dev.as_device(device)
        loss_map = {"mean": 0, "sum": 1, "cross_entropy": 2, "mse": 3}
        opt_map = {"adamw": 0, "sgd": 1}
        opt_code = opt_map.get(optimizer.lower(), 0)
        self._ptr = _C.check_ptr(
            _C.lib.gt_opt_graph_create(loss_map[loss_type.lower()], opt_code, int(nbatch_physical), int(nbatch_logical)),
            "opt_graph_create",
        )

    def __del__(self):
        ptr = getattr(self, "_ptr", 0)
        if ptr and _C is not None and getattr(_C, "lib", None) is not None:
            _C.lib.gt_opt_graph_free(ptr)
            self._ptr = 0

    def input(self, shape: tuple[int, ...]) -> int:
        return int(_C.lib.gt_opt_graph_input(self._ptr, len(shape), _c_i64_arr(shape)))

    def param(self, tensor: Tensor) -> int:
        return int(_C.lib.gt_opt_graph_param(self._ptr, tensor._ptr))

    def linear(self, x_id: int, w_id: int, b_id: int = -1) -> int:
        return int(_C.lib.gt_opt_graph_linear(self._ptr, x_id, w_id, b_id))

    def unary(self, x_id: int, op: str) -> int:
        op_map = {"relu": _C.UNARY_RELU, "silu": _C.UNARY_SILU, "gelu": _C.UNARY_GELU, "tanh": _C.UNARY_TANH, "sigmoid": _C.UNARY_SIGMOID}
        return int(_C.lib.gt_opt_graph_unary(self._ptr, x_id, op_map[op]))

    def add(self, a_id: int, b_id: int) -> int:
        return int(_C.lib.gt_opt_graph_add(self._ptr, a_id, b_id))

    def mul(self, a_id: int, b_id: int) -> int:
        return int(_C.lib.gt_opt_graph_mul(self._ptr, a_id, b_id))

    def rms_norm(self, x_id: int, w_id: int = -1, eps: float = 1e-6) -> int:
        return int(_C.lib.gt_opt_graph_rms_norm(self._ptr, x_id, w_id, float(eps)))

    def softmax(self, x_id: int) -> int:
        return int(_C.lib.gt_opt_graph_softmax(self._ptr, x_id))

    def fit(self, output_id: int, dataset_x: Tensor, dataset_y: Tensor, epochs: int = 10, lr: float = 1e-3, wd: float = 0.0, val_split: float = 0.0, silent: bool = True) -> tuple[float, float]:
        t_loss = ctypes.c_double(0.0)
        v_loss = ctypes.c_double(0.0)
        _C.check_rc(
            _C.lib.gt_opt_graph_fit(
                self._ptr,
                int(output_id),
                dataset_x._ptr,
                dataset_y._ptr,
                int(epochs),
                float(lr),
                float(wd),
                float(val_split),
                1 if silent else 0,
                ctypes.byref(t_loss),
                ctypes.byref(v_loss),
            ),
            "opt_graph_fit",
        )
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
