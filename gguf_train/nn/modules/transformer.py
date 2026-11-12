"""Transformer and MultiheadAttention modules compatible with torch.nn."""

from __future__ import annotations

import copy
import math

from ... import _dtype as _dt
from ..._creation import empty, zeros
from ..._ops import cat
from ..._tensor import Tensor
from .. import functional as F, init
from ..parameter import Parameter
from .container import ModuleList
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .normalization import LayerNorm


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = float(dropout)
        self.batch_first = bool(batch_first)
        self.kdim = self.embed_dim if kdim is None else int(kdim)
        self.vdim = self.embed_dim if vdim is None else int(vdim)

        dt = dtype or _dt.get_default_dtype()
        self._qkv_same = (self.kdim == self.embed_dim and self.vdim == self.embed_dim)
        if self._qkv_same:
            self.in_proj_weight = Parameter(empty((3 * self.embed_dim, self.embed_dim), dtype=dt, device=device))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
            init.xavier_uniform_(self.in_proj_weight)
        else:
            self.register_parameter("in_proj_weight", None)
            self.q_proj_weight = Parameter(empty((self.embed_dim, self.embed_dim), dtype=dt, device=device))
            self.k_proj_weight = Parameter(empty((self.embed_dim, self.kdim), dtype=dt, device=device))
            self.v_proj_weight = Parameter(empty((self.embed_dim, self.vdim), dtype=dt, device=device))
            init.xavier_uniform_(self.q_proj_weight)
            init.xavier_uniform_(self.k_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)

        if bias:
            self.in_proj_bias = Parameter(zeros((3 * self.embed_dim,), dtype=dt, device=device))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dt)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        is_batched = query.ndim == 3
        if not is_batched:
            if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
                raise RuntimeError(
                    "MultiheadAttention: query, key and value must all be 3-D (batched) or all 2-D (unbatched), "
                    f"got {query.ndim}-D, {key.ndim}-D and {value.ndim}-D"
                )
            # (L, E) has no batch dimension; add one where this layout keeps it
            batch_dim = 0 if self.batch_first else 1
            query, key, value = query.unsqueeze(batch_dim), key.unsqueeze(batch_dim), value.unsqueeze(batch_dim)
            if key_padding_mask is not None:
                if key_padding_mask.ndim != 1:
                    raise RuntimeError(
                        f"MultiheadAttention: key_padding_mask of an unbatched input must be 1-D (S), got {list(key_padding_mask.shape)}"
                    )
                key_padding_mask = key_padding_mask.unsqueeze(0)
        elif key.ndim != 3 or value.ndim != 3:
            raise RuntimeError(
                "MultiheadAttention: query, key and value must all be 3-D (batched) or all 2-D (unbatched), "
                f"got {query.ndim}-D, {key.ndim}-D and {value.ndim}-D"
            )
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, L, E = query.shape
        S = key.shape[1]

        if self._qkv_same:
            wq, wk, wv = self.in_proj_weight.chunk(3, dim=0)
        else:
            wq, wk, wv = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight

        if self.in_proj_bias is not None:
            bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)
        else:
            bq = bk = bv = None

        q = F.linear(query, wq, bq).reshape((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        k = F.linear(key, wk, bk).reshape((B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        v = F.linear(value, wv, bv).reshape((B, S, self.num_heads, self.head_dim)).transpose(1, 2)

        # One additive float mask, broadcastable to the scores (B, H, L, S).
        mask = None
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                if tuple(attn_mask.shape) != (L, S):
                    raise RuntimeError(f"MultiheadAttention: the 2D attn_mask has shape {list(attn_mask.shape)}, expected {[L, S]}")
                mask = self._additive_mask(attn_mask, q.dtype, "attn_mask").reshape((1, 1, L, S))
            elif attn_mask.ndim == 3:
                if tuple(attn_mask.shape) != (B * self.num_heads, L, S):
                    raise RuntimeError(
                        f"MultiheadAttention: the 3D attn_mask has shape {list(attn_mask.shape)}, expected {[B * self.num_heads, L, S]}"
                    )
                mask = self._additive_mask(attn_mask, q.dtype, "attn_mask").reshape((B, self.num_heads, L, S))
            else:
                raise RuntimeError(f"MultiheadAttention: attn_mask must be 2D or 3D, got {attn_mask.ndim}D")
        if key_padding_mask is not None:
            if tuple(key_padding_mask.shape) != (B, S):
                raise RuntimeError(
                    f"MultiheadAttention: key_padding_mask has shape {list(key_padding_mask.shape)}, expected {[B, S]}"
                )
            km = self._additive_mask(key_padding_mask, q.dtype, "key_padding_mask").reshape((B, 1, 1, S))
            mask = km if mask is None else (mask + km)

        scale = 1.0 / math.sqrt(self.head_dim)
        if not need_weights:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=scale
            )
            attn_weights = None
        else:
            scores = q.matmul(k.transpose(-2, -1)) * scale
            if is_causal:
                from ..._creation import ones
                cm = ones((L, S), dtype=q.dtype, device=q.device).triu(diagonal=1) * (-1e9)
                scores = scores + cm
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
            attn_out = attn_weights.matmul(v)
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)

        out = attn_out.transpose(1, 2).reshape((B, L, E))
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        if not is_batched:
            out = out.squeeze(1 if not self.batch_first else 0)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)
        return out, attn_weights

    @staticmethod
    def _additive_mask(mask: Tensor, dtype, name: str) -> Tensor:
        """A mask as something to add to the attention scores.

        torch.nn convention: in a boolean attn_mask or key_padding_mask, True marks a position
        that must NOT be attended to (the opposite of F.scaled_dot_product_attention). A float
        mask is added as it is.
        """
        if mask.dtype == _dt.bool:
            return zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(mask, -1e9)
        if not mask.is_floating_point():
            raise RuntimeError(f"MultiheadAttention: {name} must be a boolean or floating point tensor, got {mask.dtype}")
        return mask if mask.dtype == dtype else mask.to(dtype)


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, device=device, dtype=dtype)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm_first = norm_first
        self.activation = F.gelu if activation == "gelu" else (activation if callable(activation) else F.relu)

    def _sa_block(self, x: Tensor, attn_mask, key_padding_mask, is_causal: bool = False) -> Tensor:
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None, is_causal: bool = False) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoder(Module):
    def __init__(self, encoder_layer: Module, num_layers: int, norm: Module | None = None) -> None:
        super().__init__()
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src: Tensor, mask=None, src_key_padding_mask=None, is_causal: bool = False) -> Tensor:
        out = src
        for mod in self.layers:
            out = mod(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, device=device, dtype=dtype)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, device=device, dtype=dtype)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.norm_first = norm_first
        self.activation = F.gelu if activation == "gelu" else (activation if callable(activation) else F.relu)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        x = tgt
        if self.norm_first:
            sa, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False, is_causal=tgt_is_causal)
            x = x + self.dropout1(sa)
            ca, _ = self.multihead_attn(self.norm2(x), memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False, is_causal=memory_is_causal)
            x = x + self.dropout2(ca)
            x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x))))))
        else:
            sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False, is_causal=tgt_is_causal)
            x = self.norm1(x + self.dropout1(sa))
            ca, _ = self.multihead_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False, is_causal=memory_is_causal)
            x = self.norm2(x + self.dropout2(ca))
            x = self.norm3(x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x


class TransformerDecoder(Module):
    def __init__(self, decoder_layer: Module, num_layers: int, norm: Module | None = None) -> None:
        super().__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        out = tgt
        for mod in self.layers:
            out = mod(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        if self.norm is not None:
            out = self.norm(out)
        return out


class Transformer(Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype))
        dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers, LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype))
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device=None, dtype=None) -> Tensor:
        from ..._creation import full
        dt = dtype or _dt.get_default_dtype()
        return full((sz, sz), float("-inf"), dtype=dt, device=device).triu(diagonal=1)

    def forward(self, src: Tensor, tgt: Tensor, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, src_is_causal: bool = False, tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        mem = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=src_is_causal)
        return self.decoder(tgt, mem, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
