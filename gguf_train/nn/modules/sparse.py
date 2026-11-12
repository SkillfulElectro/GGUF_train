"""Embedding layers compatible with torch.nn."""

from __future__ import annotations

from ... import _autograd_state, _dtype as _dt
from ..._creation import empty
from ..._tensor import Tensor
from .. import functional as F, init
from ..parameter import Parameter
from .module import Module


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = float(norm_type)
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        dt = dtype or _dt.get_default_dtype()
        if _weight is None:
            self.weight = Parameter(empty((self.num_embeddings, self.embedding_dim), dtype=dt, device=device))
            self.reset_parameters()
        else:
            self.weight = Parameter(_weight)

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.padding_idx is not None:
            with _autograd_state.no_grad():
                self.weight[self.padding_idx].zero_()

    @classmethod
    def from_pretrained(cls, embeddings: Tensor, freeze: bool = True, padding_idx: int | None = None, **kwargs):
        emb = cls(embeddings.shape[0], embeddings.shape[1], padding_idx=padding_idx, _weight=embeddings.clone(), **kwargs)
        emb.weight.requires_grad_(not freeze)
        return emb

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s


class EmbeddingBag(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "mean",
        padding_idx: int | None = None,
        device=None,
        dtype=None,
        *,
        include_last_offset: bool = False,
        _weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("sum", "mean", "max"):
            raise ValueError(f"EmbeddingBag: mode has to be one of 'sum', 'mean' or 'max', got {mode!r}")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.mode = mode
        self.padding_idx = padding_idx
        self.include_last_offset = bool(include_last_offset)
        dt = dtype or _dt.get_default_dtype()
        if _weight is None:
            self.weight = Parameter(empty((self.num_embeddings, self.embedding_dim), dtype=dt, device=device))
            init.normal_(self.weight)
            if self.padding_idx is not None:
                with _autograd_state.no_grad():
                    self.weight[self.padding_idx].zero_()
        else:
            if tuple(_weight.shape) != (self.num_embeddings, self.embedding_dim):
                raise ValueError("EmbeddingBag: the shape of _weight does not match num_embeddings and embedding_dim")
            self.weight = Parameter(_weight)

    @classmethod
    def from_pretrained(cls, embeddings: Tensor, freeze: bool = True, mode: str = "mean", padding_idx: int | None = None,
                        include_last_offset: bool = False):
        bag = cls(embeddings.shape[0], embeddings.shape[1], mode=mode, padding_idx=padding_idx,
                  include_last_offset=include_last_offset, _weight=embeddings.clone())
        bag.weight.requires_grad_(not freeze)
        return bag

    def forward(self, input: Tensor, offsets: Tensor | None = None, per_sample_weights: Tensor | None = None) -> Tensor:
        return F.embedding_bag(
            input, self.weight, offsets, mode=self.mode, per_sample_weights=per_sample_weights,
            include_last_offset=self.include_last_offset, padding_idx=self.padding_idx,
        )

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}, mode={self.mode!r}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s
