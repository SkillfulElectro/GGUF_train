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
    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = "mean", padding_idx: int | None = None, device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.mode = mode
        self.padding_idx = padding_idx
        dt = dtype or _dt.get_default_dtype()
        self.weight = Parameter(empty((self.num_embeddings, self.embedding_dim), dtype=dt, device=device))
        init.normal_(self.weight)

    def forward(self, input: Tensor, offsets: Tensor | None = None, per_sample_weights: Tensor | None = None) -> Tensor:
        if input.ndim == 2:
            emb = F.embedding(input, self.weight, self.padding_idx)
            if per_sample_weights is not None:
                emb = emb * per_sample_weights.unsqueeze(-1)
            if self.mode == "sum":
                return emb.sum(dim=1)
            if self.mode == "mean":
                return emb.mean(dim=1)
            return emb.amax(dim=1)
        assert offsets is not None
        offs = offsets.tolist() + [input.shape[0]]
        outs = []
        from ..._ops import stack
        for i in range(len(offs) - 1):
            sl = F.embedding(input[offs[i] : offs[i + 1]], self.weight, self.padding_idx)
            if self.mode == "sum":
                outs.append(sl.sum(dim=0))
            elif self.mode == "mean":
                outs.append(sl.mean(dim=0))
            else:
                outs.append(sl.amax(dim=0))
        return stack(outs, dim=0)
