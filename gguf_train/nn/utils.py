"""Neural network utilities compatible with torch.nn.utils."""

from __future__ import annotations

import math

from .. import _autograd_state
from .._ops import cat
from .._tensor import Tensor


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if not grads:
        from .._tensor import _from_python
        return _from_python(0.0)
    with _autograd_state.no_grad():
        if math.isinf(norm_type):
            norms = [g.detach().abs().max() for g in grads]
            total_norm = norms[0]
            for n in norms[1:]:
                total_norm = total_norm.maximum(n)
        else:
            norms = [g.detach().norm(p=norm_type) for g in grads]
            total_norm = norms[0].new_zeros(())
            for n in norms:
                total_norm = total_norm + n ** norm_type
            total_norm = total_norm ** (1.0 / norm_type)

        if error_if_nonfinite and not bool(total_norm.isfinite().item()):
            raise RuntimeError(f"The total norm of order {norm_type} for gradients from `parameters` is non-finite")
        clip_coef = max_norm / (float(total_norm.item()) + 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
    return total_norm


def clip_grad_value_(parameters, clip_value: float) -> None:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    cv = float(clip_value)
    with _autograd_state.no_grad():
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(-cv, cv)


def parameters_to_vector(parameters) -> Tensor:
    vecs = [p.reshape((-1,)) for p in parameters]
    return cat(vecs, dim=0)


def vector_to_parameters(vec: Tensor, parameters) -> None:
    pointer = 0
    with _autograd_state.no_grad():
        for p in parameters:
            n = p.numel()
            p.copy_(vec[pointer : pointer + n].reshape(p.shape))
            pointer += n


class rnn:
    @staticmethod
    def pack_padded_sequence(input: Tensor, lengths, batch_first: bool = False, enforce_sorted: bool = True):
        (lengths, batch_first, enforce_sorted)
        return input

    @staticmethod
    def pad_packed_sequence(sequence: Tensor, batch_first: bool = False, padding_value: float = 0.0, total_length=None):
        (batch_first, padding_value, total_length)
        return sequence, None

    @staticmethod
    def pad_sequence(sequences, batch_first: bool = False, padding_value: float = 0.0) -> Tensor:
        seqs = list(sequences)
        max_len = max(s.shape[0] for s in seqs)
        trailing = seqs[0].shape[1:]
        out_sh = (len(seqs), max_len, *trailing) if batch_first else (max_len, len(seqs), *trailing)
        from .._creation import full
        out = full(out_sh, padding_value, dtype=seqs[0].dtype, device=seqs[0].device)
        for i, s in enumerate(seqs):
            l = s.shape[0]
            if batch_first:
                out[i, :l] = s
            else:
                out[:l, i] = s
        return out
