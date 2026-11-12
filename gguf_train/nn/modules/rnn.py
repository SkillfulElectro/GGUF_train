"""Recurrent layers (RNN, LSTM, GRU) compatible with torch.nn."""

from __future__ import annotations

import math

from ... import _dtype as _dt
from ..._creation import empty, zeros
from ..._ops import cat, stack
from ..._tensor import Tensor
from .. import functional as F, init
from ..parameter import Parameter
from .module import Module


def _rnn_step(x: Tensor, h: Tensor, w_ih, w_hh, b_ih, b_hh, nonlinearity: str) -> Tensor:
    pre = F.linear(x, w_ih, b_ih) + F.linear(h, w_hh, b_hh)
    return pre.relu() if nonlinearity == "relu" else pre.tanh()


def _lstm_step(x: Tensor, h: Tensor, c: Tensor, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(x, w_ih, b_ih) + F.linear(h, w_hh, b_hh)
    i, f, g, o = gates.chunk(4, dim=1)
    i = i.sigmoid()
    f = f.sigmoid()
    g = g.tanh()
    o = o.sigmoid()
    cy = f * c + i * g
    hy = o * cy.tanh()
    return hy, cy


def _gru_step(x: Tensor, h: Tensor, w_ih, w_hh, b_ih, b_hh) -> Tensor:
    gi = F.linear(x, w_ih, b_ih)
    gh = F.linear(h, w_hh, b_hh)
    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)
    r = (i_r + h_r).sigmoid()
    z = (i_z + h_z).sigmoid()
    n = (i_n + r * h_n).tanh()
    return n + z * (h - n)


class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh", device=None, dtype=None) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.nonlinearity = nonlinearity
        dt = dtype or _dt.get_default_dtype()
        self.weight_ih = Parameter(empty((self.hidden_size, self.input_size), dtype=dt, device=device))
        self.weight_hh = Parameter(empty((self.hidden_size, self.hidden_size), dtype=dt, device=device))
        if bias:
            self.bias_ih = Parameter(empty((self.hidden_size,), dtype=dt, device=device))
            self.bias_hh = Parameter(empty((self.hidden_size,), dtype=dt, device=device))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if hx is None:
            hx = zeros((input.shape[0], self.hidden_size), dtype=input.dtype, device=input.device)
        return _rnn_step(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.nonlinearity)


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        dt = dtype or _dt.get_default_dtype()
        self.weight_ih = Parameter(empty((4 * self.hidden_size, self.input_size), dtype=dt, device=device))
        self.weight_hh = Parameter(empty((4 * self.hidden_size, self.hidden_size), dtype=dt, device=device))
        if bias:
            self.bias_ih = Parameter(empty((4 * self.hidden_size,), dtype=dt, device=device))
            self.bias_hh = Parameter(empty((4 * self.hidden_size,), dtype=dt, device=device))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None):
        if hx is None:
            h0 = zeros((input.shape[0], self.hidden_size), dtype=input.dtype, device=input.device)
            c0 = zeros((input.shape[0], self.hidden_size), dtype=input.dtype, device=input.device)
        else:
            h0, c0 = hx
        return _lstm_step(input, h0, c0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        dt = dtype or _dt.get_default_dtype()
        self.weight_ih = Parameter(empty((3 * self.hidden_size, self.input_size), dtype=dt, device=device))
        self.weight_hh = Parameter(empty((3 * self.hidden_size, self.hidden_size), dtype=dt, device=device))
        if bias:
            self.bias_ih = Parameter(empty((3 * self.hidden_size,), dtype=dt, device=device))
            self.bias_hh = Parameter(empty((3 * self.hidden_size,), dtype=dt, device=device))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if hx is None:
            hx = zeros((input.shape[0], self.hidden_size), dtype=input.dtype, device=input.device)
        return _gru_step(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class _RNNBase(Module):
    """Multi-layer RNN / LSTM / GRU.

    The parameters carry the names torch.nn uses - weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k},
    bias_hh_l{k}, with the suffix _reverse for the second direction - so a state_dict can be
    exchanged with a torch.nn module of the same configuration.
    """

    _GATES = {"RNN": 1, "LSTM": 4, "GRU": 3}

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.nonlinearity = nonlinearity
        num_dirs = 2 if bidirectional else 1
        gates = self._GATES[mode]
        dt = dtype or _dt.get_default_dtype()

        for layer in range(self.num_layers):
            in_sz = self.input_size if layer == 0 else self.hidden_size * num_dirs
            for d in range(num_dirs):
                sfx = self._suffix(layer, d)
                setattr(self, "weight_ih" + sfx, Parameter(empty((gates * self.hidden_size, in_sz), dtype=dt, device=device)))
                setattr(self, "weight_hh" + sfx, Parameter(empty((gates * self.hidden_size, self.hidden_size), dtype=dt, device=device)))
                if self.bias:
                    setattr(self, "bias_ih" + sfx, Parameter(empty((gates * self.hidden_size,), dtype=dt, device=device)))
                    setattr(self, "bias_hh" + sfx, Parameter(empty((gates * self.hidden_size,), dtype=dt, device=device)))
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    @staticmethod
    def _suffix(layer: int, direction: int) -> str:
        return f"_l{layer}" + ("_reverse" if direction == 1 else "")

    def _weights(self, layer: int, direction: int):
        sfx = self._suffix(layer, direction)
        w_ih = getattr(self, "weight_ih" + sfx)
        w_hh = getattr(self, "weight_hh" + sfx)
        if not self.bias:
            return w_ih, w_hh, None, None
        return w_ih, w_hh, getattr(self, "bias_ih" + sfx), getattr(self, "bias_hh" + sfx)

    def _legacy_state_dict_keys(self) -> dict:
        # before the torch names: one cell module per layer and direction, "cell_l0.weight_ih"
        keys = {}
        for layer in range(self.num_layers):
            for d in range(2 if self.bidirectional else 1):
                sfx = self._suffix(layer, d)
                for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    keys[f"cell{sfx}.{kind}"] = kind + sfx
        return keys

    def flatten_parameters(self) -> None:
        """Nothing to do; present because code written for torch.nn calls it."""

    def forward(self, input: Tensor, hx=None):
        x = input.transpose(0, 1) if self.batch_first else input
        T, B, _ = x.shape
        num_dirs = 2 if self.bidirectional else 1
        total_states = self.num_layers * num_dirs

        if hx is None:
            h0 = zeros((total_states, B, self.hidden_size), dtype=x.dtype, device=x.device)
            c0 = zeros((total_states, B, self.hidden_size), dtype=x.dtype, device=x.device) if self.mode == "LSTM" else None
        else:
            if self.mode == "LSTM":
                h0, c0 = hx
            else:
                h0, c0 = hx, None

        final_h = []
        final_c = []
        cur_seq = x
        for layer in range(self.num_layers):
            dir_outs = []
            for d in range(num_dirs):
                w_ih, w_hh, b_ih, b_hh = self._weights(layer, d)
                s_idx = layer * num_dirs + d
                h_t = h0[s_idx]
                c_t = c0[s_idx] if c0 is not None else None
                steps = range(T) if d == 0 else range(T - 1, -1, -1)
                ys = []
                for t in steps:
                    if self.mode == "LSTM":
                        h_t, c_t = _lstm_step(cur_seq[t], h_t, c_t, w_ih, w_hh, b_ih, b_hh)
                    elif self.mode == "GRU":
                        h_t = _gru_step(cur_seq[t], h_t, w_ih, w_hh, b_ih, b_hh)
                    else:
                        h_t = _rnn_step(cur_seq[t], h_t, w_ih, w_hh, b_ih, b_hh, self.nonlinearity)
                    ys.append(h_t)
                if d == 1:
                    ys.reverse()
                dir_outs.append(stack(ys, dim=0))
                final_h.append(h_t)
                if c_t is not None:
                    final_c.append(c_t)
            cur_seq = cat(dir_outs, dim=-1) if num_dirs == 2 else dir_outs[0]
            if self.dropout > 0.0 and layer < self.num_layers - 1:
                cur_seq = F.dropout(cur_seq, p=self.dropout, training=self.training)

        out = cur_seq.transpose(0, 1) if self.batch_first else cur_seq
        hn = stack(final_h, dim=0)
        if self.mode == "LSTM":
            cn = stack(final_c, dim=0)
            return out, (hn, cn)
        return out, hn


class RNN(_RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, nonlinearity: str = "tanh", bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device=None, dtype=None) -> None:
        super().__init__("RNN", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, nonlinearity, device, dtype)


class LSTM(_RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device=None, dtype=None) -> None:
        super().__init__("LSTM", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, "tanh", device, dtype)


class GRU(_RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device=None, dtype=None) -> None:
        super().__init__("GRU", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, "tanh", device, dtype)
