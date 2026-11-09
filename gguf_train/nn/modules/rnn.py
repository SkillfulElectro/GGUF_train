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
        pre = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        return pre.relu() if self.nonlinearity == "relu" else pre.tanh()


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
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(h0, self.weight_hh, self.bias_hh)
        i, f, g, o = gates.chunk(4, dim=1)
        i = i.sigmoid()
        f = f.sigmoid()
        g = g.tanh()
        o = o.sigmoid()
        cy = f * c0 + i * g
        hy = o * cy.tanh()
        return hy, cy


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
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = (i_r + h_r).sigmoid()
        z = (i_z + h_z).sigmoid()
        n = (i_n + r * h_n).tanh()
        return n + z * (hx - n)


class _RNNBase(Module):
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
        num_dirs = 2 if bidirectional else 1

        cell_cls = {"RNN": RNNCell, "LSTM": LSTMCell, "GRU": GRUCell}[mode]
        self._cells = []
        for layer in range(self.num_layers):
            in_sz = self.input_size if layer == 0 else self.hidden_size * num_dirs
            for d in range(num_dirs):
                kw = {"bias": bias, "device": device, "dtype": dtype}
                if mode == "RNN":
                    kw["nonlinearity"] = nonlinearity
                c = cell_cls(in_sz, self.hidden_size, **kw)
                suffix = "_reverse" if d == 1 else ""
                name = f"cell_l{layer}{suffix}"
                self.add_module(name, c)
                self._cells.append((layer, d, c))

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
                cell = getattr(self, f"cell_l{layer}{'_reverse' if d == 1 else ''}")
                s_idx = layer * num_dirs + d
                h_t = h0[s_idx]
                c_t = c0[s_idx] if c0 is not None else None
                steps = range(T) if d == 0 else range(T - 1, -1, -1)
                ys = []
                for t in steps:
                    if self.mode == "LSTM":
                        h_t, c_t = cell(cur_seq[t], (h_t, c_t))
                    else:
                        h_t = cell(cur_seq[t], h_t)
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
