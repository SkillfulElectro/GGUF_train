"""Compatibility torch.random module."""

from __future__ import annotations

import contextlib

from ._creation import get_rng_state, initial_seed, manual_seed, seed, set_rng_state


@contextlib.contextmanager
def fork_rng(devices=None, enabled: bool = True, _caller="fork_rng", _devices_kw="devices"):
    (devices, _caller, _devices_kw)
    if not enabled:
        yield
        return
    st = get_rng_state()
    try:
        yield
    finally:
        set_rng_state(st)


__all__ = ["get_rng_state", "set_rng_state", "manual_seed", "seed", "initial_seed", "fork_rng"]
