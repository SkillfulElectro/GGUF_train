"""Profiler with real record_function timing aggregation compatible with torch.profiler."""

from __future__ import annotations

import enum
import threading
import time

_TLS = threading.local()


def _active_stack() -> list["profile"]:
    st = getattr(_TLS, "stack", None)
    if st is None:
        st = []
        _TLS.stack = st
    return st


class ProfilerActivity(enum.Enum):
    CPU = "CPU"
    CUDA = "CUDA"


class record_function:
    def __init__(self, name: str) -> None:
        self.name = str(name)
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        (args,)
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        for prof in _active_stack():
            prof._record(self.name, dt_ms)
        return False


class _EventAvg:
    def __init__(self, key: str, count: int, cpu_time_total_ms: float) -> None:
        self.key = key
        self.count = count
        self.cpu_time_total = cpu_time_total_ms * 1000.0  # us, like PyTorch
        self.cpu_time = self.cpu_time_total / max(count, 1)


class _EventTable(list):
    def __init__(self, events: list[_EventAvg], elapsed_ms: float) -> None:
        super().__init__(events)
        self.elapsed_ms = elapsed_ms

    def table(self, sort_by: str = "cpu_time_total", row_limit: int = 10) -> str:
        items = sorted(
            self,
            key=lambda e: getattr(e, sort_by, e.cpu_time_total),
            reverse=True,
        )[:row_limit]
        lines = [
            f"{'Name':<28} {'Calls':>8} {'CPU total':>14} {'CPU avg':>14}",
            "-" * 68,
        ]
        for ev in items:
            tot_ms = ev.cpu_time_total / 1000.0
            avg_ms = ev.cpu_time / 1000.0
            lines.append(f"{ev.key:<28} {ev.count:>8d} {tot_ms:>11.3f} ms {avg_ms:>11.3f} ms")
        lines.append("-" * 68)
        lines.append(f"Self CPU time total: {self.elapsed_ms:.3f} ms")
        return "\n".join(lines)


class profile:
    def __init__(self, activities=None, **kwargs) -> None:
        (activities, kwargs)
        self._t0 = 0.0
        self._elapsed_ms = 0.0
        self._counts: dict[str, int] = {}
        self._totals_ms: dict[str, float] = {}

    def _record(self, name: str, dt_ms: float) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        self._totals_ms[name] = self._totals_ms.get(name, 0.0) + dt_ms

    def __enter__(self):
        self._t0 = time.perf_counter()
        _active_stack().append(self)
        return self

    def __exit__(self, *args):
        st = _active_stack()
        if st and st[-1] is self:
            st.pop()
        elif self in st:
            st.remove(self)
        self._elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return False

    def step(self) -> None:
        pass

    def key_averages(self) -> _EventTable:
        events = [
            _EventAvg(name, self._counts[name], self._totals_ms[name])
            for name in self._counts
        ]
        return _EventTable(events, self._elapsed_ms)
