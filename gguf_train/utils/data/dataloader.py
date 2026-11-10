"""DataLoader and default_collate compatible with torch.utils.data."""

from __future__ import annotations

from ..._ops import stack
from ..._tensor import Tensor, _from_python
from .dataset import IterableDataset
from .sampler import BatchSampler, RandomSampler, SequentialSampler


def default_collate(batch):
    elem = batch[0]
    if isinstance(elem, Tensor):
        return stack(batch, dim=0)
    if isinstance(elem, (int, float, bool)):
        return _from_python(batch)
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, tuple):
        return tuple(default_collate(samples) for samples in zip(*batch))
    if isinstance(elem, list):
        return [default_collate(samples) for samples in zip(*batch)]
    if isinstance(elem, dict):
        return {k: default_collate([b[k] for b in batch]) for k in elem}
    return batch


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = False,
        sampler=None,
        batch_sampler=None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        generator=None,
    ) -> None:
        (num_workers, pin_memory, timeout, worker_init_fn)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

        if isinstance(dataset, IterableDataset):
            self.batch_sampler = None
        elif batch_sampler is not None:
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
            self.sampler = sampler
            if batch_size is not None:
                self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
            else:
                self.batch_sampler = None

    def __len__(self) -> int:
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        return len(self.dataset)

    def __iter__(self):
        if isinstance(self.dataset, IterableDataset):
            if self.batch_size is None:
                yield from self.dataset
                return
            buf = []
            for item in self.dataset:
                buf.append(item)
                if len(buf) == self.batch_size:
                    yield self.collate_fn(buf)
                    buf = []
            if buf and not self.drop_last:
                yield self.collate_fn(buf)
            return

        if self.batch_sampler is not None:
            for idxs in self.batch_sampler:
                yield self.collate_fn([self.dataset[i] for i in idxs])
        else:
            for idx in self.sampler:
                yield self.dataset[idx]
