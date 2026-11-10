"""Samplers compatible with torch.utils.data."""

from __future__ import annotations

from ..._creation import randperm
from ..._tensor import _from_python


class Sampler:
    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement: bool = False, num_samples: int | None = None, generator=None) -> None:
        (generator,)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        return len(self.data_source) if self._num_samples is None else self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            from ..._creation import randint
            return iter(randint(0, n, (self.num_samples,)).tolist())
        return iter(randperm(n).tolist()[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    def __init__(self, indices, generator=None) -> None:
        (generator,)
        self.indices = list(indices)

    def __iter__(self):
        perm = randperm(len(self.indices)).tolist()
        for i in perm:
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    def __init__(self, weights, num_samples: int, replacement: bool = True, generator=None) -> None:
        (generator,)
        self.weights = weights if hasattr(weights, "multinomial") else _from_python(list(weights))
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)

    def __iter__(self):
        return iter(self.weights.multinomial(self.num_samples, replacement=self.replacement).tolist())

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
