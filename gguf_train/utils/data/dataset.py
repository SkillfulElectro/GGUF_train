"""Dataset classes compatible with torch.utils.data."""

from __future__ import annotations

from ..._creation import randperm
from ..._tensor import Tensor


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other: Dataset):
        return ConcatDataset([self, other])


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == t.size(0) for t in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].size(0)


class ConcatDataset(Dataset):
    def __init__(self, datasets) -> None:
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        total = 0
        for d in self.datasets:
            total += len(d)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        for d_idx, csz in enumerate(self.cumulative_sizes):
            if idx < csz:
                prev = 0 if d_idx == 0 else self.cumulative_sizes[d_idx - 1]
                return self.datasets[d_idx][idx - prev]
        raise IndexError(idx)


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(dataset: Dataset, lengths, generator=None) -> list[Subset]:
    (generator,)
    total = len(dataset)
    if all(isinstance(l, float) for l in lengths):
        lens = [int(round(total * l)) for l in lengths]
        lens[-1] = total - sum(lens[:-1])
    else:
        lens = [int(l) for l in lengths]
    indices = randperm(total).tolist()
    res = []
    off = 0
    for l in lens:
        res.append(Subset(dataset, indices[off : off + l]))
        off += l
    return res
