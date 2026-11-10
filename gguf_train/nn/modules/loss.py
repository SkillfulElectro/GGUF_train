"""Loss modules compatible with torch.nn."""

from __future__ import annotations

from ..._tensor import Tensor
from .. import functional as F
from .module import Module


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        (size_average, reduce)
        self.reduction = reduction


class MSELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


class L1Loss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0) -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = float(beta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(_Loss):
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        self.delta = float(delta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class CrossEntropyLoss(_Loss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index,
            reduction=self.reduction, label_smoothing=self.label_smoothing,
        )


class NLLLoss(_Loss):
    def __init__(self, weight: Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.ignore_index = int(ignore_index)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class BCELoss(_Loss):
    def __init__(self, weight: Tensor | None = None, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight: Tensor | None = None, size_average=None, reduce=None, reduction: str = "mean", pos_weight: Tensor | None = None) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            input, target, weight=self.weight, reduction=self.reduction, pos_weight=self.pos_weight
        )


class KLDivLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__(size_average, reduce, reduction)
        self.log_target = bool(log_target)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class CosineEmbeddingLoss(_Loss):
    def __init__(self, margin: float = 0.0, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = float(margin)

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)


class MarginRankingLoss(_Loss):
    def __init__(self, margin: float = 0.0, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = float(margin)

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.margin_ranking_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)


class TripletMarginLoss(_Loss):
    def __init__(self, margin: float = 1.0, p: float = 2.0, eps: float = 1e-6, swap: bool = False, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = float(margin)
        self.p = float(p)
        self.eps = float(eps)
        self.swap = bool(swap)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p, eps=self.eps, swap=self.swap, reduction=self.reduction)
