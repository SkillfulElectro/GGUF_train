"""Parameter and Buffer subclasses compatible with torch.nn.parameter."""

from __future__ import annotations

from .._tensor import Tensor


class Parameter(Tensor):
    """A Tensor automatically registered as a parameter when assigned to an nn.Module attribute."""

    def __init__(self, data=None, requires_grad: bool = True):
        if data is None:
            from .._creation import empty
            t = empty((0,))
        elif isinstance(data, Tensor):
            t = data
        else:
            from .._tensor import _from_python
            t = _from_python(data)

        super().__init__(t)
        self._is_leaf = True
        if requires_grad and (self.dtype.is_floating_point or self.dtype.is_complex):
            self._requires_grad = True
        else:
            self._requires_grad = False

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        res = Parameter(self.data.clone(), requires_grad=self.requires_grad)
        memo[id(self)] = res
        return res


class Buffer(Tensor):
    """A Tensor registered as a persistent buffer on an nn.Module."""

    def __init__(self, data=None, persistent: bool = True):
        if isinstance(data, Tensor):
            t = data
        else:
            from .._tensor import _from_python
            t = _from_python(data)
        super().__init__(t)
        self.persistent = persistent


class UninitializedParameter(Parameter):
    pass


class UninitializedBuffer(Buffer):
    pass
