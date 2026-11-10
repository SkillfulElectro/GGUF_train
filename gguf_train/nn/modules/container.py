"""Container modules compatible with torch.nn."""

from __future__ import annotations

from collections import OrderedDict

from ..parameter import Parameter
from .module import Module


class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, mod in args[0].items():
                self.add_module(key, mod)
        else:
            for idx, mod in enumerate(args):
                self.add_module(str(idx), mod)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            mods = list(self._modules.values())[idx]
            return Sequential(*mods)
        if isinstance(idx, str):
            return self._modules[idx]
        keys = list(self._modules.keys())
        return self._modules[keys[idx]]

    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def forward(self, input):
        x = input
        for mod in self._modules.values():
            x = mod(x)
        return x


class ModuleList(Module):
    def __init__(self, modules=None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        keys = list(self._modules.keys())
        return self._modules[keys[idx]]

    def __setitem__(self, idx: int, module: Module) -> None:
        keys = list(self._modules.keys())
        self._modules[keys[idx]] = module

    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        for m in modules:
            self.append(m)
        return self


class ModuleDict(Module):
    def __init__(self, modules=None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules) -> None:
        items = modules.items() if isinstance(modules, dict) else modules
        for k, v in items:
            self.add_module(k, v)


class ParameterList(Module):
    def __init__(self, parameters=None) -> None:
        super().__init__()
        if parameters is not None:
            for p in parameters:
                self.append(p)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __getitem__(self, idx):
        keys = list(self._parameters.keys())
        return self._parameters[keys[idx]]

    def append(self, parameter: Parameter):
        self.register_parameter(str(len(self)), parameter)
        return self


class ParameterDict(Module):
    def __init__(self, parameters=None) -> None:
        super().__init__()
        if parameters is not None:
            for k, v in (parameters.items() if isinstance(parameters, dict) else parameters):
                self.register_parameter(k, v)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __getitem__(self, key: str) -> Parameter:
        return self._parameters[key]

    def __setitem__(self, key: str, param: Parameter) -> None:
        self.register_parameter(key, param)

    def keys(self):
        return self._parameters.keys()

    def values(self):
        return self._parameters.values()

    def items(self):
        return self._parameters.items()
