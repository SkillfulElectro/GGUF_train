"""Base Module class compatible with torch.nn.Module."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from typing import Iterator

from ... import _autograd_state, _device as _dev, _dtype as _dt
from ..._tensor import Tensor
from ..parameter import Parameter


_IncompatibleKeys = namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class _HookHandle:
    def __init__(self, hooks_dict: OrderedDict, hid: int):
        self._hooks = hooks_dict
        self._id = hid

    def remove(self) -> None:
        self._hooks.pop(self._id, None)


class Module:
    """Base class for all neural network modules."""

    training: bool

    def __init__(self) -> None:
        super().__setattr__("training", True)
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_backward_hooks", OrderedDict())
        super().__setattr__("_next_hook_id", 0)

    def forward(self, *input, **kwargs):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required 'forward' function")

    def __call__(self, *args, **kwargs):
        for hook in list(self._forward_pre_hooks.values()):
            res = hook(self, args)
            if res is not None:
                args = res if isinstance(res, tuple) else (res,)
        result = self.forward(*args, **kwargs)
        for hook in list(self._forward_hooks.values()):
            res = hook(self, args, result)
            if res is not None:
                result = res
        return result

    def register_forward_pre_hook(self, hook):
        hid = self._next_hook_id
        super().__setattr__("_next_hook_id", hid + 1)
        self._forward_pre_hooks[hid] = hook
        return _HookHandle(self._forward_pre_hooks, hid)

    def register_forward_hook(self, hook):
        hid = self._next_hook_id
        super().__setattr__("_next_hook_id", hid + 1)
        self._forward_hooks[hid] = hook
        return _HookHandle(self._forward_hooks, hid)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        if "." in name or name == "":
            raise KeyError("parameter name can't be empty or contain '.'")
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param).__name__}' object to parameter '{name}' (Parameter or None required)")
        else:
            self._parameters[name] = param
        self.__dict__.pop(name, None)

    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> None:
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        if "." in name or name == "":
            raise KeyError("buffer name can't be empty or contain '.'")
        self._buffers[name] = tensor
        self.__dict__.pop(name, None)
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def add_module(self, name: str, module: Module | None) -> None:
        if "_modules" not in self.__dict__:
            raise AttributeError("cannot assign module before Module.__init__() call")
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        self._modules[name] = module
        self.__dict__.pop(name, None)

    def register_module(self, name: str, module: Module | None) -> None:
        self.add_module(name, module)

    def get_submodule(self, target: str) -> Module:
        if target == "":
            return self
        cur = self
        for part in target.split("."):
            cur = cur._modules[part]
        return cur

    def get_parameter(self, target: str) -> Parameter:
        parts = target.rsplit(".", 1)
        mod = self.get_submodule(parts[0]) if len(parts) == 2 else self
        return mod._parameters[parts[-1]]

    def get_buffer(self, target: str) -> Tensor:
        parts = target.rsplit(".", 1)
        mod = self.get_submodule(parts[0]) if len(parts) == 2 else self
        return mod._buffers[parts[-1]]

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            params = self.__dict__["_parameters"]
            if name in params:
                return params[name]
        if "_buffers" in self.__dict__:
            bufs = self.__dict__["_buffers"]
            if name in bufs:
                return bufs[name]
        if "_modules" in self.__dict__:
            mods = self.__dict__["_modules"]
            if name in mods:
                return mods[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            self.register_parameter(name, value)
            return
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value).__name__}' as parameter '{name}' (Parameter or None expected)")
            self.register_parameter(name, value)
            return

        mods = self.__dict__.get("_modules")
        if isinstance(value, Module):
            if mods is None:
                raise AttributeError("cannot assign module before Module.__init__() call")
            self.add_module(name, value)
            return
        elif mods is not None and name in mods:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value).__name__}' as child module '{name}' (Module or None expected)")
            self.add_module(name, value)
            return

        bufs = self.__dict__.get("_buffers")
        if bufs is not None and name in bufs:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"cannot assign '{type(value).__name__}' as buffer '{name}'")
            bufs[name] = value
            return

        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _named_members(self, get_members_fn, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        memo = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for mod_prefix, mod in modules:
            for k, v in get_members_fn(mod):
                if v is None or (remove_duplicate and id(v) in memo):
                    continue
                memo.add(id(v))
                name = f"{mod_prefix}.{k}" if mod_prefix else k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self._named_members(lambda m: m._parameters.items(), prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self._named_members(lambda m: m._buffers.items(), prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def children(self) -> Iterator[Module]:
        for _, m in self.named_children():
            yield m

    def named_children(self):
        memo = set()
        for name, m in self._modules.items():
            if m is not None and id(m) not in memo:
                memo.add(id(m))
                yield name, m

    def modules(self) -> Iterator[Module]:
        for _, m in self.named_modules():
            yield m

    def named_modules(self, memo: set[int] | None = None, prefix: str = "", remove_duplicate: bool = True):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            if remove_duplicate:
                memo.add(id(self))
            yield prefix, self
            for name, m in self._modules.items():
                if m is None:
                    continue
                sub_prefix = f"{prefix}.{name}" if prefix else name
                yield from m.named_modules(memo, sub_prefix, remove_duplicate)

    def train(self, mode: bool = True):
        self.training = bool(mode)
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def _apply(self, fn):
        for m in self.children():
            m._apply(fn)
        with _autograd_state.no_grad():
            for k, p in self._parameters.items():
                if p is not None:
                    new_t = fn(p)
                    if new_t is not p:
                        self._parameters[k] = Parameter(new_t, requires_grad=p.requires_grad)
            for k, b in self._buffers.items():
                if b is not None:
                    self._buffers[k] = fn(b)
        return self

    def to(self, *args, **kwargs):
        return self._apply(lambda t: t.to(*args, **kwargs))

    def cpu(self):
        return self._apply(lambda t: t.cpu())

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device=device))

    def mps(self):
        return self._apply(lambda t: t.mps())

    def float(self):
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self):
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self):
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False) -> OrderedDict:
        if destination is None:
            destination = OrderedDict()
        for k, p in self._parameters.items():
            if p is not None:
                destination[prefix + k] = p if keep_vars else p.detach()
        for k, b in self._buffers.items():
            if b is not None and k not in self._non_persistent_buffers_set:
                destination[prefix + k] = b if keep_vars else b.detach()
        for name, m in self._modules.items():
            if m is not None:
                m.state_dict(destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        (assign,)
        own = self.state_dict(keep_vars=True)
        missing = []
        unexpected = []
        with _autograd_state.no_grad():
            for k, v in state_dict.items():
                if k in own:
                    dst = own[k]
                    if dst.shape != v.shape:
                        raise RuntimeError(f"size mismatch for {k}: copying shape {tuple(v.shape)} vs {tuple(dst.shape)}")
                    dst.copy_(v)
                else:
                    unexpected.append(k)
            for k in own:
                if k not in state_dict:
                    missing.append(k)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Error(s) in loading state_dict: missing={missing}, unexpected={unexpected}")
        return _IncompatibleKeys(missing, unexpected)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra_lines = []
        extra = self.extra_repr()
        if extra:
            extra_lines = extra.split("\n")
        child_lines = []
        for k, m in self._modules.items():
            mod_str = repr(m)
            mod_str = "\n  ".join(mod_str.split("\n"))
            child_lines.append(f"({k}): {mod_str}")
        lines = extra_lines + child_lines
        main_str = type(self).__name__ + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
