"""L-BFGS optimizer compatible with torch.optim.LBFGS."""

from __future__ import annotations

from collections import deque

from .. import _autograd_state
from .._creation import zeros
from .._ops import cat
from .optimizer import Optimizer


class LBFGS(Optimizer):
    """Limited-memory BFGS optimizer with closure support and two-loop recursion."""

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = None,
    ) -> None:
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")
        self._params = self.param_groups[0]["params"]

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                views.append(zeros((p.numel(),), dtype=p.dtype, device=p.device))
            else:
                views.append(p.grad.detach().reshape((-1,)))
        return cat(views, dim=0)

    def _add_grad(self, step_size: float, update) -> None:
        offset = 0
        with _autograd_state.no_grad():
            for p in self._params:
                n = p.numel()
                p.add_(update.narrow(0, offset, n).reshape(p.shape), alpha=step_size)
                offset += n

    def _clone_params(self):
        return [p.detach().clone() for p in self._params]

    def _set_params(self, saved) -> None:
        with _autograd_state.no_grad():
            for p, s in zip(self._params, saved):
                p.copy_(s)

    @staticmethod
    def _history(state: dict, key: str, history_size: int) -> deque:
        """state[key] as a deque bounded to history_size; a loaded state_dict holds lists."""
        hist = state.get(key)
        if not isinstance(hist, deque) or hist.maxlen != history_size:
            hist = deque(hist or (), maxlen=history_size)
            state[key] = hist
        return hist

    def step(self, closure):
        if closure is None:
            raise RuntimeError("LBFGS requires a closure")

        group = self.param_groups[0]
        lr = float(group["lr"])
        max_iter = int(group["max_iter"])
        max_eval = int(group["max_eval"])
        tol_grad = float(group["tolerance_grad"])
        tol_change = float(group["tolerance_change"])
        history_size = int(group["history_size"])

        # Kept under the first parameter, where Optimizer.state_dict() and load_state_dict()
        # look: the curvature history is what makes this L-BFGS, and a checkpoint has to carry it.
        state = self.state[self._params[0]]
        old_dirs = self._history(state, "old_dirs", history_size)
        old_stps = self._history(state, "old_stps", history_size)
        ro = self._history(state, "ro", history_size)

        orig_loss = self._evaluate(closure)
        loss = float(orig_loss.item())
        g = self._gather_flat_grad()
        g_norm = float(g.abs().max().item()) if g.numel() > 0 else 0.0
        if g_norm <= tol_grad:
            return orig_loss

        n_iter = 0
        func_evals = 1
        H_diag = float(state.get("H_diag", 1.0))
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")
        # the length of the last step taken along d: together they give the displacement the
        # next curvature pair is built from
        t = float(state.get("t", lr))
        d = state.get("d")

        while n_iter < max_iter:
            n_iter += 1
            if n_iter == 1 and d is None:
                d = -g
                H_diag = 1.0
            else:
                if prev_flat_grad is not None and d is not None:
                    y = g - prev_flat_grad
                    s = d * t
                    ys = float((y * s).sum().item())
                    if ys > 1e-10:
                        old_dirs.append(y)
                        old_stps.append(s)
                        ro.append(1.0 / ys)
                        yy = float((y * y).sum().item())
                        H_diag = ys / max(yy, 1e-12)

                # Two-loop recursion
                q = -g
                num_old = len(old_dirs)
                al = [0.0] * num_old
                for i in range(num_old - 1, -1, -1):
                    al[i] = float((old_stps[i] * q).sum().item()) * ro[i]
                    q = q - old_dirs[i] * al[i]
                r = q * H_diag
                for i in range(num_old):
                    be = float((old_dirs[i] * r).sum().item()) * ro[i]
                    r = r + old_stps[i] * (al[i] - be)
                d = r

            prev_flat_grad = g.clone()
            prev_loss = loss
            gtd = float((g * d).sum().item())
            if gtd > -tol_change:
                break

            # Armijo backtracking line search
            t = 1.0 if n_iter > 1 else min(1.0, lr / max(float(g.abs().sum().item()), 1.0))
            if group["line_search_fn"] is None:
                t = lr

            saved = self._clone_params()
            accepted = False
            step_t = t
            for _ in range(10):
                self._set_params(saved)
                self._add_grad(step_t, d)
                if func_evals >= max_eval:
                    break
                cand = self._evaluate(closure)
                func_evals += 1
                cand_loss = float(cand.item())
                if cand_loss <= prev_loss + 1e-4 * step_t * gtd or group["line_search_fn"] is None:
                    loss = cand_loss
                    orig_loss = cand
                    accepted = True
                    break
                step_t *= 0.5
            if not accepted:
                self._set_params(saved)
                break
            t = step_t
            g = self._gather_flat_grad()
            if float(g.abs().max().item()) <= tol_grad:
                break
            if float((d * t).abs().max().item()) <= tol_change:
                break
            if abs(loss - prev_loss) <= tol_change:
                break

        state["d"] = d
        state["t"] = t
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = loss
        state["H_diag"] = H_diag
        return orig_loss
