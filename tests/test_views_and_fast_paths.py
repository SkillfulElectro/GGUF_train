"""Views and the ggml fast paths (BUG-08).

A view shares storage with its base and is described by an offset and strides. Two things
went wrong around that:

* code that took the storage pointer and indexed it from zero, ignoring the offset and the
  strides of the view (conv bias, batch-norm running statistics, optimizer state), and
* fast paths that handed a tensor to ggml and then mis-delivered or dropped the result
  (one-sided clamp, float16 embeddings, a bias the conv graph could not use in place).

The rule the tests pin down: an operation gives the same values and the same gradients
whether an argument is fresh storage, a slice in the middle of a larger tensor or a strided
view - and it never touches the part of the base that lies outside the view.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import _C

REPO = Path(__file__).resolve().parents[1]
JUNK = 99.0


def fresh(t):
    """The same values in new storage that starts at offset 0."""
    return torch.tensor(t.tolist(), dtype=t.dtype)


def as_offset(t):
    """A contiguous view that starts in the middle of a larger storage."""
    pad_shape = (3,) + tuple(t.shape[1:])
    if t.dtype.is_floating_point:
        pad = torch.full(pad_shape, JUNK).to(t.dtype)
    else:
        pad = torch.zeros(pad_shape, dtype=t.dtype)
    view = torch.cat([pad, t], dim=0)[3:]
    assert view.is_contiguous()
    return view


def as_strided(t):
    """A non-contiguous view."""
    if t.ndim == 1:
        if t.dtype.is_floating_point:
            junk = torch.full(tuple(t.shape), JUNK).to(t.dtype)
        else:
            junk = torch.zeros(tuple(t.shape), dtype=t.dtype)
        view = torch.stack([t, junk], dim=1)[:, 0]
    else:
        last = t.ndim - 1
        view = t.transpose(0, last).contiguous().transpose(0, last)
    assert not view.is_contiguous() or t.numel() <= 1
    return view


KINDS = {"offset": as_offset, "strided": as_strided}


def flat(t):
    return t.reshape(-1).tolist()


class ViewTestCase(unittest.TestCase):
    def assertClose(self, got, want, tol=2e-4, msg=""):
        self.assertEqual(tuple(got.shape), tuple(want.shape), msg)
        for i, (g, w) in enumerate(zip(flat(got), flat(want))):
            if isinstance(w, float) and math.isnan(w):
                self.assertTrue(math.isnan(g), f"{msg}: element {i} is {g}, expected nan")
                continue
            self.assertFalse(isinstance(g, float) and math.isnan(g), f"{msg}: element {i} is nan, expected {w}")
            self.assertLessEqual(abs(g - w), tol * max(1.0, abs(w)), f"{msg}: element {i} is {g}, expected {w}")

    def _run(self, fn, tensors, which=None, kind=None, grad="plain"):
        leaves = [fresh(t) for t in tensors]
        for leaf in leaves:
            if leaf.dtype.is_floating_point:
                leaf.requires_grad_(True)
        args = [
            KINDS[kind](leaf) if (kind is not None and (which is None or i == which)) else leaf
            for i, leaf in enumerate(leaves)
        ]
        out = fn(*args)
        grads = None
        if out.dtype.is_floating_point and out.requires_grad:
            if out.ndim == 0:
                loss = out
            else:
                weights = torch.arange(out.numel(), dtype=torch.float32).reshape(*out.shape) * 0.01 + 0.5
                if grad == "plain":
                    loss = (out * weights).sum()
                elif grad == "offset":
                    # downstream cat: the gradient that arrives is a slice of a larger one
                    loss = (torch.cat([out * 0.0, out], dim=0) * torch.cat([weights, weights], dim=0)).sum()
                else:
                    # downstream transpose: the gradient that arrives is strided
                    last = out.ndim - 1
                    loss = (out.transpose(0, last) * weights.transpose(0, last)).sum()
            loss.backward()
            grads = [leaf.grad for leaf in leaves]
        return out.detach(), grads

    def check_views(self, fn, *tensors):
        """fn gives the same values and gradients for views as for fresh storage."""
        want, want_grads = self._run(fn, tensors)
        variants = [(None, kind, "plain") for kind in KINDS]
        variants += [(i, kind, "plain") for i in range(len(tensors)) for kind in KINDS]
        variants += [(None, None, "offset"), (None, None, "strided")]
        for which, kind, grad in variants:
            label = f"{'every argument' if which is None else 'argument %d' % which} as {kind or 'fresh'}, {grad} gradient"
            with self.subTest(label):
                got, got_grads = self._run(fn, tensors, which, kind, grad)
                self.assertClose(got, want, msg=f"forward ({label})")
                if want_grads is None:
                    continue
                for i, (g, w) in enumerate(zip(got_grads, want_grads)):
                    self.assertEqual(g is None, w is None, f"gradient of argument {i} ({label})")
                    if w is not None:
                        self.assertClose(g, w, msg=f"gradient of argument {i} ({label})")


class TestElementwiseAndReductionsOnViews(ViewTestCase):
    """The paths the audit names: gt_try_ggml_unary, gt_try_ggml_binary, softmax, sum/mean."""

    # 4x6 stays below the size where the fast paths start to reshape; 64x128 is above it
    SIZES = ((4, 6), (64, 128))

    def _views(self, rows, cols):
        torch.manual_seed(rows)
        big = torch.randn(rows + 3, cols + 2)
        return {
            "row slice": big[2:2 + rows],
            "inner block": big[1:1 + rows, 1:1 + cols],
            "transposed": torch.randn(cols, rows).t(),
        }

    def test_unary_ops(self):
        ops = ["neg", "abs", "sign", "sqrt", "square", "exp", "expm1", "log", "sin", "cos", "tanh",
               "floor", "ceil", "round", "trunc", "relu", "sigmoid", "erf", "log1p", "reciprocal", "rsqrt"]
        activations = ["silu", "gelu", "softplus", "hardswish", "hardsigmoid", "elu", "leaky_relu", "relu6", "mish"]
        for rows, cols in self.SIZES:
            for name, view in self._views(rows, cols).items():
                snapshot = view.tolist()
                positive = view.abs() + 0.5
                for op in ops:
                    source = positive if op in ("sqrt", "log", "log1p", "rsqrt", "reciprocal") else view
                    with self.subTest(op=op, view=name, size=(rows, cols)):
                        self.assertClose(getattr(torch, op)(source), getattr(torch, op)(fresh(source)))
                for op in activations:
                    with self.subTest(op=op, view=name, size=(rows, cols)):
                        self.assertClose(getattr(F, op)(view), getattr(F, op)(fresh(view)))
                self.assertEqual(view.tolist(), snapshot, "a unary op modified its input")

    def test_binary_ops(self):
        for rows, cols in self.SIZES:
            for name, view in self._views(rows, cols).items():
                torch.manual_seed(7)
                other = torch.randn(*view.shape)
                snapshot = view.tolist()
                for op in ("add", "sub", "mul", "div", "maximum", "minimum"):
                    fn = getattr(torch, op)
                    rhs = other.abs() + 0.5 if op == "div" else other
                    with self.subTest(op=op, view=name, size=(rows, cols)):
                        self.assertClose(fn(view, rhs), fn(fresh(view), rhs))
                        self.assertClose(fn(rhs, view.abs() + 0.5), fn(rhs, fresh(view.abs() + 0.5)))
                        self.assertClose(fn(view, view.abs() + 0.5), fn(fresh(view), fresh(view.abs() + 0.5)))
                with self.subTest(op="scalar", view=name, size=(rows, cols)):
                    self.assertClose(view + 1.5, fresh(view) + 1.5)
                    self.assertClose(view * 1.5, fresh(view) * 1.5)
                with self.subTest(op="broadcast", view=name, size=(rows, cols)):
                    row = view[0]
                    self.assertClose(other + row, other + fresh(row))
                    self.assertClose(other * row, other * fresh(row))
                self.assertEqual(view.tolist(), snapshot, "a binary op modified its input")

    def test_reductions_and_softmax(self):
        for rows, cols in self.SIZES:
            for name, view in self._views(rows, cols).items():
                with self.subTest(view=name, size=(rows, cols)):
                    plain = fresh(view)
                    self.assertClose(view.sum(), plain.sum())
                    self.assertClose(view.mean(), plain.mean())
                    self.assertClose(view.sum(dim=1), plain.sum(dim=1))
                    self.assertClose(view.sum(dim=0), plain.sum(dim=0))
                    self.assertClose(view.mean(dim=1), plain.mean(dim=1))
                    self.assertEqual(view.argmax(dim=1).tolist(), plain.argmax(dim=1).tolist())
                    self.assertClose(view.max(), plain.max())
                    self.assertClose(view.std(), plain.std())
                    self.assertClose(torch.softmax(view, dim=-1), torch.softmax(plain, dim=-1))
                    self.assertClose(torch.softmax(view, dim=0), torch.softmax(plain, dim=0))
                    self.assertClose(torch.log_softmax(view, dim=-1), torch.log_softmax(plain, dim=-1))
                    right, left = torch.ones(view.shape[1], 3), torch.ones(3, view.shape[0])
                    self.assertClose(view @ right, plain @ right)
                    self.assertClose(left @ view, left @ plain)

    def test_in_place_ops_write_through_the_view_only(self):
        base = torch.zeros(4, 3)
        base[2:].add_(torch.ones(2, 3))
        self.assertEqual(base.tolist(), [[0.0] * 3, [0.0] * 3, [1.0] * 3, [1.0] * 3])

        base = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        base[1:3].mul_(2.0)
        self.assertEqual(base.tolist(), [[0, 1, 2], [6, 8, 10], [12, 14, 16], [9, 10, 11]])

        base = torch.arange(12, dtype=torch.float32).reshape(4, 3) - 6
        base[2:].relu_()
        self.assertEqual(base.tolist(), [[-6, -5, -4], [-3, -2, -1], [0, 1, 2], [3, 4, 5]])

        base = torch.arange(12, dtype=torch.float32).reshape(4, 3) - 6
        base[:, 1].clamp_(-1.0, 1.0)
        self.assertEqual(base.tolist(), [[-6, -1, -4], [-3, -1, -1], [0, 1, 2], [3, 1, 5]])

        base = torch.arange(12, dtype=torch.float32).reshape(4, 3) - 6
        base[:, 2].clamp_(min=0.0)
        self.assertEqual(base.tolist(), [[-6, -5, 0], [-3, -2, 0], [0, 1, 2], [3, 4, 5]])


class TestLayersOnViews(ViewTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_linear(self):
        self.check_views(lambda x, w, b: F.linear(x, w, b), torch.randn(5, 8), torch.randn(4, 8), torch.randn(4))

    def test_linear_on_chunks_of_a_packed_weight(self):
        # nn.MultiheadAttention keeps q, k and v projections in one tensor and chunks it
        packed_w, packed_b = torch.randn(12, 4), torch.randn(12)
        x = torch.randn(3, 4)
        for w, b in zip(packed_w.chunk(3), packed_b.chunk(3)):
            self.assertClose(F.linear(x, w, b), F.linear(x, fresh(w), fresh(b)))

    def test_conv2d(self):
        # the bias used to be left out when it was a view
        args = (torch.randn(2, 4, 6, 6), torch.randn(6, 4, 3, 3), torch.randn(6))
        self.check_views(lambda i, w, b: F.conv2d(i, w, b, padding=1), *args)
        self.check_views(lambda i, w, b: F.conv2d(i, w, b, stride=2), *args)

    def test_conv2d_grouped(self):
        self.check_views(lambda i, w, b: F.conv2d(i, w, b, padding=1, groups=2),
                         torch.randn(2, 4, 6, 6), torch.randn(6, 2, 3, 3), torch.randn(6))

    def test_conv1d(self):
        self.check_views(lambda i, w, b: F.conv1d(i, w, b, padding=1),
                         torch.randn(2, 4, 10), torch.randn(5, 4, 3), torch.randn(5))

    def test_conv2d_adds_a_bias_of_another_dtype(self):
        image, weight, bias = torch.randn(2, 4, 6, 6), torch.randn(6, 4, 3, 3), torch.randn(6)
        want = F.conv2d(image, weight) + bias.reshape(1, 6, 1, 1)
        self.assertClose(F.conv2d(image, weight, bias), want)
        self.assertClose(F.conv2d(image, weight, bias.to(torch.float64)), want)

    def test_conv2d_bias_in_the_middle_of_a_parameter_vector(self):
        image, weight = torch.randn(1, 2, 5, 5), torch.randn(3, 2, 3, 3)
        packed = torch.tensor([100.0, 200.0, 0.5, -1.5, 2.5, 300.0])
        bias = packed[2:5]
        got = F.conv2d(image, weight, bias)
        self.assertClose(got, F.conv2d(image, weight) + torch.tensor([0.5, -1.5, 2.5]).reshape(1, 3, 1, 1))
        self.assertEqual(packed.tolist(), [100.0, 200.0, 0.5, -1.5, 2.5, 300.0])

    def test_normalisation_layers(self):
        x = torch.randn(5, 8)
        self.check_views(lambda x, w, b: F.layer_norm(x, (8,), w, b), x, torch.randn(8), torch.randn(8))
        self.check_views(lambda x, w, b: F.layer_norm(x, (4, 6), w, b),
                         torch.randn(3, 4, 6), torch.randn(4, 6), torch.randn(4, 6))
        self.check_views(lambda x, w: F.rms_norm(x, (8,), w), x, torch.randn(8))
        self.check_views(lambda x, w, b: F.group_norm(x, 2, w, b), torch.randn(2, 4, 6, 6), torch.randn(4), torch.randn(4))

    def test_batch_norm_reads_running_statistics_through_the_view(self):
        image = torch.randn(2, 4, 6, 6)
        self.check_views(lambda x, m, v, w, b: F.batch_norm(x, m, v, w, b, training=False),
                         image, torch.randn(4), torch.rand(4) + 0.5, torch.randn(4), torch.randn(4))
        self.check_views(lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True),
                         image, torch.randn(4), torch.randn(4))

    def test_batch_norm_updates_running_statistics_in_the_view_only(self):
        image = torch.randn(8, 3, 4, 4)
        plain_mean, plain_var = torch.zeros(3), torch.ones(3)
        F.batch_norm(image, plain_mean, plain_var, training=True, momentum=0.5)

        # contiguous slices in the middle of one buffer: [junk | mean | junk | var | junk]
        packed = torch.full((12,), JUNK)
        mean, var = packed[2:5], packed[7:10]
        mean.fill_(0.0)
        var.fill_(1.0)
        F.batch_norm(image, mean, var, training=True, momentum=0.5)
        self.assertClose(mean, plain_mean)
        self.assertClose(var, plain_var)
        self.assertEqual([packed[i].item() for i in (0, 1, 5, 6, 10, 11)], [JUNK] * 6)

        # strided: every other element of a buffer
        grid = torch.full((3, 2), JUNK)
        mean = grid[:, 0]
        mean.fill_(0.0)
        var = torch.ones(3)
        F.batch_norm(image, mean, var, training=True, momentum=0.5)
        self.assertClose(mean, plain_mean)
        self.assertEqual(grid[:, 1].tolist(), [JUNK] * 3)

    def test_batch_norm_rejects_statistics_of_the_wrong_length(self):
        with self.assertRaises(RuntimeError) as ctx:
            F.batch_norm(torch.randn(2, 4, 3, 3), torch.zeros(3), torch.ones(3), training=False)
        self.assertIn("one element per channel", str(ctx.exception))

    def test_embedding(self):
        indices = torch.tensor([[1, 3, 9], [0, 2, 5], [4, 4, 1]])
        self.check_views(lambda e, i: F.embedding(i, e), torch.randn(10, 6), indices)
        self.check_views(lambda e, i: F.embedding(i, e, padding_idx=0), torch.randn(10, 6), indices)

    def test_embedding_with_half_precision_rows(self):
        # ggml_get_rows returns float32; the float16 output used to be left uninitialised
        table = torch.randn(10, 6)
        indices = torch.tensor([[1, 3, 9], [0, 2, 5]])
        want = F.embedding(indices, table)
        for dtype, tol in ((torch.float16, 2e-3), (torch.float64, 1e-6)):
            got = F.embedding(indices, table.to(dtype))
            self.assertEqual(got.dtype, dtype)
            self.assertClose(got.to(torch.float32), want, tol=tol, msg=str(dtype))

    def test_losses(self):
        x, target = torch.randn(5, 8), torch.tensor([1, 0, 3, 2, 7])
        self.check_views(lambda x, t: F.cross_entropy(x, t), x, target)
        self.check_views(lambda x, t, w: F.cross_entropy(x, t, weight=w), x, target, torch.rand(8) + 0.5)
        self.check_views(lambda x, t: F.nll_loss(torch.log_softmax(x, dim=1), t), x, target)
        self.check_views(lambda a, b: F.mse_loss(a, b), x, torch.randn(5, 8))

    def test_attention_and_matmul(self):
        self.check_views(lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
                         torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4))
        self.check_views(lambda a, b: a @ b, torch.randn(3, 4, 5), torch.randn(3, 5, 2))
        self.check_views(lambda a, b: a @ b, torch.randn(4, 5), torch.randn(5))

    def test_indexing_and_shuffling(self):
        x = torch.randn(5, 8)
        index = torch.tensor([[0, 7], [1, 1], [2, 3], [4, 5], [6, 0]])
        self.check_views(lambda a, i: torch.gather(a, 1, i), x, index)
        self.check_views(lambda a, i: a[i], x, torch.tensor([0, 4, 2]))
        self.check_views(lambda c, a, b: torch.where(c > 0, a, b), torch.randn(5, 8), x, torch.randn(5, 8))
        self.check_views(lambda a: a.cumsum(1), x)
        self.check_views(lambda a: torch.sort(a, dim=1)[0], x)
        self.check_views(lambda a, b: torch.cat([a, b], dim=1), x, torch.randn(5, 3))


class TestOptimizerStepsOnViews(ViewTestCase):
    def test_parameter_that_is_a_slice(self):
        for name in ("SGD", "Adam", "AdamW"):
            with self.subTest(optimizer=name):
                store = torch.zeros(3, 4)
                param = store[1:2]
                param.requires_grad_(True)
                reference = torch.zeros(1, 4, requires_grad=True)
                for p in (param, reference):
                    opt = getattr(torch.optim, name)([p], lr=0.1)
                    ((p - 1.0) ** 2).sum().backward()
                    opt.step()
                self.assertClose(store[1:2].detach(), reference.detach())
                self.assertGreater(abs(store[1, 0].item()), 1e-3)
                self.assertEqual(store[0].tolist(), [0.0] * 4)
                self.assertEqual(store[2].tolist(), [0.0] * 4)

    def _adamw(self, param, grad, m, v, vmax=None):
        rc = _C.lib.gt_adamw_step(param._ptr, grad._ptr, m._ptr, v._ptr, vmax._ptr if vmax is not None else None,
                                  0.1, 0.9, 0.999, 1e-8, 0.01, 1, 1 if vmax is not None else 0, 1)
        self.assertEqual(rc, 0)

    def test_adam_state_buffers_that_are_slices(self):
        grad = torch.tensor([1.0, -2.0, 3.0])
        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                param = torch.tensor([0.5, 0.5, 0.5])
                m, v = torch.zeros(3), torch.zeros(3)
                vmax = torch.zeros(3) if amsgrad else None
                self._adamw(param, grad, m, v, vmax)

                # the same step with every state buffer in the middle of a larger tensor
                packed = torch.full((4, 5), JUNK)
                param2 = torch.tensor([0.5, 0.5, 0.5])
                m2, v2, vmax2 = packed[0, 1:4], packed[1, 2:5], (packed[2, 1:4] if amsgrad else None)
                for state in (m2, v2) + ((vmax2,) if amsgrad else ()):
                    state.fill_(0.0)
                # a float64 gradient keeps the step on the general path, like amsgrad does
                self._adamw(param2, grad.to(torch.float64), m2, v2, vmax2)

                self.assertClose(param2, param)
                self.assertClose(m2, m)
                self.assertClose(v2, v)
                if amsgrad:
                    self.assertClose(vmax2, vmax)
                untouched = [packed[0, 0], packed[0, 4], packed[1, 0], packed[1, 1], packed[3, 0], packed[3, 4]]
                self.assertEqual([u.item() for u in untouched], [JUNK] * 6)

    def test_sgd_momentum_buffer_that_is_strided(self):
        grad = torch.tensor([1.0, -2.0, 3.0])
        param, buf = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.1, 0.2, 0.3])
        self.assertEqual(_C.lib.gt_sgd_step(param._ptr, grad._ptr, buf._ptr, 0.1, 0.9, 0.0, 0.0, 0, 0), 0)

        grid = torch.full((3, 2), JUNK)
        buf2 = grid[:, 0]
        buf2.copy_(torch.tensor([0.1, 0.2, 0.3]))
        param2 = torch.tensor([0.5, 0.5, 0.5])
        self.assertEqual(_C.lib.gt_sgd_step(param2._ptr, grad._ptr, buf2._ptr, 0.1, 0.9, 0.0, 0.0, 0, 0), 0)
        self.assertClose(param2, param)
        self.assertClose(buf2, buf)
        self.assertEqual(grid[:, 1].tolist(), [JUNK] * 3)


class TestClamp(ViewTestCase):
    """clamp went through ggml_clamp with NaN standing in for a missing bound."""

    VALUES = [-1.0, -0.2, 0.0, 0.5, 2.0]

    def test_lower_bound_only(self):
        x = torch.tensor(self.VALUES)
        self.assertClose(x.clamp(min=-0.3), torch.tensor([-0.3, -0.2, 0.0, 0.5, 2.0]))
        self.assertClose(torch.clamp(x, min=0.0), torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]))
        self.assertEqual(x.tolist(), fresh(torch.tensor(self.VALUES)).tolist())

    def test_upper_bound_only(self):
        x = torch.tensor(self.VALUES)
        self.assertClose(x.clamp(max=0.4), torch.tensor([-1.0, -0.2, 0.0, 0.4, 0.4]))

    def test_both_bounds(self):
        x = torch.tensor(self.VALUES)
        self.assertClose(x.clamp(-0.3, 0.4), torch.tensor([-0.3, -0.2, 0.0, 0.4, 0.4]))
        self.assertClose(x.clip(min=-0.3, max=0.4), torch.tensor([-0.3, -0.2, 0.0, 0.4, 0.4]))

    def test_nan_elements_stay_nan(self):
        x = torch.tensor([float("nan"), -5.0, 5.0])
        for kwargs in ({"min": 0.0}, {"max": 1.0}, {"min": 0.0, "max": 1.0}):
            with self.subTest(**kwargs):
                got = x.clamp(**kwargs).tolist()
                self.assertTrue(math.isnan(got[0]), got)
                self.assertEqual(got[1], kwargs.get("min", -5.0))
                self.assertEqual(got[2], kwargs.get("max", 5.0))

    def test_large_tensors_and_other_dtypes(self):
        torch.manual_seed(0)
        big = torch.randn(64, 128)
        low = big.clamp(min=0.25)
        self.assertAlmostEqual(low.min().item(), 0.25, places=6)
        self.assertClose(low.max(), big.max())
        high = big.clamp(max=-0.25)
        self.assertAlmostEqual(high.max().item(), -0.25, places=6)
        self.assertClose(high.min(), big.min())

        self.assertEqual(torch.tensor([-3, 0, 5]).clamp(min=-1).tolist(), [-1, 0, 5])
        self.assertEqual(torch.tensor([-3, 0, 5]).clamp(max=1).tolist(), [-3, 0, 1])
        x64 = torch.tensor(self.VALUES, dtype=torch.float64)
        self.assertEqual(x64.clamp(min=0.0).tolist(), [0.0, 0.0, 0.0, 0.5, 2.0])

    def test_in_place_and_gradient(self):
        x = torch.tensor(self.VALUES)
        x.clamp_(min=0.0)
        self.assertEqual(x.tolist(), [0.0, 0.0, 0.0, 0.5, 2.0])
        x.clamp_(max=1.0)
        self.assertEqual(x.tolist(), [0.0, 0.0, 0.0, 0.5, 1.0])

        g = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        y = g.clamp(min=0.0)
        y.sum().backward()
        self.assertEqual(y.tolist(), [1.0, 0.0, 3.0])
        self.assertEqual(g.grad.tolist(), [1.0, 0.0, 1.0])

    def test_functions_built_on_a_one_sided_clamp(self):
        x = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        self.assertClose(F.normalize(x, dim=1), torch.tensor([[0.6, 0.8], [0.6, 0.8]]))
        self.assertClose(F.cosine_similarity(x, x, dim=1), torch.tensor([1.0, 1.0]))
        self.assertClose(F.softshrink(torch.tensor([-2.0, -0.2, 0.3, 1.5]), 0.5), torch.tensor([-1.5, 0.0, 0.0, 1.0]))
        self.assertClose(
            F.margin_ranking_loss(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0]), margin=0.5),
            torch.tensor(0.75),
        )

        log_probs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]]), dim=1)
        target = torch.tensor([2, 1, 0, 1])
        mean_nll = -sum(log_probs[i, int(target[i].item())].item() for i in range(4)) / 4
        self.assertAlmostEqual(F.nll_loss(log_probs, target).item(), mean_nll, places=5)

        kl = F.kl_div(torch.tensor([-1.0, -2.0]), torch.tensor([0.5, 0.5]), reduction="sum").item()
        self.assertAlmostEqual(kl, sum(0.5 * (math.log(0.5) - lp) for lp in (-1.0, -2.0)), places=5)

        logits, labels = torch.tensor([0.5, -1.0, 2.0]), torch.tensor([1.0, 0.0, 1.0])
        want = sum(max(l, 0.0) - l * t + math.log1p(math.exp(-abs(l))) for l, t in zip(logits.tolist(), labels.tolist())) / 3
        self.assertAlmostEqual(F.binary_cross_entropy_with_logits(logits, labels).item(), want, places=5)


class TestRound(ViewTestCase):
    """torch.round rounds halves to even. The ggml kernel rounds them away from zero, so a
    contiguous float32 tensor used to give [1, 2, 3] for [0.5, 1.5, 2.5] while its float64
    copy, or a strided view of it, gave [0, 2, 2]."""

    VALUES = [-3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 2.4, 2.6, -2.4, -2.6, 1e6 + 0.5]
    WANT = [-4.0, -2.0, -2.0, -0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 2.0, 3.0, -2.0, -3.0, 1e6]

    def test_halves_go_to_the_even_neighbour_on_every_path(self):
        x = torch.tensor(self.VALUES)
        layouts = {"contiguous": x, "offset": as_offset(x), "strided": as_strided(x), "float64": x.double(), "float16": x[:13].half()}
        for name, t in layouts.items():
            with self.subTest(name):
                self.assertEqual(t.round().tolist(), self.WANT[: t.numel()])
                self.assertEqual(torch.round(t).dtype, t.dtype)

    def test_large_contiguous_tensor(self):
        # above the size where the elementwise loop runs in parallel
        x = (torch.arange(20000, dtype=torch.float32) - 10000) / 2      # every half from -5000 to 5000
        got = x.round()
        want = [float(round(v)) for v in x.tolist()]                    # Python rounds halves to even too
        self.assertEqual(got.tolist(), want)

    def test_integers_are_left_alone(self):
        t = torch.tensor([-3, 0, 7])
        self.assertEqual((t.round().tolist(), t.round().dtype), ([-3, 0, 7], torch.int64))


class TestAtan2OfIntegers(unittest.TestCase):
    def test_the_result_is_floating_point(self):
        y, x = torch.tensor([1, 2, -3]), torch.tensor([1, -1, 2])
        got = torch.atan2(y, x)
        self.assertEqual(got.dtype, torch.float32)
        want = [math.atan2(a, b) for a, b in zip(y.tolist(), x.tolist())]
        for g, w in zip(got.tolist(), want):
            self.assertAlmostEqual(g, w, places=6)
        mixed = torch.atan2(y, x.double())
        self.assertEqual(mixed.dtype, torch.float64)
        self.assertAlmostEqual(mixed[1].item(), math.atan2(2, -1), places=12)


def _compiler():
    for name in ("g++", "c++", "clang++"):
        path = shutil.which(name)
        if path:
            return path
    return None


class TestGraphResultDelivery(unittest.TestCase):
    """gt_exec_ggml_graph with outputs and result nodes the Python API never produces."""

    def test_native_harness(self):
        llama = REPO / "third_party" / "llama.cpp"
        libs = Path(_C.__file__).resolve().parent / "_libs"
        compiler = _compiler()
        if sys.platform == "win32" or compiler is None:
            self.skipTest("needs a C++ compiler driver (g++/c++/clang++)")
        if not (llama / "ggml" / "include" / "ggml.h").exists() or not (REPO / "csrc" / "gt_internal.h").exists():
            self.skipTest("needs the source checkout with third_party/llama.cpp")
        if not any(libs.glob("*gguf_train.*")):
            self.skipTest("needs the native libraries next to the package")

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "exec_graph_harness"
            cmd = [
                compiler, "-std=c++17", "-O1",
                f"-I{REPO / 'csrc'}", f"-I{llama / 'ggml' / 'include'}", f"-I{llama / 'ggml' / 'src'}", f"-I{llama / 'include'}",
                str(REPO / "tests" / "native" / "exec_graph_harness.cpp"), "-o", str(exe),
                f"-L{libs}", "-lgguf_train", "-lggml_train_ggml", f"-Wl,-rpath,{libs}",
            ]
            built = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr[-2000:])
            env = dict(os.environ)
            if sys.platform == "darwin":
                env["DYLD_LIBRARY_PATH"] = str(libs)
            ran = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("0 failure(s)", ran.stdout)
            self.assertNotIn("FAIL", ran.stdout)


if __name__ == "__main__":
    unittest.main()
