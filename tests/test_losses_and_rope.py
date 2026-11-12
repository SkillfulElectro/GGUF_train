"""cross_entropy / nll_loss argument checking and RoPE (BUG-10).

Before: the native cross entropy took the number of targets and the length of the weight vector
on trust and read past the end of whatever it was given - `float` class indices, a short weight
vector and a short target tensor all produced a number - and its backward pass indexed the
weights with unchecked targets. RoPE returned float32 whatever the input dtype, read positions
that were never passed, ended the process on an odd or oversized n_dims and on every mode but
0 and 2, and never returned for n_dims <= 0.
"""

import math
import unittest

import torch
import torch.nn.functional as F
from gguf_train import _C
from gguf_train._functions import CrossEntropyFn


LOGITS = [[0.2, 1.5, -0.3, 0.9], [1.1, -0.7, 0.4, 0.0], [0.3, 0.3, 2.0, -1.0]]


def log_softmax_row(row):
    m = max(row)
    lse = m + math.log(sum(math.exp(v - m) for v in row))
    return [v - lse for v in row]


def reference_items(logits, targets, weight=None, ignore_index=-100, smoothing=0.0):
    """Per-item (loss, weight of the target class) for class-index targets, in plain Python."""
    n_classes = len(logits[0])
    w = weight or [1.0] * n_classes
    items = []
    for row, t in zip(logits, targets):
        if t == ignore_index:
            items.append((0.0, 0.0))
            continue
        logp = log_softmax_row(row)
        nll = -w[t] * logp[t]
        smooth = -sum(wc * lp for wc, lp in zip(w, logp)) / n_classes
        items.append(((1.0 - smoothing) * nll + smoothing * smooth, w[t]))
    return items


class TestCrossEntropyValues(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(LOGITS)

    def assertClose(self, got, want, places=5):
        got = got.tolist() if hasattr(got, "tolist") else got
        if isinstance(want, list):
            self.assertEqual(len(got), len(want))
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=places)
        else:
            self.assertAlmostEqual(got, want, places=places)

    def test_class_indices(self):
        targets = [1, 0, 2]
        items = reference_items(LOGITS, targets)
        t = torch.tensor(targets)
        self.assertClose(F.cross_entropy(self.logits, t, reduction="none"), [l for l, _ in items])
        self.assertClose(F.cross_entropy(self.logits, t, reduction="sum"), sum(l for l, _ in items))
        self.assertClose(F.cross_entropy(self.logits, t), sum(l for l, _ in items) / 3)

    def test_weights_and_ignore_index(self):
        weight = [1.0, 2.0, 3.0, 4.0]
        targets = [1, -100, 3]
        items = reference_items(LOGITS, targets, weight)
        t, w = torch.tensor(targets), torch.tensor(weight)
        self.assertClose(F.cross_entropy(self.logits, t, weight=w, reduction="none"), [l for l, _ in items])
        self.assertClose(F.cross_entropy(self.logits, t, weight=w), sum(l for l, _ in items) / sum(wt for _, wt in items))
        self.assertClose(F.cross_entropy(self.logits, torch.tensor([7, 0, 2]), ignore_index=7),
                         sum(l for l, _ in reference_items(LOGITS, [7, 0, 2], ignore_index=7)) / 2)
        self.assertEqual(F.cross_entropy(self.logits, torch.tensor([-100, -100, -100])).item(), 0.0)

    def test_label_smoothing(self):
        targets = [1, 0, 2]
        t = torch.tensor(targets)
        items = reference_items(LOGITS, targets, smoothing=0.2)
        self.assertClose(F.cross_entropy(self.logits, t, label_smoothing=0.2), sum(l for l, _ in items) / 3)
        weight = [1.0, 2.0, 3.0, 4.0]
        items = reference_items(LOGITS, targets, weight, smoothing=0.2)
        got = F.cross_entropy(self.logits, t, weight=torch.tensor(weight), label_smoothing=0.2, reduction="none")
        self.assertClose(got, [l for l, _ in items])

    def test_extra_dimensions(self):
        # (N, C, d): the class dimension is 1, the target has the shape (N, d)
        rows = [[0.1 * (n + c + 2 * d) * (-1) ** c for c in range(3)] for n in range(2) for d in range(4)]
        targets = [0, 1, 2, 0, 2, 2, 1, 0]
        logits = torch.tensor(rows).reshape(2, 4, 3).permute(0, 2, 1)
        t = torch.tensor(targets).reshape(2, 4)
        items = reference_items(rows, targets)
        got = F.cross_entropy(logits, t, reduction="none")
        self.assertEqual(tuple(got.shape), (2, 4))
        self.assertClose(got.reshape(-1), [l for l, _ in items])
        self.assertClose(F.cross_entropy(logits, t), sum(l for l, _ in items) / 8)

    def test_unbatched_input(self):
        want = reference_items(LOGITS[:1], [1])[0][0]
        self.assertClose(F.cross_entropy(self.logits[0], torch.tensor(1)), want)
        self.assertClose(F.cross_entropy(self.logits[0], torch.tensor([1])), want)
        self.assertClose(F.nll_loss(F.log_softmax(self.logits[0], dim=0), torch.tensor(1)), want)

    def test_class_probabilities(self):
        probs = [[0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]]
        weight = [1.0, 2.0, 3.0, 4.0]
        per_item = [-sum(p * lp for p, lp in zip(pr, log_softmax_row(row))) for pr, row in zip(probs, LOGITS)]
        weighted = [-sum(w * p * lp for w, p, lp in zip(weight, pr, log_softmax_row(row))) for pr, row in zip(probs, LOGITS)]
        p, w = torch.tensor(probs), torch.tensor(weight)
        self.assertClose(F.cross_entropy(self.logits, p, reduction="none"), per_item)
        self.assertClose(F.cross_entropy(self.logits, p), sum(per_item) / 3)
        self.assertClose(F.cross_entropy(self.logits, p, weight=w), sum(weighted) / 3)
        # a single unbatched item used to drop the weights
        self.assertClose(F.cross_entropy(self.logits[2], p[2], weight=w), weighted[2])
        # the native implementation of the same case
        native = CrossEntropyFn.apply(self.logits, p, w, -100, 0.0, 0)
        self.assertClose(native, weighted)

    def test_nll_loss_matches_cross_entropy(self):
        t = torch.tensor([1, -100, 3])
        logp = F.log_softmax(self.logits, dim=1)
        for weight in (None, torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0.1, 0.1, 0.1, 0.1]), torch.tensor([0.0, 0.0, 0.0, 0.0])):
            for reduction in ("none", "sum", "mean"):
                with self.subTest(weight=None if weight is None else weight.tolist(), reduction=reduction):
                    a = F.nll_loss(logp, t, weight=weight, reduction=reduction)
                    b = F.cross_entropy(self.logits, t, weight=weight, reduction=reduction)
                    self.assertClose(a.reshape(-1), b.reshape(-1).tolist())

    def test_gradients(self):
        from torch.autograd import gradcheck

        logits = torch.tensor(LOGITS, dtype=torch.float64, requires_grad=True)
        t = torch.tensor([1, -100, 3])
        w = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        p = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
        for reduction in ("mean", "sum"):
            with self.subTest(reduction=reduction):
                self.assertTrue(gradcheck(lambda z: F.cross_entropy(z, t, weight=w, reduction=reduction), (logits,)))
                self.assertTrue(gradcheck(lambda z: F.cross_entropy(z, t, weight=w, label_smoothing=0.3, reduction=reduction), (logits,)))
                self.assertTrue(gradcheck(lambda z: F.cross_entropy(z, p, weight=w, reduction=reduction), (logits,)))
                self.assertTrue(gradcheck(lambda z: F.nll_loss(F.log_softmax(z, dim=1), t, weight=w, reduction=reduction), (logits,)))
        self.assertTrue(gradcheck(lambda z: F.cross_entropy(z, t, reduction="none").sum(), (logits,)))


class TestCrossEntropyArguments(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(LOGITS)
        self.targets = torch.tensor([1, 0, 2])

    def assertRejected(self, fragment, fn, exc=RuntimeError):
        with self.assertRaises(exc) as ctx:
            fn()
        self.assertIn(fragment, str(ctx.exception))

    def test_target_shape(self):
        for name, loss in (("cross_entropy", F.cross_entropy), ("nll_loss", F.nll_loss)):
            for bad in ([1, 0], [1, 0, 2, 3, 1], [[1], [0], [2]]):
                with self.subTest(loss=name, target=bad):
                    self.assertRejected("expected a target of shape [3]", lambda: loss(self.logits, torch.tensor(bad)))
            self.assertRejected("expected a target of shape []", lambda: loss(self.logits[0], self.targets))
        # the same number of elements in the wrong arrangement
        logits = torch.zeros(2, 3, 4)
        self.assertRejected("expected a target of shape [2, 4]", lambda: F.cross_entropy(logits, torch.zeros(4, 2, dtype=torch.int64)))
        self.assertRejected("expected a target of shape [2, 4]", lambda: F.nll_loss(logits, torch.zeros(2, 3, dtype=torch.int64)))

    def test_class_indices_as_floats_are_not_read_as_probabilities(self):
        # three floats were read as a 3 x 4 table of probabilities: nine values from outside the tensor
        self.assertRejected("class probabilities", lambda: F.cross_entropy(self.logits, torch.tensor([1.0, 0.0, 2.0])))
        self.assertRejected("class probabilities", lambda: F.cross_entropy(self.logits, torch.zeros(3, 2)))
        self.assertRejected("integer", lambda: F.nll_loss(self.logits, torch.tensor([1.0, 0.0, 2.0])))
        self.assertRejected("integer", lambda: F.cross_entropy(self.logits, torch.tensor([True, False, True])))

    def test_weight_shape(self):
        for name, loss in (("cross_entropy", F.cross_entropy), ("nll_loss", F.nll_loss)):
            for bad in (torch.ones(2), torch.ones(5), torch.ones(4, 1), torch.tensor(1.0)):
                with self.subTest(loss=name, weight=list(bad.shape)):
                    self.assertRejected("for all 4 classes", lambda: loss(self.logits, self.targets, weight=bad))
        self.assertRejected("for all 4 classes", lambda: F.cross_entropy(self.logits, torch.zeros(3, 4), weight=torch.ones(2)))

    def test_target_values(self):
        for name, loss in (("cross_entropy", F.cross_entropy), ("nll_loss", F.nll_loss)):
            for bad in (4, 99, -1):
                with self.subTest(loss=name, target=bad):
                    self.assertRejected(f"target {bad} is out of bounds for 4 classes", lambda: loss(self.logits, torch.tensor([1, bad, 2])))
            # ignore_index is not a class and may be anything
            self.assertTrue(math.isfinite(loss(self.logits, torch.tensor([1, 99, 2]), ignore_index=99).item()))

    def test_other_arguments(self):
        self.assertRejected("label_smoothing must be between 0.0 and 1.0", lambda: F.cross_entropy(self.logits, self.targets, label_smoothing=1.5))
        self.assertRejected("label_smoothing must be between 0.0 and 1.0", lambda: F.cross_entropy(self.logits, self.targets, label_smoothing=-0.5))
        self.assertRejected("Invalid reduction", lambda: F.cross_entropy(self.logits, self.targets, reduction="avg"), ValueError)
        self.assertRejected("Invalid reduction", lambda: F.nll_loss(self.logits, self.targets, reduction="avg"), ValueError)
        self.assertRejected("floating point", lambda: F.cross_entropy(torch.tensor([[1, 2, 3]]), torch.tensor([1])))

    def test_empty_batch(self):
        self.assertEqual(F.cross_entropy(torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64)).item(), 0.0)
        self.assertEqual(tuple(F.cross_entropy(torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64), reduction="none").shape), (0,))


class TestNativeCrossEntropyChecks(unittest.TestCase):
    """The native entry points check for themselves: they are the ones that index memory."""

    def setUp(self):
        self.logits = torch.tensor(LOGITS)
        self.targets = torch.tensor([1, 0, 2])

    def forward_error(self, logits, targets, weight=None, ignore_index=-100, smoothing=0.0, reduction=1):
        with self.assertRaises(RuntimeError) as ctx:
            CrossEntropyFn.apply(logits, targets, weight, ignore_index, smoothing, reduction)
        return str(ctx.exception)

    def backward_error(self, grad, logits, targets, weight=None, ignore_index=-100, smoothing=0.0, reduction=1):
        ptr = _C.lib.gt_cross_entropy_loss_backward(
            grad._ptr, logits._ptr, targets._ptr, weight._ptr if weight is not None else 0, ignore_index, smoothing, reduction
        )
        self.assertFalse(ptr, "the backward pass accepted invalid arguments")
        return _C.last_error()

    def test_forward(self):
        self.assertIn("expected 3 class indices", self.forward_error(self.logits, torch.tensor([1, 0])))
        self.assertIn("expected 3 class indices", self.forward_error(self.logits, torch.tensor([1, 0, 2, 3])))
        self.assertIn("class probabilities", self.forward_error(self.logits, torch.tensor([1.0, 0.0, 2.0])))
        self.assertIn("class probabilities", self.forward_error(self.logits, torch.zeros(4, 3)))
        self.assertIn("one value per class", self.forward_error(self.logits, self.targets, torch.ones(2)))
        self.assertIn("one value per class", self.forward_error(self.logits, torch.zeros(3, 4), torch.ones(2)))
        self.assertIn("target 4 is out of bounds for 4 classes", self.forward_error(self.logits, torch.tensor([1, 4, 2])))
        self.assertIn("label_smoothing", self.forward_error(self.logits, self.targets, smoothing=2.0))
        self.assertIn("reduction", self.forward_error(self.logits, self.targets, reduction=7))
        self.assertIn("floating point", self.forward_error(torch.tensor([[1, 2, 3]]), torch.tensor([1])))
        self.assertIn("integer class indices", self.forward_error(self.logits, torch.tensor([True, False, True])))

    def test_backward(self):
        one = torch.tensor(1.0)
        w = torch.ones(4)
        self.assertIn("expected 3 class indices", self.backward_error(one, self.logits, torch.tensor([1, 0])))
        self.assertIn("one value per class", self.backward_error(one, self.logits, self.targets, torch.ones(2)))
        # used to index the weight vector with the target, unchecked
        self.assertIn("target 1000000 is out of bounds", self.backward_error(one, self.logits, torch.tensor([1, 1000000, 2]), w))
        self.assertIn("target -7 is out of bounds", self.backward_error(one, self.logits, torch.tensor([1, -7, 2]), w))
        self.assertIn("gradient of 3 element(s)", self.backward_error(torch.ones(2), self.logits, self.targets, reduction=0))
        self.assertIn("gradient of 1 element(s)", self.backward_error(torch.ones(0), self.logits, self.targets, reduction=1))

    def test_targets_changed_after_the_forward_pass(self):
        logits = torch.tensor(LOGITS, requires_grad=True)
        targets = torch.tensor([1, 0, 2])
        loss = F.cross_entropy(logits, targets, weight=torch.ones(4))
        targets[1] = 123456
        with self.assertRaises(RuntimeError) as ctx:
            loss.backward()
        self.assertIn("target 123456 is out of bounds for 4 classes", str(ctx.exception))


def rope_reference(x, positions, n_dims=None, neox=False, freq_base=10000.0, freq_scale=1.0):
    """RoPE of one head: x[seq][dim] -> rotated copy, in plain Python."""
    d = len(x[0])
    n_dims = d if n_dims is None else n_dims
    out = [list(row) for row in x]
    for s, row in enumerate(x):
        for i in range(n_dims // 2):
            theta = positions[s] * freq_scale * freq_base ** (-2.0 * i / n_dims)
            a, b = (i, i + n_dims // 2) if neox else (2 * i, 2 * i + 1)
            out[s][a] = row[a] * math.cos(theta) - row[b] * math.sin(theta)
            out[s][b] = row[a] * math.sin(theta) + row[b] * math.cos(theta)
    return out


class TestRoPE(unittest.TestCase):
    def head(self, seq=3, dim=6):
        return [[0.5 + 0.25 * s - 0.125 * d + 0.03 * s * d for d in range(dim)] for s in range(seq)]

    def assertRows(self, got, want, places=4):
        got = got.tolist()
        for grow, wrow in zip(got, want):
            for g, w in zip(grow, wrow):
                self.assertAlmostEqual(g, w, places=places)

    def test_values(self):
        x = self.head()
        pos = [0, 1, 7]
        for kwargs in ({}, {"n_dims": 4}, {"neox": True}, {"n_dims": 2, "neox": True}, {"freq_base": 100.0}, {"freq_scale": 0.5}):
            with self.subTest(**kwargs):
                call = dict(kwargs)
                mode = 2 if call.pop("neox", False) else 0
                want = rope_reference(x, pos, **kwargs)
                t = torch.tensor([x, x])                                   # (heads, seq, dim)
                got = F.rope(t, torch.tensor(pos), mode=mode, **call)
                self.assertEqual(tuple(got.shape), (2, 3, 6))
                self.assertRows(got[0], want)
                self.assertRows(got[1], want)
                got4 = F.rope(t.reshape(1, 2, 3, 6), torch.tensor(pos), mode=mode, **call)   # (batch, heads, seq, dim)
                self.assertRows(got4[0, 1], want)

    def test_result_keeps_the_dtype_of_the_input(self):
        x = self.head()
        want = rope_reference(x, [0, 1, 2])
        for dtype, places in ((torch.float32, 5), (torch.float64, 5), (torch.float16, 2), (torch.bfloat16, 1)):
            with self.subTest(dtype=str(dtype)):
                t = torch.tensor([x], dtype=dtype, requires_grad=True)
                out = F.rope(t, torch.arange(3))
                self.assertEqual(out.dtype, dtype)
                self.assertRows(out[0], want, places=places)
                out.sum().backward()
                self.assertEqual(t.grad.dtype, dtype)

    def test_backward_is_the_inverse_rotation(self):
        x = torch.randn(2, 3, 5, 8)
        y = torch.randn(2, 3, 5, 8)
        pos = torch.tensor([4, 0, 9, 2, 2])
        for mode in (0, 2):
            for n_dims in (8, 4):
                with self.subTest(mode=mode, n_dims=n_dims):
                    xr = x.clone().requires_grad_(True)
                    out = F.rope(xr, pos, n_dims=n_dims, mode=mode)
                    (out * y).sum().backward()
                    # <rope(x), y> == <x, rope^T(y)>, and a rotation keeps the norm
                    self.assertAlmostEqual((out * y).sum().item(), (x * xr.grad).sum().item(), places=3)
                    self.assertAlmostEqual((out * out).sum().item(), (x * x).sum().item(), places=3)
                    self.assertAlmostEqual((xr.grad * xr.grad).sum().item(), (y * y).sum().item(), places=3)

    def test_views_and_other_layouts(self):
        x = self.head(seq=3, dim=4)
        wide = torch.tensor([[v for value in row for v in (value, -1.0)] for row in x]).reshape(1, 3, 8)
        got = F.rope(wide[..., ::2], torch.arange(3))
        self.assertRows(got[0], rope_reference(x, [0, 1, 2]))
        # positions as int32, as a view, and as floats holding whole numbers
        want = rope_reference(x, [3, 5, 7])
        t = torch.tensor([x])
        self.assertRows(F.rope(t, torch.tensor([3, 5, 7], dtype=torch.int32))[0], want)
        self.assertRows(F.rope(t, torch.tensor([3, 0, 5, 0, 7, 0])[::2])[0], want)

    def test_positions(self):
        x = torch.tensor([self.head(seq=3, dim=4)])
        want = rope_reference(self.head(seq=3, dim=4), [0, 1, 2])
        # a longer position table: the first seq_len entries are used
        self.assertRows(F.rope(x, torch.arange(10))[0], want)
        # too short: three of the positions used to come from behind the tensor
        with self.assertRaises(RuntimeError) as ctx:
            F.rope(x, torch.arange(2))
        self.assertIn("a position for each of the 3 sequence elements, got shape [2]", str(ctx.exception))
        for bad in (torch.arange(6).reshape(2, 3), torch.tensor(0)):
            with self.assertRaises(RuntimeError) as ctx:
                F.rope(x, bad)
            self.assertIn("pos must be a 1-D tensor", str(ctx.exception))
        g, short = torch.ones(1, 3, 4), torch.arange(2)
        self.assertFalse(_C.lib.gt_rope_backward(g._ptr, short._ptr, 4, 0, 0, 10000.0, 1.0))
        self.assertIn("a position for each of the 3 sequence elements", _C.last_error())

    def test_arguments_that_used_to_end_or_hang_the_process(self):
        x = torch.ones(1, 1, 2, 4)
        pos = torch.arange(2)
        cases = {
            "odd n_dims": dict(n_dims=3),
            "n_dims above the head dimension": dict(n_dims=8),
            "n_dims 0 (never returned)": dict(n_dims=0),
            "negative n_dims (never returned)": dict(n_dims=-2),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError) as ctx:
                    F.rope(x, pos, **kwargs)
                self.assertIn("n_dims must be an even number between 2 and the head dimension 4", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            F.rope(torch.ones(1, 1, 2, 5), pos)            # n_dims defaults to the head dimension
        self.assertIn("head dimension 5, got 5", str(ctx.exception))
        for mode in (1, 4, 8, 24, 40, -1):
            with self.subTest(mode=mode):
                with self.assertRaises(RuntimeError) as ctx:
                    F.rope(x, pos, mode=mode)
                self.assertIn("mode must be 0 (normal) or 2 (NeoX)", str(ctx.exception))
        for base in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(freq_base=base):
                with self.assertRaises(RuntimeError) as ctx:
                    F.rope(x, pos, freq_base=base)
                self.assertIn("freq_base must be positive", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            F.rope(torch.ones(2, 4), pos)
        self.assertIn("expected x of shape", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            F.rope(torch.ones(1, 1, 2, 4, dtype=torch.int64), pos)
        self.assertIn("floating point", str(ctx.exception))
        # and the same checks guard the backward entry point
        g = torch.ones(1, 1, 2, 4)
        for args in ((3, 0), (0, 0), (8, 0), (4, 8)):
            self.assertFalse(_C.lib.gt_rope_backward(g._ptr, pos._ptr, args[0], args[1], 0, 10000.0, 1.0), args)

    def test_empty_input(self):
        out = F.rope(torch.ones(1, 2, 0, 4, dtype=torch.float64), torch.arange(0))
        self.assertEqual(tuple(out.shape), (1, 2, 0, 4))
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(tuple(F.rope(torch.ones(0, 2, 3, 4), torch.arange(3)).shape), (0, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
