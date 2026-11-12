"""In-place operations are operations of the autograd graph; the engine handles graphs of any
depth and lets go of them after the backward pass."""

import gc
import unittest
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def flat(t):
    return None if t is None else t.reshape(-1).tolist()


class Base(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.a = torch.tensor([[0.5, -1.5, 2.0], [-0.25, 3.0, -2.5]], requires_grad=True)
        self.b = torch.tensor([[1.5, 0.5, -2.0], [2.5, -1.0, 0.75]], requires_grad=True)
        self.weights = torch.tensor([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]])

    def assert_close(self, got, want, places=5):
        if got is None or want is None:        # no gradient and a gradient of zeros are the same thing
            self.assertFalse(any(got or want or ()), f"{got} != {want}")
            return
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=places)

    def compare(self, in_place, out_of_place, start=None, leaves=None):
        """`in_place(h)` changes h and `out_of_place(h)` returns the same thing as a new
        tensor; values and gradients have to agree."""
        leaves = leaves or [self.a, self.b]
        start = start or (lambda: self.a * 1.5 + 0.25)
        results = []
        for fn, is_in_place in ((out_of_place, False), (in_place, True)):
            for leaf in leaves:
                leaf.grad = None
            h = start()
            out = fn(h)
            if is_in_place:
                self.assertIs(out if out is not None else h, h)
                out = h
            (out * self.weights).sum().backward()
            results.append((flat(out), [flat(leaf.grad) for leaf in leaves]))
        (want_value, want_grads), (got_value, got_grads) = results
        self.assert_close(got_value, want_value)
        for g, w in zip(got_grads, want_grads):
            self.assert_close(g, w)
        return want_grads


class TestInPlaceMethods(Base):
    def test_arithmetic(self):
        b = self.b
        cases = (
            (lambda h: h.add_(2.0), lambda h: h + 2.0),
            (lambda h: h.add_(b), lambda h: h + b),
            (lambda h: h.add_(b, alpha=2.5), lambda h: h + 2.5 * b),
            (lambda h: h.sub_(b), lambda h: h - b),
            (lambda h: h.sub_(b, alpha=0.5), lambda h: h - 0.5 * b),
            (lambda h: h.mul_(3.0), lambda h: h * 3.0),
            (lambda h: h.mul_(b), lambda h: h * b),
            (lambda h: h.div_(4.0), lambda h: h / 4.0),
            (lambda h: h.div_(b), lambda h: h / b),
            (lambda h: h.mul_(h.detach().clone()), lambda h: h * h.detach()),
        )
        for i, (in_place, out_of_place) in enumerate(cases):
            with self.subTest(case=i):
                grads = self.compare(in_place, out_of_place)
                self.assertTrue(any(v != 0.0 for v in grads[0]))

    def test_the_gradient_knows_the_operation(self):
        # h.mul_(3) used to be differentiated as if it had not happened
        a = self.a
        h = a * 1.0
        h.mul_(3.0)
        h.sum().backward()
        self.assertEqual(flat(a.grad), [3.0] * 6)

    def test_elementwise_functions(self):
        positive = lambda: self.a * self.a + 0.5          # noqa: E731
        cases = (
            (lambda h: h.relu_(), lambda h: h.relu(), None),
            (lambda h: h.sigmoid_(), lambda h: h.sigmoid(), None),
            (lambda h: h.tanh_(), lambda h: h.tanh(), None),
            (lambda h: h.neg_(), lambda h: -h, None),
            (lambda h: h.abs_(), lambda h: h.abs(), None),
            (lambda h: h.exp_(), lambda h: h.exp(), None),
            (lambda h: h.sqrt_(), lambda h: h.sqrt(), positive),
            (lambda h: h.log_(), lambda h: h.log(), positive),
            (lambda h: h.clamp_(-1.0, 2.0), lambda h: h.clamp(-1.0, 2.0), None),
            (lambda h: h.clamp_(min=0.5), lambda h: h.clamp(min=0.5), None),
            (lambda h: h.clip_(max=1.0), lambda h: h.clamp(max=1.0), None),
        )
        for i, (in_place, out_of_place, start) in enumerate(cases):
            with self.subTest(case=i):
                self.compare(in_place, out_of_place, start=start)

    def test_fills_and_copies(self):
        b = self.b
        mask = torch.tensor([[True, False, True], [False, False, True]])
        index = torch.tensor([[2, 0, 1], [1, 1, 0]])
        cases = (
            (lambda h: h.masked_fill_(mask, 7.0), lambda h: h.masked_fill(mask, 7.0)),
            (lambda h: h.scatter_(1, index, b), lambda h: h.scatter(1, index, b)),
            (lambda h: h.scatter_add_(1, index, b), lambda h: h.scatter_add(1, index, b)),
            (lambda h: h.copy_(b * 2.0), lambda h: b * 2.0 + h * 0.0),
            (lambda h: h.copy_(b[0]), lambda h: b[0].expand(2, 3) + h * 0.0),            # broadcast
            (lambda h: h.fill_(b.sum()), lambda h: b.sum().expand(2, 3) + h * 0.0),
        )
        for i, (in_place, out_of_place) in enumerate(cases):
            with self.subTest(case=i):
                self.compare(in_place, out_of_place)

    def test_constants_cut_the_history(self):
        a = self.a
        for fill in (lambda h: h.zero_(), lambda h: h.fill_(3.0), lambda h: h.copy_(torch.ones(2, 3)),
                     lambda h: h.normal_(), lambda h: h.uniform_(), lambda h: h.bernoulli_(0.5)):  # fmt: skip
            a.grad = None
            h = a * 2.0
            before = (h * self.weights).sum()         # an earlier use keeps its gradient
            self.assertIs(fill(h), h)
            self.assertFalse(h.requires_grad)
            self.assertIsNone(h.grad_fn)
            (before + (h * a).sum()).backward()
            want = [2.0 * w + v for w, v in zip(flat(self.weights), flat(h))]
            self.assert_close(flat(a.grad), want)

    def test_operators(self):
        b = self.b

        def iadd(h):
            h += b

        def isub(h):
            h -= 2.0 * b

        def imul(h):
            h *= b

        def idiv(h):
            h /= b

        for in_place, out_of_place in ((iadd, lambda h: h + b), (isub, lambda h: h - 2.0 * b), (imul, lambda h: h * b), (idiv, lambda h: h / b)):
            with self.subTest(op=in_place.__name__):
                self.compare(in_place, out_of_place)

    def test_accumulating_a_loss(self):
        # the usual training loop: every term has to reach the parameters
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        total = 0.0
        for i in range(1, 4):
            total += (w * i).sum()
        total.backward()
        self.assertEqual(flat(w.grad), [6.0, 6.0])
        self.assertEqual(total.item(), 18.0)

    def test_a_plain_tensor_takes_up_the_history_of_what_is_put_into_it(self):
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        acc = torch.zeros(2)
        acc += w * 3.0
        self.assertTrue(acc.requires_grad)
        acc.mul_(w)
        acc.sum().backward()
        self.assertEqual(flat(w.grad), [6.0, 12.0])              # d/dw 3 w^2
        buf = torch.zeros(3)
        buf[1] = w.sum()
        buf[2] = 5.0
        self.assertEqual(flat(buf), [0.0, 3.0, 5.0])
        w.grad = None
        (buf * torch.tensor([1.0, 10.0, 100.0])).sum().backward()
        self.assertEqual(flat(w.grad), [10.0, 10.0])
        # an integer tensor cannot carry a gradient
        counts = torch.zeros(2, dtype=torch.int64)
        counts.add_(w.detach().long())
        self.assertFalse(counts.requires_grad)

    def test_earlier_uses_keep_the_earlier_value(self):
        a = self.a
        h = a * 1.5
        early = h * 2.0                    # uses h before the change
        h.add_(1.0)
        h.mul_(a)                          # h = (1.5 a + 1) a
        late = h * 3.0
        (early + late).sum().backward()
        want = [2.0 * 1.5 + 3.0 * (3.0 * v + 1.0) for v in flat(a)]
        self.assert_close(flat(a.grad), want)

    def test_it_is_the_same_tensor(self):
        a = self.a
        h = a * 2.0
        alias = h
        view = h[0]
        ident = id(h)
        self.assertIs(h.add_(1.0), h)
        h *= 2.0
        self.assertEqual(id(h), ident)
        self.assertIs(alias, h)
        self.assertEqual(flat(view), [(2.0 * v + 1.0) * 2.0 for v in flat(a)[:3]])      # the storage was written
        self.assertFalse(h.is_leaf)
        self.assertIsNotNone(h.grad_fn)

    def test_leaves(self):
        a = self.a
        for op in (lambda: a.add_(1.0), lambda: a.relu_(), lambda: a.zero_(), lambda: a.copy_(self.b), lambda: a.__setitem__(0, 1.0)):
            with self.assertRaisesRegex(RuntimeError, "leaf Variable that requires grad"):
                op()
        with self.assertRaisesRegex(RuntimeError, "leaf Variable that requires grad"):
            a += 1.0
        before = flat(a)
        with torch.no_grad():
            a.add_(1.0)
            a[0, 0] = 9.0
        self.assertTrue(a.is_leaf and a.requires_grad and a.grad_fn is None)
        self.assertEqual(flat(a)[0], 9.0)
        self.assertEqual(flat(a)[1:], [v + 1.0 for v in before[1:]])

    def test_nothing_is_recorded_without_a_reason(self):
        x = torch.ones(2, 3)
        x.add_(1.0).mul_(torch.full((2, 3), 2.0)).relu_()
        x[0] = 7.0
        self.assertFalse(x.requires_grad)
        self.assertIsNone(x.grad_fn)
        self.assertTrue(x.is_leaf)
        h = self.a * 2.0
        with torch.no_grad():
            h.mul_(10.0)                   # data only: the graph does not see it
        h.sum().backward()
        self.assertEqual(flat(self.a.grad), [2.0] * 6)

    def test_shapes_and_dtypes(self):
        h = self.a * 1.0
        with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast shape"):
            h.add_(torch.ones(4, 2, 3, requires_grad=True))
        h.add_(torch.ones(3, dtype=torch.float64, requires_grad=True))
        self.assertEqual(h.dtype, torch.float32)
        self.assertEqual(h.shape, (2, 3))

    def test_second_derivative(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        h = x * x
        h *= x                             # x^3
        (g,) = torch.autograd.grad(h.sum(), x, create_graph=True)
        self.assertEqual(flat(g), [3.0, 12.0, 27.0])
        g.sum().backward()
        self.assertEqual(flat(x.grad), [6.0, 12.0, 18.0])

    def test_hooks_and_retain_grad(self):
        a = self.a
        h = a * 2.0
        seen = []
        h.register_hook(lambda g: seen.append(flat(g)))       # for the value h has now
        h.mul_(3.0)
        h.retain_grad()
        (h * self.weights).sum().backward()
        self.assertEqual(seen, [[3.0 * w for w in flat(self.weights)]])
        self.assertEqual(flat(h.grad), flat(self.weights))    # of the present value
        self.assertEqual(flat(a.grad), [6.0 * w for w in flat(self.weights)])

    def test_detach_after_use(self):
        # an edge remembers the node: detaching the tensor later does not cut what used it
        a = self.a
        h = a * 2.0
        out = h * 3.0
        h.detach_()
        out.sum().backward()
        self.assertEqual(flat(a.grad), [6.0] * 6)


class TestIndexAssignment(Base):
    def test_assignment(self):
        b = self.b
        mask = torch.tensor([[True, False, True], [False, False, True]])
        rows = torch.tensor([1, 0])
        positions = torch.arange(6).reshape(2, 3)

        def reference(h, key, v):
            """`h` with `v` at `key`, built from operations that do not assign."""
            keep = torch.ones(2, 3)
            keep[key] = 0.0
            where = positions[key]
            v = v if isinstance(v, torch.Tensor) else torch.tensor(float(v))
            placed = torch.zeros(6).scatter_add(0, where.reshape(-1), v.expand(where.shape).reshape(-1))
            return h * keep + placed.reshape(2, 3)

        cases = (
            (0, lambda h: 5.0), ((slice(None), 1), lambda h: 2.0), (mask, lambda h: -1.0), ((1, 2), lambda h: 4.0),
            (0, lambda h: b[1]), ((slice(None), 1), lambda h: b[:, 0]), (mask, lambda h: b[mask]), (rows, lambda h: b * 2.0),
            ((slice(None), slice(0, 2)), lambda h: b.sum()), (1, lambda h: b[0, :1]), ((1, 2), lambda h: b[0, 0]),
            (0, lambda h: h[1] * 2.0),               # from the tensor itself
            (0, lambda h: torch.tensor([1.0, 2.0, 3.0])),
        )  # fmt: skip
        for i, (key, value) in enumerate(cases):
            with self.subTest(case=i):

                def in_place(h):
                    h[key] = value(h)

                self.compare(in_place, lambda h: reference(h, key, value(h)))

    def test_augmented_assignment_on_an_index(self):
        b = self.b
        mask = torch.tensor([[True, False, True], [False, False, True]])

        def add_row(h):
            h[0] += b[1]

        def scale_column(h):
            h[:, 1] *= 3.0

        def sub_masked(h):
            h[mask] -= b[mask]

        row0 = torch.tensor([[1.0], [0.0]])
        col1 = torch.tensor([0.0, 1.0, 0.0])
        cases = (
            (add_row, lambda h: h + row0 * b[1]),
            (scale_column, lambda h: h * (1.0 + 2.0 * col1)),
            (sub_masked, lambda h: h - b * mask.float()),
        )
        for in_place, out_of_place in cases:
            with self.subTest(op=in_place.__name__):
                self.compare(in_place, out_of_place)

    def test_gradient_of_what_was_overwritten_is_zero(self):
        a, b = self.a, self.b
        h = a * 2.0
        h[0] = b[1] * 3.0
        h.sum().backward()
        self.assertEqual(flat(a.grad), [0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
        self.assertEqual(flat(b.grad), [0.0, 0.0, 0.0, 3.0, 3.0, 3.0])


class TestFunctionalInPlace(Base):
    def test_activations(self):
        cases = (
            ("relu", F.relu, {}), ("relu6", F.relu6, {}), ("leaky_relu", F.leaky_relu, {"negative_slope": 0.2}),
            ("elu", F.elu, {}), ("celu", F.celu, {"alpha": 1.5}), ("selu", F.selu, {}), ("silu", F.silu, {}),
            ("mish", F.mish, {}), ("hardswish", F.hardswish, {}), ("hardsigmoid", F.hardsigmoid, {}),
            ("hardtanh", F.hardtanh, {"min_val": -0.5, "max_val": 1.0}), ("threshold", F.threshold, {"threshold": 0.3, "value": -2.0}),
        )  # fmt: skip
        for name, fn, kwargs in cases:
            with self.subTest(fn=name):
                grads = self.compare(lambda h: fn(h, inplace=True, **kwargs), lambda h: fn(h, **kwargs), start=lambda: self.a * 2.0 + 0.25)
                # the gradient is the one of the function, not of the identity (2.0 * weights)
                self.assertNotEqual([round(g, 4) for g in grads[0]], [round(2.0 * w, 4) for w in flat(self.weights)])
        self.compare(lambda h: F.relu_(h), lambda h: F.relu(h))

    def test_modules(self):
        for module, reference in ((nn.ReLU(inplace=True), nn.ReLU()), (nn.ReLU6(inplace=True), nn.ReLU6()), (nn.SiLU(inplace=True), nn.SiLU()),
                                  (nn.LeakyReLU(0.1, inplace=True), nn.LeakyReLU(0.1)), (nn.Hardtanh(inplace=True), nn.Hardtanh())):  # fmt: skip
            with self.subTest(module=type(module).__name__):
                self.compare(module, reference)

    def test_dropout(self):
        for fn in (F.dropout, F.dropout2d):
            with self.subTest(fn=fn.__name__):
                a4 = torch.randn(2, 3, 2, 2, requires_grad=True)
                results = []
                for inplace in (False, True):
                    a4.grad = None
                    torch.manual_seed(4)
                    h = a4 * 2.0
                    out = fn(h, p=0.5, training=True, inplace=inplace)
                    self.assertEqual(out is h, inplace)
                    out.sum().backward()
                    results.append((flat(out), flat(a4.grad)))
                self.assertEqual(results[0], results[1])
                self.assertIn(0.0, results[0][1])                     # the mask is in the gradient
        h = self.a * 2.0
        self.assertIs(F.dropout(h, p=1.0, inplace=True), h)
        self.assertEqual(flat(h), [0.0] * 6)

    def test_after_the_usual_layers(self):
        # nn.ReLU(inplace=True) behind every kind of layer a model puts it behind
        torch.manual_seed(1)
        x4 = torch.randn(2, 4, 5, 5)
        x2 = torch.randn(6, 8)
        tokens = torch.tensor([[1, 3, 0], [2, 2, 4]])
        w = torch.randn(8, 8, requires_grad=True)
        layers = (
            (nn.Linear(8, 8), x2), (nn.Conv2d(4, 4, 3, padding=1), x4), (nn.BatchNorm2d(4), x4), (nn.LayerNorm(8), x2),
            (nn.GroupNorm(2, 4), x4), (nn.Embedding(5, 8), tokens), (nn.Sequential(nn.Linear(8, 8), nn.Tanh()), x2),
            (lambda t: t @ w, x2), (lambda t: (t.unsqueeze(0) + w.unsqueeze(1)).sum(0), x2), (nn.MaxPool2d(2), None),
        )  # fmt: skip
        for i, (layer, x) in enumerate(layers):
            with self.subTest(layer=i):
                if x is None:
                    scale = torch.ones(1, requires_grad=True)
                    params = [scale]
                    run = lambda: layer(x4 * scale)             # noqa: E731
                else:
                    params = list(layer.parameters()) if isinstance(layer, nn.Module) else [w]
                    run = lambda: layer(x)                      # noqa: E731
                grads = []
                for inplace in (False, True):
                    for p in params:
                        p.grad = None
                    out = F.relu(run() - 0.1, inplace=inplace)
                    out = out.mul_(2.0) if inplace else out * 2.0
                    out.pow(2).sum().backward()
                    grads.append([flat(p.grad) for p in params])
                for g, want in zip(grads[1], grads[0]):
                    self.assert_close(g, want, places=4)


class TestCheckpointAndInPlace(Base):
    def test_changing_an_outside_tensor_in_place_is_refused(self):
        lin = nn.Linear(3, 3)
        h = lin(self.a)
        for fn in (nn.Sequential(nn.ReLU(inplace=True), lin), lambda t: lin(t.mul_(2.0)), lambda t: lin(F.silu(t, inplace=True))):
            with self.assertRaisesRegex(RuntimeError, "changes a tensor that requires grad"):
                checkpoint(fn, h)

    def test_in_place_on_what_the_function_made(self):
        lin = nn.Linear(3, 3)
        block = nn.Sequential(lin, nn.ReLU(inplace=True), nn.Linear(3, 3), nn.SiLU(inplace=True))
        params = list(block.parameters()) + [self.a]

        def accumulate(t):
            total = torch.zeros(2, 3)
            for scale in (1.0, 2.0):
                total += block(t) * scale
            total[0] *= 2.0
            return total

        grads = []
        for use_checkpoint in (False, True):
            for p in params:
                p.grad = None
            out = checkpoint(accumulate, self.a) if use_checkpoint else accumulate(self.a)
            (out * self.weights).sum().backward()
            grads.append([flat(p.grad) for p in params])
        for g, want in zip(grads[1], grads[0]):
            self.assert_close(g, want)


class TestEngine(unittest.TestCase):
    def test_deep_graphs(self):
        # the pass used to recurse once per node and hit Python's recursion limit
        w = torch.tensor([1.5], requires_grad=True)
        x = w
        for _ in range(5000):
            x = x * 1.0
        x.sum().backward()
        self.assertEqual(w.grad.tolist(), [1.0])
        (g,) = torch.autograd.grad((w * 2.0 + sum(w for _ in range(3000))).sum(), w)
        self.assertEqual(g.tolist(), [3002.0])

    def test_the_graph_is_released_after_backward(self):
        # ... by reference counting: nothing waits for the garbage collector
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        gc.collect()
        gc.disable()
        try:
            h = (w * 2.0).exp()
            ref = weakref.ref(h)
            loss = (h * 3.0).sum()
            del h
            loss.backward()
            del loss
            self.assertIsNone(ref())
        finally:
            gc.enable()

    def test_diamonds_and_reuse(self):
        w = torch.tensor([2.0], requires_grad=True)
        a = w * 3.0
        b = a * a                      # a twice
        c = a + b
        (c * a).sum().backward()       # (a + a^2) a = a^2 + a^3 -> (2a + 3a^2) * 3
        self.assertEqual(w.grad.tolist(), [(2 * 6.0 + 3 * 36.0) * 3.0])

    def test_a_second_pass_over_a_freed_graph_says_so(self):
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = (w * w).exp().sum()
        loss.backward()
        with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
            loss.backward()
        w.grad = None
        loss = (w * w).exp().sum()
        loss.backward(retain_graph=True)
        first = w.grad.tolist()
        loss.backward()
        self.assertEqual(w.grad.tolist(), [2 * g for g in first])

    def test_next_functions(self):
        w = torch.tensor([2.0], requires_grad=True)
        a = w * 3.0
        out = a + 1.0
        self.assertIs(out.grad_fn.next_functions[0][0], a.grad_fn)
        self.assertIsNone(a.grad_fn.next_functions[0][0])       # a leaf


if __name__ == "__main__":
    unittest.main()
