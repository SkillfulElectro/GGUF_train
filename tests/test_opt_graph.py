"""Tests for OptGraph, the static graph trained natively by ggml_opt.

Nearly every case here used to end the Python process: ggml checks its arguments with
GGML_ASSERT (abort), and the graph was built into a fixed pool that every fit() consumed
a little more of. The cases that did not crash trained on, or wrote to, the wrong memory.
"""

import gc
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch.llama import OptGraph

REPO_ROOT = Path(__file__).resolve().parents[1]

W_TRUE = [[2.0, -1.0, 0.5, 3.0]]


def regression_data(rows=64, seed=5):
    torch.manual_seed(seed)
    x = torch.randn(rows, 4)
    return x, x @ torch.tensor(W_TRUE).t()


def linear_graph(batch=16, **kwargs):
    kwargs.setdefault("loss_type", "mse")
    g = OptGraph(nbatch_physical=batch, nbatch_logical=kwargs.pop("nbatch_logical", batch), **kwargs)
    w = torch.zeros(1, 4)
    b = torch.zeros(1)
    out = g.linear(g.input((batch, 4)), g.param(w), g.param(b))
    return g, out, w, b


def max_abs(t):
    return float(t.abs().max().item())


class TestRepeatedFit(unittest.TestCase):
    def test_fit_can_be_called_many_times(self):
        # The 19th fit() on one graph used to run off the end of an 8 MB pool (SIGSEGV).
        # Run in a child process so that a regression is reported as a failure of this
        # test rather than as the death of the test runner.
        code = (
            "import torch\n"
            "from torch.llama import OptGraph\n"
            "torch.manual_seed(5)\n"
            "X = torch.randn(64, 4)\n"
            f"y = X @ torch.tensor({W_TRUE!r}).t()\n"
            "g = OptGraph(loss_type='mse', nbatch_physical=16, nbatch_logical=16)\n"
            "W = torch.zeros(1, 4); b = torch.zeros(1)\n"
            "out = g.linear(g.input((16, 4)), g.param(W), g.param(b))\n"
            "losses = [g.fit(out, X, y, epochs=1, lr=0.05)[0] for _ in range(60)]\n"
            "print('LOSSES', losses[0], losses[-1])\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, f"child died with {proc.returncode}:\n{proc.stderr[-2000:]}")
        first, last = (float(v) for v in proc.stdout.split("LOSSES")[1].split())
        self.assertLess(last, first * 0.01)

    def test_training_continues_from_the_previous_fit(self):
        x, y = regression_data()
        g, out, w, b = linear_graph()
        # Each call starts from the weights the previous one left behind.
        losses = [g.fit(out, x, y, epochs=2, lr=0.02)[0] for _ in range(6)]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertLess(losses[-1], losses[0] * 0.5)
        g.fit(out, x, y, epochs=150, lr=0.1)
        self.assertTrue(w.allclose(torch.tensor(W_TRUE), atol=0.02))

    def test_graph_can_grow_between_fits(self):
        # Parameters added after the first fit() had no storage and aborted the second one.
        x, y = regression_data()
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        x_id = g.input((16, 4))
        w = torch.zeros(1, 4)
        g.fit(g.linear(x_id, g.param(w)), x, y, epochs=5, lr=0.1)
        b = torch.full((1,), 0.5)
        loss, _ = g.fit(g.linear(x_id, g.param(w), g.param(b)), x, y, epochs=150, lr=0.1)
        self.assertLess(loss, 1e-3)
        self.assertLess(abs(b.item()), 0.05)

    def test_zero_epochs_validates_but_does_not_train(self):
        x, y = regression_data()
        g, out, w, b = linear_graph()
        self.assertEqual(g.fit(out, x, y, epochs=0), (0.0, 0.0))
        self.assertEqual(max_abs(w), 0.0)
        with self.assertRaisesRegex(RuntimeError, "features"):
            g.fit(out, torch.randn(64, 7), y, epochs=0)


class TestFitValidation(unittest.TestCase):
    def test_input_batch_dimension_must_equal_nbatch_physical(self):
        x, y = regression_data()
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        w = torch.zeros(1, 4)
        out = g.linear(g.input((8, 4)), g.param(w))
        with self.assertRaisesRegex(RuntimeError, r"nbatch_physical.*\(16, features\).*\(8, 4\)"):
            g.fit(out, x, y, epochs=1)

        # Anything but (batch, features) cannot be fed batch by batch either.
        g = OptGraph(loss_type="mean", nbatch_physical=16, nbatch_logical=16)
        w = torch.zeros(1, 4)
        out = g.linear(g.input((2, 16, 4)), g.param(w))
        with self.assertRaisesRegex(RuntimeError, r"\(2, 16, 4\)"):
            g.fit(out, torch.randn(64, 4), epochs=1)

    def test_dataset_shape_is_checked(self):
        x, y = regression_data()
        g, out, w, b = linear_graph()
        cases = [
            ((torch.randn(64, 5), y), "5 values .* takes 4 features"),
            ((torch.randn(8, 4), torch.randn(8, 1)), "fewer than one batch"),
            ((x, torch.randn(32, 1)), "one row per row of training data"),
            ((x, torch.randn(64, 3)), "3 values .* has 1 per row"),
            ((x, None), "needs labels"),
        ]
        for args, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    g.fit(out, *args, epochs=1)
        self.assertEqual(max_abs(w), 0.0)
        # ... and the graph is still usable afterwards.
        self.assertLess(g.fit(out, x, y, epochs=60, lr=0.1)[0], 0.05)

    def test_losses_without_labels_reject_labels(self):
        x, y = regression_data()
        for loss_type in ("mean", "sum"):
            with self.subTest(loss_type=loss_type):
                g, out, w, b = linear_graph(loss_type=loss_type)
                with self.assertRaisesRegex(RuntimeError, "takes no labels"):
                    g.fit(out, x, y, epochs=1)

    def test_cross_entropy_explains_class_index_labels(self):
        g = OptGraph(loss_type="cross_entropy", nbatch_physical=16, nbatch_logical=16)
        w = torch.zeros(3, 4)
        out = g.linear(g.input((16, 4)), g.param(w))
        with self.assertRaisesRegex(RuntimeError, "one-hot"):
            g.fit(out, torch.randn(64, 4), torch.zeros(64, 1), epochs=1)

    def test_hyperparameters_are_checked(self):
        x, y = regression_data()
        g, out, w, b = linear_graph()
        for kwargs, message in (
            (dict(lr=0.0), "lr must be"),
            (dict(lr=-0.1), "lr must be"),
            (dict(lr=float("nan")), "lr must be"),
            (dict(lr=float("inf")), "lr must be"),
            (dict(wd=-0.5), "weight decay"),
            (dict(wd=1.5), "weight decay"),
            (dict(val_split=1.0), "val_split"),
            (dict(val_split=-0.1), "val_split"),
            (dict(val_split=float("nan")), "val_split"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(RuntimeError, message):
                    g.fit(out, x, y, epochs=1, **kwargs)
        with self.assertRaisesRegex(ValueError, "epochs"):
            g.fit(out, x, y, epochs=-1)
        with self.assertRaises(TypeError):
            g.fit(out, [[1.0, 2.0, 3.0, 4.0]], y)
        self.assertEqual(max_abs(w), 0.0)

    def test_output_must_depend_on_a_parameter(self):
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        x_id = g.input((16, 4))
        w = torch.zeros(1, 4)
        g.param(w)
        with self.assertRaisesRegex(RuntimeError, "does not depend on any parameter"):
            g.fit(g.unary(x_id, "relu"), torch.randn(64, 4), torch.randn(64, 4), epochs=1)

    def test_missing_input_and_unknown_output(self):
        x, y = regression_data()
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        w = torch.zeros(1, 4)
        w_id = g.param(w)
        with self.assertRaisesRegex(RuntimeError, "no input"):
            g.fit(w_id, x, y, epochs=1)
        g, out, w, b = linear_graph()
        for bad in (out + 1, -1, 10_000):
            with self.subTest(output_id=bad):
                with self.assertRaisesRegex(RuntimeError, "does not exist"):
                    g.fit(bad, x, y, epochs=1)

    def test_graph_size_limit_is_an_error_not_an_abort(self):
        # ggml_opt holds forward pass, backward pass and optimizer steps in one fixed-size graph.
        x = torch.randn(8, 3)
        y = torch.randn(8, 3)

        def chain(n):
            g = OptGraph(loss_type="mse", nbatch_physical=4, nbatch_logical=8)
            w = torch.eye(3)
            h = g.linear(g.input((4, 3)), g.param(w))
            for _ in range(n):
                h = g.unary(h, "tanh")
            return g, h

        g, out = chain(115)  # the largest that is accepted
        loss, _ = g.fit(out, x, y, epochs=1, lr=1e-4)
        self.assertTrue(math.isfinite(loss))
        g, out = chain(116)
        with self.assertRaisesRegex(RuntimeError, "too large"):
            g.fit(out, x, y, epochs=1, lr=1e-4)


class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        self.g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        self.x = self.g.input((16, 4))
        self.w = torch.zeros(3, 4)
        self.w_id = self.g.param(self.w)

    def test_unknown_node_ids_raise(self):
        g = self.g
        for call in (
            lambda: g.linear(self.x, 57),
            lambda: g.linear(99, self.w_id),
            lambda: g.linear(self.x, self.w_id, 7),
            lambda: g.linear(self.x, self.w_id, -2),
            lambda: g.unary(12, "relu"),
            lambda: g.add(self.x, 31),
            lambda: g.mul(-1, self.x),
            lambda: g.rms_norm(40),
            lambda: g.rms_norm(self.x, 40),
            lambda: g.softmax(-3),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                call()

    def test_shapes_are_checked_when_the_node_is_added(self):
        g = self.g
        wrong_width = torch.zeros(3, 7)
        with self.assertRaisesRegex(RuntimeError, r"weight \(3, 7\) expects 7 input features but the input \(16, 4\) has 4"):
            g.linear(self.x, g.param(wrong_width))
        vector = torch.zeros(4)
        with self.assertRaisesRegex(RuntimeError, "must be 2-D"):
            g.linear(self.x, g.param(vector))
        bad_bias = torch.zeros(5)
        with self.assertRaisesRegex(RuntimeError, r"bias \(5,\) cannot be broadcast to the output \(16, 3\)"):
            g.linear(self.x, self.w_id, g.param(bad_bias))
        with self.assertRaisesRegex(RuntimeError, "cannot be broadcast"):
            g.add(self.x, g.param(bad_bias))
        with self.assertRaisesRegex(RuntimeError, "cannot be broadcast"):
            g.mul(g.param(bad_bias), self.x)
        with self.assertRaisesRegex(RuntimeError, "cannot be broadcast"):
            g.rms_norm(self.x, g.param(bad_bias))

    def test_only_one_input(self):
        with self.assertRaisesRegex(RuntimeError, "already has an input"):
            self.g.input((16, 4))
        g = OptGraph(nbatch_physical=16, nbatch_logical=16)
        for shape in ((), (16, 0), (1, 2, 3, 4, 5)):
            with self.subTest(shape=shape):
                with self.assertRaises(RuntimeError):
                    g.input(shape)

    def test_parameters_must_be_float32_tensors(self):
        for bad in (torch.zeros(3, 4, dtype=torch.float64), torch.zeros(3, 4, dtype=torch.int64)):
            with self.assertRaisesRegex(RuntimeError, "float32"):
                self.g.param(bad)
        with self.assertRaises(TypeError):
            self.g.param([[0.0] * 4] * 3)

    def test_unary_ops_without_a_backward_pass_are_refused(self):
        # ggml aborts in the backward pass for these; "gelu" used to be offered.
        for op in ("gelu", "erf", "no-such-op"):
            with self.subTest(op=op):
                with self.assertRaisesRegex(ValueError, "supported"):
                    self.g.unary(self.x, op)

    def test_constructor_arguments(self):
        with self.assertRaisesRegex(ValueError, "loss_type"):
            OptGraph(loss_type="hinge")
        with self.assertRaisesRegex(ValueError, "optimizer"):
            OptGraph(optimizer="adam")
        with self.assertRaisesRegex(ValueError, "nbatch_physical"):
            OptGraph(nbatch_physical=0)

    def test_same_tensor_registered_twice_is_one_parameter(self):
        self.assertEqual(self.g.param(self.w), self.w_id)
        other = torch.zeros(3, 4)
        self.assertNotEqual(self.g.param(other), self.w_id)


class TestTrainingIsCorrect(unittest.TestCase):
    def reference_step(self, fn, x, y, w0, b0, lr):
        w = w0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        loss = ((fn(x @ w.t() + b) - y) ** 2).mean()
        loss.backward()
        return float(loss.item()), w0 - lr * w.grad, b0 - lr * b.grad

    def test_every_unary_op_matches_autograd(self):
        # One SGD step on one batch: the reported loss is that of the initial weights, the
        # new weights are w - lr * grad. sigmoid and tanh are assembled from other ggml
        # operations (ggml cannot differentiate its own), so both directions are checked.
        torch.manual_seed(3)
        x = torch.randn(16, 4)
        y = torch.randn(16, 3)
        ops = {
            "relu": torch.relu,
            "silu": torch.nn.functional.silu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "softplus": torch.nn.functional.softplus,
            "exp": torch.exp,
            "square": lambda t: t * t,
            "neg": lambda t: -t,
            "abs": torch.abs,
        }
        lr = 0.1
        for name, fn in ops.items():
            with self.subTest(op=name):
                w0 = torch.randn(3, 4) * 0.5
                b0 = torch.randn(3) * 0.1
                ref_loss, ref_w, ref_b = self.reference_step(fn, x, y, w0, b0, lr)
                w, b = w0.clone(), b0.clone()
                g = OptGraph(loss_type="mse", optimizer="sgd", nbatch_physical=16, nbatch_logical=16)
                out = g.unary(g.linear(g.input((16, 4)), g.param(w), g.param(b)), name)
                loss, _ = g.fit(out, x, y, epochs=1, lr=lr)
                self.assertAlmostEqual(loss, ref_loss, delta=1e-5 * max(1.0, abs(ref_loss)))
                self.assertGreater(max_abs(w - w0), 1e-4)
                self.assertLess(max_abs(w - ref_w), 1e-5)
                self.assertLess(max_abs(b - ref_b), 1e-5)

    def test_log_and_sqrt_match_autograd(self):
        torch.manual_seed(4)
        x = torch.rand(16, 4) + 0.5
        y = torch.randn(16, 3)
        zero_bias = torch.zeros(3)
        for name, fn in (("log", torch.log), ("sqrt", torch.sqrt)):
            with self.subTest(op=name):
                w0 = torch.rand(3, 4) * 0.5 + 0.1
                ref_loss, ref_w, _ = self.reference_step(fn, x, y, w0, zero_bias, 0.01)
                w = w0.clone()
                g = OptGraph(loss_type="mse", optimizer="sgd", nbatch_physical=16, nbatch_logical=16)
                out = g.unary(g.linear(g.input((16, 4)), g.param(w)), name)
                loss, _ = g.fit(out, x, y, epochs=1, lr=0.01)
                self.assertAlmostEqual(loss, ref_loss, delta=1e-5)
                self.assertLess(max_abs(w - ref_w), 1e-5)

    def test_sigmoid_and_tanh_saturate_cleanly(self):
        values = [-60.0, -20.0, -1e-4, 0.0, 1e-4, 3.0, 20.0, 60.0]
        x = torch.tensor([[v] for v in values * 2])
        for name, fn in (("sigmoid", torch.sigmoid), ("tanh", torch.tanh)):
            with self.subTest(op=name):
                w = torch.ones(1, 1)
                g = OptGraph(loss_type="mse", optimizer="sgd", nbatch_physical=16, nbatch_logical=16)
                out = g.unary(g.linear(g.input((16, 1)), g.param(w)), name)
                loss, _ = g.fit(out, x, fn(x), epochs=1, lr=1e-6)
                self.assertLess(loss, 1e-12)  # i.e. the forward values agree with the exact function
                self.assertTrue(math.isfinite(w.item()))

    def test_parameter_that_is_a_view_is_updated_in_place(self):
        # The trained values used to be written to the start of the underlying storage,
        # i.e. into row 0 of `table` here.
        x, y = regression_data()
        table = torch.zeros(3, 4)
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        out = g.linear(g.input((16, 4)), g.param(table[1:2]))
        g.fit(out, x, y, epochs=150, lr=0.1)
        self.assertTrue(table[1:2].allclose(torch.tensor(W_TRUE), atol=0.02))
        self.assertEqual(max_abs(table[0]), 0.0)
        self.assertEqual(max_abs(table[2]), 0.0)

        # A transposed (non-contiguous) parameter: read and written through its strides.
        column = torch.zeros(4, 1)
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        out = g.linear(g.input((16, 4)), g.param(column.t()))
        g.fit(out, x, y, epochs=150, lr=0.1)
        self.assertTrue(column.t().allclose(torch.tensor(W_TRUE), atol=0.02))

    def test_dataset_views_are_read_from_their_own_offset(self):
        # Slices of a larger tensor used to be read from the start of its storage: the
        # zeros here, on which nothing can be learned.
        x, y = regression_data()
        padded_x = torch.cat([torch.zeros(64, 4), x])
        padded_y = torch.cat([torch.zeros(64, 1), y])
        g, out, w, b = linear_graph()
        loss, _ = g.fit(out, padded_x[64:], padded_y[64:], epochs=150, lr=0.1)
        self.assertLess(loss, 1e-3)
        self.assertTrue(w.allclose(torch.tensor(W_TRUE), atol=0.02))

        # Other dtypes are converted rather than reinterpreted.
        g, out, w, b = linear_graph()
        loss, _ = g.fit(out, x.to(torch.float64), y.to(torch.float64), epochs=150, lr=0.1)
        self.assertLess(loss, 1e-3)

    def test_graph_keeps_parameters_alive(self):
        # The graph used to hold a bare pointer to the caller's tensor.
        x, y = regression_data()
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        out = g.linear(g.input((16, 4)), g.param(torch.zeros(1, 4)), g.param(torch.zeros(1)))
        gc.collect()
        clutter = [torch.randn(1, 4) for _ in range(64)]
        first, _ = g.fit(out, x, y, epochs=1, lr=0.1)
        later, _ = g.fit(out, x, y, epochs=80, lr=0.1)
        self.assertLess(later, first * 0.01)
        del clutter

    def test_weight_sharing(self):
        # y = W (W x): registering W twice must train one tensor, not two copies of it.
        torch.manual_seed(7)
        x = torch.randn(64, 2)
        target = torch.tensor([[1.5, 0.0], [0.0, 0.5]])
        y = x @ (target @ target).t()
        w = torch.eye(2)
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        h = g.linear(g.input((16, 2)), g.param(w))
        out = g.linear(h, g.param(w))
        loss, _ = g.fit(out, x, y, epochs=300, lr=0.02)
        self.assertLess(loss, 1e-4)
        self.assertTrue((w @ w).allclose(target @ target, atol=0.02))

    def test_losses_optimizers_and_accumulation(self):
        x, y = regression_data()
        for optimizer, logical, lr, epochs in (("adamw", 16, 0.1, 120), ("adamw", 32, 0.1, 200), ("sgd", 16, 0.05, 200), ("sgd", 32, 0.1, 200)):
            with self.subTest(optimizer=optimizer, nbatch_logical=logical):
                g, out, w, b = linear_graph(optimizer=optimizer, nbatch_logical=logical)
                train, val = g.fit(out, x, y, epochs=epochs, lr=lr, val_split=0.25)
                self.assertLess(train, 1e-3)
                self.assertLess(val, 1e-3)
                self.assertGreater(val, 0.0)
                self.assertTrue(w.allclose(torch.tensor(W_TRUE), atol=0.05))

        # "sum" and "mean" minimise the output itself: no labels involved.
        for loss_type in ("mean", "sum"):
            with self.subTest(loss_type=loss_type):
                g = OptGraph(loss_type=loss_type, optimizer="sgd", nbatch_physical=16, nbatch_logical=16)
                w = torch.full((1, 4), 0.5)
                out = g.unary(g.linear(g.input((16, 4)), g.param(w)), "square")
                first, _ = g.fit(out, x, epochs=1, lr=0.01)
                last, _ = g.fit(out, x, epochs=30, lr=0.01)
                self.assertLess(last, first * 0.5)

    def test_cross_entropy_classifier(self):
        torch.manual_seed(11)
        x = torch.randn(96, 4)
        classes = (x @ torch.randn(4, 3)).argmax(dim=-1).tolist()
        y = torch.zeros(96, 3)
        for row, c in enumerate(classes):
            y[row, int(c)] = 1.0
        g = OptGraph(loss_type="cross_entropy", nbatch_physical=16, nbatch_logical=16)
        w1, b1 = torch.randn(16, 4) * 0.2, torch.zeros(16)
        w2, b2 = torch.randn(3, 16) * 0.2, torch.zeros(3)
        hidden = g.unary(g.linear(g.input((16, 4)), g.param(w1), g.param(b1)), "tanh")
        out = g.linear(hidden, g.param(w2), g.param(b2))
        first, _ = g.fit(out, x, y, epochs=1, lr=0.05)
        loss, val = g.fit(out, x, y, epochs=60, lr=0.05, val_split=0.17)
        self.assertLess(loss, 0.35)
        self.assertLess(loss, first)
        self.assertGreater(val, 0.0)
        predicted = (torch.tanh(x @ w1.t() + b1) @ w2.t() + b2).argmax(dim=-1).tolist()
        accuracy = sum(int(p) == int(c) for p, c in zip(predicted, classes)) / len(classes)
        self.assertGreater(accuracy, 0.85)

    def test_add_mul_rms_norm_and_softmax_nodes(self):
        torch.manual_seed(13)
        x = torch.randn(64, 4)
        true_scale = torch.tensor([1.0, 2.0, -1.0, 0.5])
        true_shift = torch.tensor([0.3, -0.2, 0.1, 0.0])
        y = torch.softmax(x * true_scale + true_shift, dim=-1)

        scale = torch.ones(4)
        shift = torch.zeros(4)
        g = OptGraph(loss_type="mse", nbatch_physical=16, nbatch_logical=16)
        x_id = g.input((16, 4))
        # The smaller operand may come first: add and mul commute.
        out = g.softmax(g.add(g.param(shift), g.mul(g.param(scale), x_id)))
        loss, _ = g.fit(out, x, y, epochs=400, lr=0.05)
        self.assertLess(loss, 1e-6)
        self.assertLess(max_abs(scale - true_scale), 0.05)
        # softmax ignores a constant added to every logit, so the shift is fixed up to one.
        self.assertLess(max_abs((shift - shift.mean()) - (true_shift - true_shift.mean())), 0.05)

        # rms_norm with a learned gain: one SGD step against autograd.
        gain0 = torch.tensor([1.0, 0.5, 2.0, 1.5])
        target = torch.randn(16, 4)
        xb = x[:16]
        gain_ref = gain0.clone().requires_grad_(True)
        normed = xb / torch.sqrt((xb * xb).mean(dim=-1, keepdim=True) + 1e-6)
        ref_loss = ((normed * gain_ref - target) ** 2).mean()
        ref_loss.backward()
        gain = gain0.clone()
        g = OptGraph(loss_type="mse", optimizer="sgd", nbatch_physical=16, nbatch_logical=16)
        out = g.rms_norm(g.input((16, 4)), g.param(gain), eps=1e-6)
        loss, _ = g.fit(out, xb, target, epochs=1, lr=0.1)
        self.assertAlmostEqual(loss, float(ref_loss.item()), delta=1e-5)
        self.assertLess(max_abs(gain - (gain0 - 0.1 * gain_ref.grad)), 1e-5)

    def test_incomplete_last_batch_is_dropped(self):
        x, y = regression_data(rows=70)
        g, out, w, b = linear_graph()
        loss, _ = g.fit(out, x, y, epochs=150, lr=0.1)
        self.assertLess(loss, 1e-3)


if __name__ == "__main__":
    unittest.main()
