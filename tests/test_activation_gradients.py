"""First and second derivatives of the activation functions (BUG-11).

A backward that calls a native "derivative" kernel returns a constant: fine for a plain
backward(), wrong the moment the gradient is differentiated again (gradient penalties,
Hessian-vector products, MAML). And a derivative written as log(exp(x) + 1) turns into inf,
then nan, for inputs a float32 exp() cannot hold.
"""

import math
import unittest

import torch
import torch.nn.functional as F


ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "mish": F.mish,
    "softplus": F.softplus,
    "elu": F.elu,
    "celu": F.celu,
    "selu": F.selu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "logsigmoid": F.logsigmoid,
    "hardswish": F.hardswish,
    "softsign": F.softsign,
    "tanhshrink": F.tanhshrink,
    "erf": torch.erf,
    "exp": torch.exp,
    "sin": torch.sin,
    "atan": torch.atan,
}


def first_derivative(fn, value, dtype=torch.float64):
    x = torch.tensor([value], dtype=dtype, requires_grad=True)
    (g,) = torch.autograd.grad(fn(x).sum(), x, create_graph=True)
    return x, g


class TestSecondDerivatives(unittest.TestCase):
    def test_second_derivative_matches_a_finite_difference_of_the_first(self):
        eps = 1e-5
        for name, fn in ACTIVATIONS.items():
            for value in (-2.3, -0.7, 0.4, 1.9):
                with self.subTest(activation=name, x=value):
                    x, g = first_derivative(fn, value)
                    self.assertTrue(g.requires_grad, "the first derivative is a constant: it has no graph")
                    (h,) = torch.autograd.grad(g.sum(), x)
                    numeric = (first_derivative(fn, value + eps)[1].item() - first_derivative(fn, value - eps)[1].item()) / (2 * eps)
                    self.assertAlmostEqual(h.item(), numeric, places=5)

    def test_first_derivative_matches_a_finite_difference_of_the_function(self):
        eps = 1e-6
        for name, fn in ACTIVATIONS.items():
            for value in (-2.3, -0.7, 0.4, 1.9):
                with self.subTest(activation=name, x=value):
                    _, g = first_derivative(fn, value)
                    hi = fn(torch.tensor([value + eps], dtype=torch.float64)).item()
                    lo = fn(torch.tensor([value - eps], dtype=torch.float64)).item()
                    self.assertAlmostEqual(g.item(), (hi - lo) / (2 * eps), places=6)

    def test_with_and_without_create_graph_agree(self):
        # two code paths per activation: differentiable ops, or one native derivative kernel
        for name, fn in ACTIVATIONS.items():
            for dtype in (torch.float32, torch.float64):
                with self.subTest(activation=name, dtype=str(dtype)):
                    values = [-6.0, -2.3, -0.7, 0.0, 0.4, 1.9, 6.0]
                    x = torch.tensor(values, dtype=dtype, requires_grad=True)
                    (with_graph,) = torch.autograd.grad(fn(x).sum(), x, create_graph=True)
                    (plain,) = torch.autograd.grad(fn(x).sum(), x)
                    for a, b in zip(with_graph.tolist(), plain.tolist()):
                        self.assertAlmostEqual(a, b, places=5)


class TestMish(unittest.TestCase):
    def test_derivatives_stay_finite_for_large_inputs(self):
        x = torch.tensor([-200.0, -1.0, 0.0, 1.0, 50.0, 100.0, 1000.0], requires_grad=True)
        y = F.mish(x)
        (first,) = torch.autograd.grad(y.sum(), x, create_graph=True)
        first.sum().backward()
        second = x.grad.tolist()
        self.assertTrue(all(math.isfinite(v) for v in y.tolist() + first.tolist() + second), (first.tolist(), second))
        # mish'(x) -> 1 and mish''(x) -> 0 for large x; log(exp(x) + 1) used to give nan there
        self.assertEqual(first.tolist()[4:], [1.0, 1.0, 1.0])
        self.assertEqual(second[4:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(first.tolist()[2], 0.6, places=5)
        self.assertAlmostEqual(second[2], 0.64, places=5)

    def test_a_gradient_penalty_on_large_activations_trains(self):
        w = torch.tensor([120.0], requires_grad=True)
        (g,) = torch.autograd.grad(F.mish(w).sum(), w, create_graph=True)
        ((g - 0.5) ** 2).sum().backward()
        self.assertTrue(math.isfinite(w.grad.item()), w.grad.tolist())


if __name__ == "__main__":
    unittest.main()
