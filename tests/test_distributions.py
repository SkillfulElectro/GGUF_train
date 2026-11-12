"""torch.distributions: batch_shape / event_shape, broadcast parameters, KL at the ends of [0, 1]."""

import itertools
import math
import unittest

import torch
from torch.distributions import Bernoulli, Categorical, Distribution, Exponential, Laplace, Normal, Uniform, kl_divergence


def flat(t):
    return t.reshape(-1).tolist()


class Base(unittest.TestCase):
    def assert_close(self, got, want, tol=1e-5):
        got = flat(got) if isinstance(got, torch.Tensor) else list(got)
        want = flat(want) if isinstance(want, torch.Tensor) else list(want)
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            if math.isinf(w) or math.isnan(w):
                self.assertEqual(str(g), str(w))
            else:
                self.assertLessEqual(abs(g - w), tol * max(1.0, abs(w)), f"{g} != {w}")


class TestShapes(Base):
    def test_scalar_parameters(self):
        for dist in (Normal(0.0, 1.0), Uniform(0.0, 2.0), Laplace(0.0, 1.0), Exponential(2.0), Bernoulli(0.5),
                     Bernoulli(logits=0.0), Categorical(probs=torch.ones(3)), Categorical(logits=torch.zeros(3))):  # fmt: skip
            with self.subTest(dist=type(dist).__name__):
                self.assertEqual(dist.batch_shape, ())
                self.assertEqual(dist.event_shape, ())
                self.assertIsInstance(dist.batch_shape, torch.Size)
                self.assertIsInstance(dist.event_shape, torch.Size)
                self.assertEqual(dist.sample().shape, ())
                self.assertEqual(dist.sample((4,)).shape, (4,))
                self.assertEqual(dist.sample((2, 3)).shape, (2, 3))
                self.assertEqual(dist.sample(torch.Size([5])).shape, (5,))

    def test_scalar_parameters_are_tensors(self):
        # they were kept as Python floats (or whatever was passed in)
        n = Normal(0, 1)
        for value in (n.loc, n.scale, Uniform(0.0, 2.0).low, Uniform(0, 2).high, Laplace(0.0, 1.0).loc,
                      Laplace(0.0, 1).scale, Exponential(2.0).rate, Bernoulli(0.5).probs, Bernoulli(0.5).logits):  # fmt: skip
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.shape, ())
            self.assertEqual(value.dtype, torch.get_default_dtype())
        self.assertEqual(Exponential(2).rate.item(), 2.0)

    def test_tensor_parameters(self):
        self.assertEqual(Normal(torch.zeros(2, 3), torch.ones(2, 3)).batch_shape, (2, 3))
        self.assertEqual(Exponential(torch.ones(4)).batch_shape, (4,))
        self.assertEqual(Bernoulli(torch.full((2, 5), 0.5)).batch_shape, (2, 5))
        self.assertEqual(Bernoulli(logits=torch.zeros(7)).batch_shape, (7,))
        cat = Categorical(probs=torch.ones(2, 4, 3))
        self.assertEqual(cat.batch_shape, (2, 4))               # the last dimension runs over the categories
        self.assertEqual(cat.event_shape, ())
        self.assertEqual(Categorical(logits=torch.zeros(5, 3)).batch_shape, (5,))

    def test_parameters_broadcast(self):
        loc, scale = torch.zeros(3, 1), torch.ones(1, 4)
        for cls in (Normal, Laplace):
            with self.subTest(cls=cls.__name__):
                dist = cls(loc, scale)
                self.assertEqual(dist.batch_shape, (3, 4))
                self.assertEqual(dist.loc.shape, (3, 4))
                self.assertEqual(dist.scale.shape, (3, 4))
                self.assertEqual(dist.sample().shape, (3, 4))
                self.assertEqual(dist.rsample((5,)).shape, (5, 3, 4))
                self.assertEqual(dist.sample((2, 2)).shape, (2, 2, 3, 4))
                self.assertEqual(dist.log_prob(torch.zeros(3, 4)).shape, (3, 4))
                self.assertEqual(dist.log_prob(torch.tensor(0.5)).shape, (3, 4))
                self.assertEqual(dist.log_prob(torch.zeros(6, 3, 4)).shape, (6, 3, 4))
                self.assertEqual(dist.entropy().shape, (3, 4))
                self.assertEqual(dist.mean.shape, (3, 4))
                self.assertEqual(dist.variance.shape, (3, 4))
                self.assertEqual(dist.stddev.shape, (3, 4))
        uni = Uniform(torch.zeros(3, 1), torch.ones(1, 4) * 2.0)
        self.assertEqual(uni.batch_shape, (3, 4))
        self.assertEqual((uni.low.shape, uni.high.shape), ((3, 4), (3, 4)))
        self.assertEqual(uni.sample((5,)).shape, (5, 3, 4))
        self.assertEqual(uni.log_prob(torch.tensor(1.0)).shape, (3, 4))
        self.assertEqual(uni.entropy().shape, (3, 4))

    def test_a_number_takes_the_shape_dtype_and_values_it_should(self):
        dist = Normal(torch.zeros(3, dtype=torch.float64), 2.0)
        self.assertEqual(dist.scale.dtype, torch.float64)
        self.assertEqual(dist.scale.tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(dist.batch_shape, (3,))
        self.assertEqual(dist.sample().dtype, torch.float64)
        dist = Normal(1, torch.ones(2, 2))
        self.assertEqual(dist.loc.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(Uniform(0.0, torch.tensor([1.0, 2.0])).batch_shape, (2,))
        self.assertEqual(Laplace(torch.zeros(2), 3.0).scale.tolist(), [3.0, 3.0])
        # integer tensors become floating point instead of failing in the sampler
        dist = Normal(torch.tensor([0, 1]), 1)
        self.assertEqual(dist.loc.dtype, torch.get_default_dtype())
        self.assertEqual(dist.sample().dtype, torch.get_default_dtype())
        for cls, args in ((Uniform, (torch.zeros(2, dtype=torch.float64), 1.0)), (Laplace, (torch.zeros(2, dtype=torch.float64), 1.0)),
                          (Exponential, (torch.ones(2, dtype=torch.float64),)), (Bernoulli, (torch.full((2,), 0.5, dtype=torch.float64),))):  # fmt: skip
            self.assertEqual(cls(*args).sample((3,)).dtype, torch.float64)

    def test_parameters_that_do_not_broadcast(self):
        for cls in (Normal, Uniform, Laplace):
            with self.assertRaises(RuntimeError):
                cls(torch.zeros(3), torch.ones(4))
        for cls in (Bernoulli, Categorical):
            with self.assertRaises(ValueError):
                cls()
            with self.assertRaises(ValueError):
                cls(probs=torch.ones(2), logits=torch.zeros(2))
        with self.assertRaises(ValueError):
            Categorical(probs=torch.tensor(1.0))

    def test_base_class(self):
        self.assertEqual(Distribution().batch_shape, ())
        self.assertEqual(Distribution((2,), (3,)).event_shape, (3,))

        class Custom(Distribution):               # never calls Distribution.__init__
            def __init__(self):
                self.rate = torch.ones(2)

            def rsample(self, sample_shape=()):
                return torch.zeros(tuple(sample_shape) + (2,), requires_grad=True) * 1.0

        self.assertEqual(Custom().batch_shape, ())
        self.assertFalse(Custom().sample((3,)).requires_grad)
        with self.assertRaises(NotImplementedError):
            Distribution().sample()
        with self.assertRaises(NotImplementedError):
            Bernoulli(0.5).rsample()

    def test_accepts_validate_args(self):
        self.assertEqual(Normal(0.0, 1.0, validate_args=False).batch_shape, ())
        self.assertEqual(Bernoulli(0.5, validate_args=False).batch_shape, ())
        self.assertIn("Normal(loc:", repr(Normal(0.0, 1.0)))


class TestValues(Base):
    def test_normal(self):
        loc = [0.0, 1.0, 2.0]
        scale = [1.0, 2.0, 3.0, 4.0]
        value = [0.3, -1.0, 2.0, 0.0]
        dist = Normal(torch.tensor(loc).reshape(3, 1), torch.tensor(scale).reshape(1, 4))
        want = [-((v - m) ** 2) / (2 * s * s) - math.log(s) - 0.5 * math.log(2 * math.pi) for m in loc for v, s in zip(value, scale)]
        self.assert_close(dist.log_prob(torch.tensor(value)), want)
        self.assert_close(dist.entropy(), [0.5 + 0.5 * math.log(2 * math.pi) + math.log(s) for _ in loc for s in scale])
        self.assert_close(dist.variance, [s * s for _ in loc for s in scale])
        self.assert_close(Normal(0.0, 1.0).entropy(), [1.4189385332046727])
        self.assert_close(Normal(0.0, 1.0).log_prob(0.0), [-0.5 * math.log(2 * math.pi)])

    def test_uniform(self):
        dist = Uniform(0.0, 2.0)
        self.assert_close(dist.log_prob(torch.tensor([-0.1, 0.0, 1.0, 2.0, 2.1])), [-math.inf, -math.log(2), -math.log(2), -math.inf, -math.inf])
        self.assert_close(dist.entropy(), [math.log(2)])
        self.assert_close(dist.mean, [1.0])
        self.assert_close(dist.variance, [1.0 / 3.0])
        dist = Uniform(torch.tensor([0.0, 1.0]), torch.tensor([[2.0], [5.0]]))
        self.assert_close(dist.log_prob(torch.tensor(1.5)), [-math.log(2), -0.0, -math.log(5), -math.log(4)])

    def test_laplace_and_exponential(self):
        dist = Laplace(torch.zeros(2, 1), torch.tensor([1.0, 2.0, 4.0]))
        self.assert_close(dist.entropy(), [1 + math.log(2 * s) for _ in range(2) for s in (1.0, 2.0, 4.0)])
        self.assert_close(dist.log_prob(torch.tensor(0.5)), [-math.log(2 * s) - 0.5 / s for _ in range(2) for s in (1.0, 2.0, 4.0)])
        self.assert_close(dist.variance, [2 * s * s for _ in range(2) for s in (1.0, 2.0, 4.0)])
        dist = Exponential(torch.tensor([1.0, 2.0]))
        self.assert_close(dist.log_prob(torch.tensor(1.0)), [-1.0, math.log(2) - 2])
        self.assert_close(dist.entropy(), [1.0, 1 - math.log(2)])
        self.assert_close(dist.mean, [1.0, 0.5])
        self.assert_close(dist.variance, [1.0, 0.25])

    def test_bernoulli(self):
        dist = Bernoulli(torch.tensor([0.0, 0.25, 1.0]))
        eps = torch.finfo(torch.float32).eps
        # the logits come from probabilities clamped to [eps, 1 - eps]: every value is finite
        self.assert_close(dist.log_prob(torch.tensor([0.0, 1.0, 1.0])), [0.0, math.log(0.25), 0.0])
        self.assert_close(dist.log_prob(torch.tensor([1.0, 0.0, 0.0])), [math.log(eps), math.log(0.75), math.log(eps)], tol=1e-4)
        self.assert_close(dist.entropy(), [0.0, -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)), 0.0])
        self.assert_close(dist.mean, [0.0, 0.25, 1.0])
        self.assert_close(dist.variance, [0.0, 0.1875, 0.0])
        self.assertEqual(dist.sample().tolist()[0], 0.0)
        self.assertEqual(dist.sample().tolist()[2], 1.0)
        from_logits = Bernoulli(logits=torch.tensor([0.0, 2.0, -50.0, 50.0]))
        self.assert_close(from_logits.probs, [0.5, 1 / (1 + math.exp(-2.0)), 0.0, 1.0])
        self.assert_close(from_logits.log_prob(torch.tensor([1.0, 0.0, 1.0, 0.0])), [math.log(0.5), -2.0 - math.log1p(math.exp(-2.0)), -50.0, -50.0])
        self.assert_close(Bernoulli(0.25).logits, [math.log(0.25 / 0.75)])

    def test_categorical(self):
        dist = Categorical(probs=torch.tensor([[2.0, 5.0, 3.0], [1.0, 0.0, 0.0]]))           # normalized by the constructor
        self.assert_close(dist.probs, [0.2, 0.5, 0.3, 1.0, 0.0, 0.0])
        self.assert_close(dist.log_prob(torch.tensor([1, 0])), [math.log(0.5), 0.0])
        self.assertEqual(dist.log_prob(torch.tensor(1)).shape, (2,))                         # one value, every batch entry
        self.assertEqual(dist.log_prob(torch.tensor([[0], [1], [2]])).shape, (3, 2))
        self.assert_close(dist.entropy(), [-(0.2 * math.log(0.2) + 0.5 * math.log(0.5) + 0.3 * math.log(0.3)), 0.0])
        self.assert_close(Categorical(logits=torch.zeros(4)).entropy(), [math.log(4)])
        self.assert_close(Categorical(logits=torch.tensor([1.0, 2.0, 3.0])).probs, [math.exp(v) / (math.e + math.e ** 2 + math.e ** 3) for v in (1, 2, 3)])
        self.assertTrue(math.isfinite(Categorical(logits=torch.tensor([0.0, -math.inf])).entropy().item()))

    def test_categorical_sample_shapes(self):
        dist = Categorical(probs=torch.tensor([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]]))
        self.assertEqual(dist.sample().shape, (2,))
        self.assertEqual(dist.sample((1,)).shape, (1, 2))        # used to lose the sample dimension
        self.assertEqual(dist.sample((5,)).shape, (5, 2))
        self.assertEqual(dist.sample((2, 3)).shape, (2, 3, 2))
        self.assertEqual(dist.sample((0,)).shape, (0, 2))
        draws = dist.sample((50,))
        self.assertEqual(draws.dtype, torch.int64)
        self.assertEqual(set(draws[:, 1].tolist()), {0})          # the second distribution always gives 0
        self.assertTrue(set(draws[:, 0].tolist()) <= {0, 1, 2})
        one_d = Categorical(probs=torch.ones(3))
        self.assertEqual(one_d.sample((4,)).shape, (4,))
        self.assertEqual(one_d.sample((1,)).shape, (1,))


class TestSampling(Base):
    N = 20000

    def setUp(self):
        torch.manual_seed(1234)

    def test_broadcast_entries_draw_their_own_noise(self):
        draws = Normal(torch.zeros(3, 1), torch.ones(1, 4)).sample()
        self.assertEqual(len(set(flat(draws))), 12)
        draws = Uniform(torch.zeros(3, 1), torch.ones(1, 4)).sample()
        self.assertEqual(len(set(flat(draws))), 12)
        draws = Laplace(torch.zeros(3, 1), torch.ones(1, 4)).sample()
        self.assertEqual(len(set(flat(draws))), 12)

    def test_moments(self):
        loc = torch.tensor([[0.0], [5.0]])
        scale = torch.tensor([1.0, 3.0])
        for cls in (Normal, Laplace):
            with self.subTest(cls=cls.__name__):
                dist = cls(loc, scale)
                draws = dist.sample((self.N,))
                self.assertTrue(bool(draws.isfinite().all()))
                self.assert_close(draws.mean(0), dist.mean, tol=0.05)
                self.assert_close(draws.std(0), dist.stddev, tol=0.05)
        dist = Uniform(torch.tensor([0.0, -2.0]), torch.tensor([[1.0], [4.0]]))
        draws = dist.sample((self.N,))
        self.assertTrue(bool(((draws >= dist.low) & (draws <= dist.high)).all()))
        self.assert_close(draws.mean(0), dist.mean, tol=0.05)
        self.assert_close(draws.var(0), dist.variance, tol=0.05)
        dist = Exponential(torch.tensor([0.5, 2.0]))
        draws = dist.sample((self.N,))
        self.assertTrue(bool(((draws >= 0) & draws.isfinite()).all()))
        self.assert_close(draws.mean(0), [2.0, 0.5], tol=0.05)
        self.assert_close(draws.std(0), [2.0, 0.5], tol=0.05)
        dist = Bernoulli(torch.tensor([0.1, 0.5, 0.9]))
        draws = dist.sample((self.N,))
        self.assertEqual(set(flat(draws)), {0.0, 1.0})
        self.assert_close(draws.mean(0), [0.1, 0.5, 0.9], tol=0.02)
        dist = Categorical(probs=torch.tensor([0.2, 0.5, 0.3]))
        draws = dist.sample((self.N,))
        self.assert_close([(draws == k).float().mean().item() for k in range(3)], [0.2, 0.5, 0.3], tol=0.02)

    def test_rsample_is_differentiable_and_sample_is_not(self):
        loc = torch.tensor([0.5, -0.5], requires_grad=True)
        scale = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        for cls in (Normal, Laplace):
            with self.subTest(cls=cls.__name__):
                dist = cls(loc, scale)
                self.assertFalse(dist.sample().requires_grad)
                draws = dist.rsample((4,))
                self.assertTrue(draws.requires_grad)
                loc.grad = scale.grad = None
                draws.sum().backward()
                self.assertEqual(loc.grad.tolist(), [12.0, 12.0])                # 4 samples x 3 scales each
                noise = ((draws.detach() - loc.detach()) / scale.detach()).sum((0, 2))
                self.assert_close(scale.grad, noise, tol=1e-4)
        low = torch.tensor([0.0, 1.0], requires_grad=True)
        high = torch.tensor([[2.0], [5.0]], requires_grad=True)
        draws = Uniform(low, high).rsample((3,))
        draws.sum().backward()
        u = (draws.detach() - low.detach()) / (high.detach() - low.detach())      # draw = low + u (high - low)
        self.assert_close(low.grad, (1.0 - u).sum((0, 1)), tol=1e-4)
        self.assert_close(high.grad, u.sum((0, 2)), tol=1e-4)
        rate = torch.tensor([1.0, 2.0], requires_grad=True)
        draws = Exponential(rate).rsample((5,))
        draws.sum().backward()
        self.assert_close(rate.grad, -(draws.detach().sum(0)) / rate.detach(), tol=1e-4)
        self.assertFalse(Bernoulli(torch.tensor([0.5], requires_grad=True)).sample().requires_grad)

    def test_log_prob_gradients_reach_broadcast_parameters(self):
        loc = torch.tensor([0.5, -0.5], requires_grad=True)
        scale = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        Normal(loc, scale).log_prob(torch.tensor(0.25)).sum().backward()
        self.assert_close(loc.grad, [sum((0.25 - m) / s**2 for s in (1, 2, 3)) for m in (0.5, -0.5)])
        self.assert_close(scale.grad, [sum((0.25 - m) ** 2 / s**3 - 1 / s for m in (0.5, -0.5)) for s in (1, 2, 3)])


def kl_bernoulli(p, q):
    total = 0.0
    for a, b in ((p, q), (1 - p, 1 - q)):
        if a == 0:
            continue                # 0 log 0 = 0
        if b == 0:
            return math.inf
        total += a * math.log(a / b)
    return total


class TestKL(Base):
    def test_bernoulli_at_zero_and_one(self):
        # all of these were nan
        self.assert_close(kl_divergence(Bernoulli(0.0), Bernoulli(0.5)), [math.log(2)])
        self.assert_close(kl_divergence(Bernoulli(1.0), Bernoulli(0.5)), [math.log(2)])
        self.assert_close(kl_divergence(Bernoulli(0.0), Bernoulli(0.0)), [0.0])
        self.assert_close(kl_divergence(Bernoulli(1.0), Bernoulli(1.0)), [0.0])
        # mass where q has none
        self.assertEqual(kl_divergence(Bernoulli(0.5), Bernoulli(0.0)).item(), math.inf)
        self.assertEqual(kl_divergence(Bernoulli(0.5), Bernoulli(1.0)).item(), math.inf)
        self.assertEqual(kl_divergence(Bernoulli(1.0), Bernoulli(0.0)).item(), math.inf)
        self.assertEqual(kl_divergence(Bernoulli(0.0), Bernoulli(1.0)).item(), math.inf)

    def test_bernoulli_grid(self):
        for dtype, probs, tol in (
            (torch.float64, (0.0, 1e-12, 1e-9, 0.1, 0.5, 0.9, 1.0), 1e-12),
            (torch.float32, (0.0, 1e-3, 0.1, 0.5, 0.9, 1.0), 1e-5),
        ):
            for p, q in itertools.product(probs, probs):
                with self.subTest(dtype=dtype, p=p, q=q):
                    got = kl_divergence(Bernoulli(torch.tensor(p, dtype=dtype)), Bernoulli(torch.tensor(q, dtype=dtype)))
                    self.assertEqual(got.dtype, dtype)
                    self.assert_close(got, [kl_bernoulli(p, q)], tol=tol)

    def test_bernoulli_batches(self):
        p = torch.tensor([[0.0], [0.5], [1.0]])
        q = torch.tensor([0.25, 0.5, 1.0, 0.0])
        got = kl_divergence(Bernoulli(p), Bernoulli(q))
        self.assertEqual(got.shape, (3, 4))
        self.assert_close(got, [kl_bernoulli(a, b) for a in (0.0, 0.5, 1.0) for b in (0.25, 0.5, 1.0, 0.0)])
        got = kl_divergence(Bernoulli(logits=torch.tensor([0.0, 2.0, -300.0, 300.0])), Bernoulli(logits=torch.tensor([1.0, 2.0, 0.0, 0.0])))
        self.assert_close(got, [kl_bernoulli(0.5, 1 / (1 + math.exp(-1.0))), 0.0, math.log(2), math.log(2)])

    def test_bernoulli_gradients_are_finite(self):
        p = torch.tensor([0.0, 0.3, 1.0], requires_grad=True)
        q = torch.tensor([0.5, 0.6, 0.5], requires_grad=True)
        kl_divergence(Bernoulli(p), Bernoulli(q)).sum().backward()
        self.assertTrue(all(math.isfinite(v) for v in p.grad.tolist()), p.grad.tolist())
        self.assertTrue(all(math.isfinite(v) for v in q.grad.tolist()), q.grad.tolist())
        self.assertAlmostEqual(p.grad.tolist()[1], math.log(0.3 / 0.7) - math.log(0.6 / 0.4), places=5)
        self.assertAlmostEqual(q.grad.tolist()[1], -0.3 / 0.6 + 0.7 / 0.4, places=5)
        # an infinite divergence does not leave nan behind either
        p = torch.tensor([0.5, 0.5], requires_grad=True)
        kl = kl_divergence(Bernoulli(p), Bernoulli(torch.tensor([0.0, 0.5])))
        self.assertEqual(kl.tolist(), [math.inf, 0.0])
        kl.sum().backward()
        self.assertTrue(all(math.isfinite(v) for v in p.grad.tolist()), p.grad.tolist())

    def test_normal(self):
        self.assert_close(kl_divergence(Normal(0.0, 1.0), Normal(1.0, 2.0)), [math.log(2) - 0.25])
        got = kl_divergence(Normal(torch.zeros(3, 1), 1.0), Normal(torch.tensor([0.0, 1.0, 2.0, 3.0]), 2.0))
        self.assertEqual(got.shape, (3, 4))
        self.assert_close(got, [math.log(2) + (1 + m * m) / 8 - 0.5 for _ in range(3) for m in (0.0, 1.0, 2.0, 3.0)])
        self.assert_close(kl_divergence(Normal(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])), Normal(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))), [0.0, 0.0])

    def test_categorical(self):
        p = Categorical(probs=torch.tensor([0.5, 0.5, 0.0]))
        q = Categorical(probs=torch.tensor([0.25, 0.25, 0.5]))
        self.assert_close(kl_divergence(p, q), [math.log(2)])                     # 0 log 0 = 0
        self.assertEqual(kl_divergence(q, p).item(), math.inf)
        self.assert_close(kl_divergence(p, p), [0.0])
        batch = kl_divergence(Categorical(probs=torch.tensor([[0.5, 0.5], [0.9, 0.1]])), Categorical(probs=torch.tensor([0.5, 0.5])))
        self.assert_close(batch, [0.0, 0.9 * math.log(1.8) + 0.1 * math.log(0.2)])

    def test_uniform_exponential_laplace(self):
        self.assert_close(kl_divergence(Uniform(0.0, 1.0), Uniform(-1.0, 3.0)), [math.log(4)])
        self.assertEqual(kl_divergence(Uniform(0.0, 5.0), Uniform(0.0, 3.0)).item(), math.inf)
        self.assert_close(kl_divergence(Exponential(2.0), Exponential(3.0)), [math.log(2 / 3) + 1.5 - 1])
        self.assert_close(kl_divergence(Laplace(0.0, 1.0), Laplace(1.0, 2.0)), [math.log(2) + 0.5 + 0.5 * math.exp(-1) - 1])
        self.assertEqual(kl_divergence(Laplace(torch.zeros(2, 1), 1.0), Laplace(torch.zeros(3), 1.0)).shape, (2, 3))

    def test_matches_a_monte_carlo_estimate(self):
        torch.manual_seed(7)
        for p, q in ((Normal(0.5, 1.5), Normal(-1.0, 2.0)), (Laplace(0.5, 1.5), Laplace(-1.0, 2.0)),
                     (Exponential(2.0), Exponential(0.7)), (Uniform(0.0, 1.0), Uniform(-1.0, 3.0))):  # fmt: skip
            with self.subTest(dist=type(p).__name__):
                draws = p.sample((40000,))
                estimate = (p.log_prob(draws) - q.log_prob(draws)).mean().item()
                self.assertAlmostEqual(kl_divergence(p, q).item(), estimate, delta=0.03)

    def test_pairs_without_a_formula(self):
        with self.assertRaises(NotImplementedError):
            kl_divergence(Normal(0.0, 1.0), Laplace(0.0, 1.0))


class TestBroadcastHelpers(Base):
    def test_broadcast_tensors(self):
        a, b, c = torch.broadcast_tensors(torch.arange(3.0).reshape(3, 1), torch.arange(4.0).reshape(1, 4), torch.tensor(2.0))
        self.assertEqual((a.shape, b.shape, c.shape), ((3, 4), (3, 4), (3, 4)))
        self.assertEqual(a.tolist()[1], [1.0] * 4)
        self.assertEqual(b.tolist()[2], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(c.tolist()[0], [2.0] * 4)
        same = torch.ones(2, 2)
        self.assertIs(torch.broadcast_tensors(same, torch.zeros(2, 2))[0], same)
        self.assertEqual(torch.broadcast_tensors([torch.zeros(2), torch.zeros(1)])[1].shape, (2,))
        with self.assertRaises(RuntimeError):
            torch.broadcast_tensors(torch.zeros(3), torch.zeros(4))
        with self.assertRaises(TypeError):
            torch.broadcast_tensors(torch.zeros(3), 1.0)
        x = torch.ones(3, 1, requires_grad=True)
        (torch.broadcast_tensors(x, torch.zeros(3, 4))[0] * 2.0).sum().backward()
        self.assertEqual(flat(x.grad), [8.0, 8.0, 8.0])

    def test_broadcast_shapes_and_broadcast_to(self):
        self.assertEqual(torch.broadcast_shapes((3, 1), (1, 4), ()), (3, 4))
        self.assertEqual(torch.broadcast_shapes(2, (3, 1)), (3, 2))
        self.assertEqual(torch.broadcast_shapes((0,), (1,)), (0,))
        self.assertEqual(torch.broadcast_shapes(), ())
        self.assertIsInstance(torch.broadcast_shapes((2,), (2,)), torch.Size)
        with self.assertRaises(RuntimeError):
            torch.broadcast_shapes((3,), (4,))
        with self.assertRaises(RuntimeError):
            torch.broadcast_shapes((0,), (2,))
        self.assertEqual(torch.broadcast_to(torch.ones(3, 1), (2, 3, 4)).shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
