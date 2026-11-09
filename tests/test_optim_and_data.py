"""Comprehensive tests for optimizers, LR schedulers, AMP, DataLoader, distributions, and profiler."""

import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.distributions import Bernoulli, Categorical, Exponential, Laplace, Normal, Uniform, kl_divergence
from torch.optim import lr_scheduler as lrs
from torch.profiler import ProfilerActivity, profile, record_function
from torch.testing import assert_close
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset, random_split


class TestOptimAndData(unittest.TestCase):
    def test_optimizers_convergence(self):
        torch.manual_seed(42)
        X = torch.randn(32, 4)
        w_true = torch.tensor([[1.5], [-2.0], [0.5], [3.0]])
        y = X @ w_true

        for opt_fn, steps in [
            (lambda p: optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True), 80),
            (lambda p: optim.Adam(p, lr=0.1), 80),
            (lambda p: optim.AdamW(p, lr=0.1, weight_decay=1e-4), 80),
            (lambda p: optim.RMSprop(p, lr=0.05), 80),
            (lambda p: optim.Adagrad(p, lr=0.2), 80),
            (lambda p: optim.Adadelta(p, lr=1.0, eps=1e-3), 150),
            (lambda p: optim.Adamax(p, lr=0.1), 80),
            (lambda p: optim.NAdam(p, lr=0.1), 80),
            (lambda p: optim.RAdam(p, lr=0.1), 80),
        ]:
            lin = nn.Linear(4, 1, bias=False)
            opt = opt_fn(lin.parameters())
            init_loss = float(((lin(X) - y) ** 2).mean().item())
            for _ in range(steps):
                opt.zero_grad()
                loss = ((lin(X) - y) ** 2).mean()
                loss.backward()
                opt.step()
            final_loss = float(((lin(X) - y) ** 2).mean().item())
            self.assertLess(final_loss, init_loss * 0.1)

        # LBFGS closure
        lin = nn.Linear(4, 1, bias=False)
        lbfgs = optim.LBFGS(lin.parameters(), lr=0.5)
        init_loss = float(((lin(X) - y) ** 2).mean().item())
        for _ in range(10):
            def closure():
                lbfgs.zero_grad()
                loss = ((lin(X) - y) ** 2).mean()
                loss.backward()
                return loss
            lbfgs.step(closure)
        final_loss = float(((lin(X) - y) ** 2).mean().item())
        self.assertLess(final_loss, init_loss * 0.05)

    def test_lr_schedulers_and_amp(self):
        p = nn.Parameter(torch.ones(2, 2))
        opt = optim.AdamW([p], lr=0.1)
        scheds = [
            lrs.StepLR(opt, step_size=2, gamma=0.5),
            lrs.MultiStepLR(opt, milestones=[2, 4], gamma=0.5),
            lrs.ExponentialLR(opt, gamma=0.9),
            lrs.CosineAnnealingLR(opt, T_max=10),
            lrs.CosineAnnealingWarmRestarts(opt, T_0=5),
            lrs.OneCycleLR(opt, max_lr=0.2, total_steps=10),
            lrs.LinearLR(opt, start_factor=0.1, total_iters=5),
            lrs.LambdaLR(opt, lr_lambda=lambda e: 0.95 ** e),
        ]
        for s in scheds:
            for _ in range(3):
                opt.step()
                s.step()
            self.assertGreater(opt.param_groups[0]["lr"], 0.0)

        rp = lrs.ReduceLROnPlateau(optim.SGD([p], lr=0.1), mode="min", patience=1)
        rp.step(1.0)
        rp.step(1.1)
        rp.step(1.2)

        # AMP autocast + GradScaler
        lin = nn.Linear(4, 2)
        opt2 = optim.AdamW(lin.parameters(), lr=1e-2)
        scaler = GradScaler("cpu", enabled=True)
        with autocast("cpu", dtype=torch.bfloat16):
            loss = lin(torch.randn(4, 4)).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(opt2)
        scaler.step(opt2)
        scaler.update()

    def test_datasets_and_dataloader(self):
        X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        y = torch.arange(10, dtype=torch.int64)
        ds = TensorDataset(X, y)
        s1, s2 = random_split(ds, [7, 3])
        self.assertEqual(len(s1), 7)
        self.assertEqual(len(s2), 3)
        cds = ConcatDataset([s1, s2])
        self.assertEqual(len(cds), 10)
        sub = Subset(ds, [0, 2, 4])
        self.assertEqual(len(sub), 3)

        dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=False)
        batches = list(dl)
        self.assertEqual(len(batches), 3)
        self.assertEqual(sum(xb.shape[0] for xb, _ in batches), 10)

    def test_distributions_profiler_and_testing(self):
        torch.manual_seed(9)
        n1 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        n2 = Normal(torch.tensor([1.0]), torch.tensor([2.0]))
        smp = n1.rsample((50,))
        self.assertEqual(smp.shape, (50, 1))
        kl = kl_divergence(n1, n2)
        # KL(N(0,1) || N(1,2)) = log(2) + (1 + 1)/8 - 0.5 = log(2) - 0.25
        import math
        assert_close(kl, torch.tensor([math.log(2.0) - 0.25]), atol=1e-4, rtol=1e-4)

        cat = Categorical(probs=torch.tensor([0.2, 0.5, 0.3]))
        self.assertEqual(cat.sample((4,)).shape, (4,))
        self.assertGreater(float(cat.entropy().item()), 0.0)
        self.assertEqual(Uniform(0.0, 2.0).sample((3,)).shape, (3,))
        self.assertEqual(Bernoulli(0.5).sample((3,)).shape, (3,))
        self.assertEqual(Exponential(2.0).sample((3,)).shape, (3,))
        self.assertEqual(Laplace(0.0, 1.0).sample((3,)).shape, (3,))

        lin = nn.Linear(4, 4)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("forward"):
                y = lin(torch.randn(8, 4))
            with record_function("backward"):
                y.sum().backward()
        table = prof.key_averages().table()
        self.assertIn("forward", table)
        self.assertIn("backward", table)


if __name__ == "__main__":
    unittest.main()
