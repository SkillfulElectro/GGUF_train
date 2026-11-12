"""Optimizer state, closures and SequentialLR (BUG-14), and the copy protocols they rely on.

Before: LBFGS kept its curvature history where state_dict() never looked, so a checkpoint
restarted it as steepest descent; six optimizers ran the closure in whatever gradient mode
step() was called in; SequentialLR began training with the learning rate of whichever scheduler
had been constructed last and started every later scheduler at its epoch 1. Copying or pickling
a tensor duplicated its native handle - copy.deepcopy(nn.BatchNorm1d(4)) crashed the process.
"""

import collections
import copy
import gc
import io
import pickle
import unittest

import torch

S = torch.optim.lr_scheduler


def rosenbrock_like(w):
    return ((w[0] - 1.5) ** 4).sum() + ((w[1] + 0.5) ** 2).sum() + (w[0] * w[1] - 0.3) ** 2


def lbfgs(start, **kwargs):
    w = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], **kwargs)

    def closure():
        opt.zero_grad()
        loss = rosenbrock_like(w)
        loss.backward()
        return loss

    return w, opt, closure


class TestLBFGSState(unittest.TestCase):
    CONFIGS = [dict(lr=0.5, max_iter=4), dict(lr=1.0, max_iter=3, line_search_fn="strong_wolfe", history_size=5)]

    def test_state_dict_holds_the_curvature_history(self):
        w, opt, closure = lbfgs([-1.0, 2.0], lr=0.5, max_iter=4)
        opt.step(closure)
        state = opt.state_dict()["state"]
        self.assertEqual(list(state), [0])
        saved = state[0]
        for key in ("old_dirs", "old_stps", "ro", "d", "t", "prev_flat_grad", "prev_loss", "H_diag"):
            self.assertIn(key, saved)
        self.assertGreaterEqual(len(saved["old_dirs"]), 1)
        self.assertEqual(len(saved["old_dirs"]), len(saved["old_stps"]))
        self.assertEqual(len(saved["old_dirs"]), len(saved["ro"]))
        self.assertIsInstance(saved["old_dirs"], list)                 # what torch.save can store
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in saved["old_dirs"]))
        self.assertEqual(tuple(saved["d"].shape), (2,))

    def test_a_checkpoint_resumes_the_same_optimization(self):
        for config in self.CONFIGS:
            with self.subTest(**config):
                w, opt, closure = lbfgs([-1.0, 2.0], **config)
                opt.step(closure)
                buf = io.BytesIO()
                torch.save({"w": w.detach(), "opt": opt.state_dict()}, buf)
                opt.step(closure)
                uninterrupted = w.tolist()

                buf.seek(0)
                ckpt = torch.load(buf)
                w2, opt2, closure2 = lbfgs(ckpt["w"].tolist(), **config)
                opt2.load_state_dict(ckpt["opt"])
                opt2.step(closure2)
                for a, b in zip(w2.tolist(), uninterrupted):
                    self.assertAlmostEqual(a, b, places=12)

                # and the history is what made the difference: without it the step is another one
                w3, opt3, closure3 = lbfgs(ckpt["w"].tolist(), **config)
                opt3.step(closure3)
                self.assertGreater(max(abs(a - b) for a, b in zip(w3.tolist(), uninterrupted)), 1e-6)

    def test_loaded_state_is_a_copy_and_stays_bounded(self):
        w, opt, closure = lbfgs([-1.0, 2.0], lr=0.5, max_iter=6, history_size=2)
        opt.step(closure)
        state = opt.state_dict()
        snapshot = [v.tolist() for v in state["state"][0]["old_dirs"]]
        self.assertLessEqual(len(snapshot), 2)
        w2, opt2, closure2 = lbfgs(w.tolist(), lr=0.5, max_iter=6, history_size=2)
        opt2.load_state_dict(state)
        opt2.step(closure2)
        opt.step(closure)
        self.assertEqual([v.tolist() for v in state["state"][0]["old_dirs"]], snapshot)   # neither optimizer wrote into it
        self.assertLessEqual(len(opt2.state[w2]["old_dirs"]), 2)
        self.assertIsInstance(opt2.state[w2]["old_dirs"], collections.deque)

    def test_converges(self):
        # the minimum, 0.33708786..., found with plain gradient descent on floats
        for config in (dict(lr=1.0, max_iter=50, line_search_fn="strong_wolfe"), dict(lr=0.5, max_iter=20)):
            with self.subTest(**config):
                w, opt, closure = lbfgs([-1.0, 2.0], **config)
                for _ in range(5):
                    opt.step(closure)
                self.assertAlmostEqual(closure().item(), 0.3370878628, places=7)
                self.assertAlmostEqual(w[0].item(), 1.30916, places=3)
                self.assertAlmostEqual(w[1].item(), -0.03952, places=3)


class TestClosures(unittest.TestCase):
    def test_closures_run_with_gradients_enabled(self):
        classes = [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, torch.optim.RMSprop, torch.optim.Adagrad,
                   torch.optim.Adadelta, torch.optim.Adamax, torch.optim.NAdam, torch.optim.RAdam, torch.optim.LBFGS]
        for cls in classes:
            with self.subTest(cls.__name__):
                w = torch.tensor([1.0, -2.0], requires_grad=True)
                opt = cls([w], lr=0.1)

                def closure():
                    opt.zero_grad()
                    loss = (w * w).sum()
                    loss.backward()
                    return loss

                with torch.no_grad():                      # step() is often called from such a block
                    loss = opt.step(closure)
                self.assertTrue(torch.is_grad_enabled())
                self.assertAlmostEqual(loss.item(), 5.0, places=5) if cls is not torch.optim.LBFGS else self.assertLess(loss.item(), 5.0)
                self.assertNotEqual(w.tolist(), [1.0, -2.0])
                self.assertIsNone(opt.step() if cls is not torch.optim.LBFGS else None)


class TestOptimizerStateDict(unittest.TestCase):
    def make(self, cls=torch.optim.Adam, **kwargs):
        w = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
        b = torch.tensor([0.1], requires_grad=True)
        opt = cls([{"params": [w]}, {"params": [b], "lr": 0.01}], lr=0.1, **kwargs)
        return w, b, opt

    def train(self, w, b, opt, steps):
        for _ in range(steps):
            opt.zero_grad()
            ((w * w).sum() + (b - 1.0).pow(2).sum()).backward()
            opt.step()

    def test_round_trip_continues_identically(self):
        for cls in (torch.optim.Adam, torch.optim.RMSprop, torch.optim.SGD):
            with self.subTest(cls.__name__):
                kwargs = {"momentum": 0.9} if cls is not torch.optim.Adam else {}
                w, b, opt = self.make(cls, **kwargs)
                self.train(w, b, opt, 3)
                buf = io.BytesIO()
                torch.save(opt.state_dict(), buf)
                w2, b2, opt2 = self.make(cls, **kwargs)
                with torch.no_grad():
                    w2.copy_(w)
                    b2.copy_(b)
                buf.seek(0)
                opt2.load_state_dict(torch.load(buf))
                self.train(w, b, opt, 2)
                self.train(w2, b2, opt2, 2)
                self.assertEqual(w2.tolist(), w.tolist())
                self.assertEqual(b2.tolist(), b.tolist())

    def test_snapshots_do_not_alias_the_live_state(self):
        w, b, opt = self.make()
        self.train(w, b, opt, 2)
        state = opt.state_dict()
        before = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in state["state"][0].items()}
        self.train(w, b, opt, 2)
        after = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in state["state"][0].items()}
        self.assertEqual(before, after)
        w2, b2, opt2 = self.make()
        opt2.load_state_dict(state)
        self.train(w2, b2, opt2, 2)
        after = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in state["state"][0].items()}
        self.assertEqual(before, after)

    def test_mismatched_groups_are_refused(self):
        w, b, opt = self.make()
        single = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        with self.assertRaises(ValueError):
            single.load_state_dict(opt.state_dict())
        state = opt.state_dict()
        state["param_groups"][0]["params"] = [0, 5, 6]
        with self.assertRaises(ValueError):
            opt.load_state_dict(state)


def sgd(lr=1.0):
    return torch.optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=lr)


class TestSequentialLR(unittest.TestCase):
    def lrs(self, opt, sched, steps):
        out = [opt.param_groups[0]["lr"]]
        for _ in range(steps):
            opt.step()
            sched.step()
            out.append(opt.param_groups[0]["lr"])
        return out

    def assertSchedule(self, got, want):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=9)

    def test_warmup_then_decay(self):
        opt = sgd()
        warmup = S.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=3)
        decay = S.ExponentialLR(opt, gamma=0.5)
        seq = S.SequentialLR(opt, [warmup, decay], milestones=[3])
        # epoch 0: the first scheduler's start, not the last-constructed scheduler's
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.1, places=9)
        self.assertSchedule(seq.get_last_lr(), [0.1])
        # at the milestone the second scheduler starts at its epoch 0
        self.assertSchedule(self.lrs(opt, seq, 6), [0.1, 0.4, 0.7, 1.0, 0.5, 0.25, 0.125])
        self.assertSchedule(seq.get_last_lr(), [0.125])
        self.assertIsInstance(seq, S.LRScheduler)

    def test_three_schedulers(self):
        opt = sgd(lr=2.0)
        seq = S.SequentialLR(
            opt,
            [S.ConstantLR(opt, factor=0.5, total_iters=10), S.StepLR(opt, step_size=2, gamma=0.1), S.ConstantLR(opt, factor=0.25, total_iters=1)],
            milestones=[2, 6],
        )
        want = [1.0, 1.0,                      # constant 0.5 x 2.0
                2.0, 2.0, 0.2, 0.2,            # StepLR epochs 0..3
                0.5, 2.0, 2.0]                 # ConstantLR(0.25) epoch 0, then its factor ends
        self.assertSchedule(self.lrs(opt, seq, 8), want)

    def test_matches_the_schedulers_run_one_after_another(self):
        def reference():
            opt = sgd()
            first = S.CosineAnnealingLR(opt, T_max=4, eta_min=0.1)
            out = [opt.param_groups[0]["lr"]]
            for _ in range(3):
                first.step()
                out.append(opt.param_groups[0]["lr"])
            opt2 = sgd()
            second = S.StepLR(opt2, step_size=2, gamma=0.5)
            out.append(opt2.param_groups[0]["lr"])
            for _ in range(4):
                second.step()
                out.append(opt2.param_groups[0]["lr"])
            return out

        opt = sgd()
        seq = S.SequentialLR(opt, [S.CosineAnnealingLR(opt, T_max=4, eta_min=0.1), S.StepLR(opt, step_size=2, gamma=0.5)], milestones=[4])
        self.assertSchedule(self.lrs(opt, seq, 8), reference())

    def test_state_dict_round_trip(self):
        def build():
            opt = sgd()
            seq = S.SequentialLR(opt, [S.LinearLR(opt, start_factor=0.1, total_iters=3), S.ExponentialLR(opt, gamma=0.5)], milestones=[3])
            return opt, seq

        for resume_at in (1, 3, 5):
            with self.subTest(resume_at=resume_at):
                opt, seq = build()
                full = self.lrs(opt, seq, 7)
                opt, seq = build()
                self.lrs(opt, seq, resume_at)
                saved = pickle.loads(pickle.dumps({"sched": seq.state_dict(), "opt": opt.state_dict()}))
                opt2, seq2 = build()
                opt2.load_state_dict(saved["opt"])
                seq2.load_state_dict(saved["sched"])
                self.assertSchedule(self.lrs(opt2, seq2, 7 - resume_at), full[resume_at:])

    def test_arguments(self):
        opt, other = sgd(), sgd()
        a, b = S.ConstantLR(opt), S.ConstantLR(opt)
        cases = {
            "milestones": lambda: S.SequentialLR(opt, [a, b], milestones=[]),
            "too many milestones": lambda: S.SequentialLR(opt, [a, b], milestones=[1, 2]),
            "descending milestones": lambda: S.SequentialLR(opt, [a, b, S.ConstantLR(opt)], milestones=[4, 2]),
            "no schedulers": lambda: S.SequentialLR(opt, [], milestones=[]),
            "another optimizer": lambda: S.SequentialLR(opt, [a, S.ConstantLR(other)], milestones=[1]),
            "plateau": lambda: S.SequentialLR(opt, [a, S.ReduceLROnPlateau(opt)], milestones=[1]),
        }
        for name, build in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    build()


class TestTensorCopyProtocols(unittest.TestCase):
    def churn(self):
        gc.collect()
        return [torch.randn(64, 64) for _ in range(20)]

    def test_deepcopy_is_an_independent_tensor(self):
        a = torch.arange(6.0).reshape(2, 3)
        a.requires_grad_(True)
        a.grad = torch.ones(2, 3)
        b = copy.deepcopy(a)
        self.assertNotEqual(a._ptr, b._ptr)
        self.assertTrue(b.requires_grad and b.is_leaf)
        self.assertEqual(b.grad.tolist(), [[1.0] * 3] * 2)
        self.assertIsNot(b.grad, a.grad)
        with torch.no_grad():
            a.mul_(2.0)
        self.assertEqual(b.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        del a
        self.churn()
        self.assertEqual(b.sum().item(), 15.0)
        with self.assertRaises(RuntimeError):
            copy.deepcopy(b * 2.0)                     # not a leaf
        twice = [b, b]
        c = copy.deepcopy(twice)
        self.assertIs(c[0], c[1])

    def test_shallow_copy_shares_the_data(self):
        a = torch.arange(4.0)
        b = copy.copy(a)
        self.assertIsNot(a, b)
        self.assertNotEqual(a._ptr, b._ptr)
        a[0] = 9.0
        self.assertEqual(b.tolist(), [9.0, 1.0, 2.0, 3.0])
        del a
        self.churn()
        self.assertEqual(b.tolist(), [9.0, 1.0, 2.0, 3.0])

    def test_pickle_round_trip(self):
        tensors = {
            "float": torch.arange(6.0).reshape(2, 3),
            "view": torch.arange(10)[1::3],
            "bool": torch.tensor([True, False]),
            "scalar": torch.tensor(2.5, dtype=torch.float64),
            "empty": torch.zeros(0, 3),
            "grad": torch.ones(2, requires_grad=True),
            "param": torch.nn.Parameter(torch.ones(2)),
        }
        back = pickle.loads(pickle.dumps(tensors))
        self.churn()
        for name, t in tensors.items():
            with self.subTest(name):
                self.assertEqual(back[name].tolist(), t.tolist())
                self.assertEqual(back[name].dtype, t.dtype)
                self.assertEqual(tuple(back[name].shape), tuple(t.shape))
                self.assertEqual(back[name].requires_grad, t.requires_grad)
        self.assertIsInstance(back["param"], torch.nn.Parameter)

    def test_torch_save_of_other_containers(self):
        payload = {"history": collections.deque([torch.ones(2), torch.zeros(2)], maxlen=4), "od": collections.OrderedDict(a=torch.arange(3))}
        buf = io.BytesIO()
        torch.save(payload, buf)
        buf.seek(0)
        back = torch.load(buf)
        self.churn()
        self.assertEqual([t.tolist() for t in back["history"]], [[1.0, 1.0], [0.0, 0.0]])
        self.assertEqual(back["od"]["a"].tolist(), [0, 1, 2])

    def test_deepcopy_of_a_module_with_buffers(self):
        bn = torch.nn.BatchNorm1d(4)
        bn(torch.randn(8, 4))
        clone = copy.deepcopy(bn)                      # used to end the process
        self.assertNotEqual(clone.running_mean._ptr, bn.running_mean._ptr)
        self.assertEqual(clone.running_mean.tolist(), bn.running_mean.tolist())
        self.assertEqual(clone.num_batches_tracked.item(), 1)
        bn(torch.randn(8, 4))
        self.assertNotEqual(clone.running_mean.tolist(), bn.running_mean.tolist())
        del bn
        self.churn()
        self.assertEqual(len(clone.running_var.tolist()), 4)
        ema = copy.deepcopy(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)))
        self.assertEqual(tuple(ema(torch.randn(8, 4)).shape), (8, 4))


if __name__ == "__main__":
    unittest.main()
