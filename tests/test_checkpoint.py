"""torch.utils.checkpoint: the recomputation replays the random numbers of the forward pass,
checkpoint_sequential cuts the segments PyTorch cuts, and gradients reach everything the
checkpointed function used."""

import collections
import contextlib
import functools
import gc
import unittest
import weakref

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_module
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

GLOBAL_LINEAR = nn.Linear(4, 4)


def uses_a_global(x):
    return GLOBAL_LINEAR(x).tanh()


def calls_a_function_that_uses_a_global(x):
    return uses_a_global(x) * 2.0


class CallableObject:
    def __init__(self):
        self.lin = nn.Linear(4, 4)

    def __call__(self, x):
        return self.lin(x).tanh()


def flat(t):
    return None if t is None else t.reshape(-1).tolist()


class Base(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(3, 4)
        self.xr = torch.randn(3, 4, requires_grad=True)
        self.lin = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.params = list(self.lin.parameters()) + list(self.lin2.parameters())

    def gradients(self, loss_fn, tensors, seed=None):
        for t in tensors:
            t.grad = None
        if seed is not None:
            torch.manual_seed(seed)
        loss = loss_fn()
        loss.backward()
        return loss.item(), [flat(t.grad) for t in tensors]

    def assert_same_gradients(self, plain, checkpointed, tensors, seed=None):
        """Both closures compute a loss; the second one uses checkpoint()."""
        want_loss, want = self.gradients(plain, tensors, seed)
        got_loss, got = self.gradients(checkpointed, tensors, seed)
        self.assertAlmostEqual(got_loss, want_loss, places=5)
        for i, (g, w) in enumerate(zip(got, want)):
            self.assertEqual(g is None, w is None, f"tensor {i}: gradient {'missing' if g is None else 'unexpected'}")
            if g is not None:
                self.assertEqual(len(g), len(w))
                for a, b in zip(g, w):
                    self.assertAlmostEqual(a, b, places=5, msg=f"tensor {i}")
        self.assertTrue(any(w is not None and any(v != 0.0 for v in w) for w in want), "the reference has no gradient at all")


class TestRandomState(Base):
    def test_dropout_draws_the_same_mask_when_recomputed(self):
        drop = nn.Dropout(0.5)
        x = torch.ones(8, 16, requires_grad=True)
        torch.manual_seed(11)
        out = checkpoint(drop, x)
        out.sum().backward()
        kept = [v != 0.0 for v in flat(out)]
        self.assertTrue(any(kept) and not all(kept))
        # d out / d x is 1 / (1 - p) where the mask kept the element and 0 elsewhere: the mask
        # of the recomputation has to be the mask of the forward pass
        self.assertEqual(flat(x.grad), [2.0 if k else 0.0 for k in kept])

    def test_gradients_of_a_stochastic_block(self):
        drop = nn.Dropout(0.3)
        lin, lin2, x = self.lin, self.lin2, self.x
        self.assert_same_gradients(
            lambda: lin2(drop(lin(x))).pow(2).sum(),
            lambda: checkpoint(lambda t: lin2(drop(lin(t))), x).pow(2).sum(),
            self.params,
            seed=5,
        )

    def test_several_random_draws_per_run(self):
        x = torch.ones(4, 8, requires_grad=True)

        def noisy(t):
            return t * torch.rand(4, 8) + t * torch.randn(4, 8) * torch.bernoulli(torch.full((4, 8), 0.5))

        torch.manual_seed(3)
        want = noisy(x)
        want.sum().backward()
        want_grad, x.grad = flat(x.grad), None
        torch.manual_seed(3)
        got = checkpoint(noisy, x)
        got.sum().backward()
        self.assertEqual(flat(got), flat(want))
        self.assertEqual(flat(x.grad), want_grad)

    def test_the_stream_around_the_checkpoint_is_untouched(self):
        drop = nn.Dropout(0.5)
        x = torch.ones(4, 4, requires_grad=True)

        def run(use_checkpoint):
            torch.manual_seed(21)
            out = checkpoint(drop, x) if use_checkpoint else drop(x)
            after_forward = torch.rand(3).tolist()
            out.sum().backward()
            after_backward = torch.rand(3).tolist()
            return after_forward, after_backward

        self.assertEqual(run(True), run(False))

    def test_backward_twice_replays_twice(self):
        drop = nn.Dropout(0.5)
        x = torch.ones(6, 6, requires_grad=True)
        torch.manual_seed(2)
        out = checkpoint(drop, x)
        out.sum().backward(retain_graph=True)
        first, x.grad = flat(x.grad), None
        torch.rand(5)                                  # the surrounding code moves the generator on
        out.sum().backward()
        self.assertEqual(flat(x.grad), first)
        self.assertEqual(first, flat(out))             # out = x * mask / (1 - p) with x = 1

    def test_preserve_rng_state_false(self):
        seen = []

        def noisy(t, **kwargs):
            seen.append(kwargs)
            return t * torch.rand(6, 6)

        x = torch.ones(6, 6, requires_grad=True)
        torch.manual_seed(2)
        out = checkpoint(noisy, x, preserve_rng_state=False)
        out.sum().backward()
        self.assertEqual(seen, [{}, {}])               # the option is not handed to the function
        self.assertNotEqual(flat(x.grad), flat(out))   # new random numbers: not the forward's factors
        # ... and the recomputation's draws are taken from the stream
        torch.manual_seed(2)
        checkpoint(noisy, x, preserve_rng_state=False).sum()
        without_backward = torch.rand(2).tolist()
        torch.manual_seed(2)
        checkpoint(noisy, x, preserve_rng_state=False).sum().backward()
        self.assertNotEqual(torch.rand(2).tolist(), without_backward)

    def test_checkpoint_sequential_with_dropout(self):
        seq = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.4), nn.Linear(4, 4), nn.Dropout(0.4), nn.Linear(4, 4))
        x = self.x
        for segments in (1, 2, 3, 5):
            with self.subTest(segments=segments):
                self.assert_same_gradients(
                    lambda: seq(x).pow(2).sum(),
                    lambda: checkpoint_sequential(seq, segments, x).pow(2).sum(),
                    list(seq.parameters()),
                    seed=9,
                )


class TestCheckpointSequential(Base):
    def segment_sizes(self, count, segments, **kwargs):
        """(sizes of the checkpointed chunks, modules run outside of checkpoint())."""
        log = []

        class Step(nn.Module):
            def forward(self, x):
                log.append("step")
                return x * 1.0

        chunks = []
        original = checkpoint_module.checkpoint

        def spy(function, *args, **kw):
            before = len(log)
            result = original(function, *args, **kw)
            chunks.append(len(log) - before)
            return result

        checkpoint_module.checkpoint = spy
        try:
            checkpoint_sequential([Step() for _ in range(count)], segments, self.xr, **kwargs)
        finally:
            checkpoint_module.checkpoint = original
        return chunks, len(log) - sum(chunks)

    def test_boundaries(self):
        # segments - 1 checkpointed chunks of len // segments functions; the rest runs directly
        for count, segments, chunks, direct in (
            (5, 2, [2], 3),            # used to be three checkpointed chunks: 2, 2, 1
            (5, 3, [1, 1], 3),
            (6, 3, [2, 2], 2),
            (7, 3, [2, 2], 3),
            (4, 4, [1, 1, 1], 1),
            (5, 5, [1, 1, 1, 1], 1),
            (5, 1, [], 5),             # one segment: nothing is checkpointed
            (10, 4, [2, 2, 2], 4),
            (1, 1, [], 1),
        ):
            with self.subTest(count=count, segments=segments):
                self.assertEqual(self.segment_sizes(count, segments), (chunks, direct))

    def test_more_segments_than_functions(self):
        self.assertEqual(self.segment_sizes(3, 5), ([1, 1], 1))
        self.assertEqual(self.segment_sizes(1, 4), ([], 1))
        self.assertIs(checkpoint_sequential([], 2, self.x), self.x)

    def test_arguments(self):
        with self.assertRaisesRegex(ValueError, "Unexpected keyword arguments: scale"):
            checkpoint_sequential([nn.Tanh()], 1, self.x, scale=2.0)
        with self.assertRaises(ValueError):
            checkpoint_sequential([nn.Tanh()], 0, self.x)
        self.assertEqual(self.segment_sizes(5, 2, use_reentrant=False, preserve_rng_state=False), ([2], 3))

    def test_results_and_gradients(self):
        seq = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
        x, xr = self.x, self.xr
        for segments in (1, 2, 3, 4, 5, 9):
            with self.subTest(segments=segments):
                self.assertEqual(flat(checkpoint_sequential(seq, segments, x)), flat(seq(x)))
                self.assert_same_gradients(
                    lambda: seq(xr).pow(2).sum(),
                    lambda: checkpoint_sequential(seq, segments, xr).pow(2).sum(),
                    list(seq.parameters()) + [xr],
                )
        # a plain list of callables
        steps = [self.lin, torch.tanh, self.lin2]
        self.assert_same_gradients(
            lambda: self.lin2(self.lin(x).tanh()).sum(),
            lambda: checkpoint_sequential(steps, 2, x).sum(),
            self.params,
        )


class TestGradientsReachEverything(Base):
    """However the function gets hold of a tensor, the gradient gets back to it."""

    def test_module_bound_method_and_closure(self):
        lin, lin2, x, xr = self.lin, self.lin2, self.x, self.xr
        self.assert_same_gradients(lambda: lin(x).sum(), lambda: checkpoint(lin, x).sum(), self.params[:2])
        self.assert_same_gradients(lambda: lin(xr).pow(2).sum(), lambda: checkpoint(lin, xr).pow(2).sum(), self.params[:2] + [xr])
        self.assert_same_gradients(lambda: lin(x).pow(2).sum(), lambda: checkpoint(lin.forward, x).pow(2).sum(), self.params[:2])
        self.assert_same_gradients(lambda: lin2(lin(x).tanh()).sum(), lambda: checkpoint(lambda t: lin2(lin(t).tanh()), x).sum(), self.params)

    def test_globals_partials_and_callable_objects(self):
        x = self.x
        params = list(GLOBAL_LINEAR.parameters())
        self.assert_same_gradients(lambda: uses_a_global(x).sum(), lambda: checkpoint(uses_a_global, x).sum(), params)
        self.assert_same_gradients(
            lambda: calls_a_function_that_uses_a_global(x).sum(),
            lambda: checkpoint(calls_a_function_that_uses_a_global, x).sum(),
            params,
        )
        lin = self.lin
        partial = functools.partial(lambda module, t: module(t).tanh(), lin)
        self.assert_same_gradients(lambda: lin(x).tanh().sum(), lambda: checkpoint(partial, x).sum(), self.params[:2])
        obj = CallableObject()
        self.assert_same_gradients(lambda: obj(x).sum(), lambda: checkpoint(obj, x).sum(), list(obj.lin.parameters()))

    def test_a_global_module_and_an_input_that_requires_grad(self):
        # the output required grad through the input, and the parameters silently got nothing
        xr = self.xr
        self.assert_same_gradients(
            lambda: uses_a_global(xr).pow(2).sum(),
            lambda: checkpoint(uses_a_global, xr).pow(2).sum(),
            list(GLOBAL_LINEAR.parameters()) + [xr],
        )

    def test_computed_tensors_from_a_closure(self):
        lin, lin2, x, xr = self.lin, self.lin2, self.x, self.xr

        def loss(use_checkpoint, also_outside):
            h = lin(x).tanh()                          # computed outside, used inside
            fn = lambda t: lin2(t) * h                 # noqa: E731
            y = checkpoint(fn, xr) if use_checkpoint else fn(xr)
            return ((y + h) if also_outside else y).pow(2).sum()

        for also_outside in (False, True):
            with self.subTest(also_outside=also_outside):
                self.assert_same_gradients(lambda: loss(False, also_outside), lambda: loss(True, also_outside), self.params + [xr])

    def test_tensor_arguments(self):
        lin, lin2, x, xr = self.lin, self.lin2, self.x, self.xr

        def loss(use_checkpoint, as_keyword):
            h = lin(x).tanh()
            fn = lambda t, scale=None: lin2(t) * scale  # noqa: E731
            if as_keyword:
                y = checkpoint(fn, xr, scale=h) if use_checkpoint else fn(xr, scale=h)
            else:
                y = checkpoint(fn, xr, h) if use_checkpoint else fn(xr, h)
            return y.pow(2).sum()

        for as_keyword in (False, True):
            with self.subTest(as_keyword=as_keyword):
                self.assert_same_gradients(lambda: loss(False, as_keyword), lambda: loss(True, as_keyword), self.params + [xr])

    def test_the_input_used_twice(self):
        # once as the argument, once through a tensor computed from it outside
        lin, lin2, x = self.lin, self.lin2, self.x

        def loss(use_checkpoint):
            a = lin(x)
            h = a.tanh()
            fn = lambda t: lin2(t) * h + t              # noqa: E731
            return (checkpoint(fn, a) if use_checkpoint else fn(a)).pow(2).sum()

        self.assert_same_gradients(lambda: loss(False), lambda: loss(True), self.params)

    def test_chains_and_nesting(self):
        lin, lin2, x = self.lin, self.lin2, self.x
        self.assert_same_gradients(
            lambda: lin2(lin(x).relu()).pow(2).sum(),
            lambda: checkpoint(lin2, checkpoint(lin, x).relu()).pow(2).sum(),
            self.params,
        )
        self.assert_same_gradients(
            lambda: lin2(lin(x).relu()).pow(2).sum(),
            lambda: checkpoint(lambda t: checkpoint(lin2, checkpoint(lin, t).relu()), x).pow(2).sum(),
            self.params,
        )

    def test_autograd_grad(self):
        lin, x = self.lin, self.x
        params = list(lin.parameters())
        want = torch.autograd.grad(lin(x).pow(2).sum(), params)
        got = torch.autograd.grad(checkpoint(lin, x).pow(2).sum(), params)
        for g, w in zip(got, want):
            self.assertEqual(flat(g), flat(w))
        self.assertTrue(all(p.grad is None for p in params))          # nothing accumulated on the side

    def test_second_derivatives(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        w = torch.tensor([2.0], requires_grad=True)
        y = checkpoint(lambda t: (t**3) * w, x).sum()
        (g,) = torch.autograd.grad(y, x, create_graph=True)
        self.assertEqual(flat(g), [6.0, 24.0, 54.0])                  # 3 x^2 w
        g.sum().backward()
        self.assertEqual(flat(x.grad), [12.0, 24.0, 36.0])            # 6 x w
        self.assertEqual(flat(w.grad), [42.0])                        # sum 3 x^2

    def test_parameter_hooks_run_once(self):
        lin, x = self.lin, self.x
        calls = []
        handle = lin.weight.register_hook(lambda g: calls.append(1) or g * 0.5)
        try:
            want = self.gradients(lambda: lin(x).pow(2).sum() + lin(x).sum(), [lin.weight])[1]
            calls.clear()
            got = self.gradients(lambda: checkpoint(lin, x).pow(2).sum() + lin(x).sum(), [lin.weight])[1]
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        for a, b in zip(got[0], want[0]):
            self.assertAlmostEqual(a, b, places=5)


class TestBehaviour(Base):
    def test_outputs(self):
        lin, lin2, x = self.lin, self.lin2, self.x
        self.assert_same_gradients(
            lambda: sum(o.pow(2).sum() for o in (lin(x), lin2(x))),
            lambda: sum(o.pow(2).sum() for o in checkpoint(lambda t: (lin(t), lin2(t)), x)),
            self.params,
        )
        Pair = collections.namedtuple("Pair", "first second")
        out = checkpoint(lambda t: {"a": lin(t), "b": [lin2(t), 7, None], "c": Pair(t, "text")}, x)
        self.assertEqual(sorted(out), ["a", "b", "c"])
        self.assertTrue(out["a"].requires_grad and out["b"][0].requires_grad)
        self.assertEqual(out["b"][1:], [7, None])
        self.assertIsInstance(out["c"], Pair)
        self.assertIs(out["c"].first, x)                 # an argument handed back is itself
        self.assertEqual(out["c"].second, "text")
        self.assertEqual(flat(out["a"]), flat(lin(x)))
        # values that are not floating point are passed through
        values, indices = checkpoint(lambda t: lin(t).max(dim=1), x)
        self.assertTrue(values.requires_grad)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertFalse(indices.requires_grad)

    def test_returning_what_was_given(self):
        lin, x, xr = self.lin, self.x, self.xr
        h = lin(x)
        a, b, c = checkpoint(lambda t: (t, h, lin(t)), xr)
        self.assertIs(a, xr)
        self.assertIs(b, h)
        self.assert_same_gradients(
            lambda: (xr * 2.0 + lin(x) + lin(xr)).sum(),
            lambda: sum(checkpoint(lambda t: (t * 2.0, lin(x), lin(t)), xr)).sum(),
            self.params[:2] + [xr],
        )

    def test_nothing_to_differentiate(self):
        x = self.x
        out = checkpoint(lambda t: t * 2.0, x)
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        with torch.no_grad():
            out = checkpoint(self.lin, self.xr)
        self.assertFalse(out.requires_grad)
        self.assertEqual(checkpoint(lambda t, k=1: k + 1, x, k=4), 5)

    def test_the_function_runs_once_per_pass(self):
        calls = []

        def fn(t):
            calls.append(torch.is_grad_enabled())
            return self.lin(t)

        out = checkpoint(fn, self.x)
        self.assertEqual(calls, [False])
        out.sum().backward(retain_graph=True)
        self.assertEqual(calls, [False, True])
        out.sum().backward()
        self.assertEqual(calls, [False, True, True])

    def test_intermediate_tensors_are_not_kept(self):
        lin, lin2, x = self.lin, self.lin2, self.x
        inside = []

        def fn(t):
            h = lin(t).exp()                             # exp keeps its result for the backward pass
            inside.append(weakref.ref(h))
            return lin2(h)

        gc.collect()
        gc.disable()                                     # freed by reference counting, not by the collector
        try:
            out = checkpoint(fn, x)
            self.assertIsNone(inside[0]())
            plain = fn(x)
            self.assertIsNotNone(inside[1]())            # an ordinary call keeps it for backward
            out.sum().backward()
            self.assertIsNone(inside[2]())               # the recomputed graph is gone as well
            del out, plain
        finally:
            gc.enable()

    def test_the_graph_around_it_survives(self):
        # the recomputation used to be differentiated right through `h` into the graph outside
        # and freed what that graph had saved
        lin, lin2, x, xr = self.lin, self.lin2, self.x, self.xr
        h = lin(x).exp()
        y = checkpoint(lambda t: lin2(t) * h, xr)
        (y.sum() + h.pow(2).sum()).backward()
        want = self.gradients(lambda: (lin2(xr) * lin(x).exp()).sum() + lin(x).exp().pow(2).sum(), self.params + [xr])[1]
        got = [flat(p.grad) for p in self.params + [xr]]
        for g, w in zip(got, want):
            for a, b in zip(g, w):
                self.assertAlmostEqual(a, b, places=4)

    def test_autocast_is_restored_for_the_recomputation(self):
        lin, x = self.lin, self.x
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reference = lin(x)
            out = checkpoint(lin, x)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(flat(out), flat(reference))
        seen = []
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = checkpoint(lambda t: seen.append(torch.is_autocast_enabled("cpu")) or lin(t), x)
        want = self.gradients(lambda: reference.float().pow(2).sum(), self.params[:2])[1]
        got = self.gradients(lambda: out.float().pow(2).sum(), self.params[:2])[1]      # backward outside the block
        self.assertEqual(seen, [True, True])
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertEqual(got, want)

    def test_context_fn(self):
        events = []

        @contextlib.contextmanager
        def note(name):
            events.append(name + " in")
            yield
            events.append(name + " out")

        out = checkpoint(lambda t: events.append("run") or self.lin(t), self.x, context_fn=lambda: (note("forward"), note("recompute")))
        self.assertEqual(events, ["forward in", "run", "forward out"])
        out.sum().backward()
        self.assertEqual(events[3:], ["recompute in", "run", "recompute out"])

    def test_keyword_arguments(self):
        lin, x = self.lin, self.x
        out = checkpoint(lambda t, scale=1.0, shift=0.0: lin(t) * scale + shift, x, scale=3.0, shift=1.0, use_reentrant=False)
        self.assertEqual(flat(out), flat(lin(x) * 3.0 + 1.0))
        with self.assertRaises(ValueError):
            checkpoint(lin, x, determinism_check="sometimes")

    def test_a_function_that_changes_its_mind(self):
        lin, lin2, x = self.lin, self.lin2, self.x
        a, b = lin(x), lin2(x)
        state = {"calls": 0}

        def fickle(t):
            state["calls"] += 1
            return t * (a if state["calls"] == 1 else b)

        out = checkpoint(fickle, self.xr)
        with self.assertRaisesRegex(RuntimeError, "had not used in the forward pass"):
            out.sum().backward()

        state["calls"] = 0

        def reshaping(t):
            state["calls"] += 1
            return lin(t) if state["calls"] == 1 else lin(t)[:2]

        out = checkpoint(reshaping, x)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            out.sum().backward()
        state["calls"] = 0
        out = checkpoint(reshaping, x, determinism_check="none")       # not checked: the engine reports the shapes itself
        with self.assertRaises(RuntimeError):
            out.sum().backward()


if __name__ == "__main__":
    unittest.main()
