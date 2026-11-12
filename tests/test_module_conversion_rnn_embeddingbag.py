"""Module.to()/half()/double() on shared parameters, RNN parameter names, EmbeddingBag (BUG-13).

Before: every module replaced its parameters by new Parameter objects on conversion, so tied
embedding / output weights became two tensors, optimizers were left holding the old ones and
.grad was dropped; Module.to(dtype) also converted integer buffers. RNN/LSTM/GRU state dicts
used "cell_l0.weight_ih" where torch.nn uses "weight_ih_l0". EmbeddingBag ignored
per_sample_weights for a 1-D input with offsets, counted padding entries in the mean and took
any offsets it was given.
"""

import math
import unittest

import torch
import torch.nn.functional as F

nn = torch.nn


class TiedLM(nn.Module):
    def __init__(self, vocab=5, dim=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight            # weight tying

    def forward(self, tokens):
        return self.head(self.embed(tokens))


class TestModuleConversion(unittest.TestCase):
    CONVERSIONS = {
        "double()": (lambda m: m.double(), torch.float64),
        "half()": (lambda m: m.half(), torch.float16),
        "to(float64)": (lambda m: m.to(torch.float64), torch.float64),
        "to(dtype=float16)": (lambda m: m.to(dtype=torch.float16), torch.float16),
        "bfloat16()": (lambda m: m.bfloat16(), torch.bfloat16),
        "double().float()": (lambda m: m.double().float(), torch.float32),
        "to('cpu', float64)": (lambda m: m.to("cpu", torch.float64), torch.float64),
        "to(tensor)": (lambda m: m.to(torch.zeros(1, dtype=torch.float64)), torch.float64),
    }

    def test_tied_weights_stay_tied(self):
        for name, (convert, dtype) in self.CONVERSIONS.items():
            with self.subTest(name):
                m = TiedLM()
                before = m.embed.weight
                values = before.tolist()
                self.assertIs(convert(m), m)
                self.assertIs(m.head.weight, m.embed.weight)
                self.assertIs(m.embed.weight, before)               # the Parameter object is not replaced
                self.assertEqual(len(list(m.parameters())), 1)
                self.assertEqual(m.embed.weight.dtype, dtype)
                self.assertTrue(m.embed.weight.requires_grad)
                self.assertTrue(m.embed.weight.is_leaf)
                places = 5 if dtype != torch.float16 and dtype != torch.bfloat16 else 1
                for got, want in zip(m.embed.weight.reshape(-1).tolist(), [v for row in values for v in row]):
                    self.assertAlmostEqual(got, want, places=places)

    def test_tied_weights_train_as_one_parameter_after_conversion(self):
        torch.manual_seed(0)
        m = TiedLM().double()
        tokens = torch.tensor([0, 3, 1])
        loss = F.cross_entropy(m(tokens), torch.tensor([1, 2, 4]))
        loss.backward()
        grad = m.embed.weight.grad
        self.assertIsNotNone(grad)
        self.assertIs(m.head.weight.grad, grad)
        self.assertEqual(grad.dtype, torch.float64)
        # row 2 is never looked up: its gradient can only come from the output layer
        self.assertGreater(grad[2].abs().sum().item(), 0.0)

    def test_an_optimizer_created_before_the_conversion_still_trains_the_model(self):
        torch.manual_seed(0)
        m = TiedLM()
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        m.double()
        self.assertTrue(any(p is m.embed.weight for g in opt.param_groups for p in g["params"]))
        before = m.embed.weight.tolist()
        F.cross_entropy(m(torch.tensor([0, 3, 1])), torch.tensor([1, 2, 4])).backward()
        opt.step()
        self.assertNotEqual(m.embed.weight.tolist(), before)
        self.assertEqual(m.embed.weight.dtype, torch.float64)

    def test_gradients_are_converted_with_the_parameter(self):
        m = nn.Linear(3, 2)
        m.weight.grad = torch.full((2, 3), 0.25)
        m.double()
        self.assertIsNotNone(m.weight.grad)
        self.assertEqual(m.weight.grad.dtype, torch.float64)
        self.assertEqual(m.weight.grad.tolist(), [[0.25] * 3] * 2)
        self.assertIsNone(m.bias.grad)

    def test_requires_grad_flags_survive(self):
        m = nn.Linear(3, 2)
        m.bias.requires_grad_(False)
        m.half()
        self.assertTrue(m.weight.requires_grad)
        self.assertFalse(m.bias.requires_grad)

    def test_integer_buffers_keep_their_dtype(self):
        bn = nn.BatchNorm1d(4)
        bn(torch.randn(8, 4))
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        bn.to(torch.float64)
        self.assertEqual(bn.running_mean.dtype, torch.float64)
        self.assertEqual(bn.weight.dtype, torch.float64)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.int64)     # a counter, not a weight
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        bn.half()
        self.assertEqual(bn.num_batches_tracked.dtype, torch.int64)
        with self.assertRaises(TypeError) as ctx:
            bn.to(torch.int32)
        self.assertIn("floating point or complex", str(ctx.exception))
        with self.assertRaises(TypeError):
            bn.to(dtyp=torch.float64)
        with self.assertRaises(TypeError):
            bn.to(3)

    def test_shared_modules_and_buffers_are_converted_once(self):
        shared = nn.Linear(2, 2)

        class Twice(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = shared
                self.b = nn.Sequential(shared)
                self.register_buffer("scale", torch.ones(2))
                self.inner = nn.Module()
                self.inner.register_buffer("scale", self.scale)

        m = Twice()
        seen = []

        def halve(t):
            seen.append(t)
            return t * 0.5                                  # not idempotent: a second visit would show

        weight = shared.weight.tolist()
        m._apply(halve)
        self.assertEqual(len(seen), 3)                      # weight, bias, scale
        self.assertEqual(shared.weight.tolist(), [[v * 0.5 for v in row] for row in weight])
        self.assertEqual(m.scale.tolist(), [0.5, 0.5])
        self.assertIs(m.inner.scale, m.scale)

    def test_overriding_apply_with_the_one_argument_signature(self):
        class Extra(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.table = torch.ones(2)                  # a plain attribute the module moves itself

            def _apply(self, fn):
                super()._apply(fn)
                self.table = fn(self.table)
                return self

        outer = nn.Sequential(Extra(), nn.Linear(2, 2))
        outer.double()
        self.assertEqual(outer[0].table.dtype, torch.float64)
        self.assertEqual(outer[0].lin.weight.dtype, torch.float64)
        self.assertEqual(outer[1].weight.dtype, torch.float64)

    def test_state_dict_after_conversion(self):
        m = TiedLM().double()
        state = m.state_dict()
        self.assertEqual(sorted(state), ["embed.weight", "head.weight"])
        self.assertTrue(all(v.dtype == torch.float64 for v in state.values()))
        other = TiedLM()
        other.load_state_dict(state)
        self.assertEqual(other.embed.weight.dtype, torch.float32)
        for a, b in zip(other.embed.weight.reshape(-1).tolist(), m.embed.weight.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=6)


class TestRNNParameterNames(unittest.TestCase):
    def test_state_dict_keys_are_the_torch_names(self):
        lstm = nn.LSTM(3, 4, num_layers=2, bidirectional=True)
        want = []
        for layer in range(2):
            for sfx in ("", "_reverse"):
                want += [f"weight_ih_l{layer}{sfx}", f"weight_hh_l{layer}{sfx}", f"bias_ih_l{layer}{sfx}", f"bias_hh_l{layer}{sfx}"]
        state = lstm.state_dict()
        self.assertEqual(list(state), want)
        self.assertEqual(tuple(state["weight_ih_l0"].shape), (16, 3))
        self.assertEqual(tuple(state["weight_hh_l0_reverse"].shape), (16, 4))
        self.assertEqual(tuple(state["weight_ih_l1"].shape), (16, 8))        # both directions of layer 0
        self.assertEqual(tuple(state["bias_hh_l1_reverse"].shape), (16,))
        self.assertEqual(list(nn.GRU(3, 4).state_dict()), ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"])
        self.assertEqual(tuple(nn.GRU(3, 4).weight_ih_l0.shape), (12, 3))
        self.assertEqual(list(nn.RNN(3, 4, bias=False).state_dict()), ["weight_ih_l0", "weight_hh_l0"])
        self.assertEqual(tuple(nn.RNN(3, 4).weight_hh_l0.shape), (4, 4))
        self.assertEqual([n for n, _ in nn.LSTM(3, 4).named_parameters()], ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"])

    def test_a_state_dict_with_the_torch_names_loads_and_drives_the_forward_pass(self):
        torch.manual_seed(0)
        H, I, T, B = 4, 3, 5, 2
        x = torch.randn(T, B, I)
        for mode, cell_cls, rnn_cls in (("LSTM", nn.LSTMCell, nn.LSTM), ("GRU", nn.GRUCell, nn.GRU), ("RNN", nn.RNNCell, nn.RNN)):
            with self.subTest(mode):
                cell = cell_cls(I, H)
                rnn = rnn_cls(I, H)
                rnn.load_state_dict({
                    "weight_ih_l0": cell.weight_ih.detach(), "weight_hh_l0": cell.weight_hh.detach(),
                    "bias_ih_l0": cell.bias_ih.detach(), "bias_hh_l0": cell.bias_hh.detach(),
                })
                out, _ = rnn(x)
                state, steps = None, []
                for t in range(T):
                    state = cell(x[t], state)
                    steps.append(state[0] if mode == "LSTM" else state)
                self.assertLess((out - torch.stack(steps, dim=0)).abs().max().item(), 1e-6)

    def test_state_dicts_written_with_the_old_names_still_load(self):
        torch.manual_seed(1)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(3, 4, num_layers=2, bidirectional=True)
                self.out = nn.Linear(8, 2)

        src, dst = Net(), Net()
        old = {}
        for key, value in src.state_dict().items():
            if key.startswith("rnn."):
                kind, layer = key[4:].split("_l", 1)                 # weight_ih, 0_reverse
                key = f"rnn.cell_l{layer}.{kind}"
            old[key] = value
        self.assertIn("rnn.cell_l1_reverse.bias_hh", old)
        result = dst.load_state_dict(old)
        self.assertEqual((list(result.missing_keys), list(result.unexpected_keys)), ([], []))
        x = torch.randn(4, 2, 3)
        self.assertEqual(dst.rnn(x)[0].tolist(), src.rnn(x)[0].tolist())
        with self.assertRaises(RuntimeError):
            dst.load_state_dict({"rnn.cell_l9.weight_ih": torch.zeros(1)})

    def test_every_parameter_gets_a_gradient_and_converts(self):
        torch.manual_seed(2)
        for rnn in (nn.LSTM(3, 4, num_layers=2, bidirectional=True, batch_first=True), nn.GRU(3, 4, num_layers=2), nn.RNN(3, 4, nonlinearity="relu")):
            with self.subTest(type(rnn).__name__):
                x = torch.randn(2, 5, 3)
                out, _ = rnn(x)
                out.sum().backward()
                for name, p in rnn.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                rnn.double()
                rnn.flatten_parameters()
                out, _ = rnn(x.double())
                self.assertEqual(out.dtype, torch.float64)


def bag_reference(weight, bags, mode, sample_weights=None, padding_idx=None):
    dim = len(weight[0])
    out = []
    for b, bag in enumerate(bags):
        rows = []
        for j, idx in enumerate(bag):
            if idx == padding_idx:
                continue
            w = 1.0 if sample_weights is None else sample_weights[b][j]
            rows.append([w * v for v in weight[idx]])
        if not rows:
            out.append([0.0] * dim)
        elif mode == "max":
            out.append([max(col) for col in zip(*rows)])
        else:
            total = [sum(col) for col in zip(*rows)]
            out.append(total if mode == "sum" else [v / len(rows) for v in total])
    return out


class TestEmbeddingBag(unittest.TestCase):
    WEIGHT = [[1.0, -10.0], [2.0, -20.0], [-3.0, 30.0], [4.0, -40.0], [-5.0, 50.0], [6.0, -60.0]]

    def bag(self, mode, **kwargs):
        return nn.EmbeddingBag.from_pretrained(torch.tensor(self.WEIGHT), freeze=False, mode=mode, **kwargs)

    def assertRows(self, got, want):
        got = got.tolist()
        self.assertEqual(len(got), len(want))
        for grow, wrow in zip(got, want):
            self.assertEqual(len(grow), len(wrow))
            for g, w in zip(grow, wrow):
                self.assertAlmostEqual(g, w, places=5)

    def ragged(self, bags):
        flat = [i for bag in bags for i in bag]
        offsets, pos = [], 0
        for bag in bags:
            offsets.append(pos)
            pos += len(bag)
        return torch.tensor(flat, dtype=torch.int64), torch.tensor(offsets, dtype=torch.int64)

    def test_offsets_and_per_sample_weights(self):
        bags = [[0, 1], [2, 3, 4]]
        weights = [[1.0, 0.5], [2.0, 1.0, 0.0]]
        idx, offsets = self.ragged(bags)
        psw = torch.tensor([w for bag in weights for w in bag])
        got = self.bag("sum")(idx, offsets, per_sample_weights=psw)
        self.assertRows(got, bag_reference(self.WEIGHT, bags, "sum", weights))           # used to ignore the weights
        self.assertNotEqual(got.tolist(), self.bag("sum")(idx, offsets).tolist())
        # the same bags as rows of a 2-D input
        got2d = self.bag("sum")(torch.tensor([[0, 1], [2, 3]]), per_sample_weights=torch.tensor([[1.0, 0.5], [2.0, 1.0]]))
        self.assertRows(got2d, bag_reference(self.WEIGHT, [[0, 1], [2, 3]], "sum", [[1.0, 0.5], [2.0, 1.0]]))

    def test_all_modes_on_ragged_and_regular_bags(self):
        ragged = [[0, 1], [2, 3, 4], [5], [1, 1, 2, 0]]
        regular = [[0, 1, 2], [3, 4, 5], [1, 1, 1]]
        for mode in ("sum", "mean", "max"):
            with self.subTest(mode=mode):
                idx, offsets = self.ragged(ragged)
                self.assertRows(self.bag(mode)(idx, offsets), bag_reference(self.WEIGHT, ragged, mode))
                self.assertRows(self.bag(mode)(torch.tensor(regular)), bag_reference(self.WEIGHT, regular, mode))
                self.assertRows(F.embedding_bag(idx, torch.tensor(self.WEIGHT), offsets, mode=mode), bag_reference(self.WEIGHT, ragged, mode))

    def test_empty_bags_give_zeros(self):
        bags = [[], [0, 1], [], [], [2, 3, 4], []]
        idx, offsets = self.ragged(bags)
        for mode in ("sum", "mean", "max"):
            with self.subTest(mode=mode):
                module = self.bag(mode)
                out = module(idx, offsets)
                self.assertRows(out, bag_reference(self.WEIGHT, bags, mode))
                out.sum().backward()
                self.assertTrue(all(math.isfinite(g) for g in module.weight.grad.reshape(-1).tolist()))
                # nothing but empty bags, and no bags at all
                none = module(torch.tensor([], dtype=torch.int64), torch.tensor([0, 0], dtype=torch.int64))
                self.assertEqual(none.tolist(), [[0.0, 0.0], [0.0, 0.0]])
                self.assertEqual(tuple(module(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)).shape), (0, 2))
                self.assertEqual(module(torch.zeros(2, 0, dtype=torch.int64)).tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_padding_entries_are_left_out(self):
        ragged = [[0, 1, 2], [1, 1], [4, 1, 2, 1], []]
        regular = [[0, 1, 2], [1, 1, 1], [2, 4, 1]]
        for mode in ("sum", "mean", "max"):
            with self.subTest(mode=mode):
                module = self.bag(mode, padding_idx=1)
                idx, offsets = self.ragged(ragged)
                self.assertRows(module(idx, offsets), bag_reference(self.WEIGHT, ragged, mode, padding_idx=1))
                self.assertRows(module(torch.tensor(regular)), bag_reference(self.WEIGHT, regular, mode, padding_idx=1))
        # the mean divides by the entries that count: (w0 + w2) / 2, not / 3
        mean = self.bag("mean", padding_idx=1)(torch.tensor([[0, 1, 2]]))
        self.assertRows(mean, [[-1.0, 10.0]])
        # the padding row takes no gradient
        module = self.bag("sum", padding_idx=1)
        module(torch.tensor([[0, 1, 2]])).sum().backward()
        self.assertEqual(module.weight.grad.tolist()[:3], [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        fresh = nn.EmbeddingBag(6, 2, padding_idx=-1)
        self.assertEqual(fresh.weight[5].tolist(), [0.0, 0.0])
        self.assertRows(F.embedding_bag(torch.tensor([[0, 5]]), torch.tensor(self.WEIGHT), padding_idx=-1), [self.WEIGHT[0]])

    def test_gradients(self):
        module = self.bag("sum")
        idx, offsets = self.ragged([[0, 1], [1, 3, 1]])
        psw = torch.tensor([1.0, 0.5, 2.0, 1.0, 3.0], requires_grad=True)
        module(idx, offsets, per_sample_weights=psw).sum().backward()
        self.assertEqual(module.weight.grad.tolist(), [[1.0, 1.0], [5.5, 5.5], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(psw.grad.tolist(), [sum(self.WEIGHT[i]) for i in (0, 1, 1, 3, 1)])
        module = self.bag("mean")
        module(idx, offsets).sum().backward()
        want = [[0.5, 0.5], [0.5 + 2 / 3, 0.5 + 2 / 3], [0.0, 0.0], [1 / 3, 1 / 3], [0.0, 0.0], [0.0, 0.0]]
        self.assertRows(module.weight.grad, want)

    def test_include_last_offset(self):
        idx = torch.tensor([0, 1, 2, 3, 4])
        module = self.bag("sum", include_last_offset=True)
        self.assertRows(module(idx, torch.tensor([0, 2, 5])), bag_reference(self.WEIGHT, [[0, 1], [2, 3, 4]], "sum"))
        # the last offset ends the last bag: what follows belongs to no bag
        self.assertRows(module(idx, torch.tensor([0, 2, 4])), bag_reference(self.WEIGHT, [[0, 1], [2, 3]], "sum"))
        got = F.embedding_bag(idx, torch.tensor(self.WEIGHT), torch.tensor([0, 2, 4]), mode="sum", include_last_offset=True,
                              per_sample_weights=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertRows(got, bag_reference(self.WEIGHT, [[0, 1], [2, 3]], "sum", [[1.0, 2.0], [3.0, 4.0]]))

    def test_invalid_arguments(self):
        idx = torch.tensor([0, 1, 2, 3, 4])
        module = self.bag("sum")
        cases = {
            "offsets must start at 0": (ValueError, lambda: module(idx, torch.tensor([1, 3]))),
            "offsets beyond the input": (ValueError, lambda: module(idx, torch.tensor([0, 9]))),
            "descending offsets": (ValueError, lambda: module(idx, torch.tensor([0, 3, 1]))),
            "negative offsets": (ValueError, lambda: module(idx, torch.tensor([-2, 3]))),
            "1-D input without offsets": (ValueError, lambda: module(idx)),
            "2-D input with offsets": (ValueError, lambda: module(idx[:4].reshape(2, 2), torch.tensor([0, 2]))),
            "3-D input": (ValueError, lambda: module(torch.zeros(1, 2, 2, dtype=torch.int64))),
            "float offsets": (TypeError, lambda: module(idx, torch.tensor([0.0, 2.0]))),
            "weights of another shape": (ValueError, lambda: module(idx, torch.tensor([0, 2]), per_sample_weights=torch.ones(4))),
            "weights with mean": (NotImplementedError, lambda: self.bag("mean")(idx, torch.tensor([0, 2]), per_sample_weights=torch.ones(5))),
            "weights with max": (NotImplementedError, lambda: self.bag("max")(idx, torch.tensor([0, 2]), per_sample_weights=torch.ones(5))),
            "unknown mode": (ValueError, lambda: nn.EmbeddingBag(6, 2, mode="median")),
            "index out of range": (IndexError, lambda: module(torch.tensor([0, 6]), torch.tensor([0]))),
            "padding_idx out of range": (ValueError, lambda: F.embedding_bag(idx, torch.tensor(self.WEIGHT), torch.tensor([0]), padding_idx=6)),
        }
        for name, (exc, call) in cases.items():
            with self.subTest(name):
                with self.assertRaises(exc):
                    call()


if __name__ == "__main__":
    unittest.main()
