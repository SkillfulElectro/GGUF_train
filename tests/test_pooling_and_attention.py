"""Adaptive pooling windows, pairwise_distance and MultiheadAttention masks/layouts (BUG-12).

Before: adaptive_avg_pool2d ran bilinear interpolation whenever the input size was not a
multiple of the output size, adaptive_max_pool2d used one window size for all outputs, and
MultiheadAttention read a boolean attn_mask the wrong way round, multiplied a float
key_padding_mask by -1e9, lost a boolean attn_mask as soon as a key_padding_mask was given,
and treated an unbatched (L, E) query as L batches of length 1 under batch_first=True.
"""

import itertools
import math
import unittest

import torch
import torch.nn.functional as F
from gguf_train._functions import FlashAttentionFn

NEG_INF = float("-inf")


def windows(in_size, out_size):
    return [(math.floor(i * in_size / out_size), math.ceil((i + 1) * in_size / out_size)) for i in range(out_size)]


def plane(h, w):
    return [[float((3 * r * r + 7 * c + r * c) % 11) - 0.25 * r for c in range(w)] for r in range(h)]


def avg_reference(grid, oh, ow):
    ih, iw = len(grid), len(grid[0])
    out = []
    for r0, r1 in windows(ih, oh):
        row = []
        for c0, c1 in windows(iw, ow):
            cells = [grid[r][c] for r in range(r0, r1) for c in range(c0, c1)]
            row.append(sum(cells) / len(cells))
        out.append(row)
    return out


def max_reference(grid, oh, ow):
    ih, iw = len(grid), len(grid[0])
    values, indices = [], []
    for r0, r1 in windows(ih, oh):
        vrow, irow = [], []
        for c0, c1 in windows(iw, ow):
            best = max((grid[r][c], -(r * iw + c)) for r in range(r0, r1) for c in range(c0, c1))
            vrow.append(best[0])
            irow.append(-best[1])        # the first maximum in row-major order
        values.append(vrow)
        indices.append(irow)
    return values, indices


class TestAdaptivePooling(unittest.TestCase):
    SIZES = [((4, 5), (2, 2)), ((4, 5), (3, 3)), ((7, 7), (3, 5)), ((5, 3), (4, 2)), ((2, 3), (3, 5)),
             ((6, 6), (3, 2)), ((5, 5), (5, 3)), ((9, 4), (1, 3)), ((3, 8), (2, 1)), ((5, 7), (1, 1))]

    def assertGrid(self, got, want, places=5):
        got = got.tolist()
        self.assertEqual((len(got), len(got[0])), (len(want), len(want[0])))
        for grow, wrow in zip(got, want):
            for g, w in zip(grow, wrow):
                self.assertAlmostEqual(g, w, places=places)

    def test_average_uses_the_adaptive_windows(self):
        for (ih, iw), (oh, ow) in self.SIZES:
            with self.subTest(input=(ih, iw), output=(oh, ow)):
                grid = plane(ih, iw)
                x = torch.tensor([[grid, [[2 * v for v in row] for row in grid]]])        # (1, 2, H, W)
                out = F.adaptive_avg_pool2d(x, (oh, ow))
                self.assertEqual(tuple(out.shape), (1, 2, oh, ow))
                want = avg_reference(grid, oh, ow)
                self.assertGrid(out[0, 0], want)
                self.assertGrid(out[0, 1], [[2 * v for v in row] for row in want])
                # unbatched (C, H, W), and the module
                self.assertGrid(F.adaptive_avg_pool2d(x[0], (oh, ow))[0], want)
                self.assertGrid(torch.nn.AdaptiveAvgPool2d((oh, ow))(x)[0, 0], want)

    def test_average_is_not_an_interpolation(self):
        # 4 x 5 -> 2 x 2: rows [0, 2), [2, 4); columns [0, 3), [2, 5)
        x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
        self.assertEqual(F.adaptive_avg_pool2d(x, 2).tolist(), [[[[3.5, 5.5], [13.5, 15.5]]]])
        bilinear = F.interpolate(x, size=(2, 2), mode="bilinear").tolist()
        self.assertNotEqual(bilinear, [[[[3.5, 5.5], [13.5, 15.5]]]])

    def test_output_size_forms(self):
        grid = plane(4, 5)
        x = torch.tensor([[grid]])
        self.assertGrid(F.adaptive_avg_pool2d(x, 3)[0, 0], avg_reference(grid, 3, 3))
        self.assertGrid(F.adaptive_avg_pool2d(x, (None, 2))[0, 0], avg_reference(grid, 4, 2))
        self.assertGrid(F.adaptive_avg_pool2d(x, (3, None))[0, 0], avg_reference(grid, 3, 5))
        self.assertGrid(F.adaptive_avg_pool2d(x, [2, 4])[0, 0], avg_reference(grid, 2, 4))
        self.assertEqual(tuple(F.adaptive_avg_pool2d(x, (0, 2)).shape), (1, 1, 0, 2))
        for bad in ((1, 2, 3), -1):
            with self.assertRaises(RuntimeError):
                F.adaptive_avg_pool2d(x, bad)
        with self.assertRaises(RuntimeError):
            F.adaptive_avg_pool2d(torch.zeros(4, 5), 2)
        with self.assertRaises(RuntimeError):
            F.adaptive_avg_pool2d(torch.arange(20).reshape(1, 1, 4, 5), 3)     # integers have no mean
        with self.assertRaises(RuntimeError):
            F.adaptive_avg_pool2d(torch.zeros(1, 1, 0, 5), 3)

    def test_one_dimensional_average(self):
        row = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0]
        for out_len in (1, 2, 3, 5, 7, 9):
            with self.subTest(output=out_len):
                want = avg_reference([row], 1, out_len)[0]
                for x in (torch.tensor([[row]]), torch.tensor([row])):           # (N, C, L) and (C, L)
                    got = F.adaptive_avg_pool1d(x, out_len)
                    self.assertEqual(tuple(got.shape), tuple(x.shape[:-1]) + (out_len,))
                    for g, w in zip(got.reshape(-1).tolist(), want):
                        self.assertAlmostEqual(g, w, places=5)
                got = torch.nn.AdaptiveAvgPool1d(out_len)(torch.tensor([[row]]))
                self.assertAlmostEqual(got.reshape(-1).tolist()[0], want[0], places=5)

    def test_average_gradient(self):
        from torch.autograd import gradcheck

        x = torch.tensor([[plane(4, 5)]], dtype=torch.float64, requires_grad=True)
        w = torch.tensor(plane(3, 3), dtype=torch.float64)
        self.assertTrue(gradcheck(lambda z: (F.adaptive_avg_pool2d(z, 3) * w).sum(), (x,)))
        # every output spreads its gradient evenly over its window
        x = torch.zeros(1, 1, 4, 5, requires_grad=True)
        F.adaptive_avg_pool2d(x, 2)[0, 0, 0, 1].backward()                       # rows [0, 2), columns [2, 5)
        want = [[1 / 6 if r < 2 and c >= 2 else 0.0 for c in range(5)] for r in range(4)]
        for grow, wrow in zip(x.grad[0, 0].tolist(), want):
            for g, v in zip(grow, wrow):
                self.assertAlmostEqual(g, v, places=6)

    def test_max_uses_the_adaptive_windows(self):
        for (ih, iw), (oh, ow) in self.SIZES:
            with self.subTest(input=(ih, iw), output=(oh, ow)):
                grid = plane(ih, iw)
                want_v, want_i = max_reference(grid, oh, ow)
                x = torch.tensor([[grid]])
                values, indices = F.adaptive_max_pool2d(x, (oh, ow), return_indices=True)
                self.assertEqual(values[0, 0].tolist(), want_v)
                self.assertEqual(indices[0, 0].tolist(), want_i)
                self.assertEqual(F.adaptive_max_pool2d(x, (oh, ow))[0, 0].tolist(), want_v)
                self.assertEqual(F.adaptive_max_pool2d(x[0], (oh, ow))[0].tolist(), want_v)
                values, indices = torch.nn.AdaptiveMaxPool2d((oh, ow), return_indices=True)(x[0])
                self.assertEqual((values[0].tolist(), indices[0].tolist()), (want_v, want_i))

    def test_max_gradient_goes_to_the_selected_element(self):
        x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5).requires_grad_(True)
        out = F.adaptive_max_pool2d(x, (3, 2))                                   # 7 9 / 12 14 / 17 19
        (out * torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).sum().backward()
        want = [0.0] * 20
        for pos, g in ((7, 1.0), (9, 2.0), (12, 3.0), (14, 4.0), (17, 5.0), (19, 6.0)):
            want[pos] = g
        self.assertEqual(x.grad.reshape(-1).tolist(), want)


class TestPairwiseDistance(unittest.TestCase):
    def test_follows_the_torch_definition(self):
        # torch: || x1 - x2 + eps ||_p, eps added to every component of the difference.
        # The audit asked for (sum |d|^p + eps^p)^(1/p); that is a different function from the one
        # torch.nn.functional.pairwise_distance computes, so the torch definition stays.
        a = [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.5, 0.0]]
        b = [[1.5, 0.0, 3.0, 1.0], [0.5, -1.0, 2.5, 0.0]]
        for p, eps in ((2.0, 1e-6), (1.0, 1e-6), (3.0, 0.5), (2.0, 0.0)):
            with self.subTest(p=p, eps=eps):
                want = [sum(abs(x - y + eps) ** p for x, y in zip(ra, rb)) ** (1.0 / p) for ra, rb in zip(a, b)]
                got = F.pairwise_distance(torch.tensor(a), torch.tensor(b), p=p, eps=eps)
                for g, w in zip(got.tolist(), want):
                    self.assertAlmostEqual(g, w, places=5)
        got = F.pairwise_distance(torch.tensor(a), torch.tensor(b), keepdim=True)
        self.assertEqual(tuple(got.shape), (2, 1))
        # identical inputs: eps * sqrt(n), as in torch
        same = F.pairwise_distance(torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64), eps=1e-3)
        self.assertAlmostEqual(same.item(), 2e-3, places=9)


def mha_reference(m, q, k, v, key_padding_mask=None, attn_mask=None):
    """torch.nn.MultiheadAttention on batch-first inputs, from basic operations.

    Boolean masks: True marks a position that must not be attended to. Float masks are added.
    """
    E, H = m.embed_dim, m.num_heads
    D = E // H
    B, L, _ = q.shape
    S = k.shape[1]

    def project(x, i):
        y = x.matmul(m.in_proj_weight[i * E:(i + 1) * E].transpose(0, 1)) + m.in_proj_bias[i * E:(i + 1) * E]
        return y.reshape(B, x.shape[1], H, D).transpose(1, 2)

    scores = project(q, 0).matmul(project(k, 1).transpose(-2, -1)) / math.sqrt(D)
    for mask, shape in ((attn_mask, None), (key_padding_mask, (B, 1, 1, S))):
        if mask is None:
            continue
        if shape is None:
            shape = (1, 1, L, S) if mask.ndim == 2 else (B, H, L, S)
        mask = mask.reshape(shape)
        scores = scores.masked_fill(mask, NEG_INF) if mask.dtype == torch.bool else scores + mask
    weights = F.softmax(scores, dim=-1)
    out = weights.matmul(project(v, 2)).transpose(1, 2).reshape(B, L, E)
    return out.matmul(m.out_proj.weight.transpose(0, 1)) + m.out_proj.bias, weights


class TestMultiheadAttention(unittest.TestCase):
    E, H, B, L, S = 8, 2, 3, 4, 5

    def assertClose(self, got, want, msg=""):
        self.assertEqual(tuple(got.shape), tuple(want.shape), msg)
        self.assertLess((got - want).abs().max().item(), 1e-4, msg)

    def masks(self, batch):
        L, S, H = self.L, self.S, self.H
        kpm = torch.zeros(batch, S, dtype=torch.bool)
        kpm[:, -1] = True
        if batch > 1:
            kpm[1, 1] = True
        causal = torch.triu(torch.ones(L, S), diagonal=1).bool()                # True above the diagonal: blocked
        per_head = torch.rand(batch * H, L, S) < 0.3
        per_head[:, :, 0] = False                                                # every query keeps one key
        return {
            "none": (None, None),
            "bool key_padding_mask": (kpm, None),
            "float key_padding_mask": (torch.zeros(batch, S).masked_fill(kpm, NEG_INF), None),
            "bool attn_mask": (None, causal),
            "float attn_mask": (None, torch.zeros(L, S).masked_fill(causal, NEG_INF)),
            "3D bool attn_mask": (None, per_head),
            "bool key_padding_mask + bool attn_mask": (kpm, causal),
            "bool key_padding_mask + float attn_mask": (kpm, torch.zeros(L, S).masked_fill(causal, NEG_INF)),
            "float key_padding_mask + 3D bool attn_mask": (torch.zeros(batch, S).masked_fill(kpm, NEG_INF), per_head),
        }

    def test_matches_the_reference_in_every_layout(self):
        torch.manual_seed(0)
        for batch_first, batched, need_weights in itertools.product((False, True), (True, False), (True, False)):
            m = torch.nn.MultiheadAttention(self.E, self.H, batch_first=batch_first)
            with torch.no_grad():
                m.in_proj_bias.copy_(torch.randn(3 * self.E) * 0.1)
            m.eval()
            batch = self.B if batched else 1
            q, k, v = torch.randn(batch, self.L, self.E), torch.randn(batch, self.S, self.E), torch.randn(batch, self.S, self.E)

            def layout(x):
                if not batched:
                    return x[0]
                return x if batch_first else x.transpose(0, 1)

            for name, (kpm, am) in self.masks(batch).items():
                label = f"batch_first={batch_first} batched={batched} need_weights={need_weights} masks: {name}"
                with self.subTest(label):
                    want, weights = mha_reference(m, q, k, v, kpm, am)
                    out, got_weights = m(
                        layout(q), layout(k), layout(v),
                        key_padding_mask=kpm if (batched or kpm is None) else kpm[0],
                        attn_mask=am, need_weights=need_weights,
                    )
                    self.assertClose(out, layout(want), label)
                    if need_weights:
                        averaged = weights.mean(dim=1)
                        self.assertClose(got_weights, averaged if batched else averaged[0], label)
                    else:
                        self.assertIsNone(got_weights)

    def test_unbatched_input_with_batch_first(self):
        torch.manual_seed(1)
        m = torch.nn.MultiheadAttention(self.E, self.H, batch_first=True).eval()
        q, kv = torch.randn(self.L, self.E), torch.randn(self.S, self.E)
        out, weights = m(q, kv, kv)
        self.assertEqual(tuple(out.shape), (self.L, self.E))                     # was (L, 1, E): L batches of one query
        self.assertEqual(tuple(weights.shape), (self.L, self.S))
        batched_out, batched_weights = m(q.unsqueeze(0), kv.unsqueeze(0), kv.unsqueeze(0))
        self.assertClose(out, batched_out[0])
        self.assertClose(weights, batched_weights[0])
        _, per_head = m(q, kv, kv, average_attn_weights=False)
        self.assertEqual(tuple(per_head.shape), (self.H, self.L, self.S))

    def test_true_in_a_boolean_attn_mask_blocks_the_position(self):
        torch.manual_seed(2)
        m = torch.nn.MultiheadAttention(self.E, self.H, batch_first=True).eval()
        x = torch.randn(1, self.L, self.E)
        causal = torch.triu(torch.ones(self.L, self.L), diagonal=1).bool()
        for need_weights in (True, False):
            with self.subTest(need_weights=need_weights):
                out, weights = m(x, x, x, attn_mask=causal, need_weights=need_weights)
                changed = x.clone()
                changed[0, -1] = changed[0, -1] + 5.0                           # the future of every earlier position
                out2, _ = m(changed, changed, changed, attn_mask=causal, need_weights=need_weights)
                self.assertLess((out[0, :-1] - out2[0, :-1]).abs().max().item(), 1e-5)
                self.assertGreater((out[0, -1] - out2[0, -1]).abs().max().item(), 1e-3)
                if need_weights:
                    upper = weights[0].masked_select(causal)
                    self.assertEqual(upper.abs().max().item(), 0.0)

    def test_padded_keys_get_no_attention_when_an_attn_mask_is_given_too(self):
        torch.manual_seed(3)
        m = torch.nn.MultiheadAttention(self.E, self.H, batch_first=True).eval()
        x = torch.randn(2, self.L, self.E)
        kpm = torch.tensor([[False, False, False, True], [False, True, True, True]])
        nothing_blocked = torch.zeros(self.L, self.L, dtype=torch.bool)
        _, weights = m(x, x, x, key_padding_mask=kpm, attn_mask=nothing_blocked)
        self.assertEqual(weights.masked_select(kpm.unsqueeze(1).expand(2, self.L, self.L)).abs().max().item(), 0.0)
        for row in weights.sum(dim=-1).reshape(-1).tolist():
            self.assertAlmostEqual(row, 1.0, places=5)

    def test_mask_shapes_are_checked(self):
        m = torch.nn.MultiheadAttention(self.E, self.H, batch_first=True).eval()
        x = torch.randn(self.B, self.L, self.E)
        cases = {
            "2D attn_mask": dict(attn_mask=torch.zeros(self.L, self.L + 1)),
            "3D attn_mask": dict(attn_mask=torch.zeros(self.B, self.L, self.L)),         # needs B * num_heads
            "4D attn_mask": dict(attn_mask=torch.zeros(self.B, self.H, self.L, self.L)),
            "key_padding_mask": dict(key_padding_mask=torch.zeros(self.B, self.L + 1, dtype=torch.bool)),
            "integer mask": dict(attn_mask=torch.zeros(self.L, self.L, dtype=torch.int64)),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError) as ctx:
                    m(x, x, x, **kwargs)
                self.assertIn("MultiheadAttention", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            m(x[0], x, x)

    def test_gradients_reach_every_parameter(self):
        torch.manual_seed(4)
        m = torch.nn.MultiheadAttention(self.E, self.H, batch_first=True)
        x = torch.randn(2, self.L, self.E)
        kpm = torch.tensor([[False, False, False, True], [False, False, True, True]])
        causal = torch.triu(torch.ones(self.L, self.L), diagonal=1).bool()
        for need_weights in (True, False):
            m.zero_grad()
            out, _ = m(x, x, x, key_padding_mask=kpm, attn_mask=causal, need_weights=need_weights)
            out.sum().backward()
            for name, p in m.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(all(math.isfinite(g) for g in p.grad.reshape(-1).tolist()), name)

    def test_encoder_layer_with_masks(self):
        torch.manual_seed(5)
        layer = torch.nn.TransformerEncoderLayer(self.E, self.H, dim_feedforward=16, dropout=0.0, batch_first=True).eval()
        x = torch.randn(2, self.L, self.E)
        kpm = torch.tensor([[False, False, False, True], [False, False, False, False]])
        out = layer(x, src_key_padding_mask=kpm)
        changed = x.clone()
        changed[0, -1] = changed[0, -1] + 5.0                                    # a padded position
        out2 = layer(changed, src_key_padding_mask=kpm)
        self.assertLess((out[0, :-1] - out2[0, :-1]).abs().max().item(), 1e-5)


class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.q, self.k, self.v = torch.randn(2, 3, 4, 6), torch.randn(2, 3, 5, 6), torch.randn(2, 3, 5, 6)

    def reference(self, mask=None):
        scores = self.q.matmul(self.k.transpose(-2, -1)) / math.sqrt(6)
        if mask is not None:
            scores = scores.masked_fill(~mask, NEG_INF) if mask.dtype == torch.bool else scores + mask
        return F.softmax(scores, dim=-1).matmul(self.v)

    def test_true_in_a_boolean_mask_means_attend(self):
        allowed = torch.rand(4, 5) < 0.6
        allowed[:, 0] = True
        got = F.scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=allowed)
        self.assertLess((got - self.reference(allowed)).abs().max().item(), 1e-5)

    def test_masks_are_broadcast_from_the_right(self):
        base = torch.rand(2, 3, 4, 5) < 0.6
        base[..., 0] = True
        for mask in (base, base[:, :1], base[:1], base[0], base[0, 0], base[0, 0, :1], base[..., :1, :]):
            with self.subTest(shape=list(mask.shape)):
                got = F.scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=mask)
                self.assertLess((got - self.reference(mask)).abs().max().item(), 1e-5)
                additive = torch.zeros(mask.shape).masked_fill(~mask, -1e9)
                got = F.scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=additive)
                self.assertLess((got - self.reference(mask)).abs().max().item(), 1e-5)

    def test_a_mask_that_does_not_fit_is_an_error(self):
        # the native loop indexed the mask by L and S whatever its shape
        for shape in ((2, 3), (4, 6), (7, 4, 5), (2, 3, 4, 5, 1)):
            with self.subTest(shape=shape):
                with self.assertRaises(RuntimeError) as ctx:
                    F.scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=torch.ones(*shape, dtype=torch.bool))
                self.assertIn("cannot be broadcast", str(ctx.exception))
        for shape in ((2, 3), (5,), (2, 4, 5)):
            with self.subTest(native=shape):
                with self.assertRaises(RuntimeError) as ctx:
                    FlashAttentionFn.apply(self.q, self.k, self.v, torch.ones(*shape, dtype=torch.bool), 0.4, False)
                self.assertIn("cannot be broadcast", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
