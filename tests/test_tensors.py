"""Comprehensive tests for Tensor creation, dtypes, indexing, shape ops, reductions, and einsum."""

import math
import unittest

import gguf_train as gt
import torch


class TestTensorsAndOps(unittest.TestCase):
    def test_dropin_identity(self):
        import importlib
        self.assertIs(torch.Tensor, gt.Tensor)
        self.assertIs(torch.nn.Linear, gt.nn.Linear)
        self.assertIs(torch.optim.AdamW, gt.optim.AdamW)
        for modname in ("torch.jit", "torch.fx", "torch.onnx", "torch.random", "torch.cpu"):
            importlib.import_module(modname)
        self.assertTrue(torch.cpu.is_available())
        self.assertIsNotNone(torch.jit.script)
        self.assertIsNotNone(torch.fx.symbolic_trace)
        self.assertIsNotNone(torch.onnx.export)

    def test_creation_and_dtypes(self):
        z = torch.zeros(2, 3)
        o = torch.ones((2, 3), dtype=torch.float64)
        f = torch.full((2, 3), 7.5)
        e = torch.eye(3, 4)
        ar = torch.arange(1, 10, 2)
        rg = torch.range(0, 4, 1)
        ls = torch.linspace(0.0, 1.0, 5)
        lgs = torch.logspace(0.0, 2.0, 3, base=10.0)
        sc = torch.scalar_tensor(3.25)

        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(o.dtype, torch.float64)
        self.assertTrue(f.allclose(torch.tensor([[7.5, 7.5, 7.5], [7.5, 7.5, 7.5]])))
        self.assertEqual(e.shape, (3, 4))
        self.assertAlmostEqual(float(e.trace().item()), 3.0)
        self.assertEqual(ar.tolist(), [1, 3, 5, 7, 9])
        self.assertEqual(len(rg.tolist()), 5)
        self.assertTrue(ls.allclose(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])))
        self.assertTrue(lgs.allclose(torch.tensor([1.0, 10.0, 100.0])))
        self.assertEqual(sc.ndim, 0)
        self.assertAlmostEqual(float(sc.item()), 3.25)

        for dt in [
            torch.float32, torch.float16, torch.bfloat16, torch.float64,
            torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool,
        ]:
            t = torch.zeros(2, 2, dtype=dt)
            self.assertEqual(t.dtype, dt)

        self.assertEqual(torch.finfo(torch.float32).bits, 32)
        self.assertEqual(torch.iinfo(torch.int16).max, 32767)
        self.assertEqual(torch.promote_types(torch.int32, torch.float16), torch.float16)

    def test_rng_reproducibility(self):
        torch.manual_seed(1234)
        a = torch.randn(4, 4)
        b = torch.randint(10, size=(3, 3))
        c = torch.randint(5, 15, (3, 3))
        p = torch.randperm(8)
        st = torch.get_rng_state()
        a2 = torch.randn(4, 4)
        torch.set_rng_state(st)
        a3 = torch.randn(4, 4)
        self.assertTrue(a2.allclose(a3))
        torch.manual_seed(1234)
        a_repeat = torch.randn(4, 4)
        self.assertTrue(a.allclose(a_repeat))
        self.assertEqual(sorted(p.tolist()), list(range(8)))
        self.assertTrue(((b >= 0) & (b < 10)).all().item())
        self.assertTrue(((c >= 5) & (c < 15)).all().item())

        probs = torch.tensor([0.1, 0.7, 0.2])
        samples = torch.multinomial(probs, 20, replacement=True)
        self.assertEqual(samples.shape, (20,))

    def test_views_slicing_indexing(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        self.assertEqual(x.transpose(0, 2).shape, (4, 3, 2))
        self.assertEqual(x.permute(2, 0, 1).shape, (4, 2, 3))
        self.assertEqual(x[:, 1:3, ::2].shape, (2, 2, 2))
        self.assertEqual(x[..., -1].tolist(), [[3.0, 7.0, 11.0], [15.0, 19.0, 23.0]])

        # View shares underlying storage
        v = x.view(6, 4)
        v[0, 0] = 999.0
        self.assertAlmostEqual(float(x[0, 0, 0].item()), 999.0)

        # Boolean mask & index assign
        m = x > 20.0
        sel = x[m]
        self.assertEqual(sorted(sel.tolist()), [21.0, 22.0, 23.0, 999.0])
        y = torch.zeros(2, 3)
        idx = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out_sc = y.scatter(1, idx, src)
        self.assertEqual(out_sc.tolist(), [[1.0, 0.0, 2.0], [4.0, 3.0, 0.0]])
        self.assertEqual(out_sc.gather(1, idx).tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_math_broadcast_and_scalar_ops(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([10.0, 20.0])
        self.assertEqual((a + b).tolist(), [[11.0, 22.0], [13.0, 24.0]])
        self.assertEqual((10.0 - a).tolist(), [[9.0, 8.0], [7.0, 6.0]])
        self.assertEqual((2.0 ** a).tolist(), [[2.0, 4.0], [8.0, 16.0]])
        self.assertEqual((a ** 2).tolist(), [[1.0, 4.0], [9.0, 16.0]])
        self.assertEqual((a // 2).tolist(), [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual((a % 2).tolist(), [[1.0, 0.0], [1.0, 0.0]])
        self.assertTrue((a.sin() ** 2 + a.cos() ** 2).allclose(torch.ones(2, 2), atol=1e-5))
        self.assertTrue(torch.tensor([0.0]).erf().allclose(torch.tensor([0.0]), atol=1e-6))

        c = torch.tensor([float("nan"), float("inf"), -2.0, 5.0])
        self.assertEqual(c.isnan().tolist(), [True, False, False, False])
        self.assertEqual(c.isinf().tolist(), [False, True, False, False])
        self.assertEqual(c.isfinite().tolist(), [False, False, True, True])
        cleaned = c.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0)
        self.assertEqual(cleaned.clamp(-5.0, 5.0).tolist(), [0.0, 5.0, -2.0, 5.0])

    def test_shape_helpers_and_reductions(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertAlmostEqual(float(a.sum().item()), 21.0)
        self.assertAlmostEqual(float(a.mean().item()), 3.5)
        self.assertAlmostEqual(float(a.prod().item()), 720.0)
        self.assertEqual(a.argmax(dim=1).tolist(), [2, 2])
        self.assertEqual(a.argmin(dim=0).tolist(), [0, 0, 0])
        self.assertEqual(a.cumsum(dim=1).tolist(), [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]])
        self.assertEqual(a.cumprod(dim=1).tolist(), [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]])

        vals, idxs = a.topk(2, dim=1, largest=True)
        self.assertEqual(vals.tolist(), [[3.0, 2.0], [6.0, 5.0]])
        self.assertEqual(idxs.tolist(), [[2, 1], [2, 1]])

        tcat = torch.cat([a, a], dim=0)
        self.assertEqual(tcat.shape, (4, 3))
        tstack = torch.stack([a, a], dim=1)
        self.assertEqual(tstack.shape, (2, 2, 3))
        p = torch.pad(a, (1, 1, 1, 0), mode="constant", value=-1.0)
        self.assertEqual(p.shape, (3, 5))
        self.assertEqual(p[0].tolist(), [-1.0, -1.0, -1.0, -1.0, -1.0])

        u, inv, cnt = torch.unique(torch.tensor([3, 1, 2, 1, 3, 3]), sorted=True, return_inverse=True, return_counts=True)
        self.assertEqual(u.tolist(), [1, 2, 3])
        self.assertEqual(cnt.tolist(), [2, 1, 3])
        self.assertEqual(inv.tolist(), [2, 0, 1, 0, 2, 2])

        bc = torch.bincount(torch.tensor([0, 1, 1, 3]), minlength=5)
        self.assertEqual(bc.tolist(), [1, 2, 0, 1, 0])

        nz = torch.nonzero(torch.tensor([[0, 1], [2, 0]]))
        self.assertEqual(nz.tolist(), [[0, 1], [1, 0]])

    def test_matmul_and_einsum(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(4, 5)
        ab = a @ b
        self.assertEqual(ab.shape, (2, 3, 5))
        ein = torch.einsum("bij,jk->bik", a, b)
        self.assertTrue(ab.allclose(ein, atol=1e-5))

        tr = torch.einsum("ii->", torch.eye(4))
        self.assertAlmostEqual(float(tr.item()), 4.0)

        # Batched outer + trace + contract
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([4.0, 5.0])
        self.assertTrue(torch.einsum("i,j->ij", v1, v2).allclose(torch.outer(v1, v2)))


if __name__ == "__main__":
    unittest.main()
