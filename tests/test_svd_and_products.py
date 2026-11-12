"""torch.linalg.svd (shapes, orthonormal factors also for rank-deficient input) and the
rank-checked products mm / bmm / mv / dot / addmm / baddbmm."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
F64 = torch.float64


def eye(n):
    return torch.eye(n, dtype=F64)


def max_abs(t):
    return float(t.abs().max().item()) if t.numel() else 0.0


class SVDTestCase(unittest.TestCase):
    def check_svd(self, a, full_matrices, tol=1e-10):
        """U and Vh have the documented shapes and orthonormal columns/rows, S is sorted and
        U diag(S) Vh gives `a` back."""
        m, n = a.shape[-2:]
        k = min(m, n)
        u, s, vh = torch.linalg.svd(a, full_matrices=full_matrices)
        batch = tuple(a.shape[:-2])
        self.assertEqual(tuple(u.shape), batch + ((m, m) if full_matrices else (m, k)))
        self.assertEqual(tuple(s.shape), batch + (k,))
        self.assertEqual(tuple(vh.shape), batch + ((n, n) if full_matrices else (k, n)))
        self.assertEqual((u.dtype, s.dtype, vh.dtype), (a.dtype,) * 3)
        u, s, vh, a64 = u.double(), s.double(), vh.double(), a.double()
        self.assertLess(max_abs(u.transpose(-2, -1) @ u - eye(u.shape[-1])), tol)
        self.assertLess(max_abs(vh @ vh.transpose(-2, -1) - eye(vh.shape[-2])), tol)
        self.assertLess(max_abs((u[..., :, :k] * s.unsqueeze(-2)) @ vh[..., :k, :] - a64), tol * max(1.0, max_abs(a64)))
        self.assertTrue(bool((s >= 0).all().item()))
        if k > 1:
            self.assertTrue(bool((s[..., :-1] >= s[..., 1:]).all().item()))
        return u, s, vh


class TestSVDShapes(SVDTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_full_and_reduced_shapes(self):
        for shape in ((5, 3), (3, 5), (4, 4), (2, 4, 3), (2, 3, 4), (2, 3, 6, 2), (1, 4), (4, 1), (1, 1), (40, 25), (25, 40)):
            a = torch.randn(*shape, dtype=F64)
            for full in (True, False):
                with self.subTest(shape=shape, full_matrices=full):
                    self.check_svd(a, full)

    def test_full_matrices_is_the_default(self):
        u, s, vh = torch.linalg.svd(torch.randn(5, 3, dtype=F64))
        self.assertEqual((tuple(u.shape), tuple(s.shape), tuple(vh.shape)), ((5, 5), (3,), (3, 3)))
        u, s, vh = torch.linalg.svd(torch.randn(3, 5, dtype=F64))
        self.assertEqual((tuple(u.shape), tuple(s.shape), tuple(vh.shape)), ((3, 3), (3,), (5, 5)))

    def test_known_decomposition(self):
        a = torch.tensor([[3.0, 0.0], [0.0, -2.0], [0.0, 0.0]], dtype=F64)
        u, s, vh = self.check_svd(a, True)
        self.assertEqual(s.tolist(), [3.0, 2.0])
        self.assertAlmostEqual(abs(u[2, 2].item()), 1.0, places=12)     # the completed third column
        # singular values of [[1, 2], [3, 4]]: sqrt(15 +- sqrt(221))
        s = torch.linalg.svdvals(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=F64))
        self.assertAlmostEqual(s[0].item(), (15 + 221 ** 0.5) ** 0.5, places=12)
        self.assertAlmostEqual(s[1].item(), (15 - 221 ** 0.5) ** 0.5, places=12)

    def test_very_large_and_very_small_entries(self):
        # squared column norms of such matrices overflow / vanish in double precision
        base = torch.randn(7, 4, dtype=F64)
        want = torch.linalg.svdvals(base)
        for scale in (1e150, 1e-150, 1e300, 1e-300):
            with self.subTest(scale=scale):
                a = base * scale
                for full in (True, False):
                    u, s, vh = torch.linalg.svd(a, full_matrices=full)
                    self.assertLess(max_abs(u.t() @ u - eye(u.shape[1])), 1e-10)
                    self.assertLess(max_abs(vh @ vh.t() - eye(4)), 1e-10)
                    self.assertLess(max_abs((u[:, :4] * s) @ vh / scale - base), 1e-10)
                    self.assertLess(max_abs(s / scale - want), 1e-10)

    def test_float32_input(self):
        a = torch.randn(6, 4)
        for full in (True, False):
            u, s, vh = self.check_svd(a, full, tol=1e-5)
        self.assertEqual(s.dtype, F64)                                   # check_svd hands back doubles
        self.assertEqual(torch.linalg.svd(a)[1].dtype, torch.float32)

    def test_non_contiguous_input(self):
        base = torch.randn(6, 8, dtype=F64)
        view = base.t()[1:7:2]                                           # (3, 6), strided
        self.assertFalse(view.is_contiguous())
        self.check_svd(view, True)
        u, s, vh = torch.linalg.svd(view, full_matrices=False)
        u2, s2, vh2 = torch.linalg.svd(view.contiguous(), full_matrices=False)
        self.assertEqual(s.tolist(), s2.tolist())

    def test_tensor_svd_returns_v_not_vh(self):
        a = torch.randn(5, 3, dtype=F64)
        u, s, v = a.svd()
        self.assertEqual((tuple(u.shape), tuple(v.shape)), ((5, 3), (3, 3)))
        self.assertLess(max_abs((u * s) @ v.t() - a), 1e-12)
        u, s, v = a.svd(some=False)
        self.assertEqual((tuple(u.shape), tuple(v.shape)), ((5, 5), (3, 3)))
        wide = torch.randn(3, 5, dtype=F64)
        u, s, v = wide.svd(some=False)
        self.assertEqual((tuple(u.shape), tuple(v.shape)), ((3, 3), (5, 5)))
        self.assertLess(max_abs((u * s) @ v[:, :3].t() - wide), 1e-12)

    def test_empty_matrices(self):
        for shape, shapes in (
            ((0, 3), ((0, 0), (0,), (3, 3))),
            ((3, 0), ((3, 3), (0,), (0, 0))),
            ((2, 0, 3), ((2, 0, 0), (2, 0), (2, 3, 3))),
        ):
            u, s, vh = torch.linalg.svd(torch.zeros(*shape, dtype=F64))
            self.assertEqual((tuple(u.shape), tuple(s.shape), tuple(vh.shape)), shapes)
        # the factor that is not empty is still orthogonal: the identity
        self.assertEqual(torch.linalg.svd(torch.zeros(0, 3, dtype=F64))[2].tolist(), eye(3).tolist())
        self.assertEqual(torch.linalg.svd(torch.zeros(3, 0, dtype=F64))[0].tolist(), eye(3).tolist())


class TestSVDRankDeficient(SVDTestCase):
    """The vectors that go with zero singular values have to complete an orthonormal basis.
    The old code left them as coordinate vectors, which are not orthogonal to the others."""

    def setUp(self):
        torch.manual_seed(1)

    def test_rank_one(self):
        tall = torch.outer(torch.randn(5, dtype=F64), torch.randn(3, dtype=F64))
        wide = torch.outer(torch.randn(3, dtype=F64), torch.randn(5, dtype=F64))
        for a in (tall, wide, tall[:3], torch.stack([tall, 2 * tall])):
            for full in (True, False):
                with self.subTest(shape=tuple(a.shape), full_matrices=full):
                    u, s, vh = self.check_svd(a, full)
                    self.assertLess(max_abs(s[..., 1:]), 1e-12 * max_abs(s))

    def test_rank_two_square(self):
        a = torch.randn(4, 2, dtype=F64) @ torch.randn(2, 4, dtype=F64)
        for full in (True, False):
            u, s, vh = self.check_svd(a, full)
            self.assertLess(s[2].item(), 1e-12 * s[0].item())

    def test_zero_matrix(self):
        for shape in ((3, 2), (2, 3), (3, 3), (2, 3, 2)):
            for full in (True, False):
                with self.subTest(shape=shape, full_matrices=full):
                    u, s, vh = self.check_svd(torch.zeros(*shape, dtype=F64), full)
                    self.assertEqual(max_abs(s), 0.0)

    def test_columns_that_are_coordinate_vectors(self):
        # the columns already span e_4, e_5 and e_1: completing U with e_4 and e_5 again, as the
        # old code did, repeated two columns
        a = eye(5)[:, [3, 4, 0]]
        u, s, vh = self.check_svd(a, True)
        self.assertEqual(s.tolist(), [1.0, 1.0, 1.0])
        self.check_svd(a.t().contiguous(), True)
        self.check_svd(eye(5)[:, :3], True)
        self.check_svd(eye(5)[:2], True)

    def test_repeated_singular_values(self):
        u, s, vh = self.check_svd(torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0], dtype=F64)), True)
        self.assertEqual(s.tolist(), [2.0, 2.0, 2.0, 1.0])

    def test_functions_built_on_svd(self):
        a = torch.outer(torch.randn(5, dtype=F64), torch.randn(3, dtype=F64))
        self.assertEqual(torch.linalg.matrix_rank(a).item(), 1)
        self.assertEqual(torch.linalg.matrix_rank(torch.randn(5, 3, dtype=F64)).item(), 3)
        pinv = torch.linalg.pinv(a)
        self.assertEqual(tuple(pinv.shape), (3, 5))
        self.assertLess(max_abs(a @ pinv @ a - a), 1e-10)
        self.assertLess(max_abs(pinv @ a @ pinv - pinv), 1e-10)
        nuc = torch.linalg.matrix_norm(a, ord="nuc")
        self.assertAlmostEqual(nuc.item(), torch.linalg.svdvals(a).sum().item(), places=10)


class TestSVDErrors(unittest.TestCase):
    def test_bad_input_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "at least 2 dimensions"):
            torch.linalg.svd(torch.ones(3))
        with self.assertRaisesRegex(RuntimeError, "floating point"):
            torch.linalg.svd(torch.ones(2, 2, dtype=torch.int64))
        for bad in (float("nan"), float("inf")):
            a = torch.ones(3, 2)
            a[1, 1] = bad
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                torch.linalg.svd(a)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                a.svd()


class TestProductRanks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.v = torch.tensor([1.0, 2.0, 3.0])
        self.m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.b = torch.randn(4, 2, 3)

    def test_values(self):
        v, m, b = self.v, self.m, self.b
        self.assertEqual(torch.dot(v, v).item(), 14.0)
        self.assertEqual(v.dot(v).shape, ())
        self.assertEqual(torch.mv(m, v).tolist(), [14.0, 32.0])
        self.assertEqual(torch.mm(m, m.t()).tolist(), [[14.0, 32.0], [32.0, 77.0]])
        self.assertEqual(m.mm(m.t()).tolist(), [[14.0, 32.0], [32.0, 77.0]])
        got = torch.bmm(b, b.transpose(1, 2))
        self.assertEqual(tuple(got.shape), (4, 2, 2))
        self.assertTrue(got.allclose(b @ b.transpose(1, 2)))

    def test_dot_wants_two_vectors(self):
        v, m = self.v, self.m
        for a, b in ((m, m), (m, v), (v, m), (torch.tensor(2.0), v), (m.t(), m), (self.b, self.b)):
            with self.subTest(shapes=(tuple(a.shape), tuple(b.shape))):
                with self.assertRaisesRegex(RuntimeError, "1D tensors expected, but got"):
                    torch.dot(a, b)
        with self.assertRaisesRegex(RuntimeError, "inconsistent tensor size"):
            torch.dot(v, torch.ones(4))
        with self.assertRaises(TypeError):
            torch.dot(v, [1.0, 2.0, 3.0])

    def test_mv_wants_a_matrix_and_a_vector(self):
        v, m = self.v, self.m
        for a, b in ((v, v), (m, m.t()), (self.b, v), (m, torch.tensor(1.0)), (v, m)):
            with self.subTest(shapes=(tuple(a.shape), tuple(b.shape))):
                with self.assertRaisesRegex(RuntimeError, "mv: expected 2D matrix and 1D vector"):
                    torch.mv(a, b)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            torch.mv(m, torch.ones(2))

    def test_mm_wants_two_matrices(self):
        v, m = self.v, self.m
        for a, b in ((m, v), (v, m.t()), (self.b, self.b.transpose(1, 2)), (v, v), (self.b, m.t())):
            with self.subTest(shapes=(tuple(a.shape), tuple(b.shape))):
                with self.assertRaisesRegex(RuntimeError, "mm: expected 2D tensors"):
                    torch.mm(a, b)
        with self.assertRaisesRegex(RuntimeError, r"shapes cannot be multiplied \(2x3 and 2x3\)"):
            torch.mm(m, m)

    def test_bmm_wants_two_batches_of_matrices(self):
        m, b = self.m, self.b
        for x, y in ((m, m.t()), (b, m.t()), (m, b.transpose(1, 2)), (torch.randn(2, 4, 2, 3), torch.randn(2, 4, 3, 2))):
            with self.subTest(shapes=(tuple(x.shape), tuple(y.shape))):
                with self.assertRaisesRegex(RuntimeError, "bmm: expected 3D tensors"):
                    torch.bmm(x, y)
        with self.assertRaisesRegex(RuntimeError, "Expected size for first two dimensions of batch2"):
            torch.bmm(b, b)                                   # inner sizes differ
        with self.assertRaisesRegex(RuntimeError, "Expected size for first two dimensions of batch2"):
            torch.bmm(b, torch.randn(1, 3, 2))                # bmm does not broadcast the batch

    def test_matmul_itself_still_broadcasts(self):
        m, b, v = self.m, self.b, self.v
        self.assertEqual(tuple((b @ v).shape), (4, 2))
        self.assertEqual(tuple((b @ m.t()).shape), (4, 2, 2))
        self.assertEqual(tuple((v @ b.transpose(1, 2)).shape), (4, 2))
        self.assertEqual((v @ v).item(), 14.0)

    def test_dot_keeps_the_dtype_and_the_gradient(self):
        for dtype in (torch.int32, torch.int64, torch.float64, torch.float16, torch.bfloat16):
            a = torch.tensor([1, 2, 3]).to(dtype)
            got = torch.dot(a, a)
            self.assertEqual((got.dtype, got.item()), (dtype, 14))
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        torch.dot(a, b).backward()
        self.assertEqual((a.grad.tolist(), b.grad.tolist()), ([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]))
        (g,) = torch.autograd.grad(torch.dot(a, a * a), a, create_graph=True)        # sum a^3 -> 3 a^2
        self.assertEqual(g.tolist(), [3.0, 12.0, 27.0])
        (gg,) = torch.autograd.grad(g.sum(), a)
        self.assertEqual(gg.tolist(), [6.0, 12.0, 18.0])

    def test_addmm_and_baddbmm(self):
        m, b = self.m, self.b
        self.assertEqual(torch.addmm(torch.ones(2), m, m.t(), beta=2.0, alpha=0.5).tolist(), [[9.0, 18.0], [18.0, 40.5]])
        with self.assertRaisesRegex(RuntimeError, "addmm: expected 2D matrices"):
            torch.addmm(torch.zeros(4, 2, 2), b, b.transpose(1, 2))
        with self.assertRaisesRegex(RuntimeError, "addmm: expected 2D matrices"):
            torch.addmm(torch.zeros(2), m, self.v)
        with self.assertRaisesRegex(RuntimeError, "shapes cannot be multiplied"):
            torch.addmm(torch.zeros(2, 3), m, m)
        with self.assertRaisesRegex(RuntimeError, "cannot be broadcast"):
            torch.addmm(torch.zeros(3, 2), m, m.t())
        bt = b.transpose(1, 2)
        inp = torch.randn(4, 2, 2)
        self.assertTrue(torch.baddbmm(inp, b, bt, beta=0.5, alpha=2.0).allclose(0.5 * inp + 2.0 * (b @ bt)))
        self.assertTrue(inp.baddbmm(b, bt).allclose(inp + b @ bt))
        self.assertEqual(tuple(torch.baddbmm(torch.zeros(2), b, bt).shape), (4, 2, 2))
        with self.assertRaisesRegex(RuntimeError, "baddbmm: expected 3D tensors"):
            torch.baddbmm(m @ m.t(), m, m.t())
        with self.assertRaisesRegex(RuntimeError, "Expected size for first two dimensions of batch2"):
            torch.baddbmm(inp, b[:1], bt)
        with self.assertRaisesRegex(RuntimeError, "cannot be broadcast"):
            torch.baddbmm(torch.zeros(3, 2, 2), b, bt)
        # beta = 0 ignores the input, nan and all
        got = torch.baddbmm(torch.full((4, 2, 2), float("nan")), b, bt, beta=0)
        self.assertTrue(got.allclose(b @ bt))

    def test_baddbmm_has_a_gradient(self):
        inp = torch.randn(2, 2, 2, requires_grad=True)
        x = torch.randn(2, 2, 3, requires_grad=True)
        y = torch.randn(2, 3, 2, requires_grad=True)
        out = torch.baddbmm(inp, x, y, beta=0.5, alpha=2.0)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertEqual(inp.grad.tolist(), torch.full((2, 2, 2), 0.5).tolist())
        ones = torch.ones(2, 2, 2)
        self.assertTrue(x.grad.allclose(2.0 * ones @ y.detach().transpose(1, 2)))
        self.assertTrue(y.grad.allclose(2.0 * x.detach().transpose(1, 2) @ ones))


class TestEmptyInnerDimension(unittest.TestCase):
    """(M, 0) @ (0, N) is an (M, N) matrix of zeros. For float32, float16 and bfloat16 the
    product used to reach ggml, which aborted the process - so this runs in a child."""

    def test_products_over_an_empty_dimension(self):
        code = (
            "import torch\n"
            "Z = torch.zeros\n"
            "for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64):\n"
            "    r = Z(2, 0, dtype=dt) @ Z(0, 3, dtype=dt)\n"
            "    assert (tuple(r.shape), r.dtype, r.tolist()) == ((2, 3), dt, [[0] * 3] * 2), (dt, r)\n"
            "assert (Z(0) @ Z(0)).tolist() == 0.0\n"
            "assert torch.dot(Z(0), Z(0)).tolist() == 0.0\n"
            "assert torch.mv(Z(2, 0), Z(0)).tolist() == [0.0, 0.0]\n"
            "assert (Z(0) @ Z(0, 3)).tolist() == [0.0, 0.0, 0.0]\n"
            "assert torch.mm(Z(2, 0), Z(0, 3)).tolist() == [[0.0] * 3] * 2\n"
            "assert tuple(torch.bmm(Z(4, 2, 0), Z(4, 0, 3)).shape) == (4, 2, 3)\n"
            "assert torch.addmm(torch.ones(2, 3), Z(2, 0), Z(0, 3)).tolist() == [[1.0] * 3] * 2\n"
            "assert torch.baddbmm(torch.ones(1, 2, 3), Z(1, 2, 0), Z(1, 0, 3)).tolist() == [[[1.0] * 3] * 2]\n"
            "assert torch.einsum('ik,kj->ij', Z(2, 0), Z(0, 3)).tolist() == [[0.0] * 3] * 2\n"
            "assert torch.nn.functional.linear(Z(2, 0), Z(3, 0), torch.ones(3)).tolist() == [[1.0] * 3] * 2\n"
            "assert tuple((Z(0, 3) @ Z(3, 2)).shape) == (0, 2) and tuple((Z(2, 3) @ Z(3, 0)).shape) == (2, 0)\n"
            "a = Z(2, 0, requires_grad=True); b = Z(0, 3, requires_grad=True)\n"
            "(a @ b).sum().backward()\n"
            "assert tuple(a.grad.shape) == (2, 0) and tuple(b.grad.shape) == (0, 3)\n"
            "print('EMPTY-OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, f"child died with {proc.returncode}:\n{proc.stderr[-2000:]}")
        self.assertIn("EMPTY-OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
