"""Comprehensive tests for torch.linalg, torch.fft, and torch.special (including autograd)."""

import math
import unittest

import torch
import torch.fft as fft
import torch.linalg as linalg
import torch.special as special
from torch.autograd import gradcheck


class TestLinalgFftSpecial(unittest.TestCase):
    def test_linalg_solvers_and_decompositions(self):
        torch.manual_seed(5)
        a = torch.randn(3, 3, dtype=torch.float64) + 4.0 * torch.eye(3, dtype=torch.float64)
        spd = a @ a.t() + torch.eye(3, dtype=torch.float64)
        b = torch.randn(3, 2, dtype=torch.float64)

        inv_a = linalg.inv(a)
        self.assertTrue((a @ inv_a).allclose(torch.eye(3, dtype=torch.float64), atol=1e-5))

        x = linalg.solve(a, b)
        self.assertTrue((a @ x).allclose(b, atol=1e-5))

        L = linalg.cholesky(spd)
        self.assertTrue((L @ L.t()).allclose(spd, atol=1e-5))

        Q, R = linalg.qr(a)
        self.assertTrue((Q @ Q.t()).allclose(torch.eye(3, dtype=torch.float64), atol=1e-5))
        self.assertTrue((Q @ R).allclose(a, atol=1e-5))

        U, S, Vh = linalg.svd(a)
        self.assertTrue((U @ torch.diag(S) @ Vh).allclose(a, atol=1e-5))

        evals, evecs = linalg.eigh(spd)
        self.assertTrue((evecs @ torch.diag(evals) @ evecs.t()).allclose(spd, atol=1e-5))

        d = linalg.det(spd)
        sign, logabs = linalg.slogdet(spd)
        self.assertAlmostEqual(float(sign.item()), 1.0, places=5)
        self.assertAlmostEqual(math.exp(float(logabs.item())), float(d.item()), places=4)

        self.assertEqual(int(linalg.matrix_rank(torch.eye(4)).item()), 4)
        self.assertTrue(linalg.pinv(a).allclose(inv_a, atol=1e-4))
        self.assertTrue((a @ linalg.lstsq(a, b).solution).allclose(b, atol=1e-4))
        self.assertGreater(float(linalg.cond(spd).item()), 1.0 - 1e-5)
        self.assertTrue(linalg.matrix_power(a, 2).allclose(a @ a, atol=1e-5))
        self.assertTrue(linalg.multi_dot([a, spd, inv_a]).allclose(a @ spd @ inv_a, atol=1e-5))

        # Autograd through inv, solve, det
        ag_a = a.detach().clone().requires_grad_(True)
        ag_b = b.detach().clone().requires_grad_(True)
        self.assertTrue(gradcheck(lambda m: linalg.inv(m).sum(), (ag_a,), eps=1e-5, atol=1e-4))
        self.assertTrue(gradcheck(lambda m, r: linalg.solve(m, r).sum(), (ag_a, ag_b), eps=1e-5, atol=1e-4))
        self.assertTrue(gradcheck(lambda m: linalg.det(m), (ag_a,), eps=1e-5, atol=1e-4))

    def test_fft_roundtrip_and_autograd(self):
        x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
        spec = fft.fft(x, dim=-1)
        rec = fft.ifft(spec, dim=-1).real
        self.assertTrue(rec.allclose(x, atol=1e-4))

        rspec = fft.rfft(x, dim=-1)
        self.assertEqual(rspec.shape, (2, 5))
        rrec = fft.irfft(rspec, n=8, dim=-1)
        self.assertTrue(rrec.allclose(x, atol=1e-4))

        # Backward through FFT
        (spec.real.sum() + spec.imag.sum()).backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # 2D FFT + shift + freqs
        img = torch.randn(4, 6)
        f2 = fft.fftn(img, dim=(0, 1))
        rec2 = fft.ifftn(f2, dim=(0, 1)).real
        self.assertTrue(rec2.allclose(img, atol=1e-4))
        self.assertEqual(fft.fftshift(fft.fftfreq(6)).shape, (6,))
        self.assertEqual(fft.rfftfreq(6).shape, (4,))

    def test_special_functions(self):
        x = torch.tensor([-1.5, 0.0, 0.5, 2.0])
        self.assertTrue(special.erfinv(special.erf(x)).allclose(x, atol=1e-4))
        self.assertTrue((special.erf(x) + special.erfc(x)).allclose(torch.ones_like(x), atol=1e-5))
        p = torch.tensor([0.2, 0.5, 0.8])
        self.assertTrue(special.expit(special.logit(p)).allclose(p, atol=1e-5))
        pos = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # lgamma(n) == log((n-1)!)
        self.assertTrue(special.gammaln(pos).allclose(torch.tensor([0.0, 0.0, math.log(2.0), math.log(6.0)]), atol=1e-4))
        self.assertAlmostEqual(float(special.sinc(torch.tensor(0.0)).item()), 1.0, places=5)
        self.assertGreater(float(special.i0(torch.tensor(1.0)).item()), 1.2)
        self.assertEqual(special.digamma(pos).shape, (4,))
        self.assertEqual(special.entr(p).shape, (3,))
        self.assertEqual(special.xlogy(pos, pos).shape, (4,))
        self.assertAlmostEqual(float(special.ndtr(torch.tensor(0.0)).item()), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
