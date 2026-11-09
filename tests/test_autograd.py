"""Comprehensive autograd, gradcheck, gradgradcheck, functional AD, and checkpointing tests."""

import math
import unittest

import torch
import torch.nn.functional as F
from torch.autograd import Function, grad, gradcheck, gradgradcheck
from torch.autograd.functional import hessian, hvp, jacobian, jvp, vhp, vjp
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class TestAutograd(unittest.TestCase):
    def test_basic_backward_hooks_and_contexts(self):
        x = torch.tensor([2.0, -3.0, 4.0], requires_grad=True)
        seen = []
        h = x.register_hook(lambda g: seen.append(g.clone()) or g * 2.0)
        y = (x * x).sum()
        y.backward()
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].allclose(torch.tensor([4.0, -6.0, 8.0])))
        self.assertTrue(x.grad.allclose(torch.tensor([8.0, -12.0, 16.0])))
        h.remove()

        with torch.no_grad():
            z = x * 5.0
            self.assertFalse(z.requires_grad)

        with torch.inference_mode():
            z2 = x + 1.0
            self.assertFalse(z2.requires_grad)

        # retain_grad on intermediate
        a = torch.tensor([1.5, -2.5], requires_grad=True)
        b = a * 3.0
        b.retain_grad()
        c = (b ** 2).sum()
        c.backward()
        self.assertTrue(b.grad.allclose(2.0 * b.detach()))
        self.assertTrue(a.grad.allclose(6.0 * b.detach()))

    def test_custom_function_and_higher_order(self):
        class ScaleSquare(Function):
            @staticmethod
            def forward(ctx, x, k):
                ctx.save_for_backward(x)
                ctx.k = k
                return k * (x ** 2)

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * (2.0 * ctx.k * x), None

        x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        y = ScaleSquare.apply(x, 2.5).sum()
        g1 = grad(y, x, create_graph=True)[0]
        self.assertTrue(g1.allclose(torch.tensor([5.0, -10.0, 15.0])))
        g2 = grad(g1.sum(), x)[0]
        self.assertTrue(g2.allclose(torch.tensor([5.0, 5.0, 5.0])))

    def test_gradcheck_elementwise_and_reductions(self):
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        y = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda a, b: (a * b + (a ** 2).sin() - (b.abs() + 0.5).log()).sum(), (x, y)))

        # Reductions including prod with zeros and single zero
        xp = torch.tensor([[2.0, 0.0, 3.0], [1.5, 4.0, 2.0]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda t: t.prod(dim=1).sum(), (xp,)))
        self.assertTrue(gradcheck(lambda t: t.logsumexp(dim=-1), (x,)))
        self.assertTrue(gradcheck(lambda t: t.var(dim=-1, unbiased=True), (x,)))
        self.assertTrue(gradcheck(lambda t: t.norm(dim=-1), (x,)))
        self.assertTrue(gradcheck(lambda t: t.cumsum(dim=-1), (x,)))

        # gradgradcheck
        xg = torch.tensor([0.7, -1.2, 0.4], dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradgradcheck(lambda t: (t ** 4 + t.sin()).sum(), (xg,)))

    def test_gradcheck_nn_ops(self):
        torch.manual_seed(7)
        # Linear / matmul
        a = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, wt, bs: F.linear(x, wt, bs), (a, w, b)))

        # Softmax / log_softmax / cross_entropy (float probs)
        logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda z: F.softmax(z, dim=-1), (logits,)))
        self.assertTrue(gradcheck(lambda z: F.log_softmax(z, dim=-1), (logits,)))
        targets = torch.tensor([1, 3, 0], dtype=torch.int64)
        self.assertTrue(gradcheck(lambda z: F.cross_entropy(z, targets), (logits,)))

        # LayerNorm, RMSNorm, GroupNorm, BatchNorm
        ln_x = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        ln_w = torch.randn(6, dtype=torch.float64, requires_grad=True)
        ln_b = torch.randn(6, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, wt, bs: F.layer_norm(x, (6,), wt, bs), (ln_x, ln_w, ln_b)))
        self.assertTrue(gradcheck(lambda x, wt: F.rms_norm(x, (6,), wt), (ln_x, ln_w)))

        gn_x = torch.randn(2, 4, 3, 3, dtype=torch.float64, requires_grad=True)
        gn_w = torch.randn(4, dtype=torch.float64, requires_grad=True)
        gn_b = torch.randn(4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, wt, bs: F.group_norm(x, 2, wt, bs), (gn_x, gn_w, gn_b)))

        bn_x = torch.randn(3, 4, 2, 2, dtype=torch.float64, requires_grad=True)
        bn_w = torch.randn(4, dtype=torch.float64, requires_grad=True)
        bn_b = torch.randn(4, dtype=torch.float64, requires_grad=True)
        rm = torch.zeros(4, dtype=torch.float64)
        rv = torch.ones(4, dtype=torch.float64)
        self.assertTrue(gradcheck(lambda x, wt, bs: F.batch_norm(x, rm.clone(), rv.clone(), wt, bs, training=True), (bn_x, bn_w, bn_b)))

    def test_gradcheck_conv_pool_attention_rope(self):
        torch.manual_seed(11)
        # Conv1d, Conv2d, ConvTranspose2d, Conv3d
        x1 = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
        w1 = torch.randn(4, 3, 3, dtype=torch.float64, requires_grad=True)
        b1 = torch.randn(4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, wt, bs: F.conv1d(x, wt, bs, padding=1), (x1, w1, b1)))

        x2 = torch.randn(2, 2, 5, 5, dtype=torch.float64, requires_grad=True)
        w2 = torch.randn(3, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        b2 = torch.randn(3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, wt, bs: F.conv2d(x, wt, bs, padding=1), (x2, w2, b2)))
        self.assertTrue(gradcheck(lambda x, wt, bs: F.conv_transpose2d(x, torch.randn_like(w2.transpose(0, 1)) if False else wt.transpose(0, 1), bs, padding=1), (x2, w2, b2)))

        x3 = torch.randn(1, 2, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        w3 = torch.randn(2, 2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, wt: F.conv3d(x, wt), (x3, w3)))

        # Pool + Upsample
        xp = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.avg_pool2d(x, 2), (xp,)))
        self.assertTrue(gradcheck(lambda x: F.interpolate(x, size=(6, 6), mode="bilinear", align_corners=False), (xp,)))

        # FlashAttention & SwiGLU & RoPE
        q = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda qq, kk, vv: F.scaled_dot_product_attention(qq, kk, vv, is_causal=True), (q, k, v)))

        g_in = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        u_in = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda g, u: F.swiglu(g, u), (g_in, u_in)))

        rx = torch.randn(1, 2, 4, 8, dtype=torch.float32, requires_grad=True)
        pos = torch.arange(4, dtype=torch.int32)
        ro = F.rope(rx, pos, n_dims=8)
        ro.sum().backward()
        self.assertIsNotNone(rx.grad)
        self.assertTrue(gradcheck(lambda x: F.rope(x, pos, n_dims=8), (rx,), eps=1e-3, atol=1e-3, rtol=1e-2))

    def test_gradcheck_shape_ops_and_functional_ad(self):
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda t: F.pad(t, (1, 1, 0, 2), value=0.5), (x,)))
        self.assertTrue(gradcheck(lambda t: t.flip(dims=(1, 2)).roll(shifts=1, dims=2), (x,)))

        idx = torch.tensor([0, 2], dtype=torch.int64)
        emb_w = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda w: F.embedding(idx, w), (emb_w,)))

        # Functional AD: vjp, jvp, jacobian, hessian, hvp, vhp
        p = torch.tensor([1.0, 2.0], dtype=torch.float64)
        fn = lambda z: (z ** 3).sum()
        val, vjp_res = vjp(fn, p, v=torch.tensor(1.0, dtype=torch.float64))
        self.assertTrue(vjp_res.allclose(torch.tensor([3.0, 12.0], dtype=torch.float64)))
        val2, jvp_res = jvp(fn, p, v=torch.tensor([1.0, 0.5], dtype=torch.float64))
        self.assertAlmostEqual(float(jvp_res.item()), 9.0, places=5)

        jac = jacobian(lambda z: torch.stack([z[0] * z[1], z[0] + z[1] ** 2]), p)
        self.assertTrue(jac.allclose(torch.tensor([[2.0, 1.0], [1.0, 4.0]], dtype=torch.float64), atol=1e-5))
        hes = hessian(lambda z: z[0] ** 2 + 3.0 * z[0] * z[1] + z[1] ** 3, p)
        self.assertTrue(hes.allclose(torch.tensor([[2.0, 3.0], [3.0, 12.0]], dtype=torch.float64), atol=1e-4))
        _, hvp_res = hvp(lambda z: (z ** 3).sum(), p, v=torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.assertTrue(hvp_res.allclose(torch.tensor([6.0, 24.0], dtype=torch.float64), atol=1e-4))
        _, vhp_res = vhp(lambda z: (z ** 3).sum(), p, v=torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.assertTrue(vhp_res.allclose(torch.tensor([6.0, 24.0], dtype=torch.float64), atol=1e-4))

    def test_activation_checkpointing(self):
        x = torch.randn(4, 6, requires_grad=True)
        w1 = torch.randn(6, 6, requires_grad=True)
        w2 = torch.randn(6, 6, requires_grad=True)

        def block1(t):
            return (t @ w1).relu()

        def block2(t):
            return (t @ w2).tanh()

        ref = block2(block1(x)).sum()
        ref.backward()
        ref_gx, ref_gw1, ref_gw2 = x.grad.clone(), w1.grad.clone(), w2.grad.clone()

        x.grad = w1.grad = w2.grad = None
        out = checkpoint_sequential([block1, block2], 2, x)
        out.sum().backward()
        self.assertTrue(x.grad.allclose(ref_gx, atol=1e-5))
        self.assertTrue(w1.grad.allclose(ref_gw1, atol=1e-5))
        self.assertTrue(w2.grad.allclose(ref_gw2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
