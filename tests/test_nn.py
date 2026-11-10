"""Comprehensive tests for torch.nn modules, losses, activations, init, utils, and training."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TestNN(unittest.TestCase):
    def test_state_dict_hooks_and_containers(self):
        net = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.LayerNorm(8),
            nn.Linear(8, 2),
        )
        sd = net.state_dict()
        self.assertIn("0.weight", sd)
        self.assertIn("2.bias", sd)

        net2 = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.LayerNorm(8),
            nn.Linear(8, 2),
        )
        missing, unexpected = net2.load_state_dict(sd, strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

        x = torch.randn(3, 4)
        self.assertTrue(net(x).allclose(net2(x), atol=1e-6))

        # strict=False check
        partial = {"0.weight": sd["0.weight"], "extra.key": torch.zeros(2)}
        m2, u2 = net2.load_state_dict(partial, strict=False)
        self.assertIn("0.bias", m2)
        self.assertEqual(u2, ["extra.key"])

        ml = nn.ModuleList([nn.Linear(4, 4), nn.ReLU()])
        md = nn.ModuleDict({"proj": nn.Linear(4, 2)})
        pl = nn.ParameterList([nn.Parameter(torch.ones(2))])
        pd = nn.ParameterDict({"scale": nn.Parameter(torch.ones(2))})
        self.assertEqual(len(ml), 2)
        self.assertIn("proj", md)
        self.assertEqual(len(list(pl.parameters())), 1)
        self.assertEqual(len(list(pd.parameters())), 1)

    def test_linear_bilinear_lazy_conv_norm_pool(self):
        torch.manual_seed(0)
        # Linear, Bilinear, LazyLinear
        ll = nn.LazyLinear(5)
        x = torch.randn(2, 6)
        out_ll = ll(x)
        self.assertEqual(out_ll.shape, (2, 5))
        bi = nn.Bilinear(6, 4, 3)
        self.assertEqual(bi(x, torch.randn(2, 4)).shape, (2, 3))

        # Convolutions & transposed conv
        c1 = nn.Conv1d(3, 5, kernel_size=3, padding=1)
        self.assertEqual(c1(torch.randn(2, 3, 10)).shape, (2, 5, 10))
        c2 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.assertEqual(c2(torch.randn(2, 3, 8, 8)).shape, (2, 6, 8, 8))
        ct2 = nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2)
        self.assertEqual(ct2(torch.randn(2, 6, 4, 4)).shape, (2, 3, 8, 8))
        c3 = nn.Conv3d(2, 4, kernel_size=2)
        self.assertEqual(c3(torch.randn(1, 2, 4, 4, 4)).shape, (1, 4, 3, 3, 3))

        # Norms
        for norm_mod, inp in [
            (nn.BatchNorm1d(4), torch.randn(3, 4, requires_grad=True)),
            (nn.BatchNorm2d(4), torch.randn(3, 4, 5, 5, requires_grad=True)),
            (nn.BatchNorm3d(4), torch.randn(2, 4, 3, 3, 3, requires_grad=True)),
            (nn.LayerNorm(4), torch.randn(2, 3, 4, requires_grad=True)),
            (nn.RMSNorm(4), torch.randn(2, 3, 4, requires_grad=True)),
            (nn.GroupNorm(2, 4), torch.randn(2, 4, 3, 3, requires_grad=True)),
            (nn.InstanceNorm1d(4), torch.randn(2, 4, 6, requires_grad=True)),
            (nn.InstanceNorm2d(4), torch.randn(2, 4, 5, 5, requires_grad=True)),
        ]:
            y = norm_mod(inp)
            y.sum().backward()
            self.assertEqual(y.shape, inp.shape)

        # Pooling, Upsample, Padding, Flatten
        img = torch.randn(2, 4, 8, 8, requires_grad=True)
        for mod, exp_sh in [
            (nn.MaxPool2d(2), (2, 4, 4, 4)),
            (nn.AvgPool2d(2), (2, 4, 4, 4)),
            (nn.AdaptiveAvgPool2d((2, 3)), (2, 4, 2, 3)),
            (nn.AdaptiveMaxPool2d((2, 2)), (2, 4, 2, 2)),
            (nn.Upsample(scale_factor=2, mode="bilinear"), (2, 4, 16, 16)),
            (nn.ZeroPad2d(1), (2, 4, 10, 10)),
            (nn.ReflectionPad2d(1), (2, 4, 10, 10)),
            (nn.ReplicationPad2d(1), (2, 4, 10, 10)),
        ]:
            y = mod(img)
            self.assertEqual(y.shape, exp_sh)
            y.sum().backward()

        fl = nn.Flatten(1)
        uf = nn.Unflatten(1, (4, 8, 8))
        self.assertEqual(uf(fl(img)).shape, (2, 4, 8, 8))

    def test_activations_losses_and_sparse(self):
        x = torch.randn(4, 6, requires_grad=True)
        for act in [
            nn.ReLU(), nn.ReLU6(), nn.GELU(), nn.SiLU(), nn.Sigmoid(), nn.Tanh(),
            nn.ELU(), nn.CELU(), nn.SELU(), nn.LeakyReLU(0.1), nn.PReLU(6),
            nn.Mish(), nn.Softplus(), nn.Softsign(), nn.Hardtanh(), nn.Hardswish(),
            nn.Hardsigmoid(), nn.LogSigmoid(), nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1),
        ]:
            y = act(x)
            self.assertEqual(y.shape, x.shape)
            y.sum().backward()

        self.assertEqual(nn.GLU(dim=-1)(x).shape, (4, 3))

        # Losses
        logits = torch.randn(5, 4, requires_grad=True)
        tgt_idx = torch.tensor([0, 2, 1, 3, 0], dtype=torch.int64)
        tgt_soft = F.softmax(torch.randn(5, 4), dim=-1)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        l1 = ce(logits, tgt_idx)
        l2 = ce(logits, tgt_soft)
        (l1 + l2).backward()
        self.assertIsNotNone(logits.grad)

        p = torch.randn(5, 4, requires_grad=True)
        q = torch.randn(5, 4)
        for lmod in [
            nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss(), nn.HuberLoss(),
            nn.BCEWithLogitsLoss(),
        ]:
            lmod(p, q).backward()

        nn.BCELoss()(p.sigmoid(), q.sigmoid()).backward()
        nn.NLLLoss()(F.log_softmax(p, dim=-1), tgt_idx).backward()
        nn.KLDivLoss(reduction="batchmean")(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1)).backward()
        nn.CosineEmbeddingLoss()(p, q, torch.ones(5)).backward()
        nn.MarginRankingLoss()(p[:, 0], q[:, 0], torch.ones(5)).backward()
        nn.TripletMarginLoss()(p, q, q * 0.5).backward()

        emb = nn.Embedding(10, 4, padding_idx=0)
        ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int64)
        eo = emb(ids)
        eo.sum().backward()
        self.assertTrue((emb.weight.grad[0] == 0).all().item())

        ebag = nn.EmbeddingBag(10, 4, mode="mean")
        ebag(ids).sum().backward()
        self.assertIsNotNone(ebag.weight.grad)

    def test_rnn_lstm_gru_and_transformer(self):
        torch.manual_seed(3)
        seq = torch.randn(4, 5, 8, requires_grad=True)  # (N, L, E) batch_first=True
        for rnn_mod in [
            nn.RNN(8, 6, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(8, 6, num_layers=2, batch_first=True),
            nn.GRU(8, 6, num_layers=2, batch_first=True),
        ]:
            out, _ = rnn_mod(seq)
            out.sum().backward()
            self.assertEqual(out.shape[:2], (4, 5))

        mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        q = torch.randn(2, 4, 8, requires_grad=True)
        attn_out, attn_w = mha(q, q, q)
        attn_out.sum().backward()
        self.assertEqual(attn_out.shape, (2, 4, 8))
        self.assertEqual(attn_w.shape, (2, 4, 4))

        enc_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True)
        enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        eo = enc(q)
        self.assertEqual(eo.shape, (2, 4, 8))
        eo.sum().backward()

        tf = nn.Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        tgt = torch.randn(2, 3, 8, requires_grad=True)
        t_out = tf(q, tgt)
        self.assertEqual(t_out.shape, (2, 3, 8))
        t_out.sum().backward()

    def test_init_and_utils_training(self):
        w = torch.empty(16, 16)
        nn.init.xavier_uniform_(w)
        nn.init.kaiming_normal_(w)
        nn.init.orthogonal_(w)
        self.assertTrue((w @ w.t()).allclose(torch.eye(16), atol=1e-4))

        p1 = nn.Parameter(torch.tensor([3.0, 4.0]))
        p1.grad = torch.tensor([6.0, 8.0])
        total_n = nn.utils.clip_grad_norm_([p1], max_norm=5.0)
        self.assertAlmostEqual(float(total_n.item()), 10.0, places=4)
        self.assertAlmostEqual(float(p1.grad.norm().item()), 5.0, places=4)

        seqs = [torch.ones(2, 3), torch.ones(4, 3)]
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-1.0)
        self.assertEqual(padded.shape, (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
