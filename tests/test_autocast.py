"""torch.autocast casts the operands of the operations PyTorch's autocast lists name - per device
type and per thread - and gradients come back to the parameters in their own dtype.
GradScaler unscales the gradients of an optimizer once per step."""

import threading
import unittest
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast

BF16, F16, F32, F64 = torch.bfloat16, torch.float16, torch.float32, torch.float64


def dtype_of(out):
    return (out[0] if isinstance(out, (tuple, list)) else out).dtype


class AutocastTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, 8)
        self.seq = torch.randn(2, 5, 8)
        self.img = torch.randn(2, 3, 8, 8)

    def tearDown(self):
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    def assert_dtypes(self, cases, inside, outside, dtype=BF16):
        """Every case gives `inside` in an autocast block and `outside` without one."""
        with autocast("cpu", dtype=dtype):
            for name, fn in cases.items():
                with self.subTest(name):
                    self.assertEqual(dtype_of(fn()), inside)
        for name, fn in cases.items():
            with self.subTest(name + " (no autocast)"):
                self.assertEqual(dtype_of(fn()), outside)


class TestAutocastState(AutocastTestCase):
    def test_enabled_only_inside_the_block(self):
        self.assertFalse(torch.is_autocast_enabled())
        with autocast("cpu", dtype=BF16) as ctx:
            self.assertIsInstance(ctx, autocast)
            self.assertTrue(torch.is_autocast_enabled())
            self.assertTrue(torch.amp.is_autocast_enabled("cpu"))
            self.assertEqual(torch.get_autocast_dtype("cpu"), BF16)
        self.assertFalse(torch.is_autocast_enabled())

    def test_defaults(self):
        self.assertEqual(autocast("cpu").dtype, BF16)
        self.assertEqual(autocast("cuda").dtype, F16)
        self.assertEqual(torch.get_autocast_dtype("cpu"), BF16)
        self.assertEqual(torch.get_autocast_dtype("cuda"), F16)
        self.assertIs(torch.autocast, autocast)
        self.assertEqual(autocast(torch.device("cpu")).device_type, "cpu")
        with self.assertRaises(ValueError):
            autocast(3)

    def test_state_is_restored_after_an_exception(self):
        with self.assertRaises(KeyError):
            with autocast("cpu"):
                raise KeyError("boom")
        self.assertFalse(torch.is_autocast_enabled())

    def test_nesting(self):
        with autocast("cpu", dtype=BF16):
            with autocast("cpu", dtype=F16):
                self.assertEqual(torch.get_autocast_dtype(), F16)
                with autocast("cpu", enabled=False):
                    self.assertFalse(torch.is_autocast_enabled())
                    self.assertEqual((self.x @ self.x.t()).dtype, F32)
                self.assertEqual((self.x @ self.x.t()).dtype, F16)
            self.assertEqual(torch.get_autocast_dtype(), BF16)
            self.assertEqual((self.x @ self.x.t()).dtype, BF16)

    def test_the_same_object_can_be_entered_again(self):
        ctx = autocast("cpu")
        with ctx:
            with ctx:
                self.assertTrue(torch.is_autocast_enabled())
            self.assertTrue(torch.is_autocast_enabled())
        self.assertFalse(torch.is_autocast_enabled())

    def test_device_types_are_separate(self):
        with autocast("cuda", dtype=F16):
            self.assertTrue(torch.is_autocast_enabled("cuda"))
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertEqual((self.x @ self.x.t()).dtype, F32)       # CPU tensors: untouched
            self.assertEqual(self.x.to(BF16).sum().dtype, BF16)
            with autocast("cpu", dtype=BF16):
                self.assertTrue(torch.is_autocast_enabled("cuda"))
                self.assertEqual((self.x @ self.x.t()).dtype, BF16)
            self.assertTrue(torch.is_autocast_enabled("cuda"))
        with torch.cuda.amp.autocast():
            self.assertTrue(torch.is_autocast_enabled("cuda"))
            self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_each_thread_has_its_own_state(self):
        seen = {}

        def worker():
            seen["enabled"] = torch.is_autocast_enabled()
            seen["dtype"] = (self.x @ self.x.t()).dtype
            with autocast("cpu", dtype=F16):
                seen["inside"] = (self.x @ self.x.t()).dtype

        with autocast("cpu", dtype=BF16):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertEqual((self.x @ self.x.t()).dtype, BF16)
        self.assertEqual(seen, {"enabled": False, "dtype": F32, "inside": F16})

    def test_one_object_shared_by_two_threads(self):
        # thread B enters and leaves the block while thread A is inside it, with another
        # surrounding state: each thread gets back what it had
        shared = autocast("cpu", dtype=F16)
        a_inside, b_done = threading.Event(), threading.Event()
        seen = {}

        def thread_a():
            with shared:
                a_inside.set()
                b_done.wait(5)
                seen["a inside"] = torch.get_autocast_dtype() if torch.is_autocast_enabled() else None
            seen["a after"] = torch.is_autocast_enabled()

        def thread_b():
            a_inside.wait(5)
            with autocast("cpu", dtype=BF16):
                with shared:
                    seen["b inside"] = torch.get_autocast_dtype()
                seen["b after"] = torch.get_autocast_dtype() if torch.is_autocast_enabled() else None
            b_done.set()

        threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(seen, {"a inside": F16, "a after": False, "b inside": F16, "b after": BF16})

    def test_decorator(self):
        @autocast("cpu", dtype=BF16)
        def forward(v):
            """docstring"""
            return torch.is_autocast_enabled(), (v @ v.t()).dtype

        self.assertEqual(forward(self.x), (True, BF16))
        self.assertEqual((forward.__name__, forward.__doc__), ("forward", "docstring"))
        self.assertFalse(torch.is_autocast_enabled())

        @autocast("cpu")
        def recursive(n):
            return torch.is_autocast_enabled() and (n == 0 or recursive(n - 1))

        self.assertTrue(recursive(3))

    def test_unsupported_dtype_disables_with_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx = autocast("cpu", dtype=F32)
        self.assertEqual(len(caught), 1)
        self.assertIn("not supported. Disabling autocast", str(caught[0].message))
        with ctx:
            self.assertFalse(torch.is_autocast_enabled())
            self.assertEqual((self.x @ self.x.t()).dtype, F32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            autocast("cpu", dtype=F32, enabled=False)                # nothing to warn about


class TestAutocastPolicy(AutocastTestCase):
    def test_products_layers_and_convolutions_run_in_the_autocast_dtype(self):
        x, seq, img = self.x, self.seq, self.img
        heads = torch.randn(2, 2, 5, 4)
        layers = {
            "linear": nn.Linear(8, 3), "conv1d": nn.Conv1d(8, 4, 3), "conv2d": nn.Conv2d(3, 4, 3),
            "conv3d": nn.Conv3d(3, 4, 3), "deconv": nn.ConvTranspose2d(3, 4, 3), "prelu": nn.PReLU(),
            "attention": nn.MultiheadAttention(8, 2, batch_first=True), "rnn": nn.RNN(8, 6, batch_first=True),
        }  # fmt: skip
        cases = {
            "@": lambda: x @ x.t(),
            "matmul": lambda: torch.matmul(x, x.t()),
            "mm": lambda: torch.mm(x, x.t()),
            "bmm": lambda: torch.bmm(seq, seq.transpose(1, 2)),
            "mv": lambda: torch.mv(x, x[0]),
            "addmm": lambda: torch.addmm(torch.zeros(4, 4), x, x.t()),
            "baddbmm": lambda: torch.baddbmm(torch.zeros(2, 5, 5), seq, seq.transpose(1, 2)),
            "einsum": lambda: torch.einsum("ij,kj->ik", x, x),
            "einsum of a list": lambda: torch.einsum("ij,jk,kl->il", [x, x.t(), x]),
            "einsum of one operand": lambda: torch.einsum("ij->i", x),
            "linalg.multi_dot": lambda: torch.linalg.multi_dot([x, x.t(), x]),
            "linalg.vecdot": lambda: torch.linalg.vecdot(x, x),
            "linalg.matrix_power": lambda: torch.linalg.matrix_power(x @ x.t(), 2),
            "F.linear": lambda: F.linear(x, torch.randn(3, 8), torch.randn(3)),
            "nn.Linear": lambda: layers["linear"](x),
            "nn.Conv1d": lambda: layers["conv1d"](seq.transpose(1, 2)),
            "nn.Conv2d": lambda: layers["conv2d"](img),
            "nn.Conv3d": lambda: layers["conv3d"](torch.randn(1, 3, 4, 4, 4)),
            "nn.ConvTranspose2d": lambda: layers["deconv"](img),
            "nn.PReLU": lambda: layers["prelu"](x),
            "attention": lambda: F.scaled_dot_product_attention(heads, heads, heads),
            "attention with dropout": lambda: F.scaled_dot_product_attention(heads, heads, heads, dropout_p=0.1),
            "nn.MultiheadAttention": lambda: layers["attention"](seq, seq, seq),
            "nn.RNN": lambda: layers["rnn"](seq),
        }
        self.assert_dtypes(cases, inside=BF16, outside=F32)
        self.assert_dtypes({k: cases[k] for k in ("@", "nn.Linear", "nn.Conv2d", "attention")}, inside=F16, outside=F32, dtype=F16)

    def test_softmax_norms_losses_and_sums_run_in_float32(self):
        x = self.x
        low = (torch.rand(4, 8) * 0.8 + 0.1).to(BF16)         # what a layer hands on inside the block
        labels = torch.tensor([0, 1, 2, 3])
        norms = {"layer": nn.LayerNorm(8), "group": nn.GroupNorm(2, 8)}
        cases = {
            "softmax": lambda: F.softmax(low, dim=-1),
            "Tensor.softmax": lambda: low.softmax(-1),
            "torch.softmax": lambda: torch.softmax(low, dim=-1),
            "log_softmax": lambda: F.log_softmax(low, dim=-1),
            "Tensor.log_softmax": lambda: low.log_softmax(-1),
            "softmin": lambda: F.softmin(low, dim=-1),
            "nn.Softmax": lambda: nn.Softmax(dim=-1)(low),
            "softplus": lambda: F.softplus(low),
            "layer_norm": lambda: F.layer_norm(low, (8,)),
            "group_norm": lambda: F.group_norm(low, 2),
            "normalize": lambda: F.normalize(low),
            "cosine_similarity": lambda: F.cosine_similarity(low, low),
            "pairwise_distance": lambda: F.pairwise_distance(low, low),
            "cdist": lambda: torch.cdist(low, low),
            "pdist": lambda: torch.pdist(low),
            "interpolate": lambda: F.interpolate(self.img.to(BF16), scale_factor=2),
            "cross_entropy": lambda: F.cross_entropy(low, labels),
            "nn.CrossEntropyLoss": lambda: nn.CrossEntropyLoss()(low, labels),
            "nll_loss": lambda: F.nll_loss(low, labels),
            "mse_loss": lambda: F.mse_loss(low, low),
            "l1_loss": lambda: F.l1_loss(low, low),
            "smooth_l1_loss": lambda: F.smooth_l1_loss(low, low),
            "huber_loss": lambda: F.huber_loss(low, low),
            "bce_with_logits": lambda: F.binary_cross_entropy_with_logits(low, low),
            "bce": lambda: F.binary_cross_entropy(low, low),
            "nn.BCELoss": lambda: nn.BCELoss()(low, low),
            "kl_div": lambda: F.kl_div(low, low, reduction="batchmean"),
            "cosine_embedding_loss": lambda: F.cosine_embedding_loss(low, low, torch.ones(4).to(BF16)),
            "margin_ranking_loss": lambda: F.margin_ranking_loss(low[:, 0], low[:, 1], torch.ones(4).to(BF16)),
            "triplet_margin_loss": lambda: F.triplet_margin_loss(low, low, low),
            "sum": lambda: low.sum(),
            "sum(dim)": lambda: low.sum(dim=1),
            "torch.sum": lambda: torch.sum(low),
            "prod": lambda: low.prod(1),
            "norm": lambda: low.norm(),
            "linalg.norm": lambda: torch.linalg.norm(low),
            "linalg.vector_norm": lambda: torch.linalg.vector_norm(low, dim=1),
            "linalg.matrix_norm": lambda: torch.linalg.matrix_norm(low),
            "cumsum": lambda: low.cumsum(1),
            "cumprod": lambda: low.cumprod(1),
            "logsumexp": lambda: low.logsumexp(1),
            "x ** 2": lambda: low ** 2,
            "x ** x": lambda: low ** low,
            "2 ** x": lambda: 2 ** low,
            "torch.pow": lambda: torch.pow(low, 3),
        }
        for name in ("acos", "asin", "cosh", "erfinv", "exp", "expm1", "log", "log10", "log2", "log1p", "reciprocal", "rsqrt", "sinh", "tan"):
            cases[name] = lambda name=name: getattr(low, name)()
            cases["torch." + name] = lambda name=name: getattr(torch, name)(low)
        self.assert_dtypes(cases, inside=F32, outside=BF16)
        with autocast("cpu", dtype=BF16):                          # layers that hold float32 weights
            self.assertEqual(norms["layer"](low).dtype, F32)
            self.assertEqual(norms["group"](low).dtype, F32)
        # float32 means computed in float32, not rounded to it afterwards
        with autocast("cpu", dtype=BF16):
            got = low.exp()
        self.assertEqual(got.tolist(), low.float().exp().tolist())
        self.assertNotEqual(got.tolist(), low.exp().float().tolist())

    def test_dot_tensordot_and_bilinear_use_the_widest_operand_type(self):
        x = self.x
        low = x.to(BF16)
        weight = torch.randn(2, 8, 8)
        with autocast("cpu", dtype=BF16):
            self.assertEqual(torch.dot(x[0], x[1]).dtype, F32)
            self.assertEqual(torch.dot(low[0], low[1]).dtype, BF16)
            self.assertEqual(torch.dot(low[0], x[1]).dtype, F32)
            self.assertEqual(torch.tensordot(x, x.t(), dims=1).dtype, F32)
            self.assertEqual(torch.tensordot(low, low.t(), dims=1).dtype, BF16)
            self.assertEqual(torch.tensordot(low, x.t(), dims=1).dtype, F32)
            self.assertEqual(F.bilinear(x, x, weight).dtype, F32)
            self.assertEqual(F.bilinear(low, low, weight.to(BF16)).dtype, BF16)
            self.assertEqual(F.bilinear(low, low, weight).dtype, F32)
            self.assertEqual(nn.Bilinear(8, 8, 2)(x, x).dtype, F32)
            # float32 operands: the very numbers computed without autocast
            self.assertEqual(torch.tensordot(x, x.t(), dims=1).tolist(), self._without_autocast(lambda: torch.tensordot(x, x.t(), dims=1)))
            self.assertEqual(F.bilinear(x, x, weight).tolist(), self._without_autocast(lambda: F.bilinear(x, x, weight)))
            # the operations that promote on their own
            self.assertEqual(torch.cat([low, x]).dtype, F32)
            self.assertEqual(torch.stack([low, x]).dtype, F32)
            self.assertEqual(torch.atan2(low, x).dtype, F32)
            self.assertEqual((low + x).dtype, F32)

    @staticmethod
    def _without_autocast(fn):
        with autocast("cpu", enabled=False):
            return fn().tolist()

    def test_other_operations_keep_the_dtype_of_their_inputs(self):
        low = (torch.rand(4, 8) * 0.8 + 0.1).to(BF16)
        img = self.img.to(BF16)
        square = torch.eye(4) + 0.1 * torch.rand(4, 4)
        spd = square @ square.t()
        cases = {
            "relu": lambda: F.relu(low), "gelu": lambda: F.gelu(low), "silu": lambda: F.silu(low), "elu": lambda: F.elu(low),
            "leaky_relu": lambda: F.leaky_relu(low), "mish": lambda: F.mish(low), "hardswish": lambda: F.hardswish(low),
            "sigmoid": lambda: low.sigmoid(), "logsigmoid": lambda: F.logsigmoid(low), "tanh": lambda: low.tanh(), "sqrt": lambda: low.sqrt(),
            "square": lambda: low.square(), "sin": lambda: low.sin(), "erf": lambda: low.erf(), "abs": lambda: low.abs(),
            "scale": lambda: low * 2, "add": lambda: low + low, "divide": lambda: 1 / low, "clamp": lambda: low.clamp(0.2, 0.7),
            "mean": lambda: low.mean(), "mean(dim)": lambda: low.mean(1), "var": lambda: low.var(), "std": lambda: low.std(1),
            "max": lambda: low.max(), "amax": lambda: low.amax(1), "trace": lambda: low[:, :4].trace(),
            "reshape": lambda: low.reshape(8, 4), "index": lambda: low[torch.tensor([0, 2])], "where": lambda: torch.where(low > 0.5, low, low * 2),
            "batch_norm": lambda: F.batch_norm(low, None, None, training=True), "instance_norm": lambda: F.instance_norm(img),
            "rms_norm": lambda: F.rms_norm(low, (8,)), "dropout": lambda: F.dropout(low, 0.5),
            "max_pool2d": lambda: F.max_pool2d(img, 2), "avg_pool2d": lambda: F.avg_pool2d(img, 2), "adaptive_avg_pool2d": lambda: F.adaptive_avg_pool2d(img, 2),
            "pad": lambda: F.pad(low, (1, 1)), "embedding": lambda: F.embedding(torch.tensor([1, 2]), low),
            "embedding_bag": lambda: F.embedding_bag(torch.tensor([[0, 1], [2, 3]]), low),
            "special.logit": lambda: torch.special.logit(low), "special.entr": lambda: torch.special.entr(low),
            "special.xlogy": lambda: torch.special.xlogy(low, low), "special.i0": lambda: torch.special.i0(low),
        }  # fmt: skip
        self.assert_dtypes(cases, inside=BF16, outside=BF16)
        # decompositions and solvers: float32 stays float32, with every digit - no product inside is cast
        solvers = {
            "linalg.inv": lambda: torch.linalg.inv(square), "linalg.solve": lambda: torch.linalg.solve(square, square),
            "linalg.pinv": lambda: torch.linalg.pinv(square), "linalg.lstsq": lambda: torch.linalg.lstsq(square, square)[0],
            "linalg.det": lambda: torch.linalg.det(square), "linalg.qr": lambda: torch.linalg.qr(square)[0],
            "linalg.svd": lambda: torch.linalg.svd(square)[1], "linalg.eigh": lambda: torch.linalg.eigh(spd)[0],
            "linalg.cholesky": lambda: torch.linalg.cholesky(spd),
            "linalg.cond": lambda: torch.linalg.cond(square), "linalg.matrix_rank": lambda: torch.linalg.matrix_rank(square),
        }  # fmt: skip
        for name, fn in solvers.items():
            with self.subTest(name):
                want = fn()
                with autocast("cpu", dtype=BF16):
                    got = fn()
                self.assertEqual(got.dtype, want.dtype)
                self.assertEqual(got.tolist(), want.tolist())
        with autocast("cpu", dtype=BF16):
            # tensors created inside the block are ordinary tensors
            self.assertEqual(torch.zeros(2).dtype, F32)
            self.assertEqual(nn.Embedding(10, 8)(torch.tensor([1, 2])).dtype, F32)

    def test_float64_and_integers_are_not_autocast(self):
        with autocast("cpu", dtype=BF16):
            double = self.x.double()
            self.assertEqual((double @ double.t()).dtype, F64)
            self.assertEqual(nn.Linear(8, 3).double()(double).dtype, F64)
            self.assertEqual(double.softmax(-1).dtype, F64)
            self.assertEqual(double.sum().dtype, F64)
            self.assertEqual(double.exp().dtype, F64)
            self.assertEqual(F.mse_loss(double, double).dtype, F64)
            self.assertEqual(torch.einsum("ij,kj->ik", double, double).dtype, F64)
            ints = torch.ones(2, 2, dtype=torch.int64)
            self.assertEqual((ints @ ints).dtype, torch.int64)
            self.assertEqual(ints.sum().dtype, torch.int64)
            self.assertEqual(ints.cumsum(0).dtype, torch.int64)
            self.assertEqual((ints ** 2).dtype, torch.int64)

    def test_explicit_dtype_and_out_are_not_autocast(self):
        x = self.x
        low = x.to(BF16)
        with autocast("cpu", dtype=BF16):
            self.assertEqual(low.softmax(-1, dtype=F64).dtype, F64)
            self.assertEqual(low.sum(dtype=BF16).dtype, BF16)
            self.assertEqual(low.prod(dtype=F64).dtype, F64)
            self.assertEqual(torch.linalg.vector_norm(low, dtype=F64).dtype, F64)
            # out=: the product is computed in the types of the operands and stored - not rounded
            # through the autocast dtype on the way
            out = torch.empty(4, 4)
            self.assertIs(torch.matmul(x, x.t(), out=out), out)
            self.assertEqual(out.dtype, F32)
            for fn in (torch.matmul, torch.mm):
                fn(x, x.t(), out=out)
                with autocast("cpu", enabled=False):
                    self.assertEqual(out.tolist(), (x @ x.t()).tolist())
            torch.addmm(torch.zeros(4, 4), x, x.t(), out=out)
            with autocast("cpu", enabled=False):
                self.assertEqual(out.tolist(), torch.addmm(torch.zeros(4, 4), x, x.t()).tolist())
            # in-place operations change values, never the dtype
            buf = low.clone()
            buf.exp_()
            self.assertEqual(buf.dtype, BF16)

    def test_values_are_those_of_the_lower_precision(self):
        x, w, b = self.x, torch.randn(3, 8), torch.randn(3)
        with autocast("cpu", dtype=BF16):
            got = F.linear(x, w, b)
        want = F.linear(x.to(BF16), w.to(BF16), b.to(BF16))
        self.assertEqual(got.float().tolist(), want.float().tolist())
        exact = F.linear(x, w, b)
        self.assertNotEqual(got.float().tolist(), exact.tolist())
        self.assertLess((got.float() - exact).abs().max().item(), 0.1)


class TestAutocastTraining(AutocastTestCase):
    def test_gradients_reach_float32_parameters(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        labels = torch.tensor([0, 1, 2, 3])
        with autocast("cpu", dtype=BF16):
            logits = model(self.x)
            loss = F.cross_entropy(logits, labels)
        self.assertEqual((logits.dtype, loss.dtype), (BF16, F32))
        loss.backward()
        low_grads = [p.grad for p in model.parameters()]
        for p, g in zip(model.parameters(), low_grads):
            self.assertEqual((p.dtype, g.dtype, tuple(g.shape)), (F32, F32, tuple(p.shape)))
            self.assertTrue(bool(g.isfinite().all().item()))
        # close to the full-precision gradients
        for p in model.parameters():
            p.grad = None
        F.cross_entropy(model(self.x), labels).backward()
        for p, low in zip(model.parameters(), low_grads):
            self.assertLess((p.grad - low).abs().max().item(), 0.05)
            self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_backward_inside_the_block_is_not_autocast(self):
        # addmm and outer compute their gradients with matrix products; called inside the block,
        # those must not be cast: the gradients are the ones backward() gives outside it
        def run(backward_inside):
            torch.manual_seed(3)
            a = torch.randn(4, 5, requires_grad=True)
            b = torch.randn(5, 3, requires_grad=True)
            bias = torch.randn(3, requires_grad=True)
            v = torch.randn(3, requires_grad=True)
            with autocast("cpu", dtype=BF16):
                loss = (torch.addmm(bias, a, b).float() * torch.outer(v, v).sum()).sum()
                if backward_inside:
                    loss.backward()
            if not backward_inside:
                loss.backward()
            return [t.grad for t in (a, b, bias, v)]

        inside, outside = run(True), run(False)
        for got, want in zip(inside, outside):
            self.assertEqual(got.dtype, F32)
            self.assertEqual(got.tolist(), want.tolist())
            self.assertGreater(got.abs().max().item(), 0.0)
        # torch.autograd.grad as well, with a graph for a second derivative
        x = torch.randn(3, 3, requires_grad=True)
        with autocast("cpu", dtype=BF16):
            y = (x @ x).float().pow(2).sum()
            (g,) = torch.autograd.grad(y, x, create_graph=True)
            self.assertTrue(torch.is_autocast_enabled())
        self.assertEqual(g.dtype, F32)
        self.assertTrue(g.requires_grad)

    def test_training_with_grad_scaler_reduces_the_loss(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
        opt = optim.SGD(model.parameters(), lr=0.05)
        scaler = GradScaler("cpu", init_scale=8.0)
        x = torch.randn(32, 8)
        y = x[:, :1] * 2.0 - x[:, 1:2]
        losses = []
        for _ in range(40):
            opt.zero_grad()
            with autocast("cpu", dtype=BF16):
                pred = model(x)
                loss = F.mse_loss(pred, y)
            self.assertEqual((pred.dtype, loss.dtype), (BF16, F32))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5)
        self.assertTrue(all(p.dtype == F32 for p in model.parameters()))

    def test_no_grad_and_autocast_together(self):
        layer = nn.Linear(8, 3)
        with torch.no_grad(), autocast("cpu", dtype=BF16):
            out = layer(self.x)
        self.assertEqual(out.dtype, BF16)
        self.assertFalse(out.requires_grad)


class TestGradScaler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layer = nn.Linear(3, 1)
        self.opt = optim.SGD(self.layer.parameters(), lr=1.0)
        self.scaler = GradScaler("cpu", init_scale=1024.0)
        self.before = self.layer.weight.detach().clone()

    def scaled_backward(self):
        self.opt.zero_grad()
        self.scaler.scale(self.layer(torch.ones(2, 3)).sum()).backward()

    def weight_change(self):
        return (self.before - self.layer.weight.detach()).tolist()

    def test_step_unscales_the_gradients(self):
        self.scaled_backward()
        self.assertEqual(self.layer.weight.grad.tolist(), [[2048.0, 2048.0, 2048.0]])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.assertEqual(self.weight_change(), [[2.0, 2.0, 2.0]])

    def test_unscale_then_step_unscales_once(self):
        # the gradient-clipping recipe: unscale_, clip, step, update
        self.scaled_backward()
        self.scaler.unscale_(self.opt)
        self.assertEqual(self.layer.weight.grad.tolist(), [[2.0, 2.0, 2.0]])
        nn.utils.clip_grad_norm_(self.layer.parameters(), max_norm=1e9)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.assertEqual(self.weight_change(), [[2.0, 2.0, 2.0]])
        # and again in the next iteration
        self.scaled_backward()
        self.scaler.unscale_(self.opt)
        self.scaler.step(self.opt)
        self.scaler.update()
        for change in self.weight_change()[0]:
            self.assertAlmostEqual(change, 4.0, places=5)

    def test_misuse_is_an_error(self):
        self.scaled_backward()
        self.scaler.unscale_(self.opt)
        with self.assertRaisesRegex(RuntimeError, "already been called"):
            self.scaler.unscale_(self.opt)
        self.scaler.step(self.opt)
        with self.assertRaisesRegex(RuntimeError, "step\\(\\) has already been called"):
            self.scaler.step(self.opt)
        with self.assertRaisesRegex(RuntimeError, "after step"):
            self.scaler.unscale_(self.opt)
        self.scaler.update()
        self.scaled_backward()
        with self.assertRaisesRegex(RuntimeError, "Closure"):
            self.scaler.step(self.opt, closure=lambda: None)

    def test_non_finite_gradients_skip_the_step_and_lower_the_scale(self):
        self.scaled_backward()
        self.layer.weight.grad[0, 0] = float("inf")
        self.scaler.unscale_(self.opt)
        self.assertIsNone(self.scaler.step(self.opt))
        self.assertEqual(self.weight_change(), [[0.0, 0.0, 0.0]])
        self.scaler.update()
        self.assertEqual(self.scaler.get_scale(), 512.0)
        self.scaled_backward()                                   # finite again: the step is taken
        self.scaler.step(self.opt)
        self.scaler.update()
        self.assertEqual(self.weight_change(), [[2.0, 2.0, 2.0]])
        self.assertEqual(self.scaler.get_scale(), 512.0)

    def test_two_optimizers_share_the_inf_check(self):
        other_layer = nn.Linear(3, 1)
        other = optim.SGD(other_layer.parameters(), lr=1.0)
        self.opt.zero_grad()
        other.zero_grad()
        self.scaler.scale(self.layer(torch.ones(2, 3)).sum() + other_layer(torch.ones(2, 3)).sum()).backward()
        self.layer.weight.grad[0, 0] = float("nan")
        self.scaler.step(self.opt)                                # skipped
        self.scaler.step(other)                                   # unscaling this one must not forget the nan
        self.scaler.update()
        self.assertEqual(self.scaler.get_scale(), 512.0)

    def test_disabled_scaler_is_transparent(self):
        scaler = GradScaler("cpu", enabled=False)
        loss = self.layer(torch.ones(2, 3)).sum()
        self.assertIs(scaler.scale(loss), loss)
        loss.backward()
        scaler.unscale_(self.opt)
        scaler.unscale_(self.opt)
        scaler.step(self.opt)
        scaler.update()
        self.assertEqual(self.weight_change(), [[2.0, 2.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
