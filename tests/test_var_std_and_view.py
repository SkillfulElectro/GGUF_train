"""Tensor.var / Tensor.std with a numeric `correction`, and Tensor.view, which never copies."""

import ctypes
import math
import random
import unittest

import torch
from torch import _C
from gguf_train._functions import _can_view_reshape

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

DATA = [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]
# sum of squared deviations from the mean: all six values, and row by row
SS_ALL = 703.5
SS_ROWS = [14.0 / 3.0, 896.0 / 3.0]


class TestVarianceCorrection(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(DATA, dtype=torch.float64)

    def assert_close(self, got, want, places=10):
        got = got.tolist() if isinstance(got, torch.Tensor) else got
        if isinstance(want, list):
            self.assertEqual(len(got), len(want))
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=places)
        else:
            self.assertAlmostEqual(got, want, places=places)

    def test_correction_is_the_number_subtracted_from_n(self):
        x = self.x
        for correction in (0, 1, 2, 3, 5, 1.5, 0.25, -1, 2.0):
            with self.subTest(correction=correction):
                self.assert_close(x.var(correction=correction), SS_ALL / (6 - correction))
                self.assert_close(x.std(correction=correction), math.sqrt(SS_ALL / (6 - correction)))
                self.assert_close(torch.var(x, correction=correction), SS_ALL / (6 - correction))
                self.assert_close(torch.std(x, correction=correction), math.sqrt(SS_ALL / (6 - correction)))

    def test_correction_two_is_not_the_sample_variance(self):
        # the original turned every correction other than 0 into N - 1
        self.assert_close(self.x.var(correction=2), 175.875)
        self.assert_close(self.x.var(), 140.7)
        self.assertNotAlmostEqual(self.x.var(correction=2).item(), self.x.var().item(), places=3)

    def test_along_dimensions(self):
        x = self.x
        self.assert_close(x.var(1, correction=2), [s / 1 for s in SS_ROWS])
        self.assert_close(x.var(dim=1, correction=0.5), [s / 2.5 for s in SS_ROWS])
        self.assert_close(x.std(-1, correction=2), [math.sqrt(s) for s in SS_ROWS])
        self.assert_close(x.var((0, 1), correction=2), SS_ALL / 4)
        self.assert_close(x.var((), correction=2), SS_ALL / 4)
        self.assertEqual(tuple(x.var(1, keepdim=True, correction=2).shape), (2, 1))
        self.assertEqual(tuple(x.std(0, keepdim=True, correction=0.5).shape), (1, 3))
        self.assert_close(x.var(0, correction=0.5), [(a - b) ** 2 / 2 / 1.5 for a, b in zip(*DATA)])
        with self.assertRaises(IndexError):
            x.var(2, correction=2)

    def test_defaults_and_the_unbiased_spelling(self):
        x = self.x
        self.assert_close(x.var(), SS_ALL / 5)
        self.assert_close(x.var(unbiased=False), SS_ALL / 6)
        self.assert_close(x.var(1, False), [s / 3 for s in SS_ROWS])
        self.assert_close(x.var(1, True), [s / 2 for s in SS_ROWS])
        self.assert_close(x.std(1, unbiased=False), [math.sqrt(s / 3) for s in SS_ROWS])
        # var(unbiased) / std(unbiased): a bool in first place is not dimension 0 or 1
        self.assert_close(x.var(False), SS_ALL / 6)
        self.assert_close(x.var(True), SS_ALL / 5)
        self.assert_close(x.std(False), math.sqrt(SS_ALL / 6))
        self.assertEqual(x.var(False).shape, ())
        # correction wins over the default of `unbiased`
        self.assert_close(x.var(1, correction=0), [s / 3 for s in SS_ROWS])

    def test_no_degrees_of_freedom(self):
        x = self.x
        self.assertEqual(x.var(correction=6).item(), math.inf)             # sum / 0
        self.assertEqual(x.var(correction=7).item(), math.inf)             # max(0, N - correction)
        self.assertEqual(x.std(correction=6).item(), math.inf)
        self.assertEqual(x.var(1, correction=3).tolist(), [math.inf, math.inf])
        self.assertTrue(math.isnan(torch.ones(3).var(correction=3).item()))    # 0 / 0
        self.assertTrue(math.isnan(torch.tensor([3.0]).var().item()))
        self.assertTrue(math.isnan(torch.tensor([3.0]).std().item()))
        self.assertEqual(torch.tensor([3.0]).var(correction=0).item(), 0.0)
        # no elements at all
        for correction in (0, 1, 2):
            self.assertTrue(math.isnan(torch.zeros(0).var(correction=correction).item()))
            self.assertTrue(math.isnan(torch.zeros(0).std(correction=correction).item()))
        self.assertTrue(all(math.isnan(v) for v in torch.zeros(0, 3).var(0).tolist()))
        self.assertEqual(tuple(torch.zeros(0, 3).var(1).shape), (0,))

    def test_dtypes(self):
        for dtype, places in ((torch.float32, 4), (torch.float64, 10)):
            x = self.x.to(dtype)
            got = x.var(correction=2)
            self.assertEqual(got.dtype, dtype)
            self.assert_close(got, 175.875, places=places)
            self.assertEqual(x.std(1, correction=2).dtype, dtype)
        self.assertEqual(self.x.to(torch.bfloat16).var(correction=2).dtype, torch.bfloat16)
        self.assert_close(self.x.float().var(correction=2, dtype=torch.float64), 175.875)

    def test_bad_correction(self):
        with self.assertRaisesRegex(TypeError, "correction"):
            self.x.var(correction="2")
        with self.assertRaisesRegex(TypeError, "correction"):
            self.x.std(correction=[1])
        with self.assertRaisesRegex(TypeError, "both"):
            self.x.var(correction=1, ddof=1)

    def test_companions(self):
        x = self.x
        var, mean = torch.var_mean(x, dim=1, correction=2)
        self.assert_close(var, SS_ROWS)
        self.assert_close(mean, [7.0 / 3.0, 56.0 / 3.0])
        std, mean = torch.std_mean(x, correction=0)
        self.assert_close(std, math.sqrt(SS_ALL / 6))
        self.assert_close(torch.var_mean(x)[0], SS_ALL / 5)

    def test_gradients(self):
        for correction in (0, 1, 2, 2.5):
            with self.subTest(correction=correction):
                x = self.x.clone().requires_grad_(True)
                x.var(correction=correction).backward()
                want = (2 * (self.x - self.x.mean()) / (6 - correction)).tolist()
                for got_row, want_row in zip(x.grad.tolist(), want):
                    self.assert_close(got_row, want_row)
                x = self.x.clone().requires_grad_(True)
                x.std(1, correction=correction).sum().backward()
                dev = self.x - self.x.mean(1, keepdim=True)
                std = (dev.pow(2).sum(1, keepdim=True) / (3 - correction)).sqrt()
                for got_row, want_row in zip(x.grad.tolist(), (dev / ((3 - correction) * std)).tolist()):
                    self.assert_close(got_row, want_row)

    def test_numpy_spellings(self):
        x = self.x
        self.assert_close(x.var(axis=1, ddof=2), SS_ROWS)
        self.assert_close(x.std(axis=0, keepdims=True, ddof=0)[0], [abs(a - b) / 2 for a, b in zip(*DATA)])
        if np is None:
            self.skipTest("NumPy is not installed")
        # np.var / np.std hand the tensor ddof=0: NumPy's default is the population statistic
        self.assertAlmostEqual(float(np.var(x)), SS_ALL / 6, places=10)
        self.assertAlmostEqual(float(np.std(x, ddof=2)), math.sqrt(SS_ALL / 4), places=10)
        self.assert_close(np.var(x, axis=1, ddof=1), [s / 2 for s in SS_ROWS])
        self.assert_close(x.var(correction=np.int64(2)), SS_ALL / 4)
        self.assert_close(x.var(correction=np.float32(1.5)), SS_ALL / 4.5)


def compute_stride(oldshape, oldstride, newshape):
    """PyTorch's at::detail::computeStride, line for line: the strides of `newshape` as a view,
    or None when the elements cannot be reached without copying."""
    if not oldshape:
        return tuple([1] * len(newshape))
    numel = math.prod(oldshape)
    if numel == 0 and tuple(oldshape) == tuple(newshape):
        return tuple(oldstride)
    newstride = [0] * len(newshape)
    if numel == 0:
        for view_d in range(len(newshape) - 1, -1, -1):
            newstride[view_d] = 1 if view_d == len(newshape) - 1 else max(newshape[view_d + 1], 1) * newstride[view_d + 1]
        return tuple(newstride)
    view_d = len(newshape) - 1
    chunk_base_stride = oldstride[-1]
    tensor_numel = 1
    view_numel = 1
    for tensor_d in range(len(oldshape) - 1, -1, -1):
        tensor_numel *= oldshape[tensor_d]
        if tensor_d == 0 or (oldshape[tensor_d - 1] != 1 and oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride):
            while view_d >= 0 and (view_numel < tensor_numel or newshape[view_d] == 1):
                newstride[view_d] = view_numel * chunk_base_stride
                view_numel *= newshape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = oldstride[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return tuple(newstride)


VIEW_ERROR = "view size is not compatible with input tensor's size and stride"


class TestView(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(24.0).reshape(2, 3, 4)

    def assert_is_view(self, view, base):
        """Writing through `view` changes `base`."""
        before = base.clone()
        view.reshape(-1)[0] = -99.0 if view.is_contiguous() else None
        idx = (0,) * view.ndim
        view[idx] = -99.0
        self.assertNotEqual(base.tolist(), before.tolist())
        base.copy_(before)

    def test_contiguous_tensors_view_in_any_shape(self):
        x = self.x
        for shape in ((24,), (4, 6), (6, 4), (2, 12), (2, 3, 4), (1, 2, 3, 4, 1), (2, 3, 2, 2), (-1,), (3, -1), (-1, 2, 2)):
            with self.subTest(shape=shape):
                v = x.view(*shape)
                self.assertEqual(v.reshape(-1).tolist(), x.reshape(-1).tolist())
                self.assert_is_view(v, x)
        self.assertEqual(tuple(x.view((4, 6)).shape), (4, 6))
        self.assertEqual(tuple(x.view([6, 4]).shape), (6, 4))
        self.assertEqual(tuple(x.view_as(torch.zeros(8, 3)).shape), (8, 3))
        self.assertEqual(x[1].view(12).tolist(), list(map(float, range(12, 24))))        # an offset into the storage
        self.assertEqual(torch.tensor(3.0).view(1, 1).tolist(), [[3.0]])
        self.assertEqual(torch.ones(1).view(()).shape, ())

    def test_view_never_copies(self):
        # the original went through reshape and handed back a copy
        t = self.x.transpose(0, 2)                    # (4, 3, 2), not contiguous
        self.assertFalse(t.is_contiguous())
        for shape in ((24,), (12, 2), (4, 6), (2, 12), (-1,), (8, 3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, VIEW_ERROR):
                    t.view(*shape)
        with self.assertRaisesRegex(RuntimeError, "Use .reshape"):
            t.view(24)
        with self.assertRaisesRegex(RuntimeError, VIEW_ERROR):
            t.view_as(torch.zeros(24))
        with self.assertRaisesRegex(RuntimeError, VIEW_ERROR):
            self.x[:, ::2].view(16)                   # a step in an inner dimension
        self.assertEqual(self.x[:, :, ::2].view(12).stride(), (2,))     # an even step all the way: still a view
        with self.assertRaisesRegex(RuntimeError, VIEW_ERROR):
            self.x[:, 1:].view(16)                    # a slice of an inner dimension
        with self.assertRaisesRegex(RuntimeError, VIEW_ERROR):
            torch.ones(1, 3).expand(2, 3).view(6)     # expanded: several elements, one location
        # reshape is the one that copies, and the copy is right
        self.assertEqual(t.reshape(24).tolist(), t.contiguous().view(24).tolist())
        self.assertEqual(t.contiguous().view(24)[:4].tolist(), [0.0, 12.0, 4.0, 16.0])

    def test_views_the_strides_do_allow(self):
        x = self.x
        t = x.transpose(0, 2)                          # (4, 3, 2)
        cases = (
            (t, (4, 3, 2)), (t, (4, 1, 3, 2)), (t, (4, 3, 2, 1)), (t, (1, 4, 3, 1, 2, 1)),      # same dims, ones added
            (x[:, :, ::2], (2, 3, 2)), (x[:, :, ::2], (6, 2)), (x[:, :, ::2], (2, 3, 1, 2)),   # leading dims merge
            (x[:, 1:], (2, 2, 4)), (x[:, 1:], (2, 8)), (x[:, 1:], (2, 2, 2, 2)),                # trailing dims merge / split
            (x.permute(1, 0, 2), (3, 2, 4)), (x.permute(1, 0, 2), (3, 2, 2, 2)), (x[0], (12,)), (x[:, 0], (2, 2, 2)),
            (torch.ones(1, 3).expand(2, 3), (2, 1, 3)),
        )  # fmt: skip
        for base, shape in cases:
            with self.subTest(base=tuple(base.shape), shape=shape):
                v = base.view(*shape)
                self.assertEqual(tuple(v.shape), shape)
                self.assertEqual(v.reshape(-1).tolist(), base.reshape(-1).tolist())
        v = t.view(4, 1, 3, 2)
        v[0, 0, 0, 0] = -1.0
        self.assertEqual(x[0, 0, 0].item(), -1.0)

    def test_agrees_with_pytorch_on_which_views_exist(self):
        rng = random.Random(0)
        checked = viewable = 0
        for _ in range(250):
            nd = rng.randint(1, 4)
            base = [rng.choice([1, 2, 3, 4]) for _ in range(nd)]
            if rng.random() < 0.1:
                base[rng.randrange(nd)] = 0
            x = torch.arange(float(math.prod(base))).reshape(base)
            perm = list(range(nd))
            if rng.random() < 0.5:
                rng.shuffle(perm)
            x = x.permute(*perm)
            for d in range(nd):
                if x.shape[d] > 1 and rng.random() < 0.25:
                    x = x[(slice(None),) * d + (slice(None, None, 2),)]
                elif x.shape[d] == 1 and rng.random() < 0.2:
                    x = x.expand(*[3 if i == d else -1 for i in range(nd)])
            numel = x.numel()
            targets = [tuple(x.shape), (numel,), (1, numel), (numel, 1)]
            for a in range(1, numel + 1):
                if numel % a == 0:
                    targets.append((a, numel // a))
                    for b in range(1, numel // a + 1):
                        if (numel // a) % b == 0:
                            targets.append((a, b, numel // a // b))
            for shape in targets:
                want = compute_stride(tuple(x.shape), tuple(x.stride()), shape)
                checked += 1
                if want is None:
                    self.assertIsNone(_can_view_reshape(tuple(x.shape), tuple(x.stride()), shape))
                    with self.assertRaisesRegex(RuntimeError, VIEW_ERROR):
                        x.view(*shape)
                    self.assertEqual(x.reshape(*shape).reshape(-1).tolist(), x.contiguous().reshape(-1).tolist())
                else:
                    viewable += 1
                    v = x.view(*shape)
                    self.assertEqual(tuple(v.shape), shape)
                    if numel:
                        self.assertEqual([s for s, n in zip(v.stride(), shape) if n > 1], [s for s, n in zip(want, shape) if n > 1])
                        self.assertEqual(v.reshape(-1).tolist(), x.contiguous().reshape(-1).tolist())
        self.assertGreater(viewable, 1000)
        self.assertGreater(checked - viewable, 500)

    def test_the_number_of_elements_must_match(self):
        # view(7) of six elements used to hand out a tensor reaching past the end of the memory,
        # view(5) one that silently lost an element; reshape did the same
        x = torch.arange(6.0).reshape(2, 3)
        for method in (x.view, x.reshape, x.t().reshape):
            for shape in ((5,), (7,), (2, 4), (3, 3), (6, 2), (0,), (1,), (2, 3, 2)):
                with self.subTest(method=method.__name__, shape=shape):
                    with self.assertRaisesRegex(RuntimeError, r"shape '\[.*\]' is invalid for input of size 6"):
                        method(*shape)
            with self.assertRaisesRegex(RuntimeError, "invalid for input of size 6"):
                method(-1, 4)
            with self.assertRaisesRegex(RuntimeError, "only one dimension can be inferred"):
                method(-1, -1)
            with self.assertRaises(RuntimeError):
                method(-2, 3)
        with self.assertRaisesRegex(RuntimeError, "invalid for input of size 6"):
            x.view_as(torch.zeros(7))
        with self.assertRaisesRegex(RuntimeError, "invalid for input of size 6"):
            x.reshape_as(torch.zeros(4))
        with self.assertRaisesRegex(RuntimeError, "invalid for input of size 6"):
            torch.reshape(x, (4, 2))

    def test_tensors_without_elements(self):
        empty = torch.zeros(0, 3)
        for shape in ((0,), (3, 0), (0, 3), (2, 0, 5), (-1,), (-1, 3)):
            with self.subTest(shape=shape):
                self.assertEqual(empty.view(*shape).numel(), 0)
                self.assertEqual(empty.t().view(*shape).numel(), 0)     # any strides will do
        self.assertEqual(tuple(empty.view(-1, 3).shape), (0, 3))
        for shape in ((-1, 0), (0, -1)):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                empty.view(*shape)
        with self.assertRaisesRegex(RuntimeError, "invalid for input of size 0"):
            empty.view(1)
        with self.assertRaisesRegex(RuntimeError, "invalid for input of size 6"):
            torch.zeros(6).view(-1, 0)

    def test_reshape_views_when_it_can_and_copies_when_it_must(self):
        x = self.x
        r = x.reshape(4, 6)
        r[0, 0] = -1.0
        self.assertEqual(x[0, 0, 0].item(), -1.0)                 # a view
        x[0, 0, 0] = 0.0
        t = x.transpose(0, 2)
        c = t.reshape(24)
        c[0] = -1.0
        self.assertEqual(x[0, 0, 0].item(), 0.0)                  # a copy
        split = t.reshape(4, 3, 1, 2)                             # reachable without copying
        split[0, 0, 0, 0] = -2.0
        self.assertEqual(x[0, 0, 0].item(), -2.0)

    def test_gradient_through_view(self):
        x = torch.arange(6.0, requires_grad=True)
        (x.view(2, 3) * torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum().backward()
        self.assertEqual(x.grad.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = torch.arange(6.0, requires_grad=True)
        (y.reshape(2, 3).t().reshape(6) * torch.arange(6.0)).sum().backward()
        self.assertEqual(y.grad.tolist(), [0.0, 2.0, 4.0, 1.0, 3.0, 5.0])


class TestNativeViewBounds(unittest.TestCase):
    """gt_tensor_view itself refuses a view that leaves the storage, whoever asks for it."""

    def native_view(self, t, shape, strides, offset):
        arr = ctypes.c_int64 * len(shape)
        return _C.lib.gt_tensor_view(t._ptr, len(shape), arr(*shape), arr(*strides), offset)

    def test_views_outside_the_storage_are_refused(self):
        t = torch.arange(6.0)
        for shape, strides, offset in (
            ((7,), (1,), 0),            # one element too many
            ((6,), (1,), 1),            # shifted past the end
            ((3,), (3,), 0),            # stride reaches element 6
            ((2, 4), (4, 1), 0),
            ((6,), (1,), -1),           # before the start
            ((3,), (-1,), 1),           # a negative stride walking out of the front
            ((-1,), (1,), 0),
        ):
            with self.subTest(shape=shape, strides=strides, offset=offset):
                ptr = self.native_view(t, shape, strides, offset)
                self.assertFalse(ptr)
                message = _C.last_error()
                self.assertTrue("outside the storage" in message or "negative dimension" in message, message)

    def test_views_inside_the_storage_are_made(self):
        t = torch.arange(6.0)
        for shape, strides, offset, want in (
            ((6,), (1,), 0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            ((3,), (2,), 1, [1.0, 3.0, 5.0]),
            ((2, 3), (3, 1), 0, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            ((4, 2), (0, 5), 0, [[0.0, 5.0]] * 4),           # expanded: stride 0
            ((3,), (-1,), 2, [2.0, 1.0, 0.0]),
            ((0,), (1,), 6, []),                               # no elements: nothing to check
        ):
            with self.subTest(shape=shape, strides=strides, offset=offset):
                ptr = self.native_view(t, shape, strides, offset)
                self.assertTrue(ptr)
                view = torch.Tensor(_ptr=ptr)
                if want and min(strides) >= 0:
                    self.assertEqual(view.tolist(), want)
                self.assertEqual(tuple(view.shape), shape)


if __name__ == "__main__":
    unittest.main()
