"""NumPy interoperability: shared memory (from_numpy, Tensor.numpy), the array protocols
(__array__, __array_ufunc__) and NumPy scalars and arrays wherever Python numbers and tensors
are accepted.

NumPy is optional for the package, so everything that needs it is skipped without it. The
native side of from_numpy - a storage on memory it does not own - is tested through ctypes
buffers and runs everywhere.
"""

import ctypes
import gc
import unittest
import warnings
import weakref

import torch
import torch.nn as nn
from gguf_train import _C
from gguf_train import _tensor as tensor_module

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on machines without numpy
    np = None

needs_numpy = unittest.skipIf(np is None, "numpy is not installed")


def drain_released():
    """Tokens of external storages destroyed since the last call."""
    out = []
    buf = (ctypes.c_uint64 * 8)()
    while True:
        n = _C.lib.gt_external_released(buf, 8)
        out.extend(int(buf[i]) for i in range(n))
        if n < 8:
            return out


def external(memory, shape, strides, dtype, token, offset=0):
    """A raw tensor handle on a ctypes buffer (strides in elements)."""
    return _C.lib.gt_tensor_from_external(
        len(shape),
        (ctypes.c_int64 * max(1, len(shape)))(*shape),
        (ctypes.c_int64 * max(1, len(strides)))(*strides),
        dtype.code,
        ctypes.c_void_p(ctypes.addressof(memory) + offset),
        token,
    )


class TestExternalStorage(unittest.TestCase):
    """gt_tensor_from_external: tensors on memory that belongs to the caller."""

    def setUp(self):
        gc.collect()
        drain_released()

    def test_reads_and_writes_go_to_the_callers_memory(self):
        memory = (ctypes.c_float * 6)(0, 1, 2, 3, 4, 5)
        t = torch.Tensor(_ptr=_C.check_ptr(external(memory, (2, 3), (3, 1), torch.float32, 101)))
        self.assertEqual(t.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(t.data_ptr(), ctypes.addressof(memory))
        memory[4] = 40.0
        self.assertEqual(t[1, 1].item(), 40.0)
        t[0, 2] = -2.0
        t.mul_(2)
        self.assertEqual(list(memory), [0.0, 2.0, -4.0, 6.0, 80.0, 10.0])
        # through the ggml fast paths as well: results are right and nothing else is written
        self.assertEqual((t + t).tolist(), [[0.0, 4.0, -8.0], [12.0, 160.0, 20.0]])
        self.assertEqual(torch.relu(t).tolist(), [[0.0, 2.0, 0.0], [6.0, 80.0, 10.0]])
        self.assertEqual((t @ t.t()).tolist(), [[20.0, 120.0], [120.0, 6536.0]])
        self.assertEqual(list(memory), [0.0, 2.0, -4.0, 6.0, 80.0, 10.0])

    def test_strides_select_the_elements(self):
        memory = (ctypes.c_int32 * 12)(*range(12))
        every_other = torch.Tensor(_ptr=_C.check_ptr(external(memory, (2, 3), (6, 2), torch.int32, 102)))
        self.assertEqual(every_other.tolist(), [[0, 2, 4], [6, 8, 10]])
        self.assertFalse(every_other.is_contiguous())
        transposed = torch.Tensor(_ptr=_C.check_ptr(external(memory, (4, 3), (1, 4), torch.int32, 103)))
        self.assertEqual(transposed.tolist(), [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])
        from_the_middle = torch.Tensor(_ptr=_C.check_ptr(external(memory, (2,), (1,), torch.int32, 104, offset=5 * 4)))
        self.assertEqual(from_the_middle.tolist(), [5, 6])
        every_other.fill_(-1)
        self.assertEqual(list(memory), [-1, 1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11])
        scalar = torch.Tensor(_ptr=_C.check_ptr(external(memory, (), (), torch.int32, 105, offset=4)))
        self.assertEqual((tuple(scalar.shape), scalar.item()), ((), 1))

    def test_the_token_is_reported_once_the_storage_is_gone(self):
        memory = (ctypes.c_double * 4)(1, 2, 3, 4)
        t = torch.Tensor(_ptr=_C.check_ptr(external(memory, (4,), (1,), torch.float64, 201)))
        view = t[1:3]
        other = t.detach()
        del t
        gc.collect()
        self.assertEqual(drain_released(), [])          # two tensors still use the memory
        self.assertEqual(view.tolist(), [2.0, 3.0])
        del view
        self.assertEqual(drain_released(), [])
        del other
        gc.collect()
        self.assertEqual(drain_released(), [201])
        self.assertEqual(drain_released(), [])

    def test_a_copy_does_not_keep_the_memory(self):
        memory = (ctypes.c_float * 3)(1, 2, 3)
        t = torch.Tensor(_ptr=_C.check_ptr(external(memory, (3,), (1,), torch.float32, 202)))
        copy, doubled, converted = t.clone(), t * 2, t.double()
        del t
        gc.collect()
        self.assertEqual(drain_released(), [202])
        memory[0] = 100.0
        self.assertEqual((copy.tolist(), doubled.tolist(), converted.tolist()), ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 2.0, 3.0]))

    def test_invalid_requests_are_errors_and_report_no_token(self):
        memory = (ctypes.c_float * 8)()
        cases = {
            "negative stride": lambda: external(memory, (2,), (-1,), torch.float32, 301),
            "empty": lambda: external(memory, (0,), (1,), torch.float32, 302),
            "no token": lambda: external(memory, (2,), (1,), torch.float32, 0),
            "unaligned": lambda: external(memory, (2,), (1,), torch.float32, 303, offset=1),
            "unaligned complex": lambda: external(memory, (1,), (1,), torch.complex64, 304, offset=2),
            "quantized": lambda: external(memory, (2,), (1,), torch.q8_0, 305),
            "too many dimensions": lambda: external(memory, (1,) * 13, (1,) * 13, torch.float32, 306),
            "overflow": lambda: external(memory, (2, 2), (2**62, 1), torch.float32, 307),
        }
        for name, make in cases.items():
            with self.subTest(name):
                self.assertFalse(make())
                self.assertIn("from_external", _C.last_error())
        null = _C.lib.gt_tensor_from_external(
            1, (ctypes.c_int64 * 1)(2), (ctypes.c_int64 * 1)(1), torch.float32.code, None, 308
        )
        self.assertFalse(null)
        self.assertEqual(drain_released(), [])

    def test_memory_that_is_not_aligned_for_ggml_buffers(self):
        # element-aligned, but not at the 32-byte boundary ggml asks of its buffers
        memory = (ctypes.c_float * 40)(*range(40))
        base = ctypes.addressof(memory)
        for start in range(1, 9):
            if (base + 4 * start) % 32 == 0:
                continue
            t = torch.Tensor(_ptr=_C.check_ptr(external(memory, (4,), (1,), torch.float32, 400 + start, offset=4 * start)))
            self.assertEqual(t.tolist(), [float(start + k) for k in range(4)])
            self.assertEqual((t * 2).sum().item(), float(2 * (4 * start + 6)))
            t.add_(0.5)
            self.assertEqual(memory[start], start + 0.5)
            t.sub_(0.5)
        self.assertEqual(list(memory), [float(k) for k in range(40)])


@needs_numpy
class TestFromNumpy(unittest.TestCase):
    def test_shares_memory_in_both_directions(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = torch.from_numpy(a)
        self.assertEqual((t.dtype, tuple(t.shape)), (torch.float32, (2, 3)))
        self.assertEqual(t.data_ptr(), a.ctypes.data)
        a[0, 0] = 100.0
        self.assertEqual(t[0, 0].item(), 100.0)
        t[1, 2] = -5.0
        self.assertEqual(a[1, 2], -5.0)
        t.add_(1)
        t.mul_(2)
        self.assertEqual(a.tolist(), [[202.0, 4.0, 6.0], [8.0, 10.0, -8.0]])
        t.zero_()
        self.assertEqual(a.sum(), 0.0)
        a += 3
        self.assertEqual(t.tolist(), [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
        with torch.no_grad():
            t.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.assertEqual(a.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    def test_views_of_the_tensor_share_too(self):
        a = np.zeros((3, 4), dtype=np.float64)
        t = torch.from_numpy(a)
        t[1].fill_(7)
        t.t()[3, 2] = -1
        t.reshape(12)[0] = 5
        self.assertEqual(a.tolist(), [[5.0, 0.0, 0.0, 0.0], [7.0, 7.0, 7.0, 7.0], [0.0, 0.0, 0.0, -1.0]])

    def test_every_supported_dtype(self):
        pairs = {
            np.float16: torch.float16, np.float32: torch.float32, np.float64: torch.float64,
            np.int8: torch.int8, np.uint8: torch.uint8, np.int16: torch.int16, np.int32: torch.int32,
            np.int64: torch.int64, np.bool_: torch.bool, np.complex64: torch.complex64, np.complex128: torch.complex128,
        }  # fmt: skip
        for np_dtype, dtype in pairs.items():
            with self.subTest(np.dtype(np_dtype).name):
                a = np.array([[0, 1, 0], [1, 1, 0]], dtype=np_dtype)
                t = torch.from_numpy(a)
                self.assertEqual(t.dtype, dtype)
                self.assertEqual(t.tolist(), a.tolist())
                a[0, 0] = 1
                self.assertEqual(t.tolist(), a.tolist())
                back = t.numpy()
                self.assertEqual(back.dtype, a.dtype)
                self.assertTrue(np.shares_memory(back, a))

    def test_strided_arrays_are_shared_as_they_are(self):
        a = np.arange(24, dtype=np.int64).reshape(4, 6)
        cases = {"step": a[::2, ::3], "transposed": a.T, "column": a[:, 4], "row": a[2], "tail": a[1:, 2:], "fortran": np.asfortranarray(a)}
        for name, arr in cases.items():
            with self.subTest(name):
                t = torch.from_numpy(arr)
                self.assertEqual(t.tolist(), arr.tolist())
                self.assertEqual(tuple(t.stride()), tuple(s // 8 for s in arr.strides))
                self.assertEqual(t.is_contiguous(), bool(arr.flags.c_contiguous))
                self.assertEqual(t.sum().item(), int(arr.sum()))
                self.assertEqual((t * 2).tolist(), (arr * 2).tolist())
        t = torch.from_numpy(a[::2, ::3])
        t[1, 1] = -1
        self.assertEqual(a[2, 3], -1)
        self.assertEqual(int((a == -1).sum()), 1)
        broadcast = np.broadcast_to(np.arange(3, dtype=np.float32), (2, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(torch.from_numpy(broadcast).tolist(), [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    def test_zero_dimensional_and_empty_arrays(self):
        a = np.array(3.5)
        t = torch.from_numpy(a)
        self.assertEqual((tuple(t.shape), t.dtype, t.item()), ((), torch.float64, 3.5))
        a[()] = 4.5
        self.assertEqual(t.item(), 4.5)
        e = torch.from_numpy(np.zeros((0, 3), dtype=np.int16))
        self.assertEqual((tuple(e.shape), e.dtype, e.tolist()), ((0, 3), torch.int16, []))

    def test_only_arrays_are_accepted(self):
        for value in ([1.0, 2.0], (1, 2), 3.0, 7, "abc", None, torch.ones(2), np.float32(1.5), np.int64(3)):
            with self.subTest(type(value).__name__):
                with self.assertRaises(TypeError) as ctx:
                    torch.from_numpy(value)
                self.assertEqual(str(ctx.exception), f"expected np.ndarray (got {type(value).__name__})")

    def test_element_types_without_a_tensor_dtype(self):
        arrays = {
            "uint16": np.array([1], dtype=np.uint16), "uint32": np.array([1], dtype=np.uint32),
            "uint64": np.array([1], dtype=np.uint64), "longdouble": np.array([1.0], dtype=np.longdouble),
            "str_": np.array(["a"]), "object_": np.array([object()]), "void": np.zeros(2, dtype=[("x", "f4")]),
            "datetime64": np.array(["2024-01-01"], dtype="datetime64[D]"),
        }  # fmt: skip
        for name, arr in arrays.items():
            with self.subTest(name):
                with self.assertRaises(TypeError) as ctx:
                    torch.from_numpy(arr)
                self.assertIn(f"can't convert np.ndarray of type numpy.{name}", str(ctx.exception))
                self.assertIn("The only supported types are: float64, float32", str(ctx.exception))

    def test_layouts_that_cannot_be_shared(self):
        a = np.arange(6, dtype=np.float32)
        raw = bytearray(16)
        cases = {
            "negative": (a[::-1], "stride in the given numpy array is negative"),
            "negative 2d": (a.reshape(2, 3)[:, ::-1], "stride in the given numpy array is negative"),
            "byte order": (a.astype(">f4"), "byte order different from the native byte order"),
            "unaligned": (np.frombuffer(raw, dtype=np.float32, count=3, offset=1), "not aligned"),
            "fractional stride": (
                np.lib.stride_tricks.as_strided(np.frombuffer(raw, dtype=np.float32, count=2), shape=(2,), strides=(2,)),
                "strides not a multiple of the element byte size",
            ),
        }
        for name, (arr, message) in cases.items():
            with self.subTest(name):
                for expected in (TypeError, ValueError):       # the audit's choice and PyTorch's
                    with self.assertRaises(expected) as ctx:
                        torch.from_numpy(arr)
                    self.assertIn(message, str(ctx.exception))
        # a copy is the documented way out
        self.assertEqual(torch.from_numpy(a[::-1].copy()).tolist(), [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    def test_the_array_lives_as_long_as_a_tensor_uses_it(self):
        class Tracked(np.ndarray):
            pass

        def make():
            arr = np.arange(1000, dtype=np.float64).view(Tracked)
            return torch.from_numpy(arr), weakref.ref(arr)

        t, ref = make()
        gc.collect()
        churn = [np.full(1000, -1.0) for _ in range(20)]
        self.assertIsNotNone(ref())
        self.assertEqual(t.sum().item(), 499500.0)
        view = t[10:13]
        detached = t.detach()
        del t
        gc.collect()
        self.assertIsNotNone(ref())                        # a view and a detached handle remain
        self.assertEqual(view.tolist(), [10.0, 11.0, 12.0])
        del view
        gc.collect()
        self.assertIsNotNone(ref())
        del detached
        gc.collect()
        self.assertIsNone(ref())                           # released with the last user
        del churn

    def test_the_autograd_graph_keeps_the_array_too(self):
        class Tracked(np.ndarray):
            pass

        def loss_on_temporary_data(w):
            data = np.arange(4, dtype=np.float32).view(Tracked)
            return (w * torch.from_numpy(data)).sum(), weakref.ref(data)

        w = torch.ones(4, requires_grad=True)
        loss, ref = loss_on_temporary_data(w)
        gc.collect()
        self.assertIsNotNone(ref())                        # saved for the backward pass
        loss.backward()
        self.assertEqual(w.grad.tolist(), [0.0, 1.0, 2.0, 3.0])
        del loss
        gc.collect()
        self.assertIsNone(ref())

    def test_a_shared_array_cannot_be_resized(self):
        a = np.arange(4.0)
        t = torch.from_numpy(a)
        with self.assertRaises(ValueError):
            a.resize(1000)                                 # numpy sees the extra reference
        self.assertEqual(t.tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_read_only_arrays_warn(self):
        a = np.arange(3.0)
        a.flags.writeable = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = torch.from_numpy(a)
        self.assertEqual(t.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("not writable", str(caught[0].message))
        self.assertTrue(caught[0].filename.endswith("test_numpy_interop.py"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch.from_numpy(np.arange(3.0))               # writable: no warning

    def test_as_tensor_shares_and_tensor_copies(self):
        a = np.arange(4, dtype=np.float32)
        shared = torch.as_tensor(a)
        copied = torch.tensor(a)
        a[0] = 9.0
        self.assertEqual(shared.tolist(), [9.0, 1.0, 2.0, 3.0])
        self.assertEqual(copied.tolist(), [0.0, 1.0, 2.0, 3.0])
        converted = torch.as_tensor(a, dtype=torch.float64)   # another dtype needs a copy
        a[1] = 8.0
        self.assertEqual((converted.dtype, converted.tolist()), (torch.float64, [9.0, 1.0, 2.0, 3.0]))
        t = torch.ones(2)
        self.assertIs(torch.as_tensor(t), t)
        self.assertEqual(torch.as_tensor([1, 2]).tolist(), [1, 2])


@needs_numpy
class TestTensorNumpy(unittest.TestCase):
    def test_shares_memory_in_both_directions(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        a = t.numpy()
        self.assertEqual((a.dtype, a.shape), (np.float32, (3, 4)))
        self.assertEqual(a.ctypes.data, t.data_ptr())
        self.assertFalse(a.flags.owndata)
        self.assertTrue(a.flags.writeable)
        a[0, 0] = 50.0
        self.assertEqual(t[0, 0].item(), 50.0)
        t[2, 3] = -1.0
        self.assertEqual(a[2, 3], -1.0)
        t.add_(1)
        self.assertEqual(a[0].tolist(), [51.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.shares_memory(a, t.numpy()))

    def test_views_keep_their_strides(self):
        t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        base = t.numpy()
        cases = {
            "transposed": t.transpose(0, 2), "permuted": t.permute(1, 2, 0), "step": t[:, ::2, 1::2],
            "row": t[1, 2], "tail": t[1:, 1:, 2:], "expanded": t[:, :1].expand(2, 5, 4), "element": t[1, 2, 3],
        }  # fmt: skip
        for name, view in cases.items():
            with self.subTest(name):
                arr = view.numpy()
                self.assertEqual(arr.tolist(), view.tolist())
                self.assertEqual(arr.strides, tuple(4 * s for s in view.stride()))
                self.assertTrue(np.shares_memory(arr, base))
        transposed = t.transpose(0, 2).numpy()
        transposed[3, 2, 1] = -7.0                          # element [1, 2, 3] of t
        self.assertEqual(t[1, 2, 3].item(), -7.0)

    def test_every_dtype(self):
        for dtype in (torch.float16, torch.float32, torch.float64, torch.int8, torch.uint8, torch.int16,
                      torch.int32, torch.int64, torch.bool, torch.complex64, torch.complex128):  # fmt: skip
            with self.subTest(str(dtype)):
                t = torch.tensor([[0, 1], [1, 1]], dtype=dtype)
                a = t.numpy()
                self.assertEqual(a.dtype.name, dtype.name)
                self.assertEqual(a.tolist(), t.tolist())
                a[0, 0] = 1
                self.assertEqual(t.tolist(), a.tolist())

    def test_zero_dimensional_and_empty(self):
        a = torch.tensor(2.5).numpy()
        self.assertEqual((a.shape, a.dtype, float(a)), ((), np.float32, 2.5))
        e = torch.zeros(0, 3, dtype=torch.int32).numpy()
        self.assertEqual((e.shape, e.dtype), ((0, 3), np.int32))

    def test_the_array_keeps_the_memory_alive(self):
        a = torch.arange(1000, dtype=torch.float64).numpy()       # the tensor is gone at once
        gc.collect()
        churn = [torch.full((1000,), -1.0, dtype=torch.float64) for _ in range(20)]
        self.assertEqual(float(a.sum()), 499500.0)
        part = torch.arange(1000, dtype=torch.float64).numpy()[10:13]   # so is the first array
        gc.collect()
        churn += [torch.full((1000,), -1.0, dtype=torch.float64) for _ in range(20)]
        self.assertEqual(part.tolist(), [10.0, 11.0, 12.0])

    def test_the_array_survives_a_conversion_of_the_tensor_in_place(self):
        # Module.double() gives the Parameter object new storage; an array taken before
        # must keep the old memory rather than point into freed memory.
        layer = nn.Linear(3, 2)
        before = layer.weight.detach().numpy()
        values = before.tolist()
        layer.double()
        gc.collect()
        churn = [torch.full((2, 3), -1.0) for _ in range(50)]
        self.assertEqual(before.tolist(), values)
        self.assertEqual(before.dtype, np.float32)
        self.assertEqual(layer.weight.detach().numpy().dtype, np.float64)
        del churn

    def test_requires_grad(self):
        w = torch.ones(3, requires_grad=True)
        with self.assertRaises(RuntimeError) as ctx:
            w.numpy()
        self.assertEqual(str(ctx.exception), "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        with self.assertRaises(RuntimeError):
            (w * 2).numpy()
        forced = w.numpy(force=True)
        self.assertEqual(forced.tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(np.shares_memory(forced, w.detach().numpy()))

    def test_types_numpy_does_not_have(self):
        bf = torch.tensor([1.0, 2.5], dtype=torch.bfloat16)
        quantized = (torch.arange(64, dtype=torch.float32) / 64).quantize(torch.q8_0)
        for name, t in {"bfloat16": bf, "q8_0": quantized}.items():
            with self.subTest(name):
                with self.assertRaises(TypeError) as ctx:
                    t.numpy()
                self.assertIn(f"Got unsupported ScalarType {name}", str(ctx.exception))
                with self.assertRaises(TypeError):
                    t.numpy(force=False)
                forced = t.numpy(force=True)
                self.assertEqual((forced.dtype, forced.shape), (np.float32, tuple(t.shape)))
        self.assertEqual(bf.numpy(force=True).tolist(), [1.0, 2.5])
        self.assertLess(float(np.abs(quantized.numpy(force=True) - np.arange(64) / 64).max()), 0.01)

    def test_round_trip(self):
        a = np.arange(6, dtype=np.int32).reshape(2, 3)
        back = torch.from_numpy(a).t().numpy()
        self.assertTrue(np.shares_memory(back, a))
        self.assertEqual(back.tolist(), a.T.tolist())
        t = torch.arange(6.0)
        again = torch.from_numpy(t.numpy()[::2])
        again[1] = -1.0
        self.assertEqual(t.tolist(), [0.0, 1.0, -1.0, 3.0, 4.0, 5.0])


@needs_numpy
class TestArrayProtocol(unittest.TestCase):
    def setUp(self):
        self.t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def test_asarray_gives_a_typed_array_on_the_same_memory(self):
        a = np.asarray(self.t)
        self.assertEqual((a.dtype, a.shape, a.tolist()), (np.float32, (2, 2), [[1.0, 2.0], [3.0, 4.0]]))
        self.assertTrue(np.shares_memory(a, self.t.numpy()))
        self.assertEqual(np.asarray(torch.tensor(3)).shape, ())
        self.assertEqual(np.asarray(torch.tensor([True, False])).dtype, np.bool_)
        self.assertEqual(np.asarray(self.t.t()).tolist(), [[1.0, 3.0], [2.0, 4.0]])

    def test_array_copies(self):
        a = np.array(self.t)
        self.assertFalse(np.shares_memory(a, self.t.numpy()))
        a[0, 0] = 9.0
        self.assertEqual(self.t[0, 0].item(), 1.0)

    def test_dtype_and_copy_arguments(self):
        as_double = np.asarray(self.t, dtype=np.float64)
        self.assertEqual((as_double.dtype, as_double.tolist()), (np.float64, [[1.0, 2.0], [3.0, 4.0]]))
        self.assertTrue(np.shares_memory(np.asarray(self.t, dtype=np.float32), self.t.numpy()))
        self.assertEqual(self.t.__array__(np.int32).dtype, np.int32)             # the NumPy 1 call
        self.assertTrue(np.shares_memory(self.t.__array__(copy=False), self.t.numpy()))
        self.assertFalse(np.shares_memory(self.t.__array__(copy=True), self.t.numpy()))
        with self.assertRaises(ValueError):
            self.t.__array__(dtype=np.float64, copy=False)

    def test_a_tensor_that_requires_grad_is_not_converted_silently(self):
        with self.assertRaises(RuntimeError):
            np.asarray(torch.ones(2, requires_grad=True))

    def test_functions_that_convert_their_arguments(self):
        t = self.t
        self.assertEqual(np.concatenate([t, t]).shape, (4, 2))
        self.assertEqual(np.stack([t, t.numpy()]).dtype, np.float32)
        self.assertTrue(np.allclose(t, t.numpy()))
        np.testing.assert_allclose(t, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(np.dot(t, t).tolist(), [[7.0, 10.0], [15.0, 22.0]])
        self.assertAlmostEqual(float(np.linalg.norm(t)), 30.0**0.5, places=5)
        self.assertEqual(np.einsum("ij,ij->i", t, t).tolist(), [5.0, 25.0])
        self.assertEqual([x.tolist() for x in np.where(t > 2)], [[1, 1], [0, 1]])

    def test_numpy_reductions_call_the_tensor_methods(self):
        t = self.t
        for name, got, want in (
            ("sum", np.sum(t), t.sum()),
            ("sum axis", np.sum(t, axis=0), t.sum(0)),
            ("sum keepdims", np.sum(t, axis=1, keepdims=True), t.sum(1, keepdim=True)),
            ("mean", np.mean(t), t.mean()),
            ("mean axis", np.mean(t, axis=1), t.mean(1)),
            ("prod", np.prod(t), t.prod()),
            ("max", np.max(t), t.max()),
            ("min", np.min(t), t.min()),
            ("all", np.all(t > 0), (t > 0).all()),
            ("any", np.any(t > 3, axis=1), (t > 3).any(1)),
            ("argmax", np.argmax(t, axis=1), t.argmax(1)),
        ):
            with self.subTest(name):
                self.assertIsInstance(got, torch.Tensor)
                self.assertEqual((got.tolist(), got.dtype), (want.tolist(), want.dtype))

    def test_axis_and_keepdims_spellings(self):
        t = self.t
        self.assertEqual(t.sum(axis=0).tolist(), [4.0, 6.0])
        self.assertEqual(t.mean(axis=1, keepdims=True).tolist(), [[1.5], [3.5]])
        self.assertEqual(t.max(axis=1)[0].tolist(), [2.0, 4.0])
        self.assertEqual(t.argmin(axis=0).tolist(), [0, 0])
        self.assertEqual(t.sum(axis=(0, 1), out=None).item(), 10.0)
        for call in (lambda: t.sum(axis=0, dim=0), lambda: t.sum(keepdim=True, keepdims=True), lambda: t.sum(out=torch.zeros(()))):
            with self.assertRaises(TypeError):
                call()


@needs_numpy
class TestUfuncs(unittest.TestCase):
    def setUp(self):
        self.t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.a = np.array([[10.0, 20.0], [30.0, 40.0]])

    def assert_tensor(self, got, values, dtype=None):
        self.assertIsInstance(got, torch.Tensor)
        self.assertEqual(got.tolist(), values)
        if dtype is not None:
            self.assertEqual(got.dtype, dtype)

    def test_a_ufunc_of_a_tensor_is_a_tensor(self):
        got = np.sin(self.t)
        self.assert_tensor(got, self.t.sin().tolist(), torch.float32)
        self.assert_tensor(np.sqrt(torch.tensor(4.0)), 2.0, torch.float32)       # 0-d stays a tensor
        self.assert_tensor(np.negative(torch.tensor([1, -2])), [-1, 2], torch.int64)
        self.assert_tensor(np.abs(-self.t), self.t.tolist(), torch.float32)
        self.assert_tensor(np.isnan(self.t), [[False, False], [False, False]], torch.bool)

    def test_array_and_tensor_operands(self):
        t, a = self.t, self.a
        self.assert_tensor(a + t, [[11.0, 22.0], [33.0, 44.0]], torch.float64)
        self.assert_tensor(t + a, [[11.0, 22.0], [33.0, 44.0]], torch.float64)
        self.assert_tensor(a - t, [[9.0, 18.0], [27.0, 36.0]], torch.float64)
        self.assert_tensor(a * t, [[10.0, 40.0], [90.0, 160.0]], torch.float64)
        self.assert_tensor(a / t, [[10.0, 10.0], [10.0, 10.0]], torch.float64)
        self.assert_tensor(a // t, [[10.0, 10.0], [10.0, 10.0]])
        self.assert_tensor(a % t, [[0.0, 0.0], [0.0, 0.0]])
        self.assert_tensor(a @ t, [[70.0, 100.0], [150.0, 220.0]], torch.float64)
        self.assert_tensor(a.astype(np.float32) ** t, [[10.0, 400.0], [27000.0, 2560000.0]], torch.float32)
        self.assert_tensor(a > t * 10, [[False, False], [False, False]], torch.bool)
        self.assert_tensor(a == t * 10, [[True, True], [True, True]], torch.bool)
        self.assert_tensor(a != t * 10, [[False, False], [False, False]], torch.bool)
        self.assert_tensor(np.arange(4).reshape(2, 2) & torch.tensor([[1, 1], [2, 2]]), [[0, 1], [2, 2]], torch.int64)
        self.assert_tensor(np.maximum(a, t * 15), [[15.0, 30.0], [45.0, 60.0]])
        # the array is not written
        self.assertEqual(a.tolist(), [[10.0, 20.0], [30.0, 40.0]])

    def test_numpy_scalar_on_the_left(self):
        t = self.t
        ints = torch.tensor([1, 2, 3])
        self.assert_tensor(np.float32(2) + t, [[3.0, 4.0], [5.0, 6.0]], torch.float32)
        self.assert_tensor(np.float64(2) * t, [[2.0, 4.0], [6.0, 8.0]], torch.float32)     # a scalar does not widen
        self.assert_tensor(np.float64(12) / t, [[12.0, 6.0], [4.0, 3.0]], torch.float32)
        self.assert_tensor(np.int64(2) - ints, [1, 0, -1], torch.int64)
        self.assert_tensor(np.float32(2.5) - ints, [1.5, 0.5, -0.5], torch.float32)
        self.assert_tensor(np.float32(2) ** t, [[2.0, 4.0], [8.0, 16.0]], torch.float32)
        self.assert_tensor(np.int64(6) & ints, [0, 2, 2], torch.int64)
        self.assert_tensor(np.float32(2.5) > ints, [True, True, False], torch.bool)
        self.assert_tensor(np.int32(2) == ints, [False, True, False], torch.bool)
        # exactly what the Python number of the same value gives
        for value in (np.float32(2), np.int64(3), np.float64(0.5)):
            for op in (lambda s, x: s + x, lambda s, x: s - x, lambda s, x: s * x, lambda s, x: s / x, lambda s, x: s**x):
                got, want = op(value, ints), op(value.item(), ints)
                self.assertEqual((got.tolist(), got.dtype), (want.tolist(), want.dtype))

    def test_gradients_flow_through_dispatched_ufuncs(self):
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        loss = (self.a * w).sum() + np.sin(w).sum() + (np.float32(3) - w).sum() + (self.a @ w).sum()
        self.assertIsInstance(loss, torch.Tensor)
        loss.backward()
        want = self.a + np.cos(w.detach().numpy()) - 1.0 + self.a.sum(axis=0)[:, None]
        np.testing.assert_allclose(w.grad.numpy(), want, rtol=1e-6)

    def test_ufuncs_without_a_tensor_counterpart_are_computed_by_numpy(self):
        t, a = self.t, self.a
        got = np.hypot(t, a)
        self.assertIsInstance(got, torch.Tensor)
        np.testing.assert_allclose(got.numpy(), np.hypot(t.numpy(), a))
        frac, whole = np.modf(t + 0.5)
        self.assert_tensor(frac, [[0.5, 0.5], [0.5, 0.5]], torch.float32)
        self.assert_tensor(whole, [[1.0, 2.0], [3.0, 4.0]], torch.float32)
        # ... which cannot follow a gradient, and says so instead of dropping it
        w = torch.ones(2, requires_grad=True)
        with self.assertRaises(RuntimeError) as ctx:
            np.hypot(w, 1.0)
        self.assertIn("requires grad", str(ctx.exception))

    def test_out_argument(self):
        t, a = self.t, self.a
        target = np.zeros((2, 2))
        self.assertIs(np.add(t, a, out=target), target)
        self.assertEqual(target.tolist(), [[11.0, 22.0], [33.0, 44.0]])
        acc = a.copy()
        acc += t                                             # in place on the array
        self.assertIsInstance(acc, np.ndarray)
        self.assertEqual(acc.tolist(), [[11.0, 22.0], [33.0, 44.0]])
        out = torch.zeros(2, 2)
        version = out._version
        self.assertIs(np.multiply(t, 3, out=out), out)       # in place on a tensor
        self.assertEqual(out.tolist(), [[3.0, 6.0], [9.0, 12.0]])
        self.assertGreater(out._version, version)
        masked = np.sin(t, where=np.array([[True, False], [True, True]]), out=np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(masked[0, 1], 0.0)
        self.assertAlmostEqual(float(masked[1, 1]), float(np.sin(4.0)), places=6)

    def test_ufunc_methods(self):
        t = self.t
        self.assert_tensor(np.add.reduce(t), [4.0, 6.0], torch.float32)
        self.assert_tensor(np.add.reduce(t, axis=None), 10.0, torch.float32)
        self.assert_tensor(np.multiply.outer(torch.tensor([1, 2]), torch.tensor([3, 4])), [[3, 4], [6, 8]], torch.int64)
        self.assert_tensor(np.add.accumulate(t, axis=1), [[1.0, 3.0], [3.0, 7.0]], torch.float32)
        counts = torch.zeros(4)
        version = counts._version
        self.assertIsNone(np.add.at(counts, [0, 0, 2], 1.0))  # writes into the tensor
        self.assertEqual(counts.tolist(), [2.0, 0.0, 1.0, 0.0])
        self.assertGreater(counts._version, version)

    def test_every_dispatched_ufunc_agrees_with_numpy(self):
        rng = np.random.default_rng(0)
        domains = {
            "arccosh": (1.1, 3.0), "arcsin": (-0.9, 0.9), "arccos": (-0.9, 0.9), "arctanh": (-0.9, 0.9),
            "log": (0.2, 3.0), "log2": (0.2, 3.0), "log10": (0.2, 3.0), "log1p": (0.2, 3.0), "sqrt": (0.2, 3.0),
            "power": (0.2, 3.0),
        }  # fmt: skip
        halves = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.49, -3.51])
        checked = 0
        for name in sorted(tensor_module._UFUNC_TENSOR_OPS):
            ufunc = getattr(np, name)
            for np_dtype in (np.float32, np.float64):
                if name in ("bitwise_and", "bitwise_or", "bitwise_xor", "invert"):
                    xs = [rng.integers(0, 16, 9) for _ in range(ufunc.nin)]
                elif name == "matmul":
                    xs = [rng.uniform(0.5, 2.0, (3, 3)).astype(np_dtype) for _ in range(2)]
                elif name in ("rint", "floor", "ceil", "trunc", "sign", "fabs", "absolute", "negative"):
                    xs = [halves.astype(np_dtype)]
                else:
                    lo, hi = domains.get(name, (-2.0, 2.0))
                    xs = [rng.uniform(lo, hi, 9).astype(np_dtype) for _ in range(ufunc.nin)]
                with self.subTest(ufunc=name, dtype=np.dtype(np_dtype).name):
                    want = ufunc(*xs)
                    got = ufunc(*[torch.tensor(x) for x in xs])
                    self.assertIsInstance(got, torch.Tensor)
                    self.assertEqual(got.numpy().dtype, want.dtype)
                    np.testing.assert_allclose(got.numpy(), want, rtol=2e-5, atol=1e-6)
                    if ufunc.nin == 2:                       # an array first, a tensor second
                        mixed = ufunc(xs[0], torch.tensor(xs[1]))
                        np.testing.assert_allclose(mixed.numpy(), want, rtol=2e-5, atol=1e-6)
                    checked += 1
        self.assertGreater(checked, 100)


@needs_numpy
class TestNumpyScalarsAndNestedArrays(unittest.TestCase):
    def test_a_numpy_scalar_becomes_a_tensor_of_its_type(self):
        cases = [
            (np.float16(1.5), torch.float16), (np.float32(1.5), torch.float32), (np.float64(1.5), torch.float64),
            (np.int8(3), torch.int8), (np.uint8(3), torch.uint8), (np.int16(3), torch.int16), (np.int32(3), torch.int32),
            (np.int64(3), torch.int64), (np.bool_(True), torch.bool), (np.complex64(1 + 2j), torch.complex64),
            (np.complex128(1 - 2j), torch.complex128),
        ]  # fmt: skip
        for value, dtype in cases:
            with self.subTest(type(value).__name__):
                t = torch.tensor(value)
                self.assertEqual((t.dtype, tuple(t.shape), t.item()), (dtype, (), value.item()))
        self.assertEqual(torch.tensor(np.float64(1.5), dtype=torch.int32).tolist(), 1)
        self.assertEqual(torch.as_tensor(np.int32(7)).dtype, torch.int32)
        self.assertEqual(torch.tensor(np.array(1.5)).shape, ())                  # 0-d stays 0-d

    def test_lists_of_numpy_scalars(self):
        self.assertEqual(torch.tensor([np.int64(1), np.int64(2)]).tolist(), [1, 2])
        self.assertEqual(torch.tensor([np.int64(1), np.int64(2)]).dtype, torch.int64)
        self.assertEqual(torch.tensor([np.float32(1), np.float32(2)]).dtype, torch.float32)
        self.assertEqual(torch.tensor([[np.float16(1)], [np.float16(2)]]).dtype, torch.float16)
        self.assertEqual(torch.tensor([np.bool_(True), np.bool_(False)]).tolist(), [True, False])
        # mixed with Python numbers: promoted
        self.assertEqual(torch.tensor([np.float32(1), 2.0]).dtype, torch.float32)
        self.assertEqual(torch.tensor([np.float64(1), 2.0]).dtype, torch.float64)
        self.assertEqual(torch.tensor([np.int32(1), 2]).dtype, torch.int64)
        self.assertEqual(torch.tensor([np.int8(1), 2.5]).tolist(), [1.0, 2.5])
        self.assertEqual(torch.tensor([np.float64(1.5)], dtype=torch.float32).dtype, torch.float32)

    def test_nested_arrays(self):
        t = torch.tensor([np.zeros(3), np.ones(3)])
        self.assertEqual((tuple(t.shape), t.dtype, t.tolist()), ((2, 3), torch.float64, [[0.0] * 3, [1.0] * 3]))
        t = torch.tensor([np.arange(3, dtype=np.float32), [1, 2, 3]])
        self.assertEqual((tuple(t.shape), t.dtype), ((2, 3), torch.float32))
        t = torch.tensor([[np.arange(2), np.arange(2)], [np.arange(2), np.arange(2)]])
        self.assertEqual((tuple(t.shape), t.dtype), ((2, 2, 2), torch.int64))
        t = torch.tensor([torch.ones(2), np.zeros(2, dtype=np.float32)])
        self.assertEqual(t.tolist(), [[1.0, 1.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            torch.tensor([np.zeros(3), np.zeros(2)])

    def test_arrays_of_any_layout_are_copied_right(self):
        a = np.arange(12, dtype=np.int32).reshape(3, 4)
        for name, arr in {"reversed": a[::-1, ::2], "fortran": np.asfortranarray(a), "transposed": a.T, "big endian": a.astype(">i4")}.items():
            with self.subTest(name):
                t = torch.tensor(arr)
                self.assertEqual((t.tolist(), t.dtype), (arr.tolist(), torch.int32))
                self.assertFalse(np.shares_memory(t.numpy(), a))

    def test_types_without_a_tensor_dtype_are_type_errors(self):
        for value in (np.uint16(3), np.array([1, 2], dtype=np.uint64), np.longdouble(1.5), [np.uint32(1)], np.array(["a"])):
            with self.subTest(repr(value)):
                with self.assertRaises(TypeError) as ctx:
                    torch.tensor(value)
                self.assertIn("The only supported types are", str(ctx.exception))

    def test_objects_with_the_array_protocol(self):
        class Wrapped:
            def __array__(self, dtype=None, copy=None):
                return np.arange(3, dtype=np.int16)

        self.assertEqual(torch.tensor(Wrapped()).tolist(), [0, 1, 2])
        self.assertEqual(torch.tensor([Wrapped(), Wrapped()]).dtype, torch.int16)

    def test_numpy_scalars_in_arithmetic_behave_like_python_numbers(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ints = torch.tensor([1, 2, 3])
        cases = [
            (lambda: t + np.float32(2.0), t + 2.0), (lambda: t - np.int32(1), t - 1), (lambda: t * np.float64(2.5), t * 2.5),
            (lambda: t / np.float16(2), t / 2.0), (lambda: t ** np.int64(2), t**2), (lambda: t // np.int64(2), t // 2),
            (lambda: t % np.float32(2), t % 2.0), (lambda: ints + np.float32(2.5), ints + 2.5),
            (lambda: ints * np.int64(3), ints * 3), (lambda: ints // np.int64(2), ints // 2), (lambda: ints % np.int64(2), ints % 2),
            (lambda: ints & np.int64(1), ints & 1), (lambda: ints | np.int64(4), ints | 4), (lambda: ints ^ np.int64(1), ints ^ 1),
            (lambda: t > np.float32(2), t > 2.0), (lambda: t >= np.int64(2), t >= 2), (lambda: t == np.int64(2), t == 2),
            (lambda: t != np.float64(2), t != 2.0), (lambda: t < np.float16(2.5), t < 2.5), (lambda: t <= np.int8(3), t <= 3),
            (lambda: (ints > 1) & np.bool_(True), (ints > 1) & True), (lambda: t * np.bool_(True), t * True),
            (lambda: t.add(np.float32(1)), t.add(1.0)), (lambda: t.sub(np.int64(1)), t.sub(1)),
            (lambda: t.mul(np.float32(2)), t.mul(2.0)), (lambda: t.div(np.float32(2)), t.div(2.0)),
            (lambda: t.pow(np.float32(2)), t.pow(2.0)), (lambda: torch.add(t, np.float32(1)), torch.add(t, 1.0)),
            (lambda: torch.pow(np.float32(2), t), torch.pow(2.0, t)), (lambda: t.clamp(np.float32(1.5), np.int64(3)), t.clamp(1.5, 3)),
            (lambda: t.masked_fill(t > 2, np.float32(-1)), t.masked_fill(t > 2, -1.0)),
            (lambda: torch.where(t > 2, t, np.float32(0)), torch.where(t > 2, t, 0.0)),
            (lambda: torch.maximum(t, torch.as_tensor(np.float32(2.5))), torch.maximum(t, torch.tensor(2.5))),
        ]  # fmt: skip
        for k, (make, want) in enumerate(cases):
            with self.subTest(case=k):
                got = make()
                self.assertEqual((got.tolist(), got.dtype), (want.tolist(), want.dtype))

    def test_a_scalar_does_not_decide_the_dtype(self):
        t32 = torch.ones(2)
        self.assertEqual((t32 + np.float64(1.0)).dtype, torch.float32)
        self.assertEqual((np.float64(1.0) + t32).dtype, torch.float32)
        self.assertEqual((torch.ones(2, dtype=torch.float64) * np.float32(2)).dtype, torch.float64)
        self.assertEqual((torch.tensor([1, 2]) + np.int8(1)).dtype, torch.int64)

    def test_in_place_operations(self):
        t = torch.tensor([1.0, 2.0])
        t += np.float32(1)
        t *= np.int64(2)
        t -= np.float64(0.5)
        t /= np.float32(2)
        self.assertEqual(t.tolist(), [1.75, 2.75])
        self.assertEqual(t.fill_(np.float32(7)).tolist(), [7.0, 7.0])
        self.assertEqual(torch.zeros(2, dtype=torch.int64).fill_(np.int64(2**40)).tolist(), [2**40, 2**40])
        self.assertEqual(torch.zeros(2, dtype=torch.bool).fill_(np.bool_(True)).tolist(), [True, True])
        self.assertEqual(t.clamp_(np.float32(8), np.int64(9)).tolist(), [8.0, 8.0])

    def test_creation_functions(self):
        self.assertEqual(torch.full((2,), np.float32(1.5)).tolist(), [1.5, 1.5])
        self.assertEqual(torch.full((np.int64(2),), np.int64(3)).dtype, torch.int64)
        self.assertEqual(torch.full(np.int64(2), np.bool_(True)).tolist(), [True, True])
        self.assertEqual(torch.arange(np.int64(1), np.int64(4)).tolist(), [1, 2, 3])
        floats = torch.arange(np.float32(0), np.float32(2), np.float32(0.5))
        self.assertEqual((floats.tolist(), floats.dtype), ([0.0, 0.5, 1.0, 1.5], torch.float32))
        self.assertEqual(torch.arange(np.float64(3)).dtype, torch.float32)
        self.assertEqual(tuple(torch.zeros(np.int64(2), np.int32(3)).shape), (2, 3))
        self.assertEqual(tuple(torch.ones((np.int64(2),)).shape), (2,))
        self.assertEqual(torch.linspace(0, 1, np.int64(5)).tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(tuple(nn.Linear(np.int64(4), np.int64(2))(torch.ones(3, 4)).shape), (3, 2))

    def test_dimensions_given_as_numpy_integers(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        i0, i1, i2 = np.int64(0), np.int64(1), np.int32(2)
        cases = [
            (lambda: t.sum(dim=i0), t.sum(dim=0)), (lambda: t.sum(dim=(i0, i1)), t.sum(dim=(0, 1))), (lambda: t.mean(i1), t.mean(1)),
            (lambda: t.max(dim=i1)[1], t.max(dim=1)[1]), (lambda: t.argmax(dim=i1), t.argmax(dim=1)), (lambda: t.cumsum(i1), t.cumsum(1)),
            (lambda: t.transpose(i0, i1), t.transpose(0, 1)), (lambda: t.permute(i1, i0), t.permute(1, 0)),
            (lambda: t.unsqueeze(i1), t.unsqueeze(1)), (lambda: t[:1].squeeze(i0), t[:1].squeeze(0)),
            (lambda: t.view(i2, np.int64(6)), t.view(2, 6)), (lambda: t.reshape((np.int64(4), np.int64(3))), t.reshape(4, 3)),
            (lambda: t.select(i0, i1), t.select(0, 1)), (lambda: t.flip(i1), t.flip(1)), (lambda: t.roll(i1, i1), t.roll(1, 1)),
            (lambda: t.movedim(i0, i1), t.movedim(0, 1)), (lambda: t.split(i2, i1)[1], t.split(2, 1)[1]),
            (lambda: t.chunk(i2, i1)[0], t.chunk(2, 1)[0]), (lambda: torch.cat([t, t], dim=i1), torch.cat([t, t], dim=1)),
            (lambda: t.topk(i2)[0], t.topk(2)[0]), (lambda: torch.softmax(t, dim=i1), torch.softmax(t, dim=1)),
        ]  # fmt: skip
        for k, (make, want) in enumerate(cases):
            with self.subTest(case=k):
                got = make()
                self.assertEqual((got.tolist(), tuple(got.shape)), (want.tolist(), tuple(want.shape)))


@needs_numpy
class TestNumpyIndices(unittest.TestCase):
    def setUp(self):
        self.t = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def test_numpy_integers(self):
        t = self.t
        self.assertEqual(t[np.int64(0)].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(t[np.int32(1), np.int64(-1)].item(), 7.0)
        self.assertEqual(t[np.uint8(2), np.int16(0)].item(), 8.0)
        self.assertEqual(t[..., np.int64(1)].tolist(), [1.0, 5.0, 9.0])
        self.assertEqual(t[None, np.int64(1)].tolist(), [[4.0, 5.0, 6.0, 7.0]])
        with self.assertRaises(IndexError):
            t[np.int64(3)]

    def test_slices_with_numpy_bounds(self):
        t = self.t
        self.assertEqual(t[np.int64(1) : np.int64(3), np.int64(0)].tolist(), [4.0, 8.0])
        self.assertEqual(t[0, :: np.int64(2)].tolist(), [0.0, 2.0])
        self.assertEqual(t[0, np.int64(-2) :].tolist(), [2.0, 3.0])

    def test_arrays_as_indices(self):
        t = self.t
        self.assertEqual(t[np.array([0, 2])].tolist(), [[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]])
        self.assertEqual(t[np.array([0, 2]), np.array([1, 3])].tolist(), [1.0, 11.0])
        self.assertEqual(t[:, np.array([[0, 1], [2, 3]])].shape, (3, 2, 2))
        self.assertEqual(t[np.array([0, 2], dtype=np.uint32)].tolist(), t[[0, 2]].tolist())
        self.assertEqual(t[np.array([True, False, True])].tolist(), t[[0, 2]].tolist())
        self.assertEqual(t[t.numpy() > 8].tolist(), [9.0, 10.0, 11.0])
        self.assertEqual(t[np.int64(1), np.array([0, 3])].tolist(), [4.0, 7.0])
        self.assertEqual(t[(0, 2), :].tolist(), t[[0, 2]].tolist())             # a tuple inside the index
        with self.assertRaises(IndexError):
            t[np.array([0, 3])]
        with self.assertRaises(IndexError):
            t[np.array([0.0, 1.0])]
        with self.assertRaises(IndexError):
            t[np.float32(1)]

    def test_assignment(self):
        t = self.t.clone()
        t[np.int64(0)] = np.float32(9)
        t[np.int64(1), np.array([0, 3])] = np.array([70.0, 71.0])
        t[np.array([[False] * 4, [False] * 4, [True, False, False, True]])] = np.float64(5)
        t[2, np.int64(1) : np.int64(3)] = np.arange(2, dtype=np.int64)
        self.assertEqual(t.tolist(), [[9.0, 9.0, 9.0, 9.0], [70.0, 5.0, 6.0, 71.0], [5.0, 0.0, 1.0, 5.0]])
        self.assertEqual(t.dtype, torch.float32)

    def test_gradient_through_an_array_index(self):
        x = torch.arange(6, dtype=torch.float32).requires_grad_(True)
        (x[np.array([0, 0, 5])] * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
        self.assertEqual(x.grad.tolist(), [3.0, 0.0, 0.0, 0.0, 0.0, 3.0])


class TestBooleanScalarIndex(unittest.TestCase):
    """True / False as an index are a 0-d mask (numpy, torch): not the integers 1 and 0."""

    def test_true_adds_a_dimension_and_false_selects_nothing(self):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertEqual(t[True].tolist(), [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])
        self.assertEqual(tuple(t[False].shape), (0, 2, 3))
        self.assertEqual(tuple(t[1, True].shape), (1, 3))
        self.assertEqual(t[1, True].tolist(), [[3.0, 4.0, 5.0]])
        self.assertEqual(tuple(t[..., False].shape), (2, 3, 0))
        self.assertEqual(tuple(torch.tensor(5.0)[True].shape), (1,))

    def test_assignment(self):
        t = torch.zeros(2, 3)
        t[True] = 5.0
        self.assertEqual(t.tolist(), [[5.0] * 3, [5.0] * 3])
        t[False] = 9.0
        self.assertEqual(t.tolist(), [[5.0] * 3, [5.0] * 3])

    @needs_numpy
    def test_numpy_bool(self):
        t = torch.arange(3)
        self.assertEqual(t[np.bool_(True)].tolist(), [[0, 1, 2]])
        self.assertEqual(tuple(t[np.bool_(False)].shape), (0, 3))


class TestNestedDataWithoutNumpy(unittest.TestCase):
    """The parts of the new nested-data code that do not need NumPy."""

    def test_lists_of_tensors(self):
        rows = torch.tensor([torch.ones(3), torch.zeros(3)])
        self.assertEqual((tuple(rows.shape), rows.tolist()), ((2, 3), [[1.0] * 3, [0.0] * 3]))
        self.assertEqual(torch.tensor([torch.tensor(1), torch.tensor(2)]).tolist(), [1, 2])
        self.assertEqual(torch.tensor([torch.tensor(1.0, dtype=torch.float64)]).dtype, torch.float64)
        self.assertEqual(torch.tensor([[torch.ones(2), [3, 4]]]).tolist(), [[[1.0, 1.0], [3.0, 4.0]]])
        with self.assertRaises(ValueError):
            torch.tensor([torch.ones(3), torch.ones(2)])

    def test_python_numbers_keep_their_default_types(self):
        self.assertEqual(torch.tensor([1, 2]).dtype, torch.int64)
        self.assertEqual(torch.tensor([1.5, 2]).dtype, torch.float32)
        self.assertEqual(torch.tensor([True, False]).dtype, torch.bool)
        self.assertEqual(torch.tensor([True, 2]).dtype, torch.int64)
        self.assertEqual(torch.tensor([1.5, 2 + 1j]).dtype, torch.complex64)
        self.assertEqual(torch.tensor([]).dtype, torch.float32)
        self.assertEqual(tuple(torch.tensor([[], []]).shape), (2, 0))
        self.assertEqual(torch.tensor(range(4)).tolist(), [0, 1, 2, 3])
        self.assertEqual(torch.tensor((1, (2,))[1]).tolist(), [2])

    def test_things_that_are_not_data(self):
        for value in ("abc", [1, "a"], object(), [None], {"a": 1}, float):
            with self.subTest(repr(value)):
                with self.assertRaises(TypeError):
                    torch.tensor(value)


if __name__ == "__main__":
    unittest.main()
