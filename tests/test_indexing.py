"""Index operations: bounds, shapes and gradients (BUG-09, BUG-11).

Before: an index outside its dimension was clamped to the nearest valid element (gather,
index_select, x[tensor], x[tensor] = v), skipped (scatter), replaced by row 0 or fatal
(embedding); an oversized gather index and an empty dimension read memory that belongs to
nobody; masks were matched element by element instead of broadcast; and x[tensor], x[mask],
x[list] and masked_select returned tensors cut off from the autograd graph.
"""

import unittest

import torch
import torch.nn.functional as F


def grid(*shape):
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=torch.float32).reshape(*shape)


class TestGatherScatterBounds(unittest.TestCase):
    def setUp(self):
        self.a = grid(3, 4)

    def test_gather_values_and_negative_indices(self):
        self.assertEqual(torch.gather(self.a, 1, torch.tensor([[0, 3], [1, 1], [2, 0]])).tolist(), [[0, 3], [5, 5], [10, 8]])
        self.assertEqual(torch.gather(self.a, 1, torch.tensor([[-1], [-4], [0]])).tolist(), [[3], [4], [8]])
        self.assertEqual(torch.gather(self.a, 0, torch.tensor([[2, 0, 1, 2]])).tolist(), [[8, 1, 6, 11]])

    def test_gather_rejects_an_index_outside_the_dimension(self):
        for bad in (4, 7, -5):
            with self.subTest(index=bad):
                with self.assertRaises(RuntimeError) as ctx:
                    torch.gather(self.a, 1, torch.tensor([[0], [bad], [1]]))
                self.assertIn(f"index {bad} is out of bounds for dimension 1 with size 4", str(ctx.exception))

    def test_gather_rejects_an_index_larger_than_the_input(self):
        # five rows of indices for a three-row input used to read past the end of the storage
        with self.assertRaises(RuntimeError) as ctx:
            torch.gather(self.a, 1, torch.zeros(5, 2, dtype=torch.int64))
        self.assertIn("larger than the input", str(ctx.exception))

    def test_gather_on_an_empty_dimension(self):
        with self.assertRaises(RuntimeError) as ctx:
            torch.gather(torch.zeros(3, 0), 1, torch.zeros(3, 1, dtype=torch.int64))
        self.assertIn("with size 0", str(ctx.exception))
        self.assertEqual(tuple(torch.gather(torch.zeros(3, 0), 1, torch.zeros(3, 0, dtype=torch.int64)).shape), (3, 0))

    def test_gather_requires_integer_indices(self):
        with self.assertRaises(RuntimeError) as ctx:
            torch.gather(self.a, 1, torch.tensor([[0.0], [1.0], [2.0]]))
        self.assertIn("integer", str(ctx.exception))

    def test_gather_keeps_integers_and_complex_numbers_exact(self):
        big = torch.tensor([[2**53 + 1, 5], [7, -(2**53) - 1]])
        self.assertEqual(torch.gather(big, 1, torch.tensor([[0], [1]])).tolist(), [[2**53 + 1], [-(2**53) - 1]])
        z = torch.tensor([[1 + 2j, 3 - 4j]])
        self.assertEqual(torch.gather(z, 1, torch.tensor([[1, 0]])).tolist(), [[3 - 4j, 1 + 2j]])

    def test_scatter_values(self):
        out = torch.zeros(2, 4).scatter(1, torch.tensor([[1, 3], [0, 0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        self.assertEqual(out.tolist(), [[0, 5, 0, 6], [8, 0, 0, 0]])
        out = torch.zeros(2, 4).scatter_add(1, torch.tensor([[1, 1], [0, 0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        self.assertEqual(out.tolist(), [[0, 11, 0, 0], [15, 0, 0, 0]])
        out = torch.zeros(2, 3).scatter(1, torch.tensor([[2], [0]]), 9.0)
        self.assertEqual(out.tolist(), [[0, 0, 9], [9, 0, 0]])

    def test_scatter_rejects_bad_indices_and_leaves_the_destination_alone(self):
        dst = torch.zeros(3, 4)
        index = torch.tensor([[0, 1], [2, 9]])
        for call in (dst.scatter_, dst.scatter_add_):
            with self.assertRaises(RuntimeError) as ctx:
                call(1, index, torch.ones(2, 2))
            self.assertIn("index 9 is out of bounds for dimension 1 with size 4", str(ctx.exception))
            self.assertEqual(dst.tolist(), [[0.0] * 4] * 3)

    def test_scatter_rejects_an_index_larger_than_its_source(self):
        with self.assertRaises(RuntimeError) as ctx:
            torch.zeros(3, 4).scatter(1, torch.zeros(3, 3, dtype=torch.int64), torch.ones(3, 2))
        self.assertIn("does not fit", str(ctx.exception))

    def test_one_hot_rejects_a_class_outside_the_range(self):
        self.assertEqual(F.one_hot(torch.tensor([0, 2]), 3).tolist(), [[1, 0, 0], [0, 0, 1]])
        self.assertEqual(F.one_hot(torch.tensor([[1], [0]])).tolist(), [[[0, 1]], [[1, 0]]])
        with self.assertRaises(RuntimeError) as ctx:
            F.one_hot(torch.tensor([0, 5]), 3)
        self.assertIn("class value 5 is out of range for num_classes=3", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            F.one_hot(torch.tensor([0, -1]), 3)      # would otherwise mark the last class
        self.assertIn("non-negative", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            F.one_hot(torch.tensor([0.0, 1.0]), 3)


class TestIndexSelectAndEmbeddingBounds(unittest.TestCase):
    def test_index_select(self):
        a = grid(3, 4)
        self.assertEqual(torch.index_select(a, 1, torch.tensor([0, 3, -1])).tolist(), [[0, 3, 3], [4, 7, 7], [8, 11, 11]])
        self.assertEqual(torch.index_select(a, 0, torch.tensor([2])).tolist(), [[8, 9, 10, 11]])
        for bad in (4, -5):
            with self.assertRaises(IndexError) as ctx:
                torch.index_select(a, 1, torch.tensor([0, bad]))
            self.assertIn(f"index {bad} is out of bounds for dimension 1 with size 4", str(ctx.exception))
        with self.assertRaises(IndexError):
            torch.index_select(torch.zeros(3, 0), 1, torch.tensor([0]))
        # an empty result in another dimension still has its indices checked
        with self.assertRaises(IndexError):
            torch.index_select(torch.zeros(0, 4), 1, torch.tensor([4]))

    def test_embedding_rejects_rows_that_do_not_exist(self):
        table = grid(4, 2)
        self.assertEqual(F.embedding(torch.tensor([[3, 0]]), table).tolist(), [[[6, 7], [0, 1]]])
        for kwargs in ({}, {"padding_idx": 0}):      # the ggml path and the general path
            for bad in (4, 99, -1):
                with self.subTest(index=bad, **kwargs):
                    with self.assertRaises(IndexError) as ctx:
                        F.embedding(torch.tensor([0, bad]), table, **kwargs)
                    self.assertIn(f"embedding index {bad} is out of range for a table of 4 rows", str(ctx.exception))
        layer = torch.nn.Embedding(4, 2)
        with self.assertRaises(IndexError):
            layer(torch.tensor([4]))


class TestAdvancedIndexing(unittest.TestCase):
    def setUp(self):
        self.a = grid(3, 4)
        self.c = grid(2, 3, 4, 5)

    def test_out_of_range_indices_raise_index_error(self):
        cases = {
            "tensor": lambda: self.a[torch.tensor([0, 5])],
            "negative tensor": lambda: self.a[torch.tensor([-4])],
            "list": lambda: self.a[[0, 5]],
            "second dimension": lambda: self.a[:, torch.tensor([0, 4])],
            "pair": lambda: self.a[torch.tensor([0, 2]), torch.tensor([1, 4])],
            "empty tensor": lambda: torch.zeros(0, 4)[torch.tensor([0])],
        }
        for name, fn in cases.items():
            with self.subTest(name):
                with self.assertRaises(IndexError) as ctx:
                    fn()
                self.assertIn("is out of bounds for dimension", str(ctx.exception))
        with self.assertRaises(IndexError) as ctx:
            self.a[torch.tensor([0, 5])]
        self.assertEqual(str(ctx.exception), "index 5 is out of bounds for dimension 0 with size 3")

    def test_integer_tensors_and_lists(self):
        a = self.a
        self.assertEqual(a[torch.tensor([-1, 0])].tolist(), [[8, 9, 10, 11], [0, 1, 2, 3]])
        self.assertEqual(a[[2, 0]].tolist(), [[8, 9, 10, 11], [0, 1, 2, 3]])
        self.assertEqual(a[:, torch.tensor([3, 0])].tolist(), [[3, 0], [7, 4], [11, 8]])
        self.assertEqual(a[torch.tensor([0, 2]), torch.tensor([1, 3])].tolist(), [1, 11])
        self.assertEqual(a[torch.tensor([[0], [2]]), torch.tensor([1, 3])].tolist(), [[1, 3], [9, 11]])
        self.assertEqual(a[1, torch.tensor([3, 0])].tolist(), [7, 4])
        self.assertEqual(a[torch.tensor([2, 0]), 1].tolist(), [9, 1])
        self.assertEqual(a[torch.tensor(1)].tolist(), [4, 5, 6, 7])
        self.assertEqual(tuple(a[torch.tensor([[0, 1], [2, 2]])].shape), (2, 2, 4))
        self.assertEqual(tuple(a[None, torch.tensor([0, 2])].shape), (1, 2, 4))
        self.assertEqual(tuple(a[torch.tensor([0, 2]), None].shape), (2, 1, 4))
        self.assertEqual(tuple(a[..., torch.tensor([0, 2])].shape), (3, 2))

    def test_index_tensors_separated_by_a_slice_come_first(self):
        c = self.c
        i, j = torch.tensor([0, 2]), torch.tensor([1, 4])
        got = c[:, i, :, j]                       # (index, dim 0, dim 2), as in numpy and torch
        self.assertEqual(tuple(got.shape), (2, 2, 4))
        for k in range(2):
            for d0 in range(2):
                self.assertEqual(got[k, d0].tolist(), c[d0, int(i[k].item()), :, int(j[k].item())].tolist())
        # next to each other: the index shape replaces the indexed dimensions
        got = c[:, i, torch.tensor([1, 3])]
        self.assertEqual(tuple(got.shape), (2, 2, 5))
        self.assertEqual(got[1, 0].tolist(), c[1, 0, 1].tolist())
        self.assertEqual(tuple(c[torch.tensor([0, 1]), 1, :, j].shape), (2, 4))

    def test_an_int_next_to_index_tensors_is_an_index_too(self):
        # numpy/torch: with an index tensor present, an int is a (0-d) index tensor as well. A
        # slice between the two separates them, and the indexed dimensions then come first.
        c = self.c                                # (2, 3, 4, 5)
        j = torch.tensor([1, 4])
        got = c[:, 1, :, j]
        self.assertEqual(tuple(got.shape), (2, 2, 4))
        for k in range(2):
            self.assertEqual(got[k].tolist(), c[:, 1, :, int(j[k].item())].tolist())
        k = torch.tensor([0, 2])
        got = c[1, :, k]
        self.assertEqual(tuple(got.shape), (2, 3, 5))
        self.assertEqual(got[1].tolist(), c[1, :, 2].tolist())
        got = c[-1, None, k]
        self.assertEqual(tuple(got.shape), (2, 1, 4, 5))
        self.assertEqual(got[1, 0].tolist(), c[1, 2].tolist())
        # next to each other, the int only removes its dimension
        self.assertEqual(tuple(c[:, 1, k].shape), (2, 2, 5))
        self.assertEqual(c[:, 1, k].tolist(), c[:, 1][:, k].tolist())
        self.assertEqual(tuple(c[1, k].shape), (2, 4, 5))
        self.assertEqual(tuple(c[j[:1], 1, 2].shape), (1, 5))
        with self.assertRaises(IndexError):
            c[:, 3, :, j]

    def test_assignment_with_an_int_separated_from_an_index_tensor(self):
        c = self.c.clone()
        j = torch.tensor([1, 4])
        value = torch.arange(16, dtype=c.dtype).reshape(2, 2, 4) + 1000   # (index, dim 0, dim 2)
        c[:, 1, :, j] = value
        self.assertEqual(c[:, 1, :, j].tolist(), value.tolist())
        self.assertEqual(c[0, 1, :, 4].tolist(), value[1, 0].tolist())
        untouched = self.c.clone()
        self.assertEqual(c[:, 0].tolist(), untouched[:, 0].tolist())
        self.assertEqual(c[:, 1, :, 0].tolist(), untouched[:, 1, :, 0].tolist())

    def test_gradient_with_an_int_separated_from_an_index_tensor(self):
        x = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5).requires_grad_(True)
        j = torch.tensor([1, 4, 1])
        x[:, 1, :, j].sum().backward()
        want = torch.zeros(2, 3, 4, 5)
        want[:, 1, :, 1] = 2.0
        want[:, 1, :, 4] = 1.0
        self.assertEqual(x.grad.tolist(), want.tolist())

    def test_boolean_masks(self):
        a = self.a
        self.assertEqual(a[a > 8].tolist(), [9, 10, 11])
        self.assertEqual(a[torch.tensor([True, False, True])].tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])
        self.assertEqual(a[[True, False, True]].tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])
        self.assertEqual(a[:, torch.tensor([True, False, False, True])].tolist(), [[0, 3], [4, 7], [8, 11]])
        self.assertEqual(tuple(a[torch.tensor([False, False, False])].shape), (0, 4))
        with self.assertRaises(IndexError) as ctx:
            a[torch.tensor([True, False])]
        self.assertIn("The shape of the mask [2] at index 0 does not match", str(ctx.exception))

    def test_invalid_index_types(self):
        a = self.a
        for bad in (torch.tensor([0.0, 1.0]), [0.5], "x"):
            with self.subTest(index=repr(bad)[:20]):
                with self.assertRaises(IndexError):
                    a[bad]
        with self.assertRaises(IndexError) as ctx:
            a[torch.tensor([0]), torch.tensor([0]), torch.tensor([0])]
        self.assertIn("too many indices", str(ctx.exception))
        with self.assertRaises(IndexError):
            a[torch.tensor([0, 1, 2]), torch.tensor([0, 1])]     # index shapes do not broadcast

    def test_masked_select_broadcasts_both_operands(self):
        a = self.a
        self.assertEqual(torch.masked_select(a, a > 8).tolist(), [9, 10, 11])
        self.assertEqual(torch.masked_select(a, torch.tensor([True, False, True, False])).tolist(), [0, 2, 4, 6, 8, 10])
        self.assertEqual(torch.masked_select(a, torch.tensor([[True], [False], [True]])).tolist(), [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(torch.masked_select(torch.tensor([1.0, 2.0, 3.0, 4.0]), a > 5).tolist(), [3, 4, 1, 2, 3, 4])
        self.assertEqual(torch.masked_select(torch.tensor(3.0), torch.tensor(True)).tolist(), [3.0])
        with self.assertRaises(RuntimeError) as ctx:
            torch.masked_select(a, torch.tensor([True, False, True]))
        self.assertIn("cannot be broadcast", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            torch.masked_select(a, torch.ones(3, 4))
        self.assertIn("boolean", str(ctx.exception))


class TestIndexedAssignment(unittest.TestCase):
    def put(self, shape, idx, value):
        z = torch.zeros(*shape)
        z[idx] = value
        return z.tolist()

    def test_out_of_range_index_raises_and_writes_nothing(self):
        z = torch.zeros(3, 4)
        with self.assertRaises(IndexError):
            z[torch.tensor([0, 5])] = 1.0
        with self.assertRaises(IndexError):
            z[torch.tensor([1]), torch.tensor([-5])] = 1.0
        self.assertEqual(z.tolist(), [[0.0] * 4] * 3)
        z[torch.tensor([0, -1])] = 1.0
        self.assertEqual(z.tolist(), [[1.0] * 4, [0.0] * 4, [1.0] * 4])

    def test_values_are_broadcast_to_the_indexed_shape(self):
        rows = torch.tensor([0, 2])
        self.assertEqual(self.put((3, 4), rows, 5.0), [[5] * 4, [0] * 4, [5] * 4])
        self.assertEqual(self.put((3, 4), rows, torch.tensor([1.0, 2.0, 3.0, 4.0])), [[1, 2, 3, 4], [0] * 4, [1, 2, 3, 4]])
        self.assertEqual(self.put((3, 4), rows, torch.tensor([[1.0], [2.0]])), [[1] * 4, [0] * 4, [2] * 4])
        self.assertEqual(self.put((3, 4), (slice(None), rows), torch.tensor([[1.0], [2.0], [3.0]])),
                         [[1, 0, 1, 0], [2, 0, 2, 0], [3, 0, 3, 0]])
        self.assertEqual(self.put((3, 4), (rows, torch.tensor([1, 3])), torch.tensor([7.0, 8.0])),
                         [[0, 7, 0, 0], [0] * 4, [0, 0, 0, 8]])
        for value in (torch.tensor([1.0, 2.0, 3.0]), torch.ones(3, 4)):
            with self.assertRaises(RuntimeError) as ctx:
                self.put((3, 4), rows, value)
            self.assertIn("shape mismatch", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            self.put((3, 4), slice(1, 3), torch.ones(3, 4))
        self.assertIn("shape mismatch", str(ctx.exception))

    def test_ints_and_slices_combine_with_index_tensors(self):
        cols = torch.tensor([0, 2])
        # these used to write every row, ignoring the int and the slice
        self.assertEqual(self.put((3, 4), (1, cols), torch.tensor([7.0, 8.0])), [[0] * 4, [7, 0, 8, 0], [0] * 4])
        self.assertEqual(self.put((3, 4), (cols, 1), torch.tensor([7.0, 8.0])), [[0, 7, 0, 0], [0] * 4, [0, 8, 0, 0]])
        self.assertEqual(self.put((3, 4), (slice(1, None), cols), 5.0), [[0] * 4, [5, 0, 5, 0], [5, 0, 5, 0]])
        self.assertEqual(self.put((3, 4), [0, 2], 5.0), [[5] * 4, [0] * 4, [5] * 4])

    def test_boolean_masks(self):
        full = grid(3, 4) % 5 == 0
        self.assertEqual(self.put((3, 4), full, 9.0), [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0]])
        self.assertEqual(self.put((3, 4), full, torch.tensor([1.0, 2.0, 3.0])), [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]])
        row = torch.tensor([True, False, False])
        # a row mask on a square tensor used to be applied to the columns
        self.assertEqual(self.put((3, 3), row, 5.0), [[5, 5, 5], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(self.put((3, 4), row, 5.0), [[5] * 4, [0] * 4, [0] * 4])
        self.assertEqual(self.put((3, 4), row, torch.tensor([1.0, 2.0, 3.0, 4.0])), [[1, 2, 3, 4], [0] * 4, [0] * 4])
        with self.assertRaises(IndexError):
            self.put((3, 4), torch.tensor([True, False]), 1.0)
        with self.assertRaises(RuntimeError):
            self.put((3, 4), full, torch.tensor([1.0, 2.0]))

    def test_assignment_into_a_leaf_that_requires_grad(self):
        p = torch.zeros(3, requires_grad=True)
        for idx in (0, slice(0, 2), torch.tensor([1]), torch.tensor([True, False, False])):
            with self.subTest(index=type(idx).__name__):
                with self.assertRaises(RuntimeError) as ctx:
                    p[idx] = 1.0
                self.assertIn("in-place", str(ctx.exception))
        self.assertEqual(p.tolist(), [0.0, 0.0, 0.0])
        with torch.no_grad():
            p[0] = 1.0
            p[torch.tensor([2])] = 3.0
        self.assertEqual(p.tolist(), [1.0, 0.0, 3.0])

    def test_assignment_moves_the_version_counter(self):
        t = torch.zeros(3)
        before = t._version
        t[0] = 1.0
        t[torch.tensor([1])] = 2.0
        t[torch.tensor([False, False, True])] = 3.0
        self.assertEqual(t._version, before + 3)
        self.assertEqual(t.tolist(), [1.0, 2.0, 3.0])


class TestIndexingGradients(unittest.TestCase):
    def grad(self, fn, shape=(2, 3)):
        x = grid(*shape).requires_grad_(True)
        y = fn(x)
        self.assertTrue(y.requires_grad, "the result is detached from the graph")
        y.sum().backward()
        return x.grad.tolist()

    def test_integer_indexing(self):
        self.assertEqual(self.grad(lambda x: x[torch.tensor([1, 0, 1])]), [[1, 1, 1], [2, 2, 2]])
        self.assertEqual(self.grad(lambda x: x[[1, 0, 1]]), [[1, 1, 1], [2, 2, 2]])
        self.assertEqual(self.grad(lambda x: x[torch.tensor([0, 1, 0]), torch.tensor([2, 0, 2])]), [[0, 0, 2], [1, 0, 0]])
        self.assertEqual(self.grad(lambda x: x[:, torch.tensor([2, 2])] * torch.tensor([1.0, 10.0])), [[0, 0, 11], [0, 0, 11]])
        self.assertEqual(self.grad(lambda x: x[1, torch.tensor([0, 2])] * 3.0), [[0, 0, 0], [3, 0, 3]])

    def test_boolean_masks(self):
        self.assertEqual(self.grad(lambda x: x[x > 2] * 2), [[0, 0, 0], [2, 2, 2]])
        self.assertEqual(self.grad(lambda x: x[torch.tensor([False, True])]), [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(self.grad(lambda x: x.masked_select(x > 3)), [[0, 0, 0], [0, 1, 1]])
        # the tensor is broadcast against the mask: each element is selected once per row
        self.assertEqual(self.grad(lambda x: x.masked_select(torch.tensor([[True, False, True]] * 4)), (1, 3)), [[4, 0, 4]])
        self.assertEqual(self.grad(lambda x: torch.masked_select(x, torch.tensor([True, False, True]))), [[1, 0, 1], [1, 0, 1]])

    def test_second_derivative_through_an_index(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x[torch.tensor([0, 2, 2])] ** 3).sum()
        (first,) = torch.autograd.grad(y, x, create_graph=True)
        self.assertEqual(first.tolist(), [3.0, 0.0, 54.0])
        first.sum().backward()
        self.assertEqual(x.grad.tolist(), [6.0, 0.0, 36.0])

    def test_training_through_an_index_moves_the_selected_rows_only(self):
        table = torch.zeros(4, 2, requires_grad=True)
        opt = torch.optim.SGD([table], lr=0.5)
        loss = ((table[torch.tensor([1, 3])] - 1.0) ** 2).sum()
        loss.backward()
        opt.step()
        self.assertEqual(table.tolist(), [[0, 0], [1, 1], [0, 0], [1, 1]])

    def test_diagonal_trace_and_diag(self):
        self.assertEqual(self.grad(lambda x: x.trace(), (3, 3)), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(self.grad(lambda x: x.diagonal(1) * torch.tensor([1.0, 2.0]), (3, 3)), [[0, 1, 0], [0, 0, 2], [0, 0, 0]])
        self.assertEqual(self.grad(lambda x: x.diag(-1), (3, 3)), [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.assertEqual(grid(2, 3, 3).diagonal(dim1=1, dim2=2).tolist(), [[0, 4, 8], [9, 13, 17]])
        self.assertEqual(self.grad(lambda x: x.reshape(2, 2, 2).diagonal(dim1=1, dim2=2), (2, 4)), [[1, 0, 0, 1], [1, 0, 0, 1]])
        self.assertEqual(torch.tensor([1.0, 2.0]).diag(1).tolist(), [[0, 1, 0], [0, 0, 2], [0, 0, 0]])
        v = torch.tensor([1.0, 2.0], requires_grad=True)
        (v.diag() * torch.tensor([[1.0, 5.0], [5.0, 3.0]])).sum().backward()
        self.assertEqual(v.grad.tolist(), [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
