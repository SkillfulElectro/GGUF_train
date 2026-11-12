"""GGUF files must give back the tensors and metadata that were put in.

GGUF itself stores at most four dimensions, drops outer dimensions of size one, and has
no uint8, bool or complex element type; metadata used to be read through a 4096-byte
buffer with arrays replaced by a placeholder. Each test saves something GGUF cannot hold
directly, or reads something another writer produced, and checks that nothing changed.
"""

import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import _C
from torch.gguf import inspect_gguf, load_gguf, quantize_gguf, save_gguf
from torch.llama import LlamaModel

import _gguf_raw as raw
import _tiny_llama as tiny

HAVE_BRIDGE = (_C._LIBS_DIR / _C._lib_name("llama_bridge")).is_file()


class GGUFTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        # ggml reports unreadable files on stderr, and llama.cpp logs every model load.
        self.addCleanup(tiny.silence_native_stderr())

    def roundtrip(self, tensor, name="t"):
        """Save ``tensor`` next to an ordinary one and return what is read back."""
        path = self.dir / "roundtrip.gguf"
        save_gguf(path, {name: tensor, "neighbour": torch.arange(5, dtype=torch.float32)})
        loaded = load_gguf(path)
        # A tensor GGUF cannot describe must not damage the rest of the file.
        self.assertEqual(loaded["neighbour"].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])
        return loaded[name]

    def assert_same(self, tensor, loaded):
        self.assertEqual(loaded.dtype, tensor.dtype)
        self.assertEqual(tuple(loaded.shape), tuple(tensor.shape))
        self.assertEqual(loaded.tolist(), tensor.tolist())


class TestTensorRoundTrip(GGUFTestCase):
    def test_uint8_keeps_its_dtype_and_values_above_127(self):
        t = torch.tensor([0, 1, 127, 128, 200, 255], dtype=torch.uint8)
        self.assert_same(t, self.roundtrip(t))

    def test_bool_keeps_its_dtype(self):
        t = torch.tensor([[True, False, True], [False, False, True]])
        self.assert_same(t, self.roundtrip(t))

    def test_complex_keeps_both_parts(self):
        c64 = torch.tensor([[1 + 2j, 3 - 4j, 5 + 6j], [7 + 8j, 9 - 1j, 2 + 2j]])
        self.assertEqual(c64.dtype, torch.complex64)
        self.assert_same(c64, self.roundtrip(c64))
        c128 = c64.to(torch.complex128)
        self.assert_same(c128, self.roundtrip(c128))
        scalar = torch.tensor(1.5 - 2.5j)
        self.assert_same(scalar, self.roundtrip(scalar))

    def test_complex_views_and_many_dimensions(self):
        c = torch.tensor([[1 + 2j, 3 - 4j, 5 + 6j], [7 + 8j, 9 - 1j, 2 + 2j]])
        for view in (c[:, 1:], c.t(), c.reshape(1, 6)):
            self.assert_same(view, self.roundtrip(view))
        # Four dimensions plus the (real, imaginary) pair is more than GGUF stores.
        big = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5).to(torch.complex64) * (1 + 1j)
        self.assert_same(big, self.roundtrip(big))

    def test_a_float_tensor_of_two_values_is_not_taken_for_a_complex_number(self):
        t = torch.tensor([1.0, 2.0])
        self.assert_same(t, self.roundtrip(t))

    def test_scalars_stay_zero_dimensional(self):
        for t in (torch.tensor(3.5), torch.tensor(7, dtype=torch.int64), torch.tensor(True)):
            loaded = self.roundtrip(t)
            self.assertEqual(tuple(loaded.shape), ())
            self.assert_same(t, loaded)
        one = torch.ones(1)
        self.assertEqual(tuple(self.roundtrip(one).shape), (1,))

    def test_more_than_four_dimensions(self):
        t5 = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
        self.assert_same(t5, self.roundtrip(t5))
        t6 = torch.arange(64, dtype=torch.int32).reshape(2, 2, 2, 2, 2, 2)
        self.assert_same(t6, self.roundtrip(t6))

    def test_dimensions_of_size_one_survive(self):
        for shape in ((1, 4), (1, 1, 4), (4, 1), (1, 4, 1), (1, 1, 1)):
            t = torch.arange(4 if 4 in shape else 1, dtype=torch.float32).reshape(*shape)
            self.assert_same(t, self.roundtrip(t))

    def test_empty_tensors(self):
        # (0, 4) used to produce a file that could not be opened again at all.
        for shape in ((0,), (0, 4), (4, 0), (3, 0, 2)):
            t = torch.zeros(*shape)
            loaded = self.roundtrip(t)
            self.assertEqual(tuple(loaded.shape), shape)
            self.assertEqual(loaded.dtype, torch.float32)
        empty_bool = torch.zeros(0, 3).to(torch.bool)
        self.assert_same(empty_bool, self.roundtrip(empty_bool))

    def test_views_are_saved_from_their_own_elements(self):
        base = torch.arange(12, dtype=torch.uint8).reshape(3, 4)
        for view in (base[1:], base[:, 1:3], base.t()):
            self.assert_same(view, self.roundtrip(view))

    def test_quantized_tensors_keep_their_shape(self):
        torch.manual_seed(0)
        # A leading dimension of one, five dimensions, and rows that are not whole blocks.
        for shape in ((1, 64), (2, 2, 2, 2, 32), (64, 16)):
            q = torch.randn(*shape).quantize("q4_0")
            loaded = self.roundtrip(q)
            self.assertEqual(loaded.dtype, q.dtype)
            self.assertEqual(tuple(loaded.shape), shape)
            self.assertEqual(loaded.dequantize().tolist(), q.dequantize().tolist())

    def test_state_dict_with_a_zero_dimensional_buffer(self):
        bn = torch.nn.BatchNorm1d(4)
        bn(torch.randn(8, 4))
        path = str(self.dir / "bn.gguf")
        torch.save(bn.state_dict(), path)
        loaded = torch.load(path)
        self.assertEqual(tuple(loaded["num_batches_tracked"].shape), ())
        self.assertEqual(loaded["num_batches_tracked"].dtype, torch.int64)
        fresh = torch.nn.BatchNorm1d(4)
        fresh.load_state_dict(loaded)
        self.assertEqual(fresh.num_batches_tracked.item(), 1)
        self.assertEqual(fresh.running_mean.tolist(), bn.running_mean.tolist())


class TestStoredForm(GGUFTestCase):
    """What ends up in the file, read without the package's own reader."""

    def test_ordinary_tensors_add_no_metadata(self):
        path = self.dir / "plain.gguf"
        save_gguf(path, {"w": torch.zeros(3, 4), "b": torch.zeros(4), "i": torch.arange(6, dtype=torch.int64)})
        metadata, tensors = raw.read_gguf(path)
        self.assertEqual(metadata, {})
        self.assertEqual(tensors["w"].dims, [4, 3])
        self.assertEqual(tensors["w"].type, raw.F32)
        self.assertEqual(tensors["i"].type, raw.I64)

    def test_shape_and_dtype_are_recorded_only_where_needed(self):
        path = self.dir / "special.gguf"
        save_gguf(
            path,
            {
                "scalar": torch.tensor(2.0),
                "mask": torch.tensor([True, False]),
                "bytes": torch.zeros(1, 3).to(torch.uint8),
                "z": torch.tensor([1 + 2j, 3 + 4j, 5 + 6j]),
                "five": torch.zeros(2, 3, 4, 5, 6),
            },
        )
        metadata, tensors = raw.read_gguf(path)
        self.assertEqual(
            sorted(metadata),
            [
                "gguf_train.dtype.bytes",
                "gguf_train.dtype.mask",
                "gguf_train.dtype.z",
                "gguf_train.shape.bytes",
                "gguf_train.shape.five",
                "gguf_train.shape.scalar",
            ],
        )
        self.assertEqual(metadata["gguf_train.shape.scalar"], raw.KV(raw.ARRAY, raw.INT64, []))
        self.assertEqual(metadata["gguf_train.shape.five"], raw.KV(raw.ARRAY, raw.INT64, [2, 3, 4, 5, 6]))
        self.assertEqual(metadata["gguf_train.dtype.mask"], raw.KV(raw.STRING, None, "bool"))
        self.assertEqual(metadata["gguf_train.dtype.bytes"].value, "uint8")
        self.assertEqual(metadata["gguf_train.dtype.z"].value, "complex64")
        self.assertEqual(tensors["mask"].type, raw.I8)
        # Complex values are pairs of floats: the innermost stored dimension is the pair.
        self.assertEqual((tensors["z"].dims, tensors["z"].type), ([2, 3], raw.F32))
        # Dimensions beyond the fourth are folded into the outermost stored one.
        self.assertEqual(tensors["five"].dims, [6, 5, 4, 6])

    def test_the_recorded_keys_are_not_user_metadata(self):
        path = self.dir / "hidden.gguf"
        save_gguf(path, {"scalar": torch.tensor(2.0), "mask": torch.tensor([True])}, metadata={"note": "kept"})
        self.assertEqual(load_gguf(path).metadata, {"note": "kept"})
        self.assertEqual(inspect_gguf(path)["metadata"], {"note": "kept"})
        for key in ("gguf_train.shape.scalar", "gguf_train.dtype.mask"):
            with self.assertRaisesRegex(RuntimeError, "reserved"):
                save_gguf(self.dir / "bad.gguf", {"t": torch.zeros(2)}, metadata={key: "x"})

    def test_files_from_other_writers_load_as_before(self):
        path = raw.write_gguf(
            self.dir / "foreign.gguf",
            [raw.kv("general.name", raw.STRING, "foreign")],
            [
                ("m", [3, 2], raw.F32, struct.pack("<6f", *range(6))),
                ("b", [4], raw.I8, struct.pack("<4b", -1, 0, 1, 127)),
            ],
        )
        loaded = load_gguf(path)
        self.assertEqual(loaded["m"].tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(loaded["b"].dtype, torch.int8)
        self.assertEqual(loaded["b"].tolist(), [-1, 0, 1, 127])

    def test_recorded_dtype_is_applied_to_any_file(self):
        path = raw.write_gguf(
            self.dir / "typed.gguf",
            [raw.kv("gguf_train.dtype.b", raw.STRING, "uint8")],
            [("b", [3], raw.I8, bytes([1, 200, 255]))],
        )
        loaded = load_gguf(path)["b"]
        self.assertEqual(loaded.dtype, torch.uint8)
        self.assertEqual(loaded.tolist(), [1, 200, 255])

    def test_inconsistent_records_are_an_error_not_a_wrong_tensor(self):
        floats = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        cases = {
            "shape": [raw.kv("gguf_train.shape.t", raw.ARRAY, [3, 3], raw.INT64)],
            "negative": [raw.kv("gguf_train.shape.t", raw.ARRAY, [-2, -2], raw.INT64)],
            "shape type": [raw.kv("gguf_train.shape.t", raw.STRING, "2,2")],
            "carrier": [raw.kv("gguf_train.dtype.t", raw.STRING, "bool")],
            "unknown dtype": [raw.kv("gguf_train.dtype.t", raw.STRING, "float128")],
            "dtype type": [raw.kv("gguf_train.dtype.t", raw.UINT32, 9)],
        }
        for label, kvs in cases.items():
            with self.subTest(label):
                path = raw.write_gguf(self.dir / "inconsistent.gguf", kvs, [("t", [4], raw.F32, floats)])
                with self.assertRaisesRegex(RuntimeError, "tensor 't'"):
                    load_gguf(path)
        # An odd number of floats cannot be complex numbers.
        path = raw.write_gguf(
            self.dir / "odd.gguf",
            [raw.kv("gguf_train.dtype.t", raw.STRING, "complex64")],
            [("t", [3], raw.F32, floats[:12])],
        )
        with self.assertRaisesRegex(RuntimeError, "even number"):
            load_gguf(path)

    def test_tensor_names_gguf_cannot_hold_are_refused(self):
        # GGUF tensor names are at most 63 bytes; longer ones used to be cut short silently.
        save_gguf(self.dir / "ok.gguf", {"n" * 63: torch.zeros(2)})
        self.assertEqual(list(load_gguf(self.dir / "ok.gguf").keys()), ["n" * 63])
        with self.assertRaisesRegex(RuntimeError, "64 bytes long"):
            save_gguf(self.dir / "long.gguf", {"n" * 64: torch.zeros(2)})
        with self.assertRaises(TypeError):
            save_gguf(self.dir / "bad.gguf", {"a": [1.0, 2.0]})

    def test_unwritable_path_reports_the_path(self):
        target = self.dir / "missing-directory" / "x.gguf"
        with self.assertRaisesRegex(RuntimeError, "could not write"):
            save_gguf(target, {"t": torch.zeros(2)})


class TestMetadata(GGUFTestCase):
    def save_and_load(self, metadata):
        path = self.dir / "meta.gguf"
        save_gguf(path, {"t": torch.zeros(2)}, metadata=metadata)
        return load_gguf(path).metadata

    def test_strings_longer_than_4096_bytes(self):
        template = "{% for message in messages %}" + "x" * 20000 + "{% endfor %}"
        loaded = self.save_and_load({"tokenizer.chat_template": template, "short": "s", "unicode": "h\u00e9llo \u4e16\u754c"})
        self.assertEqual(loaded["tokenizer.chat_template"], template)
        self.assertEqual(loaded["short"], "s")
        self.assertEqual(loaded["unicode"], "h\u00e9llo \u4e16\u754c")

    def test_scalars_keep_their_value(self):
        values = {
            "pi": math.pi,
            "tenth": 0.1,
            "tiny": 5e-324,
            "huge": 1.7976931348623157e308,
            "minus_inf": float("-inf"),
            "int": 2**40,
            "int64_min": -(2**63),
            "uint64_max": 2**64 - 1,
            "yes": True,
            "no": False,
        }
        loaded = self.save_and_load(dict(values, nan=float("nan")))
        self.assertTrue(math.isnan(loaded.pop("nan")))
        self.assertEqual(loaded, values)
        for key in ("yes", "no"):
            self.assertIs(type(loaded[key]), bool)
        self.assertIs(type(loaded["int"]), int)

    def test_integers_gguf_cannot_hold_are_refused(self):
        for value in (2**64, -(2**63) - 1, [-1, 2**63]):
            with self.assertRaises(ValueError):
                self.save_and_load({"n": value})

    def test_arrays(self):
        values = {
            "ints": [1, -2, 3],
            "big": [2**63, 1],
            "floats": [0.5, 0.1, -2.25],
            "strings": ["a", "b\u00e9", ""],
            "bools": [True, False, True],
            "empty": [],
        }
        loaded = self.save_and_load(dict(values, pair=(4, 5), mixed_numbers=[1, 0.5]))
        self.assertEqual(loaded.pop("pair"), [4, 5])
        self.assertEqual(loaded.pop("mixed_numbers"), [1.0, 0.5])
        self.assertEqual(loaded, values)
        self.assertIs(type(loaded["bools"][0]), bool)

    def test_lists_of_mixed_kinds_are_refused(self):
        for value in ([1, "a"], [True, 2], [None], [[1, 2], [3, 4]]):
            with self.assertRaises(TypeError):
                self.save_and_load({"x": value})

    def test_nul_bytes_are_refused_instead_of_truncating(self):
        for metadata in ({"x": "a\0b"}, {"a\0b": 1}, {"x": ["ok", "a\0b"]}):
            with self.assertRaises(ValueError):
                self.save_and_load(metadata)

    def test_alignment_key_cannot_abort_the_process(self):
        # ggml aborts when general.alignment is not a u32, and Python ints are written as
        # int64. The default value is accepted and left out; anything else is an error.
        path = self.dir / "aligned.gguf"
        save_gguf(path, {"t": torch.zeros(2), "u": torch.ones(40)}, metadata={"general.alignment": 32, "k": 1})
        loaded = load_gguf(path)
        self.assertEqual(loaded.metadata, {"k": 1})
        self.assertEqual(loaded["u"].sum().item(), 40.0)
        for value in (64, 3, "32", 32.0, [32]):
            with self.assertRaisesRegex(RuntimeError, "general.alignment"):
                save_gguf(path, {"t": torch.zeros(2)}, metadata={"general.alignment": value})

    def test_metadata_written_by_another_tool(self):
        path = raw.write_gguf(
            self.dir / "typed.gguf",
            [
                raw.kv("u8", raw.UINT8, 255),
                raw.kv("i8", raw.INT8, -128),
                raw.kv("u16", raw.UINT16, 65535),
                raw.kv("i16", raw.INT16, -32768),
                raw.kv("u32", raw.UINT32, 4294967295),
                raw.kv("i32", raw.INT32, -2147483648),
                raw.kv("u64", raw.UINT64, 2**64 - 1),
                raw.kv("i64", raw.INT64, -(2**63)),
                raw.kv("f32", raw.FLOAT32, 1e-5),
                raw.kv("f64", raw.FLOAT64, 0.1),
                raw.kv("flag", raw.BOOL, True),
                raw.kv("u8s", raw.ARRAY, [0, 200, 255], raw.UINT8),
                raw.kv("i16s", raw.ARRAY, [-3, 4], raw.INT16),
                raw.kv("f32s", raw.ARRAY, [0.5, -1.5], raw.FLOAT32),
                raw.kv("flags", raw.ARRAY, [True, False], raw.BOOL),
                raw.kv("words", raw.ARRAY, ["x" * 5000, "y"], raw.STRING),
                raw.kv("nothing", raw.ARRAY, [], raw.UINT32),
            ],
            [("t", [2], raw.F32, struct.pack("<2f", 1.0, 2.0))],
        )
        loaded = load_gguf(path).metadata
        self.assertEqual(
            loaded,
            {
                "u8": 255,
                "i8": -128,
                "u16": 65535,
                "i16": -32768,
                "u32": 4294967295,
                "i32": -2147483648,
                "u64": 2**64 - 1,
                "i64": -(2**63),
                # A float32 is read as exactly the value it holds, not six digits of it.
                "f32": struct.unpack("<f", struct.pack("<f", 1e-5))[0],
                "f64": 0.1,
                "flag": True,
                "u8s": [0, 200, 255],
                "i16s": [-3, 4],
                "f32s": [0.5, -1.5],
                "flags": [True, False],
                "words": ["x" * 5000, "y"],
                "nothing": [],
            },
        )

    def test_metadata_survives_a_load_and_save(self):
        first = self.dir / "first.gguf"
        save_gguf(first, {"t": torch.zeros(2)}, metadata={"labels": ["cat", "dog"], "lr": 0.001, "steps": [100, 200]})
        tensors, metadata = load_gguf(first, return_metadata=True)
        second = self.dir / "second.gguf"
        save_gguf(second, tensors, metadata=metadata)
        self.assertEqual(load_gguf(second).metadata, {"labels": ["cat", "dog"], "lr": 0.001, "steps": [100, 200]})

    def test_copy_metadata_from_keeps_exact_types(self):
        source = raw.write_gguf(
            self.dir / "source.gguf",
            [
                raw.kv("count", raw.UINT32, 7),
                raw.kv("scale", raw.FLOAT32, 0.25),
                raw.kv("ids", raw.ARRAY, [1, 2, 3], raw.INT32),
                raw.kv("general.alignment", raw.UINT32, 32),
                raw.kv("gguf_train.dtype.old", raw.STRING, "uint8"),
            ],
            [("old", [2], raw.I8, bytes([1, 2]))],
        )
        target = self.dir / "target.gguf"
        save_gguf(target, {"mask": torch.tensor([True, False])}, metadata={"count": 9, "extra": "e"}, copy_metadata_from=source)
        metadata, _ = raw.read_gguf(target)
        self.assertEqual(metadata["scale"], raw.KV(raw.FLOAT32, None, 0.25))
        self.assertEqual(metadata["ids"], raw.KV(raw.ARRAY, raw.INT32, [1, 2, 3]))
        # Explicit metadata is applied on top of the copy.
        self.assertEqual(metadata["count"], raw.KV(raw.INT64, None, 9))
        self.assertEqual(metadata["extra"].value, "e")
        # What described the source's tensors and layout is not carried over.
        self.assertNotIn("gguf_train.dtype.old", metadata)
        self.assertNotIn("general.alignment", metadata)
        self.assertEqual(metadata["gguf_train.dtype.mask"].value, "bool")
        self.assertEqual(load_gguf(target)["mask"].tolist(), [True, False])
        with self.assertRaisesRegex(RuntimeError, "could not read metadata"):
            save_gguf(target, {"t": torch.zeros(2)}, copy_metadata_from=self.dir / "absent.gguf")


class TestModelFiles(GGUFTestCase):
    """A llama.cpp model file: typed metadata, large arrays, a long chat template."""

    TEMPLATE = "{% for message in messages %}" + "{{ message['content'] }}" * 400 + "{% endfor %}"

    def setUp(self):
        super().setUp()
        self.model_path = tiny.write_tiny_llama(self.dir / "tiny.gguf", chat_template=self.TEMPLATE)

    def test_metadata_is_read_in_full(self):
        self.assertGreater(len(self.TEMPLATE), 4096)
        metadata = load_gguf(self.model_path).metadata
        self.assertEqual(metadata["tokenizer.chat_template"], self.TEMPLATE)
        self.assertEqual(metadata["llama.context_length"], 1024)
        tokens = metadata["tokenizer.ggml.tokens"]
        self.assertEqual(len(tokens), 320)
        self.assertEqual(tokens[:4], ["<unk>", "<s>", "</s>", "<0x00>"])
        self.assertEqual(tokens[tiny.FIRST_WORD_ID], "\u2581w0")
        self.assertEqual(metadata["tokenizer.ggml.scores"][tiny.FIRST_WORD_ID + 3], -3.0)
        self.assertEqual(metadata["tokenizer.ggml.token_type"][:3], [2, 3, 3])
        self.assertIs(metadata["tokenizer.ggml.add_bos_token"], True)

    def test_quantize_gguf_copies_the_metadata_exactly(self):
        target = self.dir / "quantized.gguf"
        quantize_gguf(self.model_path, target, qtype="q8_0")
        before, tensors_before = raw.read_gguf(self.model_path)
        after, tensors_after = raw.read_gguf(target)
        # Same keys, same GGUF types (llama.cpp rejects a uint32 that became an int64), same values.
        self.assertEqual(after, before)
        self.assertEqual(list(tensors_after), list(tensors_before))
        self.assertEqual(tensors_after["token_embd.weight"].type, raw.Q8_0)
        self.assertEqual(tensors_after["token_embd.weight"].dims, tensors_before["token_embd.weight"].dims)
        self.assertEqual(tensors_after["output_norm.weight"].type, raw.F32)

    @unittest.skipUnless(HAVE_BRIDGE, "the llama.cpp bridge has not been built")
    def test_llama_cpp_loads_the_quantized_copy(self):
        target = self.dir / "quantized.gguf"
        quantize_gguf(self.model_path, target, qtype="q8_0")
        tokens = tiny.sample_rows(1, 9)[0]
        logits = []
        for path in (self.model_path, target):
            model = LlamaModel(str(path))
            context = model.new_context(n_ctx=256, n_batch=64, n_threads=2)
            logits.append(context.forward_logits(tokens, only_last=True))
        self.assertEqual(tuple(logits[0].shape), tuple(logits[1].shape))
        self.assertLess((logits[0] - logits[1]).abs().max().item(), 0.05)

    @unittest.skipUnless(HAVE_BRIDGE, "the llama.cpp bridge has not been built")
    def test_llama_model_metadata_is_not_truncated(self):
        metadata = LlamaModel(str(self.model_path)).metadata
        self.assertEqual(metadata["tokenizer.chat_template"], self.TEMPLATE)
        self.assertEqual(metadata["general.architecture"], "llama")
        self.assertEqual(metadata["llama.context_length"], "1024")


if __name__ == "__main__":
    unittest.main()
