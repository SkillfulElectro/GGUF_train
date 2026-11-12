"""End-to-end tests of the llama.cpp bridge on a tiny generated model.

Everything here used to take the whole Python process down (GGML_ASSERT aborts, a
segfault on read-only weights, an uncaught C++ exception) or to train the model on
padding; the tests run the real native code paths to pin the fixed behaviour.
"""

import ctypes
import math
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import _C
from torch.gguf import load_gguf
from torch.llama import LlamaModel

import _tiny_llama as tiny

HAVE_BRIDGE = (_C._LIBS_DIR / _C._lib_name("llama_bridge")).is_file()
silence_native_stderr = tiny.silence_native_stderr


def max_abs_diff(a, b):
    return max(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


@unittest.skipUnless(HAVE_BRIDGE, "the llama.cpp bridge has not been built")
class LlamaBridgeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = Path(tempfile.mkdtemp(prefix="gguf_train_llama_test_"))
        cls.model_path = tiny.write_tiny_llama(cls._dir / "tiny.gguf")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def setUp(self):
        self.addCleanup(silence_native_stderr())

    def load(self, path=None, **kwargs):
        kwargs.setdefault("use_mmap", False)
        return LlamaModel(path or self.model_path, **kwargs)

    def context(self, model, n_ctx=256, n_batch=256, **kwargs):
        return model.new_context(n_ctx=n_ctx, n_batch=n_batch, n_threads=2, **kwargs)

    def logits(self, ctx, tokens, **kwargs):
        ctx.clear_kv()
        return ctx.forward_logits(tokens, **kwargs)

    def loss(self, ctx, rows):
        """Mean next-token cross-entropy, every row evaluated as an independent sequence."""
        total, count = 0.0, 0
        for row in rows:
            log_probs = torch.log_softmax(self.logits(ctx, row[:-1]), dim=-1).tolist()
            for k in range(len(row) - 1):
                total -= log_probs[k][row[k + 1]]
                count += 1
        return total / count


class TestModelAndInference(LlamaBridgeTestCase):
    def test_model_properties(self):
        model = self.load()
        self.assertEqual(model.n_vocab, 320)
        self.assertEqual(model.n_ctx_train, 1024)
        self.assertEqual(model.n_embd, 32)
        self.assertEqual(model.n_layer, 2)
        self.assertEqual((model.bos_token_id, model.eos_token_id), (tiny.BOS_ID, tiny.EOS_ID))
        self.assertEqual(model.metadata["general.architecture"], "llama")

    def test_missing_model_file_raises(self):
        with self.assertRaisesRegex(RuntimeError, "does-not-exist.gguf"):
            self.load(self._dir / "does-not-exist.gguf")

    def test_tokenize_round_trip(self):
        model = self.load()
        tokens = model.tokenize("hello world", add_special=True)
        self.assertEqual(tokens[0], tiny.BOS_ID)
        self.assertTrue(all(0 <= t < model.n_vocab for t in tokens))
        # The tiny vocabulary spells everything with byte tokens, including the
        # SentencePiece word-boundary marker.
        self.assertEqual(model.detokenize(tokens[1:]).replace("\u2581", " ").strip(), "hello world")
        self.assertEqual(model.detokenize([]), "")

    def test_forward_logits(self):
        model = self.load()
        ctx = self.context(model)
        row = tiny.sample_rows(1, 9)[0]

        full = self.logits(ctx, row)
        self.assertEqual(tuple(full.shape), (9, 320))
        self.assertTrue(all(math.isfinite(v) for v in full.tolist()[0]))

        last = self.logits(ctx, row, only_last=True)
        self.assertEqual(tuple(last.shape), (1, 320))
        self.assertLess(max_abs_diff([full.tolist()[-1]], last.tolist()), 1e-5)

        # Feeding the sequence in two pieces continues from the cached prefix.
        ctx.clear_kv()
        ctx.forward_logits(row[:5])
        tail = ctx.forward_logits(row[5:], pos_offset=5)
        self.assertLess(max_abs_diff(full.tolist()[5:], tail.tolist()), 1e-4)

    def test_forward_logits_reports_errors_instead_of_crashing(self):
        model = self.load()
        ctx = self.context(model, n_ctx=256, n_batch=16)
        row = tiny.sample_rows(1, 32)[0]
        cases = [
            ("n_batch", lambda: ctx.forward_logits(row[:17])),
            ("context window", lambda: ctx.forward_logits(row[:8], pos_offset=250)),
            ("context window", lambda: ctx.forward_logits(row[:8], pos_offset=-1)),
            ("llama_decode failed", lambda: ctx.forward_logits([tiny.BOS_ID, 100000])),
            ("empty token list", lambda: ctx.forward_logits([])),
        ]
        for message, call in cases:
            with self.subTest(message=message):
                ctx.clear_kv()
                with self.assertRaisesRegex(RuntimeError, message):
                    call()
        # The context survives all of the above.
        self.assertEqual(tuple(self.logits(ctx, row[:8]).shape), (8, 320))

    def test_forward_embeddings(self):
        model = self.load()
        ctx = self.context(model, embeddings=True)
        row = tiny.sample_rows(1, 6)[0]
        emb = ctx.forward_embeddings(row)
        self.assertEqual(tuple(emb.shape), (6, 32))
        self.assertTrue(any(v != 0.0 for v in emb.tolist()[0]))

    def test_missing_embeddings_are_an_error_not_zeros(self):
        # A context created without embeddings=True has no embeddings to return; this used
        # to hand back a tensor of zeros as if it were the result.
        model = self.load()
        ctx = self.context(model)
        with self.assertRaisesRegex(RuntimeError, "no embedding"):
            ctx.forward_embeddings(tiny.sample_rows(1, 6)[0])


class TestFinetune(LlamaBridgeTestCase):
    def test_finetune_learns_and_can_be_repeated(self):
        model = self.load()
        ctx = self.context(model)
        rows = tiny.sample_rows(4, 17)
        before = self.loss(ctx, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.finetune(rows, epochs=3, lr=1e-2)
            first = self.loss(ctx, rows)
            # A second call on the same context used to abort the process
            # (llama_opt_init: GGML_ASSERT(!opt_ctx)).
            ctx.finetune(rows, epochs=3, lr=1e-2)
            second = self.loss(ctx, rows)
            ctx.finetune(rows, epochs=3, lr=1e-2, optimizer="sgd", weight_decay=0.01)
            third = self.loss(ctx, rows)
        self.assertLess(first, before - 0.5)
        self.assertLess(second, first - 0.2)
        self.assertLess(third, second)

    def test_context_stays_usable_for_inference_after_finetune(self):
        # llama_decode() on a context that had been trained on used to abort.
        model = self.load()
        ctx = self.context(model)
        rows = tiny.sample_rows(3, 17)
        prompt = rows[0][:8]
        before = self.logits(ctx, prompt).tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.finetune(rows, epochs=2, lr=1e-2)
        after = self.logits(ctx, prompt).tolist()
        self.assertGreater(max_abs_diff(before, after), 1e-3, "the weights did not change")
        # The long-lived context sees the same weights as a brand-new one.
        fresh = self.logits(self.context(model), prompt).tolist()
        self.assertLess(max_abs_diff(after, fresh), 1e-5)

    def test_padding_is_never_trained_on(self):
        # Sequences shorter than the context window used to be padded with EOS, and the
        # model was trained to predict EOS after EOS on every padded position (240 of 256
        # here), which swamps the 16 real positions.
        model = self.load()
        ctx = self.context(model)
        rows = tiny.sample_rows(4, 17)
        self.assertFalse(any(tiny.EOS_ID in row for row in rows))
        eos_context = [tiny.BOS_ID] + [tiny.EOS_ID] * 6

        def p_eos_after_eos():
            return torch.softmax(self.logits(ctx, eos_context, only_last=True)[0], dim=-1).tolist()[tiny.EOS_ID]

        before = p_eos_after_eos()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.finetune(rows, epochs=8, lr=1e-2)
        after = p_eos_after_eos()
        self.assertLess(before, 0.02)
        self.assertLess(after, 0.02, "the model was taught to emit EOS after EOS, i.e. trained on padding")

    def test_result_does_not_depend_on_the_inference_context_size(self):
        # Only the sequences determine what is trained, not how large (or how batched) the
        # context that finetune() is called on happens to be.
        rows = tiny.sample_rows(3, 17)
        prompt = rows[0][:8]
        results = []
        for n_ctx, n_batch in ((256, 256), (512, 512), (768, 100)):
            model = self.load()
            ctx = self.context(model, n_ctx=n_ctx, n_batch=n_batch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # n_batch=100 does not divide n_ctx=768: llama_opt_init() on such a context aborts.
                ctx.finetune(rows, epochs=2, lr=1e-2)
            results.append(self.logits(self.context(model), prompt).tolist())
        self.assertLess(max_abs_diff(results[0], results[1]), 1e-5)
        self.assertLess(max_abs_diff(results[0], results[2]), 1e-5)

    def test_model_context_length_survives_finetune_and_save(self):
        # llama.cpp's trainer overwrites the model's n_ctx_train, which is also what is
        # written to a saved model as its context length.
        model = self.load()
        ctx = self.context(model)
        rows = tiny.sample_rows(3, 17)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.finetune(rows, epochs=1, lr=1e-2)
        self.assertEqual(model.n_ctx_train, 1024)

        saved = self._dir / "finetuned.gguf"
        model.save_model(saved)
        reloaded = self.load(saved)
        self.assertEqual(reloaded.n_ctx_train, 1024)
        self.assertEqual(reloaded.metadata["llama.context_length"], "1024")
        prompt = rows[0][:8]
        live = self.logits(ctx, prompt).tolist()
        again = self.logits(self.context(reloaded), prompt).tolist()
        self.assertLess(max_abs_diff(live, again), 1e-5)

    def _changed_tensors(self, rows):
        model = self.load()
        ctx = self.context(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx.finetune(rows, epochs=1, lr=1e-2)
        saved = self._dir / f"changed_{len(rows[0])}.gguf"
        model.save_model(saved)
        old, new = load_gguf(self.model_path), load_gguf(saved)
        self.assertEqual(sorted(old), sorted(new))
        changed = {name for name in old if float((old[name] - new[name]).abs().max().item()) > 0.0}
        return changed, set(old) - changed, [str(w.message) for w in caught]

    def test_which_weights_are_trained(self):
        kv = {f"blk.{i}.{name}.weight" for i in range(2) for name in ("attn_k", "attn_v")}

        # 257 tokens = 256 trained positions fill a training window: everything trainable moves.
        changed, unchanged, messages = self._changed_tensors(tiny.sample_rows(2, 257))
        self.assertEqual(unchanged, {"token_embd.weight"})  # frozen by llama.cpp itself
        self.assertEqual(messages, [])

        # Shorter sequences cannot fill a window. llama.cpp then leaves the K/V projections
        # without gradient; finetune() says so instead of doing it silently.
        changed, unchanged, messages = self._changed_tensors(tiny.sample_rows(2, 17))
        self.assertEqual(unchanged, {"token_embd.weight"} | kv)
        self.assertIn("blk.0.attn_q.weight", changed)
        self.assertIn("output.weight", changed)
        self.assertEqual(len(messages), 1)
        self.assertIn("key/value projection", messages[0])

    def test_validation_split_and_long_sequences(self):
        model = self.load()
        ctx = self.context(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.finetune(tiny.sample_rows(6, 33), epochs=2, lr=1e-2, val_split=0.34)
            # Longer than the context window (256): truncated to it rather than rejected.
            ctx.finetune(tiny.sample_rows(2, 300), epochs=1, lr=1e-2)
            ctx.finetune(torch.tensor(tiny.sample_rows(2, 17)), epochs=1, lr=1e-2)
            ctx.finetune(tiny.sample_rows(2, 17), epochs=0)
        self.assertEqual(model.n_ctx_train, 1024)
        self.assertEqual(tuple(self.logits(ctx, [tiny.BOS_ID, 300, 301]).shape), (3, 320))

    def test_memory_mapped_model_is_rejected_instead_of_segfaulting(self):
        # The default LlamaModel(path) maps the weights read-only; the optimizer writing to
        # them killed the process with SIGSEGV.
        model = self.load(use_mmap=True)
        ctx = self.context(model)
        with self.assertRaisesRegex(RuntimeError, "use_mmap=False"):
            ctx.finetune(tiny.sample_rows(2, 17), epochs=1, lr=1e-2)
        self.assertEqual(tuple(self.logits(ctx, [tiny.BOS_ID, 300]).shape), (2, 320))

    def test_unsupported_arguments_raise_before_reaching_llama_cpp(self):
        model = self.load()
        ctx = self.context(model)
        rows = tiny.sample_rows(2, 17)
        for kwargs, message in (
            (dict(batch_size=2), "batch_size"),
            (dict(batch_size=0), "batch_size"),
            (dict(lr=0.0), "lr"),
            (dict(lr=float("nan")), "lr"),
            (dict(lr=-1e-3), "lr"),
            (dict(weight_decay=1.5), "weight_decay"),
            (dict(weight_decay=-0.1), "weight_decay"),
            (dict(epochs=-1), "epochs"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    ctx.finetune(rows, **kwargs)
        for bad in ([[tiny.BOS_ID, 300, 100000]], [[tiny.BOS_ID, -5, 300]]):
            with self.subTest(tokens=bad):
                with self.assertRaisesRegex(RuntimeError, "outside the vocabulary"):
                    ctx.finetune(bad)
        self.assertEqual(model.n_ctx_train, 1024)

    def test_native_entry_point_validates_on_its_own(self):
        # ggml enforces these with GGML_ASSERT, i.e. abort(); the C function must refuse
        # them itself rather than rely on the Python wrapper.
        model = self.load()
        ctx = self.context(model)
        lib = _C.llama_lib()
        rows = tiny.sample_rows(2, 17)
        flat = [t for row in rows for t in row]
        arr = (ctypes.c_int32 * len(flat))(*flat)

        def call(epochs=1, batch_size=1, lr=1e-3, wd=0.0, val_split=0.0, optimizer=0, n=2, seq_len=17):
            return lib.gt_llama_finetune(ctx._ptr, arr, n, seq_len, epochs, batch_size, lr, wd, val_split, optimizer)

        for kwargs, message in (
            (dict(batch_size=4), "batch_size must be 1"),
            (dict(lr=0.0), "lr must be"),
            (dict(lr=1e-60), "lr must be"),  # underflows to 0 as a float
            (dict(wd=3.0), "weight decay"),
            (dict(val_split=1.0), "val_split"),
            (dict(optimizer=7), "optimizer"),
            (dict(seq_len=1), "two or more tokens"),
            (dict(n=0), "two or more tokens"),
        ):
            with self.subTest(**kwargs):
                self.assertEqual(call(**kwargs), -1)
                self.assertIn(message, _C.last_error())
        self.assertEqual(call(epochs=0), 0)


if __name__ == "__main__":
    unittest.main()
