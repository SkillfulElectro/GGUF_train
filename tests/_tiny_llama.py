"""Writes a tiny random-weight llama-architecture GGUF model for the llama.cpp bridge tests.

The repository ships no model file, and a real one is far too large for a unit test. A
2-layer, 32-dimensional transformer with a 320-token vocabulary is about 100 kB, loads in
milliseconds and still goes through exactly the same llama.cpp code paths (tokenizer,
llama_decode, llama_opt_*) as a real model.

The file is written with nothing but ``struct`` (see ``_gguf_raw``) so the tests need
neither NumPy nor llama.cpp's gguf-py, and do not depend on the GGUF writer of the package
under test.
"""

from __future__ import annotations

import gc
import os
import random
import struct
import sys
from array import array
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gguf_raw import ARRAY as _ARRAY, BOOL as _BOOL, F32 as _F32, FLOAT32 as _FLOAT32, INT32 as _INT32  # noqa: E402
from _gguf_raw import STRING as _STRING, UINT32 as _UINT32, kv as _kv, write_gguf  # noqa: E402

# llama.cpp token types
_TOKEN_NORMAL, _TOKEN_UNKNOWN, _TOKEN_CONTROL, _TOKEN_BYTE = 1, 2, 3, 6

UNK_ID, BOS_ID, EOS_ID = 0, 1, 2
N_SPECIAL = 3
N_BYTE_TOKENS = 256
#: Ids below this are special/byte tokens; ordinary word tokens start here.
FIRST_WORD_ID = N_SPECIAL + N_BYTE_TOKENS


def write_tiny_llama(
    path,
    n_vocab: int = 320,
    n_embd: int = 32,
    n_head: int = 4,
    n_layer: int = 2,
    n_ff: int = 64,
    n_ctx_train: int = 1024,
    seed: int = 0,
    chat_template: str | None = None,
) -> Path:
    """Write the model to ``path`` and return it."""
    if n_vocab <= FIRST_WORD_ID:
        raise ValueError(f"n_vocab must exceed {FIRST_WORD_ID}")
    if n_embd % n_head:
        raise ValueError("n_embd must be divisible by n_head")
    rng = random.Random(seed)
    n_words = n_vocab - FIRST_WORD_ID

    # A SentencePiece-style vocabulary: the byte tokens are what the tokenizer falls back to.
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(N_BYTE_TOKENS)] + [f"\u2581w{i}" for i in range(n_words)]
    scores = [0.0] * FIRST_WORD_ID + [-float(i) for i in range(n_words)]
    types = [_TOKEN_UNKNOWN, _TOKEN_CONTROL, _TOKEN_CONTROL] + [_TOKEN_BYTE] * N_BYTE_TOKENS + [_TOKEN_NORMAL] * n_words

    kvs = [
        _kv("general.architecture", _STRING, "llama"),
        _kv("general.name", _STRING, "gguf-train-tiny-test-model"),
        _kv("general.file_type", _UINT32, 0),
        _kv("llama.context_length", _UINT32, n_ctx_train),
        _kv("llama.embedding_length", _UINT32, n_embd),
        _kv("llama.block_count", _UINT32, n_layer),
        _kv("llama.feed_forward_length", _UINT32, n_ff),
        _kv("llama.attention.head_count", _UINT32, n_head),
        _kv("llama.attention.head_count_kv", _UINT32, n_head),
        _kv("llama.attention.layer_norm_rms_epsilon", _FLOAT32, 1e-5),
        _kv("llama.rope.dimension_count", _UINT32, n_embd // n_head),
        _kv("llama.vocab_size", _UINT32, n_vocab),
        _kv("tokenizer.ggml.model", _STRING, "llama"),
        _kv("tokenizer.ggml.pre", _STRING, "default"),
        _kv("tokenizer.ggml.tokens", _ARRAY, tokens, _STRING),
        _kv("tokenizer.ggml.scores", _ARRAY, scores, _FLOAT32),
        _kv("tokenizer.ggml.token_type", _ARRAY, types, _INT32),
        _kv("tokenizer.ggml.unknown_token_id", _UINT32, UNK_ID),
        _kv("tokenizer.ggml.bos_token_id", _UINT32, BOS_ID),
        _kv("tokenizer.ggml.eos_token_id", _UINT32, EOS_ID),
        _kv("tokenizer.ggml.add_bos_token", _BOOL, True),
        _kv("tokenizer.ggml.add_eos_token", _BOOL, False),
    ]
    if chat_template is not None:
        kvs.append(_kv("tokenizer.chat_template", _STRING, chat_template))

    # (name, GGUF dims). GGUF stores dimensions innermost-first, i.e. a (rows, cols) matrix
    # is written as [cols, rows].
    specs = [("token_embd.weight", [n_embd, n_vocab])]
    for i in range(n_layer):
        specs.append((f"blk.{i}.attn_norm.weight", [n_embd]))
        specs += [(f"blk.{i}.{name}.weight", [n_embd, n_embd]) for name in ("attn_q", "attn_k", "attn_v", "attn_output")]
        specs.append((f"blk.{i}.ffn_norm.weight", [n_embd]))
        specs.append((f"blk.{i}.ffn_gate.weight", [n_embd, n_ff]))
        specs.append((f"blk.{i}.ffn_down.weight", [n_ff, n_embd]))
        specs.append((f"blk.{i}.ffn_up.weight", [n_embd, n_ff]))
    specs.append(("output_norm.weight", [n_embd]))
    specs.append(("output.weight", [n_embd, n_vocab]))

    tensors = []
    for name, dims in specs:
        count = 1
        for d in dims:
            count *= d
        if name.endswith("_norm.weight"):
            values = array("f", [1.0] * count)
        else:
            values = array("f", [rng.gauss(0.0, 0.05) for _ in range(count)])
        tensors.append((name, dims, _F32, struct.pack(f"<{count}f", *values)))
    return write_gguf(path, kvs, tensors)


def silence_native_stderr():
    """Send file descriptor 2 to /dev/null; returns the function that undoes it.

    llama.cpp logs hundreds of lines per model load straight to stderr, which would bury
    the test report. Set GGUF_TRAIN_TEST_VERBOSE=1 to see them (e.g. to read the message
    of a GGML_ASSERT).
    """
    if os.environ.get("GGUF_TRAIN_TEST_VERBOSE") == "1":
        return lambda: None
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

    def restore():
        gc.collect()  # contexts freed after this point would log again
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)

    return restore


def sample_rows(n_rows: int = 4, length: int = 17, n_vocab: int = 320) -> list:
    """Deterministic token sequences (BOS followed by word tokens) for training tests."""
    n_words = n_vocab - FIRST_WORD_ID
    return [[BOS_ID] + [FIRST_WORD_ID + (i * 7 + j * 3) % n_words for j in range(length - 1)] for i in range(n_rows)]
