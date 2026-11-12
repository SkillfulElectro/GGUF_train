# GGUF_train

An experimental Python tensor and neural-network library backed by ggml, with GGUF utilities and a llama.cpp model bridge. It provides a **subset** of the PyTorch API, not a complete or verified drop-in replacement. The repository also supplies a `torch` import alias; use an isolated environment to avoid shadowing an existing PyTorch installation.

## Setup

### Install from a prebuilt wheel (no compiler needed)

Versioned releases with prebuilt wheels are published as GitHub Release assets. Wheels are tagged for the platform they were built on (`py3-none-<platform>`: the native libraries are loaded through `ctypes`, so one wheel serves every Python 3 version on that platform). Pick the asset matching your platform, e.g.:

```sh
pip install https://github.com/SkillfulElectro/GGUF_train/releases/download/v0.1.0/gguf_train-0.1.0-py3-none-linux_x86_64.whl
```

If no wheel exists for your platform, install from source (requires a C/C++ compiler):

```sh
git clone https://github.com/SkillfulElectro/GGUF_train.git
cd GGUF_train
python3 -m pip install .
```

The native builder downloads a **pinned** upstream llama.cpp/ggml revision at build time (`LLAMA_PINNED_COMMIT` in `build_native.py`, the revision `csrc/` is written against) and compiles it together with the `csrc/` bridge into shared libraries under `gguf_train/_libs/` using `gcc`/`g++` or `clang`/`clang++` directly (no CMake or Python headers required). The list of llama.cpp/ggml source files is discovered from the downloaded tree rather than hardcoded. It can also be run explicitly:

```sh
python3 build_native.py
```

Builds are reproducible by default. To build against a different llama.cpp revision (unsupported revisions may not compile against `csrc/`):

```sh
# exact commit
GGUF_TRAIN_LLAMA_COMMIT=<sha> python3 build_native.py
# or a branch/tag; a branch is a moving target, so add --refresh-llama to re-download it
GGUF_TRAIN_LLAMA_BRANCH=master python3 build_native.py --refresh-llama
# or another repository/mirror
GGUF_TRAIN_LLAMA_REPO=https://github.com/<you>/llama.cpp python3 build_native.py
```

The downloaded tree records which revision it holds; requesting another revision re-downloads it and rebuilds the libraries. A `third_party/llama.cpp` directory you placed there yourself is used as-is (and never overwritten if it is a git checkout) unless you pass `--refresh-llama` or request a revision explicitly.

When running from a source checkout, the first import compiles the native libraries if they are missing; build times can be substantial. An installed copy (wheel or `pip install .`) ships the libraries and never compiles on import. The builder currently targets CPU; accelerator support depends on the build configuration and hardware.

### Verify

```sh
python3 examples/linear_regression.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Examples

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(4, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
x = torch.randn(8, 4)
y = torch.randn(8, 1)
optimizer.zero_grad()
loss = ((model(x) - y) ** 2).mean()
loss.backward()
optimizer.step()
```

Save and read a tensor dictionary in GGUF format:

```python
from torch.gguf import save_gguf, load_gguf
save_gguf('weights.gguf', {'weight': model.weight.detach()})
weights = load_gguf('weights.gguf')
```

Tensors come back with the shape and dtype they were saved with. GGUF itself stores at most four dimensions, drops outer dimensions of size one and has no uint8, bool or complex type, so for such tensors the real shape and dtype are recorded in the file under the reserved metadata keys `gguf_train.shape.<name>` and `gguf_train.dtype.<name>` (other GGUF readers see the stored form: int8 bytes, float pairs, folded dimensions). Tensor names are limited to 63 bytes by the format. Metadata values are read in full as `int`, `float`, `bool`, `str`, or a list of those for GGUF arrays; `save_gguf(..., metadata=...)` writes Python values as int64/float64/bool/string (lists as arrays), and `save_gguf(..., copy_metadata_from=path)` copies another file's metadata with its exact GGUF types, which is what `quantize_gguf` does.

For a compatible GGUF language model, `torch.llama.LlamaModel(path)` provides tokenization and `new_context()` provides logits, embeddings and generation. Model files are not included. Choose `n_batch` at least as large as each input passed to a forward call; oversized inputs are rejected rather than automatically chunked. `LlamaContext.finetune(tokens)` fine-tunes the model in place with llama.cpp's native trainer: each equal-length token sequence is one independent next-token-prediction sample, and nothing but the sequence itself (no padding) is trained on. It requires a model loaded with `use_mmap=False` and takes one optimizer step per sequence (`batch_size` must be 1); only F32 tensors are updated, and the attention key/value projections are only updated when sequences have `256*k + 1` tokens (a warning is emitted otherwise). Fine-tuning remains experimental; do not use it on valuable model files without backups. `torch.llama.LoRALinear` and `inject_lora` provide a separate tensor-level LoRA path.

## Behaviour notes

- **NumPy.** `torch.from_numpy`, `torch.as_tensor` and `Tensor.numpy()` share memory with the array (no copy) for CPU tensors of a matching dtype; NumPy functions applied to tensors (`np.exp(t)`, `t + np.float32(2)`) run as tensor operations and keep autograd. NumPy is optional.
- **Autocast.** `torch.autocast` follows one policy for every device type, PyTorch's CUDA lists: matrix products, convolutions and linear layers run in the autocast dtype; `exp`, `log`, `pow`, losses, normalization layers, `softmax` and reductions run in float32; the rest follows its inputs. The table is in the docstring of `gguf_train/amp.py`.
- **In-place operations** (`x += y`, `x.mul_(w)`, `x[i] = v`, `nn.ReLU(inplace=True)`) are part of the autograd graph. Two things PyTorch does are not done: after an in-place operation on a *view* (`v = x[0]; v.mul_(2)`), a later use of `x` is differentiated without it (`x[0] *= 2` is fine), and there is no "modified by an inplace operation" error when an earlier operation still needed the overwritten values.
- **`torch.utils.checkpoint`.** The checkpointed function may reach its tensors any way it likes (arguments, module parameters, closures, globals); it must not change a tensor from outside in place, and a tensor it stores somewhere other than its return value carries no gradient. Random numbers and the autocast state of the forward pass are replayed when it is recomputed.

## Limitations

API coverage, numerical behavior, serialization, devices, and model architectures differ from PyTorch and llama.cpp. Only the included tests and specific workflows exercised by them have been validated; expect unsupported operations and incompatibilities. Treat this project as experimental rather than production-ready.
