# GGUF_train

An experimental Python tensor and neural-network library backed by ggml, with GGUF utilities and a llama.cpp model bridge. It provides a **subset** of the PyTorch API, not a complete or verified drop-in replacement. The repository also supplies a `torch` import alias; use an isolated environment to avoid shadowing an existing PyTorch installation.

## Setup

### Install from a prebuilt wheel (no compiler needed)

Versioned releases with prebuilt wheels are published as GitHub Release assets. Pick the asset matching your platform and Python, e.g.:

```sh
pip install https://github.com/SkillfulElectro/GGUF_train/releases/download/v0.1.0/gguf_train-0.1.0-cp311-cp311-linux_x86_64.whl
```

If no wheel exists for your platform, install from source (requires a C/C++ compiler):

```sh
git clone https://github.com/SkillfulElectro/GGUF_train.git
cd GGUF_train
python3 -m pip install .
```

The native builder compiles vendored llama.cpp/ggml and the `csrc/` bridge into shared libraries under `gguf_train/_libs/` using `gcc`/`g++` or `clang`/`clang++` directly (no CMake or Python headers required). It can also be run explicitly:

```sh
python3 build_native.py
```

First import may compile the native libraries if they are missing. Build times can be substantial. The builder currently targets CPU; accelerator support depends on the build configuration and hardware.

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

For a compatible GGUF language model, `torch.llama.LlamaModel(path)` provides tokenization and `new_context()` provides logits, embeddings and generation. Model files are not included. Choose `n_batch` at least as large as each input passed to a forward call; oversized inputs are rejected rather than automatically chunked. Native model fine-tuning is experimental and has not been reliably validated; do not use it on valuable model files without backups. `torch.llama.LoRALinear` and `inject_lora` provide a separate tensor-level LoRA path.

## Limitations

API coverage, numerical behavior, serialization, devices, and model architectures differ from PyTorch and llama.cpp. Only the included tests and specific workflows exercised by them have been validated; expect unsupported operations and incompatibilities. Treat this project as experimental rather than production-ready.
