"""ctypes binding loader and auto-builder for libgguf_train and libllama_bridge."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_LIBS_DIR = _PKG_DIR / "_libs"
_ROOT_DIR = _PKG_DIR.parent

# Unary op codes (matching enum gt_unary_op in csrc/gguf_train.h)
UNARY_NEG = 0
UNARY_ABS = 1
UNARY_SIGN = 2
UNARY_SQRT = 3
UNARY_RSQRT = 4
UNARY_SQUARE = 5
UNARY_EXP = 6
UNARY_EXPM1 = 7
UNARY_EXP2 = 8
UNARY_LOG = 9
UNARY_LOG2 = 10
UNARY_LOG10 = 11
UNARY_LOG1P = 12
UNARY_SIN = 13
UNARY_COS = 14
UNARY_TAN = 15
UNARY_ASIN = 16
UNARY_ACOS = 17
UNARY_ATAN = 18
UNARY_SINH = 19
UNARY_COSH = 20
UNARY_TANH = 21
UNARY_ASINH = 22
UNARY_ACOSH = 23
UNARY_ATANH = 24
UNARY_ERF = 25
UNARY_ERFC = 26
UNARY_ERFINV = 27
UNARY_LGAMMA = 28
UNARY_DIGAMMA = 29
UNARY_FLOOR = 30
UNARY_CEIL = 31
UNARY_ROUND = 32
UNARY_TRUNC = 33
UNARY_FRAC = 34
UNARY_RECIPROCAL = 35
UNARY_RELU = 36
UNARY_RELU6 = 37
UNARY_SIGMOID = 38
UNARY_SILU = 39
UNARY_GELU = 40
UNARY_GELU_TANH = 41
UNARY_GELU_QUICK = 42
UNARY_ELU = 43
UNARY_CELU = 44
UNARY_SELU = 45
UNARY_LEAKY_RELU = 46
UNARY_SOFTPLUS = 47
UNARY_MISH = 48
UNARY_HARDSWISH = 49
UNARY_HARDSIGMOID = 50
UNARY_HARDTANH = 51
UNARY_LOG_SIGMOID = 52
UNARY_CLAMP = 53
UNARY_ISNAN = 54
UNARY_ISINF = 55
UNARY_ISFINITE = 56
UNARY_LOGICAL_NOT = 57
UNARY_BITWISE_NOT = 58
UNARY_NAN_TO_NUM = 59
UNARY_SILU_DERIV = 60
UNARY_GELU_DERIV = 61
UNARY_GELU_TANH_DERIV = 62
UNARY_SIGMOID_DERIV = 63
UNARY_TANH_DERIV = 64
UNARY_ELU_DERIV = 65
UNARY_SOFTPLUS_DERIV = 66
UNARY_MISH_DERIV = 67
UNARY_REAL = 68
UNARY_IMAG = 69
UNARY_CONJ = 70
UNARY_ANGLE = 71

# Binary op codes (matching enum gt_binary_op)
BIN_ADD = 0
BIN_SUB = 1
BIN_MUL = 2
BIN_DIV = 3
BIN_FLOOR_DIV = 4
BIN_FMOD = 5
BIN_REMAINDER = 6
BIN_POW = 7
BIN_ATAN2 = 8
BIN_MAXIMUM = 9
BIN_MINIMUM = 10
BIN_EQ = 11
BIN_NE = 12
BIN_LT = 13
BIN_LE = 14
BIN_GT = 15
BIN_GE = 16
BIN_LOGICAL_AND = 17
BIN_LOGICAL_OR = 18
BIN_LOGICAL_XOR = 19
BIN_BITWISE_AND = 20
BIN_BITWISE_OR = 21
BIN_BITWISE_XOR = 22

# Reduction op codes (matching enum gt_reduce_op)
RED_SUM = 0
RED_MEAN = 1
RED_PROD = 2
RED_MAX = 3
RED_MIN = 4
RED_ALL = 5
RED_ANY = 6
RED_ARGMAX = 7
RED_ARGMIN = 8
RED_LOGSUMEXP = 9
RED_NORM_L1 = 10
RED_NORM_L2 = 11
RED_VAR_POP = 12
RED_VAR_SAMPLE = 13
RED_STD_POP = 14
RED_STD_SAMPLE = 15


def _lib_name(stem: str) -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        return f"lib{stem}.dylib"
    if sys_name == "Windows":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _source_build_script() -> Path | None:
    """build_native.py of the source checkout this package is imported from, if any.

    An installed copy (site-packages) has neither the build script nor the C++
    sources beside it. Requiring csrc/ as well keeps an unrelated build_native.py
    that merely happens to sit next to the package from ever being executed.
    """
    script = _ROOT_DIR / "build_native.py"
    if script.is_file() and (_ROOT_DIR / "csrc" / "gguf_train.h").is_file():
        return script
    return None


def ensure_built(include_llama: bool = True) -> None:
    wanted = [_LIBS_DIR / _lib_name("gguf_train")]
    if include_llama:
        wanted.append(_LIBS_DIR / _lib_name("llama_bridge"))
    missing = [p for p in wanted if not p.exists()]
    if not missing:
        return

    build_script = _source_build_script()
    if build_script is None:
        names = ", ".join(p.name for p in missing)
        raise RuntimeError(
            f"gguf_train native libraries not found in {_LIBS_DIR}: {names}. "
            "This copy of gguf_train is installed without them and cannot compile them itself, "
            "because the C++ sources and build_native.py are only part of the source tree. "
            "Reinstall from the source tree or sdist with a C/C++ compiler available and without "
            "GGUF_TRAIN_SKIP_BUILD=1, e.g. `pip install --force-reinstall --no-cache-dir .`."
        )
    import importlib.util

    sys.stderr.write("[gguf_train] First import: building native llama.cpp/ggml backend...\n")
    sys.stderr.flush()
    spec = importlib.util.spec_from_file_location("_gguf_train_builder", str(build_script))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.build_direct(build_llama=include_llama)


_dll_dir_handle = None


def _register_dll_dir() -> None:
    """Let Windows resolve the native libraries' dependencies from _libs.

    Python 3.8+ no longer searches PATH for the DLLs a loaded DLL depends on.
    gguf_train, llama_bridge, llama and ggml depend on one another and are loaded
    at different times, so their directory has to be registered explicitly.
    """
    global _dll_dir_handle
    if (
        sys.platform == "win32"
        and _dll_dir_handle is None
        and hasattr(os, "add_dll_directory")
        and _LIBS_DIR.is_dir()
    ):
        _dll_dir_handle = os.add_dll_directory(str(_LIBS_DIR))


def _load_native(stem: str) -> ctypes.CDLL:
    _register_dll_dir()
    return ctypes.CDLL(str((_LIBS_DIR / _lib_name(stem)).resolve()))


def _configure_gt_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_int64 = ctypes.c_int64
    c_uint64 = ctypes.c_uint64
    c_size_t = ctypes.c_size_t
    c_float = ctypes.c_float
    c_double = ctypes.c_double
    c_char_p = ctypes.c_char_p
    p_i64 = ctypes.POINTER(c_int64)
    p_i32 = ctypes.POINTER(c_int)
    p_f64 = ctypes.POINTER(c_double)
    p_size = ctypes.POINTER(c_size_t)
    p_vp = ctypes.POINTER(c_void_p)
    p_cp = ctypes.POINTER(c_char_p)

    # Runtime
    lib.gt_init.argtypes = []
    lib.gt_init.restype = None
    lib.gt_set_num_threads.argtypes = [c_int]
    lib.gt_set_num_threads.restype = None
    lib.gt_get_num_threads.argtypes = []
    lib.gt_get_num_threads.restype = c_int
    lib.gt_manual_seed.argtypes = [c_uint64]
    lib.gt_manual_seed.restype = None
    lib.gt_rng_get_state.argtypes = [c_char_p, c_int]
    lib.gt_rng_get_state.restype = c_int
    lib.gt_rng_set_state.argtypes = [c_char_p]
    lib.gt_rng_set_state.restype = c_int
    lib.gt_device_count.argtypes = [c_int]
    lib.gt_device_count.restype = c_int
    lib.gt_device_name.argtypes = [c_int, c_int, c_char_p, c_int]
    lib.gt_device_name.restype = None
    lib.gt_device_memory.argtypes = [c_int, c_int, p_size, p_size]
    lib.gt_device_memory.restype = None
    lib.gt_synchronize.argtypes = [c_int, c_int]
    lib.gt_synchronize.restype = None
    lib.gt_cpu_features.argtypes = [c_char_p, c_int]
    lib.gt_cpu_features.restype = None
    lib.gt_backend_summary.argtypes = [c_char_p, c_int]
    lib.gt_backend_summary.restype = None
    lib.gt_last_error.argtypes = []
    lib.gt_last_error.restype = c_char_p
    lib.gt_clear_error.argtypes = []
    lib.gt_clear_error.restype = None

    # Tensor lifecycle & metadata
    lib.gt_tensor_empty.argtypes = [c_int, p_i64, c_int, c_int, c_int]
    lib.gt_tensor_empty.restype = c_void_p
    lib.gt_tensor_from_data.argtypes = [c_int, p_i64, c_int, c_int, c_int, c_void_p, c_size_t]
    lib.gt_tensor_from_data.restype = c_void_p
    lib.gt_tensor_clone.argtypes = [c_void_p]
    lib.gt_tensor_clone.restype = c_void_p
    lib.gt_tensor_view.argtypes = [c_void_p, c_int, p_i64, p_i64, c_int64]
    lib.gt_tensor_view.restype = c_void_p
    lib.gt_tensor_contiguous.argtypes = [c_void_p]
    lib.gt_tensor_contiguous.restype = c_void_p
    lib.gt_tensor_to_dtype.argtypes = [c_void_p, c_int]
    lib.gt_tensor_to_dtype.restype = c_void_p
    lib.gt_tensor_to_device.argtypes = [c_void_p, c_int, c_int]
    lib.gt_tensor_to_device.restype = c_void_p
    lib.gt_tensor_free.argtypes = [c_void_p]
    lib.gt_tensor_free.restype = None

    lib.gt_tensor_ndim.argtypes = [c_void_p]
    lib.gt_tensor_ndim.restype = c_int
    lib.gt_tensor_shape.argtypes = [c_void_p, p_i64]
    lib.gt_tensor_shape.restype = None
    lib.gt_tensor_strides.argtypes = [c_void_p, p_i64]
    lib.gt_tensor_strides.restype = None
    lib.gt_tensor_offset.argtypes = [c_void_p]
    lib.gt_tensor_offset.restype = c_int64
    lib.gt_tensor_numel.argtypes = [c_void_p]
    lib.gt_tensor_numel.restype = c_int64
    lib.gt_tensor_dtype.argtypes = [c_void_p]
    lib.gt_tensor_dtype.restype = c_int
    lib.gt_tensor_device_type.argtypes = [c_void_p]
    lib.gt_tensor_device_type.restype = c_int
    lib.gt_tensor_device_index.argtypes = [c_void_p]
    lib.gt_tensor_device_index.restype = c_int
    lib.gt_tensor_is_contiguous.argtypes = [c_void_p]
    lib.gt_tensor_is_contiguous.restype = c_int
    lib.gt_tensor_data_ptr.argtypes = [c_void_p]
    lib.gt_tensor_data_ptr.restype = c_uint64
    lib.gt_tensor_storage_ptr.argtypes = [c_void_p]
    lib.gt_tensor_storage_ptr.restype = c_uint64
    lib.gt_tensor_storage_host_ptr.argtypes = [c_void_p]
    lib.gt_tensor_storage_host_ptr.restype = c_uint64
    lib.gt_tensor_storage_nbytes.argtypes = [c_void_p]
    lib.gt_tensor_storage_nbytes.restype = c_size_t
    lib.gt_tensor_from_external.argtypes = [c_int, p_i64, p_i64, c_int, c_void_p, c_uint64]
    lib.gt_tensor_from_external.restype = c_void_p
    lib.gt_external_released.argtypes = [ctypes.POINTER(c_uint64), c_int64]
    lib.gt_external_released.restype = c_int64
    lib.gt_tensor_nbytes.argtypes = [c_void_p]
    lib.gt_tensor_nbytes.restype = c_size_t
    lib.gt_tensor_elem_size.argtypes = [c_int]
    lib.gt_tensor_elem_size.restype = c_size_t

    lib.gt_tensor_copy_to_host.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.gt_tensor_copy_to_host.restype = c_int
    lib.gt_tensor_copy_from_host.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.gt_tensor_copy_from_host.restype = c_int
    lib.gt_tensor_copy_inplace.argtypes = [c_void_p, c_void_p]
    lib.gt_tensor_copy_inplace.restype = c_int
    lib.gt_tensor_item_f64.argtypes = [c_void_p]
    lib.gt_tensor_item_f64.restype = c_double
    lib.gt_tensor_item_i64.argtypes = [c_void_p]
    lib.gt_tensor_item_i64.restype = c_int64

    # Creation / RNG
    lib.gt_fill_scalar_f64.argtypes = [c_void_p, c_double]
    lib.gt_fill_scalar_f64.restype = c_int
    lib.gt_fill_scalar_i64.argtypes = [c_void_p, c_int64]
    lib.gt_fill_scalar_i64.restype = c_int
    lib.gt_fill_arange.argtypes = [c_void_p, c_double, c_double]
    lib.gt_fill_arange.restype = c_int
    lib.gt_fill_linspace.argtypes = [c_void_p, c_double, c_double, c_int64]
    lib.gt_fill_linspace.restype = c_int
    lib.gt_fill_eye.argtypes = [c_void_p, c_int64, c_int64]
    lib.gt_fill_eye.restype = c_int
    lib.gt_fill_uniform.argtypes = [c_void_p, c_double, c_double]
    lib.gt_fill_uniform.restype = c_int
    lib.gt_fill_normal.argtypes = [c_void_p, c_double, c_double]
    lib.gt_fill_normal.restype = c_int
    lib.gt_fill_trunc_normal.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
    lib.gt_fill_trunc_normal.restype = c_int
    lib.gt_fill_randint.argtypes = [c_void_p, c_int64, c_int64]
    lib.gt_fill_randint.restype = c_int
    lib.gt_fill_randperm.argtypes = [c_void_p, c_int64]
    lib.gt_fill_randperm.restype = c_int
    lib.gt_fill_bernoulli_prob.argtypes = [c_void_p, c_double]
    lib.gt_fill_bernoulli_prob.restype = c_int
    lib.gt_bernoulli_tensor.argtypes = [c_void_p]
    lib.gt_bernoulli_tensor.restype = c_void_p
    lib.gt_multinomial.argtypes = [c_void_p, c_int64, c_int]
    lib.gt_multinomial.restype = c_void_p

    # Unary / Binary
    lib.gt_unary_op.argtypes = [c_void_p, c_int, c_double, c_double]
    lib.gt_unary_op.restype = c_void_p
    lib.gt_unary_inplace.argtypes = [c_void_p, c_int, c_double, c_double]
    lib.gt_unary_inplace.restype = c_int
    lib.gt_binary_op.argtypes = [c_void_p, c_void_p, c_int, c_double]
    lib.gt_binary_op.restype = c_void_p
    lib.gt_binary_scalar_op.argtypes = [c_void_p, c_double, c_int, c_int]
    lib.gt_binary_scalar_op.restype = c_void_p
    lib.gt_binary_inplace.argtypes = [c_void_p, c_void_p, c_int, c_double]
    lib.gt_binary_inplace.restype = c_int
    lib.gt_binary_scalar_inplace.argtypes = [c_void_p, c_double, c_int, c_double]
    lib.gt_binary_scalar_inplace.restype = c_int
    lib.gt_where.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.gt_where.restype = c_void_p
    lib.gt_lerp.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.gt_lerp.restype = c_void_p
    lib.gt_clamp_tensor.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_double, c_double]
    lib.gt_clamp_tensor.restype = c_void_p

    # Matmul & Linalg
    lib.gt_matmul.argtypes = [c_void_p, c_void_p]
    lib.gt_matmul.restype = c_void_p
    lib.gt_linear.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.gt_linear.restype = c_void_p
    lib.gt_addmm.argtypes = [c_void_p, c_void_p, c_void_p, c_double, c_double]
    lib.gt_addmm.restype = c_void_p
    lib.gt_baddbmm.argtypes = [c_void_p, c_void_p, c_void_p, c_double, c_double]
    lib.gt_baddbmm.restype = c_void_p
    lib.gt_outer.argtypes = [c_void_p, c_void_p]
    lib.gt_outer.restype = c_void_p
    lib.gt_linalg_inv.argtypes = [c_void_p]
    lib.gt_linalg_inv.restype = c_void_p
    lib.gt_linalg_solve.argtypes = [c_void_p, c_void_p]
    lib.gt_linalg_solve.restype = c_void_p
    lib.gt_linalg_cholesky.argtypes = [c_void_p, c_int]
    lib.gt_linalg_cholesky.restype = c_void_p
    lib.gt_linalg_qr.argtypes = [c_void_p, p_vp, p_vp, c_int]
    lib.gt_linalg_qr.restype = c_int
    lib.gt_linalg_svd.argtypes = [c_void_p, c_int, p_vp, p_vp, p_vp]
    lib.gt_linalg_svd.restype = c_int
    lib.gt_linalg_eigh.argtypes = [c_void_p, c_int, p_vp, p_vp]
    lib.gt_linalg_eigh.restype = c_int
    lib.gt_linalg_det.argtypes = [c_void_p]
    lib.gt_linalg_det.restype = c_void_p
    lib.gt_linalg_slogdet.argtypes = [c_void_p, p_vp, p_vp]
    lib.gt_linalg_slogdet.restype = c_int

    # Reductions / Sort / TopK / Cumsum
    lib.gt_reduce.argtypes = [c_void_p, c_int, c_int, p_i32, c_int, c_int]
    lib.gt_reduce.restype = c_void_p
    lib.gt_norm_p.argtypes = [c_void_p, c_double, c_int, p_i32, c_int]
    lib.gt_norm_p.restype = c_void_p
    lib.gt_sort.argtypes = [c_void_p, c_int, c_int, p_vp, p_vp]
    lib.gt_sort.restype = c_int
    lib.gt_topk.argtypes = [c_void_p, c_int64, c_int, c_int, c_int, p_vp, p_vp]
    lib.gt_topk.restype = c_int
    lib.gt_cumsum.argtypes = [c_void_p, c_int]
    lib.gt_cumsum.restype = c_void_p
    lib.gt_cumprod.argtypes = [c_void_p, c_int]
    lib.gt_cumprod.restype = c_void_p

    # NN ops
    lib.gt_softmax.argtypes = [c_void_p, c_int, c_int]
    lib.gt_softmax.restype = c_void_p
    lib.gt_softmax_backward.argtypes = [c_void_p, c_void_p, c_int, c_int]
    lib.gt_softmax_backward.restype = c_void_p
    lib.gt_rms_norm.argtypes = [c_void_p, c_void_p, c_float]
    lib.gt_rms_norm.restype = c_void_p
    lib.gt_rms_norm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_float, p_vp]
    lib.gt_rms_norm_backward.restype = c_void_p
    lib.gt_layer_norm.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_float]
    lib.gt_layer_norm.restype = c_void_p
    lib.gt_layer_norm_backward.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_float, p_vp, p_vp, p_vp]
    lib.gt_layer_norm_backward.restype = c_int
    lib.gt_group_norm.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_float]
    lib.gt_group_norm.restype = c_void_p
    lib.gt_batch_norm.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_double, c_double, p_vp, p_vp]
    lib.gt_batch_norm.restype = c_void_p
    lib.gt_batch_norm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_double, p_vp, p_vp, p_vp]
    lib.gt_batch_norm_backward.restype = c_int
    lib.gt_embedding.argtypes = [c_void_p, c_void_p, c_int64]
    lib.gt_embedding.restype = c_void_p
    lib.gt_embedding_backward.argtypes = [c_void_p, c_void_p, c_int64, c_int64]
    lib.gt_embedding_backward.restype = c_void_p

    lib.gt_conv2d.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
    lib.gt_conv2d.restype = c_void_p
    lib.gt_conv2d_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, p_vp, p_vp, p_vp]
    lib.gt_conv2d_backward.restype = c_int
    lib.gt_conv1d.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    lib.gt_conv1d.restype = c_void_p
    lib.gt_conv1d_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, p_vp, p_vp, p_vp]
    lib.gt_conv1d_backward.restype = c_int
    lib.gt_conv_transpose2d.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
    lib.gt_conv_transpose2d.restype = c_void_p
    lib.gt_conv_transpose2d_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, p_vp, p_vp, p_vp]
    lib.gt_conv_transpose2d_backward.restype = c_int
    lib.gt_conv3d.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
    lib.gt_conv3d.restype = c_void_p
    lib.gt_conv3d_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, p_vp, p_vp, p_vp]
    lib.gt_conv3d_backward.restype = c_int

    lib.gt_max_pool2d.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, p_vp, p_vp]
    lib.gt_max_pool2d.restype = c_int
    lib.gt_max_pool2d_backward.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.gt_max_pool2d_backward.restype = c_void_p
    lib.gt_avg_pool2d.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
    lib.gt_avg_pool2d.restype = c_void_p
    lib.gt_avg_pool2d_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
    lib.gt_avg_pool2d_backward.restype = c_void_p
    lib.gt_upsample_2d.argtypes = [c_void_p, c_int64, c_int64, c_int, c_int]
    lib.gt_upsample_2d.restype = c_void_p
    lib.gt_upsample_2d_backward.argtypes = [c_void_p, c_int64, c_int64, c_int, c_int]
    lib.gt_upsample_2d_backward.restype = c_void_p

    lib.gt_cross_entropy_loss.argtypes = [c_void_p, c_void_p, c_void_p, c_int64, c_double, c_int]
    lib.gt_cross_entropy_loss.restype = c_void_p
    lib.gt_cross_entropy_loss_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_double, c_int]
    lib.gt_cross_entropy_loss_backward.restype = c_void_p
    lib.gt_rope.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_float, c_float]
    lib.gt_rope.restype = c_void_p
    lib.gt_rope_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_float, c_float]
    lib.gt_rope_backward.restype = c_void_p
    lib.gt_flash_attention.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_int]
    lib.gt_flash_attention.restype = c_void_p
    lib.gt_swiglu.argtypes = [c_void_p, c_void_p]
    lib.gt_swiglu.restype = c_void_p

    # Shape / indexing / pad / triu
    lib.gt_cat.argtypes = [c_int, p_vp, c_int]
    lib.gt_cat.restype = c_void_p
    lib.gt_index_select.argtypes = [c_void_p, c_int, c_void_p]
    lib.gt_index_select.restype = c_void_p
    lib.gt_index_select_add.argtypes = [c_void_p, c_int, c_void_p, c_int64]
    lib.gt_index_select_add.restype = c_void_p
    lib.gt_gather.argtypes = [c_void_p, c_int, c_void_p]
    lib.gt_gather.restype = c_void_p
    lib.gt_scatter.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_int]
    lib.gt_scatter.restype = c_void_p
    lib.gt_scatter_inplace.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_int]
    lib.gt_scatter_inplace.restype = c_int
    lib.gt_index_tensor.argtypes = [c_void_p, c_int, p_vp, p_i32]
    lib.gt_index_tensor.restype = c_void_p
    lib.gt_index_put.argtypes = [c_void_p, c_int, p_vp, p_i32, c_void_p, c_int]
    lib.gt_index_put.restype = c_int
    lib.gt_masked_select.argtypes = [c_void_p, c_void_p]
    lib.gt_masked_select.restype = c_void_p
    lib.gt_masked_fill_inplace.argtypes = [c_void_p, c_void_p, c_double]
    lib.gt_masked_fill_inplace.restype = c_int
    lib.gt_pad.argtypes = [c_void_p, c_int, p_i64, c_int, c_double]
    lib.gt_pad.restype = c_void_p
    lib.gt_tril_triu.argtypes = [c_void_p, c_int64, c_int]
    lib.gt_tril_triu.restype = c_void_p
    lib.gt_diag.argtypes = [c_void_p, c_int64]
    lib.gt_diag.restype = c_void_p
    lib.gt_flip.argtypes = [c_void_p, c_int, p_i32]
    lib.gt_flip.restype = c_void_p
    lib.gt_roll.argtypes = [c_void_p, c_int, p_i64, p_i32]
    lib.gt_roll.restype = c_void_p
    lib.gt_nonzero.argtypes = [c_void_p]
    lib.gt_nonzero.restype = c_void_p
    lib.gt_bincount.argtypes = [c_void_p, c_void_p, c_int64]
    lib.gt_bincount.restype = c_void_p
    lib.gt_unique.argtypes = [c_void_p, c_int, c_int, c_int, p_vp, p_vp, p_vp]
    lib.gt_unique.restype = c_int

    # Optimizers
    lib.gt_adamw_step.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double, c_double, c_double, c_double, c_double, c_int64, c_int, c_int]
    lib.gt_adamw_step.restype = c_int
    lib.gt_sgd_step.argtypes = [c_void_p, c_void_p, c_void_p, c_double, c_double, c_double, c_double, c_int, c_int]
    lib.gt_sgd_step.restype = c_int

    # FFT
    lib.gt_fft_1d.argtypes = [c_void_p, c_int64, c_int, c_int, c_int, c_int]
    lib.gt_fft_1d.restype = c_void_p

    # GGUF & Quantization
    lib.gt_gguf_save.argtypes = [c_char_p, c_int, p_cp, p_vp, c_int, p_cp, p_i32, p_cp]
    lib.gt_gguf_save.restype = c_int
    lib.gt_gguf_open.argtypes = [c_char_p]
    lib.gt_gguf_open.restype = c_void_p
    lib.gt_gguf_close.argtypes = [c_void_p]
    lib.gt_gguf_close.restype = None
    lib.gt_gguf_n_tensors.argtypes = [c_void_p]
    lib.gt_gguf_n_tensors.restype = c_int64
    lib.gt_gguf_tensor_name.argtypes = [c_void_p, c_int64]
    lib.gt_gguf_tensor_name.restype = c_char_p
    lib.gt_gguf_get_tensor.argtypes = [c_void_p, c_char_p, c_int]
    lib.gt_gguf_get_tensor.restype = c_void_p
    lib.gt_gguf_n_kv.argtypes = [c_void_p]
    lib.gt_gguf_n_kv.restype = c_int64
    lib.gt_gguf_kv_key.argtypes = [c_void_p, c_int64]
    lib.gt_gguf_kv_key.restype = c_char_p
    lib.gt_gguf_kv_type.argtypes = [c_void_p, c_int64]
    lib.gt_gguf_kv_type.restype = c_int
    lib.gt_gguf_kv_val_str.argtypes = [c_void_p, c_int64, c_char_p, c_int]
    lib.gt_gguf_kv_val_str.restype = c_int
    lib.gt_gguf_save_ex.argtypes = [c_char_p, c_int, p_cp, p_vp, c_int, p_cp, p_i32, p_cp, p_i32, p_i64, p_vp, c_char_p]
    lib.gt_gguf_save_ex.restype = c_int
    lib.gt_gguf_kv_arr_type.argtypes = [c_void_p, c_int64]
    lib.gt_gguf_kv_arr_type.restype = c_int
    lib.gt_gguf_kv_arr_n.argtypes = [c_void_p, c_int64]
    lib.gt_gguf_kv_arr_n.restype = c_int64
    lib.gt_gguf_kv_arr_data.argtypes = [c_void_p, c_int64]
    lib.gt_gguf_kv_arr_data.restype = c_void_p
    lib.gt_gguf_kv_arr_str.argtypes = [c_void_p, c_int64, c_int64]
    lib.gt_gguf_kv_arr_str.restype = c_char_p
    lib.gt_quantize.argtypes = [c_void_p, c_int]
    lib.gt_quantize.restype = c_void_p
    lib.gt_dequantize.argtypes = [c_void_p, c_int]
    lib.gt_dequantize.restype = c_void_p

    # Native static ggml_opt_graph
    lib.gt_opt_graph_create.argtypes = [c_int, c_int, c_int64, c_int64]
    lib.gt_opt_graph_create.restype = c_void_p
    lib.gt_opt_graph_free.argtypes = [c_void_p]
    lib.gt_opt_graph_free.restype = None
    lib.gt_opt_graph_input.argtypes = [c_void_p, c_int, p_i64]
    lib.gt_opt_graph_input.restype = c_int
    lib.gt_opt_graph_param.argtypes = [c_void_p, c_void_p]
    lib.gt_opt_graph_param.restype = c_int
    lib.gt_opt_graph_linear.argtypes = [c_void_p, c_int, c_int, c_int]
    lib.gt_opt_graph_linear.restype = c_int
    lib.gt_opt_graph_unary.argtypes = [c_void_p, c_int, c_int]
    lib.gt_opt_graph_unary.restype = c_int
    lib.gt_opt_graph_add.argtypes = [c_void_p, c_int, c_int]
    lib.gt_opt_graph_add.restype = c_int
    lib.gt_opt_graph_mul.argtypes = [c_void_p, c_int, c_int]
    lib.gt_opt_graph_mul.restype = c_int
    lib.gt_opt_graph_rms_norm.argtypes = [c_void_p, c_int, c_int, c_float]
    lib.gt_opt_graph_rms_norm.restype = c_int
    lib.gt_opt_graph_softmax.argtypes = [c_void_p, c_int]
    lib.gt_opt_graph_softmax.restype = c_int
    lib.gt_opt_graph_fit.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_int64, c_float, c_float, c_float, c_int, p_f64, p_f64]
    lib.gt_opt_graph_fit.restype = c_int

    lib.gt_init()
    return lib


def _configure_llama_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_int32 = ctypes.c_int32
    c_uint32 = ctypes.c_uint32
    c_int64 = ctypes.c_int64
    c_uint64 = ctypes.c_uint64
    c_size_t = ctypes.c_size_t
    c_float = ctypes.c_float
    c_double = ctypes.c_double
    c_char_p = ctypes.c_char_p
    p_i32 = ctypes.POINTER(c_int32)

    lib.gt_llama_backend_init.argtypes = []
    lib.gt_llama_backend_init.restype = None
    lib.gt_llama_backend_free.argtypes = []
    lib.gt_llama_backend_free.restype = None
    lib.gt_llama_load_model.argtypes = [c_char_p, c_int, c_int, c_int]
    lib.gt_llama_load_model.restype = c_void_p
    lib.gt_llama_free_model.argtypes = [c_void_p]
    lib.gt_llama_free_model.restype = None
    lib.gt_llama_save_model.argtypes = [c_void_p, c_char_p]
    lib.gt_llama_save_model.restype = c_int

    lib.gt_llama_n_vocab.argtypes = [c_void_p]
    lib.gt_llama_n_vocab.restype = c_int32
    lib.gt_llama_n_ctx_train.argtypes = [c_void_p]
    lib.gt_llama_n_ctx_train.restype = c_int32
    lib.gt_llama_n_embd.argtypes = [c_void_p]
    lib.gt_llama_n_embd.restype = c_int32
    lib.gt_llama_n_layer.argtypes = [c_void_p]
    lib.gt_llama_n_layer.restype = c_int32
    lib.gt_llama_n_head.argtypes = [c_void_p]
    lib.gt_llama_n_head.restype = c_int32
    lib.gt_llama_n_head_kv.argtypes = [c_void_p]
    lib.gt_llama_n_head_kv.restype = c_int32
    lib.gt_llama_model_n_params.argtypes = [c_void_p]
    lib.gt_llama_model_n_params.restype = c_uint64
    lib.gt_llama_model_size.argtypes = [c_void_p]
    lib.gt_llama_model_size.restype = c_uint64
    lib.gt_llama_vocab_bos.argtypes = [c_void_p]
    lib.gt_llama_vocab_bos.restype = c_int32
    lib.gt_llama_vocab_eos.argtypes = [c_void_p]
    lib.gt_llama_vocab_eos.restype = c_int32
    lib.gt_llama_vocab_pad.argtypes = [c_void_p]
    lib.gt_llama_vocab_pad.restype = c_int32
    lib.gt_llama_model_desc.argtypes = [c_void_p, c_char_p, c_size_t]
    lib.gt_llama_model_desc.restype = c_int32
    lib.gt_llama_model_meta_count.argtypes = [c_void_p]
    lib.gt_llama_model_meta_count.restype = c_int32
    lib.gt_llama_model_meta_key.argtypes = [c_void_p, c_int32, c_char_p, c_size_t]
    lib.gt_llama_model_meta_key.restype = c_int32
    lib.gt_llama_model_meta_val.argtypes = [c_void_p, c_char_p, c_char_p, c_size_t]
    lib.gt_llama_model_meta_val.restype = c_int32

    lib.gt_llama_new_context.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
    lib.gt_llama_new_context.restype = c_void_p
    lib.gt_llama_free_context.argtypes = [c_void_p]
    lib.gt_llama_free_context.restype = None
    lib.gt_llama_clear_kv.argtypes = [c_void_p]
    lib.gt_llama_clear_kv.restype = None
    lib.gt_llama_n_ctx.argtypes = [c_void_p]
    lib.gt_llama_n_ctx.restype = c_uint32

    lib.gt_llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, p_i32, c_int32, c_int, c_int]
    lib.gt_llama_tokenize.restype = c_int32
    lib.gt_llama_detokenize.argtypes = [c_void_p, p_i32, c_int32, c_char_p, c_int32, c_int, c_int]
    lib.gt_llama_detokenize.restype = c_int32
    lib.gt_llama_forward_logits.argtypes = [c_void_p, p_i32, c_int32, c_int32, c_int]
    lib.gt_llama_forward_logits.restype = c_void_p
    lib.gt_llama_forward_embeddings.argtypes = [c_void_p, p_i32, c_int32]
    lib.gt_llama_forward_embeddings.restype = c_void_p
    lib.gt_llama_finetune.argtypes = [c_void_p, p_i32, c_int64, c_int64, c_int, c_int64, c_double, c_double, c_float, c_int]
    lib.gt_llama_finetune.restype = c_int
    lib.gt_llama_backend_init()
    return lib


ensure_built(include_llama=False)
lib = _configure_gt_lib(_load_native("gguf_train"))

_llama_lib_cache: ctypes.CDLL | None = None


def llama_lib() -> ctypes.CDLL:
    global _llama_lib_cache
    if _llama_lib_cache is None:
        ensure_built(include_llama=True)
        _llama_lib_cache = _configure_llama_lib(_load_native("llama_bridge"))
    return _llama_lib_cache


def last_error() -> str:
    msg = lib.gt_last_error()
    return msg.decode("utf-8", errors="replace") if msg else ""


# A native message may name the Python exception it stands for: "IndexError: index 5 is out
# of bounds ..." is raised as IndexError("index 5 is out of bounds ..."). Everything else is
# a RuntimeError, prefixed with the operation that failed.
_NATIVE_ERROR_TYPES = {"IndexError": IndexError}


def _native_error(ctx: str, err: str, fallback: str) -> Exception:
    for name, exc_type in _NATIVE_ERROR_TYPES.items():
        tag = name + ": "
        if err.startswith(tag):
            return exc_type(err[len(tag):])
    if ctx and err:
        # the native message often names the operation already
        return RuntimeError(err if err.startswith(ctx + ":") else f"{ctx}: {err}")
    return RuntimeError(err or ctx or fallback)


def check_ptr(ptr: int | None, ctx: str = "") -> int:
    if not ptr:
        raise _native_error(ctx, last_error(), "Native operation returned NULL")
    return ptr


def check_rc(rc: int, ctx: str = "") -> None:
    if rc != 0:
        raise _native_error(ctx, last_error(), f"Native operation failed with code {rc}")
