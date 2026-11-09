#pragma once

#include "gguf_train.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct gt_storage {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    struct ggml_tensor  * raw_tensor = nullptr;
    void                * host_ptr = nullptr;
    size_t                nbytes = 0;
    int64_t               numel = 0;
    int                   dtype = GT_F32;
    int                   device_type = GT_DEV_CPU;
    int                   device_index = 0;

    ~gt_storage() {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
    }
};

struct gt_tensor {
    int     ndim = 0;
    int64_t shape[GT_MAX_DIMS] = {};
    int64_t strides[GT_MAX_DIMS] = {};
    int64_t offset = 0;
    int64_t numel = 1;
    int     dtype = GT_F32;
    int     device_type = GT_DEV_CPU;
    int     device_index = 0;
    std::shared_ptr<gt_storage> storage;
};

struct gt_backend_entry {
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_t     backend = nullptr;
    ggml_gallocr_t     galloc = nullptr;
    std::mutex         mtx;
    std::string        name;
};

struct gt_runtime_state {
    bool initialized = false;
    int  n_threads = 2;
    std::mutex init_mtx;
    std::mutex rng_mtx;
    std::mt19937_64 rng{42};

    gt_backend_entry cpu;
    std::vector<std::unique_ptr<gt_backend_entry>> cuda;
    std::vector<std::unique_ptr<gt_backend_entry>> mps;
    std::vector<std::unique_ptr<gt_backend_entry>> vulkan;
};

gt_runtime_state & gt_rt();
void gt_set_error(const std::string & msg);
gt_backend_entry * gt_get_backend(int device_type, int device_index);

inline bool gt_is_quantized(int dtype) {
    return dtype >= GT_Q_BASE;
}

inline enum ggml_type gt_dtype_to_ggml(int dtype) {
    if (dtype >= GT_Q_BASE) {
        return (enum ggml_type)(dtype - GT_Q_BASE);
    }
    switch (dtype) {
        case GT_F32:  return GGML_TYPE_F32;
        case GT_F16:  return GGML_TYPE_F16;
        case GT_BF16: return GGML_TYPE_BF16;
        case GT_F64:  return GGML_TYPE_F64;
        case GT_I8:   return GGML_TYPE_I8;
        case GT_I16:  return GGML_TYPE_I16;
        case GT_I32:  return GGML_TYPE_I32;
        case GT_I64:  return GGML_TYPE_I64;
        case GT_U8:   return GGML_TYPE_I8;
        case GT_BOOL: return GGML_TYPE_I8;
        case GT_C64:  return GGML_TYPE_F32;
        case GT_C128: return GGML_TYPE_F64;
        default:      return GGML_TYPE_F32;
    }
}

inline int gt_ggml_to_dtype(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return GT_F32;
        case GGML_TYPE_F16:  return GT_F16;
        case GGML_TYPE_BF16: return GT_BF16;
        case GGML_TYPE_F64:  return GT_F64;
        case GGML_TYPE_I8:   return GT_I8;
        case GGML_TYPE_I16:  return GT_I16;
        case GGML_TYPE_I32:  return GT_I32;
        case GGML_TYPE_I64:  return GT_I64;
        default:             return GT_Q_BASE + (int)t;
    }
}

inline size_t gt_elem_size(int dtype) {
    switch (dtype) {
        case GT_F32:  return 4;
        case GT_F16:  return 2;
        case GT_BF16: return 2;
        case GT_F64:  return 8;
        case GT_I8:   return 1;
        case GT_U8:   return 1;
        case GT_I16:  return 2;
        case GT_I32:  return 4;
        case GT_I64:  return 8;
        case GT_BOOL: return 1;
        case GT_C64:  return 8;
        case GT_C128: return 16;
        default:      return 0;
    }
}

inline bool gt_is_floating(int dtype) {
    return dtype == GT_F32 || dtype == GT_F16 || dtype == GT_BF16 || dtype == GT_F64;
}

inline bool gt_is_complex(int dtype) {
    return dtype == GT_C64 || dtype == GT_C128;
}

inline bool gt_is_integer(int dtype) {
    return dtype == GT_I8 || dtype == GT_U8 || dtype == GT_I16 || dtype == GT_I32 || dtype == GT_I64;
}

inline int gt_promote_dtype(int a, int b) {
    if (a == b) return a;
    if (gt_is_complex(a) || gt_is_complex(b)) {
        if (a == GT_C128 || b == GT_C128 || a == GT_F64 || b == GT_F64) return GT_C128;
        return GT_C64;
    }
    if (gt_is_floating(a) || gt_is_floating(b)) {
        if (a == GT_F64 || b == GT_F64) return GT_F64;
        if (a == GT_F32 || b == GT_F32) return GT_F32;
        if (a == GT_F16 && b == GT_BF16) return GT_F32;
        if (a == GT_BF16 && b == GT_F16) return GT_F32;
        if (a == GT_BF16 || b == GT_BF16) return GT_BF16;
        return GT_F16;
    }
    if (a == GT_BOOL) return b;
    if (b == GT_BOOL) return a;
    if (a == GT_I64 || b == GT_I64) return GT_I64;
    if (a == GT_I32 || b == GT_I32) return GT_I32;
    if (a == GT_I16 || b == GT_I16) return GT_I16;
    if (a == GT_I8 || b == GT_I8) return GT_I8;
    return GT_U8;
}

inline int64_t gt_compute_numel(int ndim, const int64_t * shape) {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return n;
}

inline void gt_default_strides(int ndim, const int64_t * shape, int64_t * strides) {
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = s;
        s *= std::max<int64_t>(shape[i], 1);
    }
}

inline bool gt_check_contiguous(const gt_tensor * t) {
    if (t->numel == 0 || t->ndim == 0) return true;
    int64_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->shape[i] != 1) {
            if (t->strides[i] != expected) return false;
            expected *= t->shape[i];
        }
    }
    return true;
}

inline int64_t gt_linear_to_elem_offset(const gt_tensor * t, int64_t linear_idx) {
    if (t->ndim == 0) return t->offset;
    int64_t off = t->offset;
    int64_t rem = linear_idx;
    for (int d = t->ndim - 1; d >= 0; --d) {
        int64_t sz = t->shape[d];
        int64_t c = rem % sz;
        rem /= sz;
        off += c * t->strides[d];
    }
    return off;
}

inline double gt_read_f64_at(const void * ptr, int dtype, int64_t idx) {
    switch (dtype) {
        case GT_F32:  return (double)((const float *)ptr)[idx];
        case GT_F64:  return ((const double *)ptr)[idx];
        case GT_F16:  return (double)ggml_fp16_to_fp32(((const ggml_fp16_t *)ptr)[idx]);
        case GT_BF16: return (double)ggml_bf16_to_fp32(((const ggml_bf16_t *)ptr)[idx]);
        case GT_I8:   return (double)((const int8_t *)ptr)[idx];
        case GT_U8:   return (double)((const uint8_t *)ptr)[idx];
        case GT_I16:  return (double)((const int16_t *)ptr)[idx];
        case GT_I32:  return (double)((const int32_t *)ptr)[idx];
        case GT_I64:  return (double)((const int64_t *)ptr)[idx];
        case GT_BOOL: return ((const uint8_t *)ptr)[idx] ? 1.0 : 0.0;
        case GT_C64:  return (double)((const float *)ptr)[2 * idx];
        case GT_C128: return ((const double *)ptr)[2 * idx];
        default:      return 0.0;
    }
}

inline void gt_write_f64_at(void * ptr, int dtype, int64_t idx, double v) {
    switch (dtype) {
        case GT_F32:  ((float *)ptr)[idx] = (float)v; break;
        case GT_F64:  ((double *)ptr)[idx] = v; break;
        case GT_F16:  ((ggml_fp16_t *)ptr)[idx] = ggml_fp32_to_fp16((float)v); break;
        case GT_BF16: ((ggml_bf16_t *)ptr)[idx] = ggml_fp32_to_bf16((float)v); break;
        case GT_I8:   ((int8_t *)ptr)[idx] = (int8_t)v; break;
        case GT_U8:   ((uint8_t *)ptr)[idx] = (uint8_t)v; break;
        case GT_I16:  ((int16_t *)ptr)[idx] = (int16_t)v; break;
        case GT_I32:  ((int32_t *)ptr)[idx] = (int32_t)v; break;
        case GT_I64:  ((int64_t *)ptr)[idx] = (int64_t)v; break;
        case GT_BOOL: ((uint8_t *)ptr)[idx] = (v != 0.0) ? 1 : 0; break;
        case GT_C64:  ((float *)ptr)[2 * idx] = (float)v; ((float *)ptr)[2 * idx + 1] = 0.0f; break;
        case GT_C128: ((double *)ptr)[2 * idx] = v; ((double *)ptr)[2 * idx + 1] = 0.0; break;
        default: break;
    }
}

inline int64_t gt_read_i64_at(const void * ptr, int dtype, int64_t idx) {
    switch (dtype) {
        case GT_I64:  return ((const int64_t *)ptr)[idx];
        case GT_I32:  return (int64_t)((const int32_t *)ptr)[idx];
        case GT_I16:  return (int64_t)((const int16_t *)ptr)[idx];
        case GT_I8:   return (int64_t)((const int8_t *)ptr)[idx];
        case GT_U8:   return (int64_t)((const uint8_t *)ptr)[idx];
        case GT_BOOL: return ((const uint8_t *)ptr)[idx] ? 1 : 0;
        default:      return (int64_t)gt_read_f64_at(ptr, dtype, idx);
    }
}

inline void gt_write_i64_at(void * ptr, int dtype, int64_t idx, int64_t v) {
    switch (dtype) {
        case GT_I64:  ((int64_t *)ptr)[idx] = v; break;
        case GT_I32:  ((int32_t *)ptr)[idx] = (int32_t)v; break;
        case GT_I16:  ((int16_t *)ptr)[idx] = (int16_t)v; break;
        case GT_I8:   ((int8_t *)ptr)[idx] = (int8_t)v; break;
        case GT_U8:   ((uint8_t *)ptr)[idx] = (uint8_t)v; break;
        case GT_BOOL: ((uint8_t *)ptr)[idx] = (v != 0) ? 1 : 0; break;
        default:      gt_write_f64_at(ptr, dtype, idx, (double)v); break;
    }
}

inline std::complex<double> gt_read_c128_at(const void * ptr, int dtype, int64_t idx) {
    if (dtype == GT_C64) {
        const float * p = (const float *)ptr;
        return std::complex<double>((double)p[2 * idx], (double)p[2 * idx + 1]);
    }
    if (dtype == GT_C128) {
        const double * p = (const double *)ptr;
        return std::complex<double>(p[2 * idx], p[2 * idx + 1]);
    }
    return std::complex<double>(gt_read_f64_at(ptr, dtype, idx), 0.0);
}

inline void gt_write_c128_at(void * ptr, int dtype, int64_t idx, std::complex<double> v) {
    if (dtype == GT_C64) {
        float * p = (float *)ptr;
        p[2 * idx] = (float)v.real();
        p[2 * idx + 1] = (float)v.imag();
    } else if (dtype == GT_C128) {
        double * p = (double *)ptr;
        p[2 * idx] = v.real();
        p[2 * idx + 1] = v.imag();
    } else {
        gt_write_f64_at(ptr, dtype, idx, v.real());
    }
}

// Host synchronization helper for tensors that might reside on a GPU backend
struct gt_host_view {
    std::vector<uint8_t> tmp;
    const void * rptr = nullptr;
    void       * wptr = nullptr;
    const gt_tensor * t = nullptr;
    bool copied = false;

    const void * read(const gt_tensor * tensor);
    void       * write_empty(gt_tensor * tensor);
    void       * read_write(gt_tensor * tensor);
    void         commit(gt_tensor * tensor);
};

// Ensures a contiguous tensor on CPU (returns existing if already contiguous CPU, else makes temporary copy)
struct gt_contig_owner {
    gt_tensor * owned = nullptr;
    const gt_tensor * ptr = nullptr;
    gt_contig_owner(const gt_tensor * t, int force_dtype = -1);
    ~gt_contig_owner() {
        if (owned) gt_tensor_free(owned);
    }
};

// Helper to run a small ggml compute graph on the tensor's backend
// Note: out_gt must already be allocated; out_node is either bound to out_gt's buffer directly or copied.
bool gt_exec_ggml_graph(
    gt_backend_entry * be,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    struct ggml_tensor * out_node,
    gt_tensor * out_gt);

// Helper to wrap a contiguous gt_tensor (up to 4D, ggml-supported dtype) into a leaf ggml_tensor
struct ggml_tensor * gt_wrap_ggml_4d(
    struct ggml_context * ctx,
    const gt_tensor * t,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
