#include "gt_internal.h"
#include <cstdarg>
#include <sstream>
#include <thread>

static thread_local std::string g_last_error;

void gt_set_error(const std::string & msg) {
    g_last_error = msg;
}

extern "C" const char * gt_last_error(void) {
    return g_last_error.c_str();
}

extern "C" void gt_clear_error(void) {
    g_last_error.clear();
}

static void gt_ggml_log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    const char * dbg = std::getenv("GGUF_TRAIN_LOG");
    if (dbg && std::atoi(dbg) > 0) {
        fprintf(stderr, "%s", text);
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "[ggml error] %s", text);
    }
}

gt_runtime_state & gt_rt() {
    static gt_runtime_state rt;
    return rt;
}

static bool str_contains_ci(const char * hay, const char * needle) {
    if (!hay || !needle) return false;
    std::string h(hay), n(needle);
    for (auto & c : h) c = (char)std::tolower((unsigned char)c);
    for (auto & c : n) c = (char)std::tolower((unsigned char)c);
    return h.find(n) != std::string::npos;
}

extern "C" void gt_init(void) {
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.init_mtx);
    if (rt.initialized) return;

    ggml_log_set(gt_ggml_log_cb, nullptr);
    ggml_backend_load_all();

    // Always initialize CPU backend
    ggml_cpu_init();
    rt.cpu.backend = ggml_backend_cpu_init();
    rt.cpu.dev = ggml_backend_get_device(rt.cpu.backend);
    rt.cpu.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.cpu.backend));
    rt.cpu.name = "CPU";

    unsigned hw = std::thread::hardware_concurrency();
    rt.n_threads = hw > 0 ? (int)std::min<unsigned>(hw, 8) : 2;
    ggml_backend_cpu_set_n_threads(rt.cpu.backend, rt.n_threads);

    // Discover non-CPU backends (CUDA, Metal/MPS, Vulkan)
    size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if (dtype == GGML_BACKEND_DEVICE_TYPE_CPU) continue;

        const char * dname = ggml_backend_dev_name(dev);
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) continue;

        auto entry = std::make_unique<gt_backend_entry>();
        entry->dev = dev;
        entry->backend = b;
        entry->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(b));
        entry->name = dname ? dname : "GPU";

        if (str_contains_ci(dname, "CUDA") || str_contains_ci(dname, "HIP") || str_contains_ci(dname, "ROCm")) {
            rt.cuda.push_back(std::move(entry));
        } else if (str_contains_ci(dname, "Metal") || str_contains_ci(dname, "MTL")) {
            rt.mps.push_back(std::move(entry));
        } else if (str_contains_ci(dname, "Vulkan")) {
            rt.vulkan.push_back(std::move(entry));
        } else if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU) {
            rt.cuda.push_back(std::move(entry));
        }
    }

    rt.initialized = true;
}

gt_backend_entry * gt_get_backend(int device_type, int device_index) {
    gt_init();
    gt_runtime_state & rt = gt_rt();
    switch (device_type) {
        case GT_DEV_CPU:
            return &rt.cpu;
        case GT_DEV_CUDA:
            if (device_index >= 0 && device_index < (int)rt.cuda.size()) return rt.cuda[device_index].get();
            return nullptr;
        case GT_DEV_MPS:
            if (device_index >= 0 && device_index < (int)rt.mps.size()) return rt.mps[device_index].get();
            return nullptr;
        case GT_DEV_VULKAN:
            if (device_index >= 0 && device_index < (int)rt.vulkan.size()) return rt.vulkan[device_index].get();
            return nullptr;
        default:
            return nullptr;
    }
}

extern "C" void gt_set_num_threads(int n_threads) {
    gt_init();
    if (n_threads < 1) n_threads = 1;
    gt_runtime_state & rt = gt_rt();
    rt.n_threads = n_threads;
    ggml_backend_cpu_set_n_threads(rt.cpu.backend, n_threads);
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif
}

extern "C" int gt_get_num_threads(void) {
    gt_init();
    return gt_rt().n_threads;
}

extern "C" void gt_manual_seed(uint64_t seed) {
    gt_init();
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    rt.rng.seed(seed);
}

extern "C" int gt_rng_get_state(char * buf, int buf_len) {
    gt_init();
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::ostringstream oss;
    oss << rt.rng;
    std::string s = oss.str();
    if (!buf || buf_len <= 0) return (int)s.size();
    int n = std::min((int)s.size(), buf_len - 1);
    std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
    return (int)s.size();
}

extern "C" int gt_rng_set_state(const char * buf) {
    if (!buf) return -1;
    gt_init();
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::istringstream iss(buf);
    if (!(iss >> rt.rng)) {
        gt_set_error("Invalid RNG state");
        return -1;
    }
    return 0;
}

extern "C" int gt_device_count(int device_type) {
    gt_init();
    gt_runtime_state & rt = gt_rt();
    switch (device_type) {
        case GT_DEV_CPU:    return 1;
        case GT_DEV_CUDA:   return (int)rt.cuda.size();
        case GT_DEV_MPS:    return (int)rt.mps.size();
        case GT_DEV_VULKAN: return (int)rt.vulkan.size();
        case GT_DEV_META:   return 1;
        default:            return 0;
    }
}

extern "C" void gt_device_name(int device_type, int idx, char * buf, int buf_len) {
    if (!buf || buf_len <= 0) return;
    if (device_type == GT_DEV_META) {
        snprintf(buf, buf_len, "meta");
        return;
    }
    gt_backend_entry * be = gt_get_backend(device_type, idx);
    if (!be) {
        snprintf(buf, buf_len, "unavailable");
        return;
    }
    if (be->dev) {
        const char * desc = ggml_backend_dev_description(be->dev);
        snprintf(buf, buf_len, "%s", desc ? desc : be->name.c_str());
    } else {
        snprintf(buf, buf_len, "%s", be->name.c_str());
    }
}

extern "C" void gt_device_memory(int device_type, int idx, size_t * free_bytes, size_t * total_bytes) {
    size_t f = 0, t = 0;
    gt_backend_entry * be = gt_get_backend(device_type, idx);
    if (be && be->dev) {
        ggml_backend_dev_memory(be->dev, &f, &t);
    }
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
}

extern "C" void gt_synchronize(int device_type, int idx) {
    gt_backend_entry * be = gt_get_backend(device_type, idx);
    if (be && be->backend) {
        ggml_backend_synchronize(be->backend);
    }
}

extern "C" void gt_cpu_features(char * buf, int buf_len) {
    if (!buf || buf_len <= 0) return;
    gt_init();
    std::string s;
    auto add = [&](const char * name, int flag) {
        if (flag) {
            if (!s.empty()) s += " ";
            s += name;
        }
    };
    add("SSE3", ggml_cpu_has_sse3());
    add("SSSE3", ggml_cpu_has_ssse3());
    add("AVX", ggml_cpu_has_avx());
    add("AVX2", ggml_cpu_has_avx2());
    add("FMA", ggml_cpu_has_fma());
    add("F16C", ggml_cpu_has_f16c());
    add("BMI2", ggml_cpu_has_bmi2());
    add("AVX512", ggml_cpu_has_avx512());
    add("AVX512_VNNI", ggml_cpu_has_avx512_vnni());
    add("AVX512_BF16", ggml_cpu_has_avx512_bf16());
    add("AMX_INT8", ggml_cpu_has_amx_int8());
    add("NEON", ggml_cpu_has_neon());
    add("SVE", ggml_cpu_has_sve());
    add("LLAMAFILE", ggml_cpu_has_llamafile());
#ifdef _OPENMP
    add("OpenMP", 1);
#endif
    snprintf(buf, buf_len, "%s", s.c_str());
}

extern "C" void gt_backend_summary(char * buf, int buf_len) {
    if (!buf || buf_len <= 0) return;
    gt_init();
    gt_runtime_state & rt = gt_rt();
    std::string s = "CPU";
    if (!rt.cuda.empty()) s += ", CUDA(" + std::to_string(rt.cuda.size()) + ")";
    if (!rt.mps.empty()) s += ", MPS(" + std::to_string(rt.mps.size()) + ")";
    if (!rt.vulkan.empty()) s += ", Vulkan(" + std::to_string(rt.vulkan.size()) + ")";
    snprintf(buf, buf_len, "%s", s.c_str());
}

// ---------- Host view & contiguous helpers ----------

const void * gt_host_view::read(const gt_tensor * tensor) {
    t = tensor;
    if (!tensor || !tensor->storage || !tensor->storage->raw_tensor) return nullptr;
    if (tensor->storage->host_ptr) {
        rptr = tensor->storage->host_ptr;
        return rptr;
    }
    tmp.resize(tensor->storage->nbytes);
    ggml_backend_tensor_get(tensor->storage->raw_tensor, tmp.data(), 0, tensor->storage->nbytes);
    copied = true;
    rptr = tmp.data();
    return rptr;
}

void * gt_host_view::write_empty(gt_tensor * tensor) {
    t = tensor;
    if (!tensor || !tensor->storage || !tensor->storage->raw_tensor) return nullptr;
    if (tensor->storage->host_ptr) {
        wptr = tensor->storage->host_ptr;
        return wptr;
    }
    tmp.resize(tensor->storage->nbytes);
    copied = true;
    wptr = tmp.data();
    return wptr;
}

void * gt_host_view::read_write(gt_tensor * tensor) {
    t = tensor;
    if (!tensor || !tensor->storage || !tensor->storage->raw_tensor) return nullptr;
    if (tensor->storage->host_ptr) {
        wptr = tensor->storage->host_ptr;
        return wptr;
    }
    tmp.resize(tensor->storage->nbytes);
    ggml_backend_tensor_get(tensor->storage->raw_tensor, tmp.data(), 0, tensor->storage->nbytes);
    copied = true;
    wptr = tmp.data();
    return wptr;
}

void gt_host_view::commit(gt_tensor * tensor) {
    if (copied && tensor && tensor->storage && tensor->storage->raw_tensor && !tmp.empty()) {
        ggml_backend_tensor_set(tensor->storage->raw_tensor, tmp.data(), 0, tensor->storage->nbytes);
    }
}

gt_contig_owner::gt_contig_owner(const gt_tensor * t, int force_dtype) {
    if (!t) {
        ptr = nullptr;
        return;
    }
    bool need_copy = !gt_check_contiguous(t) || t->offset != 0 || t->device_type != GT_DEV_CPU ||
                     (force_dtype >= 0 && t->dtype != force_dtype);
    if (!need_copy) {
        ptr = t;
        return;
    }
    gt_tensor * tmp = nullptr;
    if (t->device_type != GT_DEV_CPU) {
        gt_tensor * cpu_t = gt_tensor_to_device(t, GT_DEV_CPU, 0);
        if (!cpu_t) return;
        if (!gt_check_contiguous(cpu_t) || cpu_t->offset != 0) {
            tmp = gt_tensor_contiguous(cpu_t);
            gt_tensor_free(cpu_t);
        } else {
            tmp = cpu_t;
        }
    } else if (!gt_check_contiguous(t) || t->offset != 0) {
        tmp = gt_tensor_contiguous(t);
    } else {
        tmp = gt_tensor_clone(t);
    }
    if (!tmp) return;
    if (force_dtype >= 0 && tmp->dtype != force_dtype) {
        gt_tensor * casted = gt_tensor_to_dtype(tmp, force_dtype);
        gt_tensor_free(tmp);
        tmp = casted;
    }
    owned = tmp;
    ptr = tmp;
}

// ---------- GGML wrap & graph execution ----------

struct ggml_tensor * gt_wrap_ggml_4d(
    struct ggml_context * ctx,
    const gt_tensor * t,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
{
    enum ggml_type gtype = gt_dtype_to_ggml(t->dtype);
    struct ggml_tensor * g = ggml_new_tensor_4d(ctx, gtype, ne0, ne1, ne2, ne3);
    size_t elem_sz = gt_elem_size(t->dtype);
    if (t->storage && t->storage->raw_tensor) {
        size_t byte_off = (elem_sz > 0) ? (size_t)t->offset * elem_sz : 0;
        g->data = (char *)t->storage->raw_tensor->data + byte_off;
        g->buffer = t->storage->buf;
    }
    return g;
}

bool gt_exec_ggml_graph(
    gt_backend_entry * be,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    struct ggml_tensor * out_node,
    gt_tensor * out_gt)
{
    (void)ctx;
    if (!be || !be->backend || !gf || !out_node || !out_gt) return false;

    std::lock_guard<std::mutex> lock(be->mtx);

    bool direct_out = (out_node->view_src == nullptr) &&
                      (out_node->data == nullptr) &&
                      (out_gt->storage && out_gt->storage->raw_tensor) &&
                      (out_node->type == out_gt->storage->raw_tensor->type) &&
                      (ggml_nelements(out_node) == out_gt->numel) &&
                      (out_gt->offset == 0);

    if (direct_out) {
        out_node->data = out_gt->storage->raw_tensor->data;
        out_node->buffer = out_gt->storage->buf;
    }

    int n_nodes = ggml_graph_n_nodes(gf);
    bool need_alloc = false;
    for (int i = 0; i < n_nodes; ++i) {
        if (ggml_graph_node(gf, i)->data == nullptr) {
            need_alloc = true;
            break;
        }
    }

    if (need_alloc) {
        if (!ggml_gallocr_reserve(be->galloc, gf)) {
            gt_set_error("ggml_gallocr_reserve failed");
            return false;
        }
        if (!ggml_gallocr_alloc_graph(be->galloc, gf)) {
            gt_set_error("ggml_gallocr_alloc_graph failed");
            return false;
        }
    }

    enum ggml_status st = ggml_backend_graph_compute(be->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        gt_set_error("ggml_backend_graph_compute failed");
        return false;
    }

    if (!direct_out && out_gt->storage && out_gt->storage->raw_tensor && out_gt->numel > 0) {
        if (out_node->buffer != nullptr && out_node->type == out_gt->storage->raw_tensor->type &&
            ggml_nelements(out_node) == out_gt->numel && out_gt->offset == 0) {
            ggml_backend_tensor_copy(out_node, out_gt->storage->raw_tensor);
        } else if (out_node->data != nullptr && out_gt->storage->host_ptr != nullptr &&
                   out_node->type == out_gt->storage->raw_tensor->type) {
            memcpy(out_gt->storage->host_ptr, out_node->data, ggml_nbytes(out_node));
        }
    }

    return true;
}

// ---------- Storage & Tensor Lifecycle ----------

static std::shared_ptr<gt_storage> gt_alloc_storage(int64_t numel, int dtype, int device_type, int device_index) {
    auto st = std::make_shared<gt_storage>();
    st->numel = numel;
    st->dtype = dtype;
    st->device_type = device_type;
    st->device_index = device_index;

    if (device_type == GT_DEV_META) {
        st->nbytes = 0;
        return st;
    }

    gt_backend_entry * be = gt_get_backend(device_type, device_index);
    if (!be) {
        gt_set_error("Requested backend device is not available");
        return nullptr;
    }

    int64_t alloc_elems = std::max<int64_t>(numel, 1);
    enum ggml_type gtype = gt_dtype_to_ggml(dtype);

    if (dtype == GT_C64) {
        alloc_elems *= 2;
        gtype = GGML_TYPE_F32;
    } else if (dtype == GT_C128) {
        alloc_elems *= 2;
        gtype = GGML_TYPE_F64;
    } else if (dtype == GT_U8 || dtype == GT_BOOL) {
        gtype = GGML_TYPE_I8;
    }

    if (gt_is_quantized(dtype)) {
        int64_t bs = ggml_blck_size(gtype);
        if (bs > 1 && alloc_elems % bs != 0) {
            alloc_elems = ((alloc_elems + bs - 1) / bs) * bs;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    st->ctx = ggml_init(params);
    if (!st->ctx) {
        gt_set_error("ggml_init failed for tensor storage");
        return nullptr;
    }

    st->raw_tensor = ggml_new_tensor_1d(st->ctx, gtype, alloc_elems);
    st->buf = ggml_backend_alloc_ctx_tensors(st->ctx, be->backend);
    if (!st->buf) {
        gt_set_error("ggml_backend_alloc_ctx_tensors failed (out of memory?)");
        return nullptr;
    }

    st->nbytes = ggml_nbytes(st->raw_tensor);
    if (ggml_backend_buffer_is_host(st->buf)) {
        st->host_ptr = st->raw_tensor->data;
    }
    return st;
}

extern "C" size_t gt_tensor_elem_size(int dtype) {
    return gt_elem_size(dtype);
}

extern "C" gt_tensor * gt_tensor_empty(int ndim, const int64_t * shape, int dtype, int device_type, int device_index) {
    gt_init();
    if (ndim < 0 || ndim > GT_MAX_DIMS) {
        gt_set_error("Invalid ndim");
        return nullptr;
    }
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] < 0) {
            gt_set_error("Negative dimension size");
            return nullptr;
        }
    }

    auto * t = new gt_tensor();
    t->ndim = ndim;
    for (int i = 0; i < ndim; ++i) t->shape[i] = shape[i];
    gt_default_strides(ndim, t->shape, t->strides);
    t->offset = 0;
    t->numel = gt_compute_numel(ndim, t->shape);
    t->dtype = dtype;
    t->device_type = device_type;
    t->device_index = device_index;
    t->storage = gt_alloc_storage(t->numel, dtype, device_type, device_index);
    if (!t->storage) {
        delete t;
        return nullptr;
    }
    return t;
}

extern "C" gt_tensor * gt_tensor_from_data(int ndim, const int64_t * shape, int dtype, int device_type, int device_index, const void * data, size_t nbytes) {
    gt_tensor * t = gt_tensor_empty(ndim, shape, dtype, device_type, device_index);
    if (!t) return nullptr;
    if (data && nbytes > 0 && t->numel > 0 && device_type != GT_DEV_META) {
        if (gt_tensor_copy_from_host(t, data, nbytes) != 0) {
            gt_tensor_free(t);
            return nullptr;
        }
    }
    return t;
}

extern "C" gt_tensor * gt_tensor_clone(const gt_tensor * t) {
    if (!t) return nullptr;
    gt_tensor * out = gt_tensor_empty(t->ndim, t->shape, t->dtype, t->device_type, t->device_index);
    if (!out) return nullptr;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return out;

    if (gt_check_contiguous(t) && t->offset == 0 && t->storage && out->storage &&
        t->storage->raw_tensor && out->storage->raw_tensor &&
        t->storage->nbytes == out->storage->nbytes) {
        if (t->storage->host_ptr && out->storage->host_ptr) {
            memcpy(out->storage->host_ptr, t->storage->host_ptr, out->storage->nbytes);
        } else {
            ggml_backend_tensor_copy(t->storage->raw_tensor, out->storage->raw_tensor);
        }
        return out;
    }

    if (gt_tensor_copy_inplace(out, t) != 0) {
        gt_tensor_free(out);
        return nullptr;
    }
    return out;
}

extern "C" gt_tensor * gt_tensor_view(const gt_tensor * t, int ndim, const int64_t * shape, const int64_t * strides, int64_t offset) {
    if (!t || ndim < 0 || ndim > GT_MAX_DIMS) {
        gt_set_error("Invalid view arguments");
        return nullptr;
    }
    auto * out = new gt_tensor();
    out->ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        out->shape[i] = shape[i];
        out->strides[i] = strides[i];
    }
    out->offset = offset;
    out->numel = gt_compute_numel(ndim, out->shape);
    out->dtype = t->dtype;
    out->device_type = t->device_type;
    out->device_index = t->device_index;
    out->storage = t->storage;
    return out;
}

extern "C" gt_tensor * gt_tensor_contiguous(const gt_tensor * t) {
    if (!t) return nullptr;
    if (gt_check_contiguous(t) && t->offset == 0) {
        // Return a view sharing storage (just like PyTorch contiguous() when already contiguous!)
        return gt_tensor_view(t, t->ndim, t->shape, t->strides, t->offset);
    }
    gt_tensor * out = gt_tensor_empty(t->ndim, t->shape, t->dtype, t->device_type, t->device_index);
    if (!out) return nullptr;
    if (gt_tensor_copy_inplace(out, t) != 0) {
        gt_tensor_free(out);
        return nullptr;
    }
    return out;
}

extern "C" gt_tensor * gt_tensor_to_dtype(const gt_tensor * t, int dtype) {
    if (!t) return nullptr;
    if (t->dtype == dtype) return gt_tensor_clone(t);

    gt_tensor * out = gt_tensor_empty(t->ndim, t->shape, dtype, t->device_type, t->device_index);
    if (!out) return nullptr;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return out;

    // Quantization / Dequantization via ggml
    if (gt_is_quantized(t->dtype) || gt_is_quantized(dtype)) {
        gt_contig_owner src_cpu(t, -1);
        if (!src_cpu.ptr) { gt_tensor_free(out); return nullptr; }
        gt_host_view hv_in;
        const void * in_p = hv_in.read(src_cpu.ptr);

        std::vector<float> f32_buf(t->numel);
        if (gt_is_quantized(t->dtype)) {
            enum ggml_type gt_src = gt_dtype_to_ggml(t->dtype);
            const auto * traits = ggml_get_type_traits(gt_src);
            if (!traits || !traits->to_float) {
                gt_set_error("Unsupported source quantization type");
                gt_tensor_free(out);
                return nullptr;
            }
            traits->to_float(in_p, f32_buf.data(), t->numel);
        } else {
            for (int64_t i = 0; i < t->numel; ++i) {
                f32_buf[i] = (float)gt_read_f64_at(in_p, t->dtype, i);
            }
        }

        gt_host_view hv_out;
        void * out_p = hv_out.write_empty(out);
        if (gt_is_quantized(dtype)) {
            enum ggml_type gt_dst = gt_dtype_to_ggml(dtype);
            int64_t row_len = (t->ndim >= 1) ? std::max<int64_t>(t->shape[t->ndim - 1], 1) : t->numel;
            int64_t nrows = t->numel / row_len;
            int64_t bs = ggml_blck_size(gt_dst);
            if (row_len % bs != 0) {
                // Fallback single flat chunk if divisible by blck_size
                if (t->numel % bs != 0) {
                    gt_set_error("Tensor size must be a multiple of ggml quantization block size");
                    gt_tensor_free(out);
                    return nullptr;
                }
                row_len = t->numel;
                nrows = 1;
            }
            ggml_quantize_chunk(gt_dst, f32_buf.data(), out_p, 0, nrows, row_len, nullptr);
        } else {
            for (int64_t i = 0; i < t->numel; ++i) {
                gt_write_f64_at(out_p, dtype, i, (double)f32_buf[i]);
            }
        }
        hv_out.commit(out);
        return out;
    }

    if (gt_tensor_copy_inplace(out, t) != 0) {
        gt_tensor_free(out);
        return nullptr;
    }
    return out;
}

extern "C" gt_tensor * gt_tensor_to_device(const gt_tensor * t, int device_type, int device_index) {
    if (!t) return nullptr;
    if (t->device_type == device_type && t->device_index == device_index) {
        return gt_tensor_clone(t);
    }
    gt_contig_owner src_contig(t);
    if (!src_contig.ptr) return nullptr;

    gt_tensor * out = gt_tensor_empty(t->ndim, t->shape, t->dtype, device_type, device_index);
    if (!out) return nullptr;
    if (t->numel == 0 || device_type == GT_DEV_META || t->device_type == GT_DEV_META) {
        return out;
    }

    ggml_backend_tensor_copy(src_contig.ptr->storage->raw_tensor, out->storage->raw_tensor);
    return out;
}

extern "C" void gt_tensor_free(gt_tensor * t) {
    delete t;
}

extern "C" int gt_tensor_ndim(const gt_tensor * t) { return t ? t->ndim : 0; }
extern "C" void gt_tensor_shape(const gt_tensor * t, int64_t * out) {
    if (!t || !out) return;
    for (int i = 0; i < t->ndim; ++i) out[i] = t->shape[i];
}
extern "C" void gt_tensor_strides(const gt_tensor * t, int64_t * out) {
    if (!t || !out) return;
    for (int i = 0; i < t->ndim; ++i) out[i] = t->strides[i];
}
extern "C" int64_t gt_tensor_offset(const gt_tensor * t) { return t ? t->offset : 0; }
extern "C" int64_t gt_tensor_numel(const gt_tensor * t) { return t ? t->numel : 0; }
extern "C" int gt_tensor_dtype(const gt_tensor * t) { return t ? t->dtype : GT_F32; }
extern "C" int gt_tensor_device_type(const gt_tensor * t) { return t ? t->device_type : GT_DEV_CPU; }
extern "C" int gt_tensor_device_index(const gt_tensor * t) { return t ? t->device_index : 0; }
extern "C" int gt_tensor_is_contiguous(const gt_tensor * t) { return (t && gt_check_contiguous(t)) ? 1 : 0; }
extern "C" uint64_t gt_tensor_data_ptr(const gt_tensor * t) {
    if (!t || !t->storage || !t->storage->raw_tensor) return 0;
    size_t es = gt_elem_size(t->dtype);
    return (uint64_t)((uintptr_t)t->storage->raw_tensor->data + (size_t)t->offset * es);
}
extern "C" uint64_t gt_tensor_storage_ptr(const gt_tensor * t) {
    if (!t || !t->storage) return 0;
    return (uint64_t)(uintptr_t)t->storage.get();
}
extern "C" size_t gt_tensor_nbytes(const gt_tensor * t) {
    if (!t) return 0;
    if (gt_is_quantized(t->dtype)) {
        return t->storage ? t->storage->nbytes : 0;
    }
    return (size_t)t->numel * gt_elem_size(t->dtype);
}

extern "C" int gt_tensor_copy_to_host(const gt_tensor * t, void * dst, size_t dst_bytes) {
    if (!t || !dst) return -1;
    if (t->numel == 0) return 0;
    if (t->device_type == GT_DEV_META) {
        gt_set_error("Cannot copy from meta tensor");
        return -1;
    }

    if (gt_is_quantized(t->dtype)) {
        size_t nb = t->storage->nbytes;
        if (dst_bytes < nb) return -1;
        gt_host_view hv;
        const void * p = hv.read(t);
        memcpy(dst, p, nb);
        return 0;
    }

    size_t es = gt_elem_size(t->dtype);
    if (dst_bytes < (size_t)t->numel * es) return -1;

    gt_host_view hv;
    const void * src_base = hv.read(t);
    if (gt_check_contiguous(t)) {
        memcpy(dst, (const uint8_t *)src_base + (size_t)t->offset * es, (size_t)t->numel * es);
        return 0;
    }

    uint8_t * out_b = (uint8_t *)dst;
    const uint8_t * in_b = (const uint8_t *)src_base;
    for (int64_t i = 0; i < t->numel; ++i) {
        int64_t off = gt_linear_to_elem_offset(t, i);
        memcpy(out_b + (size_t)i * es, in_b + (size_t)off * es, es);
    }
    return 0;
}

extern "C" int gt_tensor_copy_from_host(gt_tensor * t, const void * src, size_t src_bytes) {
    if (!t || !src) return -1;
    if (t->numel == 0) return 0;
    if (t->device_type == GT_DEV_META) {
        gt_set_error("Cannot copy to meta tensor");
        return -1;
    }
    if (gt_is_quantized(t->dtype)) {
        size_t nb = std::min(src_bytes, t->storage->nbytes);
        if (t->storage->host_ptr) {
            memcpy(t->storage->host_ptr, src, nb);
        } else {
            ggml_backend_tensor_set(t->storage->raw_tensor, src, 0, nb);
        }
        return 0;
    }

    size_t es = gt_elem_size(t->dtype);
    if (src_bytes < (size_t)t->numel * es) return -1;

    if (gt_check_contiguous(t) && t->offset == 0) {
        if (t->storage->host_ptr) {
            memcpy(t->storage->host_ptr, src, (size_t)t->numel * es);
        } else {
            ggml_backend_tensor_set(t->storage->raw_tensor, src, 0, (size_t)t->numel * es);
        }
        return 0;
    }

    gt_host_view hv;
    uint8_t * dst_b = (uint8_t *)hv.read_write(t);
    const uint8_t * in_b = (const uint8_t *)src;
    for (int64_t i = 0; i < t->numel; ++i) {
        int64_t off = gt_linear_to_elem_offset(t, i);
        memcpy(dst_b + (size_t)off * es, in_b + (size_t)i * es, es);
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_tensor_copy_inplace(gt_tensor * dst, const gt_tensor * src) {
    if (!dst || !src) return -1;
    if (dst->numel == 0 || src->numel == 0) return 0;
    if (dst->device_type == GT_DEV_META || src->device_type == GT_DEV_META) return 0;

    if (dst->numel != src->numel) {
        // Support broadcast copy when src is broadcastable to dst
        // Check shape compatibility from right
        int nd = dst->ndim;
        if (src->ndim > nd) {
            gt_set_error("copy_: src has more dimensions than dst");
            return -1;
        }
        for (int i = 0; i < src->ndim; ++i) {
            int64_t ds = dst->shape[nd - 1 - i];
            int64_t ss = src->shape[src->ndim - 1 - i];
            if (ss != ds && ss != 1) {
                gt_set_error("copy_: incompatible broadcast shape");
                return -1;
            }
        }
    }

    if (dst->dtype == src->dtype && gt_check_contiguous(dst) && gt_check_contiguous(src) &&
        dst->offset == 0 && src->offset == 0 && dst->numel == src->numel && !gt_is_quantized(dst->dtype)) {
        if (dst->storage->host_ptr && src->storage->host_ptr) {
            memcpy(dst->storage->host_ptr, src->storage->host_ptr, (size_t)dst->numel * gt_elem_size(dst->dtype));
        } else {
            ggml_backend_tensor_copy(src->storage->raw_tensor, dst->storage->raw_tensor);
        }
        return 0;
    }

    gt_host_view hv_s, hv_d;
    const void * sp = hv_s.read(src);
    void * dp = hv_d.read_write(dst);

    int nd = dst->ndim;
    int snd = src->ndim;
    int64_t src_bstrides[GT_MAX_DIMS] = {};
    for (int i = 0; i < nd; ++i) {
        int si = snd - nd + i;
        if (si < 0 || src->shape[si] == 1) {
            src_bstrides[i] = 0;
        } else {
            src_bstrides[i] = src->strides[si];
        }
    }

    bool same_dtype = (dst->dtype == src->dtype);
    size_t es = gt_elem_size(dst->dtype);
    bool use_complex = gt_is_complex(dst->dtype) || gt_is_complex(src->dtype);
    bool use_i64 = !use_complex && (dst->dtype == GT_I64 || src->dtype == GT_I64) && !gt_is_floating(dst->dtype) && !gt_is_floating(src->dtype);

    #pragma omp parallel for if (dst->numel > 4096)
    for (int64_t idx = 0; idx < dst->numel; ++idx) {
        int64_t d_off = dst->offset;
        int64_t s_off = src->offset;
        int64_t rem = idx;
        for (int d = nd - 1; d >= 0; --d) {
            int64_t c = rem % dst->shape[d];
            rem /= dst->shape[d];
            d_off += c * dst->strides[d];
            s_off += c * src_bstrides[d];
        }
        if (same_dtype && es > 0) {
            memcpy((uint8_t *)dp + (size_t)d_off * es, (const uint8_t *)sp + (size_t)s_off * es, es);
        } else if (use_complex) {
            gt_write_c128_at(dp, dst->dtype, d_off, gt_read_c128_at(sp, src->dtype, s_off));
        } else if (use_i64) {
            gt_write_i64_at(dp, dst->dtype, d_off, gt_read_i64_at(sp, src->dtype, s_off));
        } else {
            gt_write_f64_at(dp, dst->dtype, d_off, gt_read_f64_at(sp, src->dtype, s_off));
        }
    }

    hv_d.commit(dst);
    return 0;
}

extern "C" double gt_tensor_item_f64(const gt_tensor * t) {
    if (!t || t->numel != 1) {
        gt_set_error("a Tensor with more or fewer than 1 element cannot be converted to Scalar");
        return 0.0;
    }
    gt_host_view hv;
    const void * p = hv.read(t);
    return gt_read_f64_at(p, t->dtype, t->offset);
}

extern "C" int64_t gt_tensor_item_i64(const gt_tensor * t) {
    if (!t || t->numel != 1) {
        gt_set_error("a Tensor with more or fewer than 1 element cannot be converted to Scalar");
        return 0;
    }
    gt_host_view hv;
    const void * p = hv.read(t);
    return gt_read_i64_at(p, t->dtype, t->offset);
}

// ---------- Creation / Fill / RNG ----------

extern "C" int gt_fill_scalar_f64(gt_tensor * t, double val) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_host_view hv;
    void * p = hv.read_write(t);
    if (gt_check_contiguous(t)) {
        int64_t off = t->offset;
        #pragma omp parallel for if (t->numel > 8192)
        for (int64_t i = 0; i < t->numel; ++i) {
            gt_write_f64_at(p, t->dtype, off + i, val);
        }
    } else {
        #pragma omp parallel for if (t->numel > 8192)
        for (int64_t i = 0; i < t->numel; ++i) {
            gt_write_f64_at(p, t->dtype, gt_linear_to_elem_offset(t, i), val);
        }
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_scalar_i64(gt_tensor * t, int64_t val) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_host_view hv;
    void * p = hv.read_write(t);
    if (gt_check_contiguous(t)) {
        int64_t off = t->offset;
        #pragma omp parallel for if (t->numel > 8192)
        for (int64_t i = 0; i < t->numel; ++i) {
            gt_write_i64_at(p, t->dtype, off + i, val);
        }
    } else {
        #pragma omp parallel for if (t->numel > 8192)
        for (int64_t i = 0; i < t->numel; ++i) {
            gt_write_i64_at(p, t->dtype, gt_linear_to_elem_offset(t, i), val);
        }
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_arange(gt_tensor * t, double start, double step) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_host_view hv;
    void * p = hv.write_empty(t);
    bool use_i64 = gt_is_integer(t->dtype) && (std::floor(start) == start) && (std::floor(step) == step);
    for (int64_t i = 0; i < t->numel; ++i) {
        if (use_i64) {
            gt_write_i64_at(p, t->dtype, i, (int64_t)start + i * (int64_t)step);
        } else {
            gt_write_f64_at(p, t->dtype, i, start + (double)i * step);
        }
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_linspace(gt_tensor * t, double start, double end, int64_t steps) {
    if (!t) return -1;
    if (steps <= 0 || t->device_type == GT_DEV_META) return 0;
    gt_host_view hv;
    void * p = hv.write_empty(t);
    if (steps == 1) {
        gt_write_f64_at(p, t->dtype, 0, start);
    } else {
        double step = (end - start) / (double)(steps - 1);
        for (int64_t i = 0; i < steps; ++i) {
            double v = (i == steps - 1) ? end : (start + (double)i * step);
            gt_write_f64_at(p, t->dtype, i, v);
        }
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_eye(gt_tensor * t, int64_t n, int64_t m) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_host_view hv;
    void * p = hv.write_empty(t);
    for (int64_t i = 0; i < n * m; ++i) {
        gt_write_f64_at(p, t->dtype, i, 0.0);
    }
    int64_t diag = std::min(n, m);
    for (int64_t i = 0; i < diag; ++i) {
        gt_write_f64_at(p, t->dtype, i * m + i, 1.0);
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_uniform(gt_tensor * t, double low, double high) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::uniform_real_distribution<double> dist(low, high);
    gt_host_view hv;
    void * p = hv.read_write(t);
    bool contig = gt_check_contiguous(t);
    for (int64_t i = 0; i < t->numel; ++i) {
        int64_t idx = contig ? (t->offset + i) : gt_linear_to_elem_offset(t, i);
        gt_write_f64_at(p, t->dtype, idx, dist(rt.rng));
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_normal(gt_tensor * t, double mean, double std) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::normal_distribution<double> dist(mean, std > 0.0 ? std : 1e-30);
    gt_host_view hv;
    void * p = hv.read_write(t);
    bool contig = gt_check_contiguous(t);
    for (int64_t i = 0; i < t->numel; ++i) {
        int64_t idx = contig ? (t->offset + i) : gt_linear_to_elem_offset(t, i);
        double v = (std == 0.0) ? mean : dist(rt.rng);
        gt_write_f64_at(p, t->dtype, idx, v);
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_trunc_normal(gt_tensor * t, double mean, double std, double a, double b) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::normal_distribution<double> dist(mean, std > 0.0 ? std : 1e-30);
    gt_host_view hv;
    void * p = hv.read_write(t);
    bool contig = gt_check_contiguous(t);
    for (int64_t i = 0; i < t->numel; ++i) {
        double v = mean;
        if (std > 0.0) {
            for (int tries = 0; tries < 1000; ++tries) {
                v = dist(rt.rng);
                if (v >= a && v <= b) break;
            }
            if (v < a) v = a;
            if (v > b) v = b;
        }
        int64_t idx = contig ? (t->offset + i) : gt_linear_to_elem_offset(t, i);
        gt_write_f64_at(p, t->dtype, idx, v);
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_randint(gt_tensor * t, int64_t low, int64_t high) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    if (high <= low) {
        gt_set_error("randint: high must be > low");
        return -1;
    }
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::uniform_int_distribution<int64_t> dist(low, high - 1);
    gt_host_view hv;
    void * p = hv.read_write(t);
    bool contig = gt_check_contiguous(t);
    for (int64_t i = 0; i < t->numel; ++i) {
        int64_t idx = contig ? (t->offset + i) : gt_linear_to_elem_offset(t, i);
        gt_write_i64_at(p, t->dtype, idx, dist(rt.rng));
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_randperm(gt_tensor * t, int64_t n) {
    if (!t) return -1;
    if (n <= 0 || t->device_type == GT_DEV_META) return 0;
    std::vector<int64_t> vals(n);
    for (int64_t i = 0; i < n; ++i) vals[i] = i;
    {
        gt_runtime_state & rt = gt_rt();
        std::lock_guard<std::mutex> lock(rt.rng_mtx);
        std::shuffle(vals.begin(), vals.end(), rt.rng);
    }
    gt_host_view hv;
    void * p = hv.write_empty(t);
    for (int64_t i = 0; i < n; ++i) {
        gt_write_i64_at(p, t->dtype, i, vals[i]);
    }
    hv.commit(t);
    return 0;
}

extern "C" int gt_fill_bernoulli_prob(gt_tensor * t, double p_prob) {
    if (!t) return -1;
    if (t->numel == 0 || t->device_type == GT_DEV_META) return 0;
    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::bernoulli_distribution dist(std::clamp(p_prob, 0.0, 1.0));
    gt_host_view hv;
    void * p = hv.read_write(t);
    bool contig = gt_check_contiguous(t);
    for (int64_t i = 0; i < t->numel; ++i) {
        int64_t idx = contig ? (t->offset + i) : gt_linear_to_elem_offset(t, i);
        gt_write_f64_at(p, t->dtype, idx, dist(rt.rng) ? 1.0 : 0.0);
    }
    hv.commit(t);
    return 0;
}

extern "C" gt_tensor * gt_bernoulli_tensor(const gt_tensor * probs) {
    if (!probs) return nullptr;
    gt_tensor * out = gt_tensor_empty(probs->ndim, probs->shape, probs->dtype, probs->device_type, probs->device_index);
    if (!out) return nullptr;
    if (probs->numel == 0 || probs->device_type == GT_DEV_META) return out;

    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    gt_host_view hv_in, hv_out;
    const void * sp = hv_in.read(probs);
    void * dp = hv_out.write_empty(out);
    bool contig = gt_check_contiguous(probs);
    for (int64_t i = 0; i < probs->numel; ++i) {
        int64_t sidx = contig ? (probs->offset + i) : gt_linear_to_elem_offset(probs, i);
        double pr = gt_read_f64_at(sp, probs->dtype, sidx);
        gt_write_f64_at(dp, out->dtype, i, (u01(rt.rng) < pr) ? 1.0 : 0.0);
    }
    hv_out.commit(out);
    return out;
}

extern "C" gt_tensor * gt_multinomial(const gt_tensor * probs, int64_t num_samples, int replacement) {
    if (!probs || probs->ndim < 1 || probs->ndim > 2) {
        gt_set_error("multinomial expects 1D or 2D probs tensor");
        return nullptr;
    }
    int64_t rows = (probs->ndim == 1) ? 1 : probs->shape[0];
    int64_t cols = (probs->ndim == 1) ? probs->shape[0] : probs->shape[1];
    if (!replacement && num_samples > cols) {
        gt_set_error("multinomial: cannot sample more than number of categories without replacement");
        return nullptr;
    }

    int64_t out_shape[2] = {rows, num_samples};
    gt_tensor * out = gt_tensor_empty(probs->ndim, (probs->ndim == 1) ? &num_samples : out_shape, GT_I64, probs->device_type, probs->device_index);
    if (!out) return nullptr;

    gt_contig_owner src_cpu(probs);
    gt_host_view hv_in, hv_out;
    const void * sp = hv_in.read(src_cpu.ptr);
    void * dp = hv_out.write_empty(out);

    gt_runtime_state & rt = gt_rt();
    std::lock_guard<std::mutex> lock(rt.rng_mtx);

    std::vector<double> w(cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            double val = gt_read_f64_at(sp, src_cpu.ptr->dtype, r * cols + c);
            w[c] = (val > 0.0) ? val : 0.0;
        }
        for (int64_t k = 0; k < num_samples; ++k) {
            double total = 0.0;
            for (int64_t c = 0; c < cols; ++c) total += w[c];
            int64_t picked = 0;
            if (total > 0.0) {
                std::uniform_real_distribution<double> u(0.0, total);
                double target = u(rt.rng);
                double acc = 0.0;
                for (int64_t c = 0; c < cols; ++c) {
                    acc += w[c];
                    if (acc >= target) { picked = c; break; }
                }
            }
            gt_write_i64_at(dp, GT_I64, r * num_samples + k, picked);
            if (!replacement) w[picked] = 0.0;
        }
    }
    hv_out.commit(out);
    return out;
}
