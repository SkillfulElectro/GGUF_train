#include "gt_internal.h"
#include <algorithm>
#include <limits>

// ---------- MatMul / Linear (ggml_mul_mat) ----------

static gt_tensor * gt_matmul_2d_ggml(const gt_tensor * a_in, const gt_tensor * b_in) {
    // a_in: (..., M, K), b_in: (..., K, N) -> out: (..., M, N)
    // ggml_mul_mat(ga, gb) computes ga^T * gb where ga has ne={K, N, ...}, gb has ne={K, M, ...}
    // Wait: in ggml_mul_mat(ga, gb), result has ne={ga->ne[1], gb->ne[1], ...} = {N, M, ...} -> PyTorch (..., M, N)!
    int nd_a = a_in->ndim;
    int nd_b = b_in->ndim;
    int M = (int)a_in->shape[nd_a - 2];
    int K = (int)a_in->shape[nd_a - 1];
    int K2 = (int)b_in->shape[nd_b - 2];
    int N = (int)b_in->shape[nd_b - 1];
    if (K != K2) {
        gt_set_error("mat1 and mat2 shapes cannot be multiplied");
        return nullptr;
    }

    int batch_ndim = std::max(nd_a - 2, nd_b - 2);
    int out_ndim = batch_ndim + 2;
    int64_t out_shape[GT_MAX_DIMS] = {};
    for (int i = 0; i < batch_ndim; ++i) {
        int ai = (nd_a - 2) - 1 - i;
        int bi = (nd_b - 2) - 1 - i;
        int64_t da = (ai >= 0) ? a_in->shape[ai] : 1;
        int64_t db = (bi >= 0) ? b_in->shape[bi] : 1;
        if (da != db && da != 1 && db != 1) {
            gt_set_error("matmul batch dimensions are not broadcastable");
            return nullptr;
        }
        out_shape[batch_ndim - 1 - i] = std::max(da, db);
    }
    out_shape[out_ndim - 2] = M;
    out_shape[out_ndim - 1] = N;

    int out_dtype = gt_promote_dtype(a_in->dtype, b_in->dtype);
    if (gt_is_quantized(a_in->dtype) || gt_is_quantized(b_in->dtype)) out_dtype = GT_F32;

    gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, out_dtype, a_in->device_type, a_in->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;
    if (K == 0) {
        // (M, 0) @ (0, N): every element is an empty sum. Answered here - ggml takes a tensor
        // whose first dimension is 0 for a transposed one and aborts the process.
        gt_fill_scalar_f64(out, 0.0);
        return out;
    }

    // Native ggml_mul_mat path for F32/F16/BF16/Quantized without complex batch broadcasting
    int64_t batch_a = 1, batch_b = 1, batch_out = 1;
    for (int i = 0; i < nd_a - 2; ++i) batch_a *= a_in->shape[i];
    for (int i = 0; i < nd_b - 2; ++i) batch_b *= b_in->shape[i];
    for (int i = 0; i < batch_ndim; ++i) batch_out *= out_shape[i];

    bool can_ggml = (a_in->dtype == GT_F32 || gt_is_quantized(a_in->dtype) || a_in->dtype == GT_F16 || a_in->dtype == GT_BF16) &&
                    (b_in->dtype == GT_F32 || gt_is_quantized(b_in->dtype) || b_in->dtype == GT_F16 || b_in->dtype == GT_BF16) &&
                    (batch_a == batch_out && (batch_b == batch_out || batch_b == 1));

    if (can_ggml && a_in->device_type == b_in->device_type && a_in->device_index == b_in->device_index) {
        gt_backend_entry * be = gt_get_backend(out->device_type, out->device_index);
        // Prepare contiguous a (F32) and b transposed (N, K) contiguous
        const gt_tensor * a_use = a_in;
        gt_tensor * a_tmp = nullptr;
        if (!gt_check_contiguous(a_in) || a_in->offset != 0 || (a_in->dtype != GT_F32)) {
            gt_tensor * c = gt_tensor_contiguous(a_in);
            if (c->dtype != GT_F32) {
                a_tmp = gt_tensor_to_dtype(c, GT_F32);
                gt_tensor_free(c);
            } else {
                a_tmp = c;
            }
            a_use = a_tmp;
        }

        // Transpose b last 2 dims -> bT shape (..., N, K) contiguous
        gt_tensor * b_f32 = (b_in->dtype == GT_F32 && gt_check_contiguous(b_in) && b_in->offset == 0)
                            ? nullptr
                            : (gt_is_quantized(b_in->dtype) ? gt_tensor_to_dtype(b_in, GT_F32) : gt_tensor_to_dtype(b_in, GT_F32));
        const gt_tensor * b_src = b_f32 ? b_f32 : b_in;
        gt_tensor * b_src_c = gt_check_contiguous(b_src) ? nullptr : gt_tensor_contiguous(b_src);
        const gt_tensor * b_c = b_src_c ? b_src_c : b_src;

        int64_t bt_shape[GT_MAX_DIMS];
        int64_t bt_strides[GT_MAX_DIMS];
        for (int i = 0; i < b_c->ndim; ++i) {
            bt_shape[i] = b_c->shape[i];
            bt_strides[i] = b_c->strides[i];
        }
        std::swap(bt_shape[b_c->ndim - 2], bt_shape[b_c->ndim - 1]);
        std::swap(bt_strides[b_c->ndim - 2], bt_strides[b_c->ndim - 1]);
        gt_tensor * b_view = gt_tensor_view(b_c, b_c->ndim, bt_shape, bt_strides, b_c->offset);
        gt_tensor * bT = gt_tensor_contiguous(b_view);
        gt_tensor_free(b_view);

        gt_tensor * out_f32 = (out_dtype == GT_F32) ? out : gt_tensor_empty(out_ndim, out_shape, GT_F32, out->device_type, out->device_index);

        struct ggml_init_params params = {
            ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
        };
        struct ggml_context * ctx = ggml_init(params);
        if (be && ctx && a_use && bT && out_f32) {
            struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, bT, K, N, batch_b, 1);
            struct ggml_tensor * gb = gt_wrap_ggml_4d(ctx, a_use, K, M, batch_a, 1);
            struct ggml_tensor * gout = ggml_mul_mat(ctx, ga, gb);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, gout);
            bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out_f32);
            ggml_free(ctx);
            if (ok) {
                if (out_f32 != out) {
                    gt_tensor_copy_inplace(out, out_f32);
                    gt_tensor_free(out_f32);
                }
                gt_tensor_free(bT);
                if (b_src_c) gt_tensor_free(b_src_c);
                if (b_f32) gt_tensor_free(b_f32);
                if (a_tmp) gt_tensor_free(a_tmp);
                return out;
            }
        } else {
            if (ctx) ggml_free(ctx);
        }
        if (out_f32 && out_f32 != out) gt_tensor_free(out_f32);
        gt_tensor_free(bT);
        if (b_src_c) gt_tensor_free(b_src_c);
        if (b_f32) gt_tensor_free(b_f32);
        if (a_tmp) gt_tensor_free(a_tmp);
    }

    // General fallback (F64, I32, I64, complex, arbitrary batch broadcasting)
    gt_host_view hva, hvb, hvo;
    const void * ap = hva.read(a_in);
    const void * bp = hvb.read(b_in);
    void * op = hvo.write_empty(out);

    int64_t a_bstrides[GT_MAX_DIMS] = {}, b_bstrides[GT_MAX_DIMS] = {};
    for (int i = 0; i < batch_ndim; ++i) {
        int ai = (nd_a - 2) - batch_ndim + i;
        int bi = (nd_b - 2) - batch_ndim + i;
        a_bstrides[i] = (ai < 0 || a_in->shape[ai] == 1) ? 0 : a_in->strides[ai];
        b_bstrides[i] = (bi < 0 || b_in->shape[bi] == 1) ? 0 : b_in->strides[bi];
    }

    int64_t a_sm = a_in->strides[nd_a - 2], a_sk = a_in->strides[nd_a - 1];
    int64_t b_sk = b_in->strides[nd_b - 2], b_sn = b_in->strides[nd_b - 1];

    bool use_c = gt_is_complex(out->dtype);
    bool use_i64 = !use_c && gt_is_integer(out->dtype);

    #pragma omp parallel for collapse(2) if (batch_out * M * N > 4096)
    for (int64_t b = 0; b < batch_out; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            int64_t a_boff = a_in->offset;
            int64_t b_boff = b_in->offset;
            int64_t rem = b;
            for (int d = batch_ndim - 1; d >= 0; --d) {
                int64_t c = rem % out_shape[d];
                rem /= out_shape[d];
                a_boff += c * a_bstrides[d];
                b_boff += c * b_bstrides[d];
            }
            for (int64_t n = 0; n < N; ++n) {
                int64_t oidx = (b * M + m) * N + n;
                if (use_c) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int64_t k = 0; k < K; ++k) {
                        acc += gt_read_c128_at(ap, a_in->dtype, a_boff + m * a_sm + k * a_sk) *
                               gt_read_c128_at(bp, b_in->dtype, b_boff + k * b_sk + n * b_sn);
                    }
                    gt_write_c128_at(op, out->dtype, oidx, acc);
                } else if (use_i64) {
                    int64_t acc = 0;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += gt_read_i64_at(ap, a_in->dtype, a_boff + m * a_sm + k * a_sk) *
                               gt_read_i64_at(bp, b_in->dtype, b_boff + k * b_sk + n * b_sn);
                    }
                    gt_write_i64_at(op, out->dtype, oidx, acc);
                } else {
                    double acc = 0.0;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += gt_read_f64_at(ap, a_in->dtype, a_boff + m * a_sm + k * a_sk) *
                               gt_read_f64_at(bp, b_in->dtype, b_boff + k * b_sk + n * b_sn);
                    }
                    gt_write_f64_at(op, out->dtype, oidx, acc);
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_matmul(const gt_tensor * a, const gt_tensor * b) {
    if (!a || !b) return nullptr;
    if (a->ndim == 0 || b->ndim == 0) {
        gt_set_error("both arguments to matmul need to be at least 1D");
        return nullptr;
    }
    if (a->ndim == 1 && b->ndim == 1) {
        if (a->shape[0] != b->shape[0]) {
            gt_set_error("dot: inconsistent vector lengths");
            return nullptr;
        }
        int64_t s_a[2] = {1, a->shape[0]}, st_a[2] = {a->shape[0] * a->strides[0], a->strides[0]};
        int64_t s_b[2] = {b->shape[0], 1}, st_b[2] = {b->strides[0], b->strides[0]};
        gt_tensor * va = gt_tensor_view(a, 2, s_a, st_a, a->offset);
        gt_tensor * vb = gt_tensor_view(b, 2, s_b, st_b, b->offset);
        gt_tensor * r2 = gt_matmul_2d_ggml(va, vb);
        gt_tensor_free(va); gt_tensor_free(vb);
        if (!r2) return nullptr;
        gt_tensor * r0 = gt_tensor_view(r2, 0, nullptr, nullptr, 0);
        gt_tensor_free(r2);
        return r0;
    }
    if (a->ndim == 1 && b->ndim >= 2) {
        int64_t s_a[2] = {1, a->shape[0]}, st_a[2] = {a->shape[0] * a->strides[0], a->strides[0]};
        gt_tensor * va = gt_tensor_view(a, 2, s_a, st_a, a->offset);
        gt_tensor * r = gt_matmul_2d_ggml(va, b);
        gt_tensor_free(va);
        if (!r) return nullptr;
        int nd = r->ndim;
        int64_t new_sh[GT_MAX_DIMS], new_st[GT_MAX_DIMS];
        int k = 0;
        for (int i = 0; i < nd; ++i) {
            if (i == nd - 2) continue;
            new_sh[k] = r->shape[i];
            new_st[k] = r->strides[i];
            ++k;
        }
        gt_tensor * res = gt_tensor_view(r, k, new_sh, new_st, r->offset);
        gt_tensor_free(r);
        return res;
    }
    if (a->ndim >= 2 && b->ndim == 1) {
        int64_t s_b[2] = {b->shape[0], 1}, st_b[2] = {b->strides[0], b->strides[0]};
        gt_tensor * vb = gt_tensor_view(b, 2, s_b, st_b, b->offset);
        gt_tensor * r = gt_matmul_2d_ggml(a, vb);
        gt_tensor_free(vb);
        if (!r) return nullptr;
        int nd = r->ndim;
        int64_t new_sh[GT_MAX_DIMS], new_st[GT_MAX_DIMS];
        for (int i = 0; i < nd - 1; ++i) {
            new_sh[i] = r->shape[i];
            new_st[i] = r->strides[i];
        }
        gt_tensor * res = gt_tensor_view(r, nd - 1, new_sh, new_st, r->offset);
        gt_tensor_free(r);
        return res;
    }
    return gt_matmul_2d_ggml(a, b);
}

extern "C" gt_tensor * gt_linear(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias) {
    // input: (..., in_features), weight: (out_features, in_features), bias: (out_features,) or NULL
    if (!input || !weight || weight->ndim != 2 || input->ndim < 1) {
        gt_set_error("linear: invalid input or weight dimensions");
        return nullptr;
    }
    int64_t out_features = weight->shape[0];
    int64_t in_features = weight->shape[1];
    if (input->shape[input->ndim - 1] != in_features) {
        gt_set_error("linear: matmul shape mismatch");
        return nullptr;
    }

    int64_t batch_rows = input->numel / std::max<int64_t>(in_features, 1);
    int64_t out_shape[GT_MAX_DIMS];
    for (int i = 0; i < input->ndim - 1; ++i) out_shape[i] = input->shape[i];
    out_shape[input->ndim - 1] = out_features;

    // Direct ggml_mul_mat(weight, input) — zero weight transpose! Also supports quantized GGUF weights!
    bool use_ggml = (input->dtype == GT_F32) &&
                    (weight->dtype == GT_F32 || weight->dtype == GT_F16 || weight->dtype == GT_BF16 || gt_is_quantized(weight->dtype)) &&
                    (input->device_type == weight->device_type && input->device_index == weight->device_index) &&
                    (!bias || (bias->dtype == GT_F32 && bias->device_type == input->device_type));

    if (use_ggml && input->numel > 0 && input->device_type != GT_DEV_META) {
        gt_backend_entry * be = gt_get_backend(input->device_type, input->device_index);
        gt_tensor * in_c = (gt_check_contiguous(input) && input->offset == 0) ? nullptr : gt_tensor_contiguous(input);
        const gt_tensor * in_use = in_c ? in_c : input;
        gt_tensor * w_c = (gt_check_contiguous(weight) && weight->offset == 0) ? nullptr : gt_tensor_contiguous(weight);
        const gt_tensor * w_use = w_c ? w_c : weight;
        gt_tensor * b_c = (bias && (!gt_check_contiguous(bias) || bias->offset != 0)) ? gt_tensor_contiguous(bias) : nullptr;
        const gt_tensor * b_use = b_c ? b_c : bias;

        gt_tensor * out = gt_tensor_empty(input->ndim, out_shape, GT_F32, input->device_type, input->device_index);
        if (be && out) {
            struct ggml_init_params params = {
                ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
            };
            struct ggml_context * ctx = ggml_init(params);
            if (ctx) {
                struct ggml_tensor * gw = gt_wrap_ggml_4d(ctx, w_use, in_features, out_features, 1, 1);
                struct ggml_tensor * gx = gt_wrap_ggml_4d(ctx, in_use, in_features, batch_rows, 1, 1);
                struct ggml_tensor * gout = ggml_mul_mat(ctx, gw, gx);
                if (b_use) {
                    struct ggml_tensor * gb = gt_wrap_ggml_4d(ctx, b_use, out_features, 1, 1, 1);
                    gout = ggml_add(ctx, gout, gb);
                }
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, gout);
                bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                ggml_free(ctx);
                if (in_c) gt_tensor_free(in_c);
                if (w_c) gt_tensor_free(w_c);
                if (b_c) gt_tensor_free(b_c);
                if (ok) return out;
            }
            gt_tensor_free(out);
        }
        if (in_c) gt_tensor_free(in_c);
        if (w_c) gt_tensor_free(w_c);
        if (b_c) gt_tensor_free(b_c);
    }

    // Fallback via weight^T
    gt_tensor * w_f32 = gt_is_quantized(weight->dtype) ? gt_tensor_to_dtype(weight, GT_F32) : nullptr;
    const gt_tensor * w_ref = w_f32 ? w_f32 : weight;
    int64_t wt_sh[2] = {w_ref->shape[1], w_ref->shape[0]};
    int64_t wt_st[2] = {w_ref->strides[1], w_ref->strides[0]};
    gt_tensor * wT = gt_tensor_view(w_ref, 2, wt_sh, wt_st, w_ref->offset);
    gt_tensor * mm = gt_matmul(input, wT);
    gt_tensor_free(wT);
    if (w_f32) gt_tensor_free(w_f32);
    if (!mm) return nullptr;
    if (bias) {
        gt_tensor * res = gt_binary_op(mm, bias, GT_BIN_ADD, 1.0);
        gt_tensor_free(mm);
        return res;
    }
    return mm;
}

extern "C" gt_tensor * gt_addmm(const gt_tensor * input, const gt_tensor * mat1, const gt_tensor * mat2, double beta, double alpha) {
    gt_tensor * mm = gt_matmul(mat1, mat2);
    if (!mm) return nullptr;
    if (std::abs(alpha - 1.0) > 1e-12) {
        gt_tensor * scaled = gt_binary_scalar_op(mm, alpha, 0, GT_BIN_MUL);
        gt_tensor_free(mm);
        mm = scaled;
        if (!mm) return nullptr;
    }
    if (beta == 0.0) return mm;
    gt_tensor * inp_part = (std::abs(beta - 1.0) > 1e-12) ? gt_binary_scalar_op(input, beta, 0, GT_BIN_MUL) : gt_tensor_clone(input);
    if (!inp_part) { gt_tensor_free(mm); return nullptr; }
    gt_tensor * res = gt_binary_op(inp_part, mm, GT_BIN_ADD, 1.0);
    gt_tensor_free(inp_part);
    gt_tensor_free(mm);
    return res;
}

extern "C" gt_tensor * gt_baddbmm(const gt_tensor * input, const gt_tensor * b1, const gt_tensor * b2, double beta, double alpha) {
    return gt_addmm(input, b1, b2, beta, alpha);
}

extern "C" gt_tensor * gt_outer(const gt_tensor * a, const gt_tensor * b) {
    if (!a || !b || a->ndim != 1 || b->ndim != 1) {
        gt_set_error("outer expects two 1D tensors");
        return nullptr;
    }
    int64_t ash[2] = {a->shape[0], 1}, ast[2] = {a->strides[0], 1};
    int64_t bsh[2] = {1, b->shape[0]}, bst[2] = {b->shape[0] * b->strides[0], b->strides[0]};
    gt_tensor * va = gt_tensor_view(a, 2, ash, ast, a->offset);
    gt_tensor * vb = gt_tensor_view(b, 2, bsh, bst, b->offset);
    gt_tensor * out = gt_matmul(va, vb);
    gt_tensor_free(va);
    gt_tensor_free(vb);
    return out;
}

// ---------- Softmax ----------

extern "C" gt_tensor * gt_softmax(const gt_tensor * a, int dim, int is_log) {
    if (!a) return nullptr;
    int nd = a->ndim;
    if (nd == 0) {
        gt_tensor * out = gt_tensor_empty(0, nullptr, a->dtype, a->device_type, a->device_index);
        if (out) gt_fill_scalar_f64(out, is_log ? 0.0 : 1.0);
        return out;
    }
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        gt_set_error("softmax: dim out of range");
        return nullptr;
    }

    // Native ggml_soft_max fast path when dim == last dim, F32, contiguous, <= 4D
    if (!is_log && dim == nd - 1 && a->dtype == GT_F32 && nd <= 4 &&
        gt_ggml_inplace_ok(a) && a->numel > 0 && a->device_type != GT_DEV_META) {
        gt_backend_entry * be = gt_get_backend(a->device_type, a->device_index);
        gt_tensor * out = gt_tensor_empty(nd, a->shape, GT_F32, a->device_type, a->device_index);
        if (be && out) {
            int64_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < nd; ++i) ne[i] = a->shape[nd - 1 - i];
            struct ggml_init_params params = {
                ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
            };
            struct ggml_context * ctx = ggml_init(params);
            if (ctx) {
                struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, ne[0], ne[1], ne[2], ne[3]);
                struct ggml_tensor * gout = ggml_soft_max(ctx, ga);
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, gout);
                bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                ggml_free(ctx);
                if (ok) return out;
            }
            gt_tensor_free(out);
        }
    }

    gt_tensor * out = gt_tensor_empty(nd, a->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (a->numel == 0 || a->device_type == GT_DEV_META) return out;

    gt_host_view hva, hvo;
    const void * ap = hva.read(a);
    void * op = hvo.write_empty(out);

    int64_t dim_len = a->shape[dim];
    int64_t outer = a->numel / dim_len;
    std::vector<int> other_dims;
    for (int d = 0; d < nd; ++d) if (d != dim) other_dims.push_back(d);

    #pragma omp parallel for if (outer > 128)
    for (int64_t o = 0; o < outer; ++o) {
        int64_t base_in = a->offset, base_out = 0, rem = o;
        for (int k = (int)other_dims.size() - 1; k >= 0; --k) {
            int d = other_dims[k];
            int64_t c = rem % a->shape[d];
            rem /= a->shape[d];
            base_in += c * a->strides[d];
            base_out += c * out->strides[d];
        }
        double mx = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < dim_len; ++j) {
            double v = gt_read_f64_at(ap, a->dtype, base_in + j * a->strides[dim]);
            if (v > mx) mx = v;
        }
        double sum_exp = 0.0;
        for (int64_t j = 0; j < dim_len; ++j) {
            double e = std::exp(gt_read_f64_at(ap, a->dtype, base_in + j * a->strides[dim]) - mx);
            sum_exp += e;
            if (!is_log) gt_write_f64_at(op, out->dtype, base_out + j * out->strides[dim], e);
        }
        if (is_log) {
            double lse = mx + std::log(sum_exp);
            for (int64_t j = 0; j < dim_len; ++j) {
                double v = gt_read_f64_at(ap, a->dtype, base_in + j * a->strides[dim]);
                gt_write_f64_at(op, out->dtype, base_out + j * out->strides[dim], v - lse);
            }
        } else {
            double inv_s = 1.0 / sum_exp;
            for (int64_t j = 0; j < dim_len; ++j) {
                int64_t off = base_out + j * out->strides[dim];
                gt_write_f64_at(op, out->dtype, off, gt_read_f64_at(op, out->dtype, off) * inv_s);
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_softmax_backward(const gt_tensor * grad_out, const gt_tensor * out, int dim, int is_log) {
    if (!grad_out || !out) return nullptr;
    int nd = out->ndim;
    if (nd == 0) {
        gt_tensor * dx = gt_tensor_empty(0, nullptr, out->dtype, out->device_type, out->device_index);
        if (dx) gt_fill_scalar_f64(dx, 0.0);
        return dx;
    }
    if (dim < 0) dim += nd;
    gt_tensor * dx = gt_tensor_empty(nd, out->shape, out->dtype, out->device_type, out->device_index);
    if (!dx) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return dx;

    gt_host_view hvg, hvo, hvd;
    const void * gp = hvg.read(grad_out);
    const void * op = hvo.read(out);
    void * dp = hvd.write_empty(dx);

    int64_t dim_len = out->shape[dim];
    int64_t outer = out->numel / dim_len;
    std::vector<int> other_dims;
    for (int d = 0; d < nd; ++d) if (d != dim) other_dims.push_back(d);

    #pragma omp parallel for if (outer > 128)
    for (int64_t o = 0; o < outer; ++o) {
        int64_t bg = grad_out->offset, bo = out->offset, bd = 0, rem = o;
        for (int k = (int)other_dims.size() - 1; k >= 0; --k) {
            int d = other_dims[k];
            int64_t c = rem % out->shape[d];
            rem /= out->shape[d];
            bg += c * grad_out->strides[d];
            bo += c * out->strides[d];
            bd += c * dx->strides[d];
        }
        if (is_log) {
            double sg = 0.0;
            for (int64_t j = 0; j < dim_len; ++j) {
                sg += gt_read_f64_at(gp, grad_out->dtype, bg + j * grad_out->strides[dim]);
            }
            for (int64_t j = 0; j < dim_len; ++j) {
                double g = gt_read_f64_at(gp, grad_out->dtype, bg + j * grad_out->strides[dim]);
                double p = std::exp(gt_read_f64_at(op, out->dtype, bo + j * out->strides[dim]));
                gt_write_f64_at(dp, dx->dtype, bd + j * dx->strides[dim], g - p * sg);
            }
        } else {
            double dot = 0.0;
            for (int64_t j = 0; j < dim_len; ++j) {
                dot += gt_read_f64_at(gp, grad_out->dtype, bg + j * grad_out->strides[dim]) *
                       gt_read_f64_at(op, out->dtype, bo + j * out->strides[dim]);
            }
            for (int64_t j = 0; j < dim_len; ++j) {
                double g = gt_read_f64_at(gp, grad_out->dtype, bg + j * grad_out->strides[dim]);
                double y = gt_read_f64_at(op, out->dtype, bo + j * out->strides[dim]);
                gt_write_f64_at(dp, dx->dtype, bd + j * dx->strides[dim], y * (g - dot));
            }
        }
    }
    hvd.commit(dx);
    return dx;
}

// ---------- RMSNorm, LayerNorm, GroupNorm, BatchNorm ----------

extern "C" gt_tensor * gt_rms_norm(const gt_tensor * x, const gt_tensor * weight, float eps) {
    if (!x || x->ndim < 1) { gt_set_error("rms_norm: invalid input"); return nullptr; }
    int64_t D = x->shape[x->ndim - 1];
    gt_tensor * out = gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index);
    if (!out) return nullptr;
    if (x->numel == 0 || x->device_type == GT_DEV_META) return out;

    // Native ggml_rms_norm fast path
    if (x->dtype == GT_F32 && gt_check_contiguous(x) && x->offset == 0 && x->ndim <= 4 &&
        (!weight || (weight->dtype == GT_F32 && gt_check_contiguous(weight) && weight->offset == 0 && weight->numel == D))) {
        gt_backend_entry * be = gt_get_backend(x->device_type, x->device_index);
        if (be) {
            int64_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < x->ndim; ++i) ne[i] = x->shape[x->ndim - 1 - i];
            struct ggml_init_params params = {
                ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
            };
            struct ggml_context * ctx = ggml_init(params);
            if (ctx) {
                struct ggml_tensor * gx = gt_wrap_ggml_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
                struct ggml_tensor * gout = ggml_rms_norm(ctx, gx, eps);
                if (weight) {
                    struct ggml_tensor * gw = gt_wrap_ggml_4d(ctx, weight, D, 1, 1, 1);
                    gout = ggml_mul(ctx, gout, gw);
                }
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, gout);
                bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                ggml_free(ctx);
                if (ok) return out;
            }
        }
    }

    gt_contig_owner xc(x), wc(weight);
    gt_host_view hvx, hvw, hvo;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    void * op = hvo.write_empty(out);
    int64_t rows = x->numel / std::max<int64_t>(D, 1);

    #pragma omp parallel for if (rows > 64)
    for (int64_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int64_t c = 0; c < D; ++c) {
            double v = gt_read_f64_at(xp, xc.ptr->dtype, r * D + c);
            ss += v * v;
        }
        double inv_rms = 1.0 / std::sqrt(ss / (double)D + (double)eps);
        for (int64_t c = 0; c < D; ++c) {
            double v = gt_read_f64_at(xp, xc.ptr->dtype, r * D + c) * inv_rms;
            if (wp) v *= gt_read_f64_at(wp, wc.ptr->dtype, c);
            gt_write_f64_at(op, out->dtype, r * D + c, v);
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_rms_norm_backward(const gt_tensor * grad_out, const gt_tensor * x, const gt_tensor * weight, float eps, gt_tensor ** out_dweight) {
    if (!grad_out || !x) return nullptr;
    int64_t D = x->shape[x->ndim - 1];
    int64_t rows = x->numel / std::max<int64_t>(D, 1);

    gt_tensor * dx = gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index);
    if (!dx) return nullptr;
    gt_tensor * dw = nullptr;
    if (weight && out_dweight) {
        dw = gt_tensor_empty(weight->ndim, weight->shape, weight->dtype, weight->device_type, weight->device_index);
        if (!dw) { gt_tensor_free(dx); return nullptr; }
        gt_fill_scalar_f64(dw, 0.0);
    }

    gt_contig_owner gc(grad_out), xc(x), wc(weight);
    gt_host_view hvg, hvx, hvw, hvd, hvdw;
    const void * gp = hvg.read(gc.ptr);
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    void * dxp = hvd.write_empty(dx);
    std::vector<double> dw_acc(dw ? D : 0, 0.0);

    for (int64_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int64_t c = 0; c < D; ++c) {
            double v = gt_read_f64_at(xp, xc.ptr->dtype, r * D + c);
            ss += v * v;
        }
        double rms = std::sqrt(ss / (double)D + (double)eps);
        double inv_rms = 1.0 / rms;
        double dot_gx = 0.0;
        for (int64_t c = 0; c < D; ++c) {
            double g = gt_read_f64_at(gp, gc.ptr->dtype, r * D + c);
            double xv = gt_read_f64_at(xp, xc.ptr->dtype, r * D + c);
            double w = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
            if (dw) dw_acc[c] += g * xv * inv_rms;
            dot_gx += (g * w) * xv;
        }
        double coeff = (dot_gx / (double)D) * (inv_rms * inv_rms);
        for (int64_t c = 0; c < D; ++c) {
            double g = gt_read_f64_at(gp, gc.ptr->dtype, r * D + c);
            double xv = gt_read_f64_at(xp, xc.ptr->dtype, r * D + c);
            double w = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
            double res = inv_rms * (g * w - xv * coeff);
            gt_write_f64_at(dxp, dx->dtype, r * D + c, res);
        }
    }
    hvd.commit(dx);
    if (dw) {
        void * dwp = hvdw.write_empty(dw);
        for (int64_t c = 0; c < D; ++c) gt_write_f64_at(dwp, dw->dtype, c, dw_acc[c]);
        hvdw.commit(dw);
        *out_dweight = dw;
    }
    return dx;
}

extern "C" gt_tensor * gt_layer_norm(const gt_tensor * x, int norm_ndim, const gt_tensor * weight, const gt_tensor * bias, float eps) {
    if (!x || norm_ndim < 1 || norm_ndim > x->ndim) {
        gt_set_error("layer_norm: invalid normalized_shape dims");
        return nullptr;
    }
    int64_t norm_size = 1;
    for (int i = x->ndim - norm_ndim; i < x->ndim; ++i) norm_size *= x->shape[i];
    int64_t rows = x->numel / std::max<int64_t>(norm_size, 1);

    gt_tensor * out = gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index);
    if (!out) return nullptr;
    if (x->numel == 0 || x->device_type == GT_DEV_META) return out;

    // Native ggml_norm fast path when norm_ndim == 1 and F32 contiguous
    if (norm_ndim == 1 && x->dtype == GT_F32 && gt_check_contiguous(x) && x->offset == 0 && x->ndim <= 4 &&
        (!weight || (weight->dtype == GT_F32 && gt_check_contiguous(weight) && weight->offset == 0)) &&
        (!bias || (bias->dtype == GT_F32 && gt_check_contiguous(bias) && bias->offset == 0))) {
        gt_backend_entry * be = gt_get_backend(x->device_type, x->device_index);
        if (be) {
            int64_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < x->ndim; ++i) ne[i] = x->shape[x->ndim - 1 - i];
            struct ggml_init_params params = {
                ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
            };
            struct ggml_context * ctx = ggml_init(params);
            if (ctx) {
                struct ggml_tensor * gx = gt_wrap_ggml_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
                struct ggml_tensor * gout = ggml_norm(ctx, gx, eps);
                if (weight) {
                    struct ggml_tensor * gw = gt_wrap_ggml_4d(ctx, weight, norm_size, 1, 1, 1);
                    gout = ggml_mul(ctx, gout, gw);
                }
                if (bias) {
                    struct ggml_tensor * gb = gt_wrap_ggml_4d(ctx, bias, norm_size, 1, 1, 1);
                    gout = ggml_add(ctx, gout, gb);
                }
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, gout);
                bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                ggml_free(ctx);
                if (ok) return out;
            }
        }
    }

    gt_contig_owner xc(x), wc(weight), bc(bias);
    gt_host_view hvx, hvw, hvb, hvo;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    const void * bp = bc.ptr ? hvb.read(bc.ptr) : nullptr;
    void * op = hvo.write_empty(out);

    #pragma omp parallel for if (rows > 64)
    for (int64_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int64_t c = 0; c < norm_size; ++c) mean += gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c);
        mean /= (double)norm_size;
        double var = 0.0;
        for (int64_t c = 0; c < norm_size; ++c) {
            double d = gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c) - mean;
            var += d * d;
        }
        double inv_std = 1.0 / std::sqrt(var / (double)norm_size + (double)eps);
        for (int64_t c = 0; c < norm_size; ++c) {
            double xn = (gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c) - mean) * inv_std;
            if (wp) xn *= gt_read_f64_at(wp, wc.ptr->dtype, c);
            if (bp) xn += gt_read_f64_at(bp, bc.ptr->dtype, c);
            gt_write_f64_at(op, out->dtype, r * norm_size + c, xn);
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_layer_norm_backward(const gt_tensor * grad_out, const gt_tensor * x, int norm_ndim, const gt_tensor * weight, float eps, gt_tensor ** dx, gt_tensor ** dw, gt_tensor ** db) {
    if (!grad_out || !x) return -1;
    int64_t norm_size = 1;
    for (int i = x->ndim - norm_ndim; i < x->ndim; ++i) norm_size *= x->shape[i];
    int64_t rows = x->numel / std::max<int64_t>(norm_size, 1);

    gt_tensor * t_dx = dx ? gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index) : nullptr;
    gt_tensor * t_dw = (dw && weight) ? gt_tensor_empty(weight->ndim, weight->shape, weight->dtype, weight->device_type, weight->device_index) : nullptr;
    gt_tensor * t_db = db ? gt_tensor_empty(norm_ndim, x->shape + (x->ndim - norm_ndim), x->dtype, x->device_type, x->device_index) : nullptr;

    gt_contig_owner gc(grad_out), xc(x), wc(weight);
    gt_host_view hvg, hvx, hvw, hvdx, hvdw, hvdb;
    const void * gp = hvg.read(gc.ptr);
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    void * dxp = t_dx ? hvdx.write_empty(t_dx) : nullptr;

    std::vector<double> dw_acc(t_dw ? norm_size : 0, 0.0);
    std::vector<double> db_acc(t_db ? norm_size : 0, 0.0);

    for (int64_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int64_t c = 0; c < norm_size; ++c) mean += gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c);
        mean /= (double)norm_size;
        double var = 0.0;
        for (int64_t c = 0; c < norm_size; ++c) {
            double d = gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c) - mean;
            var += d * d;
        }
        double inv_std = 1.0 / std::sqrt(var / (double)norm_size + (double)eps);

        double sum_dxhat = 0.0;
        double sum_dxhat_xhat = 0.0;
        for (int64_t c = 0; c < norm_size; ++c) {
            double g = gt_read_f64_at(gp, gc.ptr->dtype, r * norm_size + c);
            double xhat = (gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c) - mean) * inv_std;
            if (t_dw) dw_acc[c] += g * xhat;
            if (t_db) db_acc[c] += g;
            double dxhat = wp ? g * gt_read_f64_at(wp, wc.ptr->dtype, c) : g;
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
        }
        if (dxp) {
            double inv_n = 1.0 / (double)norm_size;
            for (int64_t c = 0; c < norm_size; ++c) {
                double g = gt_read_f64_at(gp, gc.ptr->dtype, r * norm_size + c);
                double xhat = (gt_read_f64_at(xp, xc.ptr->dtype, r * norm_size + c) - mean) * inv_std;
                double dxhat = wp ? g * gt_read_f64_at(wp, wc.ptr->dtype, c) : g;
                double val = inv_std * inv_n * ((double)norm_size * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
                gt_write_f64_at(dxp, t_dx->dtype, r * norm_size + c, val);
            }
        }
    }
    if (t_dx) { hvdx.commit(t_dx); *dx = t_dx; }
    if (t_dw) {
        void * p = hvdw.write_empty(t_dw);
        for (int64_t c = 0; c < norm_size; ++c) gt_write_f64_at(p, t_dw->dtype, c, dw_acc[c]);
        hvdw.commit(t_dw);
        *dw = t_dw;
    }
    if (t_db) {
        void * p = hvdb.write_empty(t_db);
        for (int64_t c = 0; c < norm_size; ++c) gt_write_f64_at(p, t_db->dtype, c, db_acc[c]);
        hvdb.commit(t_db);
        *db = t_db;
    }
    return 0;
}

extern "C" gt_tensor * gt_group_norm(const gt_tensor * x, int num_groups, const gt_tensor * weight, const gt_tensor * bias, float eps) {
    if (!x || x->ndim < 2) { gt_set_error("group_norm expects >= 2D tensor"); return nullptr; }
    int64_t N = x->shape[0];
    int64_t C = x->shape[1];
    int64_t spatial = x->numel / std::max<int64_t>(N * C, 1);
    if (C % num_groups != 0) { gt_set_error("group_norm: C must be divisible by num_groups"); return nullptr; }

    gt_tensor * out = gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index);
    if (!out) return nullptr;
    if (x->numel == 0 || x->device_type == GT_DEV_META) return out;

    gt_contig_owner xc(x), wc(weight), bc(bias);
    gt_host_view hvx, hvw, hvb, hvo;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    const void * bp = bc.ptr ? hvb.read(bc.ptr) : nullptr;
    void * op = hvo.write_empty(out);

    int64_t cpg = C / num_groups;
    int64_t gsize = cpg * spatial;

    #pragma omp parallel for collapse(2) if (N * num_groups > 32)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            double mean = 0.0;
            for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c) {
                int64_t base = (n * C + c) * spatial;
                for (int64_t s = 0; s < spatial; ++s) mean += gt_read_f64_at(xp, xc.ptr->dtype, base + s);
            }
            mean /= (double)gsize;
            double var = 0.0;
            for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c) {
                int64_t base = (n * C + c) * spatial;
                for (int64_t s = 0; s < spatial; ++s) {
                    double d = gt_read_f64_at(xp, xc.ptr->dtype, base + s) - mean;
                    var += d * d;
                }
            }
            double inv_std = 1.0 / std::sqrt(var / (double)gsize + (double)eps);
            for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c) {
                double w = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
                double b = bp ? gt_read_f64_at(bp, bc.ptr->dtype, c) : 0.0;
                int64_t base = (n * C + c) * spatial;
                for (int64_t s = 0; s < spatial; ++s) {
                    double xn = (gt_read_f64_at(xp, xc.ptr->dtype, base + s) - mean) * inv_std;
                    gt_write_f64_at(op, out->dtype, base + s, xn * w + b);
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_batch_norm(const gt_tensor * x, const gt_tensor * weight, const gt_tensor * bias,
                                      gt_tensor * running_mean, gt_tensor * running_var,
                                      int training, double momentum, double eps,
                                      gt_tensor ** save_mean, gt_tensor ** save_invstd) {
    if (!x || x->ndim < 2) { gt_set_error("batch_norm expects >= 2D tensor"); return nullptr; }
    int64_t N = x->shape[0];
    int64_t C = x->shape[1];
    int64_t spatial = x->numel / std::max<int64_t>(N * C, 1);
    int64_t count = N * spatial;
    if ((running_mean && running_mean->numel != C) || (running_var && running_var->numel != C)) {
        gt_set_error("batch_norm: running_mean and running_var must have one element per channel");
        return nullptr;
    }

    gt_tensor * out = gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index);
    gt_tensor * sm = gt_tensor_empty(1, &C, GT_F32, x->device_type, x->device_index);
    gt_tensor * si = gt_tensor_empty(1, &C, GT_F32, x->device_type, x->device_index);
    if (!out || !sm || !si) { gt_tensor_free(out); gt_tensor_free(sm); gt_tensor_free(si); return nullptr; }

    gt_contig_owner xc(x), wc(weight), bc(bias);
    gt_host_view hvx, hvw, hvb, hvo, hvrm, hvrv, hvsm, hvsi;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    const void * bp = bc.ptr ? hvb.read(bc.ptr) : nullptr;
    void * op = hvo.write_empty(out);
    void * rmp = running_mean ? hvrm.read_write(running_mean) : nullptr;
    void * rvp = running_var ? hvrv.read_write(running_var) : nullptr;
    void * smp = hvsm.write_empty(sm);
    void * sip = hvsi.write_empty(si);
    // The running statistics are read and updated in place, so they are addressed through
    // their own offset and strides: a buffer that is a slice of a larger tensor must neither
    // be read from, nor written to, the start of that tensor.
    auto rm_at = [&](int64_t c) { return gt_linear_to_elem_offset(running_mean, c); };
    auto rv_at = [&](int64_t c) { return gt_linear_to_elem_offset(running_var, c); };

    #pragma omp parallel for if (C > 16)
    for (int64_t c = 0; c < C; ++c) {
        double mean = 0.0, var = 0.0;
        if (training || !rmp || !rvp) {
            for (int64_t n = 0; n < N; ++n) {
                int64_t base = (n * C + c) * spatial;
                for (int64_t s = 0; s < spatial; ++s) mean += gt_read_f64_at(xp, xc.ptr->dtype, base + s);
            }
            mean /= (double)std::max<int64_t>(count, 1);
            for (int64_t n = 0; n < N; ++n) {
                int64_t base = (n * C + c) * spatial;
                for (int64_t s = 0; s < spatial; ++s) {
                    double d = gt_read_f64_at(xp, xc.ptr->dtype, base + s) - mean;
                    var += d * d;
                }
            }
            double biased_var = var / (double)std::max<int64_t>(count, 1);
            double unbiased_var = (count > 1) ? (var / (double)(count - 1)) : biased_var;
            if (training && rmp && rvp) {
                double old_m = gt_read_f64_at(rmp, running_mean->dtype, rm_at(c));
                double old_v = gt_read_f64_at(rvp, running_var->dtype, rv_at(c));
                gt_write_f64_at(rmp, running_mean->dtype, rm_at(c), (1.0 - momentum) * old_m + momentum * mean);
                gt_write_f64_at(rvp, running_var->dtype, rv_at(c), (1.0 - momentum) * old_v + momentum * unbiased_var);
            }
            var = biased_var;
        } else {
            mean = gt_read_f64_at(rmp, running_mean->dtype, rm_at(c));
            var  = gt_read_f64_at(rvp, running_var->dtype, rv_at(c));
        }
        double inv_std = 1.0 / std::sqrt(var + eps);
        gt_write_f64_at(smp, GT_F32, c, mean);
        gt_write_f64_at(sip, GT_F32, c, inv_std);

        double w = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
        double b = bp ? gt_read_f64_at(bp, bc.ptr->dtype, c) : 0.0;
        for (int64_t n = 0; n < N; ++n) {
            int64_t base = (n * C + c) * spatial;
            for (int64_t s = 0; s < spatial; ++s) {
                double xn = (gt_read_f64_at(xp, xc.ptr->dtype, base + s) - mean) * inv_std;
                gt_write_f64_at(op, out->dtype, base + s, xn * w + b);
            }
        }
    }

    hvo.commit(out);
    hvsm.commit(sm);
    hvsi.commit(si);
    if (running_mean) hvrm.commit(running_mean);
    if (running_var) hvrv.commit(running_var);
    if (save_mean) *save_mean = sm; else gt_tensor_free(sm);
    if (save_invstd) *save_invstd = si; else gt_tensor_free(si);
    return out;
}

extern "C" int gt_batch_norm_backward(const gt_tensor * grad_out, const gt_tensor * x, const gt_tensor * weight,
                                      const gt_tensor * save_mean, const gt_tensor * save_invstd,
                                      const gt_tensor * running_mean, const gt_tensor * running_var,
                                      int training, double eps, gt_tensor ** dx, gt_tensor ** dw, gt_tensor ** db) {
    (void)running_mean; (void)running_var; (void)eps;
    if (!grad_out || !x || !save_mean || !save_invstd) return -1;
    int64_t N = x->shape[0];
    int64_t C = x->shape[1];
    int64_t spatial = x->numel / std::max<int64_t>(N * C, 1);
    int64_t count = N * spatial;

    gt_tensor * t_dx = dx ? gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index) : nullptr;
    gt_tensor * t_dw = (dw && weight) ? gt_tensor_empty(1, &C, weight->dtype, x->device_type, x->device_index) : nullptr;
    gt_tensor * t_db = db ? gt_tensor_empty(1, &C, x->dtype, x->device_type, x->device_index) : nullptr;

    gt_contig_owner gc(grad_out), xc(x), wc(weight), smc(save_mean), sic(save_invstd);
    gt_host_view hvg, hvx, hvw, hvsm, hvsi, hvdx, hvdw, hvdb;
    const void * gp = hvg.read(gc.ptr);
    const void * xp = hvx.read(xc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    const void * smp = hvsm.read(smc.ptr);
    const void * sip = hvsi.read(sic.ptr);
    void * dxp = t_dx ? hvdx.write_empty(t_dx) : nullptr;
    void * dwp = t_dw ? hvdw.write_empty(t_dw) : nullptr;
    void * dbp = t_db ? hvdb.write_empty(t_db) : nullptr;

    #pragma omp parallel for if (C > 16)
    for (int64_t c = 0; c < C; ++c) {
        double mean = gt_read_f64_at(smp, smc.ptr->dtype, c);
        double inv_std = gt_read_f64_at(sip, sic.ptr->dtype, c);
        double w = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;

        double sum_dy = 0.0;
        double sum_dy_xhat = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            int64_t base = (n * C + c) * spatial;
            for (int64_t s = 0; s < spatial; ++s) {
                double dy = gt_read_f64_at(gp, gc.ptr->dtype, base + s);
                double xhat = (gt_read_f64_at(xp, xc.ptr->dtype, base + s) - mean) * inv_std;
                sum_dy += dy;
                sum_dy_xhat += dy * xhat;
            }
        }
        if (dwp) gt_write_f64_at(dwp, t_dw->dtype, c, sum_dy_xhat);
        if (dbp) gt_write_f64_at(dbp, t_db->dtype, c, sum_dy);
        if (dxp) {
            for (int64_t n = 0; n < N; ++n) {
                int64_t base = (n * C + c) * spatial;
                for (int64_t s = 0; s < spatial; ++s) {
                    double dy = gt_read_f64_at(gp, gc.ptr->dtype, base + s);
                    if (training) {
                        double xhat = (gt_read_f64_at(xp, xc.ptr->dtype, base + s) - mean) * inv_std;
                        double val = (w * inv_std / (double)count) * ((double)count * dy - sum_dy - xhat * sum_dy_xhat);
                        gt_write_f64_at(dxp, t_dx->dtype, base + s, val);
                    } else {
                        gt_write_f64_at(dxp, t_dx->dtype, base + s, dy * w * inv_std);
                    }
                }
            }
        }
    }
    if (t_dx) { hvdx.commit(t_dx); *dx = t_dx; }
    if (t_dw) { hvdw.commit(t_dw); *dw = t_dw; }
    if (t_db) { hvdb.commit(t_db); *db = t_db; }
    return 0;
}

// ---------- Embedding (ggml_get_rows) ----------

extern "C" gt_tensor * gt_embedding(const gt_tensor * weight, const gt_tensor * indices, int64_t padding_idx) {
    if (!weight || weight->ndim != 2 || !indices) {
        gt_set_error("embedding: weight must be 2D");
        return nullptr;
    }
    int64_t num_emb = weight->shape[0];
    int64_t emb_dim = weight->shape[1];
    int out_ndim = indices->ndim + 1;
    if (out_ndim > GT_MAX_DIMS) { gt_set_error("embedding: too many dimensions"); return nullptr; }
    int64_t out_shape[GT_MAX_DIMS];
    for (int i = 0; i < indices->ndim; ++i) out_shape[i] = indices->shape[i];
    out_shape[indices->ndim] = emb_dim;

    int out_dtype = gt_is_quantized(weight->dtype) ? GT_F32 : weight->dtype;
    gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, out_dtype, weight->device_type, weight->device_index);
    if (!out) return nullptr;
    if (indices->numel == 0 || weight->device_type == GT_DEV_META) return out;

    // Every index has to name a row of the table. ggml_get_rows asserts on one that does
    // not - which takes the whole process down - and the general path used to return row 0
    // for it. Negative indices are not rows either (as in torch).
    {
        if (!gt_is_integer(indices->dtype)) {
            gt_set_error("embedding: indices must be an integer tensor");
            gt_tensor_free(out);
            return nullptr;
        }
        gt_host_view hv_check;
        const void * ip_check = hv_check.read(indices);
        const bool i_contig = gt_check_contiguous(indices);
        for (int64_t r = 0; r < indices->numel; ++r) {
            const int64_t at = i_contig ? (indices->offset + r) : gt_linear_to_elem_offset(indices, r);
            const int64_t idx = gt_read_i64_at(ip_check, indices->dtype, at);
            if (idx < 0 || idx >= num_emb) {
                gt_set_error("IndexError: embedding index " + std::to_string(idx) + " is out of range for a table of " +
                             std::to_string(num_emb) + " rows");
                gt_tensor_free(out);
                return nullptr;
            }
        }
    }

    // Fast ggml_get_rows path for F32/Quantized weights when padding_idx < 0. ggml_get_rows
    // always produces F32, which is what `out` holds for those two; F16 weights keep their
    // dtype in `out` and therefore take the general path below.
    if (padding_idx < 0 && (weight->dtype == GT_F32 || gt_is_quantized(weight->dtype)) &&
        gt_check_contiguous(weight) && weight->offset == 0 && weight->device_type == indices->device_type) {
        gt_backend_entry * be = gt_get_backend(weight->device_type, weight->device_index);
        gt_tensor * idx_i32 = (indices->dtype == GT_I32 && gt_check_contiguous(indices) && indices->offset == 0)
                              ? nullptr : gt_tensor_to_dtype(indices, GT_I32);
        const gt_tensor * idx_use = idx_i32 ? idx_i32 : indices;
        if (be && idx_use) {
            struct ggml_init_params params = {
                ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
            };
            struct ggml_context * ctx = ggml_init(params);
            if (ctx) {
                struct ggml_tensor * gw = gt_wrap_ggml_4d(ctx, weight, emb_dim, num_emb, 1, 1);
                struct ggml_tensor * gi = gt_wrap_ggml_4d(ctx, idx_use, idx_use->numel, 1, 1, 1);
                struct ggml_tensor * gout = ggml_get_rows(ctx, gw, gi);
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, gout);
                bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                ggml_free(ctx);
                if (idx_i32) gt_tensor_free(idx_i32);
                if (ok) return out;
            }
            if (idx_i32) gt_tensor_free(idx_i32);
        }
    }

    gt_contig_owner wc(weight), ic(indices);
    gt_host_view hvw, hvi, hvo;
    const void * wp = hvw.read(wc.ptr);
    const void * ip = hvi.read(ic.ptr);
    void * op = hvo.write_empty(out);

    #pragma omp parallel for if (indices->numel > 128)
    for (int64_t r = 0; r < indices->numel; ++r) {
        const int64_t idx = gt_read_i64_at(ip, ic.ptr->dtype, r);      // validated above
        for (int64_t c = 0; c < emb_dim; ++c) {
            double v = (idx == padding_idx) ? 0.0 : gt_read_f64_at(wp, wc.ptr->dtype, idx * emb_dim + c);
            gt_write_f64_at(op, out->dtype, r * emb_dim + c, v);
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_embedding_backward(const gt_tensor * grad_out, const gt_tensor * indices, int64_t num_embeddings, int64_t padding_idx) {
    if (!grad_out || !indices || grad_out->ndim < 1) return nullptr;
    int64_t emb_dim = grad_out->shape[grad_out->ndim - 1];
    int64_t dw_shape[2] = {num_embeddings, emb_dim};
    gt_tensor * dw = gt_tensor_empty(2, dw_shape, grad_out->dtype, grad_out->device_type, grad_out->device_index);
    if (!dw) return nullptr;
    gt_fill_scalar_f64(dw, 0.0);
    if (indices->numel == 0 || grad_out->device_type == GT_DEV_META) return dw;

    gt_contig_owner gc(grad_out), ic(indices);
    gt_host_view hvg, hvi, hvd;
    const void * gp = hvg.read(gc.ptr);
    const void * ip = hvi.read(ic.ptr);
    void * dp = hvd.read_write(dw);

    for (int64_t r = 0; r < indices->numel; ++r) {
        int64_t idx = gt_read_i64_at(ip, ic.ptr->dtype, r);
        if (idx < 0) idx += num_embeddings;
        if (idx < 0 || idx >= num_embeddings || idx == padding_idx) continue;
        for (int64_t c = 0; c < emb_dim; ++c) {
            double cur = gt_read_f64_at(dp, dw->dtype, idx * emb_dim + c);
            double addv = gt_read_f64_at(gp, gc.ptr->dtype, r * emb_dim + c);
            gt_write_f64_at(dp, dw->dtype, idx * emb_dim + c, cur + addv);
        }
    }
    hvd.commit(dw);
    return dw;
}

// ---------- Convolutions (ggml_im2col + ggml_mul_mat + C++ backward) ----------

extern "C" gt_tensor * gt_conv2d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias,
                                 int sh, int sw, int ph, int pw, int dh, int dw, int groups) {
    if (!input || !weight || input->ndim != 4 || weight->ndim != 4) {
        gt_set_error("conv2d expects 4D input (N,C,H,W) and 4D weight (OC,IC/g,KH,KW)");
        return nullptr;
    }
    int64_t N = input->shape[0], IC = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t OC = weight->shape[0], IC_g = weight->shape[1], KH = weight->shape[2], KW = weight->shape[3];
    if (IC != IC_g * groups || OC % groups != 0) {
        gt_set_error("conv2d: invalid channel/group dimensions");
        return nullptr;
    }
    int64_t OH = (H + 2 * ph - dh * (KH - 1) - 1) / sh + 1;
    int64_t OW = (W + 2 * pw - dw * (KW - 1) - 1) / sw + 1;
    if (OH <= 0 || OW <= 0) {
        gt_set_error("conv2d: calculated output size is non-positive");
        return nullptr;
    }

    int64_t out_shape[4] = {N, OC, OH, OW};
    gt_tensor * out = gt_tensor_empty(4, out_shape, input->dtype, input->device_type, input->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || input->device_type == GT_DEV_META) return out;

    // Native ggml_im2col + ggml_mul_mat path for groups == 1, F32
    if (groups == 1 && input->dtype == GT_F32 && weight->dtype == GT_F32 &&
        gt_check_contiguous(input) && input->offset == 0 &&
        gt_check_contiguous(weight) && weight->offset == 0 &&
        input->device_type == weight->device_type) {
        gt_backend_entry * be = gt_get_backend(input->device_type, input->device_index);
        // The bias joins the graph as a leaf, so it needs to be F32, densely packed, at the
        // start of its storage and on the input's device. A bias that is not (a slice of a
        // larger tensor, a strided view, another dtype) is materialised here - it used to be
        // left out of the sum without a word.
        gt_tensor * bias_tmp = nullptr;
        const gt_tensor * bias_use = bias;
        bool bias_ok = true;
        if (bias) {
            if (bias->numel != OC || bias->device_type != input->device_type || bias->device_index != input->device_index) {
                bias_ok = false;       // the general path reports or handles it
            } else if (bias->dtype != GT_F32) {
                bias_tmp = gt_tensor_to_dtype(bias, GT_F32);
                bias_use = bias_tmp;
                bias_ok = (bias_tmp != nullptr);
            } else if (!gt_check_contiguous(bias) || bias->offset != 0) {
                bias_tmp = gt_tensor_contiguous(bias);
                bias_use = bias_tmp;
                bias_ok = (bias_tmp != nullptr);
            }
        }
        if (be && bias_ok) {
            struct ggml_init_params params = {
                ggml_tensor_overhead() * 24 + ggml_graph_overhead(), nullptr, true
            };
            struct ggml_context * ctx = ggml_init(params);
            if (ctx) {
                struct ggml_tensor * gw = gt_wrap_ggml_4d(ctx, weight, KW, KH, IC, OC);
                struct ggml_tensor * gx = gt_wrap_ggml_4d(ctx, input, W, H, IC, N);
                struct ggml_tensor * im2col = ggml_im2col(ctx, gw, gx, sw, sh, pw, ph, dw, dh, true, GGML_TYPE_F32);
                struct ggml_tensor * res = ggml_mul_mat(
                    ctx,
                    ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                    ggml_reshape_2d(ctx, gw, KW * KH * IC, OC));
                res = ggml_reshape_4d(ctx, res, OW, OH, N, OC);
                res = ggml_cont(ctx, ggml_permute(ctx, res, 0, 1, 3, 2)); // -> [OW, OH, OC, N] = PyTorch (N, OC, OH, OW)
                if (bias_use) {
                    struct ggml_tensor * gb = gt_wrap_ggml_4d(ctx, bias_use, 1, 1, OC, 1);
                    res = ggml_add(ctx, res, gb);
                }
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, res);
                bool ok = gt_exec_ggml_graph(be, ctx, gf, res, out);
                ggml_free(ctx);
                if (ok) {
                    if (bias_tmp) gt_tensor_free(bias_tmp);
                    return out;
                }
            }
        }
        if (bias_tmp) gt_tensor_free(bias_tmp);
    }

    gt_contig_owner xc(input), wc(weight), bc(bias);
    gt_host_view hvx, hvw, hvb, hvo;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = hvw.read(wc.ptr);
    const void * bp = bc.ptr ? hvb.read(bc.ptr) : nullptr;
    void * op = hvo.write_empty(out);
    int64_t OC_g = OC / groups;

    #pragma omp parallel for collapse(2) if (N * OC * OH * OW > 4096)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            int64_t g = oc / OC_g;
            double bval = bp ? gt_read_f64_at(bp, bc.ptr->dtype, oc) : 0.0;
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double acc = bval;
                    for (int64_t ic = 0; ic < IC_g; ++ic) {
                        int64_t in_c = g * IC_g + ic;
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            int64_t ih = oh * sh - ph + kh * dh;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                int64_t iw = ow * sw - pw + kw * dw;
                                if (iw < 0 || iw >= W) continue;
                                double iv = gt_read_f64_at(xp, xc.ptr->dtype, ((n * IC + in_c) * H + ih) * W + iw);
                                double wv = gt_read_f64_at(wp, wc.ptr->dtype, ((oc * IC_g + ic) * KH + kh) * KW + kw);
                                acc += iv * wv;
                            }
                        }
                    }
                    gt_write_f64_at(op, out->dtype, ((n * OC + oc) * OH + oh) * OW + ow, acc);
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_conv2d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight,
                                  int sh, int sw, int ph, int pw, int dh, int dw, int groups,
                                  int need_dx, int need_dw, int need_db,
                                  gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db) {
    if (!grad_out || !input || !weight) return -1;
    int64_t N = input->shape[0], IC = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t OC = weight->shape[0], IC_g = weight->shape[1], KH = weight->shape[2], KW = weight->shape[3];
    int64_t OH = grad_out->shape[2], OW = grad_out->shape[3];
    int64_t OC_g = OC / groups;

    gt_contig_owner gc(grad_out), xc(input), wc(weight);
    gt_host_view hvg, hvx, hvw;
    const void * gp = hvg.read(gc.ptr);
    const void * xp = hvx.read(xc.ptr);
    const void * wp = hvw.read(wc.ptr);

    if (need_db && db) {
        gt_tensor * tdb = gt_tensor_empty(1, &OC, weight->dtype, input->device_type, input->device_index);
        gt_host_view hvb;
        void * dbp = hvb.write_empty(tdb);
        #pragma omp parallel for if (OC > 16)
        for (int64_t oc = 0; oc < OC; ++oc) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                int64_t base = (n * OC + oc) * OH * OW;
                for (int64_t p = 0; p < OH * OW; ++p) acc += gt_read_f64_at(gp, gc.ptr->dtype, base + p);
            }
            gt_write_f64_at(dbp, tdb->dtype, oc, acc);
        }
        hvb.commit(tdb);
        *db = tdb;
    }

    if (need_dw && dw_out) {
        gt_tensor * tdw = gt_tensor_empty(4, weight->shape, weight->dtype, input->device_type, input->device_index);
        gt_host_view hv_dw;
        void * dwp = hv_dw.write_empty(tdw);
        #pragma omp parallel for collapse(2) if (OC * IC_g > 32)
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t ic = 0; ic < IC_g; ++ic) {
                int64_t g = oc / OC_g;
                int64_t in_c = g * IC_g + ic;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        double acc = 0.0;
                        for (int64_t n = 0; n < N; ++n) {
                            for (int64_t oh = 0; oh < OH; ++oh) {
                                int64_t ih = oh * sh - ph + kh * dh;
                                if (ih < 0 || ih >= H) continue;
                                for (int64_t ow = 0; ow < OW; ++ow) {
                                    int64_t iw = ow * sw - pw + kw * dw;
                                    if (iw < 0 || iw >= W) continue;
                                    double go = gt_read_f64_at(gp, gc.ptr->dtype, ((n * OC + oc) * OH + oh) * OW + ow);
                                    double iv = gt_read_f64_at(xp, xc.ptr->dtype, ((n * IC + in_c) * H + ih) * W + iw);
                                    acc += go * iv;
                                }
                            }
                        }
                        gt_write_f64_at(dwp, tdw->dtype, ((oc * IC_g + ic) * KH + kh) * KW + kw, acc);
                    }
                }
            }
        }
        hv_dw.commit(tdw);
        *dw_out = tdw;
    }

    if (need_dx && dx) {
        gt_tensor * tdx = gt_tensor_empty(4, input->shape, input->dtype, input->device_type, input->device_index);
        gt_fill_scalar_f64(tdx, 0.0);
        gt_host_view hv_dx;
        void * dxp = hv_dx.read_write(tdx);
        std::vector<double> dx_acc(input->numel, 0.0);
        #pragma omp parallel for collapse(2) if (N * IC > 32)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t in_c = 0; in_c < IC; ++in_c) {
                int64_t g = in_c / IC_g;
                int64_t ic = in_c % IC_g;
                for (int64_t oc = g * OC_g; oc < (g + 1) * OC_g; ++oc) {
                    for (int64_t oh = 0; oh < OH; ++oh) {
                        for (int64_t ow = 0; ow < OW; ++ow) {
                            double go = gt_read_f64_at(gp, gc.ptr->dtype, ((n * OC + oc) * OH + oh) * OW + ow);
                            if (go == 0.0) continue;
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                int64_t ih = oh * sh - ph + kh * dh;
                                if (ih < 0 || ih >= H) continue;
                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    int64_t iw = ow * sw - pw + kw * dw;
                                    if (iw < 0 || iw >= W) continue;
                                    double wv = gt_read_f64_at(wp, wc.ptr->dtype, ((oc * IC_g + ic) * KH + kh) * KW + kw);
                                    dx_acc[((n * IC + in_c) * H + ih) * W + iw] += go * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int64_t i = 0; i < input->numel; ++i) gt_write_f64_at(dxp, tdx->dtype, i, dx_acc[i]);
        hv_dx.commit(tdx);
        *dx = tdx;
    }
    return 0;
}

extern "C" gt_tensor * gt_conv1d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias, int s, int p, int d, int groups) {
    if (!input || !weight || input->ndim != 3 || weight->ndim != 3) {
        gt_set_error("conv1d expects 3D input and 3D weight");
        return nullptr;
    }
    int64_t in4_sh[4] = {input->shape[0], input->shape[1], 1, input->shape[2]};
    int64_t in4_st[4] = {input->strides[0], input->strides[1], input->shape[2] * input->strides[2], input->strides[2]};
    int64_t w4_sh[4]  = {weight->shape[0], weight->shape[1], 1, weight->shape[2]};
    int64_t w4_st[4]  = {weight->strides[0], weight->strides[1], weight->shape[2] * weight->strides[2], weight->strides[2]};
    gt_tensor * in4 = gt_tensor_view(input, 4, in4_sh, in4_st, input->offset);
    gt_tensor * w4  = gt_tensor_view(weight, 4, w4_sh, w4_st, weight->offset);
    gt_tensor * r4 = gt_conv2d(in4, w4, bias, 1, s, 0, p, 1, d, groups);
    gt_tensor_free(in4); gt_tensor_free(w4);
    if (!r4) return nullptr;
    int64_t out3_sh[3] = {r4->shape[0], r4->shape[1], r4->shape[3]};
    int64_t out3_st[3] = {r4->strides[0], r4->strides[1], r4->strides[3]};
    gt_tensor * r3 = gt_tensor_view(r4, 3, out3_sh, out3_st, r4->offset);
    gt_tensor_free(r4);
    return r3;
}

extern "C" int gt_conv1d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight,
                                  int s, int p, int d, int groups, int need_dx, int need_dw, int need_db,
                                  gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db) {
    if (!grad_out || !input || !weight) return -1;
    int64_t g4_sh[4] = {grad_out->shape[0], grad_out->shape[1], 1, grad_out->shape[2]};
    int64_t g4_st[4] = {grad_out->strides[0], grad_out->strides[1], grad_out->shape[2] * grad_out->strides[2], grad_out->strides[2]};
    int64_t i4_sh[4] = {input->shape[0], input->shape[1], 1, input->shape[2]};
    int64_t i4_st[4] = {input->strides[0], input->strides[1], input->shape[2] * input->strides[2], input->strides[2]};
    int64_t w4_sh[4] = {weight->shape[0], weight->shape[1], 1, weight->shape[2]};
    int64_t w4_st[4] = {weight->strides[0], weight->strides[1], weight->shape[2] * weight->strides[2], weight->strides[2]};
    gt_tensor * g4 = gt_tensor_view(grad_out, 4, g4_sh, g4_st, grad_out->offset);
    gt_tensor * i4 = gt_tensor_view(input, 4, i4_sh, i4_st, input->offset);
    gt_tensor * w4 = gt_tensor_view(weight, 4, w4_sh, w4_st, weight->offset);
    gt_tensor * dx4 = nullptr, * dw4 = nullptr;
    int rc = gt_conv2d_backward(g4, i4, w4, 1, s, 0, p, 1, d, groups, need_dx, need_dw, need_db,
                                need_dx ? &dx4 : nullptr, need_dw ? &dw4 : nullptr, db);
    gt_tensor_free(g4); gt_tensor_free(i4); gt_tensor_free(w4);
    if (dx4) {
        int64_t sh[3] = {dx4->shape[0], dx4->shape[1], dx4->shape[3]};
        int64_t st[3] = {dx4->strides[0], dx4->strides[1], dx4->strides[3]};
        *dx = gt_tensor_view(dx4, 3, sh, st, dx4->offset);
        gt_tensor_free(dx4);
    }
    if (dw4) {
        int64_t sh[3] = {dw4->shape[0], dw4->shape[1], dw4->shape[3]};
        int64_t st[3] = {dw4->strides[0], dw4->strides[1], dw4->strides[3]};
        *dw_out = gt_tensor_view(dw4, 3, sh, st, dw4->offset);
        gt_tensor_free(dw4);
    }
    return rc;
}

extern "C" gt_tensor * gt_conv_transpose2d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias,
                                           int sh, int sw, int ph, int pw, int oph, int opw, int dh, int dw, int groups) {
    if (!input || !weight || input->ndim != 4 || weight->ndim != 4) {
        gt_set_error("conv_transpose2d expects 4D input and weight");
        return nullptr;
    }
    int64_t N = input->shape[0], IC = input->shape[1], IH = input->shape[2], IW = input->shape[3];
    int64_t OC_g = weight->shape[1], KH = weight->shape[2], KW = weight->shape[3];
    int64_t IC_g = IC / groups;
    int64_t OC = OC_g * groups;
    int64_t OH = (IH - 1) * sh - 2 * ph + dh * (KH - 1) + oph + 1;
    int64_t OW = (IW - 1) * sw - 2 * pw + dw * (KW - 1) + opw + 1;
    if (OH <= 0 || OW <= 0) { gt_set_error("conv_transpose2d: negative output size"); return nullptr; }

    int64_t out_shape[4] = {N, OC, OH, OW};
    gt_tensor * out = gt_tensor_empty(4, out_shape, input->dtype, input->device_type, input->device_index);
    if (!out) return nullptr;

    gt_contig_owner xc(input), wc(weight), bc(bias);
    gt_host_view hvx, hvw, hvb, hvo;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = hvw.read(wc.ptr);
    const void * bp = bc.ptr ? hvb.read(bc.ptr) : nullptr;
    void * op = hvo.write_empty(out);

    std::vector<double> acc(out->numel, 0.0);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            double bval = bp ? gt_read_f64_at(bp, bc.ptr->dtype, oc) : 0.0;
            if (bval != 0.0) {
                int64_t base = (n * OC + oc) * OH * OW;
                for (int64_t p = 0; p < OH * OW; ++p) acc[base + p] = bval;
            }
        }
    }
    #pragma omp parallel for collapse(2) if (N * OC > 32)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            int64_t g = oc / OC_g;
            int64_t oc_l = oc % OC_g;
            for (int64_t ic = g * IC_g; ic < (g + 1) * IC_g; ++ic) {
                for (int64_t ih = 0; ih < IH; ++ih) {
                    for (int64_t iw = 0; iw < IW; ++iw) {
                        double iv = gt_read_f64_at(xp, xc.ptr->dtype, ((n * IC + ic) * IH + ih) * IW + iw);
                        if (iv == 0.0) continue;
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            int64_t oh = ih * sh - ph + kh * dh;
                            if (oh < 0 || oh >= OH) continue;
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                int64_t ow = iw * sw - pw + kw * dw;
                                if (ow < 0 || ow >= OW) continue;
                                double wv = gt_read_f64_at(wp, wc.ptr->dtype, ((ic * OC_g + oc_l) * KH + kh) * KW + kw);
                                acc[((n * OC + oc) * OH + oh) * OW + ow] += iv * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    for (int64_t i = 0; i < out->numel; ++i) gt_write_f64_at(op, out->dtype, i, acc[i]);
    hvo.commit(out);
    return out;
}

extern "C" int gt_conv_transpose2d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight,
                                            int sh, int sw, int ph, int pw, int oph, int opw, int dh, int dw, int groups,
                                            int need_dx, int need_dw, int need_db,
                                            gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db) {
    (void)oph; (void)opw;
    if (!grad_out || !input || !weight) return -1;
    int64_t N = input->shape[0], IC = input->shape[1], IH = input->shape[2], IW = input->shape[3];
    int64_t OC_g = weight->shape[1], KH = weight->shape[2], KW = weight->shape[3];
    int64_t OC = OC_g * groups, IC_g = IC / groups;
    int64_t OH = grad_out->shape[2], OW = grad_out->shape[3];

    gt_contig_owner gc(grad_out), xc(input), wc(weight);
    gt_host_view hvg, hvx, hvw;
    const void * gp = hvg.read(gc.ptr);
    const void * xp = hvx.read(xc.ptr);
    const void * wp = hvw.read(wc.ptr);

    if (need_db && db) {
        gt_tensor * tdb = gt_tensor_empty(1, &OC, weight->dtype, input->device_type, input->device_index);
        gt_host_view hvb;
        void * dbp = hvb.write_empty(tdb);
        for (int64_t oc = 0; oc < OC; ++oc) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t p = 0; p < OH * OW; ++p)
                    acc += gt_read_f64_at(gp, gc.ptr->dtype, (n * OC + oc) * OH * OW + p);
            gt_write_f64_at(dbp, tdb->dtype, oc, acc);
        }
        hvb.commit(tdb);
        *db = tdb;
    }
    if (need_dx && dx) {
        gt_tensor * tdx = gt_tensor_empty(4, input->shape, input->dtype, input->device_type, input->device_index);
        gt_host_view hv_dx;
        void * dxp = hv_dx.write_empty(tdx);
        #pragma omp parallel for collapse(2) if (N * IC > 32)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t ic = 0; ic < IC; ++ic) {
                int64_t g = ic / IC_g;
                for (int64_t ih = 0; ih < IH; ++ih) {
                    for (int64_t iw = 0; iw < IW; ++iw) {
                        double acc = 0.0;
                        for (int64_t oc_l = 0; oc_l < OC_g; ++oc_l) {
                            int64_t oc = g * OC_g + oc_l;
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                int64_t oh = ih * sh - ph + kh * dh;
                                if (oh < 0 || oh >= OH) continue;
                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    int64_t ow = iw * sw - pw + kw * dw;
                                    if (ow < 0 || ow >= OW) continue;
                                    double go = gt_read_f64_at(gp, gc.ptr->dtype, ((n * OC + oc) * OH + oh) * OW + ow);
                                    double wv = gt_read_f64_at(wp, wc.ptr->dtype, ((ic * OC_g + oc_l) * KH + kh) * KW + kw);
                                    acc += go * wv;
                                }
                            }
                        }
                        gt_write_f64_at(dxp, tdx->dtype, ((n * IC + ic) * IH + ih) * IW + iw, acc);
                    }
                }
            }
        }
        hv_dx.commit(tdx);
        *dx = tdx;
    }
    if (need_dw && dw_out) {
        gt_tensor * tdw = gt_tensor_empty(4, weight->shape, weight->dtype, input->device_type, input->device_index);
        gt_host_view hv_dw;
        void * dwp = hv_dw.write_empty(tdw);
        #pragma omp parallel for collapse(2) if (IC * OC_g > 32)
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t oc_l = 0; oc_l < OC_g; ++oc_l) {
                int64_t g = ic / IC_g;
                int64_t oc = g * OC_g + oc_l;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        double acc = 0.0;
                        for (int64_t n = 0; n < N; ++n) {
                            for (int64_t ih = 0; ih < IH; ++ih) {
                                int64_t oh = ih * sh - ph + kh * dh;
                                if (oh < 0 || oh >= OH) continue;
                                for (int64_t iw = 0; iw < IW; ++iw) {
                                    int64_t ow = iw * sw - pw + kw * dw;
                                    if (ow < 0 || ow >= OW) continue;
                                    double iv = gt_read_f64_at(xp, xc.ptr->dtype, ((n * IC + ic) * IH + ih) * IW + iw);
                                    double go = gt_read_f64_at(gp, gc.ptr->dtype, ((n * OC + oc) * OH + oh) * OW + ow);
                                    acc += iv * go;
                                }
                            }
                        }
                        gt_write_f64_at(dwp, tdw->dtype, ((ic * OC_g + oc_l) * KH + kh) * KW + kw, acc);
                    }
                }
            }
        }
        hv_dw.commit(tdw);
        *dw_out = tdw;
    }
    return 0;
}

extern "C" gt_tensor * gt_conv3d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias,
                                 int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups) {
    if (!input || !weight || input->ndim != 5 || weight->ndim != 5) {
        gt_set_error("conv3d expects 5D input and weight");
        return nullptr;
    }
    int64_t N = input->shape[0], IC = input->shape[1], ID = input->shape[2], IH = input->shape[3], IW = input->shape[4];
    int64_t OC = weight->shape[0], IC_g = weight->shape[1], KD = weight->shape[2], KH = weight->shape[3], KW = weight->shape[4];
    int64_t OD = (ID + 2 * pd - dd * (KD - 1) - 1) / sd + 1;
    int64_t OH = (IH + 2 * ph - dh * (KH - 1) - 1) / sh + 1;
    int64_t OW = (IW + 2 * pw - dw * (KW - 1) - 1) / sw + 1;
    if (OD <= 0 || OH <= 0 || OW <= 0) { gt_set_error("conv3d: negative output size"); return nullptr; }

    int64_t out_shape[5] = {N, OC, OD, OH, OW};
    gt_tensor * out = gt_tensor_empty(5, out_shape, input->dtype, input->device_type, input->device_index);
    if (!out) return nullptr;

    gt_contig_owner xc(input), wc(weight), bc(bias);
    gt_host_view hvx, hvw, hvb, hvo;
    const void * xp = hvx.read(xc.ptr);
    const void * wp = hvw.read(wc.ptr);
    const void * bp = bc.ptr ? hvb.read(bc.ptr) : nullptr;
    void * op = hvo.write_empty(out);
    int64_t OC_g = OC / groups;

    #pragma omp parallel for collapse(2) if (N * OC > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            int64_t g = oc / OC_g;
            double bval = bp ? gt_read_f64_at(bp, bc.ptr->dtype, oc) : 0.0;
            for (int64_t od = 0; od < OD; ++od) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        double acc = bval;
                        for (int64_t ic = 0; ic < IC_g; ++ic) {
                            int64_t in_c = g * IC_g + ic;
                            for (int64_t kd = 0; kd < KD; ++kd) {
                                int64_t id = od * sd - pd + kd * dd;
                                if (id < 0 || id >= ID) continue;
                                for (int64_t kh = 0; kh < KH; ++kh) {
                                    int64_t ih = oh * sh - ph + kh * dh;
                                    if (ih < 0 || ih >= IH) continue;
                                    for (int64_t kw = 0; kw < KW; ++kw) {
                                        int64_t iw = ow * sw - pw + kw * dw;
                                        if (iw < 0 || iw >= IW) continue;
                                        double iv = gt_read_f64_at(xp, xc.ptr->dtype, ((((n * IC + in_c) * ID + id) * IH + ih) * IW + iw));
                                        double wv = gt_read_f64_at(wp, wc.ptr->dtype, ((((oc * IC_g + ic) * KD + kd) * KH + kh) * KW + kw));
                                        acc += iv * wv;
                                    }
                                }
                            }
                        }
                        gt_write_f64_at(op, out->dtype, ((((n * OC + oc) * OD + od) * OH + oh) * OW + ow), acc);
                    }
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_conv3d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight,
                                  int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups,
                                  int need_dx, int need_dw, int need_db,
                                  gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db) {
    if (!grad_out || !input || !weight) return -1;
    int64_t N = input->shape[0], IC = input->shape[1], ID = input->shape[2], IH = input->shape[3], IW = input->shape[4];
    int64_t OC = weight->shape[0], IC_g = weight->shape[1], KD = weight->shape[2], KH = weight->shape[3], KW = weight->shape[4];
    int64_t OD = grad_out->shape[2], OH = grad_out->shape[3], OW = grad_out->shape[4];
    int64_t OC_g = OC / groups;

    gt_contig_owner gc(grad_out), xc(input), wc(weight);
    gt_host_view hvg, hvx, hvw;
    const void * gp = hvg.read(gc.ptr);
    const void * xp = hvx.read(xc.ptr);
    const void * wp = hvw.read(wc.ptr);

    if (need_db && db) {
        gt_tensor * tdb = gt_tensor_empty(1, &OC, weight->dtype, input->device_type, input->device_index);
        gt_host_view hvb;
        void * dbp = hvb.write_empty(tdb);
        for (int64_t oc = 0; oc < OC; ++oc) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t p = 0; p < OD * OH * OW; ++p)
                    acc += gt_read_f64_at(gp, gc.ptr->dtype, (n * OC + oc) * OD * OH * OW + p);
            gt_write_f64_at(dbp, tdb->dtype, oc, acc);
        }
        hvb.commit(tdb);
        *db = tdb;
    }
    if (need_dw && dw_out) {
        gt_tensor * tdw = gt_tensor_empty(5, weight->shape, weight->dtype, input->device_type, input->device_index);
        gt_host_view hvd;
        void * dwp = hvd.write_empty(tdw);
        for (int64_t oc = 0; oc < OC; ++oc) {
            int64_t g = oc / OC_g;
            for (int64_t ic = 0; ic < IC_g; ++ic) {
                int64_t in_c = g * IC_g + ic;
                for (int64_t kd = 0; kd < KD; ++kd)
                    for (int64_t kh = 0; kh < KH; ++kh)
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            double acc = 0.0;
                            for (int64_t n = 0; n < N; ++n)
                                for (int64_t od = 0; od < OD; ++od) {
                                    int64_t id = od * sd - pd + kd * dd;
                                    if (id < 0 || id >= ID) continue;
                                    for (int64_t oh = 0; oh < OH; ++oh) {
                                        int64_t ih = oh * sh - ph + kh * dh;
                                        if (ih < 0 || ih >= IH) continue;
                                        for (int64_t ow = 0; ow < OW; ++ow) {
                                            int64_t iw = ow * sw - pw + kw * dw;
                                            if (iw < 0 || iw >= IW) continue;
                                            acc += gt_read_f64_at(gp, gc.ptr->dtype, ((((n * OC + oc) * OD + od) * OH + oh) * OW + ow)) *
                                                   gt_read_f64_at(xp, xc.ptr->dtype, ((((n * IC + in_c) * ID + id) * IH + ih) * IW + iw));
                                        }
                                    }
                                }
                            gt_write_f64_at(dwp, tdw->dtype, ((((oc * IC_g + ic) * KD + kd) * KH + kh) * KW + kw), acc);
                        }
            }
        }
        hvd.commit(tdw);
        *dw_out = tdw;
    }
    if (need_dx && dx) {
        gt_tensor * tdx = gt_tensor_empty(5, input->shape, input->dtype, input->device_type, input->device_index);
        std::vector<double> dx_acc(input->numel, 0.0);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t oc = 0; oc < OC; ++oc) {
                int64_t g = oc / OC_g;
                for (int64_t od = 0; od < OD; ++od)
                    for (int64_t oh = 0; oh < OH; ++oh)
                        for (int64_t ow = 0; ow < OW; ++ow) {
                            double go = gt_read_f64_at(gp, gc.ptr->dtype, ((((n * OC + oc) * OD + od) * OH + oh) * OW + ow));
                            if (go == 0.0) continue;
                            for (int64_t ic = 0; ic < IC_g; ++ic) {
                                int64_t in_c = g * IC_g + ic;
                                for (int64_t kd = 0; kd < KD; ++kd) {
                                    int64_t id = od * sd - pd + kd * dd;
                                    if (id < 0 || id >= ID) continue;
                                    for (int64_t kh = 0; kh < KH; ++kh) {
                                        int64_t ih = oh * sh - ph + kh * dh;
                                        if (ih < 0 || ih >= IH) continue;
                                        for (int64_t kw = 0; kw < KW; ++kw) {
                                            int64_t iw = ow * sw - pw + kw * dw;
                                            if (iw < 0 || iw >= IW) continue;
                                            double wv = gt_read_f64_at(wp, wc.ptr->dtype, ((((oc * IC_g + ic) * KD + kd) * KH + kh) * KW + kw));
                                            dx_acc[((((n * IC + in_c) * ID + id) * IH + ih) * IW + iw)] += go * wv;
                                        }
                                    }
                                }
                            }
                        }
            }
        gt_host_view hvd;
        void * dxp = hvd.write_empty(tdx);
        for (int64_t i = 0; i < input->numel; ++i) gt_write_f64_at(dxp, tdx->dtype, i, dx_acc[i]);
        hvd.commit(tdx);
        *dx = tdx;
    }
    return 0;
}

// ---------- Pooling & Upsampling ----------

extern "C" int gt_max_pool2d(const gt_tensor * input, int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
                             gt_tensor ** out, gt_tensor ** indices) {
    if (!input || input->ndim != 4 || !out) return -1;
    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t OH = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t OW = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    int64_t osh[4] = {N, C, OH, OW};
    gt_tensor * to = gt_tensor_empty(4, osh, input->dtype, input->device_type, input->device_index);
    gt_tensor * ti = indices ? gt_tensor_empty(4, osh, GT_I64, input->device_type, input->device_index) : nullptr;
    if (!to) return -1;

    gt_contig_owner xc(input);
    gt_host_view hvx, hvo, hvi;
    const void * xp = hvx.read(xc.ptr);
    void * op = hvo.write_empty(to);
    void * ip = ti ? hvi.write_empty(ti) : nullptr;

    #pragma omp parallel for collapse(2) if (N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double best = -std::numeric_limits<double>::infinity();
                    int64_t best_idx = -1;
                    for (int i = 0; i < kh; ++i) {
                        int64_t ih = oh * sh - ph + i * dh;
                        if (ih < 0 || ih >= H) continue;
                        for (int j = 0; j < kw; ++j) {
                            int64_t iw = ow * sw - pw + j * dw;
                            if (iw < 0 || iw >= W) continue;
                            int64_t idx = ih * W + iw;
                            double v = gt_read_f64_at(xp, xc.ptr->dtype, (n * C + c) * H * W + idx);
                            if (v > best) { best = v; best_idx = idx; }
                        }
                    }
                    int64_t oidx = ((n * C + c) * OH + oh) * OW + ow;
                    gt_write_f64_at(op, to->dtype, oidx, best);
                    if (ip) gt_write_i64_at(ip, GT_I64, oidx, best_idx);
                }
            }
        }
    }
    hvo.commit(to);
    if (ti) { hvi.commit(ti); *indices = ti; }
    *out = to;
    return 0;
}

extern "C" gt_tensor * gt_max_pool2d_backward(const gt_tensor * grad_out, const gt_tensor * indices, const gt_tensor * input_ref) {
    if (!grad_out || !indices || !input_ref) return nullptr;
    int64_t N = input_ref->shape[0], C = input_ref->shape[1], H = input_ref->shape[2], W = input_ref->shape[3];
    int64_t OH = grad_out->shape[2], OW = grad_out->shape[3];
    gt_tensor * dx = gt_tensor_empty(4, input_ref->shape, input_ref->dtype, input_ref->device_type, input_ref->device_index);
    if (!dx) return nullptr;
    gt_contig_owner gc(grad_out), ic(indices);
    gt_host_view hvg, hvi, hvd;
    const void * gp = hvg.read(gc.ptr);
    const void * ip = hvi.read(ic.ptr);
    void * dp = hvd.write_empty(dx);
    std::vector<double> acc(input_ref->numel, 0.0);

    #pragma omp parallel for collapse(2) if (N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_base = (n * C + c) * H * W;
            int64_t out_base = (n * C + c) * OH * OW;
            for (int64_t p = 0; p < OH * OW; ++p) {
                int64_t idx = gt_read_i64_at(ip, ic.ptr->dtype, out_base + p);
                if (idx >= 0 && idx < H * W) {
                    acc[in_base + idx] += gt_read_f64_at(gp, gc.ptr->dtype, out_base + p);
                }
            }
        }
    }
    for (int64_t i = 0; i < input_ref->numel; ++i) gt_write_f64_at(dp, dx->dtype, i, acc[i]);
    hvd.commit(dx);
    return dx;
}

extern "C" gt_tensor * gt_avg_pool2d(const gt_tensor * input, int kh, int kw, int sh, int sw, int ph, int pw, int count_include_pad) {
    if (!input || input->ndim != 4) return nullptr;
    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t OH = (H + 2 * ph - kh) / sh + 1;
    int64_t OW = (W + 2 * pw - kw) / sw + 1;
    int64_t osh[4] = {N, C, OH, OW};
    gt_tensor * out = gt_tensor_empty(4, osh, input->dtype, input->device_type, input->device_index);
    if (!out) return nullptr;

    gt_contig_owner xc(input);
    gt_host_view hvx, hvo;
    const void * xp = hvx.read(xc.ptr);
    void * op = hvo.write_empty(out);

    #pragma omp parallel for collapse(2) if (N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double acc = 0.0;
                    int cnt = 0;
                    for (int i = 0; i < kh; ++i) {
                        int64_t ih = oh * sh - ph + i;
                        for (int j = 0; j < kw; ++j) {
                            int64_t iw = ow * sw - pw + j;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                acc += gt_read_f64_at(xp, xc.ptr->dtype, ((n * C + c) * H + ih) * W + iw);
                                ++cnt;
                            } else if (count_include_pad && ih >= -ph && ih < H + ph && iw >= -pw && iw < W + pw) {
                                ++cnt;
                            }
                        }
                    }
                    double div = (double)(count_include_pad ? (kh * kw) : std::max(cnt, 1));
                    gt_write_f64_at(op, out->dtype, ((n * C + c) * OH + oh) * OW + ow, acc / div);
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_avg_pool2d_backward(const gt_tensor * grad_out, const gt_tensor * input_ref, int kh, int kw, int sh, int sw, int ph, int pw, int count_include_pad) {
    if (!grad_out || !input_ref) return nullptr;
    int64_t N = input_ref->shape[0], C = input_ref->shape[1], H = input_ref->shape[2], W = input_ref->shape[3];
    int64_t OH = grad_out->shape[2], OW = grad_out->shape[3];
    gt_tensor * dx = gt_tensor_empty(4, input_ref->shape, input_ref->dtype, input_ref->device_type, input_ref->device_index);
    if (!dx) return nullptr;
    gt_contig_owner gc(grad_out);
    gt_host_view hvg, hvd;
    const void * gp = hvg.read(gc.ptr);
    void * dp = hvd.write_empty(dx);
    std::vector<double> acc(input_ref->numel, 0.0);

    #pragma omp parallel for collapse(2) if (N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    int cnt = 0;
                    for (int i = 0; i < kh; ++i) {
                        int64_t ih = oh * sh - ph + i;
                        for (int j = 0; j < kw; ++j) {
                            int64_t iw = ow * sw - pw + j;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) ++cnt;
                        }
                    }
                    double div = (double)(count_include_pad ? (kh * kw) : std::max(cnt, 1));
                    double go = gt_read_f64_at(gp, gc.ptr->dtype, ((n * C + c) * OH + oh) * OW + ow) / div;
                    for (int i = 0; i < kh; ++i) {
                        int64_t ih = oh * sh - ph + i;
                        if (ih < 0 || ih >= H) continue;
                        for (int j = 0; j < kw; ++j) {
                            int64_t iw = ow * sw - pw + j;
                            if (iw < 0 || iw >= W) continue;
                            acc[((n * C + c) * H + ih) * W + iw] += go;
                        }
                    }
                }
            }
        }
    }
    for (int64_t i = 0; i < input_ref->numel; ++i) gt_write_f64_at(dp, dx->dtype, i, acc[i]);
    hvd.commit(dx);
    return dx;
}

extern "C" gt_tensor * gt_upsample_2d(const gt_tensor * input, int64_t out_h, int64_t out_w, int mode, int align_corners) {
    // mode: 0 = nearest, 1 = bilinear
    if (!input || input->ndim != 4) return nullptr;
    int64_t N = input->shape[0], C = input->shape[1], IH = input->shape[2], IW = input->shape[3];
    int64_t osh[4] = {N, C, out_h, out_w};
    gt_tensor * out = gt_tensor_empty(4, osh, input->dtype, input->device_type, input->device_index);
    if (!out) return nullptr;
    gt_contig_owner xc(input);
    gt_host_view hvx, hvo;
    const void * xp = hvx.read(xc.ptr);
    void * op = hvo.write_empty(out);

    #pragma omp parallel for collapse(2) if (N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_base = (n * C + c) * IH * IW;
            int64_t out_base = (n * C + c) * out_h * out_w;
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    double val = 0.0;
                    if (mode == 0) {
                        int64_t ih = std::min<int64_t>((oh * IH) / out_h, IH - 1);
                        int64_t iw = std::min<int64_t>((ow * IW) / out_w, IW - 1);
                        val = gt_read_f64_at(xp, xc.ptr->dtype, in_base + ih * IW + iw);
                    } else {
                        double rh = (align_corners && out_h > 1) ? ((double)oh * (IH - 1) / (out_h - 1)) : (((oh + 0.5) * IH / out_h) - 0.5);
                        double rw = (align_corners && out_w > 1) ? ((double)ow * (IW - 1) / (out_w - 1)) : (((ow + 0.5) * IW / out_w) - 0.5);
                        rh = std::clamp(rh, 0.0, (double)(IH - 1));
                        rw = std::clamp(rw, 0.0, (double)(IW - 1));
                        int64_t h0 = (int64_t)std::floor(rh), w0 = (int64_t)std::floor(rw);
                        int64_t h1 = std::min(h0 + 1, IH - 1), w1 = std::min(w0 + 1, IW - 1);
                        double dh = rh - h0, dw = rw - w0;
                        double v00 = gt_read_f64_at(xp, xc.ptr->dtype, in_base + h0 * IW + w0);
                        double v01 = gt_read_f64_at(xp, xc.ptr->dtype, in_base + h0 * IW + w1);
                        double v10 = gt_read_f64_at(xp, xc.ptr->dtype, in_base + h1 * IW + w0);
                        double v11 = gt_read_f64_at(xp, xc.ptr->dtype, in_base + h1 * IW + w1);
                        val = (1 - dh) * ((1 - dw) * v00 + dw * v01) + dh * ((1 - dw) * v10 + dw * v11);
                    }
                    gt_write_f64_at(op, out->dtype, out_base + oh * out_w + ow, val);
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_upsample_2d_backward(const gt_tensor * grad_out, int64_t in_h, int64_t in_w, int mode, int align_corners) {
    if (!grad_out || grad_out->ndim != 4) return nullptr;
    int64_t N = grad_out->shape[0], C = grad_out->shape[1], OH = grad_out->shape[2], OW = grad_out->shape[3];
    int64_t ish[4] = {N, C, in_h, in_w};
    gt_tensor * dx = gt_tensor_empty(4, ish, grad_out->dtype, grad_out->device_type, grad_out->device_index);
    if (!dx) return nullptr;
    gt_contig_owner gc(grad_out);
    gt_host_view hvg, hvd;
    const void * gp = hvg.read(gc.ptr);
    void * dp = hvd.write_empty(dx);
    std::vector<double> acc(dx->numel, 0.0);

    #pragma omp parallel for collapse(2) if (N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_base = (n * C + c) * in_h * in_w;
            int64_t out_base = (n * C + c) * OH * OW;
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double go = gt_read_f64_at(gp, gc.ptr->dtype, out_base + oh * OW + ow);
                    if (mode == 0) {
                        int64_t ih = std::min<int64_t>((oh * in_h) / OH, in_h - 1);
                        int64_t iw = std::min<int64_t>((ow * in_w) / OW, in_w - 1);
                        acc[in_base + ih * in_w + iw] += go;
                    } else {
                        double rh = (align_corners && OH > 1) ? ((double)oh * (in_h - 1) / (OH - 1)) : (((oh + 0.5) * in_h / OH) - 0.5);
                        double rw = (align_corners && OW > 1) ? ((double)ow * (in_w - 1) / (OW - 1)) : (((ow + 0.5) * in_w / OW) - 0.5);
                        rh = std::clamp(rh, 0.0, (double)(in_h - 1));
                        rw = std::clamp(rw, 0.0, (double)(in_w - 1));
                        int64_t h0 = (int64_t)std::floor(rh), w0 = (int64_t)std::floor(rw);
                        int64_t h1 = std::min(h0 + 1, in_h - 1), w1 = std::min(w0 + 1, in_w - 1);
                        double dh = rh - h0, dw = rw - w0;
                        acc[in_base + h0 * in_w + w0] += go * (1 - dh) * (1 - dw);
                        acc[in_base + h0 * in_w + w1] += go * (1 - dh) * dw;
                        acc[in_base + h1 * in_w + w0] += go * dh * (1 - dw);
                        acc[in_base + h1 * in_w + w1] += go * dh * dw;
                    }
                }
            }
        }
    }
    for (int64_t i = 0; i < dx->numel; ++i) gt_write_f64_at(dp, dx->dtype, i, acc[i]);
    hvd.commit(dx);
    return dx;
}

// ---------- Cross Entropy Loss ----------

struct gt_ce_layout {
    int64_t N = 1;        // items in the batch
    int64_t C = 0;        // classes
    int64_t spatial = 1;  // elements per item behind the class dimension
    bool soft = false;    // targets are class probabilities (floating point), not class indices
};

static std::string gt_ce_shape_str(const gt_tensor * t) {
    std::string s = "[";
    for (int i = 0; i < t->ndim; ++i) s += (i ? ", " : "") + std::to_string(t->shape[i]);
    return s + "]";
}

// Everything the loops below index by: the class count, the number of targets and the length
// of the weight vector. The forward and the backward pass both start here, before anything is
// read or allocated, so a target or weight tensor of the wrong size is an error instead of a
// read past the end of its buffer.
static bool gt_cross_entropy_layout(const gt_tensor * logits, const gt_tensor * targets, const gt_tensor * weight,
                                    double label_smoothing, int reduction, gt_ce_layout & L) {
    if (!logits || !targets) {
        gt_set_error("cross_entropy: logits and targets are required");
        return false;
    }
    if (logits->ndim < 1) {
        gt_set_error("cross_entropy: logits need a class dimension: (C), (N, C) or (N, C, d1, ...)");
        return false;
    }
    if (!gt_is_floating(logits->dtype)) {
        gt_set_error("cross_entropy: logits must be a floating point tensor");
        return false;
    }
    if (reduction < 0 || reduction > 2) {
        gt_set_error("cross_entropy: reduction must be 0 (none), 1 (mean) or 2 (sum), got " + std::to_string(reduction));
        return false;
    }
    if (!(label_smoothing >= 0.0 && label_smoothing <= 1.0)) {
        gt_set_error("cross_entropy: label_smoothing must be between 0.0 and 1.0, got " + std::to_string(label_smoothing));
        return false;
    }
    L.N = (logits->ndim == 1) ? 1 : logits->shape[0];
    L.C = (logits->ndim == 1) ? logits->shape[0] : logits->shape[1];
    L.spatial = 1;
    for (int i = 2; i < logits->ndim; ++i) L.spatial *= logits->shape[i];
    L.soft = gt_is_floating(targets->dtype);

    if (weight) {
        if (gt_is_complex(weight->dtype)) {
            gt_set_error("cross_entropy: weight must be a real tensor");
            return false;
        }
        if (weight->numel != L.C) {
            gt_set_error("cross_entropy: weight must hold one value per class: expected " + std::to_string(L.C) +
                         " elements, got a tensor of shape " + gt_ce_shape_str(weight));
            return false;
        }
    }
    if (L.soft) {
        bool same = targets->ndim == logits->ndim;
        for (int i = 0; same && i < logits->ndim; ++i) same = targets->shape[i] == logits->shape[i];
        if (!same) {
            gt_set_error("cross_entropy: floating point targets are class probabilities and need the shape of the logits " +
                         gt_ce_shape_str(logits) + ", got " + gt_ce_shape_str(targets) +
                         " (class indices must be an integer tensor)");
            return false;
        }
    } else {
        if (!gt_is_integer(targets->dtype)) {
            gt_set_error("cross_entropy: targets must be integer class indices or floating point class probabilities");
            return false;
        }
        if (targets->numel != L.N * L.spatial) {
            gt_set_error("cross_entropy: expected " + std::to_string(L.N * L.spatial) + " class indices as targets for logits of shape " +
                         gt_ce_shape_str(logits) + ", got a tensor of shape " + gt_ce_shape_str(targets));
            return false;
        }
    }
    return true;
}

// Class indices are used to index the logits and the weight vector: every one has to be
// ignore_index or a class. Checked in one pass before any loss or gradient is computed, in the
// backward pass too - the targets may have been modified in place since the forward pass.
static bool gt_cross_entropy_check_targets(const void * tp, int dtype, int64_t count, int64_t C, int64_t ignore_index) {
    for (int64_t i = 0; i < count; ++i) {
        const int64_t tgt = gt_read_i64_at(tp, dtype, i);
        if (tgt != ignore_index && (tgt < 0 || tgt >= C)) {
            gt_set_error("cross_entropy: target " + std::to_string(tgt) + " is out of bounds for " + std::to_string(C) + " classes");
            return false;
        }
    }
    return true;
}

extern "C" gt_tensor * gt_cross_entropy_loss(const gt_tensor * logits, const gt_tensor * targets, const gt_tensor * weight,
                                             int64_t ignore_index, double label_smoothing, int reduction) {
    // logits: (C,) or (N, C) or (N, C, d1, ...)
    // targets: integer class indices, one per item - () or (N,) or (N, d1, ...) -
    //          OR floating point class probabilities with the shape of the logits
    gt_ce_layout L;
    if (!gt_cross_entropy_layout(logits, targets, weight, label_smoothing, reduction, L)) return nullptr;
    const bool soft_targets = L.soft;
    const int64_t N = L.N, C = L.C, spatial = L.spatial;

    int64_t per_shape[GT_MAX_DIMS];
    int per_ndim = 0;
    if (logits->ndim >= 2) {
        per_shape[0] = N;
        per_ndim = 1;
        for (int i = 2; i < logits->ndim; ++i) per_shape[per_ndim++] = logits->shape[i];
    }

    gt_contig_owner lc(logits), tc(targets), wc(weight);
    gt_host_view hvl, hvt, hvw, hvo;
    const void * lp = hvl.read(lc.ptr);
    const void * tp = hvt.read(tc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    if (!soft_targets && !gt_cross_entropy_check_targets(tp, tc.ptr->dtype, N * spatial, C, ignore_index)) return nullptr;

    gt_tensor * losses = gt_tensor_empty(per_ndim, per_shape, logits->dtype, logits->device_type, logits->device_index);
    if (!losses) return nullptr;
    void * op = hvo.write_empty(losses);

    double total_loss = 0.0;
    double total_weight = 0.0;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t sp = 0; sp < spatial; ++sp) {
            int64_t item_idx = n * spatial + sp;
            auto get_logit = [&](int64_t c) -> double {
                return gt_read_f64_at(lp, lc.ptr->dtype, (n * C + c) * spatial + sp);
            };
            double mx = -std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < C; ++c) {
                double v = get_logit(c);
                if (v > mx) mx = v;
            }
            double sum_exp = 0.0;
            for (int64_t c = 0; c < C; ++c) sum_exp += std::exp(get_logit(c) - mx);
            double lse = mx + std::log(sum_exp);

            double item_loss = 0.0;
            double item_w = 0.0;
            if (soft_targets) {
                for (int64_t c = 0; c < C; ++c) {
                    double p = gt_read_f64_at(tp, tc.ptr->dtype, (n * C + c) * spatial + sp);
                    double p_smooth = p * (1.0 - label_smoothing) + label_smoothing / (double)C;
                    double cw = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
                    double lsm = get_logit(c) - lse;
                    item_loss -= p_smooth * cw * lsm;
                }
                item_w = 1.0;
            } else {
                int64_t tgt = gt_read_i64_at(tp, tc.ptr->dtype, item_idx);
                if (tgt == ignore_index) {
                    item_loss = 0.0;
                    item_w = 0.0;
                } else {
                    double cw = wp ? gt_read_f64_at(wp, wc.ptr->dtype, tgt) : 1.0;
                    if (label_smoothing > 0.0) {
                        double nll = -(get_logit(tgt) - lse);
                        double mean_lsm = 0.0;
                        double sum_cw_lsm = 0.0;
                        double sum_cw = 0.0;
                        for (int64_t c = 0; c < C; ++c) {
                            double w_c = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
                            sum_cw_lsm += w_c * (get_logit(c) - lse);
                            sum_cw += w_c;
                        }
                        mean_lsm = -sum_cw_lsm / (double)C;
                        item_loss = (1.0 - label_smoothing) * cw * nll + label_smoothing * mean_lsm;
                        item_w = (1.0 - label_smoothing) * cw + label_smoothing * (sum_cw / (double)C);
                    } else {
                        item_loss = -cw * (get_logit(tgt) - lse);
                        item_w = cw;
                    }
                }
            }
            gt_write_f64_at(op, losses->dtype, item_idx, item_loss);
            total_loss += item_loss;
            total_weight += item_w;
        }
    }
    hvo.commit(losses);

    if (reduction == 0) return losses;
    gt_tensor_free(losses);

    gt_tensor * scalar = gt_tensor_empty(0, nullptr, logits->dtype, logits->device_type, logits->device_index);
    if (!scalar) return nullptr;
    double res = (reduction == 1) ? (total_weight > 0.0 ? total_loss / total_weight : 0.0) : total_loss;
    gt_fill_scalar_f64(scalar, res);
    return scalar;
}

extern "C" gt_tensor * gt_cross_entropy_loss_backward(const gt_tensor * grad_out, const gt_tensor * logits, const gt_tensor * targets,
                                                      const gt_tensor * weight, int64_t ignore_index, double label_smoothing, int reduction) {
    if (!grad_out) {
        gt_set_error("cross_entropy: backward needs the gradient of the loss");
        return nullptr;
    }
    gt_ce_layout L;
    if (!gt_cross_entropy_layout(logits, targets, weight, label_smoothing, reduction, L)) return nullptr;
    const bool soft_targets = L.soft;
    const int64_t N = L.N, C = L.C, spatial = L.spatial;
    // one gradient per item without a reduction, one for the reduced loss otherwise
    const int64_t grad_elems = (reduction == 0) ? N * spatial : 1;
    if (grad_out->numel != grad_elems) {
        gt_set_error("cross_entropy: backward expected a gradient of " + std::to_string(grad_elems) + " element(s), got a tensor of shape " +
                     gt_ce_shape_str(grad_out));
        return nullptr;
    }

    gt_contig_owner gc(grad_out), lc(logits), tc(targets), wc(weight);
    gt_host_view hvg, hvl, hvt, hvw, hvd;
    const void * gp = hvg.read(gc.ptr);
    const void * lp = hvl.read(lc.ptr);
    const void * tp = hvt.read(tc.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    if (!soft_targets && !gt_cross_entropy_check_targets(tp, tc.ptr->dtype, N * spatial, C, ignore_index)) return nullptr;

    gt_tensor * dx = gt_tensor_empty(logits->ndim, logits->shape, logits->dtype, logits->device_type, logits->device_index);
    if (!dx) return nullptr;
    void * dxp = hvd.write_empty(dx);

    double total_weight = 0.0;
    if (reduction == 1) {
        for (int64_t idx = 0; idx < N * spatial; ++idx) {
            if (soft_targets) total_weight += 1.0;
            else {
                int64_t tgt = gt_read_i64_at(tp, tc.ptr->dtype, idx);
                if (tgt != ignore_index) {
                    double cw = wp ? gt_read_f64_at(wp, wc.ptr->dtype, tgt) : 1.0;
                    if (label_smoothing > 0.0) {
                        double sum_cw = 0.0;
                        for (int64_t c = 0; c < C; ++c) sum_cw += wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
                        total_weight += (1.0 - label_smoothing) * cw + label_smoothing * (sum_cw / (double)C);
                    } else {
                        total_weight += cw;
                    }
                }
            }
        }
    }

    double go_scalar = (reduction != 0) ? gt_read_f64_at(gp, gc.ptr->dtype, 0) : 1.0;
    double mean_scale = (reduction == 1 && total_weight > 0.0) ? (1.0 / total_weight) : 1.0;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t sp = 0; sp < spatial; ++sp) {
            int64_t item_idx = n * spatial + sp;
            double g_item = (reduction == 0) ? gt_read_f64_at(gp, gc.ptr->dtype, item_idx) : (go_scalar * mean_scale);

            auto get_logit = [&](int64_t c) -> double {
                return gt_read_f64_at(lp, lc.ptr->dtype, (n * C + c) * spatial + sp);
            };
            double mx = -std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < C; ++c) {
                double v = get_logit(c);
                if (v > mx) mx = v;
            }
            double sum_exp = 0.0;
            for (int64_t c = 0; c < C; ++c) sum_exp += std::exp(get_logit(c) - mx);

            if (soft_targets) {
                double sum_q = 0.0;
                std::vector<double> q(C);
                for (int64_t c = 0; c < C; ++c) {
                    double p = gt_read_f64_at(tp, tc.ptr->dtype, (n * C + c) * spatial + sp);
                    double p_sm = p * (1.0 - label_smoothing) + label_smoothing / (double)C;
                    double cw = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
                    q[c] = p_sm * cw;
                    sum_q += q[c];
                }
                for (int64_t c = 0; c < C; ++c) {
                    double prob = std::exp(get_logit(c) - mx) / sum_exp;
                    gt_write_f64_at(dxp, dx->dtype, (n * C + c) * spatial + sp, g_item * (prob * sum_q - q[c]));
                }
            } else {
                int64_t tgt = gt_read_i64_at(tp, tc.ptr->dtype, item_idx);
                if (tgt == ignore_index) {
                    for (int64_t c = 0; c < C; ++c) {
                        gt_write_f64_at(dxp, dx->dtype, (n * C + c) * spatial + sp, 0.0);
                    }
                } else {
                    double sum_q = 0.0;
                    std::vector<double> q(C, 0.0);
                    for (int64_t c = 0; c < C; ++c) {
                        double cw = wp ? gt_read_f64_at(wp, wc.ptr->dtype, c) : 1.0;
                        double t_prob = (c == tgt ? (1.0 - label_smoothing) : 0.0) + label_smoothing / (double)C;
                        q[c] = t_prob * cw;
                        sum_q += q[c];
                    }
                    for (int64_t c = 0; c < C; ++c) {
                        double prob = std::exp(get_logit(c) - mx) / sum_exp;
                        gt_write_f64_at(dxp, dx->dtype, (n * C + c) * spatial + sp, g_item * (prob * sum_q - q[c]));
                    }
                }
            }
        }
    }
    hvd.commit(dx);
    return dx;
}

// ---------- RoPE, Flash Attention, SwiGLU (ggml) ----------

struct gt_rope_layout {
    int64_t B = 1;  // batch
    int64_t H = 0;  // heads
    int64_t S = 0;  // sequence length
    int64_t D = 0;  // head dimension
};

static std::string gt_rope_shape_str(const gt_tensor * t) {
    std::string s = "[";
    for (int i = 0; i < t->ndim; ++i) s += (i ? ", " : "") + std::to_string(t->shape[i]);
    return s + "]";
}

// ggml checks its RoPE arguments with GGML_ASSERT, which ends the process, takes the number of
// positions on trust, and with n_dims <= 0 never returns. Everything it would trip over is
// turned into an error here, for the forward and the backward pass alike.
static bool gt_rope_check(const gt_tensor * x, const gt_tensor * pos, int n_dims, int mode, float freq_base, float freq_scale,
                          gt_rope_layout & L) {
    if (!x || !pos) {
        gt_set_error("rope: x and pos are required");
        return false;
    }
    if (x->ndim < 3 || x->ndim > 4) {
        gt_set_error("rope: expected x of shape (batch, heads, seq_len, head_dim) or (heads, seq_len, head_dim), got " +
                     gt_rope_shape_str(x));
        return false;
    }
    if (!gt_is_floating(x->dtype)) {
        gt_set_error("rope: x must be a floating point tensor");
        return false;
    }
    L.B = (x->ndim == 4) ? x->shape[0] : 1;
    L.H = (x->ndim == 4) ? x->shape[1] : x->shape[0];
    L.S = (x->ndim == 4) ? x->shape[2] : x->shape[1];
    L.D = (x->ndim == 4) ? x->shape[3] : x->shape[2];

    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        gt_set_error("rope: mode must be 0 (normal) or 2 (NeoX), got " + std::to_string(mode) +
                     "; the multi-dimensional modes need per-section position ids and are not supported");
        return false;
    }
    if (n_dims < 2 || n_dims % 2 != 0 || n_dims > L.D) {
        gt_set_error("rope: n_dims must be an even number between 2 and the head dimension " + std::to_string(L.D) + ", got " +
                     std::to_string(n_dims));
        return false;
    }
    if (!(freq_base > 0.0f) || !std::isfinite(freq_base) || !std::isfinite(freq_scale)) {
        gt_set_error("rope: freq_base must be positive and finite, and freq_scale finite");
        return false;
    }
    if (pos->ndim != 1 || pos->shape[0] < L.S) {
        gt_set_error("rope: pos must be a 1-D tensor with a position for each of the " + std::to_string(L.S) +
                     " sequence elements, got shape " + gt_rope_shape_str(pos));
        return false;
    }
    return true;
}

// Rotate x (forward) or rotate a gradient back (backward). ggml computes in float32; an input
// of another floating type is converted, and the result is cast back so that it has the dtype
// of the input.
static gt_tensor * gt_rope_run(const gt_tensor * x, const gt_tensor * pos, const gt_rope_layout & L, int n_dims, int mode,
                               int n_ctx_orig, float freq_base, float freq_scale, bool backward) {
    if (x->numel == 0) return gt_tensor_empty(x->ndim, x->shape, x->dtype, x->device_type, x->device_index);

    gt_backend_entry * be = gt_get_backend(x->device_type, x->device_index);
    if (!be) {
        gt_set_error("rope: no backend for the device of x");
        return nullptr;
    }
    gt_tensor * out = gt_tensor_empty(x->ndim, x->shape, GT_F32, x->device_type, x->device_index);
    if (!out) return nullptr;

    gt_tensor * xf = (x->dtype == GT_F32 && gt_check_contiguous(x) && x->offset == 0) ? nullptr : gt_tensor_to_dtype(x, GT_F32);
    const gt_tensor * x_use = xf ? xf : x;
    gt_tensor * pf = (pos->dtype == GT_I32 && gt_check_contiguous(pos) && pos->offset == 0) ? nullptr : gt_tensor_to_dtype(pos, GT_I32);
    const gt_tensor * p_use = pf ? pf : pos;

    bool ok = false;
    gt_set_error("");
    struct ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx = (x_use && p_use) ? ggml_init(params) : nullptr;
    if (ctx) {
        // ggml wants [head_dim, heads, seq_len, batch]; the first seq_len positions are used
        struct ggml_tensor * gx = gt_wrap_ggml_4d(ctx, x_use, L.D, L.S, L.H, L.B);
        struct ggml_tensor * gp = gt_wrap_ggml_4d(ctx, p_use, L.S, 1, 1, 1);
        struct ggml_tensor * gx_p = ggml_cont(ctx, ggml_permute(ctx, gx, 0, 2, 1, 3));
        struct ggml_tensor * g_roped = backward
            ? ggml_rope_ext_back(ctx, gx_p, gp, nullptr, n_dims, mode, n_ctx_orig, freq_base, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f)
            : ggml_rope_ext(ctx, gx_p, gp, nullptr, n_dims, mode, n_ctx_orig, freq_base, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * gout = ggml_cont(ctx, ggml_permute(ctx, g_roped, 0, 2, 1, 3));
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gout);
        ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
        ggml_free(ctx);
    }
    if (xf) gt_tensor_free(xf);
    if (pf) gt_tensor_free(pf);
    if (!ok) {
        const std::string detail = gt_last_error();
        gt_tensor_free(out);
        gt_set_error("rope: " + (detail.empty() ? std::string("ggml execution failed") : detail));
        return nullptr;
    }
    if (x->dtype == GT_F32) return out;
    gt_tensor * cast = gt_tensor_to_dtype(out, x->dtype);
    gt_tensor_free(out);
    return cast;
}

extern "C" gt_tensor * gt_rope(const gt_tensor * x, const gt_tensor * pos, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale) {
    // x: (batch, heads, seq_len, head_dim) or (heads, seq_len, head_dim); pos: (seq_len,) or longer
    gt_rope_layout L;
    if (!gt_rope_check(x, pos, n_dims, mode, freq_base, freq_scale, L)) return nullptr;
    return gt_rope_run(x, pos, L, n_dims, mode, n_ctx_orig, freq_base, freq_scale, false);
}

extern "C" gt_tensor * gt_rope_backward(const gt_tensor * grad_out, const gt_tensor * pos, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale) {
    gt_rope_layout L;
    if (!gt_rope_check(grad_out, pos, n_dims, mode, freq_base, freq_scale, L)) return nullptr;
    return gt_rope_run(grad_out, pos, L, n_dims, mode, n_ctx_orig, freq_base, freq_scale, true);
}

extern "C" gt_tensor * gt_flash_attention(const gt_tensor * q, const gt_tensor * k, const gt_tensor * v, const gt_tensor * mask, float scale, int is_causal) {
    // q: (B, H, L, D), k: (B, H_kv, S, D), v: (B, H_kv, S, Dv)
    if (!q || !k || !v || q->ndim != 4 || k->ndim != 4 || v->ndim != 4) {
        gt_set_error("flash_attention expects 4D q, k, v tensors (B, H, L, D)");
        return nullptr;
    }
    int64_t B = q->shape[0], H = q->shape[1], L = q->shape[2], D = q->shape[3];
    int64_t H_kv = k->shape[1], S = k->shape[2], Dv = v->shape[3];
    if (H % H_kv != 0) { gt_set_error("flash_attention: num_heads must be divisible by num_kv_heads"); return nullptr; }
    int64_t group_sz = H / H_kv;
    if (scale <= 0.0f) scale = 1.0f / std::sqrt((float)D);
    if (k->shape[0] != B || v->shape[0] != B || v->shape[1] != H_kv || v->shape[2] != S || k->shape[3] != D) {
        gt_set_error("flash_attention: q (B, H, L, D), k (B, H_kv, S, D) and v (B, H_kv, S, Dv) do not fit together");
        return nullptr;
    }
    if (mask) {
        // (L, S) or (B, H, L, S), every dimension either 1 or the full size: the loop below
        // indexes the mask by these sizes
        const int64_t want[4] = {B, H, L, S};
        bool fits = mask->ndim == 2 || mask->ndim == 4;
        for (int i = 0; fits && i < mask->ndim; ++i) {
            const int64_t m = mask->shape[i];
            fits = (m == 1 || m == want[4 - mask->ndim + i]);
        }
        if (!fits) {
            std::string shape = "[";
            for (int i = 0; i < mask->ndim; ++i) shape += (i ? ", " : "") + std::to_string(mask->shape[i]);
            gt_set_error("flash_attention: attn_mask of shape " + shape + "] cannot be broadcast to the attention weights [" +
                         std::to_string(B) + ", " + std::to_string(H) + ", " + std::to_string(L) + ", " + std::to_string(S) + "]");
            return nullptr;
        }
    }

    int64_t osh[4] = {B, H, L, Dv};
    gt_tensor * out = gt_tensor_empty(4, osh, q->dtype, q->device_type, q->device_index);
    if (!out) return nullptr;

    gt_contig_owner qc(q), kc(k), vc(v), mc(mask);
    gt_host_view hvq, hvk, hvv, hvm, hvo;
    const void * qp = hvq.read(qc.ptr);
    const void * kp = hvk.read(kc.ptr);
    const void * vp = hvv.read(vc.ptr);
    const void * mp = mc.ptr ? hvm.read(mc.ptr) : nullptr;
    void * op = hvo.write_empty(out);

    #pragma omp parallel for collapse(2) if (B * H > 4)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t h_kv = h / group_sz;
            std::vector<double> scores(S);
            for (int64_t l = 0; l < L; ++l) {
                double mx = -std::numeric_limits<double>::infinity();
                for (int64_t s = 0; s < S; ++s) {
                    if (is_causal && s > l) {
                        scores[s] = -std::numeric_limits<double>::infinity();
                        continue;
                    }
                    double dot = 0.0;
                    int64_t q_off = ((b * H + h) * L + l) * D;
                    int64_t k_off = ((b * H_kv + h_kv) * S + s) * D;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += gt_read_f64_at(qp, qc.ptr->dtype, q_off + d) *
                               gt_read_f64_at(kp, kc.ptr->dtype, k_off + d);
                    }
                    dot *= (double)scale;
                    if (mp) {
                        // Support broadcast mask of (L, S), (1, 1, L, S), (B, 1, L, S), (B, H, L, S)
                        int64_t mb = (mc.ptr->ndim == 4 && mc.ptr->shape[0] > 1) ? b : 0;
                        int64_t mh = (mc.ptr->ndim == 4 && mc.ptr->shape[1] > 1) ? h : 0;
                        int64_t m_idx = 0;
                        if (mc.ptr->ndim == 2) m_idx = (mc.ptr->shape[0] > 1 ? l : 0) * mc.ptr->shape[1] + (mc.ptr->shape[1] > 1 ? s : 0);
                        else if (mc.ptr->ndim == 4) m_idx = ((mb * mc.ptr->shape[1] + mh) * mc.ptr->shape[2] + (mc.ptr->shape[2] > 1 ? l : 0)) * mc.ptr->shape[3] + (mc.ptr->shape[3] > 1 ? s : 0);
                        if (mc.ptr->dtype == GT_BOOL) {
                            if (gt_read_i64_at(mp, GT_BOOL, m_idx) == 0) dot = -std::numeric_limits<double>::infinity();
                        } else {
                            dot += gt_read_f64_at(mp, mc.ptr->dtype, m_idx);
                        }
                    }
                    scores[s] = dot;
                    if (dot > mx) mx = dot;
                }
                double sum_e = 0.0;
                for (int64_t s = 0; s < S; ++s) {
                    if (std::isinf(scores[s]) && scores[s] < 0) scores[s] = 0.0;
                    else { scores[s] = std::exp(scores[s] - mx); sum_e += scores[s]; }
                }
                double inv_se = sum_e > 0.0 ? (1.0 / sum_e) : 0.0;
                int64_t o_off = ((b * H + h) * L + l) * Dv;
                for (int64_t dv = 0; dv < Dv; ++dv) {
                    double acc = 0.0;
                    for (int64_t s = 0; s < S; ++s) {
                        double p = scores[s] * inv_se;
                        if (p == 0.0) continue;
                        acc += p * gt_read_f64_at(vp, vc.ptr->dtype, ((b * H_kv + h_kv) * S + s) * Dv + dv);
                    }
                    gt_write_f64_at(op, out->dtype, o_off + dv, acc);
                }
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_swiglu(const gt_tensor * x, const gt_tensor * up) {
    if (!x || !up) return nullptr;
    gt_tensor * sx = gt_unary_op(x, GT_UNARY_SILU, 0, 0);
    if (!sx) return nullptr;
    gt_tensor * res = gt_binary_op(sx, up, GT_BIN_MUL, 1.0);
    gt_tensor_free(sx);
    return res;
}

// ---------- Optimizers (ggml_opt_step_adamw / ggml_opt_step_sgd) ----------

extern "C" int gt_adamw_step(gt_tensor * param, const gt_tensor * grad, gt_tensor * exp_avg, gt_tensor * exp_avg_sq, gt_tensor * max_exp_avg_sq,
                             double lr, double beta1, double beta2, double eps, double weight_decay, int64_t step, int amsgrad, int decoupled_wd) {
    if (!param || !grad || !exp_avg || !exp_avg_sq) return -1;
    if (param->numel == 0 || param->device_type == GT_DEV_META) return 0;

    // Native ggml_opt_step_adamw fast path for F32 contiguous decoupled AdamW without amsgrad
    if (!amsgrad && decoupled_wd && param->dtype == GT_F32 && grad->dtype == GT_F32 &&
        exp_avg->dtype == GT_F32 && exp_avg_sq->dtype == GT_F32 &&
        gt_check_contiguous(param) && gt_check_contiguous(grad) &&
        gt_check_contiguous(exp_avg) && gt_check_contiguous(exp_avg_sq) &&
        param->offset == 0 && grad->offset == 0 && exp_avg->offset == 0 && exp_avg_sq->offset == 0 &&
        param->device_type == grad->device_type) {
        gt_backend_entry * be = gt_get_backend(param->device_type, param->device_index);
        if (be) {
            float pars[7];
            pars[0] = (float)lr;
            pars[1] = (float)beta1;
            pars[2] = (float)beta2;
            pars[3] = (float)eps;
            pars[4] = (float)weight_decay;
            pars[5] = 1.0f / (1.0f - std::pow((float)beta1, (float)step));
            pars[6] = 1.0f / (1.0f - std::pow((float)beta2, (float)step));

            int64_t psh[1] = {7};
            gt_tensor * pars_t = gt_tensor_from_data(1, psh, GT_F32, GT_DEV_CPU, 0, pars, sizeof(pars));
            struct ggml_init_params init_p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
            struct ggml_context * ctx = ggml_init(init_p);
            if (ctx && pars_t) {
                int64_t n = param->numel;
                struct ggml_tensor * gp = gt_wrap_ggml_4d(ctx, param, n, 1, 1, 1);
                ggml_set_param(gp);
                struct ggml_tensor * gg = gt_wrap_ggml_4d(ctx, grad, n, 1, 1, 1);
                struct ggml_tensor * gm = gt_wrap_ggml_4d(ctx, exp_avg, n, 1, 1, 1);
                struct ggml_tensor * gv = gt_wrap_ggml_4d(ctx, exp_avg_sq, n, 1, 1, 1);
                struct ggml_tensor * gpar = gt_wrap_ggml_4d(ctx, pars_t, 7, 1, 1, 1);
                struct ggml_tensor * opt_node = ggml_opt_step_adamw(ctx, gp, gg, gm, gv, gpar);
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, opt_node);
                {
                    std::lock_guard<std::mutex> lock(be->mtx);
                    enum ggml_status st = ggml_backend_graph_compute(be->backend, gf);
                    ggml_free(ctx);
                    gt_tensor_free(pars_t);
                    if (st == GGML_STATUS_SUCCESS) return 0;
                }
            } else {
                if (ctx) ggml_free(ctx);
                if (pars_t) gt_tensor_free(pars_t);
            }
        }
    }

    gt_host_view hvp, hvg, hvm, hvv, hvmx;
    void * pp = hvp.read_write(param);
    const void * gp = hvg.read(grad);
    void * mp = hvm.read_write(exp_avg);
    void * vp = hvv.read_write(exp_avg_sq);
    void * mxp = (amsgrad && max_exp_avg_sq) ? hvmx.read_write(max_exp_avg_sq) : nullptr;

    double bias_corr1 = 1.0 - std::pow(beta1, (double)step);
    double bias_corr2 = 1.0 - std::pow(beta2, (double)step);
    bool p_contig = gt_check_contiguous(param), g_contig = gt_check_contiguous(grad);
    // Every tensor is addressed through its own offset and strides - the state buffers too
    bool m_contig = gt_check_contiguous(exp_avg), v_contig = gt_check_contiguous(exp_avg_sq);
    bool mx_contig = mxp ? gt_check_contiguous(max_exp_avg_sq) : true;

    #pragma omp parallel for if (param->numel > 4096)
    for (int64_t i = 0; i < param->numel; ++i) {
        int64_t pidx = p_contig ? (param->offset + i) : gt_linear_to_elem_offset(param, i);
        int64_t gidx = g_contig ? (grad->offset + i) : gt_linear_to_elem_offset(grad, i);
        int64_t midx = m_contig ? (exp_avg->offset + i) : gt_linear_to_elem_offset(exp_avg, i);
        int64_t vidx = v_contig ? (exp_avg_sq->offset + i) : gt_linear_to_elem_offset(exp_avg_sq, i);
        double w = gt_read_f64_at(pp, param->dtype, pidx);
        double g = gt_read_f64_at(gp, grad->dtype, gidx);
        if (decoupled_wd) {
            if (weight_decay != 0.0) w *= (1.0 - lr * weight_decay);
        } else {
            if (weight_decay != 0.0) g += weight_decay * w;
        }
        double m = beta1 * gt_read_f64_at(mp, exp_avg->dtype, midx) + (1.0 - beta1) * g;
        double v = beta2 * gt_read_f64_at(vp, exp_avg_sq->dtype, vidx) + (1.0 - beta2) * g * g;
        gt_write_f64_at(mp, exp_avg->dtype, midx, m);
        gt_write_f64_at(vp, exp_avg_sq->dtype, vidx, v);

        double denom;
        if (amsgrad && mxp) {
            int64_t mxidx = mx_contig ? (max_exp_avg_sq->offset + i) : gt_linear_to_elem_offset(max_exp_avg_sq, i);
            double vmax = std::max(gt_read_f64_at(mxp, max_exp_avg_sq->dtype, mxidx), v);
            gt_write_f64_at(mxp, max_exp_avg_sq->dtype, mxidx, vmax);
            denom = std::sqrt(vmax / bias_corr2) + eps;
        } else {
            denom = std::sqrt(v / bias_corr2) + eps;
        }
        w -= (lr / bias_corr1) * (m / denom);
        gt_write_f64_at(pp, param->dtype, pidx, w);
    }

    hvp.commit(param);
    hvm.commit(exp_avg);
    hvv.commit(exp_avg_sq);
    if (amsgrad && max_exp_avg_sq) hvmx.commit(max_exp_avg_sq);
    return 0;
}

extern "C" int gt_sgd_step(gt_tensor * param, const gt_tensor * grad, gt_tensor * momentum_buf,
                           double lr, double momentum, double dampening, double weight_decay, int nesterov, int first_step) {
    if (!param || !grad) return -1;
    if (param->numel == 0 || param->device_type == GT_DEV_META) return 0;

    // Native ggml_opt_step_sgd fast path when momentum == 0 and F32 contiguous
    if (momentum == 0.0 && param->dtype == GT_F32 && grad->dtype == GT_F32 &&
        gt_check_contiguous(param) && gt_check_contiguous(grad) &&
        param->offset == 0 && grad->offset == 0 && param->device_type == grad->device_type) {
        gt_backend_entry * be = gt_get_backend(param->device_type, param->device_index);
        if (be) {
            float pars[2] = {(float)lr, (float)weight_decay};
            int64_t psh[1] = {2};
            gt_tensor * pars_t = gt_tensor_from_data(1, psh, GT_F32, GT_DEV_CPU, 0, pars, sizeof(pars));
            struct ggml_init_params init_p = { ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true };
            struct ggml_context * ctx = ggml_init(init_p);
            if (ctx && pars_t) {
                int64_t n = param->numel;
                struct ggml_tensor * gp = gt_wrap_ggml_4d(ctx, param, n, 1, 1, 1);
                ggml_set_param(gp);
                struct ggml_tensor * gg = gt_wrap_ggml_4d(ctx, grad, n, 1, 1, 1);
                struct ggml_tensor * gpar = gt_wrap_ggml_4d(ctx, pars_t, 2, 1, 1, 1);
                struct ggml_tensor * opt_node = ggml_opt_step_sgd(ctx, gp, gg, gpar);
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, opt_node);
                {
                    std::lock_guard<std::mutex> lock(be->mtx);
                    enum ggml_status st = ggml_backend_graph_compute(be->backend, gf);
                    ggml_free(ctx);
                    gt_tensor_free(pars_t);
                    if (st == GGML_STATUS_SUCCESS) return 0;
                }
            } else {
                if (ctx) ggml_free(ctx);
                if (pars_t) gt_tensor_free(pars_t);
            }
        }
    }

    gt_host_view hvp, hvg, hvm;
    void * pp = hvp.read_write(param);
    const void * gp = hvg.read(grad);
    void * mp = momentum_buf ? hvm.read_write(momentum_buf) : nullptr;
    bool p_contig = gt_check_contiguous(param), g_contig = gt_check_contiguous(grad);
    bool m_contig = mp ? gt_check_contiguous(momentum_buf) : true;

    #pragma omp parallel for if (param->numel > 4096)
    for (int64_t i = 0; i < param->numel; ++i) {
        int64_t pidx = p_contig ? (param->offset + i) : gt_linear_to_elem_offset(param, i);
        int64_t gidx = g_contig ? (grad->offset + i) : gt_linear_to_elem_offset(grad, i);
        double w = gt_read_f64_at(pp, param->dtype, pidx);
        double d_p = gt_read_f64_at(gp, grad->dtype, gidx);
        if (weight_decay != 0.0) d_p += weight_decay * w;
        if (momentum != 0.0 && mp) {
            int64_t midx = m_contig ? (momentum_buf->offset + i) : gt_linear_to_elem_offset(momentum_buf, i);
            double buf;
            if (first_step) {
                buf = d_p;
            } else {
                buf = gt_read_f64_at(mp, momentum_buf->dtype, midx) * momentum + (1.0 - dampening) * d_p;
            }
            gt_write_f64_at(mp, momentum_buf->dtype, midx, buf);
            d_p = nesterov ? (d_p + momentum * buf) : buf;
        }
        w -= lr * d_p;
        gt_write_f64_at(pp, param->dtype, pidx, w);
    }
    hvp.commit(param);
    if (momentum_buf) hvm.commit(momentum_buf);
    return 0;
}
