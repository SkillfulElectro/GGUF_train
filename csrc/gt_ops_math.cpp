#include "gt_internal.h"
#include <algorithm>
#include <limits>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double gt_digamma_scalar(double x) {
    double res = 0.0;
    if (x <= 0.0) {
        if (x == std::floor(x)) return std::numeric_limits<double>::quiet_NaN();
        res -= M_PI / std::tan(M_PI * x);
        x = 1.0 - x;
    }
    while (x < 8.0) {
        res -= 1.0 / x;
        x += 1.0;
    }
    double r = 1.0 / x;
    double r2 = r * r;
    res += std::log(x) - 0.5 * r - r2 * (1.0 / 12.0 - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 / 240.0)));
    return res;
}

static double gt_erfinv_scalar(double y) {
    if (y < -1.0 || y > 1.0) return std::numeric_limits<double>::quiet_NaN();
    if (y == 1.0) return std::numeric_limits<double>::infinity();
    if (y == -1.0) return -std::numeric_limits<double>::infinity();
    if (y == 0.0) return 0.0;
    double s = (y < 0.0) ? -1.0 : 1.0;
    double ay = std::abs(y);
    double x;
    if (ay <= 0.7) {
        double z = ay * ay;
        double num = (((-0.140543331 * z + 0.914624893) * z - 1.645349621) * z + 0.886226899);
        double den = ((((0.012229801 * z - 0.329097515) * z + 1.442710462) * z - 2.118377725) * z + 1.0);
        x = ay * num / den;
    } else {
        double z = std::sqrt(-std::log((1.0 - ay) * 0.5));
        double num = (((1.641345311 * z + 3.429567803) * z - 1.624906493) * z - 1.970840454);
        double den = ((1.637067800 * z + 3.543889200) * z + 1.0);
        x = num / den;
    }
    x *= s;
    const double two_over_sqrt_pi = 2.0 / std::sqrt(M_PI);
    for (int it = 0; it < 3; ++it) {
        double err = std::erf(x) - y;
        x -= err / (two_over_sqrt_pi * std::exp(-x * x));
    }
    return x;
}

static int gt_unary_out_dtype(int in_dtype, int op_id) {
    switch (op_id) {
        case GT_UNARY_ISNAN:
        case GT_UNARY_ISINF:
        case GT_UNARY_ISFINITE:
        case GT_UNARY_LOGICAL_NOT:
            return GT_BOOL;
        case GT_UNARY_REAL:
        case GT_UNARY_IMAG:
        case GT_UNARY_ANGLE:
        case GT_UNARY_ABS:
            if (in_dtype == GT_C64) return GT_F32;
            if (in_dtype == GT_C128) return GT_F64;
            return in_dtype;
        case GT_UNARY_CONJ:
            return in_dtype;
        default:
            if (gt_is_integer(in_dtype) || in_dtype == GT_BOOL) {
                switch (op_id) {
                    case GT_UNARY_NEG:
                    case GT_UNARY_ABS:
                    case GT_UNARY_SIGN:
                    case GT_UNARY_SQUARE:
                    case GT_UNARY_FLOOR:
                    case GT_UNARY_CEIL:
                    case GT_UNARY_ROUND:
                    case GT_UNARY_TRUNC:
                    case GT_UNARY_BITWISE_NOT:
                    case GT_UNARY_CLAMP:
                    case GT_UNARY_RELU:
                    case GT_UNARY_RELU6:
                    case GT_UNARY_HARDTANH:
                        return in_dtype;
                    default:
                        return GT_F32;
                }
            }
            return in_dtype;
    }
}

static bool gt_try_ggml_unary(const gt_tensor * a, gt_tensor * out, int op_id, double p1, double p2) {
    if (a->dtype != GT_F32 || out->dtype != GT_F32) return false;
    if (a->device_type == GT_DEV_META || a->numel == 0) return false;
    if (!gt_check_contiguous(a) || !gt_check_contiguous(out)) return false;
    if (a->device_type != out->device_type || a->device_index != out->device_index) return false;

    // Check if ggml supports this exact unary op
    bool supported = false;
    switch (op_id) {
        case GT_UNARY_NEG:
        case GT_UNARY_ABS:
        case GT_UNARY_SIGN:
        case GT_UNARY_SQRT:
        case GT_UNARY_SQUARE:
        case GT_UNARY_EXP:
        case GT_UNARY_EXPM1:
        case GT_UNARY_LOG:
        case GT_UNARY_SIN:
        case GT_UNARY_COS:
        case GT_UNARY_TANH:
        case GT_UNARY_FLOOR:
        case GT_UNARY_CEIL:
        case GT_UNARY_ROUND:
        case GT_UNARY_TRUNC:
        case GT_UNARY_RELU:
        case GT_UNARY_SIGMOID:
        case GT_UNARY_SILU:
        case GT_UNARY_GELU:
        case GT_UNARY_GELU_TANH:
        case GT_UNARY_GELU_QUICK:
        case GT_UNARY_HARDSWISH:
        case GT_UNARY_HARDSIGMOID:
        case GT_UNARY_LEAKY_RELU:
        case GT_UNARY_CLAMP:
            supported = true;
            break;
        case GT_UNARY_ELU:
            supported = (std::abs(p1 - 1.0) < 1e-6);
            break;
        case GT_UNARY_SOFTPLUS:
            supported = (std::abs(p1 - 1.0) < 1e-6 && std::abs(p2 - 20.0) < 1e-6);
            break;
        default:
            supported = false;
            break;
    }
    if (!supported) return false;

    gt_backend_entry * be = gt_get_backend(a->device_type, a->device_index);
    if (!be) return false;

    // Shape into 2D [cols, rows] so ggml threads parallelize cleanly on large 1D tensors too
    int64_t n = a->numel;
    int64_t ne0 = n, ne1 = 1;
    if (n >= 4096) {
        for (int64_t d = 256; d <= 1024; d *= 2) {
            if (n % d == 0) { ne0 = d; ne1 = n / d; break; }
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, ne0, ne1, 1, 1);
    struct ggml_tensor * gout = nullptr;
    switch (op_id) {
        case GT_UNARY_NEG:         gout = ggml_neg(ctx, ga); break;
        case GT_UNARY_ABS:         gout = ggml_abs(ctx, ga); break;
        case GT_UNARY_SIGN:        gout = ggml_sgn(ctx, ga); break;
        case GT_UNARY_SQRT:        gout = ggml_sqrt(ctx, ga); break;
        case GT_UNARY_SQUARE:      gout = ggml_sqr(ctx, ga); break;
        case GT_UNARY_EXP:         gout = ggml_exp(ctx, ga); break;
        case GT_UNARY_EXPM1:       gout = ggml_expm1(ctx, ga); break;
        case GT_UNARY_LOG:         gout = ggml_log(ctx, ga); break;
        case GT_UNARY_SIN:         gout = ggml_sin(ctx, ga); break;
        case GT_UNARY_COS:         gout = ggml_cos(ctx, ga); break;
        case GT_UNARY_TANH:        gout = ggml_tanh(ctx, ga); break;
        case GT_UNARY_ELU:         gout = ggml_elu(ctx, ga); break;
        case GT_UNARY_FLOOR:       gout = ggml_floor(ctx, ga); break;
        case GT_UNARY_CEIL:        gout = ggml_ceil(ctx, ga); break;
        case GT_UNARY_ROUND:       gout = ggml_round(ctx, ga); break;
        case GT_UNARY_TRUNC:       gout = ggml_trunc(ctx, ga); break;
        case GT_UNARY_RELU:        gout = ggml_relu(ctx, ga); break;
        case GT_UNARY_SIGMOID:     gout = ggml_sigmoid(ctx, ga); break;
        case GT_UNARY_SILU:        gout = ggml_silu(ctx, ga); break;
        case GT_UNARY_GELU:        gout = ggml_gelu_erf(ctx, ga); break;
        case GT_UNARY_GELU_TANH:   gout = ggml_gelu(ctx, ga); break;
        case GT_UNARY_GELU_QUICK:  gout = ggml_gelu_quick(ctx, ga); break;
        case GT_UNARY_SOFTPLUS:    gout = ggml_softplus(ctx, ga); break;
        case GT_UNARY_HARDSWISH:   gout = ggml_hardswish(ctx, ga); break;
        case GT_UNARY_HARDSIGMOID: gout = ggml_hardsigmoid(ctx, ga); break;
        case GT_UNARY_LEAKY_RELU:  gout = ggml_leaky_relu(ctx, ga, (float)p1, false); break;
        case GT_UNARY_CLAMP:       gout = ggml_clamp(ctx, ga, (float)p1, (float)p2); break;
        default: break;
    }
    if (!gout) {
        ggml_free(ctx);
        return false;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, gout);
    bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
    ggml_free(ctx);
    return ok;
}

static inline double gt_apply_unary_scalar(double x, int op_id, double p1, double p2) {
    switch (op_id) {
        case GT_UNARY_NEG: return -x;
        case GT_UNARY_ABS: return std::abs(x);
        case GT_UNARY_SIGN: return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
        case GT_UNARY_SQRT: return std::sqrt(x);
        case GT_UNARY_RSQRT: return 1.0 / std::sqrt(x);
        case GT_UNARY_SQUARE: return x * x;
        case GT_UNARY_EXP: return std::exp(x);
        case GT_UNARY_EXPM1: return std::expm1(x);
        case GT_UNARY_EXP2: return std::exp2(x);
        case GT_UNARY_LOG: return std::log(x);
        case GT_UNARY_LOG2: return std::log2(x);
        case GT_UNARY_LOG10: return std::log10(x);
        case GT_UNARY_LOG1P: return std::log1p(x);
        case GT_UNARY_SIN: return std::sin(x);
        case GT_UNARY_COS: return std::cos(x);
        case GT_UNARY_TAN: return std::tan(x);
        case GT_UNARY_ASIN: return std::asin(x);
        case GT_UNARY_ACOS: return std::acos(x);
        case GT_UNARY_ATAN: return std::atan(x);
        case GT_UNARY_SINH: return std::sinh(x);
        case GT_UNARY_COSH: return std::cosh(x);
        case GT_UNARY_TANH: return std::tanh(x);
        case GT_UNARY_ASINH: return std::asinh(x);
        case GT_UNARY_ACOSH: return std::acosh(x);
        case GT_UNARY_ATANH: return std::atanh(x);
        case GT_UNARY_ERF: return std::erf(x);
        case GT_UNARY_ERFC: return std::erfc(x);
        case GT_UNARY_ERFINV: return gt_erfinv_scalar(x);
        case GT_UNARY_LGAMMA: return std::lgamma(x);
        case GT_UNARY_DIGAMMA: return gt_digamma_scalar(x);
        case GT_UNARY_FLOOR: return std::floor(x);
        case GT_UNARY_CEIL: return std::ceil(x);
        case GT_UNARY_ROUND: return std::nearbyint(x);
        case GT_UNARY_TRUNC: return std::trunc(x);
        case GT_UNARY_FRAC: return x - std::trunc(x);
        case GT_UNARY_RECIPROCAL: return 1.0 / x;
        case GT_UNARY_RELU: return x > 0.0 ? x : 0.0;
        case GT_UNARY_RELU6: return std::min(std::max(x, 0.0), 6.0);
        case GT_UNARY_SIGMOID: return 1.0 / (1.0 + std::exp(-x));
        case GT_UNARY_SILU: return x / (1.0 + std::exp(-x));
        case GT_UNARY_GELU: return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
        case GT_UNARY_GELU_TANH: {
            const double k = 0.7978845608028654;
            return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
        }
        case GT_UNARY_GELU_QUICK: return x / (1.0 + std::exp(-1.702 * x));
        case GT_UNARY_ELU: return x >= 0.0 ? x : p1 * std::expm1(x);
        case GT_UNARY_CELU: return std::max(0.0, x) + std::min(0.0, p1 * std::expm1(x / p1));
        case GT_UNARY_SELU: {
            const double alpha = 1.6732632423543772848170429916717;
            const double scale = 1.0507009873554804934193349852946;
            return scale * (x >= 0.0 ? x : alpha * std::expm1(x));
        }
        case GT_UNARY_LEAKY_RELU: return x >= 0.0 ? x : p1 * x;
        case GT_UNARY_SOFTPLUS: {
            double bx = p1 * x;
            if (bx > p2) return x;
            return std::log1p(std::exp(bx)) / p1;
        }
        case GT_UNARY_MISH: {
            double sp = (x > 20.0) ? x : std::log1p(std::exp(x));
            return x * std::tanh(sp);
        }
        case GT_UNARY_HARDSWISH: {
            if (x <= -3.0) return 0.0;
            if (x >= 3.0) return x;
            return x * (x + 3.0) / 6.0;
        }
        case GT_UNARY_HARDSIGMOID: {
            if (x <= -3.0) return 0.0;
            if (x >= 3.0) return 1.0;
            return (x + 3.0) / 6.0;
        }
        case GT_UNARY_HARDTANH: return std::min(std::max(x, p1), p2);
        case GT_UNARY_LOG_SIGMOID: return (x >= 0.0) ? -std::log1p(std::exp(-x)) : (x - std::log1p(std::exp(x)));
        case GT_UNARY_CLAMP: {
            double v = x;
            if (!std::isnan(p1) && v < p1) v = p1;
            if (!std::isnan(p2) && v > p2) v = p2;
            return v;
        }
        case GT_UNARY_ISNAN: return std::isnan(x) ? 1.0 : 0.0;
        case GT_UNARY_ISINF: return std::isinf(x) ? 1.0 : 0.0;
        case GT_UNARY_ISFINITE: return std::isfinite(x) ? 1.0 : 0.0;
        case GT_UNARY_LOGICAL_NOT: return (x == 0.0) ? 1.0 : 0.0;
        case GT_UNARY_NAN_TO_NUM: {
            if (std::isnan(x)) return p1;
            if (std::isinf(x)) return (x > 0.0) ? p2 : -p2;
            return x;
        }
        case GT_UNARY_SILU_DERIV: {
            double s = 1.0 / (1.0 + std::exp(-x));
            return s * (1.0 + x * (1.0 - s));
        }
        case GT_UNARY_GELU_DERIV: {
            const double inv_sqrt2 = 0.70710678118654752440;
            const double inv_sqrt2pi = 0.39894228040143267794;
            double cdf = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
            double pdf = inv_sqrt2pi * std::exp(-0.5 * x * x);
            return cdf + x * pdf;
        }
        case GT_UNARY_GELU_TANH_DERIV: {
            const double k = 0.7978845608028654;
            double x2 = x * x;
            double u = k * (x + 0.044715 * x * x2);
            double th = std::tanh(u);
            double sech2 = 1.0 - th * th;
            double du = k * (1.0 + 3.0 * 0.044715 * x2);
            return 0.5 * (1.0 + th) + 0.5 * x * sech2 * du;
        }
        case GT_UNARY_SIGMOID_DERIV: {
            double s = 1.0 / (1.0 + std::exp(-x));
            return s * (1.0 - s);
        }
        case GT_UNARY_TANH_DERIV: {
            double t = std::tanh(x);
            return 1.0 - t * t;
        }
        case GT_UNARY_ELU_DERIV: {
            return (x >= 0.0) ? 1.0 : (p1 * std::exp(x));
        }
        case GT_UNARY_SOFTPLUS_DERIV: {
            double bx = p1 * x;
            if (bx > p2) return 1.0;
            return 1.0 / (1.0 + std::exp(-bx));
        }
        case GT_UNARY_MISH_DERIV: {
            double ex = std::exp(x);
            double sp = (x > 20.0) ? x : std::log1p(ex);
            double th = std::tanh(sp);
            double sig = 1.0 / (1.0 + std::exp(-x));
            return th + x * sig * (1.0 - th * th);
        }
        default: return x;
    }
}

extern "C" gt_tensor * gt_unary_op(const gt_tensor * a, int op_id, double p1, double p2) {
    if (!a) return nullptr;
    int out_dtype = gt_unary_out_dtype(a->dtype, op_id);
    gt_tensor * out = gt_tensor_empty(a->ndim, a->shape, out_dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (a->numel == 0 || a->device_type == GT_DEV_META) return out;

    if (gt_try_ggml_unary(a, out, op_id, p1, p2)) {
        return out;
    }

    gt_host_view hv_a, hv_o;
    const void * ap = hv_a.read(a);
    void * op = hv_o.write_empty(out);
    bool contig = gt_check_contiguous(a);

    if (gt_is_complex(a->dtype)) {
        #pragma omp parallel for if (a->numel > 4096)
        for (int64_t i = 0; i < a->numel; ++i) {
            int64_t aidx = contig ? (a->offset + i) : gt_linear_to_elem_offset(a, i);
            std::complex<double> z = gt_read_c128_at(ap, a->dtype, aidx);
            if (op_id == GT_UNARY_REAL) gt_write_f64_at(op, out->dtype, i, z.real());
            else if (op_id == GT_UNARY_IMAG) gt_write_f64_at(op, out->dtype, i, z.imag());
            else if (op_id == GT_UNARY_ABS) gt_write_f64_at(op, out->dtype, i, std::abs(z));
            else if (op_id == GT_UNARY_ANGLE) gt_write_f64_at(op, out->dtype, i, std::arg(z));
            else if (op_id == GT_UNARY_CONJ) gt_write_c128_at(op, out->dtype, i, std::conj(z));
            else if (op_id == GT_UNARY_NEG) gt_write_c128_at(op, out->dtype, i, -z);
            else if (op_id == GT_UNARY_EXP) gt_write_c128_at(op, out->dtype, i, std::exp(z));
            else if (op_id == GT_UNARY_LOG) gt_write_c128_at(op, out->dtype, i, std::log(z));
            else if (op_id == GT_UNARY_SQRT) gt_write_c128_at(op, out->dtype, i, std::sqrt(z));
            else if (op_id == GT_UNARY_SIN) gt_write_c128_at(op, out->dtype, i, std::sin(z));
            else if (op_id == GT_UNARY_COS) gt_write_c128_at(op, out->dtype, i, std::cos(z));
            else gt_write_c128_at(op, out->dtype, i, z);
        }
        hv_o.commit(out);
        return out;
    }

    if (op_id == GT_UNARY_BITWISE_NOT && gt_is_integer(a->dtype)) {
        #pragma omp parallel for if (a->numel > 4096)
        for (int64_t i = 0; i < a->numel; ++i) {
            int64_t aidx = contig ? (a->offset + i) : gt_linear_to_elem_offset(a, i);
            gt_write_i64_at(op, out->dtype, i, ~gt_read_i64_at(ap, a->dtype, aidx));
        }
        hv_o.commit(out);
        return out;
    }

    if (op_id == GT_UNARY_BITWISE_NOT && a->dtype == GT_BOOL) {
        #pragma omp parallel for if (a->numel > 4096)
        for (int64_t i = 0; i < a->numel; ++i) {
            int64_t aidx = contig ? (a->offset + i) : gt_linear_to_elem_offset(a, i);
            gt_write_i64_at(op, out->dtype, i, gt_read_i64_at(ap, a->dtype, aidx) ? 0 : 1);
        }
        hv_o.commit(out);
        return out;
    }

    bool int_exact = gt_is_integer(out->dtype) && gt_is_integer(a->dtype) &&
                     (op_id == GT_UNARY_NEG || op_id == GT_UNARY_ABS || op_id == GT_UNARY_SIGN ||
                      op_id == GT_UNARY_SQUARE || op_id == GT_UNARY_RELU || op_id == GT_UNARY_RELU6);

    #pragma omp parallel for if (a->numel > 4096)
    for (int64_t i = 0; i < a->numel; ++i) {
        int64_t aidx = contig ? (a->offset + i) : gt_linear_to_elem_offset(a, i);
        if (int_exact) {
            int64_t v = gt_read_i64_at(ap, a->dtype, aidx);
            int64_t r = v;
            if (op_id == GT_UNARY_NEG) r = -v;
            else if (op_id == GT_UNARY_ABS) r = v < 0 ? -v : v;
            else if (op_id == GT_UNARY_SIGN) r = (v > 0) ? 1 : (v < 0 ? -1 : 0);
            else if (op_id == GT_UNARY_SQUARE) r = v * v;
            else if (op_id == GT_UNARY_RELU) r = v > 0 ? v : 0;
            else if (op_id == GT_UNARY_RELU6) r = std::min<int64_t>(std::max<int64_t>(v, 0), 6);
            gt_write_i64_at(op, out->dtype, i, r);
        } else {
            double v = gt_read_f64_at(ap, a->dtype, aidx);
            double r = gt_apply_unary_scalar(v, op_id, p1, p2);
            gt_write_f64_at(op, out->dtype, i, r);
        }
    }

    hv_o.commit(out);
    return out;
}

extern "C" int gt_unary_inplace(gt_tensor * a, int op_id, double p1, double p2) {
    if (!a) return -1;
    gt_tensor * tmp = gt_unary_op(a, op_id, p1, p2);
    if (!tmp) return -1;
    int rc = gt_tensor_copy_inplace(a, tmp);
    gt_tensor_free(tmp);
    return rc;
}

// ---------- Binary Ops with PyTorch Broadcasting + GGML Dispatch ----------

static bool gt_broadcast_shapes(const gt_tensor * a, const gt_tensor * b, int & out_ndim, int64_t * out_shape) {
    out_ndim = std::max(a->ndim, b->ndim);
    for (int i = 0; i < out_ndim; ++i) {
        int ai = a->ndim - 1 - i;
        int bi = b->ndim - 1 - i;
        int64_t da = (ai >= 0) ? a->shape[ai] : 1;
        int64_t db = (bi >= 0) ? b->shape[bi] : 1;
        if (da != db && da != 1 && db != 1) {
            gt_set_error("The size of tensor a must match the size of tensor b at non-singleton dimension");
            return false;
        }
        out_shape[out_ndim - 1 - i] = (da == 0 || db == 0) ? 0 : std::max(da, db);
    }
    return true;
}

static bool gt_is_cmp_or_logic(int op_id) {
    return op_id >= GT_BIN_EQ && op_id <= GT_BIN_LOGICAL_XOR;
}

static int gt_binary_out_dtype(int da, int db, int op_id) {
    if (gt_is_cmp_or_logic(op_id)) return GT_BOOL;
    int p = gt_promote_dtype(da, db);
    if ((op_id == GT_BIN_DIV) && (gt_is_integer(p) || p == GT_BOOL)) {
        return GT_F32;
    }
    return p;
}

static bool gt_try_ggml_binary(const gt_tensor * a, const gt_tensor * b, gt_tensor * out, int op_id, double alpha) {
    if (a->dtype != GT_F32 || b->dtype != GT_F32 || out->dtype != GT_F32) return false;
    if (a->device_type == GT_DEV_META || out->numel == 0) return false;
    if (a->device_type != b->device_type || a->device_index != b->device_index) return false;
    if (!gt_check_contiguous(a) || !gt_check_contiguous(b) || !gt_check_contiguous(out)) return false;

    if (op_id != GT_BIN_ADD && op_id != GT_BIN_SUB && op_id != GT_BIN_MUL && op_id != GT_BIN_DIV) {
        return false;
    }
    if ((op_id == GT_BIN_ADD || op_id == GT_BIN_SUB) && std::abs(alpha - 1.0) > 1e-7) {
        return false;
    }
    if (out->ndim > 4) return false;

    // Fast exact same-shape path (arbitrary numel treated as 2D)
    bool same_shape = (a->numel == b->numel && a->numel == out->numel && a->ndim == b->ndim);
    if (same_shape) {
        for (int i = 0; i < a->ndim; ++i) {
            if (a->shape[i] != b->shape[i]) { same_shape = false; break; }
        }
    }

    gt_backend_entry * be = gt_get_backend(out->device_type, out->device_index);
    if (!be) return false;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor * ga = nullptr;
    struct ggml_tensor * gb = nullptr;
    if (same_shape) {
        int64_t n = out->numel;
        int64_t ne0 = n, ne1 = 1;
        if (n >= 4096) {
            for (int64_t d = 256; d <= 1024; d *= 2) {
                if (n % d == 0) { ne0 = d; ne1 = n / d; break; }
            }
        }
        ga = gt_wrap_ggml_4d(ctx, a, ne0, ne1, 1, 1);
        gb = gt_wrap_ggml_4d(ctx, b, ne0, ne1, 1, 1);
    } else {
        // Check if b broadcasts into a (ggml requires b shape divides a shape on each dim!)
        int64_t ane[4] = {1, 1, 1, 1};
        int64_t bne[4] = {1, 1, 1, 1};
        for (int i = 0; i < out->ndim; ++i) {
            int ai = a->ndim - 1 - i;
            int bi = b->ndim - 1 - i;
            ane[i] = (ai >= 0) ? a->shape[ai] : 1;
            bne[i] = (bi >= 0) ? b->shape[bi] : 1;
        }
        bool a_is_full = true;
        for (int i = 0; i < 4; ++i) {
            if (ane[i] != ((i < out->ndim) ? out->shape[out->ndim - 1 - i] : 1)) {
                a_is_full = false;
                break;
            }
        }
        if (!a_is_full) {
            ggml_free(ctx);
            return false;
        }
        ga = gt_wrap_ggml_4d(ctx, a, ane[0], ane[1], ane[2], ane[3]);
        gb = gt_wrap_ggml_4d(ctx, b, bne[0], bne[1], bne[2], bne[3]);
        if (!ggml_can_repeat(gb, ga)) {
            ggml_free(ctx);
            return false;
        }
    }

    struct ggml_tensor * gout = nullptr;
    switch (op_id) {
        case GT_BIN_ADD: gout = ggml_add(ctx, ga, gb); break;
        case GT_BIN_SUB: gout = ggml_sub(ctx, ga, gb); break;
        case GT_BIN_MUL: gout = ggml_mul(ctx, ga, gb); break;
        case GT_BIN_DIV: gout = ggml_div(ctx, ga, gb); break;
        default: break;
    }
    if (!gout) {
        ggml_free(ctx);
        return false;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, gout);
    bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
    ggml_free(ctx);
    return ok;
}

static inline double gt_apply_binary_f64(double x, double y, int op_id, double alpha) {
    switch (op_id) {
        case GT_BIN_ADD: return x + alpha * y;
        case GT_BIN_SUB: return x - alpha * y;
        case GT_BIN_MUL: return x * y;
        case GT_BIN_DIV: return x / y;
        case GT_BIN_FLOOR_DIV: return std::floor(x / y);
        case GT_BIN_FMOD: return std::fmod(x, y);
        case GT_BIN_REMAINDER: {
            double r = std::fmod(x, y);
            if (r != 0.0 && ((r < 0.0) != (y < 0.0))) r += y;
            return r;
        }
        case GT_BIN_POW: return std::pow(x, y);
        case GT_BIN_ATAN2: return std::atan2(x, y);
        case GT_BIN_MAXIMUM: return (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN() : std::max(x, y);
        case GT_BIN_MINIMUM: return (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN() : std::min(x, y);
        case GT_BIN_EQ: return (x == y) ? 1.0 : 0.0;
        case GT_BIN_NE: return (x != y) ? 1.0 : 0.0;
        case GT_BIN_LT: return (x < y) ? 1.0 : 0.0;
        case GT_BIN_LE: return (x <= y) ? 1.0 : 0.0;
        case GT_BIN_GT: return (x > y) ? 1.0 : 0.0;
        case GT_BIN_GE: return (x >= y) ? 1.0 : 0.0;
        case GT_BIN_LOGICAL_AND: return ((x != 0.0) && (y != 0.0)) ? 1.0 : 0.0;
        case GT_BIN_LOGICAL_OR:  return ((x != 0.0) || (y != 0.0)) ? 1.0 : 0.0;
        case GT_BIN_LOGICAL_XOR: return ((x != 0.0) != (y != 0.0)) ? 1.0 : 0.0;
        default: return 0.0;
    }
}

extern "C" gt_tensor * gt_binary_op(const gt_tensor * a, const gt_tensor * b, int op_id, double alpha) {
    if (!a || !b) return nullptr;
    if (a->device_type != b->device_type || a->device_index != b->device_index) {
        gt_set_error("Expected all tensors to be on the same device");
        return nullptr;
    }
    int out_ndim = 0;
    int64_t out_shape[GT_MAX_DIMS] = {};
    if (!gt_broadcast_shapes(a, b, out_ndim, out_shape)) return nullptr;

    int out_dtype = gt_binary_out_dtype(a->dtype, b->dtype, op_id);
    gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, out_dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    if (gt_try_ggml_binary(a, b, out, op_id, alpha)) {
        return out;
    }

    gt_host_view hv_a, hv_b, hv_o;
    const void * ap = hv_a.read(a);
    const void * bp = hv_b.read(b);
    void * op = hv_o.write_empty(out);

    int64_t a_bstrides[GT_MAX_DIMS] = {};
    int64_t b_bstrides[GT_MAX_DIMS] = {};
    for (int i = 0; i < out_ndim; ++i) {
        int ai = a->ndim - out_ndim + i;
        int bi = b->ndim - out_ndim + i;
        a_bstrides[i] = (ai < 0 || a->shape[ai] == 1) ? 0 : a->strides[ai];
        b_bstrides[i] = (bi < 0 || b->shape[bi] == 1) ? 0 : b->strides[bi];
    }

    bool use_complex = gt_is_complex(a->dtype) || gt_is_complex(b->dtype);
    bool use_i64 = !use_complex && (gt_is_integer(out->dtype) || out->dtype == GT_BOOL) &&
                   (gt_is_integer(a->dtype) || a->dtype == GT_BOOL) &&
                   (gt_is_integer(b->dtype) || b->dtype == GT_BOOL) &&
                   (op_id == GT_BIN_ADD || op_id == GT_BIN_SUB || op_id == GT_BIN_MUL ||
                    op_id == GT_BIN_FLOOR_DIV || op_id == GT_BIN_FMOD || op_id == GT_BIN_REMAINDER ||
                    op_id == GT_BIN_POW || op_id == GT_BIN_MAXIMUM || op_id == GT_BIN_MINIMUM ||
                    op_id >= GT_BIN_EQ);

    #pragma omp parallel for if (out->numel > 4096)
    for (int64_t idx = 0; idx < out->numel; ++idx) {
        int64_t a_off = a->offset;
        int64_t b_off = b->offset;
        int64_t rem = idx;
        for (int d = out_ndim - 1; d >= 0; --d) {
            int64_t c = rem % out_shape[d];
            rem /= out_shape[d];
            a_off += c * a_bstrides[d];
            b_off += c * b_bstrides[d];
        }
        if (use_complex) {
            std::complex<double> za = gt_read_c128_at(ap, a->dtype, a_off);
            std::complex<double> zb = gt_read_c128_at(bp, b->dtype, b_off);
            if (op_id == GT_BIN_ADD) gt_write_c128_at(op, out->dtype, idx, za + alpha * zb);
            else if (op_id == GT_BIN_SUB) gt_write_c128_at(op, out->dtype, idx, za - alpha * zb);
            else if (op_id == GT_BIN_MUL) gt_write_c128_at(op, out->dtype, idx, za * zb);
            else if (op_id == GT_BIN_DIV) gt_write_c128_at(op, out->dtype, idx, za / zb);
            else if (op_id == GT_BIN_POW) gt_write_c128_at(op, out->dtype, idx, std::pow(za, zb));
            else if (op_id == GT_BIN_EQ) gt_write_i64_at(op, out->dtype, idx, za == zb ? 1 : 0);
            else if (op_id == GT_BIN_NE) gt_write_i64_at(op, out->dtype, idx, za != zb ? 1 : 0);
        } else if (use_i64) {
            int64_t ia = gt_read_i64_at(ap, a->dtype, a_off);
            int64_t ib = gt_read_i64_at(bp, b->dtype, b_off);
            int64_t r = 0;
            switch (op_id) {
                case GT_BIN_ADD: r = ia + (int64_t)alpha * ib; break;
                case GT_BIN_SUB: r = ia - (int64_t)alpha * ib; break;
                case GT_BIN_MUL: r = ia * ib; break;
                case GT_BIN_FLOOR_DIV: r = (ib != 0) ? (ia / ib - ((ia % ib != 0) && ((ia < 0) != (ib < 0)) ? 1 : 0)) : 0; break;
                case GT_BIN_FMOD: r = (ib != 0) ? (ia % ib) : 0; break;
                case GT_BIN_REMAINDER: {
                    if (ib == 0) r = 0;
                    else { r = ia % ib; if (r != 0 && ((r < 0) != (ib < 0))) r += ib; }
                } break;
                case GT_BIN_POW: r = (int64_t)std::pow((double)ia, (double)ib); break;
                case GT_BIN_MAXIMUM: r = std::max(ia, ib); break;
                case GT_BIN_MINIMUM: r = std::min(ia, ib); break;
                case GT_BIN_EQ: r = (ia == ib) ? 1 : 0; break;
                case GT_BIN_NE: r = (ia != ib) ? 1 : 0; break;
                case GT_BIN_LT: r = (ia < ib) ? 1 : 0; break;
                case GT_BIN_LE: r = (ia <= ib) ? 1 : 0; break;
                case GT_BIN_GT: r = (ia > ib) ? 1 : 0; break;
                case GT_BIN_GE: r = (ia >= ib) ? 1 : 0; break;
                case GT_BIN_LOGICAL_AND: r = ((ia != 0) && (ib != 0)) ? 1 : 0; break;
                case GT_BIN_LOGICAL_OR:  r = ((ia != 0) || (ib != 0)) ? 1 : 0; break;
                case GT_BIN_LOGICAL_XOR: r = ((ia != 0) != (ib != 0)) ? 1 : 0; break;
                case GT_BIN_BITWISE_AND: r = ia & ib; break;
                case GT_BIN_BITWISE_OR:  r = ia | ib; break;
                case GT_BIN_BITWISE_XOR: r = ia ^ ib; break;
                default: break;
            }
            gt_write_i64_at(op, out->dtype, idx, r);
        } else {
            double xa = gt_read_f64_at(ap, a->dtype, a_off);
            double yb = gt_read_f64_at(bp, b->dtype, b_off);
            gt_write_f64_at(op, out->dtype, idx, gt_apply_binary_f64(xa, yb, op_id, alpha));
        }
    }

    hv_o.commit(out);
    return out;
}

extern "C" gt_tensor * gt_binary_scalar_op(const gt_tensor * a, double scalar, int scalar_first, int op_id) {
    if (!a) return nullptr;
    int s_dtype = gt_is_floating(a->dtype) ? a->dtype :
                  (std::floor(scalar) == scalar ? a->dtype : GT_F32);
    if (a->dtype == GT_F16 || a->dtype == GT_BF16) s_dtype = a->dtype;
    int64_t s_shape[1] = {1};
    gt_tensor * st = gt_tensor_empty(0, s_shape, s_dtype, a->device_type, a->device_index);
    if (!st) return nullptr;
    gt_fill_scalar_f64(st, scalar);
    gt_tensor * res = scalar_first ? gt_binary_op(st, a, op_id, 1.0) : gt_binary_op(a, st, op_id, 1.0);
    gt_tensor_free(st);
    return res;
}

extern "C" int gt_binary_inplace(gt_tensor * a, const gt_tensor * b, int op_id, double alpha) {
    if (!a || !b) return -1;
    gt_tensor * tmp = gt_binary_op(a, b, op_id, alpha);
    if (!tmp) return -1;
    int rc = gt_tensor_copy_inplace(a, tmp);
    gt_tensor_free(tmp);
    return rc;
}

extern "C" int gt_binary_scalar_inplace(gt_tensor * a, double scalar, int op_id, double alpha) {
    if (!a) return -1;
    int64_t s_shape[1] = {1};
    gt_tensor * st = gt_tensor_empty(0, s_shape, a->dtype, a->device_type, a->device_index);
    if (!st) return -1;
    gt_fill_scalar_f64(st, scalar);
    int rc = gt_binary_inplace(a, st, op_id, alpha);
    gt_tensor_free(st);
    return rc;
}

extern "C" gt_tensor * gt_where(const gt_tensor * cond, const gt_tensor * x, const gt_tensor * y) {
    if (!cond || !x || !y) return nullptr;
    int nd1 = 0;
    int64_t sh1[GT_MAX_DIMS] = {};
    if (!gt_broadcast_shapes(x, y, nd1, sh1)) return nullptr;

    int out_ndim = std::max(cond->ndim, nd1);
    int64_t out_shape[GT_MAX_DIMS] = {};
    for (int i = 0; i < out_ndim; ++i) {
        int ci = cond->ndim - 1 - i;
        int xy_i = nd1 - 1 - i;
        int64_t dc = (ci >= 0) ? cond->shape[ci] : 1;
        int64_t dxy = (xy_i >= 0) ? sh1[xy_i] : 1;
        if (dc != dxy && dc != 1 && dxy != 1) {
            gt_set_error("where: incompatible broadcast shape");
            return nullptr;
        }
        out_shape[out_ndim - 1 - i] = (dc == 0 || dxy == 0) ? 0 : std::max(dc, dxy);
    }

    int out_dtype = gt_promote_dtype(x->dtype, y->dtype);
    gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, out_dtype, x->device_type, x->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_host_view hvc, hvx, hvy, hvo;
    const void * cp = hvc.read(cond);
    const void * xp = hvx.read(x);
    const void * yp = hvy.read(y);
    void * op = hvo.write_empty(out);

    int64_t cb[GT_MAX_DIMS] = {}, xb[GT_MAX_DIMS] = {}, yb[GT_MAX_DIMS] = {};
    for (int i = 0; i < out_ndim; ++i) {
        int ci = cond->ndim - out_ndim + i;
        int xi = x->ndim - out_ndim + i;
        int yi = y->ndim - out_ndim + i;
        cb[i] = (ci < 0 || cond->shape[ci] == 1) ? 0 : cond->strides[ci];
        xb[i] = (xi < 0 || x->shape[xi] == 1) ? 0 : x->strides[xi];
        yb[i] = (yi < 0 || y->shape[yi] == 1) ? 0 : y->strides[yi];
    }

    #pragma omp parallel for if (out->numel > 4096)
    for (int64_t idx = 0; idx < out->numel; ++idx) {
        int64_t co = cond->offset, xo = x->offset, yo = y->offset;
        int64_t rem = idx;
        for (int d = out_ndim - 1; d >= 0; --d) {
            int64_t c = rem % out_shape[d];
            rem /= out_shape[d];
            co += c * cb[d];
            xo += c * xb[d];
            yo += c * yb[d];
        }
        bool pick_x = (gt_read_f64_at(cp, cond->dtype, co) != 0.0);
        if (gt_is_complex(out->dtype)) {
            gt_write_c128_at(op, out->dtype, idx, pick_x ? gt_read_c128_at(xp, x->dtype, xo) : gt_read_c128_at(yp, y->dtype, yo));
        } else if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL) {
            gt_write_i64_at(op, out->dtype, idx, pick_x ? gt_read_i64_at(xp, x->dtype, xo) : gt_read_i64_at(yp, y->dtype, yo));
        } else {
            gt_write_f64_at(op, out->dtype, idx, pick_x ? gt_read_f64_at(xp, x->dtype, xo) : gt_read_f64_at(yp, y->dtype, yo));
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_lerp(const gt_tensor * a, const gt_tensor * b, const gt_tensor * weight) {
    gt_tensor * diff = gt_binary_op(b, a, GT_BIN_SUB, 1.0);
    if (!diff) return nullptr;
    gt_tensor * scaled = gt_binary_op(diff, weight, GT_BIN_MUL, 1.0);
    gt_tensor_free(diff);
    if (!scaled) return nullptr;
    gt_tensor * res = gt_binary_op(a, scaled, GT_BIN_ADD, 1.0);
    gt_tensor_free(scaled);
    return res;
}

extern "C" gt_tensor * gt_clamp_tensor(const gt_tensor * x, const gt_tensor * min_t, const gt_tensor * max_t, int has_min, int has_max, double min_s, double max_s) {
    if (!x) return nullptr;
    if (!min_t && !max_t) {
        double lo = has_min ? min_s : std::numeric_limits<double>::quiet_NaN();
        double hi = has_max ? max_s : std::numeric_limits<double>::quiet_NaN();
        return gt_unary_op(x, GT_UNARY_CLAMP, lo, hi);
    }
    gt_tensor * cur = gt_tensor_clone(x);
    if (!cur) return nullptr;
    if (min_t) {
        gt_tensor * nxt = gt_binary_op(cur, min_t, GT_BIN_MAXIMUM, 1.0);
        gt_tensor_free(cur);
        cur = nxt;
        if (!cur) return nullptr;
    } else if (has_min) {
        gt_tensor * nxt = gt_unary_op(cur, GT_UNARY_CLAMP, min_s, std::numeric_limits<double>::quiet_NaN());
        gt_tensor_free(cur);
        cur = nxt;
        if (!cur) return nullptr;
    }
    if (max_t) {
        gt_tensor * nxt = gt_binary_op(cur, max_t, GT_BIN_MINIMUM, 1.0);
        gt_tensor_free(cur);
        cur = nxt;
        if (!cur) return nullptr;
    } else if (has_max) {
        gt_tensor * nxt = gt_unary_op(cur, GT_UNARY_CLAMP, std::numeric_limits<double>::quiet_NaN(), max_s);
        gt_tensor_free(cur);
        cur = nxt;
        if (!cur) return nullptr;
    }
    return cur;
}

// ---------- Reductions ----------

extern "C" gt_tensor * gt_reduce(const gt_tensor * a, int op_id, int n_axes, const int * axes, int keepdim, int out_dtype) {
    if (!a) return nullptr;
    int nd = a->ndim;
    bool reduce_axis[GT_MAX_DIMS] = {};
    if (n_axes <= 0) {
        for (int i = 0; i < nd; ++i) reduce_axis[i] = true;
    } else {
        for (int i = 0; i < n_axes; ++i) {
            int ax = axes[i];
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd) {
                gt_set_error("Dimension out of range in reduction");
                return nullptr;
            }
            reduce_axis[ax] = true;
        }
    }

    int out_ndim = 0;
    int64_t out_shape[GT_MAX_DIMS] = {};
    int ax_map[GT_MAX_DIMS] = {};
    for (int i = 0; i < nd; ++i) {
        if (reduce_axis[i]) {
            if (keepdim) {
                ax_map[out_ndim] = i;
                out_shape[out_ndim++] = 1;
            }
        } else {
            ax_map[out_ndim] = i;
            out_shape[out_ndim++] = a->shape[i];
        }
    }

    int dst_dtype = out_dtype;
    if (dst_dtype < 0) {
        if (op_id == GT_RED_ARGMAX || op_id == GT_RED_ARGMIN) dst_dtype = GT_I64;
        else if (op_id == GT_RED_ALL || op_id == GT_RED_ANY) dst_dtype = GT_BOOL;
        else if (op_id == GT_RED_SUM || op_id == GT_RED_PROD) {
            dst_dtype = (gt_is_integer(a->dtype) || a->dtype == GT_BOOL) ? GT_I64 : a->dtype;
        } else if (op_id == GT_RED_MEAN || op_id == GT_RED_VAR_POP || op_id == GT_RED_VAR_SAMPLE ||
                   op_id == GT_RED_STD_POP || op_id == GT_RED_STD_SAMPLE || op_id == GT_RED_LOGSUMEXP ||
                   op_id == GT_RED_NORM_L1 || op_id == GT_RED_NORM_L2) {
            dst_dtype = gt_is_floating(a->dtype) ? a->dtype : GT_F32;
        } else {
            dst_dtype = a->dtype;
        }
    }

    // Fast ggml full-tensor sum/mean path for F32
    if (n_axes <= 0 && (op_id == GT_RED_SUM || op_id == GT_RED_MEAN) &&
        a->dtype == GT_F32 && dst_dtype == GT_F32 && a->numel > 0 &&
        gt_check_contiguous(a) && a->device_type != GT_DEV_META && !keepdim && out_ndim == 0) {
        gt_backend_entry * be = gt_get_backend(a->device_type, a->device_index);
        if (be) {
            gt_tensor * out = gt_tensor_empty(0, out_shape, dst_dtype, a->device_type, a->device_index);
            if (out) {
                struct ggml_init_params params = {
                    ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
                };
                struct ggml_context * ctx = ggml_init(params);
                if (ctx) {
                    struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, a->numel, 1, 1, 1);
                    struct ggml_tensor * gout = (op_id == GT_RED_SUM) ? ggml_sum(ctx, ga) : ggml_mean(ctx, ga);
                    struct ggml_cgraph * gf = ggml_new_graph(ctx);
                    ggml_build_forward_expand(gf, gout);
                    bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                    ggml_free(ctx);
                    if (ok) return out;
                }
                gt_tensor_free(out);
            }
        }
    }

    // Fast ggml last-dim sum/argmax path for F32 2D
    if (nd == 2 && n_axes == 1 && !reduce_axis[0] && reduce_axis[1] &&
        (op_id == GT_RED_SUM || op_id == GT_RED_ARGMAX) &&
        a->dtype == GT_F32 && a->numel > 0 && gt_check_contiguous(a) && a->device_type != GT_DEV_META) {
        if ((op_id == GT_RED_SUM && dst_dtype == GT_F32) || (op_id == GT_RED_ARGMAX && dst_dtype == GT_I32)) {
            gt_backend_entry * be = gt_get_backend(a->device_type, a->device_index);
            if (be) {
                gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, dst_dtype, a->device_type, a->device_index);
                if (out) {
                    struct ggml_init_params params = {
                        ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true
                    };
                    struct ggml_context * ctx = ggml_init(params);
                    if (ctx) {
                        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, a->shape[1], a->shape[0], 1, 1);
                        struct ggml_tensor * gout = (op_id == GT_RED_SUM) ? ggml_sum_rows(ctx, ga) : ggml_argmax(ctx, ga);
                        struct ggml_cgraph * gf = ggml_new_graph(ctx);
                        ggml_build_forward_expand(gf, gout);
                        bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
                        ggml_free(ctx);
                        if (ok) return out;
                    }
                    gt_tensor_free(out);
                }
            }
        }
    }

    gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, dst_dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_host_view hv_a, hv_o;
    const void * ap = hv_a.read(a);
    void * op = hv_o.write_empty(out);

    std::vector<int> red_axes;
    int64_t red_count = 1;
    for (int i = 0; i < nd; ++i) {
        if (reduce_axis[i]) {
            red_axes.push_back(i);
            red_count *= a->shape[i];
        }
    }

    std::vector<int> keep_axes;
    for (int i = 0; i < nd; ++i) {
        if (!reduce_axis[i]) keep_axes.push_back(i);
    }

    #pragma omp parallel for if (out->numel > 256 && red_count > 16)
    for (int64_t oidx = 0; oidx < out->numel; ++oidx) {
        int64_t base_off = a->offset;
        int64_t rem = oidx;
        for (int k = (int)keep_axes.size() - 1; k >= 0; --k) {
            int ax = keep_axes[k];
            int64_t c = rem % a->shape[ax];
            rem /= a->shape[ax];
            base_off += c * a->strides[ax];
        }

        if (red_count == 0) {
            double v = 0.0;
            if (op_id == GT_RED_PROD || op_id == GT_RED_ALL) v = 1.0;
            else if (op_id == GT_RED_LOGSUMEXP) v = -std::numeric_limits<double>::infinity();
            gt_write_f64_at(op, dst_dtype, oidx, v);
            continue;
        }

        auto get_in_off = [&](int64_t ridx) {
            int64_t off = base_off;
            int64_t rr = ridx;
            for (int k = (int)red_axes.size() - 1; k >= 0; --k) {
                int ax = red_axes[k];
                int64_t c = rr % a->shape[ax];
                rr /= a->shape[ax];
                off += c * a->strides[ax];
            }
            return off;
        };

        if (op_id == GT_RED_ARGMAX || op_id == GT_RED_ARGMIN) {
            int64_t best_i = 0;
            double best_v = gt_read_f64_at(ap, a->dtype, get_in_off(0));
            for (int64_t r = 1; r < red_count; ++r) {
                double v = gt_read_f64_at(ap, a->dtype, get_in_off(r));
                if ((op_id == GT_RED_ARGMAX && v > best_v) || (op_id == GT_RED_ARGMIN && v < best_v)) {
                    best_v = v;
                    best_i = r;
                }
            }
            gt_write_i64_at(op, dst_dtype, oidx, best_i);
        } else if (op_id == GT_RED_ALL || op_id == GT_RED_ANY) {
            bool res = (op_id == GT_RED_ALL);
            for (int64_t r = 0; r < red_count; ++r) {
                bool b = (gt_read_f64_at(ap, a->dtype, get_in_off(r)) != 0.0);
                if (op_id == GT_RED_ALL && !b) { res = false; break; }
                if (op_id == GT_RED_ANY && b)  { res = true; break; }
            }
            gt_write_i64_at(op, dst_dtype, oidx, res ? 1 : 0);
        } else if (op_id == GT_RED_LOGSUMEXP) {
            double mx = -std::numeric_limits<double>::infinity();
            for (int64_t r = 0; r < red_count; ++r) {
                double v = gt_read_f64_at(ap, a->dtype, get_in_off(r));
                if (v > mx) mx = v;
            }
            if (std::isinf(mx)) {
                gt_write_f64_at(op, dst_dtype, oidx, mx);
            } else {
                double s = 0.0;
                for (int64_t r = 0; r < red_count; ++r) {
                    s += std::exp(gt_read_f64_at(ap, a->dtype, get_in_off(r)) - mx);
                }
                gt_write_f64_at(op, dst_dtype, oidx, mx + std::log(s));
            }
        } else if (op_id == GT_RED_VAR_POP || op_id == GT_RED_VAR_SAMPLE ||
                   op_id == GT_RED_STD_POP || op_id == GT_RED_STD_SAMPLE) {
            double mean = 0.0;
            for (int64_t r = 0; r < red_count; ++r) {
                mean += gt_read_f64_at(ap, a->dtype, get_in_off(r));
            }
            mean /= (double)red_count;
            double sq = 0.0;
            for (int64_t r = 0; r < red_count; ++r) {
                double d = gt_read_f64_at(ap, a->dtype, get_in_off(r)) - mean;
                sq += d * d;
            }
            double denom = (op_id == GT_RED_VAR_SAMPLE || op_id == GT_RED_STD_SAMPLE)
                           ? std::max<double>(1.0, (double)(red_count - 1))
                           : (double)red_count;
            double var = (red_count <= 1 && (op_id == GT_RED_VAR_SAMPLE || op_id == GT_RED_STD_SAMPLE))
                         ? (red_count == 1 ? std::numeric_limits<double>::quiet_NaN() : 0.0)
                         : (sq / denom);
            double res = (op_id == GT_RED_STD_POP || op_id == GT_RED_STD_SAMPLE) ? std::sqrt(var) : var;
            gt_write_f64_at(op, dst_dtype, oidx, res);
        } else if (gt_is_integer(dst_dtype) && gt_is_integer(a->dtype) &&
                   (op_id == GT_RED_SUM || op_id == GT_RED_PROD || op_id == GT_RED_MAX || op_id == GT_RED_MIN)) {
            int64_t acc = gt_read_i64_at(ap, a->dtype, get_in_off(0));
            for (int64_t r = 1; r < red_count; ++r) {
                int64_t v = gt_read_i64_at(ap, a->dtype, get_in_off(r));
                if (op_id == GT_RED_SUM) acc += v;
                else if (op_id == GT_RED_PROD) acc *= v;
                else if (op_id == GT_RED_MAX) { if (v > acc) acc = v; }
                else if (op_id == GT_RED_MIN) { if (v < acc) acc = v; }
            }
            gt_write_i64_at(op, dst_dtype, oidx, acc);
        } else {
            double acc = 0.0;
            if (op_id == GT_RED_PROD) acc = 1.0;
            else if (op_id == GT_RED_MAX || op_id == GT_RED_MIN) acc = gt_read_f64_at(ap, a->dtype, get_in_off(0));
            for (int64_t r = 0; r < red_count; ++r) {
                double v = gt_read_f64_at(ap, a->dtype, get_in_off(r));
                if (op_id == GT_RED_SUM || op_id == GT_RED_MEAN) acc += v;
                else if (op_id == GT_RED_PROD) acc *= v;
                else if (op_id == GT_RED_MAX) { if (v > acc || std::isnan(v)) acc = v; }
                else if (op_id == GT_RED_MIN) { if (v < acc || std::isnan(v)) acc = v; }
                else if (op_id == GT_RED_NORM_L1) acc += std::abs(v);
                else if (op_id == GT_RED_NORM_L2) acc += v * v;
            }
            if (op_id == GT_RED_MEAN) acc /= (double)red_count;
            else if (op_id == GT_RED_NORM_L2) acc = std::sqrt(acc);
            gt_write_f64_at(op, dst_dtype, oidx, acc);
        }
    }

    hv_o.commit(out);
    return out;
}

extern "C" gt_tensor * gt_norm_p(const gt_tensor * a, double p, int n_axes, const int * axes, int keepdim) {
    if (!a) return nullptr;
    if (p == 1.0) return gt_reduce(a, GT_RED_NORM_L1, n_axes, axes, keepdim, -1);
    if (p == 2.0) return gt_reduce(a, GT_RED_NORM_L2, n_axes, axes, keepdim, -1);
    if (std::isinf(p)) {
        gt_tensor * abs_a = gt_unary_op(a, GT_UNARY_ABS, 0, 0);
        if (!abs_a) return nullptr;
        gt_tensor * res = gt_reduce(abs_a, p > 0 ? GT_RED_MAX : GT_RED_MIN, n_axes, axes, keepdim, -1);
        gt_tensor_free(abs_a);
        return res;
    }
    gt_tensor * abs_a = gt_unary_op(a, GT_UNARY_ABS, 0, 0);
    if (!abs_a) return nullptr;
    gt_tensor * pow_a = gt_binary_scalar_op(abs_a, p, 0, GT_BIN_POW);
    gt_tensor_free(abs_a);
    if (!pow_a) return nullptr;
    gt_tensor * sum_a = gt_reduce(pow_a, GT_RED_SUM, n_axes, axes, keepdim, -1);
    gt_tensor_free(pow_a);
    if (!sum_a) return nullptr;
    gt_tensor * res = gt_binary_scalar_op(sum_a, 1.0 / p, 0, GT_BIN_POW);
    gt_tensor_free(sum_a);
    return res;
}

extern "C" int gt_sort(const gt_tensor * a, int dim, int descending, gt_tensor ** out_vals, gt_tensor ** out_idx) {
    if (!a || !out_vals || !out_idx) return -1;
    int nd = a->ndim;
    if (nd == 0) {
        *out_vals = gt_tensor_clone(a);
        int64_t z = 0;
        *out_idx = gt_tensor_from_data(0, nullptr, GT_I64, a->device_type, a->device_index, &z, sizeof(int64_t));
        return (*out_vals && *out_idx) ? 0 : -1;
    }
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) { gt_set_error("sort: dim out of range"); return -1; }

    gt_tensor * vals = gt_tensor_empty(nd, a->shape, a->dtype, a->device_type, a->device_index);
    gt_tensor * idxs = gt_tensor_empty(nd, a->shape, GT_I64, a->device_type, a->device_index);
    if (!vals || !idxs) { gt_tensor_free(vals); gt_tensor_free(idxs); return -1; }
    if (a->numel == 0 || a->device_type == GT_DEV_META) { *out_vals = vals; *out_idx = idxs; return 0; }

    gt_host_view hv_a, hv_v, hv_i;
    const void * ap = hv_a.read(a);
    void * vp = hv_v.write_empty(vals);
    void * ip = hv_i.write_empty(idxs);

    int64_t dim_len = a->shape[dim];
    int64_t outer = a->numel / dim_len;
    std::vector<int> other_dims;
    for (int d = 0; d < nd; ++d) if (d != dim) other_dims.push_back(d);

    #pragma omp parallel for if (outer > 128)
    for (int64_t o = 0; o < outer; ++o) {
        int64_t base_in = a->offset;
        int64_t base_out = 0;
        int64_t rem = o;
        for (int k = (int)other_dims.size() - 1; k >= 0; --k) {
            int d = other_dims[k];
            int64_t c = rem % a->shape[d];
            rem /= a->shape[d];
            base_in += c * a->strides[d];
            base_out += c * vals->strides[d];
        }
        std::vector<std::pair<double, int64_t>> pairs(dim_len);
        for (int64_t j = 0; j < dim_len; ++j) {
            pairs[j] = {gt_read_f64_at(ap, a->dtype, base_in + j * a->strides[dim]), j};
        }
        std::stable_sort(pairs.begin(), pairs.end(), [&](const auto & p1, const auto & p2) {
            return descending ? (p1.first > p2.first) : (p1.first < p2.first);
        });
        for (int64_t j = 0; j < dim_len; ++j) {
            int64_t ooff = base_out + j * vals->strides[dim];
            gt_write_f64_at(vp, vals->dtype, ooff, pairs[j].first);
            gt_write_i64_at(ip, GT_I64, ooff, pairs[j].second);
        }
    }
    hv_v.commit(vals);
    hv_i.commit(idxs);
    *out_vals = vals;
    *out_idx = idxs;
    return 0;
}

extern "C" int gt_topk(const gt_tensor * a, int64_t k, int dim, int largest, int sorted, gt_tensor ** out_vals, gt_tensor ** out_idx) {
    (void)sorted;
    if (!a || !out_vals || !out_idx) return -1;
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (nd == 0 || dim < 0 || dim >= nd || k < 0 || k > a->shape[dim]) {
        gt_set_error("topk: invalid k or dim");
        return -1;
    }
    gt_tensor * svals = nullptr, * sidx = nullptr;
    if (gt_sort(a, dim, largest, &svals, &sidx) != 0) return -1;

    int64_t out_shape[GT_MAX_DIMS];
    for (int i = 0; i < nd; ++i) out_shape[i] = a->shape[i];
    out_shape[dim] = k;

    gt_tensor * v_slice = gt_tensor_view(svals, nd, out_shape, svals->strides, svals->offset);
    gt_tensor * i_slice = gt_tensor_view(sidx, nd, out_shape, sidx->strides, sidx->offset);
    *out_vals = gt_tensor_clone(v_slice);
    *out_idx  = gt_tensor_clone(i_slice);
    gt_tensor_free(v_slice);
    gt_tensor_free(i_slice);
    gt_tensor_free(svals);
    gt_tensor_free(sidx);
    return (*out_vals && *out_idx) ? 0 : -1;
}

extern "C" gt_tensor * gt_cumsum(const gt_tensor * a, int dim) {
    if (!a) return nullptr;
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (nd == 0) return gt_tensor_clone(a);
    if (dim < 0 || dim >= nd) { gt_set_error("cumsum: dim out of range"); return nullptr; }

    int out_dtype = (gt_is_integer(a->dtype) || a->dtype == GT_BOOL) ? GT_I64 : a->dtype;
    gt_tensor * out = gt_tensor_empty(nd, a->shape, out_dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (a->numel == 0 || a->device_type == GT_DEV_META) return out;

    gt_host_view hv_a, hv_o;
    const void * ap = hv_a.read(a);
    void * op = hv_o.write_empty(out);

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
        if (out_dtype == GT_I64) {
            int64_t acc = 0;
            for (int64_t j = 0; j < dim_len; ++j) {
                acc += gt_read_i64_at(ap, a->dtype, base_in + j * a->strides[dim]);
                gt_write_i64_at(op, out_dtype, base_out + j * out->strides[dim], acc);
            }
        } else {
            double acc = 0.0;
            for (int64_t j = 0; j < dim_len; ++j) {
                acc += gt_read_f64_at(ap, a->dtype, base_in + j * a->strides[dim]);
                gt_write_f64_at(op, out_dtype, base_out + j * out->strides[dim], acc);
            }
        }
    }
    hv_o.commit(out);
    return out;
}

extern "C" gt_tensor * gt_cumprod(const gt_tensor * a, int dim) {
    if (!a) return nullptr;
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (nd == 0) return gt_tensor_clone(a);
    if (dim < 0 || dim >= nd) { gt_set_error("cumprod: dim out of range"); return nullptr; }

    int out_dtype = (gt_is_integer(a->dtype) || a->dtype == GT_BOOL) ? GT_I64 : a->dtype;
    gt_tensor * out = gt_tensor_empty(nd, a->shape, out_dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (a->numel == 0 || a->device_type == GT_DEV_META) return out;

    gt_host_view hv_a, hv_o;
    const void * ap = hv_a.read(a);
    void * op = hv_o.write_empty(out);

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
        if (out_dtype == GT_I64) {
            int64_t acc = 1;
            for (int64_t j = 0; j < dim_len; ++j) {
                acc *= gt_read_i64_at(ap, a->dtype, base_in + j * a->strides[dim]);
                gt_write_i64_at(op, out_dtype, base_out + j * out->strides[dim], acc);
            }
        } else {
            double acc = 1.0;
            for (int64_t j = 0; j < dim_len; ++j) {
                acc *= gt_read_f64_at(ap, a->dtype, base_in + j * a->strides[dim]);
                gt_write_f64_at(op, out_dtype, base_out + j * out->strides[dim], acc);
            }
        }
    }
    hv_o.commit(out);
    return out;
}
