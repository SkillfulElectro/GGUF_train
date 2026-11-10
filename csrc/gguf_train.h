#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && !defined(__MINGW32__)
#  ifdef GT_BUILD_SHARED
#    define GT_API __declspec(dllexport)
#  else
#    define GT_API __declspec(dllimport)
#  endif
#else
#  define GT_API __attribute__((visibility("default")))
#endif

#define GT_MAX_DIMS 12

enum gt_dtype {
    GT_F32   = 0,
    GT_F16   = 1,
    GT_BF16  = 2,
    GT_F64   = 3,
    GT_I8    = 4,
    GT_U8    = 5,
    GT_I16   = 6,
    GT_I32   = 7,
    GT_I64   = 8,
    GT_BOOL  = 9,
    GT_C64   = 10,
    GT_C128  = 11,
    GT_Q_BASE = 100
};

enum gt_device_type {
    GT_DEV_CPU    = 0,
    GT_DEV_CUDA   = 1,
    GT_DEV_MPS    = 2,
    GT_DEV_VULKAN = 3,
    GT_DEV_META   = 4
};

enum gt_unary_op {
    GT_UNARY_NEG = 0,
    GT_UNARY_ABS,
    GT_UNARY_SIGN,
    GT_UNARY_SQRT,
    GT_UNARY_RSQRT,
    GT_UNARY_SQUARE,
    GT_UNARY_EXP,
    GT_UNARY_EXPM1,
    GT_UNARY_EXP2,
    GT_UNARY_LOG,
    GT_UNARY_LOG2,
    GT_UNARY_LOG10,
    GT_UNARY_LOG1P,
    GT_UNARY_SIN,
    GT_UNARY_COS,
    GT_UNARY_TAN,
    GT_UNARY_ASIN,
    GT_UNARY_ACOS,
    GT_UNARY_ATAN,
    GT_UNARY_SINH,
    GT_UNARY_COSH,
    GT_UNARY_TANH,
    GT_UNARY_ASINH,
    GT_UNARY_ACOSH,
    GT_UNARY_ATANH,
    GT_UNARY_ERF,
    GT_UNARY_ERFC,
    GT_UNARY_ERFINV,
    GT_UNARY_LGAMMA,
    GT_UNARY_DIGAMMA,
    GT_UNARY_FLOOR,
    GT_UNARY_CEIL,
    GT_UNARY_ROUND,
    GT_UNARY_TRUNC,
    GT_UNARY_FRAC,
    GT_UNARY_RECIPROCAL,
    GT_UNARY_RELU,
    GT_UNARY_RELU6,
    GT_UNARY_SIGMOID,
    GT_UNARY_SILU,
    GT_UNARY_GELU,
    GT_UNARY_GELU_TANH,
    GT_UNARY_GELU_QUICK,
    GT_UNARY_ELU,
    GT_UNARY_CELU,
    GT_UNARY_SELU,
    GT_UNARY_LEAKY_RELU,
    GT_UNARY_SOFTPLUS,
    GT_UNARY_MISH,
    GT_UNARY_HARDSWISH,
    GT_UNARY_HARDSIGMOID,
    GT_UNARY_HARDTANH,
    GT_UNARY_LOG_SIGMOID,
    GT_UNARY_CLAMP,
    GT_UNARY_ISNAN,
    GT_UNARY_ISINF,
    GT_UNARY_ISFINITE,
    GT_UNARY_LOGICAL_NOT,
    GT_UNARY_BITWISE_NOT,
    GT_UNARY_NAN_TO_NUM,
    GT_UNARY_SILU_DERIV,
    GT_UNARY_GELU_DERIV,
    GT_UNARY_GELU_TANH_DERIV,
    GT_UNARY_SIGMOID_DERIV,
    GT_UNARY_TANH_DERIV,
    GT_UNARY_ELU_DERIV,
    GT_UNARY_SOFTPLUS_DERIV,
    GT_UNARY_MISH_DERIV,
    GT_UNARY_REAL,
    GT_UNARY_IMAG,
    GT_UNARY_CONJ,
    GT_UNARY_ANGLE
};

enum gt_binary_op {
    GT_BIN_ADD = 0,
    GT_BIN_SUB,
    GT_BIN_MUL,
    GT_BIN_DIV,
    GT_BIN_FLOOR_DIV,
    GT_BIN_FMOD,
    GT_BIN_REMAINDER,
    GT_BIN_POW,
    GT_BIN_ATAN2,
    GT_BIN_MAXIMUM,
    GT_BIN_MINIMUM,
    GT_BIN_EQ,
    GT_BIN_NE,
    GT_BIN_LT,
    GT_BIN_LE,
    GT_BIN_GT,
    GT_BIN_GE,
    GT_BIN_LOGICAL_AND,
    GT_BIN_LOGICAL_OR,
    GT_BIN_LOGICAL_XOR,
    GT_BIN_BITWISE_AND,
    GT_BIN_BITWISE_OR,
    GT_BIN_BITWISE_XOR
};

enum gt_reduce_op {
    GT_RED_SUM = 0,
    GT_RED_MEAN,
    GT_RED_PROD,
    GT_RED_MAX,
    GT_RED_MIN,
    GT_RED_ALL,
    GT_RED_ANY,
    GT_RED_ARGMAX,
    GT_RED_ARGMIN,
    GT_RED_LOGSUMEXP,
    GT_RED_NORM_L1,
    GT_RED_NORM_L2,
    GT_RED_VAR_POP,
    GT_RED_VAR_SAMPLE,
    GT_RED_STD_POP,
    GT_RED_STD_SAMPLE
};

typedef struct gt_tensor gt_tensor;
typedef struct gt_gguf_file gt_gguf_file;
typedef struct gt_opt_graph gt_opt_graph;

// Runtime / backend / info
GT_API void        gt_init(void);
GT_API void        gt_set_num_threads(int n_threads);
GT_API int         gt_get_num_threads(void);
GT_API void        gt_manual_seed(uint64_t seed);
GT_API int         gt_rng_get_state(char * buf, int buf_len);
GT_API int         gt_rng_set_state(const char * buf);
GT_API int         gt_device_count(int device_type);
GT_API void        gt_device_name(int device_type, int idx, char * buf, int buf_len);
GT_API void        gt_device_memory(int device_type, int idx, size_t * free_bytes, size_t * total_bytes);
GT_API void        gt_synchronize(int device_type, int idx);
GT_API void        gt_cpu_features(char * buf, int buf_len);
GT_API void        gt_backend_summary(char * buf, int buf_len);
GT_API const char * gt_last_error(void);
GT_API void        gt_clear_error(void);

// Tensor lifecycle & metadata
GT_API gt_tensor * gt_tensor_empty(int ndim, const int64_t * shape, int dtype, int device_type, int device_index);
GT_API gt_tensor * gt_tensor_from_data(int ndim, const int64_t * shape, int dtype, int device_type, int device_index, const void * data, size_t nbytes);
GT_API gt_tensor * gt_tensor_clone(const gt_tensor * t);
GT_API gt_tensor * gt_tensor_view(const gt_tensor * t, int ndim, const int64_t * shape, const int64_t * strides, int64_t offset);
GT_API gt_tensor * gt_tensor_contiguous(const gt_tensor * t);
GT_API gt_tensor * gt_tensor_to_dtype(const gt_tensor * t, int dtype);
GT_API gt_tensor * gt_tensor_to_device(const gt_tensor * t, int device_type, int device_index);
GT_API void        gt_tensor_free(gt_tensor * t);

GT_API int         gt_tensor_ndim(const gt_tensor * t);
GT_API void        gt_tensor_shape(const gt_tensor * t, int64_t * out);
GT_API void        gt_tensor_strides(const gt_tensor * t, int64_t * out);
GT_API int64_t     gt_tensor_offset(const gt_tensor * t);
GT_API int64_t     gt_tensor_numel(const gt_tensor * t);
GT_API int         gt_tensor_dtype(const gt_tensor * t);
GT_API int         gt_tensor_device_type(const gt_tensor * t);
GT_API int         gt_tensor_device_index(const gt_tensor * t);
GT_API int         gt_tensor_is_contiguous(const gt_tensor * t);
GT_API uint64_t    gt_tensor_data_ptr(const gt_tensor * t);
GT_API uint64_t    gt_tensor_storage_ptr(const gt_tensor * t);
GT_API size_t      gt_tensor_nbytes(const gt_tensor * t);
GT_API size_t      gt_tensor_elem_size(int dtype);

GT_API int         gt_tensor_copy_to_host(const gt_tensor * t, void * dst, size_t dst_bytes);
GT_API int         gt_tensor_copy_from_host(gt_tensor * t, const void * src, size_t src_bytes);
GT_API int         gt_tensor_copy_inplace(gt_tensor * dst, const gt_tensor * src);
GT_API double      gt_tensor_item_f64(const gt_tensor * t);
GT_API int64_t     gt_tensor_item_i64(const gt_tensor * t);

// Creation / fill / RNG
GT_API int         gt_fill_scalar_f64(gt_tensor * t, double val);
GT_API int         gt_fill_scalar_i64(gt_tensor * t, int64_t val);
GT_API int         gt_fill_arange(gt_tensor * t, double start, double step);
GT_API int         gt_fill_linspace(gt_tensor * t, double start, double end, int64_t steps);
GT_API int         gt_fill_eye(gt_tensor * t, int64_t n, int64_t m);
GT_API int         gt_fill_uniform(gt_tensor * t, double low, double high);
GT_API int         gt_fill_normal(gt_tensor * t, double mean, double std);
GT_API int         gt_fill_trunc_normal(gt_tensor * t, double mean, double std, double a, double b);
GT_API int         gt_fill_randint(gt_tensor * t, int64_t low, int64_t high);
GT_API int         gt_fill_randperm(gt_tensor * t, int64_t n);
GT_API int         gt_fill_bernoulli_prob(gt_tensor * t, double p);
GT_API gt_tensor * gt_bernoulli_tensor(const gt_tensor * probs);
GT_API gt_tensor * gt_multinomial(const gt_tensor * probs, int64_t num_samples, int replacement);

// Unary & binary ops
GT_API gt_tensor * gt_unary_op(const gt_tensor * a, int op_id, double p1, double p2);
GT_API int         gt_unary_inplace(gt_tensor * a, int op_id, double p1, double p2);
GT_API gt_tensor * gt_binary_op(const gt_tensor * a, const gt_tensor * b, int op_id, double alpha);
GT_API gt_tensor * gt_binary_scalar_op(const gt_tensor * a, double scalar, int scalar_first, int op_id);
GT_API int         gt_binary_inplace(gt_tensor * a, const gt_tensor * b, int op_id, double alpha);
GT_API int         gt_binary_scalar_inplace(gt_tensor * a, double scalar, int op_id, double alpha);
GT_API gt_tensor * gt_where(const gt_tensor * cond, const gt_tensor * x, const gt_tensor * y);
GT_API gt_tensor * gt_lerp(const gt_tensor * a, const gt_tensor * b, const gt_tensor * weight);
GT_API gt_tensor * gt_clamp_tensor(const gt_tensor * x, const gt_tensor * min_t, const gt_tensor * max_t, int has_min, int has_max, double min_s, double max_s);

// Matrix & linear algebra (ggml_mul_mat + solvers)
GT_API gt_tensor * gt_matmul(const gt_tensor * a, const gt_tensor * b);
GT_API gt_tensor * gt_linear(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias);
GT_API gt_tensor * gt_addmm(const gt_tensor * input, const gt_tensor * mat1, const gt_tensor * mat2, double beta, double alpha);
GT_API gt_tensor * gt_baddbmm(const gt_tensor * input, const gt_tensor * b1, const gt_tensor * b2, double beta, double alpha);
GT_API gt_tensor * gt_outer(const gt_tensor * a, const gt_tensor * b);
GT_API gt_tensor * gt_linalg_inv(const gt_tensor * a);
GT_API gt_tensor * gt_linalg_solve(const gt_tensor * a, const gt_tensor * b);
GT_API gt_tensor * gt_linalg_cholesky(const gt_tensor * a, int upper);
GT_API int         gt_linalg_qr(const gt_tensor * a, gt_tensor ** out_q, gt_tensor ** out_r, int mode);
GT_API int         gt_linalg_svd(const gt_tensor * a, int full_matrices, gt_tensor ** out_u, gt_tensor ** out_s, gt_tensor ** out_vh);
GT_API int         gt_linalg_eigh(const gt_tensor * a, int upper, gt_tensor ** out_evals, gt_tensor ** out_evecs);
GT_API gt_tensor * gt_linalg_det(const gt_tensor * a);
GT_API int         gt_linalg_slogdet(const gt_tensor * a, gt_tensor ** out_sign, gt_tensor ** out_logabsdet);

// Reductions, sort, topk, cumsum
GT_API gt_tensor * gt_reduce(const gt_tensor * a, int op_id, int n_axes, const int * axes, int keepdim, int out_dtype);
GT_API gt_tensor * gt_norm_p(const gt_tensor * a, double p, int n_axes, const int * axes, int keepdim);
GT_API int         gt_sort(const gt_tensor * a, int dim, int descending, gt_tensor ** out_vals, gt_tensor ** out_idx);
GT_API int         gt_topk(const gt_tensor * a, int64_t k, int dim, int largest, int sorted, gt_tensor ** out_vals, gt_tensor ** out_idx);
GT_API gt_tensor * gt_cumsum(const gt_tensor * a, int dim);
GT_API gt_tensor * gt_cumprod(const gt_tensor * a, int dim);

// Neural net & transformer ops (ggml-backed)
GT_API gt_tensor * gt_softmax(const gt_tensor * a, int dim, int is_log);
GT_API gt_tensor * gt_softmax_backward(const gt_tensor * grad_out, const gt_tensor * out, int dim, int is_log);
GT_API gt_tensor * gt_rms_norm(const gt_tensor * x, const gt_tensor * weight, float eps);
GT_API gt_tensor * gt_rms_norm_backward(const gt_tensor * grad_out, const gt_tensor * x, const gt_tensor * weight, float eps, gt_tensor ** out_dweight);
GT_API gt_tensor * gt_layer_norm(const gt_tensor * x, int norm_ndim, const gt_tensor * weight, const gt_tensor * bias, float eps);
GT_API int         gt_layer_norm_backward(const gt_tensor * grad_out, const gt_tensor * x, int norm_ndim, const gt_tensor * weight, float eps, gt_tensor ** dx, gt_tensor ** dw, gt_tensor ** db);
GT_API gt_tensor * gt_group_norm(const gt_tensor * x, int num_groups, const gt_tensor * weight, const gt_tensor * bias, float eps);
GT_API gt_tensor * gt_batch_norm(const gt_tensor * x, const gt_tensor * weight, const gt_tensor * bias, gt_tensor * running_mean, gt_tensor * running_var, int training, double momentum, double eps, gt_tensor ** save_mean, gt_tensor ** save_invstd);
GT_API int         gt_batch_norm_backward(const gt_tensor * grad_out, const gt_tensor * x, const gt_tensor * weight, const gt_tensor * save_mean, const gt_tensor * save_invstd, const gt_tensor * running_mean, const gt_tensor * running_var, int training, double eps, gt_tensor ** dx, gt_tensor ** dw, gt_tensor ** db);
GT_API gt_tensor * gt_embedding(const gt_tensor * weight, const gt_tensor * indices, int64_t padding_idx);
GT_API gt_tensor * gt_embedding_backward(const gt_tensor * grad_out, const gt_tensor * indices, int64_t num_embeddings, int64_t padding_idx);

GT_API gt_tensor * gt_conv2d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias, int sh, int sw, int ph, int pw, int dh, int dw, int groups);
GT_API int         gt_conv2d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight, int sh, int sw, int ph, int pw, int dh, int dw, int groups, int need_dx, int need_dw, int need_db, gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db);
GT_API gt_tensor * gt_conv1d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias, int s, int p, int d, int groups);
GT_API int         gt_conv1d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight, int s, int p, int d, int groups, int need_dx, int need_dw, int need_db, gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db);
GT_API gt_tensor * gt_conv_transpose2d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias, int sh, int sw, int ph, int pw, int oph, int opw, int dh, int dw, int groups);
GT_API int         gt_conv_transpose2d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight, int sh, int sw, int ph, int pw, int oph, int opw, int dh, int dw, int groups, int need_dx, int need_dw, int need_db, gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db);
GT_API gt_tensor * gt_conv3d(const gt_tensor * input, const gt_tensor * weight, const gt_tensor * bias, int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups);
GT_API int         gt_conv3d_backward(const gt_tensor * grad_out, const gt_tensor * input, const gt_tensor * weight, int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups, int need_dx, int need_dw, int need_db, gt_tensor ** dx, gt_tensor ** dw_out, gt_tensor ** db);

GT_API int         gt_max_pool2d(const gt_tensor * input, int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw, gt_tensor ** out, gt_tensor ** indices);
GT_API gt_tensor * gt_max_pool2d_backward(const gt_tensor * grad_out, const gt_tensor * indices, const gt_tensor * input_ref);
GT_API gt_tensor * gt_avg_pool2d(const gt_tensor * input, int kh, int kw, int sh, int sw, int ph, int pw, int count_include_pad);
GT_API gt_tensor * gt_avg_pool2d_backward(const gt_tensor * grad_out, const gt_tensor * input_ref, int kh, int kw, int sh, int sw, int ph, int pw, int count_include_pad);
GT_API gt_tensor * gt_upsample_2d(const gt_tensor * input, int64_t out_h, int64_t out_w, int mode, int align_corners);
GT_API gt_tensor * gt_upsample_2d_backward(const gt_tensor * grad_out, int64_t in_h, int64_t in_w, int mode, int align_corners);

GT_API gt_tensor * gt_cross_entropy_loss(const gt_tensor * logits, const gt_tensor * targets, const gt_tensor * weight, int64_t ignore_index, double label_smoothing, int reduction);
GT_API gt_tensor * gt_cross_entropy_loss_backward(const gt_tensor * grad_out, const gt_tensor * logits, const gt_tensor * targets, const gt_tensor * weight, int64_t ignore_index, double label_smoothing, int reduction);
GT_API gt_tensor * gt_rope(const gt_tensor * x, const gt_tensor * pos, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale);
GT_API gt_tensor * gt_rope_backward(const gt_tensor * grad_out, const gt_tensor * pos, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale);
GT_API gt_tensor * gt_flash_attention(const gt_tensor * q, const gt_tensor * k, const gt_tensor * v, const gt_tensor * mask, float scale, int is_causal);
GT_API gt_tensor * gt_swiglu(const gt_tensor * x, const gt_tensor * up);

// Shape / indexing / scatter / gather / pad / triu
GT_API gt_tensor * gt_cat(int n, const gt_tensor ** tensors, int dim);
GT_API gt_tensor * gt_index_select(const gt_tensor * a, int dim, const gt_tensor * index);
GT_API gt_tensor * gt_index_select_add(const gt_tensor * grad_out, int dim, const gt_tensor * index, int64_t dim_size);
GT_API gt_tensor * gt_gather(const gt_tensor * a, int dim, const gt_tensor * index);
GT_API gt_tensor * gt_scatter(const gt_tensor * a, int dim, const gt_tensor * index, const gt_tensor * src, int add_mode);
GT_API int         gt_scatter_inplace(gt_tensor * dst, int dim, const gt_tensor * index, const gt_tensor * src, int add_mode);
GT_API gt_tensor * gt_index_tensor(const gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims);
GT_API int         gt_index_put(gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims, const gt_tensor * values, int accumulate);
GT_API gt_tensor * gt_masked_select(const gt_tensor * a, const gt_tensor * mask);
GT_API int         gt_masked_fill_inplace(gt_tensor * a, const gt_tensor * mask, double val);
GT_API gt_tensor * gt_pad(const gt_tensor * a, int n_pairs, const int64_t * pads, int mode, double value);
GT_API gt_tensor * gt_tril_triu(const gt_tensor * a, int64_t diagonal, int upper);
GT_API gt_tensor * gt_diag(const gt_tensor * a, int64_t diagonal);
GT_API gt_tensor * gt_flip(const gt_tensor * a, int n_dims, const int * dims);
GT_API gt_tensor * gt_roll(const gt_tensor * a, int n_shifts, const int64_t * shifts, const int * dims);
GT_API gt_tensor * gt_nonzero(const gt_tensor * a);
GT_API gt_tensor * gt_bincount(const gt_tensor * a, const gt_tensor * weights, int64_t minlength);
GT_API int         gt_unique(const gt_tensor * a, int sorted, int return_inverse, int return_counts, gt_tensor ** out_vals, gt_tensor ** out_inv, gt_tensor ** out_cnt);

// Optimizers (using ggml_opt_step_adamw / ggml_opt_step_sgd where applicable)
GT_API int gt_adamw_step(gt_tensor * param, const gt_tensor * grad, gt_tensor * exp_avg, gt_tensor * exp_avg_sq, gt_tensor * max_exp_avg_sq,
                         double lr, double beta1, double beta2, double eps, double weight_decay, int64_t step, int amsgrad, int decoupled_wd);
GT_API int gt_sgd_step(gt_tensor * param, const gt_tensor * grad, gt_tensor * momentum_buf,
                       double lr, double momentum, double dampening, double weight_decay, int nesterov, int first_step);

// FFT (complex64 / complex128)
GT_API gt_tensor * gt_fft_1d(const gt_tensor * a, int64_t n, int dim, int inverse, int real_input, int norm_mode);

// GGUF & Quantization
GT_API int         gt_gguf_save(const char * path, int n_tensors, const char ** names, const gt_tensor ** tensors,
                                int n_kv, const char ** kv_keys, const int * kv_types, const char ** kv_vals);
GT_API gt_gguf_file * gt_gguf_open(const char * path);
GT_API void        gt_gguf_close(gt_gguf_file * f);
GT_API int64_t     gt_gguf_n_tensors(const gt_gguf_file * f);
GT_API const char * gt_gguf_tensor_name(const gt_gguf_file * f, int64_t idx);
GT_API gt_tensor * gt_gguf_get_tensor(const gt_gguf_file * f, const char * name, int dequantize_f32);
GT_API int64_t     gt_gguf_n_kv(const gt_gguf_file * f);
GT_API const char * gt_gguf_kv_key(const gt_gguf_file * f, int64_t idx);
GT_API int         gt_gguf_kv_type(const gt_gguf_file * f, int64_t idx);
GT_API int         gt_gguf_kv_val_str(const gt_gguf_file * f, int64_t idx, char * buf, int buf_len);

GT_API gt_tensor * gt_quantize(const gt_tensor * t, int ggml_type_id);
GT_API gt_tensor * gt_dequantize(const gt_tensor * t, int out_dtype);

// Native static ggml_opt graph builder & trainer (direct ggml_opt_fit / ggml_opt_epoch!)
GT_API gt_opt_graph * gt_opt_graph_create(int loss_type, int optimizer_type, int64_t nbatch_physical, int64_t nbatch_logical);
GT_API void           gt_opt_graph_free(gt_opt_graph * g);
GT_API int            gt_opt_graph_input(gt_opt_graph * g, int ndim, const int64_t * shape);
GT_API int            gt_opt_graph_param(gt_opt_graph * g, gt_tensor * param);
GT_API int            gt_opt_graph_linear(gt_opt_graph * g, int x_id, int w_id, int b_id);
GT_API int            gt_opt_graph_unary(gt_opt_graph * g, int x_id, int unary_op);
GT_API int            gt_opt_graph_add(gt_opt_graph * g, int a_id, int b_id);
GT_API int            gt_opt_graph_mul(gt_opt_graph * g, int a_id, int b_id);
GT_API int            gt_opt_graph_rms_norm(gt_opt_graph * g, int x_id, int w_id, float eps);
GT_API int            gt_opt_graph_softmax(gt_opt_graph * g, int x_id);
GT_API int            gt_opt_graph_fit(gt_opt_graph * g, int out_id, const gt_tensor * data, const gt_tensor * labels,
                                        int64_t epochs, float lr, float wd, float val_split, int silent, double * out_train_loss, double * out_val_loss);

#ifdef __cplusplus
}
#endif
