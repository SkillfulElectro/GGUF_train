#include "gt_internal.h"
#include "ggml-opt.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

// ---------- Cat ----------

extern "C" gt_tensor * gt_cat(int n_tensors, const gt_tensor ** tensors, int dim) {
    if (n_tensors <= 0 || !tensors) { gt_set_error("cat: empty tensor list"); return nullptr; }
    const gt_tensor * ref = tensors[0];
    if (!ref) return nullptr;
    int nd = ref->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) { gt_set_error("cat: dimension out of range"); return nullptr; }

    int out_dtype = ref->dtype;
    int64_t cat_dim = 0;
    for (int t = 0; t < n_tensors; ++t) {
        const gt_tensor * cur = tensors[t];
        if (!cur || cur->ndim != nd) { gt_set_error("cat: tensors must have same number of dimensions"); return nullptr; }
        for (int d = 0; d < nd; ++d) {
            if (d != dim && cur->shape[d] != ref->shape[d]) {
                gt_set_error("cat: sizes of tensors must match except in concat dimension");
                return nullptr;
            }
        }
        cat_dim += cur->shape[dim];
        out_dtype = gt_promote_dtype(out_dtype, cur->dtype);
    }

    int64_t out_shape[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) out_shape[d] = ref->shape[d];
    out_shape[dim] = cat_dim;

    gt_tensor * out = gt_tensor_empty(nd, out_shape, out_dtype, ref->device_type, ref->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_host_view hvo;
    void * op = hvo.write_empty(out);

    int64_t outer = 1, inner = 1;
    for (int d = 0; d < dim; ++d) outer *= out_shape[d];
    for (int d = dim + 1; d < nd; ++d) inner *= out_shape[d];

    int64_t dim_offset = 0;
    for (int t = 0; t < n_tensors; ++t) {
        const gt_tensor * cur = tensors[t];
        int64_t t_dim = cur->shape[dim];
        if (cur->numel == 0) continue;
        gt_contig_owner cc(cur);
        gt_host_view hvc;
        const void * cp = hvc.read(cc.ptr);
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t k = 0; k < t_dim; ++k) {
                int64_t src_base = (o * t_dim + k) * inner;
                int64_t dst_base = (o * cat_dim + dim_offset + k) * inner;
                for (int64_t j = 0; j < inner; ++j) {
                    if (gt_is_complex(out_dtype)) {
                        gt_write_c128_at(op, out_dtype, dst_base + j, gt_read_c128_at(cp, cc.ptr->dtype, src_base + j));
                    } else if (gt_is_integer(out_dtype) || out_dtype == GT_BOOL) {
                        gt_write_i64_at(op, out_dtype, dst_base + j, gt_read_i64_at(cp, cc.ptr->dtype, src_base + j));
                    } else {
                        gt_write_f64_at(op, out_dtype, dst_base + j, gt_read_f64_at(cp, cc.ptr->dtype, src_base + j));
                    }
                }
            }
        }
        dim_offset += t_dim;
    }
    hvo.commit(out);
    return out;
}

// ---------- Indexing, Gather, Scatter, IndexSelect, Advanced Indexing ----------
//
// Conventions shared by the functions in this section:
//  * an index tensor must have an integer dtype;
//  * a negative index counts from the end of its dimension;
//  * an index outside [-size, size) is an error. Nothing is clamped to the nearest valid
//    element and nothing is skipped, and every index is checked before the first element
//    is written, so an in-place function leaves its destination untouched when it fails.
// Messages that start with "IndexError: " are raised as IndexError by the Python layer
// (indexing, index_select, embedding); gather and scatter report a RuntimeError like torch.

static void gt_set_index_error(bool tag, const char * what, int64_t idx, int dim, int64_t size) {
    std::string msg = tag ? "IndexError: " : (std::string(what) + ": ");
    msg += "index " + std::to_string(idx) + " is out of bounds for dimension " + std::to_string(dim) +
           " with size " + std::to_string(size);
    gt_set_error(msg);
}

static bool gt_require_index_dtype(const gt_tensor * index, bool tag, const char * what) {
    if (gt_is_integer(index->dtype)) return true;
    if (tag) gt_set_error("IndexError: tensors used as indices must be integer tensors");
    else gt_set_error(std::string(what) + ": index must be an integer tensor");
    return false;
}

static std::string gt_shape_str(int ndim, const int64_t * shape) {
    std::string s = "[";
    for (int i = 0; i < ndim; ++i) {
        if (i) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

// Copies one element, keeping integers and complex numbers exact
static inline void gt_copy_elem(void * dp, int ddt, int64_t di, const void * sp, int sdt, int64_t si) {
    if (gt_is_complex(ddt)) gt_write_c128_at(dp, ddt, di, gt_read_c128_at(sp, sdt, si));
    else if (gt_is_integer(ddt) || ddt == GT_BOOL) gt_write_i64_at(dp, ddt, di, gt_read_i64_at(sp, sdt, si));
    else gt_write_f64_at(dp, ddt, di, gt_read_f64_at(sp, sdt, si));
}

// dst[di] += src[si] (accumulate) or dst[di] = src[si]
static inline void gt_put_elem(void * dp, int ddt, int64_t di, const void * sp, int sdt, int64_t si, bool accumulate) {
    if (!accumulate) {
        gt_copy_elem(dp, ddt, di, sp, sdt, si);
    } else if (gt_is_complex(ddt)) {
        gt_write_c128_at(dp, ddt, di, gt_read_c128_at(dp, ddt, di) + gt_read_c128_at(sp, sdt, si));
    } else if (gt_is_integer(ddt) || ddt == GT_BOOL) {
        gt_write_i64_at(dp, ddt, di, gt_read_i64_at(dp, ddt, di) + gt_read_i64_at(sp, sdt, si));
    } else {
        gt_write_f64_at(dp, ddt, di, gt_read_f64_at(dp, ddt, di) + gt_read_f64_at(sp, sdt, si));
    }
}

extern "C" gt_tensor * gt_gather(const gt_tensor * a, int dim, const gt_tensor * index) {
    if (!a || !index) { gt_set_error("gather: null tensor"); return nullptr; }
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd || index->ndim != nd) {
        gt_set_error("gather: invalid dim or index ndim");
        return nullptr;
    }
    if (!gt_require_index_dtype(index, false, "gather")) return nullptr;
    // Outside `dim` the index walks the input's own coordinates, so it cannot be larger
    for (int d = 0; d < nd; ++d) {
        if (d != dim && index->shape[d] > a->shape[d]) {
            gt_set_error("gather: index of shape " + gt_shape_str(nd, index->shape) + " is larger than the input of shape " +
                         gt_shape_str(nd, a->shape) + " at dimension " + std::to_string(d));
            return nullptr;
        }
    }
    gt_tensor * out = gt_tensor_empty(nd, index->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_host_view hva, hvi, hvo;
    const void * ap = hva.read(a);
    const void * ip = hvi.read(index);
    void * op = hvo.write_empty(out);
    const int64_t size = a->shape[dim];

    for (int64_t l = 0; l < out->numel; ++l) {
        int64_t rem = l;
        int64_t coords[GT_MAX_DIMS];
        for (int d = nd - 1; d >= 0; --d) {
            coords[d] = rem % out->shape[d];
            rem /= out->shape[d];
        }
        int64_t ioff = index->offset;
        for (int d = 0; d < nd; ++d) ioff += coords[d] * index->strides[d];
        const int64_t given = gt_read_i64_at(ip, index->dtype, ioff);
        int64_t idx = given < 0 ? given + size : given;
        if (idx < 0 || idx >= size) {
            gt_set_index_error(false, "gather", given, dim, size);
            gt_tensor_free(out);
            return nullptr;
        }
        coords[dim] = idx;
        int64_t aoff = a->offset;
        for (int d = 0; d < nd; ++d) aoff += coords[d] * a->strides[d];
        gt_copy_elem(op, out->dtype, l, ap, a->dtype, aoff);
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_scatter_inplace(gt_tensor * dst, int dim, const gt_tensor * index, const gt_tensor * src, int add_mode) {
    if (!dst || !index || !src) { gt_set_error("scatter: null tensor"); return -1; }
    int nd = dst->ndim;
    if (dim < 0) dim += nd;
    const bool scalar_src = (src->ndim == 0);
    if (dim < 0 || dim >= nd || index->ndim != nd || (!scalar_src && src->ndim != nd)) {
        gt_set_error("scatter: dimension mismatch");
        return -1;
    }
    if (!gt_require_index_dtype(index, false, "scatter")) return -1;
    // The index walks the coordinates of src everywhere, and those of dst outside `dim`
    for (int d = 0; d < nd; ++d) {
        if ((!scalar_src && index->shape[d] > src->shape[d]) || (d != dim && index->shape[d] > dst->shape[d])) {
            gt_set_error("scatter: index of shape " + gt_shape_str(nd, index->shape) + " does not fit" +
                         (scalar_src ? std::string("") : " src of shape " + gt_shape_str(nd, src->shape) + " and") +
                         " the destination of shape " + gt_shape_str(nd, dst->shape) + " (apart from dimension " +
                         std::to_string(dim) + ")");
            return -1;
        }
    }
    if (index->numel == 0 || dst->device_type == GT_DEV_META) return 0;

    gt_host_view hvi, hvs;
    const void * ip = hvi.read(index);
    const int64_t size = dst->shape[dim];
    const bool i_contig = gt_check_contiguous(index);

    // Check every index before anything is written
    for (int64_t l = 0; l < index->numel; ++l) {
        const int64_t ioff = i_contig ? (index->offset + l) : gt_linear_to_elem_offset(index, l);
        const int64_t given = gt_read_i64_at(ip, index->dtype, ioff);
        const int64_t idx = given < 0 ? given + size : given;
        if (idx < 0 || idx >= size) {
            gt_set_index_error(false, "scatter", given, dim, size);
            return -1;
        }
    }

    gt_host_view hvd;
    void * dp = hvd.read_write(dst);
    const void * sp = hvs.read(src);

    for (int64_t l = 0; l < index->numel; ++l) {
        int64_t rem = l;
        int64_t coords[GT_MAX_DIMS];
        for (int d = nd - 1; d >= 0; --d) {
            coords[d] = rem % index->shape[d];
            rem /= index->shape[d];
        }
        int64_t ioff = index->offset, soff = src->offset;
        for (int d = 0; d < nd; ++d) {
            ioff += coords[d] * index->strides[d];
            if (!scalar_src) soff += coords[d] * src->strides[d];
        }
        int64_t idx = gt_read_i64_at(ip, index->dtype, ioff);
        if (idx < 0) idx += size;
        coords[dim] = idx;
        int64_t doff = dst->offset;
        for (int d = 0; d < nd; ++d) doff += coords[d] * dst->strides[d];
        gt_put_elem(dp, dst->dtype, doff, sp, src->dtype, soff, add_mode != 0);
    }
    hvd.commit(dst);
    return 0;
}

extern "C" gt_tensor * gt_scatter(const gt_tensor * a, int dim, const gt_tensor * index, const gt_tensor * src, int add_mode) {
    gt_tensor * out = gt_tensor_clone(a);
    if (!out) return nullptr;
    if (gt_scatter_inplace(out, dim, index, src, add_mode) != 0) {
        gt_tensor_free(out);
        return nullptr;
    }
    return out;
}

extern "C" gt_tensor * gt_index_select(const gt_tensor * a, int dim, const gt_tensor * index) {
    if (!a || !index) { gt_set_error("index_select: null tensor"); return nullptr; }
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd || index->ndim > 1) {
        gt_set_error("index_select: invalid dim, or index is not a vector");
        return nullptr;
    }
    if (!gt_require_index_dtype(index, false, "index_select")) return nullptr;
    int64_t idx_len = (index->ndim == 0) ? 1 : index->shape[0];
    const int64_t size = a->shape[dim];

    // Resolve and check the indices first: they are needed even when the result is empty in
    // another dimension, and an empty `dim` has no valid index at all.
    std::vector<int64_t> sel((size_t)idx_len);
    if (idx_len > 0 && index->device_type != GT_DEV_META) {
        gt_host_view hvi;
        const void * ip = hvi.read(index);
        for (int64_t i = 0; i < idx_len; ++i) {
            const int64_t given = gt_read_i64_at(ip, index->dtype, gt_linear_to_elem_offset(index, i));
            const int64_t s = given < 0 ? given + size : given;
            if (s < 0 || s >= size) {
                gt_set_index_error(true, "index_select", given, dim, size);
                return nullptr;
            }
            sel[(size_t)i] = s;
        }
    }

    int64_t out_shape[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) out_shape[d] = (d == dim) ? idx_len : a->shape[d];
    gt_tensor * out = gt_tensor_empty(nd, out_shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_contig_owner ac(a);
    if (!ac.ptr) { gt_tensor_free(out); return nullptr; }
    gt_host_view hva, hvo;
    const void * ap = hva.read(ac.ptr);
    void * op = hvo.write_empty(out);

    int64_t outer = 1, inner = 1;
    for (int d = 0; d < dim; ++d) outer *= out_shape[d];
    for (int d = dim + 1; d < nd; ++d) inner *= out_shape[d];

    for (int64_t i = 0; i < idx_len; ++i) {
        for (int64_t o = 0; o < outer; ++o) {
            int64_t src_base = (o * size + sel[(size_t)i]) * inner;
            int64_t dst_base = (o * idx_len + i) * inner;
            for (int64_t j = 0; j < inner; ++j) {
                gt_copy_elem(op, out->dtype, dst_base + j, ap, ac.ptr->dtype, src_base + j);
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_index_select_add(const gt_tensor * grad_out, int dim, const gt_tensor * index, int64_t dim_size) {
    if (!grad_out || !index) return nullptr;
    int nd = grad_out->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) return nullptr;
    int64_t out_shape[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) out_shape[d] = (d == dim) ? dim_size : grad_out->shape[d];
    gt_tensor * out = gt_tensor_empty(nd, out_shape, grad_out->dtype, grad_out->device_type, grad_out->device_index);
    if (!out) return nullptr;
    gt_fill_scalar_f64(out, 0.0);
    if (out->numel == 0 || grad_out->numel == 0 || grad_out->device_type == GT_DEV_META) return out;

    gt_contig_owner gc(grad_out), ic(index);
    gt_host_view hvg, hvi, hvo;
    const void * gp = hvg.read(gc.ptr);
    const void * ip = hvi.read(ic.ptr);
    void * op = hvo.read_write(out);

    int64_t idx_len = (index->ndim == 0) ? 1 : index->shape[0];
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < dim; ++d) outer *= out_shape[d];
    for (int d = dim + 1; d < nd; ++d) inner *= out_shape[d];

    for (int64_t i = 0; i < idx_len; ++i) {
        int64_t sel = gt_read_i64_at(ip, ic.ptr->dtype, i);
        if (sel < 0) sel += dim_size;
        if (sel < 0 || sel >= dim_size) continue;
        for (int64_t o = 0; o < outer; ++o) {
            int64_t src_base = (o * idx_len + i) * inner;
            int64_t dst_base = (o * dim_size + sel) * inner;
            for (int64_t j = 0; j < inner; ++j) {
                gt_put_elem(op, out->dtype, dst_base + j, gp, gc.ptr->dtype, src_base + j, true);
            }
        }
    }
    hvo.commit(out);
    return out;
}

// Advanced indexing a[..., i0, i1, ...]: the index tensors (broadcast against each other)
// address the dimensions `dims`, which must be strictly ascending.
//
// Layout of the result, as in numpy and torch:
//  * indexed dimensions next to each other -> the index shape takes their place:
//        a: (P, I0, I1, Q), dims = {1, 2}  ->  (P, *index_shape, Q)
//  * indexed dimensions separated by others -> the index shape comes first:
//        a: (P, I0, Q, I1), dims = {1, 3}  ->  (*index_shape, P, Q)
struct gt_adv_plan {
    int      idx_ndim = 0;
    int64_t  idx_shape[GT_MAX_DIMS] = {};
    int64_t  idx_numel = 1;
    std::vector<int> pre_dims, post_dims;   // dimensions that are not indexed, before / after the index block
    int64_t  pre_size = 1, post_size = 1;
    int      out_ndim = 0;
    int64_t  out_shape[GT_MAX_DIMS] = {};
    std::vector<int64_t> idx_off;           // storage offset contributed by the indexed dims, per index element
};

static bool gt_adv_plan_build(const gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims,
                              const char * what, gt_adv_plan & plan) {
    const std::string w(what);
    if (n_idx > a->ndim) {
        gt_set_error("IndexError: too many indices for tensor of dimension " + std::to_string(a->ndim));
        return false;
    }
    bool indexed[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_idx; ++i) {
        if (!idx_tensors[i]) { gt_set_error(w + ": null index tensor"); return false; }
        if (dims[i] < 0 || dims[i] >= a->ndim || (i > 0 && dims[i] <= dims[i - 1])) {
            gt_set_error(w + ": indexed dimensions must be ascending and inside the tensor");
            return false;
        }
        if (!gt_require_index_dtype(idx_tensors[i], true, what)) return false;
        indexed[dims[i]] = true;
        plan.idx_ndim = std::max(plan.idx_ndim, idx_tensors[i]->ndim);
    }
    for (int k = 0; k < plan.idx_ndim; ++k) {
        int64_t m = 1;
        for (int i = 0; i < n_idx; ++i) {
            int ii = idx_tensors[i]->ndim - 1 - k;
            int64_t d = (ii >= 0) ? idx_tensors[i]->shape[ii] : 1;
            if (d == m || d == 1) continue;
            if (m != 1) {
                gt_set_error("IndexError: shape mismatch: indexing tensors could not be broadcast together");
                return false;
            }
            m = d;
        }
        plan.idx_shape[plan.idx_ndim - 1 - k] = m;
    }
    for (int k = 0; k < plan.idx_ndim; ++k) plan.idx_numel *= plan.idx_shape[k];

    const bool adjacent = (dims[n_idx - 1] - dims[0] + 1 == n_idx);
    const int block_at = adjacent ? dims[0] : 0;
    for (int d = 0; d < a->ndim; ++d) {
        if (indexed[d]) continue;
        if (d < block_at) { plan.pre_dims.push_back(d); plan.pre_size *= a->shape[d]; }
        else { plan.post_dims.push_back(d); plan.post_size *= a->shape[d]; }
    }
    if ((int)plan.pre_dims.size() + plan.idx_ndim + (int)plan.post_dims.size() > GT_MAX_DIMS) {
        gt_set_error(w + ": the result has too many dimensions");
        return false;
    }
    for (int d : plan.pre_dims) plan.out_shape[plan.out_ndim++] = a->shape[d];
    for (int k = 0; k < plan.idx_ndim; ++k) plan.out_shape[plan.out_ndim++] = plan.idx_shape[k];
    for (int d : plan.post_dims) plan.out_shape[plan.out_ndim++] = a->shape[d];

    // Resolve every index now, so that an error is reported before anything is copied
    plan.idx_off.assign((size_t)plan.idx_numel, 0);
    if (plan.idx_numel == 0 || a->device_type == GT_DEV_META) return true;
    for (int m = 0; m < n_idx; ++m) {
        const gt_tensor * it = idx_tensors[m];
        gt_host_view hv;
        const void * ip = hv.read(it);
        const int ad = dims[m];
        const int64_t size = a->shape[ad];
        for (int64_t ii = 0; ii < plan.idx_numel; ++ii) {
            int64_t ioff = it->offset, irem = ii;
            for (int k = plan.idx_ndim - 1; k >= 0; --k) {
                int64_t c = irem % plan.idx_shape[k];
                irem /= plan.idx_shape[k];
                int mi = it->ndim - plan.idx_ndim + k;
                if (mi >= 0 && it->shape[mi] > 1) ioff += c * it->strides[mi];
            }
            const int64_t given = gt_read_i64_at(ip, it->dtype, ioff);
            const int64_t coord = given < 0 ? given + size : given;
            if (coord < 0 || coord >= size) {
                gt_set_index_error(true, what, given, ad, size);
                return false;
            }
            plan.idx_off[(size_t)ii] += coord * a->strides[ad];
        }
    }
    return true;
}

// Storage offset contributed by the un-indexed dimensions `dims_list` for their combined
// row-major position `pos`
static inline int64_t gt_adv_free_offset(const gt_tensor * a, const std::vector<int> & dims_list, int64_t pos) {
    int64_t off = 0;
    for (int k = (int)dims_list.size() - 1; k >= 0; --k) {
        int d = dims_list[k];
        int64_t c = pos % a->shape[d];
        pos /= a->shape[d];
        off += c * a->strides[d];
    }
    return off;
}

extern "C" gt_tensor * gt_index_tensor(const gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims) {
    if (!a || n_idx <= 0 || !idx_tensors || !dims) { gt_set_error("index: invalid arguments"); return nullptr; }
    gt_adv_plan plan;
    if (!gt_adv_plan_build(a, n_idx, idx_tensors, dims, "index", plan)) return nullptr;

    gt_tensor * out = gt_tensor_empty(plan.out_ndim, plan.out_shape, a->dtype, a->device_type, a->device_index);
    if (!out || out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_host_view hva, hvo;
    const void * ap = hva.read(a);
    void * op = hvo.write_empty(out);

    for (int64_t p = 0; p < plan.pre_size; ++p) {
        const int64_t pre_off = gt_adv_free_offset(a, plan.pre_dims, p);
        for (int64_t ii = 0; ii < plan.idx_numel; ++ii) {
            const int64_t base = a->offset + pre_off + plan.idx_off[(size_t)ii];
            for (int64_t q = 0; q < plan.post_size; ++q) {
                const int64_t oidx = (p * plan.idx_numel + ii) * plan.post_size + q;
                gt_copy_elem(op, out->dtype, oidx, ap, a->dtype, base + gt_adv_free_offset(a, plan.post_dims, q));
            }
        }
    }
    hvo.commit(out);
    return out;
}

// a[..., i0, i1, ...] = values (or += values with `accumulate`). `values` holds either one
// element or exactly as many as the indexed result, laid out like gt_index_tensor's output;
// broadcasting anything else to that shape is the caller's job.
extern "C" int gt_index_put(gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims, const gt_tensor * values, int accumulate) {
    if (!a || n_idx <= 0 || !idx_tensors || !dims || !values) { gt_set_error("index_put: invalid arguments"); return -1; }
    gt_adv_plan plan;
    if (!gt_adv_plan_build(a, n_idx, idx_tensors, dims, "index_put", plan)) return -1;

    const int64_t total = plan.pre_size * plan.idx_numel * plan.post_size;
    if (values->numel != 1 && values->numel != total) {
        gt_set_error("index_put: shape mismatch: value tensor of shape " + gt_shape_str(values->ndim, values->shape) +
                     " cannot be broadcast to the indexing result of shape " + gt_shape_str(plan.out_ndim, plan.out_shape));
        return -1;
    }
    if (total == 0 || a->device_type == GT_DEV_META) return 0;

    gt_contig_owner vc(values);
    if (!vc.ptr) return -1;
    gt_host_view hva, hvv;
    void * ap = hva.read_write(a);
    const void * vp = hvv.read(vc.ptr);
    const bool one_value = (vc.ptr->numel == 1);

    for (int64_t p = 0; p < plan.pre_size; ++p) {
        const int64_t pre_off = gt_adv_free_offset(a, plan.pre_dims, p);
        for (int64_t ii = 0; ii < plan.idx_numel; ++ii) {
            const int64_t base = a->offset + pre_off + plan.idx_off[(size_t)ii];
            for (int64_t q = 0; q < plan.post_size; ++q) {
                const int64_t oidx = (p * plan.idx_numel + ii) * plan.post_size + q;
                const int64_t aoff = base + gt_adv_free_offset(a, plan.post_dims, q);
                gt_put_elem(ap, a->dtype, aoff, vp, vc.ptr->dtype, one_value ? 0 : oidx, accumulate != 0);
            }
        }
    }
    hva.commit(a);
    return 0;
}

// Elements of `a` where `mask` is non-zero, as a 1-D tensor in row-major order. `a` and
// `mask` are broadcast against each other first.
extern "C" gt_tensor * gt_masked_select(const gt_tensor * a, const gt_tensor * mask) {
    if (!a || !mask) { gt_set_error("masked_select: null tensor"); return nullptr; }
    const int nd = std::max(a->ndim, mask->ndim);
    int64_t shape[GT_MAX_DIMS] = {};
    int64_t a_bs[GT_MAX_DIMS] = {}, m_bs[GT_MAX_DIMS] = {};
    int64_t total = 1;
    for (int i = 0; i < nd; ++i) {
        const int ai = a->ndim - nd + i, mi = mask->ndim - nd + i;
        const int64_t da = (ai >= 0) ? a->shape[ai] : 1;
        const int64_t dm = (mi >= 0) ? mask->shape[mi] : 1;
        if (da != dm && da != 1 && dm != 1) {
            gt_set_error("masked_select: the mask of shape " + gt_shape_str(mask->ndim, mask->shape) +
                         " cannot be broadcast with the tensor of shape " + gt_shape_str(a->ndim, a->shape));
            return nullptr;
        }
        shape[i] = (da == 1) ? dm : da;
        a_bs[i] = (ai < 0 || da == 1) ? 0 : a->strides[ai];
        m_bs[i] = (mi < 0 || dm == 1) ? 0 : mask->strides[mi];
        total *= shape[i];
    }

    std::vector<int64_t> picked;     // storage offsets into `a`
    gt_host_view hva, hvm;
    const void * ap = nullptr;
    if (total > 0 && a->device_type != GT_DEV_META) {
        ap = hva.read(a);
        const void * mp = hvm.read(mask);
        for (int64_t idx = 0; idx < total; ++idx) {
            int64_t aoff = a->offset, moff = mask->offset, rem = idx;
            for (int d = nd - 1; d >= 0; --d) {
                const int64_t c = rem % shape[d];
                rem /= shape[d];
                aoff += c * a_bs[d];
                moff += c * m_bs[d];
            }
            if (gt_read_f64_at(mp, mask->dtype, moff) != 0.0) picked.push_back(aoff);
        }
    }

    int64_t cnt = (int64_t)picked.size();
    gt_tensor * out = gt_tensor_empty(1, &cnt, a->dtype, a->device_type, a->device_index);
    if (!out || cnt == 0) return out;
    gt_host_view hvo;
    void * op = hvo.write_empty(out);
    for (int64_t i = 0; i < cnt; ++i) gt_copy_elem(op, out->dtype, i, ap, a->dtype, picked[(size_t)i]);
    hvo.commit(out);
    return out;
}

extern "C" int gt_masked_fill_inplace(gt_tensor * a, const gt_tensor * mask, double value) {
    if (!a || !mask) return -1;
    int nd = a->ndim;
    int64_t m_bstrides[GT_MAX_DIMS] = {};
    for (int i = 0; i < nd; ++i) {
        int mi = mask->ndim - nd + i;
        if (mi < 0 || mask->shape[mi] == 1) m_bstrides[i] = 0;
        else if (mask->shape[mi] == a->shape[i]) m_bstrides[i] = mask->strides[mi];
        else { gt_set_error("masked_fill: incompatible mask shape"); return -1; }
    }
    gt_host_view hva, hvm;
    void * ap = hva.read_write(a);
    const void * mp = hvm.read(mask);

    for (int64_t idx = 0; idx < a->numel; ++idx) {
        int64_t aoff = a->offset, moff = mask->offset, rem = idx;
        for (int d = nd - 1; d >= 0; --d) {
            int64_t c = rem % a->shape[d]; rem /= a->shape[d];
            aoff += c * a->strides[d];
            moff += c * m_bstrides[d];
        }
        if (gt_read_i64_at(mp, mask->dtype, moff) != 0) gt_write_f64_at(ap, a->dtype, aoff, value);
    }
    hva.commit(a);
    return 0;
}

extern "C" gt_tensor * gt_nonzero(const gt_tensor * a) {
    if (!a) return nullptr;
    gt_host_view hva;
    const void * ap = hva.read(a);
    int nd = a->ndim;
    std::vector<int64_t> coords;
    for (int64_t i = 0; i < a->numel; ++i) {
        int64_t aoff = gt_check_contiguous(a) ? (a->offset + i) : gt_linear_to_elem_offset(a, i);
        if (gt_read_f64_at(ap, a->dtype, aoff) != 0.0) {
            int64_t rem = i, tmp[GT_MAX_DIMS];
            for (int d = nd - 1; d >= 0; --d) { tmp[d] = rem % a->shape[d]; rem /= a->shape[d]; }
            for (int d = 0; d < nd; ++d) coords.push_back(tmp[d]);
        }
    }
    int64_t cnt = (nd > 0) ? ((int64_t)coords.size() / nd) : ((coords.empty()) ? 0 : 1);
    int64_t osh[2] = {cnt, std::max(nd, 0)};
    gt_tensor * out = gt_tensor_empty(2, osh, GT_I64, a->device_type, a->device_index);
    if (!out || out->numel == 0) return out;
    gt_host_view hvo;
    void * op = hvo.write_empty(out);
    std::memcpy(op, coords.data(), coords.size() * sizeof(int64_t));
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_tril_triu(const gt_tensor * a, int64_t diagonal, int upper) {
    if (!a || a->ndim < 2) { gt_set_error("tril/triu expects >= 2D tensor"); return nullptr; }
    int nd = a->ndim;
    int64_t R = a->shape[nd - 2], C = a->shape[nd - 1];
    gt_tensor * out = gt_tensor_empty(nd, a->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_contig_owner ac(a);
    gt_host_view hva, hvo;
    const void * ap = hva.read(ac.ptr);
    void * op = hvo.write_empty(out);
    int64_t outer = a->numel / std::max<int64_t>(R * C, 1);
    for (int64_t b = 0; b < outer; ++b) {
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t c = 0; c < C; ++c) {
                int64_t idx = (b * R + r) * C + c;
                bool keep = upper ? (c - r >= diagonal) : (c - r <= diagonal);
                double v = keep ? gt_read_f64_at(ap, ac.ptr->dtype, idx) : 0.0;
                gt_write_f64_at(op, out->dtype, idx, v);
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_diag(const gt_tensor * a, int64_t diagonal) {
    if (!a || (a->ndim != 1 && a->ndim != 2)) { gt_set_error("diag expects 1D or 2D tensor"); return nullptr; }
    gt_contig_owner ac(a);
    gt_host_view hva;
    const void * ap = hva.read(ac.ptr);
    if (a->ndim == 1) {
        int64_t n = a->shape[0];
        int64_t sz = n + std::abs(diagonal);
        int64_t osh[2] = {sz, sz};
        gt_tensor * out = gt_tensor_empty(2, osh, a->dtype, a->device_type, a->device_index);
        gt_fill_scalar_f64(out, 0.0);
        gt_host_view hvo;
        void * op = hvo.read_write(out);
        for (int64_t i = 0; i < n; ++i) {
            int64_t r = (diagonal >= 0) ? i : (i - diagonal);
            int64_t c = (diagonal >= 0) ? (i + diagonal) : i;
            gt_write_f64_at(op, out->dtype, r * sz + c, gt_read_f64_at(ap, ac.ptr->dtype, i));
        }
        hvo.commit(out);
        return out;
    } else {
        int64_t R = a->shape[0], C = a->shape[1];
        int64_t r0 = (diagonal < 0) ? -diagonal : 0;
        int64_t c0 = (diagonal > 0) ? diagonal : 0;
        int64_t len = std::max<int64_t>(0, std::min(R - r0, C - c0));
        gt_tensor * out = gt_tensor_empty(1, &len, a->dtype, a->device_type, a->device_index);
        if (len == 0) return out;
        gt_host_view hvo;
        void * op = hvo.write_empty(out);
        for (int64_t i = 0; i < len; ++i)
            gt_write_f64_at(op, out->dtype, i, gt_read_f64_at(ap, ac.ptr->dtype, (r0 + i) * C + (c0 + i)));
        hvo.commit(out);
        return out;
    }
}

extern "C" gt_tensor * gt_pad(const gt_tensor * a, int n_pairs, const int64_t * pads, int mode, double value) {
    // n_pairs: number of (left, right) pairs from last dim backwards (pads has 2 * n_pairs ints)
    if (!a || n_pairs < 0 || n_pairs > a->ndim) {
        gt_set_error("pad: invalid pad specification");
        return nullptr;
    }
    int nd = a->ndim;
    int64_t out_shape[GT_MAX_DIMS];
    int64_t pad_left[GT_MAX_DIMS] = {};
    for (int d = 0; d < nd; ++d) out_shape[d] = a->shape[d];
    for (int p = 0; p < n_pairs; ++p) {
        int d = nd - 1 - p;
        pad_left[d] = pads[2 * p];
        out_shape[d] += pads[2 * p] + pads[2 * p + 1];
        if (out_shape[d] < 0) { gt_set_error("pad: negative output dimension"); return nullptr; }
    }

    gt_tensor * out = gt_tensor_empty(nd, out_shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_contig_owner ac(a);
    gt_host_view hva, hvo;
    const void * ap = hva.read(ac.ptr);
    void * op = hvo.write_empty(out);

    for (int64_t idx = 0; idx < out->numel; ++idx) {
        int64_t rem = idx, in_off = 0;
        bool in_bounds = true;
        for (int d = nd - 1; d >= 0; --d) {
            int64_t c = rem % out_shape[d]; rem /= out_shape[d];
            int64_t src_c = c - pad_left[d];
            int64_t S = a->shape[d];
            if (src_c < 0 || src_c >= S) {
                if (mode == 0) { in_bounds = false; break; }
                else if (mode == 2) { src_c = std::clamp<int64_t>(src_c, 0, S - 1); }
                else if (mode == 3) { src_c = ((src_c % S) + S) % S; }
                else if (mode == 1) {
                    if (S == 1) src_c = 0;
                    else while (src_c < 0 || src_c >= S) {
                        if (src_c < 0) src_c = -src_c;
                        if (src_c >= S) src_c = 2 * (S - 1) - src_c;
                    }
                }
            }
            in_off += src_c * ac.ptr->strides[d];
        }
        gt_write_f64_at(op, out->dtype, idx, in_bounds ? gt_read_f64_at(ap, ac.ptr->dtype, in_off) : value);
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_flip(const gt_tensor * a, int n_dims, const int * dims) {
    if (!a) return nullptr;
    bool flip_d[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_dims; ++i) {
        int d = dims[i];
        if (d < 0) d += a->ndim;
        if (d < 0 || d >= a->ndim) { gt_set_error("flip: dim out of range"); return nullptr; }
        flip_d[d] = true;
    }
    gt_tensor * out = gt_tensor_empty(a->ndim, a->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_host_view hva, hvo;
    const void * ap = hva.read(a);
    void * op = hvo.write_empty(out);
    for (int64_t idx = 0; idx < out->numel; ++idx) {
        int64_t rem = idx, aoff = a->offset;
        for (int d = a->ndim - 1; d >= 0; --d) {
            int64_t c = rem % a->shape[d]; rem /= a->shape[d];
            if (flip_d[d]) c = a->shape[d] - 1 - c;
            aoff += c * a->strides[d];
        }
        if (gt_is_complex(out->dtype)) gt_write_c128_at(op, out->dtype, idx, gt_read_c128_at(ap, a->dtype, aoff));
        else if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL) gt_write_i64_at(op, out->dtype, idx, gt_read_i64_at(ap, a->dtype, aoff));
        else gt_write_f64_at(op, out->dtype, idx, gt_read_f64_at(ap, a->dtype, aoff));
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_roll(const gt_tensor * a, int n_shifts, const int64_t * shifts, const int * dims) {
    if (!a) return nullptr;
    if (n_shifts == 0) return gt_tensor_clone(a);
    if (!dims) {
        gt_contig_owner ac(a);
        gt_tensor * out = gt_tensor_empty(a->ndim, a->shape, a->dtype, a->device_type, a->device_index);
        if (!out || out->numel == 0) return out;
        gt_host_view hva, hvo;
        const void * ap = hva.read(ac.ptr);
        void * op = hvo.write_empty(out);
        int64_t sh = ((shifts[0] % out->numel) + out->numel) % out->numel;
        for (int64_t i = 0; i < out->numel; ++i) {
            int64_t dst = (i + sh) % out->numel;
            if (gt_is_complex(out->dtype)) gt_write_c128_at(op, out->dtype, dst, gt_read_c128_at(ap, ac.ptr->dtype, i));
            else if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL) gt_write_i64_at(op, out->dtype, dst, gt_read_i64_at(ap, ac.ptr->dtype, i));
            else gt_write_f64_at(op, out->dtype, dst, gt_read_f64_at(ap, ac.ptr->dtype, i));
        }
        hvo.commit(out);
        return out;
    }
    int64_t shift_d[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_shifts; ++i) {
        int d = dims[i];
        if (d < 0) d += a->ndim;
        if (d < 0 || d >= a->ndim) { gt_set_error("roll: dim out of range"); return nullptr; }
        shift_d[d] += shifts[i];
    }
    gt_tensor * out = gt_tensor_empty(a->ndim, a->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_host_view hva, hvo;
    const void * ap = hva.read(a);
    void * op = hvo.write_empty(out);
    for (int64_t idx = 0; idx < out->numel; ++idx) {
        int64_t rem = idx, aoff = a->offset;
        for (int d = a->ndim - 1; d >= 0; --d) {
            int64_t c = rem % a->shape[d]; rem /= a->shape[d];
            if (shift_d[d] != 0 && a->shape[d] > 0) {
                int64_t S = a->shape[d];
                c = (((c - shift_d[d]) % S) + S) % S;
            }
            aoff += c * a->strides[d];
        }
        if (gt_is_complex(out->dtype)) gt_write_c128_at(op, out->dtype, idx, gt_read_c128_at(ap, a->dtype, aoff));
        else if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL) gt_write_i64_at(op, out->dtype, idx, gt_read_i64_at(ap, a->dtype, aoff));
        else gt_write_f64_at(op, out->dtype, idx, gt_read_f64_at(ap, a->dtype, aoff));
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_bincount(const gt_tensor * input, const gt_tensor * weights, int64_t minlength) {
    if (!input || input->ndim != 1) { gt_set_error("bincount expects 1D integer tensor"); return nullptr; }
    gt_contig_owner ic(input), wc(weights);
    gt_host_view hvi, hvw;
    const void * ip = hvi.read(ic.ptr);
    const void * wp = wc.ptr ? hvw.read(wc.ptr) : nullptr;
    int64_t mx = minlength - 1;
    for (int64_t i = 0; i < ic.ptr->numel; ++i) {
        int64_t v = gt_read_i64_at(ip, ic.ptr->dtype, i);
        if (v < 0) { gt_set_error("bincount requires non-negative values"); return nullptr; }
        if (v > mx) mx = v;
    }
    int64_t len = std::max<int64_t>(0, mx + 1);
    int out_dt = wc.ptr ? wc.ptr->dtype : GT_I64;
    gt_tensor * out = gt_tensor_empty(1, &len, out_dt, input->device_type, input->device_index);
    if (!out || len == 0) return out;
    gt_fill_scalar_f64(out, 0.0);
    gt_host_view hvo;
    void * op = hvo.read_write(out);
    for (int64_t i = 0; i < ic.ptr->numel; ++i) {
        int64_t idx = gt_read_i64_at(ip, ic.ptr->dtype, i);
        if (wp) gt_write_f64_at(op, out_dt, idx, gt_read_f64_at(op, out_dt, idx) + gt_read_f64_at(wp, wc.ptr->dtype, i));
        else gt_write_i64_at(op, GT_I64, idx, gt_read_i64_at(op, GT_I64, idx) + 1);
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_unique(const gt_tensor * input, int sorted, int return_inverse, int return_counts,
                         gt_tensor ** out_vals, gt_tensor ** out_inv, gt_tensor ** out_cnt) {
    (void)sorted;
    if (!input || !out_vals) return -1;
    gt_contig_owner ic(input);
    gt_host_view hvi;
    const void * ip = hvi.read(ic.ptr);
    int64_t N = ic.ptr->numel;
    std::vector<double> vals(N);
    for (int64_t i = 0; i < N; ++i) vals[i] = gt_read_f64_at(ip, ic.ptr->dtype, i);
    std::vector<double> uniq = vals;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    int64_t U = (int64_t)uniq.size();
    gt_tensor * v_t = gt_tensor_empty(1, &U, input->dtype, input->device_type, input->device_index);
    if (U > 0) {
        gt_host_view hvv;
        void * vp = hvv.write_empty(v_t);
        for (int64_t i = 0; i < U; ++i) gt_write_f64_at(vp, v_t->dtype, i, uniq[i]);
        hvv.commit(v_t);
    }
    *out_vals = v_t;

    if (return_inverse && out_inv) {
        gt_tensor * inv_t = gt_tensor_empty(input->ndim, input->shape, GT_I64, input->device_type, input->device_index);
        if (N > 0) {
            gt_host_view hvinv;
            void * invp = hvinv.write_empty(inv_t);
            for (int64_t i = 0; i < N; ++i)
                gt_write_i64_at(invp, GT_I64, i, (int64_t)(std::lower_bound(uniq.begin(), uniq.end(), vals[i]) - uniq.begin()));
            hvinv.commit(inv_t);
        }
        *out_inv = inv_t;
    }
    if (return_counts && out_cnt) {
        gt_tensor * cnt_t = gt_tensor_empty(1, &U, GT_I64, input->device_type, input->device_index);
        gt_fill_scalar_f64(cnt_t, 0.0);
        if (U > 0 && N > 0) {
            gt_host_view hvc;
            void * cp = hvc.read_write(cnt_t);
            for (int64_t i = 0; i < N; ++i) {
                int64_t k = (int64_t)(std::lower_bound(uniq.begin(), uniq.end(), vals[i]) - uniq.begin());
                gt_write_i64_at(cp, GT_I64, k, gt_read_i64_at(cp, GT_I64, k) + 1);
            }
            hvc.commit(cnt_t);
        }
        *out_cnt = cnt_t;
    }
    return 0;
}

// ---------- Linear Algebra ----------

static bool gt_lu_pivot_solve(int64_t N, int64_t K, std::vector<double> & A, std::vector<double> & B, double & det_out) {
    det_out = 1.0;
    for (int64_t i = 0; i < N; ++i) {
        int64_t piv = i;
        double best = std::abs(A[i * N + i]);
        for (int64_t r = i + 1; r < N; ++r) {
            double v = std::abs(A[r * N + i]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-14) { det_out = 0.0; return false; }
        if (piv != i) {
            det_out = -det_out;
            for (int64_t c = 0; c < N; ++c) std::swap(A[i * N + c], A[piv * N + c]);
            for (int64_t c = 0; c < K; ++c) std::swap(B[i * K + c], B[piv * K + c]);
        }
        double diag = A[i * N + i];
        det_out *= diag;
        for (int64_t r = i + 1; r < N; ++r) {
            double f = A[r * N + i] / diag;
            if (f == 0.0) continue;
            for (int64_t c = i; c < N; ++c) A[r * N + c] -= f * A[i * N + c];
            for (int64_t c = 0; c < K; ++c) B[r * K + c] -= f * B[i * K + c];
        }
    }
    for (int64_t i = N - 1; i >= 0; --i) {
        double diag = A[i * N + i];
        for (int64_t c = 0; c < K; ++c) {
            double s = B[i * K + c];
            for (int64_t k = i + 1; k < N; ++k) s -= A[i * N + k] * B[k * K + c];
            B[i * K + c] = s / diag;
        }
    }
    return true;
}

extern "C" gt_tensor * gt_linalg_inv(const gt_tensor * a) {
    if (!a || a->ndim < 2 || a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        gt_set_error("linalg.inv expects square matrix (..., N, N)");
        return nullptr;
    }
    int64_t N = a->shape[a->ndim - 1], B = a->numel / std::max<int64_t>(N * N, 1);
    gt_tensor * out = gt_tensor_empty(a->ndim, a->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_contig_owner ac(a);
    gt_host_view hva, hvo;
    const void * ap = hva.read(ac.ptr);
    void * op = hvo.write_empty(out);
    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> mat(N * N), rhs(N * N, 0.0);
        for (int64_t i = 0; i < N * N; ++i) mat[i] = gt_read_f64_at(ap, ac.ptr->dtype, b * N * N + i);
        for (int64_t i = 0; i < N; ++i) rhs[i * N + i] = 1.0;
        double det = 0.0;
        if (!gt_lu_pivot_solve(N, N, mat, rhs, det)) {
            gt_set_error("linalg.inv: singular matrix");
            gt_tensor_free(out);
            return nullptr;
        }
        for (int64_t i = 0; i < N * N; ++i) gt_write_f64_at(op, out->dtype, b * N * N + i, rhs[i]);
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_linalg_solve(const gt_tensor * a, const gt_tensor * b_in) {
    if (!a || !b_in || a->ndim < 2 || a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        gt_set_error("linalg.solve: invalid A shape");
        return nullptr;
    }
    int64_t N = a->shape[a->ndim - 1];
    bool vec_rhs = (b_in->ndim == a->ndim - 1);
    int64_t K = vec_rhs ? 1 : b_in->shape[b_in->ndim - 1];
    int64_t B = a->numel / std::max<int64_t>(N * N, 1);
    gt_tensor * out = gt_tensor_empty(b_in->ndim, b_in->shape, gt_promote_dtype(a->dtype, b_in->dtype), a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_contig_owner ac(a), bc(b_in);
    gt_host_view hva, hvb, hvo;
    const void * ap = hva.read(ac.ptr);
    const void * bp = hvb.read(bc.ptr);
    void * op = hvo.write_empty(out);
    for (int64_t batch = 0; batch < B; ++batch) {
        std::vector<double> mat(N * N), rhs(N * K);
        for (int64_t i = 0; i < N * N; ++i) mat[i] = gt_read_f64_at(ap, ac.ptr->dtype, batch * N * N + i);
        for (int64_t i = 0; i < N * K; ++i) rhs[i] = gt_read_f64_at(bp, bc.ptr->dtype, batch * N * K + i);
        double det = 0.0;
        if (!gt_lu_pivot_solve(N, K, mat, rhs, det)) {
            gt_set_error("linalg.solve: singular matrix");
            gt_tensor_free(out);
            return nullptr;
        }
        for (int64_t i = 0; i < N * K; ++i) gt_write_f64_at(op, out->dtype, batch * N * K + i, rhs[i]);
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_linalg_cholesky(const gt_tensor * a, int upper) {
    if (!a || a->ndim < 2 || a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        gt_set_error("linalg.cholesky expects square matrix");
        return nullptr;
    }
    int64_t N = a->shape[a->ndim - 1], B = a->numel / std::max<int64_t>(N * N, 1);
    gt_tensor * out = gt_tensor_empty(a->ndim, a->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    gt_contig_owner ac(a);
    gt_host_view hva, hvo;
    const void * ap = hva.read(ac.ptr);
    void * op = hvo.write_empty(out);
    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> L(N * N, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double s = gt_read_f64_at(ap, ac.ptr->dtype, (b * N + i) * N + j);
                for (int64_t k = 0; k < j; ++k) s -= L[i * N + k] * L[j * N + k];
                if (i == j) {
                    if (s <= 0.0) { gt_set_error("linalg.cholesky: matrix not positive definite"); gt_tensor_free(out); return nullptr; }
                    L[i * N + j] = std::sqrt(s);
                } else {
                    L[i * N + j] = s / L[j * N + j];
                }
            }
        }
        for (int64_t i = 0; i < N; ++i)
            for (int64_t j = 0; j < N; ++j)
                gt_write_f64_at(op, out->dtype, (b * N + i) * N + j, upper ? L[j * N + i] : L[i * N + j]);
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_linalg_qr(const gt_tensor * a, gt_tensor ** out_q, gt_tensor ** out_r, int mode) {
    // mode: 0 = reduced, 1 = complete
    if (!a || a->ndim < 2 || !out_q || !out_r) return -1;
    bool mode_reduced = (mode == 0);
    int nd = a->ndim;
    int64_t M = a->shape[nd - 2], N = a->shape[nd - 1], K = std::min(M, N);
    int64_t B = a->numel / std::max<int64_t>(M * N, 1);

    int64_t q_sh[GT_MAX_DIMS], r_sh[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) { q_sh[d] = a->shape[d]; r_sh[d] = a->shape[d]; }
    q_sh[nd - 1] = mode_reduced ? K : M;
    r_sh[nd - 2] = mode_reduced ? K : M;

    gt_tensor * Q = gt_tensor_empty(nd, q_sh, a->dtype, a->device_type, a->device_index);
    gt_tensor * R = gt_tensor_empty(nd, r_sh, a->dtype, a->device_type, a->device_index);
    gt_contig_owner ac(a);
    gt_host_view hva, hvq, hvr;
    const void * ap = hva.read(ac.ptr);
    void * qp = hvq.write_empty(Q);
    void * rp = hvr.write_empty(R);

    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> Rm(M * N), Qm(M * M, 0.0);
        for (int64_t i = 0; i < M * N; ++i) Rm[i] = gt_read_f64_at(ap, ac.ptr->dtype, b * M * N + i);
        for (int64_t i = 0; i < M; ++i) Qm[i * M + i] = 1.0;
        for (int64_t k = 0; k < K; ++k) {
            double norm_x = 0.0;
            for (int64_t i = k; i < M; ++i) norm_x += Rm[i * N + k] * Rm[i * N + k];
            norm_x = std::sqrt(norm_x);
            if (norm_x < 1e-15) continue;
            double x0 = Rm[k * N + k];
            double alpha = (x0 >= 0.0) ? -norm_x : norm_x;
            std::vector<double> v(M - k, 0.0);
            v[0] = x0 - alpha;
            for (int64_t i = 1; i < M - k; ++i) v[i] = Rm[(k + i) * N + k];
            double vnorm2 = 0.0;
            for (double vi : v) vnorm2 += vi * vi;
            if (vnorm2 < 1e-30) continue;
            for (int64_t j = k; j < N; ++j) {
                double dot = 0.0;
                for (int64_t i = 0; i < M - k; ++i) dot += v[i] * Rm[(k + i) * N + j];
                double sc = 2.0 * dot / vnorm2;
                for (int64_t i = 0; i < M - k; ++i) Rm[(k + i) * N + j] -= sc * v[i];
            }
            for (int64_t r = 0; r < M; ++r) {
                double dot = 0.0;
                for (int64_t i = 0; i < M - k; ++i) dot += Qm[r * M + (k + i)] * v[i];
                double sc = 2.0 * dot / vnorm2;
                for (int64_t i = 0; i < M - k; ++i) Qm[r * M + (k + i)] -= sc * v[i];
            }
        }
        int64_t qcols = q_sh[nd - 1], rrows = r_sh[nd - 2];
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < qcols; ++j)
                gt_write_f64_at(qp, Q->dtype, (b * M + i) * qcols + j, Qm[i * M + j]);
        for (int64_t i = 0; i < rrows; ++i)
            for (int64_t j = 0; j < N; ++j)
                gt_write_f64_at(rp, R->dtype, (b * rrows + i) * N + j, (i > j) ? 0.0 : Rm[i * N + j]);
    }
    hvq.commit(Q); hvr.commit(R);
    *out_q = Q; *out_r = R;
    return 0;
}

extern "C" int gt_linalg_eigh(const gt_tensor * a, int upper, gt_tensor ** out_evals, gt_tensor ** out_evecs) {
    if (!a || a->ndim < 2 || a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) return -1;
    int nd = a->ndim;
    int64_t N = a->shape[nd - 1], B = a->numel / std::max<int64_t>(N * N, 1);
    int64_t w_sh[GT_MAX_DIMS];
    for (int d = 0; d < nd - 1; ++d) w_sh[d] = a->shape[d];
    gt_tensor * W = gt_tensor_empty(nd - 1, w_sh, a->dtype, a->device_type, a->device_index);
    gt_tensor * V = out_evecs ? gt_tensor_empty(nd, a->shape, a->dtype, a->device_type, a->device_index) : nullptr;
    gt_contig_owner ac(a);
    gt_host_view hva, hvw, hvv;
    const void * ap = hva.read(ac.ptr);
    void * wp = hvw.write_empty(W);
    void * vp = V ? hvv.write_empty(V) : nullptr;

    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> M(N * N), Q(N * N, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            Q[i * N + i] = 1.0;
            for (int64_t j = 0; j < N; ++j) {
                int64_t r = upper ? std::min(i, j) : std::max(i, j);
                int64_t c = upper ? std::max(i, j) : std::min(i, j);
                M[i * N + j] = gt_read_f64_at(ap, ac.ptr->dtype, (b * N + r) * N + c);
            }
        }
        for (int sweep = 0; sweep < 50; ++sweep) {
            double off = 0.0;
            for (int64_t p = 0; p < N; ++p)
                for (int64_t q = p + 1; q < N; ++q)
                    off += M[p * N + q] * M[p * N + q];
            if (off < 1e-28) break;
            for (int64_t p = 0; p < N; ++p) {
                for (int64_t q = p + 1; q < N; ++q) {
                    double apq = M[p * N + q];
                    if (std::abs(apq) < 1e-16) continue;
                    double tau = (M[q * N + q] - M[p * N + p]) / (2.0 * apq);
                    double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                            : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                    double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                    double app = M[p * N + p], aqq = M[q * N + q];
                    M[p * N + p] = app - t * apq; M[q * N + q] = aqq + t * apq;
                    M[p * N + q] = M[q * N + p] = 0.0;
                    for (int64_t k = 0; k < N; ++k) {
                        if (k == p || k == q) continue;
                        double mkp = M[k * N + p], mkq = M[k * N + q];
                        M[k * N + p] = M[p * N + k] = c * mkp - s * mkq;
                        M[k * N + q] = M[q * N + k] = s * mkp + c * mkq;
                    }
                    for (int64_t k = 0; k < N; ++k) {
                        double qkp = Q[k * N + p], qkq = Q[k * N + q];
                        Q[k * N + p] = c * qkp - s * qkq; Q[k * N + q] = s * qkp + c * qkq;
                    }
                }
            }
        }
        std::vector<std::pair<double, int64_t>> eig(N);
        for (int64_t i = 0; i < N; ++i) eig[i] = {M[i * N + i], i};
        std::sort(eig.begin(), eig.end(), [](const auto & x, const auto & y) { return x.first < y.first; });
        for (int64_t j = 0; j < N; ++j) {
            gt_write_f64_at(wp, W->dtype, b * N + j, eig[j].first);
            if (vp) {
                int64_t col = eig[j].second;
                for (int64_t i = 0; i < N; ++i) gt_write_f64_at(vp, V->dtype, (b * N + i) * N + j, Q[i * N + col]);
            }
        }
    }
    hvw.commit(W);
    if (V) { hvv.commit(V); *out_evecs = V; }
    *out_evals = W;
    return 0;
}

// Fills column k of the M x cols matrix Q with a unit vector orthogonal to every column marked
// in `have`. Starts from the coordinate vector that keeps most of its length once the known
// columns are projected out - at least (M - known) / M of it, so the result never degenerates.
static void gt_svd_complete_column(std::vector<double> & Q, int64_t M, int64_t cols, const std::vector<char> & have, int64_t k) {
    int64_t best = 0;
    double best_left = -1.0;
    for (int64_t i = 0; i < M; ++i) {
        double proj = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            if (have[j]) proj += Q[i * cols + j] * Q[i * cols + j];
        }
        if (1.0 - proj > best_left) {
            best_left = 1.0 - proj;
            best = i;
        }
    }
    for (int64_t i = 0; i < M; ++i) Q[i * cols + k] = (i == best) ? 1.0 : 0.0;
    // Gram-Schmidt, twice: the second pass removes what rounding left behind in the first
    for (int pass = 0; pass < 2; ++pass) {
        for (int64_t j = 0; j < cols; ++j) {
            if (!have[j]) continue;
            double dot = 0.0;
            for (int64_t i = 0; i < M; ++i) dot += Q[i * cols + k] * Q[i * cols + j];
            for (int64_t i = 0; i < M; ++i) Q[i * cols + k] -= dot * Q[i * cols + j];
        }
        double nrm = 0.0;
        for (int64_t i = 0; i < M; ++i) nrm += Q[i * cols + k] * Q[i * cols + k];
        nrm = std::sqrt(nrm);
        if (nrm > 0.0) {
            for (int64_t i = 0; i < M; ++i) Q[i * cols + k] /= nrm;
        }
    }
}

// A = U diag(S) Vh for every matrix of the batch, by one-sided Jacobi rotations.
//   full_matrices == 0: U is (..., M, K), Vh is (..., K, N), K = min(M, N)
//   full_matrices != 0: U is (..., M, M), Vh is (..., N, N)
// U and Vh always have orthonormal columns/rows - also for rank-deficient matrices, where the
// vectors that belong to zero singular values are chosen to complete the basis.
extern "C" int gt_linalg_svd(const gt_tensor * a, int full_matrices, gt_tensor ** out_u, gt_tensor ** out_s, gt_tensor ** out_vh) {
    if (!out_u || !out_s || !out_vh) {
        gt_set_error("linalg.svd: missing output arguments");
        return -1;
    }
    *out_u = *out_s = *out_vh = nullptr;
    if (!a || a->ndim < 2) {
        gt_set_error("linalg.svd: The input tensor A must have at least 2 dimensions.");
        return -1;
    }
    if (!gt_is_floating(a->dtype)) {
        gt_set_error(gt_is_complex(a->dtype) ? "linalg.svd: complex matrices are not supported"
                                             : "linalg.svd: Expected a floating point tensor as input");
        return -1;
    }
    const int nd = a->ndim;
    const int64_t M0 = a->shape[nd - 2], N0 = a->shape[nd - 1], K = std::min(M0, N0);
    int64_t B = 1;
    for (int d = 0; d < nd - 2; ++d) B *= a->shape[d];

    int64_t u_sh[GT_MAX_DIMS], s_sh[GT_MAX_DIMS], vh_sh[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) { u_sh[d] = a->shape[d]; vh_sh[d] = a->shape[d]; }
    for (int d = 0; d < nd - 1; ++d) s_sh[d] = a->shape[d];
    u_sh[nd - 1] = full_matrices ? M0 : K;
    s_sh[nd - 2] = K;
    vh_sh[nd - 2] = full_matrices ? N0 : K;

    gt_contig_owner ac(a);
    if (!ac.ptr) return -1;
    gt_host_view hva;
    const void * ap = hva.read(ac.ptr);
    for (int64_t i = 0; i < a->numel; ++i) {
        if (!std::isfinite(gt_read_f64_at(ap, ac.ptr->dtype, i))) {
            gt_set_error("linalg.svd: the input matrix contains non-finite values");
            return -1;
        }
    }

    gt_tensor * U = gt_tensor_empty(nd, u_sh, a->dtype, a->device_type, a->device_index);
    gt_tensor * S = gt_tensor_empty(nd - 1, s_sh, a->dtype, a->device_type, a->device_index);
    gt_tensor * Vh = gt_tensor_empty(nd, vh_sh, a->dtype, a->device_type, a->device_index);
    if (!U || !S || !Vh) {
        gt_tensor_free(U); gt_tensor_free(S); gt_tensor_free(Vh);
        return -1;
    }
    gt_host_view hvu, hvs, hvvh;
    void * up = hvu.write_empty(U);
    void * sp = hvs.write_empty(S);
    void * vhp = hvvh.write_empty(Vh);

    // The rotations work on a matrix with at least as many rows as columns: a wide matrix is
    // decomposed through its transpose, A^T = Q diag(S) V^T, so that A = V diag(S) Q^T.
    const bool transposed = (M0 < N0);
    const int64_t M = transposed ? N0 : M0, N = transposed ? M0 : N0;
    const int64_t qcols = full_matrices ? M : N;          // left vectors computed for the working matrix
    const int64_t out_uc = u_sh[nd - 1], out_vr = vh_sh[nd - 2];

    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> W((size_t)(M * N)), V((size_t)(N * N), 0.0);
        for (int64_t i = 0; i < N; ++i) V[i * N + i] = 1.0;
        double amax = 0.0;
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                const double v = transposed ? gt_read_f64_at(ap, ac.ptr->dtype, (b * M0 + j) * N0 + i)
                                            : gt_read_f64_at(ap, ac.ptr->dtype, (b * M0 + i) * N0 + j);
                W[i * N + j] = v;
                amax = std::max(amax, std::abs(v));
            }
        // The rotations work with squared column norms, which overflow for entries around 1e154
        // and vanish around 1e-154. Scale the matrix to entries of magnitude 1 first - by a
        // power of two, so that nothing is rounded - and scale the singular values back.
        int scale_exp = 0;
        if (amax > 0.0) {
            std::frexp(amax, &scale_exp);
            if (scale_exp != 0) {
                for (double & v : W) v = std::ldexp(v, -scale_exp);
            }
        }
        for (int sweep = 0; sweep < 60; ++sweep) {
            bool conv = true;
            for (int64_t p = 0; p < N; ++p) {
                for (int64_t q = p + 1; q < N; ++q) {
                    double app = 0, aqq = 0, apq = 0;
                    for (int64_t i = 0; i < M; ++i) {
                        double wp = W[i * N + p], wq = W[i * N + q];
                        app += wp * wp; aqq += wq * wq; apq += wp * wq;
                    }
                    if (std::abs(apq) <= 1e-15 * std::sqrt(app * aqq)) continue;
                    conv = false;
                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                            : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                    double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                    for (int64_t i = 0; i < M; ++i) {
                        double wp = W[i * N + p], wq = W[i * N + q];
                        W[i * N + p] = c * wp - s * wq; W[i * N + q] = s * wp + c * wq;
                    }
                    for (int64_t i = 0; i < N; ++i) {
                        double vp = V[i * N + p], vq = V[i * N + q];
                        V[i * N + p] = c * vp - s * vq; V[i * N + q] = s * vp + c * vq;
                    }
                }
            }
            if (conv) break;
        }
        // singular values: the column norms, largest first
        std::vector<std::pair<double, int64_t>> sig((size_t)N);
        for (int64_t j = 0; j < N; ++j) {
            double n2 = 0.0;
            for (int64_t i = 0; i < M; ++i) n2 += W[i * N + j] * W[i * N + j];
            sig[j] = {std::sqrt(n2), j};
        }
        std::stable_sort(sig.begin(), sig.end(), [](const auto & x, const auto & y) { return x.first > y.first; });

        // Left vectors: a column divided by its norm - where the norm is not (numerically) zero.
        // The others, and for full matrices the M - N extra ones, complete the orthonormal basis.
        const double smax = N > 0 ? sig[0].first : 0.0;
        const double tol = smax * (double)std::max<int64_t>(M, 1) * std::numeric_limits<double>::epsilon();
        std::vector<double> Q((size_t)(M * qcols), 0.0), Vrows((size_t)(N * N), 0.0);
        std::vector<char> have((size_t)qcols, 0);
        for (int64_t k = 0; k < N; ++k) {
            const double sv = sig[k].first;
            const int64_t sc = sig[k].second;
            gt_write_f64_at(sp, S->dtype, b * K + k, std::ldexp(sv, scale_exp));
            if (sv > tol) {
                for (int64_t i = 0; i < M; ++i) Q[i * qcols + k] = W[i * N + sc] / sv;
                have[k] = 1;
            }
            for (int64_t i = 0; i < N; ++i) Vrows[k * N + i] = V[i * N + sc];
        }
        for (int64_t k = 0; k < qcols; ++k) {
            if (!have[k]) {
                gt_svd_complete_column(Q, M, qcols, have, k);
                have[k] = 1;
            }
        }

        if (!transposed) {
            for (int64_t i = 0; i < M0; ++i)
                for (int64_t j = 0; j < out_uc; ++j)
                    gt_write_f64_at(up, U->dtype, (b * M0 + i) * out_uc + j, Q[i * qcols + j]);
            for (int64_t i = 0; i < out_vr; ++i)
                for (int64_t j = 0; j < N0; ++j)
                    gt_write_f64_at(vhp, Vh->dtype, (b * out_vr + i) * N0 + j, Vrows[i * N + j]);
        } else {
            for (int64_t i = 0; i < M0; ++i)
                for (int64_t j = 0; j < out_uc; ++j)
                    gt_write_f64_at(up, U->dtype, (b * M0 + i) * out_uc + j, Vrows[j * N + i]);
            for (int64_t i = 0; i < out_vr; ++i)
                for (int64_t j = 0; j < N0; ++j)
                    gt_write_f64_at(vhp, Vh->dtype, (b * out_vr + i) * N0 + j, Q[j * qcols + i]);
        }
    }
    hvu.commit(U); hvs.commit(S); hvvh.commit(Vh);
    *out_u = U; *out_s = S; *out_vh = Vh;
    return 0;
}

extern "C" gt_tensor * gt_linalg_det(const gt_tensor * a) {
    if (!a || a->ndim < 2 || a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) return nullptr;
    int64_t N = a->shape[a->ndim - 1], B = a->numel / std::max<int64_t>(N * N, 1);
    gt_tensor * out = gt_tensor_empty(a->ndim - 2, a->shape, a->dtype, a->device_type, a->device_index);
    gt_contig_owner ac(a);
    gt_host_view hva, hvo;
    const void * ap = hva.read(ac.ptr);
    void * op = hvo.write_empty(out);
    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> mat(N * N), dummy;
        for (int64_t i = 0; i < N * N; ++i) mat[i] = gt_read_f64_at(ap, ac.ptr->dtype, b * N * N + i);
        double det = 1.0;
        gt_lu_pivot_solve(N, 0, mat, dummy, det);
        gt_write_f64_at(op, out->dtype, b, det);
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_linalg_slogdet(const gt_tensor * a, gt_tensor ** out_sign, gt_tensor ** out_logabsdet) {
    gt_tensor * det = gt_linalg_det(a);
    if (!det) return -1;
    gt_tensor * sign = gt_unary_op(det, GT_UNARY_SIGN, 0, 0);
    gt_tensor * absd = gt_unary_op(det, GT_UNARY_ABS, 0, 0);
    gt_tensor * logd = gt_unary_op(absd, GT_UNARY_LOG, 0, 0);
    gt_tensor_free(det); gt_tensor_free(absd);
    *out_sign = sign; *out_logabsdet = logd;
    return 0;
}

// ---------- 1D FFT ----------

static void gt_fft_radix2(std::vector<std::complex<double>> & x, bool inverse) {
    size_t n = x.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = (inverse ? 2.0 : -2.0) * M_PI / (double)len;
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = x[i + j];
                std::complex<double> v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

static void gt_fft_any(std::vector<std::complex<double>> & x, bool inverse) {
    size_t n = x.size();
    if (n <= 1) return;
    if ((n & (n - 1)) == 0) { gt_fft_radix2(x, inverse); return; }
    size_t m = 1;
    while (m < 2 * n - 1) m <<= 1;
    double sign = inverse ? 1.0 : -1.0;
    std::vector<std::complex<double>> chirp(n), a(m, {0.0, 0.0}), b(m, {0.0, 0.0});
    for (size_t k = 0; k < n; ++k) {
        double ang = sign * M_PI * (double)(k * k) / (double)n;
        chirp[k] = std::complex<double>(std::cos(ang), std::sin(ang));
        a[k] = x[k] * chirp[k];
        b[k] = std::conj(chirp[k]);
        if (k > 0) b[m - k] = b[k];
    }
    gt_fft_radix2(a, false); gt_fft_radix2(b, false);
    for (size_t k = 0; k < m; ++k) a[k] *= b[k];
    gt_fft_radix2(a, true);
    for (size_t k = 0; k < n; ++k) x[k] = (a[k] / (double)m) * chirp[k];
}

extern "C" gt_tensor * gt_fft_1d(const gt_tensor * a, int64_t n, int dim, int inverse, int real_input, int norm_mode) {
    if (!a || a->ndim < 1) return nullptr;
    (void)real_input;
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) { gt_set_error("fft: dim out of range"); return nullptr; }
    int64_t in_len = a->shape[dim];
    int64_t fft_n = (n > 0) ? n : in_len;
    int64_t out_shape[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) out_shape[d] = (d == dim) ? fft_n : a->shape[d];
    int out_dt = (a->dtype == GT_F64 || a->dtype == GT_C128) ? GT_C128 : GT_C64;
    gt_tensor * out = gt_tensor_empty(nd, out_shape, out_dt, a->device_type, a->device_index);
    if (!out || fft_n == 0) return out;

    gt_host_view hva, hvo;
    const void * ap = hva.read(a);
    void * op = hvo.write_empty(out);
    int64_t outer = out->numel / fft_n;
    std::vector<int> other;
    for (int d = 0; d < nd; ++d) if (d != dim) other.push_back(d);

    double scale = 1.0;
    if (norm_mode == 0) scale = inverse ? (1.0 / (double)fft_n) : 1.0;
    else if (norm_mode == 1) scale = 1.0 / std::sqrt((double)fft_n);
    else if (norm_mode == 2) scale = inverse ? 1.0 : (1.0 / (double)fft_n);

    #pragma omp parallel for if (outer > 32)
    for (int64_t o = 0; o < outer; ++o) {
        int64_t bin = a->offset, bout = 0, rem = o;
        for (int k = (int)other.size() - 1; k >= 0; --k) {
            int d = other[k];
            int64_t c = rem % a->shape[d]; rem /= a->shape[d];
            bin += c * a->strides[d]; bout += c * out->strides[d];
        }
        std::vector<std::complex<double>> buf(fft_n, {0.0, 0.0});
        int64_t copy_n = std::min(in_len, fft_n);
        for (int64_t j = 0; j < copy_n; ++j) buf[j] = gt_read_c128_at(ap, a->dtype, bin + j * a->strides[dim]);
        gt_fft_any(buf, inverse != 0);
        for (int64_t j = 0; j < fft_n; ++j) gt_write_c128_at(op, out_dt, bout + j * out->strides[dim], buf[j] * scale);
    }
    hvo.commit(out);
    return out;
}

// ---------- GGUF Native Save / Load / Quantization ----------
//
// GGUF cannot describe every tensor this library has:
//
//   * it stores at most GGML_MAX_DIMS (4) dimensions and writes ggml_n_dims() of
//     them, which drops outer dimensions of size one (and with them everything
//     outside a zero-sized dimension), so (1, 4) comes back as (4,), a scalar as
//     (1,), and a tensor such as (0, 4) used to make the whole file unreadable;
//   * it has no unsigned-byte, boolean or complex element type.
//
// Tensors are therefore written in a form GGUF can hold (uint8/bool as int8
// bytes, complex as pairs of floats, extra dimensions folded, empty tensors as
// zero elements) and whatever that loses is recorded next to them:
//
//   gguf_train.shape.<name>  int64 array   the tensor's real shape
//   gguf_train.dtype.<name>  string        "uint8", "bool", "complex64", "complex128"
//
// Both keys are written only when needed, so files holding ordinary tensors are
// byte-for-byte what they were, and files from other tools load unchanged.

static const char GT_GGUF_SHAPE_PREFIX[] = "gguf_train.shape.";
static const char GT_GGUF_DTYPE_PREFIX[] = "gguf_train.dtype.";
static const char GT_GGUF_ALIGNMENT_KEY[] = "general.alignment";

static bool gt_gguf_key_is_reserved(const char * key) {
    return std::strncmp(key, GT_GGUF_SHAPE_PREFIX, sizeof(GT_GGUF_SHAPE_PREFIX) - 1) == 0 ||
           std::strncmp(key, GT_GGUF_DTYPE_PREFIX, sizeof(GT_GGUF_DTYPE_PREFIX) - 1) == 0;
}

struct gt_gguf_layout {
    enum ggml_type type = GGML_TYPE_F32;
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    const char * dtype_name = nullptr;  // set when the ggml type alone does not identify the dtype
    bool needs_shape = false;           // set when the dimensions GGUF keeps are not the tensor's shape
};

// The shape (outermost dimension first) a reader derives from the stored dimensions alone.
static std::vector<int64_t> gt_gguf_natural_shape(const int64_t * ne, bool complex_pairs) {
    int nd = 1;
    for (int i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        if (ne[i] > 1) { nd = i + 1; break; }
    }
    std::vector<int64_t> shape;
    for (int d = nd - 1; d >= 0; --d) shape.push_back(ne[d]);
    if (complex_pairs) {
        // The innermost dimension holds (real, imaginary).
        if (!shape.empty() && shape.back() == 2) shape.pop_back();
        else shape.clear();
    }
    return shape;
}

static bool gt_gguf_plan(const gt_tensor * t, gt_gguf_layout & lay, std::string & err) {
    const int dt = t->dtype;
    if (gt_is_quantized(dt)) {
        lay.type = (enum ggml_type)(dt - GT_Q_BASE);
        if ((int)lay.type < 0 || lay.type >= GGML_TYPE_COUNT) { err = "unknown quantized dtype"; return false; }
    } else {
        switch (dt) {
            case GT_F32: case GT_F16: case GT_BF16: case GT_F64:
            case GT_I8: case GT_I16: case GT_I32: case GT_I64:
                lay.type = gt_dtype_to_ggml(dt);
                break;
            case GT_U8:   lay.type = GGML_TYPE_I8;  lay.dtype_name = "uint8";      break;
            case GT_BOOL: lay.type = GGML_TYPE_I8;  lay.dtype_name = "bool";       break;
            case GT_C64:  lay.type = GGML_TYPE_F32; lay.dtype_name = "complex64";  break;
            case GT_C128: lay.type = GGML_TYPE_F64; lay.dtype_name = "complex128"; break;
            default:
                err = "dtype cannot be stored in a GGUF file";
                return false;
        }
    }

    const bool cplx = gt_is_complex(dt);
    std::vector<int64_t> dims(t->shape, t->shape + t->ndim);
    if (cplx) dims.push_back(2);
    int64_t count = 1;
    for (int64_t d : dims) count *= d;

    const int64_t blck = ggml_blck_size(lay.type);
    if (blck > 1 && count % blck != 0) {
        err = "quantized tensor does not hold a whole number of blocks";
        return false;
    }

    const int n = (int)dims.size();
    if (count == 0) {
        lay.ne[0] = 0;
    } else if (n == 0) {
        lay.ne[0] = 1;
    } else if (blck > 1 && dims[n - 1] % blck != 0) {
        // GGUF wants every row to be whole blocks; this one is only whole blocks overall.
        lay.ne[0] = count;
    } else {
        for (int d = 0; d < std::min(n, (int)GGML_MAX_DIMS); ++d) lay.ne[d] = dims[n - 1 - d];
        for (int d = GGML_MAX_DIMS; d < n; ++d) lay.ne[GGML_MAX_DIMS - 1] *= dims[n - 1 - d];
    }

    const std::vector<int64_t> natural = gt_gguf_natural_shape(lay.ne, cplx);
    lay.needs_shape = natural != std::vector<int64_t>(t->shape, t->shape + t->ndim);
    return true;
}

static bool gt_gguf_parse_i64(const char * s, long long lo, long long hi, long long & out) {
    if (!s || !*s) return false;
    errno = 0;
    char * end = nullptr;
    out = std::strtoll(s, &end, 10);
    return errno == 0 && end != s && *end == '\0' && out >= lo && out <= hi;
}

static bool gt_gguf_parse_u64(const char * s, unsigned long long hi, unsigned long long & out) {
    if (!s || !*s) return false;
    for (const char * p = s; *p; ++p) {
        if (*p == '-') return false;  // strtoull would quietly wrap a negative number around
    }
    errno = 0;
    char * end = nullptr;
    out = std::strtoull(s, &end, 10);
    return errno == 0 && end != s && *end == '\0' && out <= hi;
}

static bool gt_gguf_parse_f64(const char * s, double & out) {
    if (!s || !*s) return false;
    char * end = nullptr;
    out = std::strtod(s, &end);
    return end != s && *end == '\0';
}

static bool gt_gguf_set_scalar(struct gguf_context * g, const char * k, int type, const char * v, std::string & err) {
    long long i = 0;
    unsigned long long u = 0;
    double f = 0.0;
    bool ok = true;
    switch (type) {
        case GGUF_TYPE_UINT8:   ok = gt_gguf_parse_u64(v, UINT8_MAX, u);  if (ok) gguf_set_val_u8(g, k, (uint8_t)u); break;
        case GGUF_TYPE_INT8:    ok = gt_gguf_parse_i64(v, INT8_MIN, INT8_MAX, i);   if (ok) gguf_set_val_i8(g, k, (int8_t)i); break;
        case GGUF_TYPE_UINT16:  ok = gt_gguf_parse_u64(v, UINT16_MAX, u); if (ok) gguf_set_val_u16(g, k, (uint16_t)u); break;
        case GGUF_TYPE_INT16:   ok = gt_gguf_parse_i64(v, INT16_MIN, INT16_MAX, i); if (ok) gguf_set_val_i16(g, k, (int16_t)i); break;
        case GGUF_TYPE_UINT32:  ok = gt_gguf_parse_u64(v, UINT32_MAX, u); if (ok) gguf_set_val_u32(g, k, (uint32_t)u); break;
        case GGUF_TYPE_INT32:   ok = gt_gguf_parse_i64(v, INT32_MIN, INT32_MAX, i); if (ok) gguf_set_val_i32(g, k, (int32_t)i); break;
        case GGUF_TYPE_UINT64:  ok = gt_gguf_parse_u64(v, UINT64_MAX, u); if (ok) gguf_set_val_u64(g, k, (uint64_t)u); break;
        case GGUF_TYPE_INT64:   ok = gt_gguf_parse_i64(v, INT64_MIN, INT64_MAX, i); if (ok) gguf_set_val_i64(g, k, (int64_t)i); break;
        case GGUF_TYPE_FLOAT32: ok = gt_gguf_parse_f64(v, f); if (ok) gguf_set_val_f32(g, k, (float)f); break;
        case GGUF_TYPE_FLOAT64: ok = gt_gguf_parse_f64(v, f); if (ok) gguf_set_val_f64(g, k, f); break;
        case GGUF_TYPE_BOOL:
            if (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0) gguf_set_val_bool(g, k, true);
            else if (std::strcmp(v, "false") == 0 || std::strcmp(v, "0") == 0) gguf_set_val_bool(g, k, false);
            else ok = false;
            break;
        case GGUF_TYPE_STRING:  gguf_set_val_str(g, k, v); break;
        default:
            err = std::string("metadata key '") + k + "' has an unknown GGUF value type " + std::to_string(type);
            return false;
    }
    if (!ok) {
        err = std::string("metadata key '") + k + "': '" + v + "' is not a valid " + gguf_type_name((enum gguf_type)type) + " value";
    }
    return ok;
}

// ggml lays a new file out with its default alignment whatever general.alignment
// says, and aborts if the key is not a u32, so the key is never written: a file
// without it is read with that same default.
static bool gt_gguf_check_alignment(size_t alignment, int type, const char * v, std::string & err) {
    unsigned long long u = 0;
    const bool integral = type != GGUF_TYPE_FLOAT32 && type != GGUF_TYPE_FLOAT64 && type != GGUF_TYPE_BOOL &&
                          type != GGUF_TYPE_STRING && type != GGUF_TYPE_ARRAY;
    if (integral && gt_gguf_parse_u64(v, UINT32_MAX, u) && u == (unsigned long long)alignment) return true;
    err = std::string("metadata key '") + GT_GGUF_ALIGNMENT_KEY + "' can only be " + std::to_string(alignment) +
          ", the alignment files are written with";
    return false;
}

extern "C" int gt_gguf_save_ex(const char * path, int n_tensors, const char ** names, const gt_tensor ** tensors,
                               int n_kv, const char ** kv_keys, const int * kv_types, const char ** kv_vals,
                               const int * kv_arr_types, const int64_t * kv_arr_n, const void * const * kv_arr_data,
                               const char * copy_kv_from) {
    if (!path) { gt_set_error("gguf save: no path given"); return -1; }
    if (n_tensors < 0 || n_kv < 0 || (n_tensors > 0 && (!names || !tensors)) || (n_kv > 0 && !kv_keys)) {
        gt_set_error("gguf save: invalid arguments");
        return -1;
    }

    struct gguf_context * gctx = gguf_init_empty();
    if (!gctx) { gt_set_error("gguf save: gguf_init_empty failed"); return -1; }
    const size_t alignment = gguf_get_alignment(gctx);
    struct ggml_context * mctx = nullptr;
    std::vector<std::unique_ptr<gt_contig_owner>> owners;
    std::vector<std::unique_ptr<gt_host_view>> views;
    std::string err;

    // 1. Metadata copied verbatim (types, arrays and all) from another file.
    if (copy_kv_from) {
        struct gguf_init_params sp = { true, nullptr };
        struct gguf_context * src = gguf_init_from_file(copy_kv_from, sp);
        if (!src) {
            err = std::string("could not read metadata from '") + copy_kv_from + "'";
        } else {
            gguf_set_kv(gctx, src);
            gguf_free(src);
            // What described the other file's layout and tensors does not describe this one's.
            std::vector<std::string> drop;
            const int64_t n = gguf_get_n_kv(gctx);
            for (int64_t i = 0; i < n; ++i) {
                const char * k = gguf_get_key(gctx, i);
                if (gt_gguf_key_is_reserved(k) || std::strcmp(k, GT_GGUF_ALIGNMENT_KEY) == 0) drop.emplace_back(k);
            }
            for (const std::string & k : drop) gguf_remove_key(gctx, k.c_str());
        }
    }

    // 2. Metadata given by the caller.
    for (int i = 0; err.empty() && i < n_kv; ++i) {
        const char * k = kv_keys[i];
        if (!k) { err = "metadata entry " + std::to_string(i) + " has no key"; break; }
        const int kt = kv_types ? kv_types[i] : (int)GGUF_TYPE_STRING;
        if (gt_gguf_key_is_reserved(k)) {
            err = std::string("metadata key '") + k + "' is reserved: gguf_train.shape.* and gguf_train.dtype.* describe how tensors are stored";
            break;
        }
        if (kt == GGUF_TYPE_ARRAY) {
            const int et = kv_arr_types ? kv_arr_types[i] : -1;
            const int64_t n = kv_arr_n ? kv_arr_n[i] : -1;
            const void * data = kv_arr_data ? kv_arr_data[i] : nullptr;
            if (et < 0 || et >= GGUF_TYPE_COUNT || et == GGUF_TYPE_ARRAY || n < 0 || (n > 0 && !data)) {
                err = std::string("metadata key '") + k + "': invalid array description";
                break;
            }
            if (std::strcmp(k, GT_GGUF_ALIGNMENT_KEY) == 0) { gt_gguf_check_alignment(alignment, kt, "", err); break; }
            if (et == GGUF_TYPE_STRING) {
                const char ** strs = (const char **)data;
                for (int64_t j = 0; j < n; ++j) {
                    if (!strs[j]) { err = std::string("metadata key '") + k + "': array element " + std::to_string(j) + " is null"; break; }
                }
                if (!err.empty()) break;
                gguf_set_arr_str(gctx, k, strs, (size_t)n);
            } else {
                gguf_set_arr_data(gctx, k, (enum gguf_type)et, data, (size_t)n);
            }
            continue;
        }
        const char * v = kv_vals ? kv_vals[i] : nullptr;
        if (!v) { err = std::string("metadata key '") + k + "' has no value"; break; }
        if (std::strcmp(k, GT_GGUF_ALIGNMENT_KEY) == 0) {
            if (!gt_gguf_check_alignment(alignment, kt, v, err)) break;
            continue;
        }
        if (!gt_gguf_set_scalar(gctx, k, kt, v, err)) break;
    }

    // 3. Tensors, and what GGUF cannot say about them.
    if (err.empty()) {
        struct ggml_init_params p = { ggml_tensor_overhead() * (size_t)(n_tensors + 4), nullptr, true };
        mctx = ggml_init(p);
        if (!mctx) err = "ggml_init failed";
    }
    for (int i = 0; err.empty() && i < n_tensors; ++i) {
        const char * name = names[i];
        const gt_tensor * t = tensors[i];
        if (!name || !t) { err = "tensor " + std::to_string(i) + " is missing or has no name"; break; }
        const std::string label = std::string("tensor '") + name + "'";
        if (std::strlen(name) >= GGML_MAX_NAME) {
            err = label + ": the name is " + std::to_string(std::strlen(name)) + " bytes long, GGUF allows at most " +
                  std::to_string(GGML_MAX_NAME - 1);
            break;
        }
        if (gguf_find_tensor(gctx, name) != -1) { err = label + " appears more than once"; break; }
        if (t->device_type == GT_DEV_META) { err = label + " has no data (meta device)"; break; }

        gt_gguf_layout lay;
        std::string why;
        if (!gt_gguf_plan(t, lay, why)) { err = label + ": " + why; break; }

        owners.emplace_back(new gt_contig_owner(t));
        const gt_tensor * src = owners.back()->ptr;
        if (!src) { err = label + ": could not make a contiguous copy"; break; }
        views.emplace_back(new gt_host_view());
        const void * data = views.back()->read(src);

        struct ggml_tensor * gm = ggml_new_tensor(mctx, lay.type, GGML_MAX_DIMS, lay.ne);
        ggml_set_name(gm, name);
        static const char no_data = 0;
        gm->data = data ? const_cast<void *>(data) : const_cast<void *>((const void *)&no_data);
        if (!data && ggml_nbytes(gm) > 0) { err = label + ": could not read the data"; break; }
        gguf_add_tensor(gctx, gm);

        const std::string shape_key = std::string(GT_GGUF_SHAPE_PREFIX) + name;
        const std::string dtype_key = std::string(GT_GGUF_DTYPE_PREFIX) + name;
        if (lay.needs_shape) gguf_set_arr_data(gctx, shape_key.c_str(), GGUF_TYPE_INT64, t->shape, (size_t)t->ndim);
        if (lay.dtype_name) gguf_set_val_str(gctx, dtype_key.c_str(), lay.dtype_name);
    }

    if (err.empty() && !gguf_write_to_file(gctx, path, false)) {
        err = std::string("could not write '") + path + "'";
    }

    views.clear();
    owners.clear();
    if (mctx) ggml_free(mctx);
    gguf_free(gctx);
    if (!err.empty()) {
        gt_set_error("gguf save: " + err);
        return -1;
    }
    return 0;
}

extern "C" int gt_gguf_save(const char * path, int n_tensors, const char ** names, const gt_tensor ** tensors,
                            int n_kv, const char ** kv_keys, const int * kv_types, const char ** kv_vals) {
    return gt_gguf_save_ex(path, n_tensors, names, tensors, n_kv, kv_keys, kv_types, kv_vals, nullptr, nullptr, nullptr, nullptr);
}

struct gt_gguf_file {
    struct gguf_context * gctx = nullptr;
    struct ggml_context * mctx = nullptr;
};

extern "C" gt_gguf_file * gt_gguf_open(const char * path) {
    if (!path) return nullptr;
    auto * gf = new gt_gguf_file();
    struct gguf_init_params params = { false, &gf->mctx };
    gf->gctx = gguf_init_from_file(path, params);
    if (!gf->gctx) {
        if (gf->mctx) ggml_free(gf->mctx);
        delete gf;
        gt_set_error("Failed to open GGUF file");
        return nullptr;
    }
    return gf;
}

extern "C" void gt_gguf_close(gt_gguf_file * gf) {
    if (!gf) return;
    if (gf->mctx) ggml_free(gf->mctx);
    if (gf->gctx) gguf_free(gf->gctx);
    delete gf;
}

extern "C" int64_t gt_gguf_n_tensors(const gt_gguf_file * gf) {
    return gf && gf->gctx ? gguf_get_n_tensors(gf->gctx) : 0;
}

extern "C" const char * gt_gguf_tensor_name(const gt_gguf_file * gf, int64_t idx) {
    if (!gf || !gf->gctx || idx < 0 || idx >= gguf_get_n_tensors(gf->gctx)) return nullptr;
    return gguf_get_tensor_name(gf->gctx, idx);
}

extern "C" gt_tensor * gt_gguf_get_tensor(const gt_gguf_file * gf, const char * name, int dequantize_f32) {
    if (!gf || !gf->gctx || !gf->mctx || !name) return nullptr;
    struct ggml_tensor * gm = ggml_get_tensor(gf->mctx, name);
    if (!gm) {
        gt_set_error(std::string("gguf: no tensor named '") + name + "'");
        return nullptr;
    }
    const std::string label = std::string("gguf: tensor '") + name + "': ";
    int dt = gt_ggml_to_dtype(gm->type);

    // The dtype GGUF could not name.
    bool cplx = false;
    const int64_t dtype_id = gguf_find_key(gf->gctx, (std::string(GT_GGUF_DTYPE_PREFIX) + name).c_str());
    if (dtype_id >= 0) {
        if (gguf_get_kv_type(gf->gctx, dtype_id) != GGUF_TYPE_STRING) {
            gt_set_error(label + "the recorded dtype is not a string");
            return nullptr;
        }
        const std::string want = gguf_get_val_str(gf->gctx, dtype_id);
        enum ggml_type carrier = GGML_TYPE_COUNT;
        if (want == "uint8")           { carrier = GGML_TYPE_I8;  dt = GT_U8; }
        else if (want == "bool")       { carrier = GGML_TYPE_I8;  dt = GT_BOOL; }
        else if (want == "complex64")  { carrier = GGML_TYPE_F32; dt = GT_C64;  cplx = true; }
        else if (want == "complex128") { carrier = GGML_TYPE_F64; dt = GT_C128; cplx = true; }
        if (carrier != gm->type) {
            gt_set_error(label + "recorded as " + want + " but stored as " + ggml_type_name(gm->type));
            return nullptr;
        }
    }

    const int64_t count = ggml_nelements(gm);
    if (cplx && count % 2 != 0) {
        gt_set_error(label + "a complex tensor needs an even number of stored values");
        return nullptr;
    }
    const int64_t numel = cplx ? count / 2 : count;

    // The shape: what GGUF kept, unless the real one was recorded.
    std::vector<int64_t> shape = gt_gguf_natural_shape(gm->ne, cplx);
    const int64_t shape_id = gguf_find_key(gf->gctx, (std::string(GT_GGUF_SHAPE_PREFIX) + name).c_str());
    if (shape_id >= 0) {
        if (gguf_get_kv_type(gf->gctx, shape_id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gf->gctx, shape_id) != GGUF_TYPE_INT64) {
            gt_set_error(label + "the recorded shape is not an int64 array");
            return nullptr;
        }
        const size_t n = gguf_get_arr_n(gf->gctx, shape_id);
        const int64_t * dims = (const int64_t *)gguf_get_arr_data(gf->gctx, shape_id);
        if (n > GT_MAX_DIMS) {
            gt_set_error(label + "the recorded shape has too many dimensions");
            return nullptr;
        }
        shape.assign(dims, dims + n);
    }
    int64_t expect = 1;
    bool valid = true;
    for (int64_t d : shape) {
        if (d < 0 || (d > 0 && expect > INT64_MAX / d)) { valid = false; break; }
        expect *= d;
    }
    if (!valid || expect != numel) {
        gt_set_error(label + "the shape does not match the " + std::to_string(numel) + " stored elements");
        return nullptr;
    }

    gt_tensor * raw = gt_tensor_from_data((int)shape.size(), shape.data(), dt, GT_DEV_CPU, 0, gm->data, ggml_nbytes(gm));
    if (!raw) return nullptr;
    if (dequantize_f32 && gt_is_quantized(dt)) {
        gt_tensor * f32 = gt_tensor_to_dtype(raw, GT_F32);
        gt_tensor_free(raw);
        return f32;
    }
    return raw;
}

extern "C" int64_t gt_gguf_n_kv(const gt_gguf_file * gf) {
    return gf && gf->gctx ? gguf_get_n_kv(gf->gctx) : 0;
}

static bool gt_gguf_kv_valid(const gt_gguf_file * gf, int64_t idx) {
    return gf && gf->gctx && idx >= 0 && idx < gguf_get_n_kv(gf->gctx);
}

extern "C" const char * gt_gguf_kv_key(const gt_gguf_file * gf, int64_t idx) {
    return gt_gguf_kv_valid(gf, idx) ? gguf_get_key(gf->gctx, idx) : nullptr;
}

extern "C" int gt_gguf_kv_type(const gt_gguf_file * gf, int64_t idx) {
    return gt_gguf_kv_valid(gf, idx) ? (int)gguf_get_kv_type(gf->gctx, idx) : -1;
}

// Writes the value as text and returns its full length, like snprintf: a result
// >= buf_len means the text was cut short. Pass a null buffer to ask for the
// length alone.
extern "C" int gt_gguf_kv_val_str(const gt_gguf_file * gf, int64_t idx, char * buf, int buf_len) {
    if (!gt_gguf_kv_valid(gf, idx)) return -1;
    char * out = (buf && buf_len > 0) ? buf : nullptr;
    const size_t cap = out ? (size_t)buf_len : 0;
    const struct gguf_context * g = gf->gctx;
    int n = 0;
    switch (gguf_get_kv_type(g, idx)) {
        case GGUF_TYPE_STRING: {
            const char * s = gguf_get_val_str(g, idx);
            const size_t len = std::strlen(s);
            if (out) {
                const size_t ncopy = std::min(len, cap - 1);
                std::memcpy(out, s, ncopy);
                out[ncopy] = '\0';
            }
            n = len > (size_t)INT_MAX ? INT_MAX : (int)len;
        } break;
        case GGUF_TYPE_UINT8:   n = std::snprintf(out, cap, "%u", (unsigned)gguf_get_val_u8(g, idx)); break;
        case GGUF_TYPE_INT8:    n = std::snprintf(out, cap, "%d", (int)gguf_get_val_i8(g, idx)); break;
        case GGUF_TYPE_UINT16:  n = std::snprintf(out, cap, "%u", (unsigned)gguf_get_val_u16(g, idx)); break;
        case GGUF_TYPE_INT16:   n = std::snprintf(out, cap, "%d", (int)gguf_get_val_i16(g, idx)); break;
        case GGUF_TYPE_UINT32:  n = std::snprintf(out, cap, "%u", gguf_get_val_u32(g, idx)); break;
        case GGUF_TYPE_INT32:   n = std::snprintf(out, cap, "%d", gguf_get_val_i32(g, idx)); break;
        // Seventeen significant digits read back as exactly the same double (a float32 is
        // reported as the double it widens to, as its array elements are).
        case GGUF_TYPE_FLOAT32: n = std::snprintf(out, cap, "%.17g", (double)gguf_get_val_f32(g, idx)); break;
        case GGUF_TYPE_UINT64:  n = std::snprintf(out, cap, "%llu", (unsigned long long)gguf_get_val_u64(g, idx)); break;
        case GGUF_TYPE_INT64:   n = std::snprintf(out, cap, "%lld", (long long)gguf_get_val_i64(g, idx)); break;
        case GGUF_TYPE_FLOAT64: n = std::snprintf(out, cap, "%.17g", gguf_get_val_f64(g, idx)); break;
        case GGUF_TYPE_BOOL:    n = std::snprintf(out, cap, "%s", gguf_get_val_bool(g, idx) ? "true" : "false"); break;
        case GGUF_TYPE_ARRAY:   n = std::snprintf(out, cap, "<array len=%zu>", gguf_get_arr_n(g, idx)); break;
        default:
            if (out) out[0] = '\0';
            n = 0;
            break;
    }
    return n;
}

// Arrays: element type (a gguf_type, never GGUF_TYPE_ARRAY) or -1 when the value is not an array.
extern "C" int gt_gguf_kv_arr_type(const gt_gguf_file * gf, int64_t idx) {
    if (!gt_gguf_kv_valid(gf, idx) || gguf_get_kv_type(gf->gctx, idx) != GGUF_TYPE_ARRAY) return -1;
    return (int)gguf_get_arr_type(gf->gctx, idx);
}

extern "C" int64_t gt_gguf_kv_arr_n(const gt_gguf_file * gf, int64_t idx) {
    if (gt_gguf_kv_arr_type(gf, idx) < 0) return -1;
    return (int64_t)gguf_get_arr_n(gf->gctx, idx);
}

// Packed elements of a numeric or bool (one byte each) array; null for string arrays.
extern "C" const void * gt_gguf_kv_arr_data(const gt_gguf_file * gf, int64_t idx) {
    const int et = gt_gguf_kv_arr_type(gf, idx);
    if (et < 0 || et == GGUF_TYPE_STRING) return nullptr;
    return gguf_get_arr_data(gf->gctx, idx);
}

extern "C" const char * gt_gguf_kv_arr_str(const gt_gguf_file * gf, int64_t idx, int64_t i) {
    if (gt_gguf_kv_arr_type(gf, idx) != GGUF_TYPE_STRING) return nullptr;
    if (i < 0 || (size_t)i >= gguf_get_arr_n(gf->gctx, idx)) return nullptr;
    return gguf_get_arr_str(gf->gctx, idx, (size_t)i);
}

extern "C" gt_tensor * gt_quantize(const gt_tensor * a, int quant_dtype) {
    return a ? gt_tensor_to_dtype(a, quant_dtype) : nullptr;
}

extern "C" gt_tensor * gt_dequantize(const gt_tensor * a, int target_dtype) {
    return a ? gt_tensor_to_dtype(a, target_dtype) : nullptr;
}

// ---------- Native ggml_opt Static Graph Builder & Trainer ----------
//
// The builder functions (gt_opt_graph_input/param/linear/...) only *record* the graph: one
// descriptor per node holding its operation, its operands and its inferred shape. Nothing exists
// in ggml until gt_opt_graph_fit(), which builds the ggml graph from the descriptors into contexts
// that live exactly as long as that one call.
//
// It has to be this way round. ggml_opt_init() appends the backward pass, the optimizer steps and
// three graph objects to the compute context it is handed. This code used to build the forward
// graph once into a long-lived 8 MB context and pass that to every fit(), so each call left
// ~450 kB behind and the 19th one ran off the end of the pool. Recording the graph also lets every
// node id, shape and argument be checked here, where the caller gets an error it can handle,
// rather than by a GGML_ASSERT inside ggml, which aborts the process.

namespace {

enum gt_og_kind {
    GT_OG_INPUT,
    GT_OG_PARAM,
    GT_OG_LINEAR,
    GT_OG_UNARY,
    GT_OG_ADD,
    GT_OG_MUL,
    GT_OG_RMS_NORM,
    GT_OG_SOFTMAX,
};

struct gt_og_node {
    gt_og_kind kind = GT_OG_INPUT;
    int     src[3] = {-1, -1, -1}; // operand node ids, -1 = unused
    int     unary_op = 0;          // GT_OG_UNARY: a GT_UNARY_* value
    float   eps = 0.0f;            // GT_OG_RMS_NORM
    int     param = -1;            // GT_OG_PARAM: index into gt_opt_graph::params
    int     nd = 1;                // number of dimensions as the caller sees them
    int64_t ne[4] = {1, 1, 1, 1};  // shape in ggml order (innermost dimension first)
    int     n_fwd = 0;             // ggml tensors the node expands to
    int     n_bwd = 0;             // upper bound on the tensors ggml creates for its backward pass
};

// ggml_opt keeps the forward pass, the backward pass and the optimizer steps in one graph of
// GGML_DEFAULT_GRAPH_SIZE entries, and copies every node and leaf of it into a pool sized for that
// many tensors (ggml_opt_alloc). Exceeding either is a GGML_ASSERT, so the size is bounded here.
// The n_bwd values are worst cases read off ggml_compute_backward(); GT_OG_LOSS_TENSORS covers the
// loss, its gradient and the prediction/accuracy nodes; a parameter costs five tensors (itself,
// its optimizer step, a gradient accumulator and two AdamW moments).
const int GT_OG_LOSS_TENSORS  = 64;
const int GT_OG_PARAM_TENSORS = 5;

std::string gt_og_shape_str(const gt_og_node & n) {
    std::string s = "(";
    for (int d = n.nd - 1; d >= 0; --d) {
        s += std::to_string(n.ne[d]);
        if (d > 0) s += ", ";
    }
    if (n.nd == 1) s += ",";
    return s + ")";
}

// ggml broadcasts in one direction only: `small` is repeated to the shape of `big`.
bool gt_og_can_repeat(const gt_og_node & small, const gt_og_node & big) {
    for (int d = 0; d < 4; ++d) {
        if (big.ne[d] % small.ne[d] != 0) return false;
    }
    return true;
}

// Unary operations that ggml can differentiate. sigmoid and tanh have a forward op in ggml but no
// backward pass (ggml_compute_backward aborts on them), so they are expanded into ops that do.
bool gt_og_unary_cost(int op, int & n_fwd, int & n_bwd) {
    switch (op) {
        case GT_UNARY_RELU:     n_fwd = 1; n_bwd = 3;  return true;
        case GT_UNARY_SILU:     n_fwd = 1; n_bwd = 2;  return true;
        case GT_UNARY_EXP:      n_fwd = 1; n_bwd = 2;  return true;
        case GT_UNARY_LOG:      n_fwd = 1; n_bwd = 2;  return true;
        case GT_UNARY_SQUARE:   n_fwd = 1; n_bwd = 3;  return true;
        case GT_UNARY_SQRT:     n_fwd = 1; n_bwd = 3;  return true;
        case GT_UNARY_NEG:      n_fwd = 1; n_bwd = 1;  return true;
        case GT_UNARY_ABS:      n_fwd = 1; n_bwd = 3;  return true;
        case GT_UNARY_SOFTPLUS: n_fwd = 1; n_bwd = 3;  return true;
        case GT_UNARY_SIGMOID:  n_fwd = 4; n_bwd = 7;  return true;
        case GT_UNARY_TANH:     n_fwd = 6; n_bwd = 11; return true;
        default:                return false;
    }
}

// sigmoid(x) = exp(-softplus(-x)); exact, and well behaved for large |x|.
struct ggml_tensor * gt_og_sigmoid(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_exp(ctx, ggml_neg(ctx, ggml_softplus(ctx, ggml_neg(ctx, x))));
}

struct ggml_tensor * gt_og_emit_unary(struct ggml_context * ctx, int op, struct ggml_tensor * x) {
    switch (op) {
        case GT_UNARY_RELU:     return ggml_relu(ctx, x);
        case GT_UNARY_SILU:     return ggml_silu(ctx, x);
        case GT_UNARY_EXP:      return ggml_exp(ctx, x);
        case GT_UNARY_LOG:      return ggml_log(ctx, x);
        case GT_UNARY_SQUARE:   return ggml_sqr(ctx, x);
        case GT_UNARY_SQRT:     return ggml_sqrt(ctx, x);
        case GT_UNARY_NEG:      return ggml_neg(ctx, x);
        case GT_UNARY_ABS:      return ggml_abs(ctx, x);
        case GT_UNARY_SOFTPLUS: return ggml_softplus(ctx, x);
        case GT_UNARY_SIGMOID:  return gt_og_sigmoid(ctx, x);
        // tanh(x) = 2*sigmoid(2x) - 1
        case GT_UNARY_TANH:     return ggml_scale_bias(ctx, gt_og_sigmoid(ctx, ggml_scale(ctx, x, 2.0f)), 2.0f, -1.0f);
        default:                return nullptr;
    }
}

struct gt_og_hparams {
    float lr = 1e-3f;
    float wd = 0.0f;
};

struct ggml_opt_optimizer_params gt_og_get_opt_pars(void * userdata) {
    const gt_og_hparams * hp = (const gt_og_hparams *)userdata;
    struct ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = hp->lr;
    p.adamw.wd    = hp->wd;
    p.sgd.alpha   = hp->lr;
    p.sgd.wd      = hp->wd;
    return p;
}

// Everything one fit() allocates in ggml, released whichever way the call ends.
struct gt_og_run {
    struct ggml_context * ctx_static  = nullptr; // input and parameters
    struct ggml_context * ctx_compute = nullptr; // forward graph, plus whatever ggml_opt adds to it
    ggml_backend_buffer_t buf_static  = nullptr;
    ggml_backend_sched_t  sched       = nullptr;
    ggml_opt_dataset_t    dataset     = nullptr;
    ggml_opt_context_t    opt         = nullptr;
    ggml_opt_result_t     res_train   = nullptr;
    ggml_opt_result_t     res_eval    = nullptr;

    ~gt_og_run() {
        if (res_train)   ggml_opt_result_free(res_train);
        if (res_eval)    ggml_opt_result_free(res_eval);
        if (opt)         ggml_opt_free(opt);
        if (dataset)     ggml_opt_dataset_free(dataset);
        if (sched)       ggml_backend_sched_free(sched);
        if (buf_static)  ggml_backend_buffer_free(buf_static);
        if (ctx_compute) ggml_free(ctx_compute);
        if (ctx_static)  ggml_free(ctx_static);
    }
};

int gt_og_fail(const std::string & msg) {
    gt_set_error(msg);
    return -1;
}

} // namespace

struct gt_opt_graph {
    int loss_type = 3;      // 0=MEAN, 1=SUM, 2=CROSS_ENTROPY, 3=MSE
    int optimizer_type = 0; // 0=ADAMW, 1=SGD
    int64_t nbatch_physical = 32;
    int64_t nbatch_logical = 32;
    std::vector<gt_og_node> nodes;
    int input_id = -1;
    // Handles of our own onto the callers' parameter storage: the caller may free its tensor
    // handle before fit() without leaving this graph with a dangling pointer.
    std::vector<gt_tensor *> params;
};

namespace {

bool gt_og_valid_id(const gt_opt_graph * g, int id) {
    return id >= 0 && (size_t)id < g->nodes.size();
}

// Returns false (with the error set) unless every id names an existing node. `optional` ids may be -1.
bool gt_og_check_ids(const gt_opt_graph * g, const char * what, std::initializer_list<int> required,
                     std::initializer_list<int> optional = {}) {
    if (!g) {
        gt_set_error(std::string(what) + ": graph is NULL");
        return false;
    }
    for (int id : required) {
        if (!gt_og_valid_id(g, id)) {
            gt_set_error(std::string(what) + ": node id " + std::to_string(id) + " does not exist (the graph has " +
                         std::to_string(g->nodes.size()) + " nodes)");
            return false;
        }
    }
    for (int id : optional) {
        if (id != -1 && !gt_og_valid_id(g, id)) {
            gt_set_error(std::string(what) + ": node id " + std::to_string(id) + " does not exist (the graph has " +
                         std::to_string(g->nodes.size()) + " nodes; pass -1 to leave it out)");
            return false;
        }
    }
    return true;
}

int gt_og_push(gt_opt_graph * g, const gt_og_node & n) {
    g->nodes.push_back(n);
    return (int)g->nodes.size() - 1;
}

void gt_og_copy_shape(gt_og_node & dst, const gt_og_node & src) {
    dst.nd = src.nd;
    for (int d = 0; d < 4; ++d) dst.ne[d] = src.ne[d];
}

// add and mul commute, but ggml only repeats the second operand; put the smaller one second.
int gt_og_binary(gt_opt_graph * g, const char * what, gt_og_kind kind, int a_id, int b_id, int n_bwd) {
    if (!gt_og_check_ids(g, what, {a_id, b_id})) return -1;
    const gt_og_node * a = &g->nodes[a_id];
    const gt_og_node * b = &g->nodes[b_id];
    if (!gt_og_can_repeat(*b, *a)) {
        if (!gt_og_can_repeat(*a, *b)) {
            return gt_og_fail(std::string(what) + ": shapes " + gt_og_shape_str(*a) + " and " + gt_og_shape_str(*b) +
                              " cannot be broadcast (one must repeat evenly into the other)");
        }
        std::swap(a_id, b_id);
        std::swap(a, b);
    }
    gt_og_node n;
    n.kind = kind;
    n.src[0] = a_id;
    n.src[1] = b_id;
    gt_og_copy_shape(n, *a);
    n.n_fwd = 1;
    n.n_bwd = n_bwd;
    return gt_og_push(g, n);
}

} // namespace

extern "C" gt_opt_graph * gt_opt_graph_create(int loss_type, int optimizer_type, int64_t nbatch_physical, int64_t nbatch_logical) {
    if (loss_type < 0 || loss_type > 3) {
        gt_set_error("opt_graph_create: loss_type must be 0 (mean), 1 (sum), 2 (cross_entropy) or 3 (mse)");
        return nullptr;
    }
    if (optimizer_type < 0 || optimizer_type > 1) {
        gt_set_error("opt_graph_create: optimizer must be 0 (adamw) or 1 (sgd)");
        return nullptr;
    }
    gt_init();
    auto * g = new gt_opt_graph();
    g->loss_type = loss_type;
    g->optimizer_type = optimizer_type;
    g->nbatch_physical = std::max<int64_t>(1, nbatch_physical);
    g->nbatch_logical  = std::max<int64_t>(g->nbatch_physical, nbatch_logical);
    return g;
}

extern "C" void gt_opt_graph_free(gt_opt_graph * g) {
    if (!g) return;
    for (gt_tensor * p : g->params) gt_tensor_free(p);
    delete g;
}

extern "C" int gt_opt_graph_input(gt_opt_graph * g, int ndim, const int64_t * shape) {
    if (!g) return gt_og_fail("opt_graph_input: graph is NULL");
    if (g->input_id >= 0) return gt_og_fail("opt_graph_input: the graph already has an input; only one is supported");
    if (ndim < 1 || ndim > 4 || !shape) return gt_og_fail("opt_graph_input: the input must have 1 to 4 dimensions");
    gt_og_node n;
    n.kind = GT_OG_INPUT;
    n.nd = ndim;
    for (int d = 0; d < ndim; ++d) {
        if (shape[ndim - 1 - d] < 1) return gt_og_fail("opt_graph_input: every dimension of the input must be at least 1");
        n.ne[d] = shape[ndim - 1 - d];
    }
    g->input_id = gt_og_push(g, n);
    return g->input_id;
}

extern "C" int gt_opt_graph_param(gt_opt_graph * g, gt_tensor * param) {
    if (!g || !param) return gt_og_fail("opt_graph_param: graph or tensor is NULL");
    if (param->dtype != GT_F32) return gt_og_fail("opt_graph_param: parameters must be float32 tensors");
    if (param->ndim > 4) return gt_og_fail("opt_graph_param: parameters can have at most 4 dimensions");
    if (param->numel < 1 || !param->storage || !param->storage->raw_tensor) {
        return gt_og_fail("opt_graph_param: parameters must not be empty");
    }

    // The same tensor registered twice is one parameter (weight sharing), not two that would be
    // trained separately and then written back over each other.
    for (size_t id = 0; id < g->nodes.size(); ++id) {
        const gt_og_node & other = g->nodes[id];
        if (other.kind != GT_OG_PARAM) continue;
        const gt_tensor * q = g->params[other.param];
        bool same = q->storage == param->storage && q->offset == param->offset && q->ndim == param->ndim;
        for (int d = 0; same && d < param->ndim; ++d) {
            same = q->shape[d] == param->shape[d] && q->strides[d] == param->strides[d];
        }
        if (same) return (int)id;
    }

    gt_tensor * alias = gt_tensor_view(param, param->ndim, param->shape, param->strides, param->offset);
    if (!alias) return -1;

    gt_og_node n;
    n.kind = GT_OG_PARAM;
    n.nd = std::max(1, param->ndim);
    for (int d = 0; d < param->ndim; ++d) n.ne[d] = param->shape[param->ndim - 1 - d];
    n.param = (int)g->params.size();
    g->params.push_back(alias);
    return gt_og_push(g, n);
}

extern "C" int gt_opt_graph_linear(gt_opt_graph * g, int x_id, int w_id, int b_id) {
    if (!gt_og_check_ids(g, "opt_graph_linear", {x_id, w_id}, {b_id})) return -1;
    const gt_og_node & x = g->nodes[x_id];
    const gt_og_node & w = g->nodes[w_id];
    if (w.nd != 2) {
        return gt_og_fail("opt_graph_linear: the weight must be 2-D (out_features, in_features), got " + gt_og_shape_str(w));
    }
    if (w.ne[0] != x.ne[0]) {
        return gt_og_fail("opt_graph_linear: weight " + gt_og_shape_str(w) + " expects " + std::to_string(w.ne[0]) +
                          " input features but the input " + gt_og_shape_str(x) + " has " + std::to_string(x.ne[0]));
    }
    gt_og_node n;
    n.kind = GT_OG_LINEAR;
    n.src[0] = x_id;
    n.src[1] = w_id;
    n.src[2] = b_id;
    n.nd = x.nd;
    n.ne[0] = w.ne[1];
    for (int d = 1; d < 4; ++d) n.ne[d] = x.ne[d];
    n.n_fwd = 1;
    n.n_bwd = 7;
    if (b_id >= 0) {
        const gt_og_node & b = g->nodes[b_id];
        if (!gt_og_can_repeat(b, n)) {
            return gt_og_fail("opt_graph_linear: bias " + gt_og_shape_str(b) + " cannot be broadcast to the output " + gt_og_shape_str(n));
        }
        n.n_fwd += 1;
        n.n_bwd += 3;
    }
    return gt_og_push(g, n);
}

extern "C" int gt_opt_graph_unary(gt_opt_graph * g, int x_id, int unary_op) {
    if (!gt_og_check_ids(g, "opt_graph_unary", {x_id})) return -1;
    gt_og_node n;
    n.kind = GT_OG_UNARY;
    n.src[0] = x_id;
    n.unary_op = unary_op;
    gt_og_copy_shape(n, g->nodes[x_id]);
    if (!gt_og_unary_cost(unary_op, n.n_fwd, n.n_bwd)) {
        return gt_og_fail("opt_graph_unary: ggml has no backward pass for this operation (supported: relu, silu, sigmoid, tanh, "
                          "softplus, exp, log, square, sqrt, neg, abs)");
    }
    return gt_og_push(g, n);
}

extern "C" int gt_opt_graph_add(gt_opt_graph * g, int a_id, int b_id) {
    return gt_og_binary(g, "opt_graph_add", GT_OG_ADD, a_id, b_id, 3);
}

extern "C" int gt_opt_graph_mul(gt_opt_graph * g, int a_id, int b_id) {
    return gt_og_binary(g, "opt_graph_mul", GT_OG_MUL, a_id, b_id, 5);
}

extern "C" int gt_opt_graph_rms_norm(gt_opt_graph * g, int x_id, int w_id, float eps) {
    if (!gt_og_check_ids(g, "opt_graph_rms_norm", {x_id}, {w_id})) return -1;
    if (!(eps >= 0.0f) || !std::isfinite(eps)) return gt_og_fail("opt_graph_rms_norm: eps must be a finite, non-negative number");
    const gt_og_node & x = g->nodes[x_id];
    gt_og_node n;
    n.kind = GT_OG_RMS_NORM;
    n.src[0] = x_id;
    n.src[1] = w_id;
    n.eps = eps;
    gt_og_copy_shape(n, x);
    n.n_fwd = 1;
    n.n_bwd = 2;
    if (w_id >= 0) {
        const gt_og_node & w = g->nodes[w_id];
        if (!gt_og_can_repeat(w, x)) {
            return gt_og_fail("opt_graph_rms_norm: weight " + gt_og_shape_str(w) + " cannot be broadcast to the input " + gt_og_shape_str(x));
        }
        n.n_fwd += 1;
        n.n_bwd += 5;
    }
    return gt_og_push(g, n);
}

extern "C" int gt_opt_graph_softmax(gt_opt_graph * g, int x_id) {
    if (!gt_og_check_ids(g, "opt_graph_softmax", {x_id})) return -1;
    gt_og_node n;
    n.kind = GT_OG_SOFTMAX;
    n.src[0] = x_id;
    gt_og_copy_shape(n, g->nodes[x_id]);
    n.n_fwd = 1;
    n.n_bwd = 2;
    return gt_og_push(g, n);
}

static int gt_opt_graph_fit_impl(gt_opt_graph * g, int out_id, const gt_tensor * data, const gt_tensor * labels,
                                 int64_t epochs, float lr, float wd, float val_split, int silent,
                                 double * out_train_loss, double * out_val_loss) {
    if (out_train_loss) *out_train_loss = 0.0;
    if (out_val_loss)   *out_val_loss   = 0.0;

    if (!gt_og_check_ids(g, "OptGraph.fit", {out_id})) return -1;
    if (g->input_id < 0) return gt_og_fail("OptGraph.fit: the graph has no input; call input() first");
    if (!data) return gt_og_fail("OptGraph.fit: no training data given");
    if (!std::isfinite(lr) || !(lr > 0.0f)) return gt_og_fail("OptGraph.fit: lr must be a positive, finite number");
    if (!(wd >= 0.0f && wd <= 1.0f)) return gt_og_fail("OptGraph.fit: weight decay must be in [0, 1]");
    if (!(val_split >= 0.0f && val_split < 1.0f)) return gt_og_fail("OptGraph.fit: val_split must be in [0, 1)");

    const int64_t nbatch = g->nbatch_physical;
    const gt_og_node & in  = g->nodes[g->input_id];
    const gt_og_node & out = g->nodes[out_id];
    const bool has_labels = g->loss_type == 2 || g->loss_type == 3;
    const char * loss_name = g->loss_type == 0 ? "mean" : g->loss_type == 1 ? "sum" : g->loss_type == 2 ? "cross_entropy" : "mse";

    // ---- which nodes does the output depend on? (operands always have smaller ids) ----
    std::vector<char> used(g->nodes.size(), 0);
    used[out_id] = 1;
    int64_t n_param = 0, n_tensors = 0;
    for (int id = out_id; id >= 0; --id) {
        if (!used[id]) continue;
        const gt_og_node & n = g->nodes[id];
        for (int s : n.src) if (s >= 0) used[s] = 1;
        if (n.kind == GT_OG_PARAM) n_param++;
        n_tensors += n.n_fwd + n.n_bwd;
    }
    if (n_param == 0) {
        return gt_og_fail("OptGraph.fit: the output does not depend on any parameter, so there is nothing to train");
    }
    const int64_t graph_tensors = n_tensors + GT_OG_PARAM_TENSORS * n_param + GT_OG_LOSS_TENSORS;
    if (graph_tensors > GGML_DEFAULT_GRAPH_SIZE) {
        return gt_og_fail("OptGraph.fit: the graph is too large for ggml_opt (about " + std::to_string(graph_tensors) +
                          " tensors including the backward pass; the limit is " + std::to_string(GGML_DEFAULT_GRAPH_SIZE) + ")");
    }

    // ---- shapes: ggml_opt feeds the input one batch of rows at a time ----
    if (in.ne[1] != nbatch || in.ne[2] != 1 || in.ne[3] != 1) {
        return gt_og_fail("OptGraph.fit: the input must have shape (nbatch_physical, features) = (" + std::to_string(nbatch) +
                          ", features), but it was declared as " + gt_og_shape_str(in));
    }
    if (data->ndim < 1 || data->shape[0] < 1 || data->numel < 1) {
        return gt_og_fail("OptGraph.fit: the training data is empty");
    }
    const int64_t n_rows = data->shape[0];
    const int64_t ne_dp  = data->numel / n_rows;
    if (ne_dp != in.ne[0]) {
        return gt_og_fail("OptGraph.fit: each row of the training data has " + std::to_string(ne_dp) +
                          " values but the input " + gt_og_shape_str(in) + " takes " + std::to_string(in.ne[0]) + " features");
    }
    const int64_t n_data = (n_rows / nbatch) * nbatch; // an incomplete last batch is dropped
    if (n_data < nbatch) {
        return gt_og_fail("OptGraph.fit: " + std::to_string(n_rows) + " rows of training data are fewer than one batch (nbatch_physical=" +
                          std::to_string(nbatch) + ")");
    }

    int64_t ne_lbl = 0;
    if (has_labels) {
        if (!labels) return gt_og_fail(std::string("OptGraph.fit: loss_type '") + loss_name + "' needs labels");
        if (out.ne[1] != nbatch || out.ne[2] != 1 || out.ne[3] != 1) {
            return gt_og_fail(std::string("OptGraph.fit: for loss_type '") + loss_name + "' the output must have shape (nbatch_physical, n) = (" +
                              std::to_string(nbatch) + ", n), but it is " + gt_og_shape_str(out));
        }
        if (labels->ndim < 1 || labels->shape[0] != n_rows) {
            return gt_og_fail("OptGraph.fit: the labels must have one row per row of training data (" + std::to_string(n_rows) +
                              "), got " + std::to_string(labels->ndim < 1 ? 0 : labels->shape[0]));
        }
        ne_lbl = labels->numel / n_rows;
        if (ne_lbl != out.ne[0]) {
            std::string msg = "OptGraph.fit: each row of the labels has " + std::to_string(ne_lbl) + " values but the output " +
                              gt_og_shape_str(out) + " has " + std::to_string(out.ne[0]) + " per row";
            if (g->loss_type == 2 && ne_lbl == 1) msg += " (cross_entropy takes one-hot or probability rows, not class indices)";
            return gt_og_fail(msg);
        }
    } else if (labels) {
        return gt_og_fail(std::string("OptGraph.fit: loss_type '") + loss_name + "' takes no labels (the loss is computed from the output alone)");
    }

    if (epochs <= 0) return 0;

    gt_backend_entry * be = gt_get_backend(GT_DEV_CPU, 0);
    if (!be || !be->backend) return gt_og_fail("OptGraph.fit: the CPU backend is not available");

    // ---- host copies of the dataset (float32, contiguous, starting at the tensor's own offset) ----
    gt_contig_owner data_c(data, GT_F32);
    if (!data_c.ptr) return gt_og_fail("OptGraph.fit: could not read the training data as float32");
    gt_host_view data_hv;
    const void * data_ptr = data_hv.read(data_c.ptr);

    gt_contig_owner labels_c(has_labels ? labels : nullptr, GT_F32);
    gt_host_view labels_hv;
    const void * labels_ptr = nullptr;
    if (has_labels) {
        if (!labels_c.ptr) return gt_og_fail("OptGraph.fit: could not read the labels as float32");
        labels_ptr = labels_hv.read(labels_c.ptr);
    }
    if (!data_ptr || (has_labels && !labels_ptr)) return gt_og_fail("OptGraph.fit: the dataset has no accessible storage");

    gt_og_run run;

    // ---- input and parameters ----
    int64_t n_static = 1;
    for (size_t id = 0; id < g->nodes.size(); ++id) n_static += used[id] && g->nodes[id].kind == GT_OG_PARAM;
    run.ctx_static = ggml_init({ (size_t)(n_static + 1) * ggml_tensor_overhead(), nullptr, true });
    if (!run.ctx_static) return gt_og_fail("OptGraph.fit: out of memory");

    std::vector<struct ggml_tensor *> t(g->nodes.size(), nullptr);
    t[g->input_id] = ggml_new_tensor(run.ctx_static, GGML_TYPE_F32, 4, in.ne);
    for (int id = 0; id <= out_id; ++id) {
        const gt_og_node & n = g->nodes[id];
        if (!used[id] || n.kind != GT_OG_PARAM) continue;
        t[id] = ggml_new_tensor(run.ctx_static, GGML_TYPE_F32, 4, n.ne);
        ggml_set_param(t[id]);
    }
    run.buf_static = ggml_backend_alloc_ctx_tensors(run.ctx_static, be->backend);
    if (!run.buf_static) return gt_og_fail("OptGraph.fit: could not allocate the parameters");

    for (int id = 0; id <= out_id; ++id) {
        const gt_og_node & n = g->nodes[id];
        if (!used[id] || n.kind != GT_OG_PARAM) continue;
        gt_contig_owner pc(g->params[n.param], GT_F32);
        gt_host_view hv;
        const void * src = pc.ptr ? hv.read(pc.ptr) : nullptr;
        if (!src) return gt_og_fail("OptGraph.fit: could not read a parameter");
        ggml_backend_tensor_set(t[id], src, 0, ggml_nbytes(t[id]));
    }

    // ---- forward graph, rebuilt for this call only ----
    const size_t graph_bytes = ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true);
    run.ctx_compute = ggml_init({ 3 * graph_bytes + (size_t)(GGML_DEFAULT_GRAPH_SIZE + 64) * ggml_tensor_overhead(), nullptr, true });
    if (!run.ctx_compute) return gt_og_fail("OptGraph.fit: out of memory");
    struct ggml_context * cc = run.ctx_compute;

    for (int id = 0; id <= out_id; ++id) {
        const gt_og_node & n = g->nodes[id];
        if (!used[id] || t[id]) continue;
        struct ggml_tensor * a = n.src[0] >= 0 ? t[n.src[0]] : nullptr;
        struct ggml_tensor * b = n.src[1] >= 0 ? t[n.src[1]] : nullptr;
        struct ggml_tensor * c = n.src[2] >= 0 ? t[n.src[2]] : nullptr;
        struct ggml_tensor * r = nullptr;
        switch (n.kind) {
            case GT_OG_LINEAR:
                r = ggml_mul_mat(cc, b, a);
                if (c) r = ggml_add(cc, r, c);
                break;
            case GT_OG_UNARY:
                r = gt_og_emit_unary(cc, n.unary_op, a);
                break;
            case GT_OG_ADD:
                r = ggml_add(cc, a, b);
                break;
            case GT_OG_MUL:
                r = ggml_mul(cc, a, b);
                break;
            case GT_OG_RMS_NORM:
                r = ggml_rms_norm(cc, a, n.eps);
                if (b) r = ggml_mul(cc, r, b);
                break;
            case GT_OG_SOFTMAX:
                r = ggml_soft_max(cc, a);
                break;
            default:
                break;
        }
        if (!r) return gt_og_fail("OptGraph.fit: internal error while building the graph");
        t[id] = r;
    }

    // ---- dataset ----
    run.dataset = ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, ne_dp, ne_lbl, n_data, nbatch);
    if (!run.dataset || !ggml_opt_dataset_data(run.dataset)->data) return gt_og_fail("OptGraph.fit: could not allocate the dataset");
    std::memcpy(ggml_opt_dataset_data(run.dataset)->data, data_ptr, (size_t)n_data * (size_t)ne_dp * sizeof(float));
    if (has_labels) {
        std::memcpy(ggml_opt_dataset_labels(run.dataset)->data, labels_ptr, (size_t)n_data * (size_t)ne_lbl * sizeof(float));
    }

    enum ggml_opt_loss_type ltype = GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR;
    if (g->loss_type == 0) ltype = GGML_OPT_LOSS_TYPE_MEAN;
    else if (g->loss_type == 1) ltype = GGML_OPT_LOSS_TYPE_SUM;
    else if (g->loss_type == 2) ltype = GGML_OPT_LOSS_TYPE_CROSS_ENTROPY;

    gt_og_hparams hparams;
    hparams.lr = lr;
    hparams.wd = wd;

    const int64_t nbatches = n_data / nbatch;
    int64_t ibatch_split = nbatches;
    if (val_split > 0.0f && nbatches > 1) {
        ibatch_split = (int64_t)((1.0 - (double)val_split) * (double)nbatches);
        ibatch_split = std::min<int64_t>(std::max<int64_t>(ibatch_split, 1), nbatches - 1);
    }
    const int64_t idata_split = ibatch_split * nbatch;

    double last_tr = 0.0, last_ev = 0.0;
    {
        // The CPU backend object is shared with every other tensor operation.
        std::lock_guard<std::mutex> lock(be->mtx);

        ggml_backend_t backends[1] = { be->backend };
        // The scheduler's tables must hold every node *and* leaf of the optimizer graph.
        run.sched = ggml_backend_sched_new(backends, nullptr, 1, 2 * GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (!run.sched) return gt_og_fail("OptGraph.fit: could not create the ggml scheduler");

        struct ggml_opt_params opt_pars = ggml_opt_default_params(run.sched, ltype);
        opt_pars.ctx_compute     = run.ctx_compute;
        opt_pars.inputs          = t[g->input_id];
        opt_pars.outputs         = t[out_id];
        opt_pars.opt_period      = (int32_t)std::max<int64_t>(1, g->nbatch_logical / nbatch);
        opt_pars.get_opt_pars    = gt_og_get_opt_pars;
        opt_pars.get_opt_pars_ud = &hparams;
        opt_pars.optimizer       = g->optimizer_type == 1 ? GGML_OPT_OPTIMIZER_TYPE_SGD : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

        run.opt = ggml_opt_init(opt_pars);
        run.res_train = ggml_opt_result_init();
        run.res_eval  = ggml_opt_result_init();

        ggml_opt_epoch_callback cb = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;
        for (int64_t e = 0; e < epochs; ++e) {
            ggml_opt_dataset_shuffle(run.opt, run.dataset, idata_split);
            ggml_opt_result_reset(run.res_train);
            ggml_opt_result_reset(run.res_eval);
            ggml_opt_epoch(run.opt, run.dataset, run.res_train, run.res_eval, idata_split, cb, cb);
            ggml_opt_result_loss(run.res_train, &last_tr, nullptr);
            if (idata_split < n_data) ggml_opt_result_loss(run.res_eval, &last_ev, nullptr);
        }
    }

    // ---- write the trained values back through each parameter's own view ----
    for (int id = 0; id <= out_id; ++id) {
        const gt_og_node & n = g->nodes[id];
        if (!used[id] || n.kind != GT_OG_PARAM) continue;
        gt_tensor * dst = g->params[n.param];
        gt_tensor * trained = gt_tensor_empty(dst->ndim, dst->shape, GT_F32, GT_DEV_CPU, 0);
        if (!trained) return -1;
        gt_host_view hv;
        void * ptr = hv.write_empty(trained);
        int rc = -1;
        if (ptr) {
            ggml_backend_tensor_get(t[id], ptr, 0, ggml_nbytes(t[id]));
            hv.commit(trained);
            rc = gt_tensor_copy_inplace(dst, trained);
        }
        gt_tensor_free(trained);
        if (rc != 0) return gt_og_fail("OptGraph.fit: could not write a trained parameter back");
    }

    if (out_train_loss) *out_train_loss = last_tr;
    if (out_val_loss)   *out_val_loss   = last_ev;
    return 0;
}

extern "C" int gt_opt_graph_fit(gt_opt_graph * g, int out_id, const gt_tensor * data, const gt_tensor * labels,
                                int64_t epochs, float lr, float wd, float val_split, int silent,
                                double * out_train_loss, double * out_val_loss) {
    try {
        return gt_opt_graph_fit_impl(g, out_id, data, labels, epochs, lr, wd, val_split, silent, out_train_loss, out_val_loss);
    } catch (const std::exception & e) {
        return gt_og_fail(std::string("OptGraph.fit: ") + e.what());
    } catch (...) {
        return gt_og_fail("OptGraph.fit: unknown error");
    }
}
