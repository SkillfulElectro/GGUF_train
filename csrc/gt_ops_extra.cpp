#include "gt_internal.h"
#include "ggml-opt.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <cstdlib>

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

extern "C" gt_tensor * gt_gather(const gt_tensor * a, int dim, const gt_tensor * index) {
    if (!a || !index) return nullptr;
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd || index->ndim != nd) {
        gt_set_error("gather: invalid dim or index ndim");
        return nullptr;
    }
    gt_tensor * out = gt_tensor_empty(nd, index->shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_host_view hva, hvi, hvo;
    const void * ap = hva.read(a);
    const void * ip = hvi.read(index);
    void * op = hvo.write_empty(out);

    for (int64_t l = 0; l < out->numel; ++l) {
        int64_t rem = l;
        int64_t coords[GT_MAX_DIMS];
        for (int d = nd - 1; d >= 0; --d) {
            coords[d] = rem % out->shape[d];
            rem /= out->shape[d];
        }
        int64_t ioff = index->offset;
        for (int d = 0; d < nd; ++d) ioff += coords[d] * index->strides[d];
        int64_t idx = gt_read_i64_at(ip, index->dtype, ioff);
        if (idx < 0) idx += a->shape[dim];
        coords[dim] = std::clamp<int64_t>(idx, 0, std::max<int64_t>(0, a->shape[dim] - 1));
        int64_t aoff = a->offset;
        for (int d = 0; d < nd; ++d) aoff += coords[d] * a->strides[d];
        if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL)
            gt_write_i64_at(op, out->dtype, l, gt_read_i64_at(ap, a->dtype, aoff));
        else
            gt_write_f64_at(op, out->dtype, l, gt_read_f64_at(ap, a->dtype, aoff));
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_scatter_inplace(gt_tensor * dst, int dim, const gt_tensor * index, const gt_tensor * src, int add_mode) {
    if (!dst || !index || !src) return -1;
    int nd = dst->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd || index->ndim != nd || src->ndim != nd) {
        gt_set_error("scatter: dimension mismatch");
        return -1;
    }
    gt_host_view hvd, hvi, hvs;
    void * dp = hvd.read_write(dst);
    const void * ip = hvi.read(index);
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
            soff += coords[d] * src->strides[d];
        }
        int64_t idx = gt_read_i64_at(ip, index->dtype, ioff);
        if (idx < 0) idx += dst->shape[dim];
        if (idx < 0 || idx >= dst->shape[dim]) continue;
        coords[dim] = idx;
        int64_t doff = dst->offset;
        for (int d = 0; d < nd; ++d) doff += coords[d] * dst->strides[d];
        if (gt_is_integer(dst->dtype) || dst->dtype == GT_BOOL) {
            int64_t v = gt_read_i64_at(sp, src->dtype, soff);
            if (add_mode) v += gt_read_i64_at(dp, dst->dtype, doff);
            gt_write_i64_at(dp, dst->dtype, doff, v);
        } else {
            double v = gt_read_f64_at(sp, src->dtype, soff);
            if (add_mode) v += gt_read_f64_at(dp, dst->dtype, doff);
            gt_write_f64_at(dp, dst->dtype, doff, v);
        }
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
    if (!a || !index) return nullptr;
    int nd = a->ndim;
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd || index->ndim > 1) {
        gt_set_error("index_select: invalid dim or index");
        return nullptr;
    }
    int64_t idx_len = (index->ndim == 0) ? 1 : index->shape[0];
    int64_t out_shape[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) out_shape[d] = (d == dim) ? idx_len : a->shape[d];
    gt_tensor * out = gt_tensor_empty(nd, out_shape, a->dtype, a->device_type, a->device_index);
    if (!out) return nullptr;
    if (out->numel == 0 || out->device_type == GT_DEV_META) return out;

    gt_contig_owner ac(a), ic(index);
    gt_host_view hva, hvi, hvo;
    const void * ap = hva.read(ac.ptr);
    const void * ip = hvi.read(ic.ptr);
    void * op = hvo.write_empty(out);

    int64_t outer = 1, inner = 1;
    for (int d = 0; d < dim; ++d) outer *= out_shape[d];
    for (int d = dim + 1; d < nd; ++d) inner *= out_shape[d];

    for (int64_t i = 0; i < idx_len; ++i) {
        int64_t sel = gt_read_i64_at(ip, ic.ptr->dtype, i);
        if (sel < 0) sel += a->shape[dim];
        sel = std::clamp<int64_t>(sel, 0, std::max<int64_t>(0, a->shape[dim] - 1));
        for (int64_t o = 0; o < outer; ++o) {
            int64_t src_base = (o * a->shape[dim] + sel) * inner;
            int64_t dst_base = (o * idx_len + i) * inner;
            for (int64_t j = 0; j < inner; ++j) {
                if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL)
                    gt_write_i64_at(op, out->dtype, dst_base + j, gt_read_i64_at(ap, ac.ptr->dtype, src_base + j));
                else
                    gt_write_f64_at(op, out->dtype, dst_base + j, gt_read_f64_at(ap, ac.ptr->dtype, src_base + j));
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
                double cur = gt_read_f64_at(op, out->dtype, dst_base + j);
                double addv = gt_read_f64_at(gp, gc.ptr->dtype, src_base + j);
                gt_write_f64_at(op, out->dtype, dst_base + j, cur + addv);
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" gt_tensor * gt_index_tensor(const gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims) {
    if (!a || n_idx <= 0 || !idx_tensors || !dims) return nullptr;
    // Broadcast all index tensors together (supports consecutive dims[0..n_idx-1])
    int idx_ndim = 0;
    int64_t idx_shape[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_idx; ++i) idx_ndim = std::max(idx_ndim, idx_tensors[i]->ndim);
    for (int k = 0; k < idx_ndim; ++k) {
        int64_t m = 1;
        for (int i = 0; i < n_idx; ++i) {
            int ii = idx_tensors[i]->ndim - 1 - k;
            int64_t d = (ii >= 0) ? idx_tensors[i]->shape[ii] : 1;
            if (d != m && d != 1 && m != 1) { gt_set_error("index_tensor: incompatible index shapes"); return nullptr; }
            m = std::max(m, d);
        }
        idx_shape[idx_ndim - 1 - k] = m;
    }
    int64_t idx_numel = 1;
    for (int k = 0; k < idx_ndim; ++k) idx_numel *= idx_shape[k];

    // Assume indexed dims are sorted ascending; insert idx_shape at dims[0]
    int first_dim = dims[0];
    bool indexed[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_idx; ++i) indexed[dims[i]] = true;

    int out_ndim = 0;
    int64_t out_shape[GT_MAX_DIMS] = {};
    for (int d = 0; d < first_dim; ++d) if (!indexed[d]) out_shape[out_ndim++] = a->shape[d];
    for (int k = 0; k < idx_ndim; ++k) out_shape[out_ndim++] = idx_shape[k];
    for (int d = first_dim; d < a->ndim; ++d) if (!indexed[d]) out_shape[out_ndim++] = a->shape[d];

    gt_tensor * out = gt_tensor_empty(out_ndim, out_shape, a->dtype, a->device_type, a->device_index);
    if (!out || out->numel == 0) return out;

    gt_host_view hva, hvo;
    const void * ap = hva.read(a);
    void * op = hvo.write_empty(out);
    std::vector<gt_host_view> hvis(n_idx);
    std::vector<const void *> ips(n_idx);
    for (int i = 0; i < n_idx; ++i) ips[i] = hvis[i].read(idx_tensors[i]);

    int64_t pre_size = 1, post_size = 1;
    std::vector<int> pre_dims, post_dims;
    for (int d = 0; d < a->ndim; ++d) {
        if (!indexed[d]) {
            if (d < first_dim) { pre_dims.push_back(d); pre_size *= a->shape[d]; }
            else { post_dims.push_back(d); post_size *= a->shape[d]; }
        }
    }

    for (int64_t p = 0; p < pre_size; ++p) {
        int64_t pre_off = 0, rem = p;
        for (int k = (int)pre_dims.size() - 1; k >= 0; --k) {
            int d = pre_dims[k];
            int64_t c = rem % a->shape[d]; rem /= a->shape[d];
            pre_off += c * a->strides[d];
        }
        for (int64_t ii = 0; ii < idx_numel; ++ii) {
            int64_t idx_off = 0;
            for (int m = 0; m < n_idx; ++m) {
                int64_t ioff = idx_tensors[m]->offset;
                int64_t irem = ii;
                for (int k = idx_ndim - 1; k >= 0; --k) {
                    int64_t c = irem % idx_shape[k]; irem /= idx_shape[k];
                    int mi = idx_tensors[m]->ndim - idx_ndim + k;
                    if (mi >= 0 && idx_tensors[m]->shape[mi] > 1) ioff += c * idx_tensors[m]->strides[mi];
                }
                int64_t coord = gt_read_i64_at(ips[m], idx_tensors[m]->dtype, ioff);
                int ad = dims[m];
                if (coord < 0) coord += a->shape[ad];
                coord = std::clamp<int64_t>(coord, 0, a->shape[ad] - 1);
                idx_off += coord * a->strides[ad];
            }
            for (int64_t q = 0; q < post_size; ++q) {
                int64_t post_off = 0, qrem = q;
                for (int k = (int)post_dims.size() - 1; k >= 0; --k) {
                    int d = post_dims[k];
                    int64_t c = qrem % a->shape[d]; qrem /= a->shape[d];
                    post_off += c * a->strides[d];
                }
                int64_t oidx = (p * idx_numel + ii) * post_size + q;
                int64_t aoff = a->offset + pre_off + idx_off + post_off;
                if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL)
                    gt_write_i64_at(op, out->dtype, oidx, gt_read_i64_at(ap, a->dtype, aoff));
                else
                    gt_write_f64_at(op, out->dtype, oidx, gt_read_f64_at(ap, a->dtype, aoff));
            }
        }
    }
    hvo.commit(out);
    return out;
}

extern "C" int gt_index_put(gt_tensor * a, int n_idx, const gt_tensor ** idx_tensors, const int * dims, const gt_tensor * values, int accumulate) {
    if (!a || n_idx <= 0 || !idx_tensors || !dims || !values) return -1;
    int idx_ndim = 0;
    int64_t idx_shape[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_idx; ++i) idx_ndim = std::max(idx_ndim, idx_tensors[i]->ndim);
    for (int k = 0; k < idx_ndim; ++k) {
        int64_t m = 1;
        for (int i = 0; i < n_idx; ++i) {
            int ii = idx_tensors[i]->ndim - 1 - k;
            int64_t d = (ii >= 0) ? idx_tensors[i]->shape[ii] : 1;
            m = std::max(m, d);
        }
        idx_shape[idx_ndim - 1 - k] = m;
    }
    int64_t idx_numel = 1;
    for (int k = 0; k < idx_ndim; ++k) idx_numel *= idx_shape[k];

    int first_dim = dims[0];
    bool indexed[GT_MAX_DIMS] = {};
    for (int i = 0; i < n_idx; ++i) indexed[dims[i]] = true;

    std::vector<int> pre_dims, post_dims;
    int64_t pre_size = 1, post_size = 1;
    for (int d = 0; d < a->ndim; ++d) {
        if (!indexed[d]) {
            if (d < first_dim) { pre_dims.push_back(d); pre_size *= a->shape[d]; }
            else { post_dims.push_back(d); post_size *= a->shape[d]; }
        }
    }

    gt_contig_owner vc(values);
    gt_host_view hva, hvv;
    void * ap = hva.read_write(a);
    const void * vp = hvv.read(vc.ptr);
    std::vector<gt_host_view> hvis(n_idx);
    std::vector<const void *> ips(n_idx);
    for (int i = 0; i < n_idx; ++i) ips[i] = hvis[i].read(idx_tensors[i]);

    for (int64_t p = 0; p < pre_size; ++p) {
        int64_t pre_off = 0, rem = p;
        for (int k = (int)pre_dims.size() - 1; k >= 0; --k) {
            int d = pre_dims[k];
            int64_t c = rem % a->shape[d]; rem /= a->shape[d];
            pre_off += c * a->strides[d];
        }
        for (int64_t ii = 0; ii < idx_numel; ++ii) {
            int64_t idx_off = 0;
            for (int m = 0; m < n_idx; ++m) {
                int64_t ioff = idx_tensors[m]->offset, irem = ii;
                for (int k = idx_ndim - 1; k >= 0; --k) {
                    int64_t c = irem % idx_shape[k]; irem /= idx_shape[k];
                    int mi = idx_tensors[m]->ndim - idx_ndim + k;
                    if (mi >= 0 && idx_tensors[m]->shape[mi] > 1) ioff += c * idx_tensors[m]->strides[mi];
                }
                int64_t coord = gt_read_i64_at(ips[m], idx_tensors[m]->dtype, ioff);
                int ad = dims[m];
                if (coord < 0) coord += a->shape[ad];
                coord = std::clamp<int64_t>(coord, 0, a->shape[ad] - 1);
                idx_off += coord * a->strides[ad];
            }
            for (int64_t q = 0; q < post_size; ++q) {
                int64_t post_off = 0, qrem = q;
                for (int k = (int)post_dims.size() - 1; k >= 0; --k) {
                    int d = post_dims[k];
                    int64_t c = qrem % a->shape[d]; qrem /= a->shape[d];
                    post_off += c * a->strides[d];
                }
                int64_t oidx = (p * idx_numel + ii) * post_size + q;
                int64_t vidx = (vc.ptr->numel == 1) ? 0 : (oidx % vc.ptr->numel);
                int64_t aoff = a->offset + pre_off + idx_off + post_off;
                if (gt_is_integer(a->dtype) || a->dtype == GT_BOOL) {
                    int64_t val = gt_read_i64_at(vp, vc.ptr->dtype, vidx);
                    if (accumulate) val += gt_read_i64_at(ap, a->dtype, aoff);
                    gt_write_i64_at(ap, a->dtype, aoff, val);
                } else {
                    double val = gt_read_f64_at(vp, vc.ptr->dtype, vidx);
                    if (accumulate) val += gt_read_f64_at(ap, a->dtype, aoff);
                    gt_write_f64_at(ap, a->dtype, aoff, val);
                }
            }
        }
    }
    hva.commit(a);
    return 0;
}

extern "C" gt_tensor * gt_masked_select(const gt_tensor * a, const gt_tensor * mask) {
    if (!a || !mask) return nullptr;
    gt_contig_owner ac(a), mc(mask);
    gt_host_view hva, hvm;
    const void * ap = hva.read(ac.ptr);
    const void * mp = hvm.read(mc.ptr);
    int64_t n = std::min(ac.ptr->numel, mc.ptr->numel);
    std::vector<int64_t> sel;
    sel.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        if (gt_read_i64_at(mp, mc.ptr->dtype, i) != 0) sel.push_back(i);
    }
    int64_t cnt = (int64_t)sel.size();
    gt_tensor * out = gt_tensor_empty(1, &cnt, a->dtype, a->device_type, a->device_index);
    if (!out || cnt == 0) return out;
    gt_host_view hvo;
    void * op = hvo.write_empty(out);
    for (int64_t i = 0; i < cnt; ++i) {
        if (gt_is_integer(out->dtype) || out->dtype == GT_BOOL)
            gt_write_i64_at(op, out->dtype, i, gt_read_i64_at(ap, ac.ptr->dtype, sel[i]));
        else
            gt_write_f64_at(op, out->dtype, i, gt_read_f64_at(ap, ac.ptr->dtype, sel[i]));
    }
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

extern "C" int gt_linalg_svd(const gt_tensor * a, int full_matrices, gt_tensor ** out_u, gt_tensor ** out_s, gt_tensor ** out_vh) {
    if (!a || a->ndim < 2 || !out_u || !out_s || !out_vh) return -1;
    int nd = a->ndim;
    int64_t M0 = a->shape[nd - 2], N0 = a->shape[nd - 1], K = std::min(M0, N0);
    int64_t B = a->numel / std::max<int64_t>(M0 * N0, 1);

    int64_t u_sh[GT_MAX_DIMS], s_sh[GT_MAX_DIMS], vh_sh[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) { u_sh[d] = a->shape[d]; vh_sh[d] = a->shape[d]; }
    for (int d = 0; d < nd - 1; ++d) s_sh[d] = a->shape[d];
    u_sh[nd - 1] = full_matrices ? M0 : K;
    s_sh[nd - 2] = K;
    vh_sh[nd - 2] = full_matrices ? N0 : K;

    gt_tensor * U = gt_tensor_empty(nd, u_sh, a->dtype, a->device_type, a->device_index);
    gt_tensor * S = gt_tensor_empty(nd - 1, s_sh, a->dtype, a->device_type, a->device_index);
    gt_tensor * Vh = gt_tensor_empty(nd, vh_sh, a->dtype, a->device_type, a->device_index);

    gt_contig_owner ac(a);
    gt_host_view hva, hvu, hvs, hvvh;
    const void * ap = hva.read(ac.ptr);
    void * up = hvu.write_empty(U);
    void * sp = hvs.write_empty(S);
    void * vhp = hvvh.write_empty(Vh);

    for (int64_t b = 0; b < B; ++b) {
        bool transposed = (M0 < N0);
        int64_t M = transposed ? N0 : M0, N = transposed ? M0 : N0;
        std::vector<double> W(M * N), V(N * N, 0.0);
        for (int64_t i = 0; i < N; ++i) V[i * N + i] = 1.0;
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j)
                W[i * N + j] = transposed ? gt_read_f64_at(ap, ac.ptr->dtype, (b * M0 + j) * N0 + i)
                                          : gt_read_f64_at(ap, ac.ptr->dtype, (b * M0 + i) * N0 + j);
        for (int sweep = 0; sweep < 40; ++sweep) {
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
        std::vector<std::pair<double, int64_t>> sig(N);
        for (int64_t j = 0; j < N; ++j) {
            double n2 = 0.0;
            for (int64_t i = 0; i < M; ++i) n2 += W[i * N + j] * W[i * N + j];
            sig[j] = {std::sqrt(n2), j};
        }
        std::sort(sig.begin(), sig.end(), [](const auto & x, const auto & y) { return x.first > y.first; });

        int64_t ucols_tmp = full_matrices ? M : N;
        std::vector<double> Ufull(M * ucols_tmp, 0.0), Vfull(N * N, 0.0);
        for (int64_t k = 0; k < N; ++k) {
            double sv = sig[k].first;
            int64_t sc = sig[k].second;
            gt_write_f64_at(sp, S->dtype, b * K + k, sv);
            for (int64_t i = 0; i < M; ++i) Ufull[i * ucols_tmp + k] = (sv > 1e-15) ? (W[i * N + sc] / sv) : (i == k ? 1.0 : 0.0);
            for (int64_t i = 0; i < N; ++i) Vfull[k * N + i] = V[i * N + sc];
        }
        if (full_matrices && M > N) {
            for (int64_t k = N; k < M; ++k) {
                Ufull[k * M + k] = 1.0;
                for (int64_t pass = 0; pass < 2; ++pass) {
                    for (int64_t j = 0; j < k; ++j) {
                        double dot = 0.0;
                        for (int64_t i = 0; i < M; ++i) dot += Ufull[i * M + k] * Ufull[i * M + j];
                        for (int64_t i = 0; i < M; ++i) Ufull[i * M + k] -= dot * Ufull[i * M + j];
                    }
                    double nrm = 0.0;
                    for (int64_t i = 0; i < M; ++i) nrm += Ufull[i * M + k] * Ufull[i * M + k];
                    nrm = std::sqrt(nrm);
                    if (nrm < 1e-12) Ufull[0 * M + k] = 1.0;
                    else for (int64_t i = 0; i < M; ++i) Ufull[i * M + k] /= nrm;
                }
            }
        }
        int64_t out_uc = u_sh[nd - 1], out_vr = vh_sh[nd - 2];
        if (!transposed) {
            for (int64_t i = 0; i < M0; ++i)
                for (int64_t j = 0; j < out_uc; ++j)
                    gt_write_f64_at(up, U->dtype, (b * M0 + i) * out_uc + j, Ufull[i * out_uc + j]);
            for (int64_t i = 0; i < out_vr; ++i)
                for (int64_t j = 0; j < N0; ++j)
                    gt_write_f64_at(vhp, Vh->dtype, (b * out_vr + i) * N0 + j, Vfull[i * N + j]);
        } else {
            for (int64_t i = 0; i < M0; ++i)
                for (int64_t j = 0; j < out_uc; ++j)
                    gt_write_f64_at(up, U->dtype, (b * M0 + i) * out_uc + j, Vfull[j * N + i]);
            for (int64_t i = 0; i < out_vr; ++i)
                for (int64_t j = 0; j < N0; ++j)
                    gt_write_f64_at(vhp, Vh->dtype, (b * out_vr + i) * N0 + j, Ufull[j * ucols_tmp + i]);
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

extern "C" int gt_gguf_save(const char * path, int n_tensors, const char ** names, const gt_tensor ** tensors,
                            int n_kv, const char ** kv_keys, const int * kv_types, const char ** kv_vals) {
    if (!path) return -1;
    struct gguf_context * gctx = gguf_init_empty();
    if (!gctx) return -1;

    for (int i = 0; i < n_kv; ++i) {
        if (!kv_keys || !kv_keys[i] || !kv_vals || !kv_vals[i]) continue;
        int kt = kv_types ? kv_types[i] : (int)GGUF_TYPE_STRING;
        const char * k = kv_keys[i];
        const char * v = kv_vals[i];
        if (kt == GGUF_TYPE_INT32) gguf_set_val_i32(gctx, k, std::atoi(v));
        else if (kt == GGUF_TYPE_UINT32) gguf_set_val_u32(gctx, k, (uint32_t)std::strtoul(v, nullptr, 10));
        else if (kt == GGUF_TYPE_INT64) gguf_set_val_i64(gctx, k, std::strtoll(v, nullptr, 10));
        else if (kt == GGUF_TYPE_UINT64) gguf_set_val_u64(gctx, k, std::strtoull(v, nullptr, 10));
        else if (kt == GGUF_TYPE_FLOAT32) gguf_set_val_f32(gctx, k, std::strtof(v, nullptr));
        else if (kt == GGUF_TYPE_FLOAT64) gguf_set_val_f64(gctx, k, std::strtod(v, nullptr));
        else if (kt == GGUF_TYPE_BOOL) gguf_set_val_bool(gctx, k, std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0);
        else gguf_set_val_str(gctx, k, v);
    }

    size_t ctx_size = ggml_tensor_overhead() * (size_t)std::max(1, n_tensors + 4);
    struct ggml_init_params p = { ctx_size, nullptr, true };
    struct ggml_context * mctx = ggml_init(p);
    std::vector<gt_contig_owner *> owners;
    std::vector<gt_host_view *> views;
    std::vector<gt_tensor *> tmp_f32s;

    for (int i = 0; i < n_tensors; ++i) {
        if (!names[i] || !tensors[i]) continue;
        const gt_tensor * t = tensors[i];
        int nd = std::max(1, std::min(t->ndim, 4));
        int64_t ne[4] = {1, 1, 1, 1};
        if (t->ndim == 0) ne[0] = 1;
        else {
            for (int d = 0; d < std::min(t->ndim, 4); ++d) ne[d] = t->shape[t->ndim - 1 - d];
            if (t->ndim > 4) {
                int64_t extra = 1;
                for (int d = 0; d < t->ndim - 3; ++d) extra *= t->shape[d];
                ne[3] = extra;
            }
        }
        enum ggml_type gtype = gt_dtype_to_ggml(t->dtype);
        if (gtype == GGML_TYPE_COUNT) {
            gt_tensor * tf = gt_tensor_to_dtype(t, GT_F32);
            struct ggml_tensor * gm = ggml_new_tensor(mctx, GGML_TYPE_F32, nd, ne);
            ggml_set_name(gm, names[i]);
            auto * hv = new gt_host_view();
            gm->data = const_cast<void *>(hv->read(tf));
            gguf_add_tensor(gctx, gm);
            views.push_back(hv);
            tmp_f32s.push_back(tf);
            continue;
        }
        auto * co = new gt_contig_owner(t);
        auto * hv = new gt_host_view();
        struct ggml_tensor * gm = ggml_new_tensor(mctx, gtype, nd, ne);
        ggml_set_name(gm, names[i]);
        gm->data = const_cast<void *>(hv->read(co->ptr));
        gguf_add_tensor(gctx, gm);
        owners.push_back(co);
        views.push_back(hv);
    }

    bool ok = gguf_write_to_file(gctx, path, false);
    for (auto * v : views) delete v;
    for (auto * o : owners) delete o;
    for (auto * tf : tmp_f32s) gt_tensor_free(tf);
    ggml_free(mctx);
    gguf_free(gctx);
    return ok ? 0 : -1;
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
    return gf && gf->gctx ? gguf_get_tensor_name(gf->gctx, idx) : nullptr;
}

extern "C" gt_tensor * gt_gguf_get_tensor(const gt_gguf_file * gf, const char * name, int dequantize_f32) {
    if (!gf || !gf->gctx || !gf->mctx || !name) return nullptr;
    struct ggml_tensor * gm = ggml_get_tensor(gf->mctx, name);
    if (!gm) return nullptr;
    int nd = ggml_n_dims(gm);
    int64_t shape[GT_MAX_DIMS];
    for (int d = 0; d < nd; ++d) shape[d] = gm->ne[nd - 1 - d];
    int dt = gt_ggml_to_dtype(gm->type);
    if (gt_is_quantized(dt) && !dequantize_f32) {
        size_t nbytes = ggml_nbytes(gm);
        return gt_tensor_from_data(nd, shape, dt, GT_DEV_CPU, 0, gm->data, nbytes);
    }
    gt_tensor * raw = gt_tensor_from_data(nd, shape, dt, GT_DEV_CPU, 0, gm->data, ggml_nbytes(gm));
    if (!raw) return nullptr;
    if (dequantize_f32 && dt != GT_F32 && gt_is_quantized(dt)) {
        gt_tensor * f32 = gt_tensor_to_dtype(raw, GT_F32);
        gt_tensor_free(raw);
        return f32;
    }
    return raw;
}

extern "C" int64_t gt_gguf_n_kv(const gt_gguf_file * gf) {
    return gf && gf->gctx ? gguf_get_n_kv(gf->gctx) : 0;
}

extern "C" const char * gt_gguf_kv_key(const gt_gguf_file * gf, int64_t idx) {
    return gf && gf->gctx ? gguf_get_key(gf->gctx, idx) : nullptr;
}

extern "C" int gt_gguf_kv_type(const gt_gguf_file * gf, int64_t idx) {
    return gf && gf->gctx ? (int)gguf_get_kv_type(gf->gctx, idx) : -1;
}

extern "C" int gt_gguf_kv_val_str(const gt_gguf_file * gf, int64_t idx, char * buf, int buf_len) {
    if (!gf || !gf->gctx || !buf || buf_len <= 0) return -1;
    enum gguf_type t = gguf_get_kv_type(gf->gctx, idx);
    int n = 0;
    switch (t) {
        case GGUF_TYPE_STRING: n = std::snprintf(buf, buf_len, "%s", gguf_get_val_str(gf->gctx, idx)); break;
        case GGUF_TYPE_UINT8:  n = std::snprintf(buf, buf_len, "%u", (unsigned)gguf_get_val_u8(gf->gctx, idx)); break;
        case GGUF_TYPE_INT8:   n = std::snprintf(buf, buf_len, "%d", (int)gguf_get_val_i8(gf->gctx, idx)); break;
        case GGUF_TYPE_UINT16: n = std::snprintf(buf, buf_len, "%u", (unsigned)gguf_get_val_u16(gf->gctx, idx)); break;
        case GGUF_TYPE_INT16:  n = std::snprintf(buf, buf_len, "%d", (int)gguf_get_val_i16(gf->gctx, idx)); break;
        case GGUF_TYPE_UINT32: n = std::snprintf(buf, buf_len, "%u", gguf_get_val_u32(gf->gctx, idx)); break;
        case GGUF_TYPE_INT32:  n = std::snprintf(buf, buf_len, "%d", gguf_get_val_i32(gf->gctx, idx)); break;
        case GGUF_TYPE_FLOAT32: n = std::snprintf(buf, buf_len, "%g", gguf_get_val_f32(gf->gctx, idx)); break;
        case GGUF_TYPE_UINT64: n = std::snprintf(buf, buf_len, "%llu", (unsigned long long)gguf_get_val_u64(gf->gctx, idx)); break;
        case GGUF_TYPE_INT64:  n = std::snprintf(buf, buf_len, "%lld", (long long)gguf_get_val_i64(gf->gctx, idx)); break;
        case GGUF_TYPE_FLOAT64: n = std::snprintf(buf, buf_len, "%g", gguf_get_val_f64(gf->gctx, idx)); break;
        case GGUF_TYPE_BOOL:   n = std::snprintf(buf, buf_len, "%s", gguf_get_val_bool(gf->gctx, idx) ? "true" : "false"); break;
        case GGUF_TYPE_ARRAY:  n = std::snprintf(buf, buf_len, "<array len=%zu>", gguf_get_arr_n(gf->gctx, idx)); break;
        default: buf[0] = '\0'; n = 0; break;
    }
    return n;
}

extern "C" gt_tensor * gt_quantize(const gt_tensor * a, int quant_dtype) {
    return a ? gt_tensor_to_dtype(a, quant_dtype) : nullptr;
}

extern "C" gt_tensor * gt_dequantize(const gt_tensor * a, int target_dtype) {
    return a ? gt_tensor_to_dtype(a, target_dtype) : nullptr;
}

// ---------- Native ggml_opt Static Graph Builder & Trainer ----------

struct gt_opt_graph {
    int loss_type = 3;      // 0=MEAN, 1=SUM, 2=CROSS_ENTROPY, 3=MSE
    int optimizer_type = 0; // 0=ADAMW, 1=SGD
    int64_t nbatch_physical = 32;
    int64_t nbatch_logical = 32;
    struct ggml_context * ctx_static = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    std::vector<struct ggml_tensor *> nodes;
    int input_id = -1;
    std::vector<gt_tensor *> param_refs;
    std::vector<int> param_node_ids;
    std::vector<ggml_backend_buffer_t> static_bufs;
    ggml_backend_sched_t sched = nullptr;
    int64_t n_in = 0;
};

extern "C" gt_opt_graph * gt_opt_graph_create(int loss_type, int optimizer_type, int64_t nbatch_physical, int64_t nbatch_logical) {
    gt_init();
    auto * g = new gt_opt_graph();
    g->loss_type = loss_type;
    g->optimizer_type = optimizer_type;
    g->nbatch_physical = std::max<int64_t>(1, nbatch_physical);
    g->nbatch_logical  = std::max<int64_t>(g->nbatch_physical, nbatch_logical);
    size_t sz = 16ULL * 1024ULL * 1024ULL;
    g->ctx_static  = ggml_init({ sz / 2, nullptr, true });
    g->ctx_compute = ggml_init({ sz / 2, nullptr, true });
    return g;
}

extern "C" void gt_opt_graph_free(gt_opt_graph * g) {
    if (!g) return;
    if (g->sched) ggml_backend_sched_free(g->sched);
    for (auto b : g->static_bufs) if (b) ggml_backend_buffer_free(b);
    if (g->ctx_compute) ggml_free(g->ctx_compute);
    if (g->ctx_static)  ggml_free(g->ctx_static);
    delete g;
}

extern "C" int gt_opt_graph_input(gt_opt_graph * g, int ndim, const int64_t * shape) {
    if (!g || ndim < 1 || ndim > 4) return -1;
    int64_t ne[4] = {1, 1, 1, 1};
    for (int d = 0; d < ndim; ++d) ne[d] = shape[ndim - 1 - d];
    struct ggml_tensor * t = ggml_new_tensor(g->ctx_static, GGML_TYPE_F32, ndim, ne);
    g->nodes.push_back(t);
    int id = (int)g->nodes.size() - 1;
    g->input_id = id;
    g->n_in = ne[0];
    return id;
}

extern "C" int gt_opt_graph_param(gt_opt_graph * g, gt_tensor * param) {
    if (!g || !param || param->ndim > 4) return -1;
    int nd = std::max(1, param->ndim);
    int64_t ne[4] = {1, 1, 1, 1};
    for (int d = 0; d < param->ndim; ++d) ne[d] = param->shape[param->ndim - 1 - d];
    struct ggml_tensor * t = ggml_new_tensor(g->ctx_static, gt_dtype_to_ggml(param->dtype), nd, ne);
    ggml_set_param(t);
    g->nodes.push_back(t);
    int id = (int)g->nodes.size() - 1;
    g->param_refs.push_back(param);
    g->param_node_ids.push_back(id);
    return id;
}

extern "C" int gt_opt_graph_linear(gt_opt_graph * g, int x_id, int w_id, int b_id) {
    if (!g || x_id < 0 || w_id < 0) return -1;
    struct ggml_tensor * r = ggml_mul_mat(g->ctx_compute, g->nodes[w_id], g->nodes[x_id]);
    if (b_id >= 0) r = ggml_add(g->ctx_compute, r, g->nodes[b_id]);
    g->nodes.push_back(r);
    return (int)g->nodes.size() - 1;
}

extern "C" int gt_opt_graph_unary(gt_opt_graph * g, int x_id, int unary_op) {
    if (!g || x_id < 0) return -1;
    struct ggml_tensor * x = g->nodes[x_id];
    struct ggml_tensor * r = nullptr;
    switch (unary_op) {
        case GT_UNARY_RELU: r = ggml_relu(g->ctx_compute, x); break;
        case GT_UNARY_SILU: r = ggml_silu(g->ctx_compute, x); break;
        case GT_UNARY_GELU: r = ggml_gelu_erf(g->ctx_compute, x); break;
        case GT_UNARY_TANH: r = ggml_tanh(g->ctx_compute, x); break;
        case GT_UNARY_SIGMOID: r = ggml_sigmoid(g->ctx_compute, x); break;
        case GT_UNARY_EXP: r = ggml_exp(g->ctx_compute, x); break;
        case GT_UNARY_LOG: r = ggml_log(g->ctx_compute, x); break;
        case GT_UNARY_SQUARE: r = ggml_sqr(g->ctx_compute, x); break;
        default: r = ggml_relu(g->ctx_compute, x); break;
    }
    g->nodes.push_back(r);
    return (int)g->nodes.size() - 1;
}

extern "C" int gt_opt_graph_add(gt_opt_graph * g, int a_id, int b_id) {
    if (!g || a_id < 0 || b_id < 0) return -1;
    g->nodes.push_back(ggml_add(g->ctx_compute, g->nodes[a_id], g->nodes[b_id]));
    return (int)g->nodes.size() - 1;
}

extern "C" int gt_opt_graph_mul(gt_opt_graph * g, int a_id, int b_id) {
    if (!g || a_id < 0 || b_id < 0) return -1;
    g->nodes.push_back(ggml_mul(g->ctx_compute, g->nodes[a_id], g->nodes[b_id]));
    return (int)g->nodes.size() - 1;
}

extern "C" int gt_opt_graph_rms_norm(gt_opt_graph * g, int x_id, int w_id, float eps) {
    if (!g || x_id < 0) return -1;
    struct ggml_tensor * r = ggml_rms_norm(g->ctx_compute, g->nodes[x_id], eps);
    if (w_id >= 0) r = ggml_mul(g->ctx_compute, r, g->nodes[w_id]);
    g->nodes.push_back(r);
    return (int)g->nodes.size() - 1;
}

extern "C" int gt_opt_graph_softmax(gt_opt_graph * g, int x_id) {
    if (!g || x_id < 0) return -1;
    g->nodes.push_back(ggml_soft_max(g->ctx_compute, g->nodes[x_id]));
    return (int)g->nodes.size() - 1;
}

extern "C" int gt_opt_graph_fit(gt_opt_graph * g, int out_id, const gt_tensor * data, const gt_tensor * labels,
                                int64_t epochs, float lr, float wd, float val_split, int silent,
                                double * out_train_loss, double * out_val_loss) {
    if (!g || g->input_id < 0 || out_id < 0 || !data || data->ndim < 2) return -1;

    gt_backend_entry * be = gt_get_backend(GT_DEV_CPU, 0);
    if (!be) return -1;

    ggml_backend_t backends[1] = { be->backend };
    if (!g->sched) {
        g->sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
        ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(g->ctx_static, be->backend);
        g->static_bufs.push_back(sbuf);
    }

    for (size_t i = 0; i < g->param_refs.size(); ++i) {
        gt_tensor * pt = g->param_refs[i];
        gt_contig_owner pc(pt);
        gt_host_view hvp;
        const void * ptr = hvp.read(pc.ptr);
        struct ggml_tensor * gp = g->nodes[g->param_node_ids[i]];
        ggml_backend_tensor_set(gp, ptr, 0, ggml_nbytes(gp));
    }

    int64_t n_data_raw = data->shape[0];
    int64_t n_data = (n_data_raw / g->nbatch_physical) * g->nbatch_physical;
    if (n_data < g->nbatch_physical) {
        gt_set_error("OptGraph.fit requires dataset rows >= nbatch_physical");
        return -1;
    }
    int64_t ne_dp = data->numel / std::max<int64_t>(n_data_raw, 1);
    int64_t ne_lbl = labels ? (labels->numel / std::max<int64_t>(n_data_raw, 1)) : 0;

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, ne_dp, ne_lbl, n_data, g->nbatch_physical);

    gt_tensor * data_f32 = (data->dtype == GT_F32 && gt_check_contiguous(data)) ? nullptr : gt_tensor_to_dtype(data, GT_F32);
    const gt_tensor * d_use = data_f32 ? data_f32 : data;
    {
        gt_host_view hvd;
        const void * dp = hvd.read(d_use);
        std::memcpy(ggml_opt_dataset_data(dataset)->data, dp, (size_t)n_data * (size_t)ne_dp * sizeof(float));
    }
    if (data_f32) gt_tensor_free(data_f32);

    if (labels && ne_lbl > 0) {
        gt_tensor * lbl_f32 = (labels->dtype == GT_F32 && gt_check_contiguous(labels)) ? nullptr : gt_tensor_to_dtype(labels, GT_F32);
        const gt_tensor * l_use = lbl_f32 ? lbl_f32 : labels;
        gt_host_view hvl;
        const void * lp = hvl.read(l_use);
        std::memcpy(ggml_opt_dataset_labels(dataset)->data, lp, (size_t)n_data * (size_t)ne_lbl * sizeof(float));
        if (lbl_f32) gt_tensor_free(lbl_f32);
    }

    static thread_local float tl_lr = 1e-3f, tl_wd = 0.0f;
    tl_lr = lr; tl_wd = wd;
    auto get_opt_tl = [](void *) -> struct ggml_opt_optimizer_params {
        struct ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
        p.adamw.alpha = tl_lr;
        p.adamw.wd    = tl_wd;
        p.sgd.alpha   = tl_lr;
        p.sgd.wd      = tl_wd;
        return p;
    };

    enum ggml_opt_loss_type ltype = GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR;
    if (g->loss_type == 0) ltype = GGML_OPT_LOSS_TYPE_MEAN;
    else if (g->loss_type == 1) ltype = GGML_OPT_LOSS_TYPE_SUM;
    else if (g->loss_type == 2) ltype = GGML_OPT_LOSS_TYPE_CROSS_ENTROPY;

    enum ggml_opt_optimizer_type otype = (g->optimizer_type == 1) ? GGML_OPT_OPTIMIZER_TYPE_SGD : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    struct ggml_opt_params opt_pars = ggml_opt_default_params(g->sched, ltype);
    opt_pars.ctx_compute     = g->ctx_compute;
    opt_pars.inputs          = g->nodes[g->input_id];
    opt_pars.outputs         = g->nodes[out_id];
    opt_pars.opt_period      = (int32_t)(g->nbatch_logical / g->nbatch_physical);
    opt_pars.get_opt_pars    = get_opt_tl;
    opt_pars.get_opt_pars_ud = nullptr;
    opt_pars.optimizer       = otype;

    ggml_opt_context_t opt_ctx = ggml_opt_init(opt_pars);
    int64_t opt_period = std::max<int64_t>(1, g->nbatch_logical / g->nbatch_physical);
    int64_t nbatches_phys = n_data / g->nbatch_physical;
    int64_t ibatch_split = (val_split <= 0.0f) ? nbatches_phys : (int64_t)((1.0f - val_split) * (float)nbatches_phys);
    if (val_split > 0.0f && ibatch_split >= nbatches_phys && nbatches_phys > 1) ibatch_split = nbatches_phys - 1;
    if (ibatch_split < 1) ibatch_split = 1;
    (void)opt_period;
    int64_t idata_split = ibatch_split * g->nbatch_physical;

    ggml_opt_result_t res_train = ggml_opt_result_init();
    ggml_opt_result_t res_eval  = ggml_opt_result_init();

    ggml_opt_epoch_callback cb = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;
    double last_tr = 0.0, last_ev = 0.0;

    for (int64_t e = 0; e < epochs; ++e) {
        if (idata_split > 0) ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        ggml_opt_result_reset(res_train);
        ggml_opt_result_reset(res_eval);
        ggml_opt_epoch(opt_ctx, dataset, res_train, res_eval, idata_split, cb, cb);
        ggml_opt_result_loss(res_train, &last_tr, nullptr);
        if (idata_split < n_data) ggml_opt_result_loss(res_eval, &last_ev, nullptr);
    }

    if (out_train_loss) *out_train_loss = last_tr;
    if (out_val_loss)   *out_val_loss   = last_ev;

    for (size_t i = 0; i < g->param_refs.size(); ++i) {
        gt_tensor * pt = g->param_refs[i];
        gt_host_view hvp;
        void * ptr = hvp.write_empty(pt);
        struct ggml_tensor * gp = g->nodes[g->param_node_ids[i]];
        ggml_backend_tensor_get(gp, ptr, 0, ggml_nbytes(gp));
        hvp.commit(pt);
    }

    ggml_opt_result_free(res_train);
    ggml_opt_result_free(res_eval);
    ggml_opt_free(opt_ctx);
    ggml_opt_dataset_free(dataset);
    return 0;
}
