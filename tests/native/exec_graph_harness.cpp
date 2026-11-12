// Result delivery of gt_exec_ggml_graph (BUG-08).
//
// Every fast path in csrc/ calls it with a fresh output and a fresh result node, so the
// cases below - an output that is a view, a result node that is a view or permuted, a
// result ggml typed differently from the output - cannot be reached through the Python API.
// tests/test_views_and_fast_paths.py compiles this file against the built libraries and
// runs it; the process exits with the number of failed checks.
#include "gt_internal.h"
#include <cstdio>
#include <cmath>
#include <vector>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { ++fails; printf("FAIL %s\n", msg); } else printf("ok   %s\n", msg); } while (0)

static std::vector<float> host(const gt_tensor * t) {
    std::vector<float> v((size_t)t->numel);
    gt_tensor * c = gt_tensor_empty(t->ndim, t->shape, t->dtype, t->device_type, t->device_index);
    gt_tensor_copy_inplace(c, t);
    gt_host_view hv; const float * p = (const float *)hv.read(c);
    for (int64_t i = 0; i < t->numel; ++i) v[i] = p[i];
    gt_tensor_free(c);
    return v;
}

int main() {
    gt_backend_entry * be = gt_get_backend(GT_DEV_CPU, 0);
    CHECK(be != nullptr, "cpu backend");
    float src[6] = {1, 2, 3, 4, 5, 6};           // torch shape (2, 3): [[1,2,3],[4,5,6]]
    int64_t sh[2] = {2, 3};
    gt_tensor * a = gt_tensor_from_data(2, sh, GT_F32, GT_DEV_CPU, 0, src, sizeof(src));

    auto mk = [&]() {
        struct ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        return ggml_init(p);
    };

    // (1) out_node is a permuted (non-contiguous) view: transpose of a*2 -> torch shape (3, 2)
    {
        struct ggml_context * ctx = mk();
        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, 3, 2, 1, 1);
        struct ggml_tensor * gout = ggml_permute(ctx, ggml_scale(ctx, ga, 2.0f), 1, 0, 2, 3);   // ne = [2, 3]
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gout);
        int64_t osh[2] = {3, 2};
        gt_tensor * out = gt_tensor_empty(2, osh, GT_F32, GT_DEV_CPU, 0);
        bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
        std::vector<float> v = host(out);
        float want[6] = {2, 8, 4, 10, 6, 12};
        bool same = ok; for (int i = 0; i < 6; ++i) same = same && v[i] == want[i];
        CHECK(same, "non-contiguous out_node is laid out before the copy");
        ggml_free(ctx); gt_tensor_free(out);
    }
    // (2) out_node is a reshape view (contiguous view of a computed node) into an OFFSET output
    {
        struct ggml_context * ctx = mk();
        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, 3, 2, 1, 1);
        struct ggml_tensor * gout = ggml_reshape_1d(ctx, ggml_scale(ctx, ga, 3.0f), 6);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gout);
        int64_t bsh[1] = {10};
        gt_tensor * base = gt_tensor_empty(1, bsh, GT_F32, GT_DEV_CPU, 0);
        gt_fill_scalar_f64(base, -1.0);
        int64_t vsh[1] = {6}, vst[1] = {1};
        gt_tensor * view = gt_tensor_view(base, 1, vsh, vst, 3);       // base[3:9]
        bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, view);
        std::vector<float> v = host(base);
        float want[10] = {-1, -1, -1, 3, 6, 9, 12, 15, 18, -1};
        bool same = ok; for (int i = 0; i < 10; ++i) same = same && v[i] == want[i];
        CHECK(same, "result lands at the output's offset, neighbours untouched");
        ggml_free(ctx); gt_tensor_free(view); gt_tensor_free(base);
    }
    // (3) fresh node into a STRIDED output (every other element of a base)
    {
        struct ggml_context * ctx = mk();
        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, 6, 1, 1, 1);
        struct ggml_tensor * gout = ggml_scale(ctx, ga, 10.0f);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gout);
        int64_t bsh[1] = {12};
        gt_tensor * base = gt_tensor_empty(1, bsh, GT_F32, GT_DEV_CPU, 0);
        gt_fill_scalar_f64(base, -1.0);
        int64_t vsh[1] = {6}, vst[1] = {2};
        gt_tensor * view = gt_tensor_view(base, 1, vsh, vst, 1);       // base[1::2]
        bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, view);
        std::vector<float> v = host(base);
        float want[12] = {-1, 10, -1, 20, -1, 30, -1, 40, -1, 50, -1, 60};
        bool same = ok; for (int i = 0; i < 12; ++i) same = same && v[i] == want[i];
        CHECK(same, "strided output receives the result element by element");
        ggml_free(ctx); gt_tensor_free(view); gt_tensor_free(base);
    }
    // (4) the output node is an input leaf itself (data already set)
    {
        struct ggml_context * ctx = mk();
        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, 6, 1, 1, 1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ga);
        int64_t osh[1] = {6};
        gt_tensor * out = gt_tensor_empty(1, osh, GT_F32, GT_DEV_CPU, 0);
        bool ok = gt_exec_ggml_graph(be, ctx, gf, ga, out);
        std::vector<float> v = host(out);
        bool same = ok; for (int i = 0; i < 6; ++i) same = same && v[i] == src[i];
        std::vector<float> av = host(a);
        for (int i = 0; i < 6; ++i) same = same && av[i] == src[i];
        CHECK(same, "leaf output is copied, the input keeps its data");
        ggml_free(ctx); gt_tensor_free(out);
    }
    // (5) type mismatch and size mismatch are refused and leave the output alone
    {
        struct ggml_context * ctx = mk();
        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, 6, 1, 1, 1);
        struct ggml_tensor * gout = ggml_scale(ctx, ga, 2.0f);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gout);
        int64_t osh[1] = {6};
        gt_tensor * out16 = gt_tensor_empty(1, osh, GT_F16, GT_DEV_CPU, 0);
        gt_fill_scalar_f64(out16, 7.0);
        bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out16);
        gt_host_view hv; const uint16_t * p = (const uint16_t *)hv.read(out16);
        bool untouched = true; for (int i = 1; i < 6; ++i) untouched = untouched && p[i] == p[0];
        CHECK(!ok && untouched, "F32 result is refused for an F16 output");
        int64_t osh2[1] = {5};
        gt_tensor * small = gt_tensor_empty(1, osh2, GT_F32, GT_DEV_CPU, 0);
        CHECK(!gt_exec_ggml_graph(be, ctx, gf, gout, small), "6-element result is refused for a 5-element output");
        int64_t osh3[1] = {3};
        gt_tensor * cplx = gt_tensor_empty(1, osh3, GT_C64, GT_DEV_CPU, 0);
        CHECK(!gt_exec_ggml_graph(be, ctx, gf, gout, cplx), "real result is refused for a complex output");
        ggml_free(ctx); gt_tensor_free(out16); gt_tensor_free(small); gt_tensor_free(cplx);
    }
    // (6) the plain direct case still works
    {
        struct ggml_context * ctx = mk();
        struct ggml_tensor * ga = gt_wrap_ggml_4d(ctx, a, 6, 1, 1, 1);
        struct ggml_tensor * gout = ggml_scale(ctx, ga, 2.0f);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gout);
        int64_t osh[2] = {2, 3};
        gt_tensor * out = gt_tensor_empty(2, osh, GT_F32, GT_DEV_CPU, 0);
        bool ok = gt_exec_ggml_graph(be, ctx, gf, gout, out);
        std::vector<float> v = host(out);
        bool same = ok; for (int i = 0; i < 6; ++i) same = same && v[i] == 2 * src[i];
        CHECK(same, "direct output");
        ggml_free(ctx); gt_tensor_free(out);
    }
    gt_tensor_free(a);
    printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
