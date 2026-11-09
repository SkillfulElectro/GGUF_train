#include "llama_bridge.h"
#include "llama.h"
#include "ggml-opt.h"
#include <algorithm>
#include <cstring>
#include <vector>

struct gt_llama_model {
    struct llama_model * model = nullptr;
};

struct gt_llama_context {
    struct llama_context * ctx = nullptr;
    gt_llama_model * model = nullptr;
};

extern "C" void gt_llama_backend_init(void) {
    gt_init();
    llama_backend_init();
}

extern "C" void gt_llama_backend_free(void) {
    llama_backend_free();
}

extern "C" gt_llama_model * gt_llama_load_model(const char * path, int n_gpu_layers, int vocab_only, int use_mmap) {
    if (!path) return nullptr;
    gt_llama_backend_init();
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.vocab_only = (vocab_only != 0);
    mparams.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    struct llama_model * m = llama_model_load_from_file(path, mparams);
    if (!m) return nullptr;
    auto * wrap = new gt_llama_model();
    wrap->model = m;
    return wrap;
}

extern "C" void gt_llama_free_model(gt_llama_model * m) {
    if (!m) return;
    if (m->model) llama_model_free(m->model);
    delete m;
}

extern "C" int gt_llama_save_model(const gt_llama_model * m, const char * path) {
    if (!m || !m->model || !path) return -1;
    llama_model_save_to_file(m->model, path);
    return 0;
}

extern "C" int32_t gt_llama_n_vocab(const gt_llama_model * m) {
    if (!m || !m->model) return 0;
    return llama_vocab_n_tokens(llama_model_get_vocab(m->model));
}

extern "C" int32_t gt_llama_n_ctx_train(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_n_ctx_train(m->model) : 0;
}

extern "C" int32_t gt_llama_n_embd(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_n_embd(m->model) : 0;
}

extern "C" int32_t gt_llama_n_layer(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_n_layer(m->model) : 0;
}

extern "C" int32_t gt_llama_n_head(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_n_head(m->model) : 0;
}

extern "C" int32_t gt_llama_n_head_kv(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_n_head_kv(m->model) : 0;
}

extern "C" uint64_t gt_llama_model_n_params(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_n_params(m->model) : 0;
}

extern "C" uint64_t gt_llama_model_size(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_size(m->model) : 0;
}

extern "C" int32_t gt_llama_vocab_bos(const gt_llama_model * m) {
    return (m && m->model) ? llama_vocab_bos(llama_model_get_vocab(m->model)) : -1;
}

extern "C" int32_t gt_llama_vocab_eos(const gt_llama_model * m) {
    return (m && m->model) ? llama_vocab_eos(llama_model_get_vocab(m->model)) : -1;
}

extern "C" int32_t gt_llama_vocab_pad(const gt_llama_model * m) {
    return (m && m->model) ? llama_vocab_pad(llama_model_get_vocab(m->model)) : -1;
}

extern "C" int32_t gt_llama_model_desc(const gt_llama_model * m, char * buf, size_t buf_len) {
    return (m && m->model && buf) ? llama_model_desc(m->model, buf, buf_len) : -1;
}

extern "C" int32_t gt_llama_model_meta_count(const gt_llama_model * m) {
    return (m && m->model) ? llama_model_meta_count(m->model) : 0;
}

extern "C" int32_t gt_llama_model_meta_key(const gt_llama_model * m, int32_t idx, char * buf, size_t buf_len) {
    return (m && m->model && buf) ? llama_model_meta_key_by_index(m->model, idx, buf, buf_len) : -1;
}

extern "C" int32_t gt_llama_model_meta_val(const gt_llama_model * m, const char * key, char * buf, size_t buf_len) {
    return (m && m->model && key && buf) ? llama_model_meta_val_str(m->model, key, buf, buf_len) : -1;
}

extern "C" gt_llama_context * gt_llama_new_context(gt_llama_model * m, int n_ctx, int n_batch, int n_threads, int embeddings) {
    if (!m || !m->model) return nullptr;
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? (uint32_t)n_ctx : 2048;
    cparams.n_batch = n_batch > 0 ? (uint32_t)n_batch : cparams.n_ctx;
    int nt = n_threads > 0 ? n_threads : gt_get_num_threads();
    cparams.n_threads = nt;
    cparams.n_threads_batch = nt;
    cparams.embeddings = (embeddings != 0);
    struct llama_context * c = llama_init_from_model(m->model, cparams);
    if (!c) return nullptr;
    auto * wrap = new gt_llama_context();
    wrap->ctx = c;
    wrap->model = m;
    return wrap;
}

extern "C" void gt_llama_free_context(gt_llama_context * c) {
    if (!c) return;
    if (c->ctx) llama_free(c->ctx);
    delete c;
}

extern "C" void gt_llama_clear_kv(gt_llama_context * c) {
    if (!c || !c->ctx) return;
    llama_memory_t mem = llama_get_memory(c->ctx);
    if (mem) llama_memory_clear(mem, true);
}

extern "C" uint32_t gt_llama_n_ctx(const gt_llama_context * c) {
    return (c && c->ctx) ? llama_n_ctx(c->ctx) : 0;
}

extern "C" int32_t gt_llama_tokenize(const gt_llama_model * m, const char * text, int32_t text_len,
                                     int32_t * out_tokens, int32_t max_tokens, int add_special, int parse_special) {
    if (!m || !m->model || !text) return -1;
    const struct llama_vocab * vocab = llama_model_get_vocab(m->model);
    if (!vocab) return -1;
    return llama_tokenize(vocab, text, text_len, out_tokens, max_tokens, add_special != 0, parse_special != 0);
}

extern "C" int32_t gt_llama_detokenize(const gt_llama_model * m, const int32_t * tokens, int32_t n_tokens,
                                       char * out_buf, int32_t buf_len, int remove_special, int unparse_special) {
    if (!m || !m->model || !tokens || n_tokens <= 0) return 0;
    const struct llama_vocab * vocab = llama_model_get_vocab(m->model);
    if (!vocab) return -1;
    return llama_detokenize(vocab, tokens, n_tokens, out_buf, buf_len, remove_special != 0, unparse_special != 0);
}

extern "C" gt_tensor * gt_llama_forward_logits(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens, int32_t pos_offset, int only_last) {
    if (!c || !c->ctx || !c->model || !tokens || n_tokens <= 0) return nullptr;
    // llama_decode asserts on an oversized batch; reject it rather than aborting
    // the hosting Python process. Also avoid overflowing the context window.
    if ((uint32_t)n_tokens > llama_n_batch(c->ctx) || pos_offset < 0 ||
        (uint64_t)pos_offset + (uint64_t)n_tokens > llama_n_ctx(c->ctx)) return nullptr;
    int32_t n_vocab = gt_llama_n_vocab(c->model);
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_offset + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = only_last ? (i == n_tokens - 1 ? 1 : 0) : 1;
    }
    int32_t rc = llama_decode(c->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) return nullptr;

    int32_t n_out = only_last ? 1 : n_tokens;
    int64_t sh[2] = {n_out, n_vocab};
    gt_tensor * out = gt_tensor_empty(2, sh, GT_F32, GT_DEV_CPU, 0);
    if (!out) return nullptr;
    std::vector<float> buf((size_t)n_out * (size_t)n_vocab);
    if (only_last) {
        const float * lg = llama_get_logits_ith(c->ctx, n_tokens - 1);
        std::memcpy(buf.data(), lg, (size_t)n_vocab * sizeof(float));
    } else {
        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * lg = llama_get_logits_ith(c->ctx, i);
            std::memcpy(buf.data() + (size_t)i * (size_t)n_vocab, lg, (size_t)n_vocab * sizeof(float));
        }
    }
    gt_tensor * filled = gt_tensor_from_data(2, sh, GT_F32, GT_DEV_CPU, 0, buf.data(), buf.size() * sizeof(float));
    gt_tensor_free(out);
    return filled;
}

extern "C" gt_tensor * gt_llama_forward_embeddings(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens) {
    if (!c || !c->ctx || !c->model || !tokens || n_tokens <= 0) return nullptr;
    if ((uint32_t)n_tokens > llama_n_batch(c->ctx) || (uint32_t)n_tokens > llama_n_ctx(c->ctx)) return nullptr;
    gt_llama_clear_kv(c);
    int32_t n_embd = gt_llama_n_embd(c->model);
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    int32_t rc = llama_decode(c->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) return nullptr;

    const float * emb = llama_get_embeddings_seq(c->ctx, 0);
    if (emb) {
        int64_t sh[1] = {n_embd};
        return gt_tensor_from_data(1, sh, GT_F32, GT_DEV_CPU, 0, emb, (size_t)n_embd * sizeof(float));
    }
    int64_t sh[2] = {n_tokens, n_embd};
    std::vector<float> buf((size_t)n_tokens * (size_t)n_embd);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * ei = llama_get_embeddings_ith(c->ctx, i);
        if (ei) std::memcpy(buf.data() + (size_t)i * (size_t)n_embd, ei, (size_t)n_embd * sizeof(float));
    }
    return gt_tensor_from_data(2, sh, GT_F32, GT_DEV_CPU, 0, buf.data(), buf.size() * sizeof(float));
}

extern "C" int gt_llama_finetune(gt_llama_context * c, const int32_t * tokens_data, int64_t n_samples, int64_t seq_len,
                                 int epochs, int64_t batch_size, double lr, double wd, float val_split, int optimizer_type) {
    (void)batch_size;
    if (!c || !c->ctx || !c->model || !c->model->model || !tokens_data || n_samples <= 0 || seq_len <= 1) return -1;

    uint32_t ctx_len = llama_n_ctx(c->ctx);
    static thread_local float tl_lr = 1e-4f, tl_wd = 0.0f;
    tl_lr = (float)lr;
    tl_wd = (float)wd;
    auto get_opt = [](void *) -> struct ggml_opt_optimizer_params {
        struct ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
        p.adamw.alpha = tl_lr;
        p.adamw.wd    = tl_wd;
        p.sgd.alpha   = tl_lr;
        p.sgd.wd      = tl_wd;
        return p;
    };

    struct llama_opt_params lopt = {};
    lopt.n_ctx_train     = ctx_len;
    lopt.param_filter    = llama_opt_param_filter_all;
    lopt.param_filter_ud = nullptr;
    lopt.get_opt_pars    = get_opt;
    lopt.get_opt_pars_ud = nullptr;
    lopt.optimizer_type  = (optimizer_type == 1) ? GGML_OPT_OPTIMIZER_TYPE_SGD : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    llama_opt_init(c->ctx, c->model->model, lopt);

    // llama_context::opt_epoch reads n_ctx tokens and n_ctx labels per sample with ndata_shard = 1
    int32_t eos_tok = std::max<int32_t>(0, gt_llama_vocab_eos(c->model));
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, (int64_t)ctx_len, (int64_t)ctx_len, n_samples, 1);
    struct ggml_tensor * ds_d = ggml_opt_dataset_data(dataset);
    struct ggml_tensor * ds_l = ggml_opt_dataset_labels(dataset);
    int32_t * dp = (int32_t *)ds_d->data;
    int32_t * lp = (int32_t *)ds_l->data;
    for (int64_t s = 0; s < n_samples; ++s) {
        const int32_t * src = tokens_data + s * seq_len;
        int64_t copy_n = std::min<int64_t>(seq_len - 1, (int64_t)ctx_len);
        for (int64_t k = 0; k < (int64_t)ctx_len; ++k) {
            if (k < copy_n) {
                dp[s * (int64_t)ctx_len + k] = src[k];
                lp[s * (int64_t)ctx_len + k] = src[k + 1];
            } else {
                dp[s * (int64_t)ctx_len + k] = eos_tok;
                lp[s * (int64_t)ctx_len + k] = eos_tok;
            }
        }
    }

    int64_t idata_split = (int64_t)((1.0f - val_split) * (float)n_samples);
    if (idata_split < 1) idata_split = n_samples;

    ggml_opt_result_t r_train = ggml_opt_result_init();
    ggml_opt_result_t r_eval  = ggml_opt_result_init();
    for (int e = 0; e < epochs; ++e) {
        ggml_opt_result_reset(r_train);
        ggml_opt_result_reset(r_eval);
        llama_opt_epoch(c->ctx, dataset, r_train, r_eval, idata_split, nullptr, nullptr);
    }
    ggml_opt_result_free(r_train);
    ggml_opt_result_free(r_eval);
    ggml_opt_dataset_free(dataset);
    return 0;
}
