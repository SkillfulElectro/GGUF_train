#include "llama_bridge.h"
#include "llama.h"
#include "ggml-opt.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

struct gt_llama_model {
    struct llama_model * model = nullptr;
    // The optimizer updates the weights in place, which needs them in private writable
    // memory: llama.cpp maps mmap-ed model files read-only.
    bool writable = false;
};

struct gt_llama_context {
    struct llama_context * ctx = nullptr;
    gt_llama_model * model = nullptr;
    int n_threads = 1;
};

static void gt_llama_error(const std::string & msg) {
    gt_set_last_error(msg.c_str());
}

// llama.cpp reports some failures by throwing. An exception must not unwind through the
// C ABI into ctypes (that terminates the hosting Python process), so every entry point
// that can throw converts it into the usual "error return + gt_last_error()" contract.
template <typename R, typename F>
static R gt_llama_guard(const char * what, R fallback, F && fn) {
    try {
        return fn();
    } catch (const std::exception & e) {
        gt_llama_error(std::string(what) + ": " + e.what());
    } catch (...) {
        gt_llama_error(std::string(what) + ": unknown C++ exception");
    }
    return fallback;
}

extern "C" void gt_llama_backend_init(void) {
    gt_init();
    llama_backend_init();
}

extern "C" void gt_llama_backend_free(void) {
    llama_backend_free();
}

extern "C" gt_llama_model * gt_llama_load_model(const char * path, int n_gpu_layers, int vocab_only, int use_mmap) {
    if (!path) {
        gt_llama_error("model path is NULL");
        return nullptr;
    }
    return gt_llama_guard("llama load_model", (gt_llama_model *)nullptr, [&]() -> gt_llama_model * {
        gt_llama_backend_init();
        struct llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;
        mparams.vocab_only = (vocab_only != 0);
        mparams.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        struct llama_model * m = llama_model_load_from_file(path, mparams);
        if (!m) {
            gt_llama_error(std::string("failed to load model from ") + path);
            return nullptr;
        }
        auto * wrap = new gt_llama_model();
        wrap->model = m;
        wrap->writable = !use_mmap && !vocab_only;
        return wrap;
    });
}

extern "C" void gt_llama_free_model(gt_llama_model * m) {
    if (!m) return;
    if (m->model) llama_model_free(m->model);
    delete m;
}

extern "C" int gt_llama_save_model(const gt_llama_model * m, const char * path) {
    if (!m || !m->model || !path) {
        gt_llama_error("invalid arguments");
        return -1;
    }
    return gt_llama_guard("llama save_model", -1, [&]() -> int {
        llama_model_save_to_file(m->model, path);
        return 0;
    });
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
    if (!m || !m->model) {
        gt_llama_error("invalid model");
        return nullptr;
    }
    return gt_llama_guard("llama new_context", (gt_llama_context *)nullptr, [&]() -> gt_llama_context * {
        struct llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx > 0 ? (uint32_t)n_ctx : 2048;
        cparams.n_batch = n_batch > 0 ? (uint32_t)n_batch : cparams.n_ctx;
        int nt = n_threads > 0 ? n_threads : gt_get_num_threads();
        cparams.n_threads = nt;
        cparams.n_threads_batch = nt;
        cparams.embeddings = (embeddings != 0);
        struct llama_context * c = llama_init_from_model(m->model, cparams);
        if (!c) {
            gt_llama_error("llama_init_from_model failed (out of memory, or a model without weights?)");
            return nullptr;
        }
        auto * wrap = new gt_llama_context();
        wrap->ctx = c;
        wrap->model = m;
        wrap->n_threads = nt;
        return wrap;
    });
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
    // INT32_MIN is what llama_tokenize itself returns on overflow; no token count maps to it.
    return gt_llama_guard("llama tokenize", (int32_t)INT32_MIN, [&]() -> int32_t {
        return llama_tokenize(vocab, text, text_len, out_tokens, max_tokens, add_special != 0, parse_special != 0);
    });
}

extern "C" int32_t gt_llama_detokenize(const gt_llama_model * m, const int32_t * tokens, int32_t n_tokens,
                                       char * out_buf, int32_t buf_len, int remove_special, int unparse_special) {
    if (!m || !m->model || !tokens || n_tokens <= 0) return 0;
    const struct llama_vocab * vocab = llama_model_get_vocab(m->model);
    if (!vocab) return 0;
    return gt_llama_guard("llama detokenize", (int32_t)INT32_MIN, [&]() -> int32_t {
        return llama_detokenize(vocab, tokens, n_tokens, out_buf, buf_len, remove_special != 0, unparse_special != 0);
    });
}

// Runs llama_decode on `tokens` placed at positions pos0, pos0+1, ... of sequence 0.
static bool gt_llama_decode_tokens(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens, int32_t pos0, int only_last) {
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = only_last ? (i == n_tokens - 1 ? 1 : 0) : 1;
    }
    int32_t rc = 0;
    try {
        rc = llama_decode(c->ctx, batch);
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
    if (rc != 0) {
        gt_llama_error("llama_decode failed with code " + std::to_string(rc) +
                       (rc == 1 ? " (no free slot in the context window; call clear_kv() or use a larger n_ctx)"
                                : " (a token id outside the vocabulary, or positions that do not continue what the "
                                  "context already holds; see clear_kv())"));
        return false;
    }
    return true;
}

extern "C" gt_tensor * gt_llama_forward_logits(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens, int32_t pos_offset, int only_last) {
    if (!c || !c->ctx || !c->model || !tokens || n_tokens <= 0) {
        gt_llama_error("invalid context or empty token list");
        return nullptr;
    }
    return gt_llama_guard("llama forward_logits", (gt_tensor *)nullptr, [&]() -> gt_tensor * {
        // llama_decode asserts on an oversized batch; reject it rather than aborting
        // the hosting Python process. Also avoid overflowing the context window.
        if ((uint32_t)n_tokens > llama_n_batch(c->ctx)) {
            gt_llama_error("" + std::to_string(n_tokens) + " tokens exceed the context's n_batch (" +
                           std::to_string(llama_n_batch(c->ctx)) + ")");
            return nullptr;
        }
        if (pos_offset < 0 || (uint64_t)pos_offset + (uint64_t)n_tokens > llama_n_ctx(c->ctx)) {
            gt_llama_error("positions " + std::to_string(pos_offset) + ".." +
                           std::to_string((int64_t)pos_offset + n_tokens - 1) + " do not fit the context window (n_ctx = " +
                           std::to_string(llama_n_ctx(c->ctx)) + ")");
            return nullptr;
        }
        const int32_t n_vocab = gt_llama_n_vocab(c->model);
        if (n_vocab <= 0) {
            gt_llama_error("the model has no vocabulary");
            return nullptr;
        }
        if (!gt_llama_decode_tokens(c, tokens, n_tokens, pos_offset, only_last)) return nullptr;

        const int32_t n_out = only_last ? 1 : n_tokens;
        int64_t sh[2] = {n_out, n_vocab};
        gt_tensor * out = gt_tensor_empty(2, sh, GT_F32, GT_DEV_CPU, 0);
        if (!out) return nullptr;
        // Write the rows straight into the result tensor's storage.
        float * dst = (float *)(uintptr_t)gt_tensor_data_ptr(out);
        if (!dst) {
            gt_tensor_free(out);
            gt_llama_error("result tensor has no host-accessible storage");
            return nullptr;
        }
        for (int32_t i = 0; i < n_out; ++i) {
            const int32_t idx = only_last ? n_tokens - 1 : i;
            // NULL when the context produced no logits for this position (e.g. an
            // embeddings-only/pooled context).
            const float * lg = llama_get_logits_ith(c->ctx, idx);
            if (!lg) {
                gt_tensor_free(out);
                gt_llama_error("llama.cpp returned no logits for token " + std::to_string(idx) +
                               " (was the context created with embeddings=True?)");
                return nullptr;
            }
            std::memcpy(dst + (size_t)i * (size_t)n_vocab, lg, (size_t)n_vocab * sizeof(float));
        }
        return out;
    });
}

extern "C" gt_tensor * gt_llama_forward_embeddings(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens) {
    if (!c || !c->ctx || !c->model || !tokens || n_tokens <= 0) {
        gt_llama_error("invalid context or empty token list");
        return nullptr;
    }
    return gt_llama_guard("llama forward_embeddings", (gt_tensor *)nullptr, [&]() -> gt_tensor * {
        if ((uint32_t)n_tokens > llama_n_batch(c->ctx) || (uint32_t)n_tokens > llama_n_ctx(c->ctx)) {
            gt_llama_error("" + std::to_string(n_tokens) + " tokens exceed the context's n_batch/n_ctx (" +
                           std::to_string(llama_n_batch(c->ctx)) + "/" + std::to_string(llama_n_ctx(c->ctx)) + ")");
            return nullptr;
        }
        gt_llama_clear_kv(c);
        const int32_t n_embd = gt_llama_n_embd(c->model);
        if (n_embd <= 0) {
            gt_llama_error("the model reports no embedding size");
            return nullptr;
        }
        if (!gt_llama_decode_tokens(c, tokens, n_tokens, 0, 0)) return nullptr;

        const float * emb = llama_get_embeddings_seq(c->ctx, 0);
        if (emb) {
            int64_t sh[1] = {n_embd};
            return gt_tensor_from_data(1, sh, GT_F32, GT_DEV_CPU, 0, emb, (size_t)n_embd * sizeof(float));
        }
        int64_t sh[2] = {n_tokens, n_embd};
        gt_tensor * out = gt_tensor_empty(2, sh, GT_F32, GT_DEV_CPU, 0);
        if (!out) return nullptr;
        float * dst = (float *)(uintptr_t)gt_tensor_data_ptr(out);
        if (!dst) {
            gt_tensor_free(out);
            gt_llama_error("result tensor has no host-accessible storage");
            return nullptr;
        }
        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * ei = llama_get_embeddings_ith(c->ctx, i);
            if (!ei) {
                gt_tensor_free(out);
                gt_llama_error("llama.cpp returned no embedding for token " + std::to_string(i) +
                               " (was the context created with embeddings=True?)");
                return nullptr;
            }
            std::memcpy(dst + (size_t)i * (size_t)n_embd, ei, (size_t)n_embd * sizeof(float));
        }
        return out;
    });
}

// ---------------------------------------------------------------------------
// Fine-tuning
// ---------------------------------------------------------------------------

namespace {

struct gt_llama_opt_hparams {
    float lr = 1e-4f;
    float wd = 0.0f;
};

struct ggml_opt_optimizer_params gt_llama_opt_pars(void * userdata) {
    const auto * hp = static_cast<const gt_llama_opt_hparams *>(userdata);
    struct ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = hp->lr;
    p.adamw.wd    = hp->wd;
    p.sgd.alpha   = hp->lr;
    p.sgd.wd      = hp->wd;
    return p;
}

bool gt_llama_param_filter_none(const struct ggml_tensor *, void *) {
    return false;
}

struct gt_llama_ctx_holder {
    struct llama_context * ctx = nullptr;
    ~gt_llama_ctx_holder() { reset(); }
    void reset() {
        if (ctx) llama_free(ctx);
        ctx = nullptr;
    }
};

struct gt_llama_dataset_holder {
    ggml_opt_dataset_t dataset = nullptr;
    ggml_opt_result_t r_train = nullptr;
    ggml_opt_result_t r_eval = nullptr;
    ~gt_llama_dataset_holder() {
        if (r_train) ggml_opt_result_free(r_train);
        if (r_eval) ggml_opt_result_free(r_eval);
        if (dataset) ggml_opt_dataset_free(dataset);
    }
};

// llama_opt_init() overwrites the model's n_ctx_train ("context size post training") and
// llama.cpp then uses that field as the number of tokens it trains on per sample. It is
// also what llama_model_n_ctx_train() reports and what llama_model_save_to_file() writes
// as the model's context length, so it has to be put back once training is over. The
// public API offers no setter other than llama_opt_init() itself, so the value is restored
// by initialising the optimizer of a minimal throw-away context with no trainable tensors.
void gt_llama_log_discard(enum ggml_log_level, const char *, void *) {}

// Silences llama.cpp's and ggml's loggers for its lifetime.
//
// llama_log_set() replaces both loggers at once, and only ggml's can be queried. ggml's
// is the one gguf_train installs; llama.cpp's is left at its default by this library, so
// that is what it is put back to.
struct gt_llama_log_mute {
    ggml_log_callback ggml_cb = nullptr;
    void * ggml_ud = nullptr;
    gt_llama_log_mute() {
        ggml_log_get(&ggml_cb, &ggml_ud);
        llama_log_set(gt_llama_log_discard, nullptr);
    }
    ~gt_llama_log_mute() {
        llama_log_set(nullptr, nullptr);
        ggml_log_set(ggml_cb, ggml_ud);
    }
};

bool gt_llama_restore_n_ctx_train(struct llama_model * model, int32_t n_ctx_train, int n_threads) {
    if (llama_model_n_ctx_train(model) == n_ctx_train) return true;
    if (n_ctx_train <= 0) return false;
    // The helper context trains nothing; llama.cpp's messages about its (irrelevant)
    // training configuration would only mislead.
    gt_llama_log_mute mute;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1;  // llama.cpp rounds this up to its minimum context size
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.embeddings = false;
    gt_llama_ctx_holder helper;
    helper.ctx = llama_init_from_model(model, cp);
    if (!helper.ctx) return false;
    // llama_opt_init aborts unless n_ctx_train % n_batch == 0 and n_batch % n_ubatch == 0.
    if (llama_n_batch(helper.ctx) != 1 || llama_n_ubatch(helper.ctx) != 1) return false;
    struct llama_opt_params lopt = {};
    lopt.n_ctx_train     = (uint32_t)n_ctx_train;
    lopt.param_filter    = gt_llama_param_filter_none;
    lopt.param_filter_ud = nullptr;
    lopt.get_opt_pars    = ggml_opt_get_default_optimizer_params;
    lopt.get_opt_pars_ud = nullptr;
    lopt.optimizer_type  = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    llama_opt_init(helper.ctx, model, lopt);
    helper.reset();
    return llama_model_n_ctx_train(model) == n_ctx_train;
}

} // namespace

extern "C" int gt_llama_finetune(gt_llama_context * c, const int32_t * tokens_data, int64_t n_samples, int64_t seq_len,
                                 int epochs, int64_t batch_size, double lr, double wd, float val_split, int optimizer_type) {
    if (!c || !c->ctx || !c->model || !c->model->model) {
        gt_llama_error("invalid context");
        return -1;
    }
    if (!tokens_data || n_samples <= 0 || seq_len < 2) {
        gt_llama_error("tokens must hold at least one sequence of two or more tokens");
        return -1;
    }
    // Everything below is checked up front because llama.cpp/ggml enforce these conditions
    // with GGML_ASSERT, i.e. by aborting the whole process.
    if (!c->model->writable) {
        gt_llama_error("the model weights are read-only (memory-mapped or vocab-only); "
                       "load the model with use_mmap=False to fine-tune it");
        return -1;
    }
    if (batch_size != 1) {
        gt_llama_error("batch_size must be 1; llama.cpp's trainer performs one optimizer step per "
                       "sequence and cannot combine several sequences into one step");
        return -1;
    }
    if (!(lr > 0.0) || !std::isfinite(lr) || !((float)lr > 0.0f)) {
        gt_llama_error("lr must be a positive finite number");
        return -1;
    }
    if (!(wd >= 0.0 && wd <= 1.0)) {
        gt_llama_error("weight decay must be in [0, 1]");
        return -1;
    }
    if (!(val_split >= 0.0f && val_split < 1.0f)) {
        gt_llama_error("val_split must be in [0, 1)");
        return -1;
    }
    if (optimizer_type != 0 && optimizer_type != 1) {
        gt_llama_error("unknown optimizer type");
        return -1;
    }
    if (n_samples > INT32_MAX || seq_len > INT32_MAX) {
        gt_llama_error("dataset is too large");
        return -1;
    }
    struct llama_model * model = c->model->model;
    const int32_t n_vocab = gt_llama_n_vocab(c->model);
    for (int64_t i = 0; i < n_samples * seq_len; ++i) {
        if (tokens_data[i] < 0 || tokens_data[i] >= n_vocab) {
            gt_llama_error("token id " + std::to_string(tokens_data[i]) + " (sequence " +
                           std::to_string(i / seq_len) + ", position " + std::to_string(i % seq_len) +
                           ") is outside the vocabulary [0, " + std::to_string(n_vocab) + ")");
            return -1;
        }
    }
    if (epochs <= 0) return 0;

    // Sequences longer than the context's window are truncated to it, as before.
    const int64_t usable = std::min<int64_t>(seq_len - 1, (int64_t)llama_n_ctx(c->ctx));
    const int32_t n_ctx_train_saved = llama_model_n_ctx_train(model);

    int rc = gt_llama_guard("llama finetune", -1, [&]() -> int {
        // Declared before the training context so that it outlives the optimizer using it.
        gt_llama_opt_hparams hp;
        hp.lr = (float)lr;
        hp.wd = (float)wd;

        // Training runs in a context of its own, sized to the sequences:
        //  - llama_opt_init() can run only once per context and leaves it unusable for
        //    llama_decode(), so the caller's context must not be used for it (and finetune()
        //    stays callable any number of times);
        //  - the whole sequence has to be a single ubatch for it to be trained as one
        //    independent sequence.
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = (uint32_t)usable;  // llama.cpp rounds this up to a multiple of 256
        cp.n_batch         = (uint32_t)usable;
        cp.n_ubatch        = (uint32_t)usable;
        cp.n_seq_max       = 1;
        cp.n_threads       = c->n_threads;
        cp.n_threads_batch = c->n_threads;
        cp.embeddings      = false;
        // ggml has no backward pass for an f16 KV cache (see llama.cpp's finetune example).
        cp.type_k          = GGML_TYPE_F32;
        cp.type_v          = GGML_TYPE_F32;

        gt_llama_ctx_holder train;
        train.ctx = llama_init_from_model(model, cp);
        if (!train.ctx) {
            gt_llama_error("could not create the training context (out of memory?)");
            return -1;
        }
        const int64_t window = (int64_t)llama_n_ctx(train.ctx);
        if (window < usable || (int64_t)llama_n_batch(train.ctx) != usable || (int64_t)llama_n_ubatch(train.ctx) != usable) {
            gt_llama_error("llama.cpp created the training context with unexpected batch sizes");
            return -1;
        }

        // One sample is one row of `window` tokens, because llama_opt_epoch() fetches n_ctx
        // tokens per sample. Only the first `usable` of them hold data; llama.cpp is told to
        // train on exactly those through n_ctx_train (see gt_llama_restore_n_ctx_train), so
        // the filler behind them is never fed to the model, let alone trained on.
        gt_llama_dataset_holder data;
        data.dataset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, window, window, n_samples, 1);
        if (!data.dataset) {
            gt_llama_error("could not allocate the dataset");
            return -1;
        }
        int32_t * dp = (int32_t *)ggml_opt_dataset_data(data.dataset)->data;
        int32_t * lp = (int32_t *)ggml_opt_dataset_labels(data.dataset)->data;
        for (int64_t s = 0; s < n_samples; ++s) {
            const int32_t * src = tokens_data + s * seq_len;
            for (int64_t k = 0; k < window; ++k) {
                const int64_t j = std::min<int64_t>(k, usable - 1);
                dp[s * window + k] = src[j];
                lp[s * window + k] = src[j + 1];
            }
        }

        struct llama_opt_params lopt = {};
        lopt.n_ctx_train     = (uint32_t)usable;
        lopt.param_filter    = llama_opt_param_filter_all;
        lopt.param_filter_ud = nullptr;
        lopt.get_opt_pars    = gt_llama_opt_pars;
        lopt.get_opt_pars_ud = &hp;
        lopt.optimizer_type  = (optimizer_type == 1) ? GGML_OPT_OPTIMIZER_TYPE_SGD : GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        llama_opt_init(train.ctx, model, lopt);
        if ((int64_t)llama_model_n_ctx_train(model) != usable) {
            // Not the llama.cpp behaviour this code was written against: training now would
            // run over the filler tokens.
            gt_llama_error("this llama.cpp revision does not take the training length from n_ctx_train");
            return -1;
        }

        int64_t idata_split = (int64_t)((1.0f - val_split) * (float)n_samples);
        if (idata_split < 1 || idata_split > n_samples) idata_split = n_samples;

        data.r_train = ggml_opt_result_init();
        data.r_eval  = ggml_opt_result_init();
        for (int e = 0; e < epochs; ++e) {
            ggml_opt_result_reset(data.r_train);
            ggml_opt_result_reset(data.r_eval);
            llama_opt_epoch(train.ctx, data.dataset, data.r_train, data.r_eval, idata_split, nullptr, nullptr);
        }
        return 0;
    });

    // The training context is gone by now (also after a failure part-way through).
    const bool restored = gt_llama_guard("llama finetune", false, [&]() -> bool {
        return gt_llama_restore_n_ctx_train(model, n_ctx_train_saved, c->n_threads);
    });
    // Anything cached in the caller's context was computed with the old weights.
    gt_llama_clear_kv(c);
    if (rc == 0 && !restored) {
        gt_llama_error("training finished, but the model's context length could not be restored (it now reads " +
                       std::to_string(llama_model_n_ctx_train(model)) + " instead of " + std::to_string(n_ctx_train_saved) +
                       "); reload the model before saving it");
        return -2;
    }
    return rc;
}
