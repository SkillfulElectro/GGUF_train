#pragma once
#include "gguf_train.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gt_llama_model gt_llama_model;
typedef struct gt_llama_context gt_llama_context;

GT_API void gt_llama_backend_init(void);
GT_API void gt_llama_backend_free(void);

GT_API gt_llama_model * gt_llama_load_model(const char * path, int n_gpu_layers, int vocab_only, int use_mmap);
GT_API void             gt_llama_free_model(gt_llama_model * m);
GT_API int              gt_llama_save_model(const gt_llama_model * m, const char * path);

GT_API int32_t gt_llama_n_vocab(const gt_llama_model * m);
GT_API int32_t gt_llama_n_ctx_train(const gt_llama_model * m);
GT_API int32_t gt_llama_n_embd(const gt_llama_model * m);
GT_API int32_t gt_llama_n_layer(const gt_llama_model * m);
GT_API int32_t gt_llama_n_head(const gt_llama_model * m);
GT_API int32_t gt_llama_n_head_kv(const gt_llama_model * m);
GT_API uint64_t gt_llama_model_n_params(const gt_llama_model * m);
GT_API uint64_t gt_llama_model_size(const gt_llama_model * m);
GT_API int32_t gt_llama_vocab_bos(const gt_llama_model * m);
GT_API int32_t gt_llama_vocab_eos(const gt_llama_model * m);
GT_API int32_t gt_llama_vocab_pad(const gt_llama_model * m);
GT_API int32_t gt_llama_model_desc(const gt_llama_model * m, char * buf, size_t buf_len);
GT_API int32_t gt_llama_model_meta_count(const gt_llama_model * m);
GT_API int32_t gt_llama_model_meta_key(const gt_llama_model * m, int32_t idx, char * buf, size_t buf_len);
GT_API int32_t gt_llama_model_meta_val(const gt_llama_model * m, const char * key, char * buf, size_t buf_len);

GT_API gt_llama_context * gt_llama_new_context(gt_llama_model * m, int n_ctx, int n_batch, int n_threads, int embeddings);
GT_API void               gt_llama_free_context(gt_llama_context * c);
GT_API void               gt_llama_clear_kv(gt_llama_context * c);
GT_API uint32_t           gt_llama_n_ctx(const gt_llama_context * c);

GT_API int32_t gt_llama_tokenize(const gt_llama_model * m, const char * text, int32_t text_len,
                                 int32_t * out_tokens, int32_t max_tokens, int add_special, int parse_special);
GT_API int32_t gt_llama_detokenize(const gt_llama_model * m, const int32_t * tokens, int32_t n_tokens,
                                   char * out_buf, int32_t buf_len, int remove_special, int unparse_special);

// Forward decode on tokens -> returns gt_tensor of shape (n_tokens, n_vocab) F32 (or last-token (1, n_vocab) if only_last != 0)
GT_API gt_tensor * gt_llama_forward_logits(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens, int32_t pos_offset, int only_last);
// Forward embeddings on tokens -> returns gt_tensor of shape (1, n_embd) or (n_tokens, n_embd) F32
GT_API gt_tensor * gt_llama_forward_embeddings(gt_llama_context * c, const int32_t * tokens, int32_t n_tokens);

// Fine-tune a GGUF model via llama_opt_init + llama_opt_epoch on tokenized dataset
// tokens_data: array of n_samples * n_ctx_seq int32 tokens (each sequence of length seq_len)
GT_API int gt_llama_finetune(gt_llama_context * c, const int32_t * tokens_data, int64_t n_samples, int64_t seq_len,
                             int epochs, int64_t batch_size, double lr, double wd, float val_split, int optimizer_type);

#ifdef __cplusplus
}
#endif
