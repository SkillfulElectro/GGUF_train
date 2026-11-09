"""Comprehensive tests for GGUF serialization, ggml quantization, LoRA, OptGraph, and llama.cpp bridge."""

import os
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.gguf import (
    dequantize_state_dict,
    dequantize_tensor,
    inspect_gguf,
    load_gguf,
    quantize_gguf,
    quantize_state_dict,
    quantize_tensor,
    save_gguf,
)
from torch.llama import LlamaContext, LoRALinear, OptGraph, inject_lora, merge_lora


class TestGGUFAndLlama(unittest.TestCase):
    def test_finetune_rejects_invalid_input_before_native_call(self):
        context = LlamaContext.__new__(LlamaContext)
        for rows in ([], [[1]], [[1, 2], [1]]):
            with self.assertRaises(ValueError):
                context.finetune(rows)
        with self.assertRaises(ValueError):
            context.finetune([[1, 2]], val_split=1.0)
        with self.assertRaises(ValueError):
            context.finetune([[1, 2]], optimizer="unknown")

    def test_gguf_save_load_inspect_and_torch_save(self):
        torch.manual_seed(1)
        tensors = {
            "w1": torch.randn(16, 32),
            "b1": torch.randn(16),
            "w2": torch.randn(8, 32),
        }
        meta = {"general.architecture": "mlp", "model.dim": 32, "model.active": True}
        with tempfile.TemporaryDirectory() as tmp:
            gpath = os.path.join(tmp, "weights.gguf")
            save_gguf(gpath, tensors, metadata=meta)
            info = inspect_gguf(gpath)
            self.assertEqual(info["n_tensors"], 3)
            self.assertEqual(info["metadata"]["general.architecture"], "mlp")

            loaded, lmeta = load_gguf(gpath, return_metadata=True)
            self.assertTrue(loaded["w1"].allclose(tensors["w1"], atol=1e-6))
            self.assertTrue(loaded["b1"].allclose(tensors["b1"], atol=1e-6))
            self.assertEqual(lmeta["general.architecture"], "mlp")

            # torch.save / torch.load for BOTH .gguf and .pt
            t_gguf = os.path.join(tmp, "sd.gguf")
            t_pt = os.path.join(tmp, "sd.pt")
            torch.save(tensors, t_gguf)
            torch.save({"state": tensors, "epoch": 7}, t_pt)
            l_gguf = torch.load(t_gguf)
            l_pt = torch.load(t_pt)
            self.assertTrue(l_gguf["w2"].allclose(tensors["w2"], atol=1e-6))
            self.assertEqual(l_pt["epoch"], 7)
            self.assertTrue(l_pt["state"]["w1"].allclose(tensors["w1"], atol=1e-6))

    def test_all_quant_formats_and_quantize_gguf(self):
        torch.manual_seed(2)
        # Multiple of 256 for superblock k-quants (q2_k, q3_k, q4_k, q5_k, q6_k)
        w = torch.randn(4, 256)
        for qdt in [
            torch.q4_0, torch.q4_1, torch.q5_0, torch.q5_1, torch.q8_0,
            torch.q2_k, torch.q3_k, torch.q4_k, torch.q5_k, torch.q6_k,
        ]:
            qw = quantize_tensor(w, qdt)
            self.assertTrue(qw.dtype.is_quantized)
            dw = dequantize_tensor(qw, torch.float32)
            self.assertEqual(dw.shape, w.shape)
            # Cosine similarity high even at 2-bit/4-bit
            cos = torch.cosine_similarity(w.reshape((-1,)).unsqueeze(0), dw.reshape((-1,)).unsqueeze(0)).item()
            self.assertGreater(cos, 0.85, msg=f"low cosine similarity for {qdt}: {cos}")

        sd = {"blk.0.attn_q.weight": torch.randn(64, 256), "blk.0.attn_norm.weight": torch.ones(256)}
        qsd = quantize_state_dict(sd, qtype="q4_0")
        self.assertEqual(qsd["blk.0.attn_q.weight"].dtype, torch.q4_0)
        self.assertEqual(qsd["blk.0.attn_norm.weight"].dtype, torch.float32)
        dsd = dequantize_state_dict(qsd)
        self.assertEqual(dsd["blk.0.attn_q.weight"].dtype, torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "f32.gguf")
            dst = os.path.join(tmp, "q8.gguf")
            save_gguf(src, sd)
            quantize_gguf(src, dst, qtype="q8_0")
            q_loaded = load_gguf(dst, dequantize=False)
            self.assertEqual(q_loaded["blk.0.attn_q.weight"].dtype, torch.q8_0)

    def test_quantized_linear_and_lora_finetuning_and_merge(self):
        torch.manual_seed(17)
        base_model = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 8),
        )
        # Quantize base weights to q4_0 & q8_0!
        base_model[0].quantize_weight_(torch.q4_0)
        base_model[2].quantize_weight_(torch.q8_0)

        # Inject LoRA adapters on quantized Linear layers
        lora_model = inject_lora(base_model, rank=4, alpha=8.0)
        self.assertIsInstance(lora_model[0], LoRALinear)
        self.assertIsInstance(lora_model[2], LoRALinear)

        trainable = [p for p in lora_model.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 4)  # 2x (lora_A, lora_B)

        X = torch.randn(16, 64)
        target = torch.randn(16, 8)
        opt = optim.AdamW(trainable, lr=5e-2)
        init_loss = float(((lora_model(X) - target) ** 2).mean().item())
        for _ in range(40):
            opt.zero_grad()
            loss = ((lora_model(X) - target) ** 2).mean()
            loss.backward()
            opt.step()
        final_loss = float(((lora_model(X) - target) ** 2).mean().item())
        self.assertLess(final_loss, init_loss * 0.25)

        # Merge LoRA back into standard Linear & verify predictions closely match
        # (quantized ggml_mul_mat uses Q8_0 activation quant vs dequantized F32 post-merge)
        lora_model.eval()
        pred_before_merge = lora_model(X).detach()
        merged = merge_lora(lora_model)
        self.assertIsInstance(merged[0], nn.Linear)
        pred_after_merge = merged(X).detach()
        self.assertTrue(pred_before_merge.allclose(pred_after_merge, atol=2e-2))

        # Exact float32 base LoRA merge matches to 1e-5
        f32_lora = inject_lora(nn.Sequential(nn.Linear(16, 8)), rank=4, alpha=8.0)
        with torch.no_grad():
            f32_lora[0].lora_B.normal_(0.0, 0.2)
        x_s = torch.randn(4, 16)
        p_pre = f32_lora(x_s).detach()
        p_post = merge_lora(f32_lora)(x_s).detach()
        self.assertTrue(p_pre.allclose(p_post, atol=1e-5))

    def test_native_ggml_opt_graph_mse_and_cross_entropy(self):
        torch.manual_seed(5)
        # 1. MSE linear regression via static ggml_opt_fit
        X = torch.randn(64, 4)
        w_true = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
        y = X @ w_true.t()

        g = OptGraph(loss_type="mse", optimizer="adamw", nbatch_physical=16, nbatch_logical=16)
        x_id = g.input((16, 4))
        W = torch.randn(1, 4, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        w_id = g.param(W)
        b_id = g.param(b)
        out_id = g.linear(x_id, w_id, b_id)

        train_loss, val_loss = g.fit(out_id, X, y, epochs=120, lr=0.1, val_split=0.2, silent=True)
        self.assertLess(train_loss, 1e-4)
        self.assertLess(val_loss, 1e-4)
        self.assertTrue(W.allclose(w_true, atol=0.05))

        # 2. Cross-entropy 2-layer MLP via static ggml_opt_fit
        X_cls = torch.randn(64, 4)
        logits_true = X_cls @ torch.randn(4, 3)
        y_onehot = torch.zeros(64, 3)
        idx = logits_true.argmax(dim=-1)
        for i, c in enumerate(idx.tolist()):
            y_onehot[i, int(c)] = 1.0

        g2 = OptGraph(loss_type="cross_entropy", optimizer="adamw", nbatch_physical=16, nbatch_logical=16)
        x2 = g2.input((16, 4))
        w1 = torch.randn(16, 4) * 0.2
        b1 = torch.zeros(16)
        w2 = torch.randn(3, 16) * 0.2
        b2 = torch.zeros(3)
        h = g2.unary(g2.linear(x2, g2.param(w1), g2.param(b1)), "silu")
        out2 = g2.linear(h, g2.param(w2), g2.param(b2))
        ce_loss, _ = g2.fit(out2, X_cls, y_onehot, epochs=35, lr=0.05, silent=True)
        self.assertLess(ce_loss, 0.35)


if __name__ == "__main__":
    unittest.main()
