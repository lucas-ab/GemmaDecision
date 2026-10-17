"""MLX port of the decision model's text path (EmbeddingGemma-2 text encoder + Clef head), for Apple silicon.

Mirrors transformers' EmbeddingGemma2TextModel (bidirectional Gemma: per-layer embeddings, q/k/v RMSNorm,
attention scale 1.0, sliding window |i - j| <= 512 on 20 of 24 layers, per-type RoPE) and our ClefHead
(gemma_decision.model). Images and audio stay on the PyTorch path; this covers text inputs.

Weights come from a trained run (base checkpoint + trainable.safetensors), converted once to
runs/<run>/mlx_text.safetensors:

  uv run python -m gemma_decision.mlx_backend runs/broad        # convert (needs torch once)
  model = MLXDecisionModel.load("runs/broad", dtype=mx.float32)  # then torch-free inference
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

WEIGHTS = "mlx_text.safetensors"
NEG_INF = -1e9  # additive mask value; finite so fully-masked rows stay NaN-free


def rms_norm(x, weight=None, eps=1e-6):
    """EmbeddingGemma2RMSNorm: x * (mean(x^2) + eps)^-1/2 (* weight), computed in fp32."""
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    if weight is not None:
        y = y * weight.astype(mx.float32)
    return y.astype(x.dtype)


def layer_norm(x, w, b, eps=1e-5):
    return mx.fast.layer_norm(x, w, b, eps)


class TextEncoder:
    """EmbeddingGemma-2 text backbone: input_ids, attention_mask -> [B, T, 768] token states."""

    def __init__(self, w: dict, cfg: dict, dtype):
        self.w, self.cfg, self.dtype = w, cfg, dtype
        self.n_layers = cfg["num_hidden_layers"]
        self.hidden = cfg["hidden_size"]
        self.ple_dim = cfg["hidden_size_per_layer_input"]
        self.heads = cfg["num_attention_heads"]
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]
        self.layer_types = cfg["layer_types"]
        per = cfg.get("per_layer_config") or {}
        self.head_dim = [per.get(f"{i:02d}", {}).get("head_dim", cfg["head_dim"]) for i in range(self.n_layers)]
        self.kv_heads = [per.get(f"{i:02d}", {}).get("num_key_value_heads", cfg["num_key_value_heads"])
                         for i in range(self.n_layers)]
        self.theta = {t: p["rope_theta"] for t, p in cfg["rope_parameters"].items()}
        self.embed_scale = mx.array(self.hidden ** 0.5, dtype=mx.float32)

    def _lin(self, x, name):
        return x @ self.w[name].T

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        w, (B, T) = self.w, input_ids.shape
        x = (w["embed_tokens.weight"][input_ids].astype(mx.float32) * self.embed_scale).astype(self.dtype)
        ple = self._lin(x, "ple.per_layer_model_projection.weight") * (self.hidden ** -0.5)
        ple = rms_norm(ple.reshape(B, T, self.n_layers, self.ple_dim), w["ple.per_layer_projection_norm.weight"], self.eps)

        # additive masks [B, 1, T, T]: padding keys, and for sliding layers |i - j| <= window
        keep = attention_mask.astype(mx.bool_)[:, None, None, :]
        full = mx.where(keep, 0.0, NEG_INF).astype(self.dtype)
        pos = mx.arange(T)
        near = mx.abs(pos[:, None] - pos[None, :]) <= self.window
        sliding = mx.where(keep & near[None, None], 0.0, NEG_INF).astype(self.dtype) if T > self.window + 1 else full

        for i in range(self.n_layers):
            p = f"layers.{i}."
            hd, kvh = self.head_dim[i], self.kv_heads[i]
            ltype = self.layer_types[i]
            h = rms_norm(x, w[p + "input_layernorm.weight"], self.eps)
            q = self._lin(h, p + "self_attn.q_proj.weight").reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
            k = self._lin(h, p + "self_attn.k_proj.weight").reshape(B, T, kvh, hd).transpose(0, 2, 1, 3)
            v = self._lin(h, p + "self_attn.v_proj.weight").reshape(B, T, kvh, hd).transpose(0, 2, 1, 3)
            q = rms_norm(q, w[p + "self_attn.q_norm.weight"], self.eps)
            k = rms_norm(k, w[p + "self_attn.k_norm.weight"], self.eps)
            v = rms_norm(v, None, self.eps)
            q = mx.fast.rope(q, hd, traditional=False, base=self.theta[ltype], scale=1.0, offset=0)
            k = mx.fast.rope(k, hd, traditional=False, base=self.theta[ltype], scale=1.0, offset=0)
            mask = sliding if ltype == "sliding_attention" else full
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
            a = self._lin(a.transpose(0, 2, 1, 3).reshape(B, T, self.heads * hd), p + "self_attn.o_proj.weight")
            x = x + rms_norm(a, w[p + "post_attention_layernorm.weight"], self.eps)

            h = rms_norm(x, w[p + "pre_feedforward_layernorm.weight"], self.eps)
            h = nn.gelu_approx(self._lin(h, p + "mlp.gate_proj.weight")) * self._lin(h, p + "mlp.up_proj.weight")
            x = x + rms_norm(self._lin(h, p + "mlp.down_proj.weight"), w[p + "post_feedforward_layernorm.weight"], self.eps)

            g = nn.gelu_approx(self._lin(x, p + "ple_block.per_layer_input_gate.weight")) * ple[:, :, i, :]
            g = self._lin(g, p + "ple_block.per_layer_projection.weight")
            x = x + rms_norm(g, w[p + "ple_block.post_per_layer_input_norm.weight"], self.eps)
            x = x * w[p + "layer_scalar"].astype(self.dtype)

        x = rms_norm(x, w["norm.weight"], self.eps)
        return self._lin(x, "embedding_projection.weight")


class VLMEncoder:
    """Full EmbeddingGemma-2 (text + Gemma 4 vision and audio towers) from mlx-vlm (Blaizzy/mlx-vlm, PR #2446),
    with our fine-tuned text weights swapped into its language model. Same output as TextEncoder for text, plus
    image / audio soft tokens placed at their placeholder ids."""

    def __init__(self, enc_w: dict, dtype, model_id: str = "google/embeddinggemma-2"):
        from mlx_vlm import load

        model, _ = load(model_id)
        model.set_dtype(dtype)
        model.language_model.load_weights([(k, v.astype(dtype)) for k, v in enc_w.items()], strict=True)
        mx.eval(model.parameters())
        self.model = model

    def __call__(self, input_ids, attention_mask, **media):
        return self.model(input_ids, attention_mask, **media).last_hidden_state


class ClefHeadMLX:
    """gemma_decision.model.ClefHead (eval mode) on MLX arrays."""

    def __init__(self, w: dict, heads: int = 8, n_cross: int = 2, n_joint: int = 2):
        self.w, self.heads, self.n_cross, self.n_joint = w, heads, n_cross, n_joint

    def _lin(self, x, p):
        y = x @ self.w[p + ".weight"].T
        return y + self.w[p + ".bias"] if p + ".bias" in self.w else y

    def _ln(self, x, p):
        return layer_norm(x, self.w[p + ".weight"], self.w[p + ".bias"])

    def _mha(self, x, mem, pad, p):
        """nn.MultiheadAttention(batch_first) with key_padding_mask=pad (True = ignore)."""
        d = x.shape[-1]
        wq, wk, wv = mx.split(self.w[p + ".in_proj_weight"], 3, axis=0)
        bq, bk, bv = mx.split(self.w[p + ".in_proj_bias"], 3, axis=0)
        B, S, T, H = x.shape[0], x.shape[1], mem.shape[1], self.heads
        q = (x @ wq.T + bq).reshape(B, S, H, d // H).transpose(0, 2, 1, 3)
        k = (mem @ wk.T + bk).reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        v = (mem @ wv.T + bv).reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        mask = mx.where(pad[:, None, None, :], NEG_INF, 0.0).astype(x.dtype)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(d // H), mask=mask)
        return self._lin(a.transpose(0, 2, 1, 3).reshape(B, S, d), p + ".out_proj")

    def __call__(self, tokens, attention_mask, batch: dict, calibrated: bool = True) -> mx.array:
        w = self.w
        mem = self._lin(self._ln(tokens, "tok_proj.0"), "tok_proj.1")
        mem_pad = attention_mask == 0
        slots = batch["slot_weights"] @ mem
        slots = (slots + w["kind_emb.weight"][batch["slot_kind"]] + w["type_emb.weight"][batch["slot_type"]]
                 + w["qid_emb.weight"][batch["slot_qid"]] + w["level_emb.weight"][batch["slot_level"]])
        slot_pad = batch["slot_kind"] == 0
        for i in range(self.n_cross):
            p = f"cross.{i}"
            slots = slots + self._mha(self._ln(slots, p + ".norm_q"), mem, mem_pad, p + ".attn")
            h = nn.gelu(self._lin(self._ln(slots, p + ".norm_ff"), p + ".ff.0"))
            slots = slots + self._lin(h, p + ".ff.3")
        for i in range(self.n_joint):  # TransformerEncoderLayer(norm_first=True, activation=gelu)
            p = f"joint.layers.{i}"
            h = self._ln(slots, p + ".norm1")
            slots = slots + self._mha(h, h, slot_pad, p + ".self_attn")
            slots = slots + self._lin(nn.gelu(self._lin(self._ln(slots, p + ".norm2"), p + ".linear1")), p + ".linear2")
        slots = self._ln(slots, "out_norm")

        flat = slots.reshape(-1, slots.shape[-1])
        opt = flat[batch["q_opt_slots"]]
        q = mx.broadcast_to(flat[batch["q_slot"]][:, None, :], opt.shape)
        x = mx.concatenate([opt, q, opt * q], axis=-1)
        logits = self._lin(nn.gelu(self._lin(x, "scorer.0")), "scorer.3").squeeze(-1)
        if calibrated:
            logits = logits / mx.exp(w["log_temp"])[batch["q_type"]][:, None]
        return mx.where(batch["q_opt_mask"], logits, -mx.inf)


class MLXDecisionModel:
    BATCH_KEYS = ("input_ids", "attention_mask", "slot_weights", "slot_kind", "slot_type", "slot_qid", "slot_level",
                  "q_slot", "q_opt_slots", "q_opt_mask", "q_type")

    MEDIA_KEYS = ("pixel_values", "image_position_ids", "input_features", "input_features_mask")

    def __init__(self, enc_w: dict, head_w: dict, cfg: dict, dtype=mx.float32, encoder: str = "own"):
        """encoder "own": our text-only port (TextEncoder). "vlm": mlx-vlm's full model (text, images, audio)."""
        self.dtype = dtype
        cast = lambda w: {k: (v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in w.items()}  # noqa: E731
        self.encoder = TextEncoder(cast(enc_w), cfg, dtype) if encoder == "own" else VLMEncoder(enc_w, dtype)
        self.multimodal = encoder == "vlm"
        self.head = ClefHeadMLX(cast(head_w))

    @classmethod
    def load(cls, run_dir, dtype=mx.float32, encoder: str = "own"):
        run_dir = Path(run_dir)
        path = run_dir / WEIGHTS
        if not path.exists():
            convert(run_dir)
        w = mx.load(str(path))
        cfg = json.loads((run_dir / "mlx_text_config.json").read_text())
        enc = {k[len("encoder."):]: v for k, v in w.items() if k.startswith("encoder.")}
        head = {k[len("head."):]: v for k, v in w.items() if k.startswith("head.")}
        return cls(enc, head, cfg, dtype, encoder)

    def forward(self, batch: dict, calibrated: bool = True) -> mx.array:
        """`batch` from gemma_decision.data.collate (torch tensors or numpy); returns logits [Q, K] (float32)."""
        b = {k: mx.array(np.asarray(batch[k])) for k in self.BATCH_KEYS}
        b["slot_weights"] = b["slot_weights"].astype(self.dtype)
        media = {k: mx.array(np.asarray(batch[k])) for k in self.MEDIA_KEYS if k in batch}
        if media and not self.multimodal:
            raise ValueError("images/audio need encoder='vlm'")
        for k in ("pixel_values", "input_features"):
            if k in media:
                media[k] = media[k].astype(self.dtype)
        tokens = self.encoder(b["input_ids"], b["attention_mask"], **media)
        return self.head(tokens, b["attention_mask"], b, calibrated).astype(mx.float32)

    def probabilities(self, batch: dict, calibrated: bool = True) -> np.ndarray:
        p = mx.softmax(self.forward(batch, calibrated), axis=-1)
        p = mx.where(mx.isnan(p), 0.0, p)
        mx.eval(p)
        return np.array(p)


def convert(run_dir) -> Path:
    """Write runs/<run>/mlx_text.safetensors (text encoder + head) from the trained PyTorch run."""
    import torch
    import yaml
    from safetensors.torch import load_file

    from .train import build_model

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_model(cfg)
    model.load_state_dict(load_file(run_dir / "trainable.safetensors"), strict=False)
    lm = model.encoder.language_model
    out = {}
    for k, v in lm.state_dict().items():
        if "rotary_emb" in k:
            continue
        out[f"encoder.{k}"] = mx.array(v.detach().to(torch.float32).numpy())
    for k, v in model.head.state_dict().items():
        arr = v.detach().cpu()
        out[f"head.{k}"] = mx.array(arr.numpy() if arr.dtype != torch.float32 else arr.numpy())
    mx.save_safetensors(str(run_dir / WEIGHTS), out)
    tc = lm.config.to_dict()
    keep = ("num_hidden_layers", "hidden_size", "hidden_size_per_layer_input", "num_attention_heads", "head_dim",
            "num_key_value_heads", "rms_norm_eps", "sliding_window", "layer_types", "rope_parameters", "per_layer_config",
            "embedding_dim", "pad_token_id")
    cfg_out = {k: tc[k] for k in keep}
    cfg_out["per_layer_config"] = {k: v for k, v in (tc.get("per_layer_config") or {}).items() if str(k).isdigit()}
    (run_dir / "mlx_text_config.json").write_text(json.dumps(cfg_out, indent=1))
    print(f"wrote {run_dir / WEIGHTS} ({sum(v.size for v in out.values()) / 1e6:.0f}M params)")
    return run_dir / WEIGHTS


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "runs/broad")
