"""EmbeddingGemma-2 encoder + Clef-style joint decision head.

Encoder: the full EmbeddingGemma-2 model (text + vision + audio). Its `last_hidden_state` is per token,
already normed and projected 512 -> 768 (the model card's mean pooling happens outside, in
sentence-transformers), so the head reads 768-d token states directly.

Head (our reading of Clef's "joint schema head"; Cloudflare has not published the internals):
  1. span pooling     each question and each option -> mean of its token states
  2. cross-attention  every question/option slot attends to all tokens (option reads the state in context)
  3. joint mixer      self-attention across all slots of all questions, so decisions are made together
  4. scorer           logit per option from [option, question, option*question]; softmax within the question
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import KIND_PAD, TYPES

MODEL_ID = "google/embeddinggemma-2"
LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def load_encoder(mode: str, model_id: str = MODEL_ID, lora: dict | None = None, attn_implementation: str = "sdpa"):
    """mode: "full" | "lora" | "frozen". Vision/audio towers and token embeddings are always frozen."""
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(model_id, dtype=torch.float32, attn_implementation=attn_implementation)
    for p in enc.parameters():
        p.requires_grad = False

    if mode == "full":
        for name, p in enc.language_model.named_parameters():
            p.requires_grad = not name.startswith("embed_tokens")
    elif mode == "lora":
        from peft import LoraConfig, inject_adapter_in_model

        lora = lora or {}
        cfg = LoraConfig(
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.05),
            target_modules=LORA_TARGETS,
        )
        enc = inject_adapter_in_model(cfg, enc)
        for name, p in enc.named_parameters():
            p.requires_grad = "lora_" in name
    elif mode != "frozen":
        raise ValueError(f"unknown encoder mode {mode!r}")
    return enc


class CrossBlock(nn.Module):
    """Pre-norm block: slots attend to token states, then FFN."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, mem_pad):
        h = self.norm_q(x)
        x = x + self.drop(self.attn(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0])
        return x + self.drop(self.ff(self.norm_ff(x)))


class ClefHead(nn.Module):
    def __init__(
        self, d_in: int = 768, d: int = 512, heads: int = 8, n_cross: int = 2, n_joint: int = 2,
        max_questions: int = 64, max_levels: int = 16, dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.kind_emb = nn.Embedding(3, d)
        self.type_emb = nn.Embedding(len(TYPES), d)
        self.qid_emb = nn.Embedding(max_questions, d)
        self.level_emb = nn.Embedding(max_levels + 1, d)
        self.cross = nn.ModuleList([CrossBlock(d, heads, dropout) for _ in range(n_cross)])
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.joint = nn.TransformerEncoder(layer, n_joint, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.scorer = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        # Post-hoc calibration: one log-temperature per question type (fitted on held-out data, not trained).
        self.register_buffer("log_temp", torch.zeros(len(TYPES)))

    def forward(self, tokens, attention_mask, batch, calibrated: bool = True):
        """Returns logits [Q, K] (masked options = -inf) for every question in the batch."""
        mem = self.tok_proj(tokens)  # [B, T, d]
        mem_pad = attention_mask == 0

        slots = torch.bmm(batch["slot_weights"], mem)  # span mean-pooling -> [B, S, d]
        slots = (
            slots
            + self.kind_emb(batch["slot_kind"])
            + self.type_emb(batch["slot_type"])
            + self.qid_emb(batch["slot_qid"])
            + self.level_emb(batch["slot_level"])
        )
        slot_pad = batch["slot_kind"] == KIND_PAD
        for blk in self.cross:
            slots = blk(slots, mem, mem_pad)
        slots = self.out_norm(self.joint(slots, src_key_padding_mask=slot_pad))

        flat = slots.reshape(-1, slots.shape[-1])  # [B*S, d]
        opt = flat[batch["q_opt_slots"]]  # [Q, K, d]
        q = flat[batch["q_slot"]].unsqueeze(1).expand_as(opt)
        logits = self.scorer(torch.cat([opt, q, opt * q], dim=-1)).squeeze(-1)  # [Q, K]
        if calibrated:
            logits = logits / self.log_temp.exp()[batch["q_type"]].unsqueeze(1)
        return logits.masked_fill(~batch["q_opt_mask"], float("-inf"))


class DecisionModel(nn.Module):
    def __init__(self, encoder, head: ClefHead, mode: str):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.mode = mode

    def encode_tokens(self, input_ids, attention_mask, **mm):
        if self.mode == "frozen":
            with torch.no_grad():
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask, **mm).last_hidden_state
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, **mm).last_hidden_state

    def forward(self, batch, calibrated: bool = True, **mm):
        tokens = self.encode_tokens(batch["input_ids"], batch["attention_mask"], **mm)
        return self.head(tokens, batch["attention_mask"], batch, calibrated=calibrated)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Everything that differs from the base checkpoint: trainable encoder params + full head."""
        sd = {f"encoder.{n}": p.detach().cpu() for n, p in self.encoder.named_parameters() if p.requires_grad}
        sd.update({f"head.{n}": t.detach().cpu() for n, t in self.head.state_dict().items()})
        return sd


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).nan_to_num(0.0)
