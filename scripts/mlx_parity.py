"""Check the MLX port against the PyTorch model on real inputs (S1MB + typed-decisions test rows).

  uv run python scripts/mlx_parity.py runs/broad
Reports token-state and probability differences, top-1 agreement, for MLX fp32 / bf16 / fp16.
"""
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from s1mb_bench import load_cases, to_examples  # noqa: E402

from gemma_decision.data import collate, encode, load_examples  # noqa: E402
from gemma_decision.mlx_backend import MLXDecisionModel  # noqa: E402
from gemma_decision.model import probabilities, to_device  # noqa: E402
from gemma_decision.train import load_checkpoint  # noqa: E402

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/broad")
model, tok, cfg, device = load_checkpoint(run_dir)
model.eval()
spec = json.loads(Path("data/s1mb/english-v1/benchmarks.json").read_text())
rng = random.Random(0)
exs = []
for b in rng.sample(spec["benchmarks"], 40):
    case = rng.choice(load_cases(b["dataset"].split("/", 1)[1], b["task"]))
    exs.append(to_examples(case)[0][0])
exs += rng.sample(load_examples(split="test"), 20)
batches = []
for ex in exs:
    b = collate([encode(tok, ex.state, ex.questions, 2048)], tok.pad_token_id)
    b.pop("q_meta")
    batches.append(b)
# one padded multi-row batch too (exercises the padding masks)
bb = collate([encode(tok, ex.state, ex.questions, 2048) for ex in exs[:4]], tok.pad_token_id)
bb.pop("q_meta")
batches.append(bb)

with torch.inference_mode():
    ref = [probabilities(model(to_device(b, device)).float()).cpu().numpy() for b in batches]
    b0 = to_device(batches[0], device)
    ref_tokens = model.encode_tokens(b0["input_ids"], b0["attention_mask"]).float().cpu().numpy()
lengths = [b["input_ids"].shape[1] for b in batches]
print(f"{len(batches)} inputs, {min(lengths)}-{max(lengths)} tokens")
for name, dt in (("fp32", mx.float32), ("bf16", mx.bfloat16), ("fp16", mx.float16)):
    m = MLXDecisionModel.load(run_dir, dt)
    tokens = m.encoder(mx.array(batches[0]["input_ids"].numpy()), mx.array(batches[0]["attention_mask"].numpy()))
    tokens = np.array(tokens.astype(mx.float32))
    cos = (tokens * ref_tokens).sum(-1) / (np.linalg.norm(tokens, axis=-1) * np.linalg.norm(ref_tokens, axis=-1))
    diffs, agree, n = [], 0, 0
    for b, r in zip(batches, ref):
        p = m.probabilities(b)
        mask = b["q_opt_mask"].numpy()
        diffs.append(np.abs(np.where(mask, p - r, 0)).max())
        agree += int((np.where(mask, p, -1).argmax(-1) == np.where(mask, r, -1).argmax(-1)).sum())
        n += mask.shape[0]
    print(f"MLX {name}: token states max |d| {np.abs(tokens - ref_tokens).max():.2e}, min cosine {cos.min():.6f} | "
          f"probability max |d| {max(diffs):.2e} (median {np.median(diffs):.1e}) | top-1 agreement {agree}/{n}")
