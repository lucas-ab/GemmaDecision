"""Check the MLX (mlx-vlm) multimodal path against PyTorch on held-out pilot test rows with images and audio.

  uv run python scripts/mlx_parity_media.py runs/broad [--per-source 6]
Both backends get the same processor output (collate_mm); reports probability differences and top-1 agreement
per source, plus text rows through the same mlx-vlm model, for fp32 and fp16.
Timings are median per-row latency in ms; both backends get one untimed warm-up call on the first batch of each
source (so neither pays first-call kernel compilation) and every timed call is synchronized.
"""
import argparse
import statistics as st
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoProcessor

from gemma_decision.data import collate_mm, encode, load_pilot_examples, media_reserve, pick_rendering
from gemma_decision.mlx_backend import MLXDecisionModel
from gemma_decision.model import probabilities, to_device
from gemma_decision.train import load_checkpoint

IMAGE = ("oxford_pets", "eurosat", "textvqa", "gqa", "valen")
AUDIO = ("fleurs", "minds14", "cremad", "slurp", "belebele_fleurs")

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path, nargs="?", default=Path("runs/broad"))
ap.add_argument("--per-source", type=int, default=6)
args = ap.parse_args()

model, tok, cfg, device = load_checkpoint(args.run_dir)
model.eval()
processor = AutoProcessor.from_pretrained(cfg["model_id"])
rows = {}
for src in IMAGE + AUDIO:
    exs = load_pilot_examples("data/pilot", "test", [src])[: args.per_source]
    if exs:  # some sources have no test split (e.g. minds14 tests live in data/mm-decisions)
        rows[src] = [pick_rendering(ex, None, {}, prefer=("media", "both", "text")) for ex in exs]
batches = {}
for src, exs in rows.items():
    batches[src] = []
    for ex in exs:
        enc = encode(tok, ex.state, ex.questions, cfg["max_length"], reserve=media_reserve(ex))
        b = collate_mm([enc], [ex], processor)
        b.pop("q_meta")
        batches[src].append(b)


def timed_ms(fn, b):
    """One synchronized call, in ms (fn returns host data, so MLX has evaluated; torch syncs explicitly)."""
    torch.mps.synchronize()
    t = time.perf_counter()
    out = fn(b)
    torch.mps.synchronize()
    return out, 1000 * (time.perf_counter() - t)


t_ref = {}
with torch.inference_mode():
    ref = {}
    torch_probs = lambda b: probabilities(model(to_device(b, device)).float()).cpu().numpy()  # noqa: E731
    for src, bs in batches.items():
        torch_probs(bs[0])  # untimed warm-up, same as the MLX loop below
        res = [timed_ms(torch_probs, b) for b in bs]
        ref[src] = [r[0] for r in res]
        t_ref[src] = st.median(r[1] for r in res)
del model
torch.mps.empty_cache()

for name, dt in (("fp32", mx.float32), ("fp16", mx.float16)):
    m = MLXDecisionModel.load(args.run_dir, dt, encoder="vlm")
    print(f"\nMLX (mlx-vlm) {name}")
    print(f"{'source':16s} {'kind':5s} {'n':>3s} {'max |dp|':>9s} {'top-1':>7s} {'torch med ms':>13s} {'mlx med ms':>11s}")
    for src, bs in batches.items():
        m.probabilities(bs[0])  # untimed warm-up, same as the torch loop above
        diffs, agree, n = [], 0, 0
        res = [timed_ms(m.probabilities, b) for b in bs]
        outs = [r[0] for r in res]
        ms = st.median(r[1] for r in res)
        for b, p, r in zip(bs, outs, ref[src]):
            mask = b["q_opt_mask"].numpy()
            diffs.append(np.abs(np.where(mask, p - r, 0)).max())
            agree += int((np.where(mask, p, -1).argmax(-1) == np.where(mask, r, -1).argmax(-1)).sum())
            n += mask.shape[0]
        kind = "image" if src in IMAGE else "audio"
        print(f"{src:16s} {kind:5s} {len(bs):3d} {max(diffs):9.2e} {agree:3d}/{n:<3d} {t_ref[src]:13.0f} {ms:11.0f}")
    del m
    mx.clear_cache()
