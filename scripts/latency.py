"""Per-row decision latency (batch 1, all of a row's questions in one pass) for a trained run.

  uv run python scripts/latency.py runs/full --device cpu --threads 4 --rows 100
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from gemma_decision.data import collate, encode, load_examples
from gemma_decision.model import to_device
from gemma_decision.train import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--device", default="mps")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--rows", type=int, default=100)
args = ap.parse_args()

torch.set_num_threads(args.threads)
model, tok, cfg, _ = load_checkpoint(args.run_dir)
device = torch.device(args.device)
model = model.to(device).eval()
exs = load_examples(split="test")[: args.rows]
sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
times = []
with torch.inference_mode():
    for i, ex in enumerate(exs):
        t = time.perf_counter()
        b = collate([encode(tok, ex.state, ex.questions, cfg["max_length"])], tok.pad_token_id)
        b.pop("q_meta")
        model(to_device(b, device)).softmax(-1).cpu()
        sync()
        if i >= 5:  # warmup
            times.append(time.perf_counter() - t)
res = {"run": args.run_dir.name, "device": args.device, "threads": args.threads, "rows": len(times),
       "median_row_ms": statistics.median(times) * 1000, "p90_row_ms": sorted(times)[int(0.9 * len(times))] * 1000}
print(json.dumps(res))
