"""Benchmark the decision model on PyTorch MPS vs MLX (text inputs), on this Mac.

  uv run python scripts/bench_backends.py runs/broad            # writes runs/<run>/bench_backends.json

Measures, for PyTorch MPS fp32 (as gd-serve runs it: bucketed padding) and MLX fp32 / fp16 / bf16:
  latency      single request at fixed token lengths (median of 10 after warm-up)
  unseen       single requests at 20 random lengths never seen before (first-call cost)
  throughput   rows/s over 256 S1MB inputs, length-sorted batches of 16
  memory       peak GPU memory during the throughput run
"""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from s1mb_bench import load_cases, to_examples  # noqa: E402

from gemma_decision.data import collate, encode, parse_questions  # noqa: E402
from gemma_decision.mlx_backend import MLXDecisionModel  # noqa: E402
from gemma_decision.model import probabilities, to_device  # noqa: E402
from gemma_decision.serve import pad_batch  # noqa: E402
from gemma_decision.train import load_checkpoint  # noqa: E402

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/broad")
LENGTHS = (192, 256, 512, 1024, 2048, 4096)  # the three questions alone take 151 tokens
QUESTIONS = parse_questions({
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Payments.", "tech": "Bugs.", "account": "Logins.", "sales": "Pricing."}},
    "urgent": {"type": "noul", "instructions": "This is urgent."},
    "level": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "angry", "furious"]}})


def batch_of_length(tok, target: int, seed: int = 0):
    """A 3-question request whose encoded length is exactly `target` tokens (state trimmed to fit)."""
    rng = random.Random(seed)
    words = "invoice refund login outage pricing urgent password charged twice receipt delivery broken app".split()
    state = " ".join(rng.choice(words) for _ in range(target * 2))
    enc = encode(tok, state, QUESTIONS, target)
    assert len(enc.input_ids) <= target
    b = collate([enc], tok.pad_token_id)
    b.pop("q_meta")
    return b


def s1mb_inputs(tok, n=256, max_length=2048):
    spec = json.loads(Path("data/s1mb/english-v1/benchmarks.json").read_text())
    rng, out = random.Random(1), []
    pool = [(b, c) for b in spec["benchmarks"] for c in load_cases(b["dataset"].split("/", 1)[1], b["task"])[:20]]
    for b, case in rng.sample(pool, len(pool)):
        ex = to_examples(case)[0][0]
        try:
            out.append(encode(tok, ex.state, ex.questions, max_length))
        except ValueError:  # questions alone exceed max_length (a few long option lists)
            continue
        if len(out) == n:
            break
    return sorted(out, key=lambda e: len(e.input_ids))


class Torch:
    name = "PyTorch MPS fp32"

    def __init__(self, model, tok, device, pad=True):
        self.model, self.tok, self.device, self.pad = model, tok, device, pad
        if not pad:
            self.name += " (no padding)"

    @torch.inference_mode()
    def run(self, b):
        if self.pad and b["input_ids"].shape[0] == 1:
            b = pad_batch(b, self.tok.pad_token_id)
        p = probabilities(self.model(to_device(b, self.device)).float())
        torch.mps.synchronize()
        return p

    def peak_reset(self):
        torch.mps.empty_cache()

    def peak(self):
        return torch.mps.driver_allocated_memory() / 1e9


class MLX:
    def __init__(self, dtype, label):
        self.m = MLXDecisionModel.load(run_dir, dtype)
        self.name = f"MLX {label}"

    def run(self, b):
        p = mx.softmax(self.m.forward(b), axis=-1)
        mx.eval(p)
        return p

    def peak_reset(self):
        mx.reset_peak_memory()

    def peak(self):
        return mx.get_peak_memory() / 1e9


def timed(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t))
    return st.median(ts)


def bench(backend, tok, fixed, unseen, s1):
    r = {"name": backend.name}
    backend.run(fixed[192])  # warm-up
    r["latency_ms"] = {T: round(timed(lambda: backend.run(fixed[T]), 10), 1) for T in LENGTHS}
    times = []
    for b in unseen:
        t = time.perf_counter()
        backend.run(b)
        times.append(1000 * (time.perf_counter() - t))
    times.sort()
    r["unseen_ms"] = {"median": round(times[len(times) // 2], 1), "p90": round(times[int(0.9 * len(times))], 1),
                      "max": round(times[-1], 1)}
    backend.peak_reset()
    t = time.perf_counter()
    for i in range(0, len(s1), 16):
        b = collate(s1[i : i + 16], tok.pad_token_id)
        b.pop("q_meta")
        backend.run(b)
    r["throughput_rows_s"] = round(len(s1) / (time.perf_counter() - t), 1)
    r["peak_memory_gb"] = round(backend.peak(), 2)
    print(json.dumps(r), flush=True)
    return r


def main():
    model, tok, cfg, device = load_checkpoint(run_dir)
    model.eval()
    fixed = {T: batch_of_length(tok, T) for T in LENGTHS}
    rng = random.Random(7)
    unseen = [batch_of_length(tok, n, seed=n) for n in rng.sample(range(160, 700), 20)]
    s1 = s1mb_inputs(tok)
    print(f"S1MB throughput set: {len(s1)} inputs, {len(s1[0].input_ids)}-{len(s1[-1].input_ids)} tokens", flush=True)
    results = [bench(Torch(model, tok, device), tok, fixed, unseen, s1),
               bench(Torch(model, tok, device, pad=False), tok, fixed, unseen, s1)]
    del model
    torch.mps.empty_cache()
    for dt, label in ((mx.float32, "fp32"), (mx.float16, "fp16"), (mx.bfloat16, "bf16")):
        results.append(bench(MLX(dt, label), tok, fixed, unseen, s1))
        mx.clear_cache()
    (run_dir / "bench_backends.json").write_text(json.dumps(results, indent=1))

    print(f"\n{'backend':32s}" + "".join(f"{T:>7d}" for T in LENGTHS) + "   unseen med/p90   rows/s   peak GB")
    for r in results:
        lat = "".join(f"{r['latency_ms'][T]:7.1f}" for T in LENGTHS)
        u = r["unseen_ms"]
        print(f"{r['name']:32s}{lat}   {u['median']:6.1f}/{u['p90']:6.1f}   {r['throughput_rows_s']:6.1f}   {r['peak_memory_gb']:6.2f}")


if __name__ == "__main__":
    main()
