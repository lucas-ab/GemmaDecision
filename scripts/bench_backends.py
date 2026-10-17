"""Benchmark the decision model on PyTorch (MPS, CPU) vs MLX (GPU, CPU), text inputs, on this Mac.

  uv run python scripts/bench_backends.py runs/broad [--examples 50]   # writes runs/<run>/bench_backends.json

Measures, for PyTorch MPS fp32 (as gd-serve runs it: bucketed padding, and without it), PyTorch CPU fp32 (with and
without padding, 8 threads = the M1 Max performance cores), MLX GPU fp32 / fp16 / bf16 and MLX CPU fp32:
  latency      single requests at fixed token lengths: --examples distinct requests per length (different state
               text, same token count), each timed once after one untimed warm-up request of that length; median/p90
  unseen       single requests at --examples random lengths never seen before (first-call cost); median/p90/max
  throughput   rows/s over 256 S1MB inputs, length-sorted batches of 16
  memory       during the throughput run: model weight bytes (and, for PyTorch, how much of that is the vision/audio
               towers this text benchmark never uses), peak live memory, that peak minus the pre-run baseline
               (activation overhead above the weights), and reserved memory.
               PyTorch MPS: live = torch.mps allocated tensors, reserved = peak driver allocation incl. allocator cache,
               both sampled from a 1 ms background thread (a lower bound; torch.mps has no peak API).
               PyTorch CPU: live and reserved = process RSS, sampled the same way (includes the Python runtime).
               MLX: live = mx.get_peak_memory() (exact), reserved = active + cache after the run.
"""
import argparse
import json
import random
import statistics as st
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import psutil
import torch

sys.path.insert(0, str(Path(__file__).parent))
from s1mb_bench import load_cases, to_examples  # noqa: E402

from gemma_decision.data import collate, encode, parse_questions  # noqa: E402
from gemma_decision.mlx_backend import MLXDecisionModel  # noqa: E402
from gemma_decision.model import probabilities, to_device  # noqa: E402
from gemma_decision.serve import pad_batch  # noqa: E402
from gemma_decision.train import load_checkpoint  # noqa: E402

LENGTHS = (192, 256, 512, 1024, 2048, 4096)  # the three questions alone take 151 tokens
CPU_THREADS = 8  # M1 Max performance cores; 10 (incl. efficiency cores) measured ~5% slower
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


GB = 1e9


class Torch:
    def __init__(self, model, tok, device, pad=True):
        self.model, self.tok, self.device, self.pad = model, tok, device, pad
        self.name = f"PyTorch {device.type.upper()} fp32" + ("" if pad else " (no padding)")
        self._stop, self._thread = threading.Event(), None

    def _sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def run(self, b):
        if self.pad and b["input_ids"].shape[0] == 1:
            b = pad_batch(b, self.tok.pad_token_id)
        p = probabilities(self.model(to_device(b, self.device)).float())
        self._sync()
        return p

    def weights(self) -> dict:
        """Parameter+buffer bytes: total, and the vision/audio-tower share the text benchmark never touches."""
        tensors = list(self.model.named_parameters()) + list(self.model.named_buffers())
        nbytes = lambda t: t.numel() * t.element_size()  # noqa: E731
        total = sum(nbytes(t) for _, t in tensors)
        towers = sum(nbytes(t) for n, t in tensors if "vision" in n or "audio" in n)
        return {"weights_total_gb": total / GB, "weights_unused_towers_gb": towers / GB,
                "weights_text_path_gb": (total - towers) / GB}

    def _live(self) -> tuple[int, int]:
        """(live, reserved) bytes now."""
        if self.device.type == "mps":
            return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
        rss = psutil.Process().memory_info().rss
        return rss, rss

    def _sample(self):
        while not self._stop.is_set():
            live, res = self._live()
            self._peak_active, self._peak_driver = max(self._peak_active, live), max(self._peak_driver, res)
            time.sleep(0.001)

    def mem_start(self):
        self._sync()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        self._baseline, self._peak_driver = self._live()
        self._peak_active = self._baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def mem_stop(self) -> dict:
        self._sync()
        self._stop.set()
        self._thread.join()
        live, res = self._live()
        self._peak_active, self._peak_driver = max(self._peak_active, live), max(self._peak_driver, res)
        return {"baseline_active_gb": self._baseline / GB, "peak_active_gb": self._peak_active / GB,
                "peak_over_baseline_gb": (self._peak_active - self._baseline) / GB,
                "reserved_gb": self._peak_driver / GB}


class MLX:
    def __init__(self, run_dir, dtype, label, device=mx.gpu):
        self.m = MLXDecisionModel.load(run_dir, dtype)
        self.device = device
        self.name = f"MLX {'CPU ' if device == mx.cpu else ''}{label}"

    def run(self, b):
        with mx.stream(self.device):
            p = mx.softmax(self.m.forward(b), axis=-1)
            mx.eval(p)
        return p

    def weights(self) -> dict:
        """Bytes of the loaded arrays (text encoder + head only; the towers are never loaded)."""
        total = sum(a.nbytes for w in (self.m.encoder.w, self.m.head.w) for a in w.values())
        return {"weights_total_gb": total / GB, "weights_unused_towers_gb": 0.0, "weights_text_path_gb": total / GB}

    def mem_start(self):
        mx.clear_cache()
        self._baseline = mx.get_active_memory()
        mx.reset_peak_memory()

    def mem_stop(self) -> dict:
        peak = mx.get_peak_memory()
        return {"baseline_active_gb": self._baseline / GB, "peak_active_gb": peak / GB,
                "peak_over_baseline_gb": (peak - self._baseline) / GB,
                "reserved_gb": (mx.get_active_memory() + mx.get_cache_memory()) / GB}


def once_ms(fn, b) -> float:
    t = time.perf_counter()
    fn(b)
    return 1000 * (time.perf_counter() - t)


def pct(xs, q) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def bench(backend, tok, fixed, warm, unseen, s1):
    r = {"name": backend.name, "examples": len(unseen)}
    r["latency_ms"] = {}
    for T in LENGTHS:
        backend.run(warm[T])  # untimed warm-up of this length (MPS compiles kernels per shape)
        ts = [once_ms(backend.run, b) for b in fixed[T]]
        r["latency_ms"][T] = {"median": round(st.median(ts), 1), "p90": pct(ts, 0.9)}
    ts = [once_ms(backend.run, b) for b in unseen]
    r["unseen_ms"] = {"median": round(st.median(ts), 1), "p90": pct(ts, 0.9), "max": round(max(ts), 1)}
    backend.mem_start()
    t = time.perf_counter()
    for i in range(0, len(s1), 16):
        b = collate(s1[i : i + 16], tok.pad_token_id)
        b.pop("q_meta")
        backend.run(b)
    r["throughput_rows_s"] = round(len(s1) / (time.perf_counter() - t), 1)
    mem = backend.mem_stop()
    r["memory_gb"] = {k: round(v, 2) for k, v in {**backend.weights(), **mem}.items()}
    print(json.dumps(r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, nargs="?", default=Path("runs/broad"))
    ap.add_argument("--examples", type=int, default=50, help="requests per fixed length, and unseen lengths")
    args = ap.parse_args()
    n = args.examples

    model, tok, cfg, device = load_checkpoint(args.run_dir)
    model.eval()
    fixed = {T: [batch_of_length(tok, T, seed=s) for s in range(n)] for T in LENGTHS}
    warm = {T: batch_of_length(tok, T, seed=10_000) for T in LENGTHS}
    rng = random.Random(7)
    unseen = [batch_of_length(tok, k, seed=k) for k in rng.sample(range(160, 700), n)]
    s1 = s1mb_inputs(tok)
    print(f"{n} requests per length; S1MB throughput set: {len(s1)} inputs, "
          f"{len(s1[0].input_ids)}-{len(s1[-1].input_ids)} tokens", flush=True)
    results = [bench(Torch(model, tok, device), tok, fixed, warm, unseen, s1),
               bench(Torch(model, tok, device, pad=False), tok, fixed, warm, unseen, s1)]
    model = model.to("cpu")
    torch.mps.empty_cache()
    torch.set_num_threads(CPU_THREADS)
    cpu = torch.device("cpu")
    results += [bench(Torch(model, tok, cpu), tok, fixed, warm, unseen, s1),
                bench(Torch(model, tok, cpu, pad=False), tok, fixed, warm, unseen, s1)]
    del model
    for dt, label, dev in ((mx.float32, "fp32", mx.gpu), (mx.float16, "fp16", mx.gpu),
                           (mx.bfloat16, "bf16", mx.gpu), (mx.float32, "fp32", mx.cpu)):
        results.append(bench(MLX(args.run_dir, dt, label, dev), tok, fixed, warm, unseen, s1))
        mx.clear_cache()
    (args.run_dir / "bench_backends.json").write_text(json.dumps(results, indent=1))

    print(f"\nlatency in ms, median (p90) of {n} requests per length")
    print(f"{'backend':32s}" + "".join(f"{T:>14d}" for T in LENGTHS) + "   unseen med/p90   rows/s")
    for r in results:
        lat = "".join(f"{'%.0f (%.0f)' % (v['median'], v['p90']):>14s}" for v in r["latency_ms"].values())
        u = r["unseen_ms"]
        print(f"{r['name']:32s}{lat}   {u['median']:6.1f}/{u['p90']:6.1f}   {r['throughput_rows_s']:6.1f}")

    print("\nmemory during the throughput run (GB). 'weights' = parameter bytes; 'towers' = vision/audio share of it "
          "(unused by this text benchmark);\n'peak live' = max live memory (torch MPS: sampled tensors, lower bound; "
          "torch CPU: sampled process RSS; MLX: exact); 'over weights' = peak live minus pre-run baseline;\n"
          "'reserved' = torch MPS: peak driver allocation incl. allocator cache; torch CPU: peak RSS; "
          "MLX: active+cache after the run.")
    print(f"{'backend':32s}{'weights':>9s}{'towers':>8s}{'text path':>11s}{'peak live':>11s}{'over weights':>14s}{'reserved':>10s}")
    for r in results:
        m = r["memory_gb"]
        print(f"{r['name']:32s}{m['weights_total_gb']:9.2f}{m['weights_unused_towers_gb']:8.2f}{m['weights_text_path_gb']:11.2f}"
              f"{m['peak_active_gb']:11.2f}{m['peak_over_baseline_gb']:14.2f}{m['reserved_gb']:10.2f}")


if __name__ == "__main__":
    main()
