"""Run SupersonicLabs/Julia-1 on decision rows and dump raw per-question probabilities.

Runs in Julia's own environment (transformers 5.0.x), not this project's:

  data/julia/.venv/bin/python -I scripts/julia_bench.py --checkpoint data/julia/Julia-1 \
      --rows <typed-decisions all/test parquet> --out runs/julia/typed_test.jsonl

Settings follow Julia's published CPU reproduction (scripts/reproduce_typed.py): torch backend,
strict encoding, max_length 1024, head_length 512, marker_only_head False, one question per forward.
Rows: a typed-decisions parquet, or a gd-mm-build parquet whose `state_text` column is used (Julia is text-only).
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rows", required=True, help="parquet with state/questions/gold (typed-decisions or mm set)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, str(args.checkpoint.resolve()))
    import pyarrow.parquet as pq
    import torch
    from julia.router.engine import FastEngine

    torch.set_num_threads(args.threads)
    engine = FastEngine(args.checkpoint.resolve(), device=args.device, transformer_backend="torch",
                        strict_encoding=True, max_length=1024, head_length=512, batch_size=16,
                        marker_only_head=False)

    cases = pq.read_table(args.rows).to_pylist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_latency = []
    with args.out.open("w") as out:
        for n, case in enumerate(cases, 1):
            raw_state = case.get("state", case.get("state_text"))
            try:  # typed-decisions states are JSON objects; Julia's harness passes the parsed object
                state = json.loads(raw_state)
            except (json.JSONDecodeError, TypeError):
                state = raw_state
            started = time.perf_counter()
            for qid, q in json.loads(case["questions"]).items():
                kind, criteria = q["type"], q.get("criteria")
                if criteria is None and kind == "noul":
                    criteria = {"false": "false", "true": "true"}
                if isinstance(criteria, list):
                    criteria = {str(i): v for i, v in enumerate(criteria)}
                keys = ["false", "true"] if kind == "noul" else list(criteria)
                row = dict(state=state, question=q["instructions"], type=kind, options=[criteria[k] for k in keys])
                logits = engine.logits([row])[0]  # one question per forward, as in Julia's protocol
                m = max(logits)
                p = [math.exp(x - m) for x in logits]
                s = sum(p)
                out.write(json.dumps({"id": case["id"], "question": qid, "keys": keys,
                                      "probs": [x / s for x in p]}) + "\n")
            row_latency.append(time.perf_counter() - started)
            if n % 50 == 0:
                print(f"{n}/{len(cases)}  median row {statistics.median(row_latency) * 1000:.0f} ms", flush=True)
    meta = {"device": args.device, "threads": args.threads, "rows": len(cases),
            "median_row_ms": statistics.median(row_latency) * 1000,
            "p90_row_ms": sorted(row_latency)[int(0.9 * len(row_latency))] * 1000,
            "total_s": sum(row_latency)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
