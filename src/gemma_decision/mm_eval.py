"""`gd-mm-eval runs/frozen runs/lora runs/full` - text vs media vs both on the multimodal decision set.

For every run and every rendering of the same rows (text | media | both) this reports per-workflow metrics,
writes runs/<name>/eval_mm.json, and prints a table. The frozen run's encoder is untouched EmbeddingGemma-2,
so `media` accuracy relative to frozen shows how much fine-tuning on text-only data moved the image/audio path.

Note: all heads were trained only on typed-decisions text, so every number here is zero-shot transfer to new
questions; compare runs against each other, not against the text benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_mm_examples
from .evaluate import report, run

VARIANTS = ("text", "media", "both")


def chance(examples) -> float:
    qs = [q for ex in examples for q in ex.questions]
    return sum(1 / len(q.keys) for q in qs) / len(qs)


def main():
    from transformers import AutoProcessor

    from .train import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/mm-decisions"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--metric", default="acc")
    args = ap.parse_args()

    data = {v: load_mm_examples(args.data, args.split, v) for v in VARIANTS}
    workflows = sorted({ex.workflow for ex in data["text"]})
    results = {}
    for run_dir in args.runs:
        out_path = run_dir / "eval_mm.json"
        if out_path.exists():
            results[run_dir.name] = json.loads(out_path.read_text())
            continue
        model, tokenizer, cfg, device = load_checkpoint(run_dir)
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        res = {}
        for v in VARIANTS:
            records, _ = run(model, tokenizer, data[v], device, args.batch_size, args.max_length, processor=processor)
            res[v] = report(records)
            print(f"{run_dir.name:8s} {v:5s} acc {res[v]['overall']['acc']:.3f}", flush=True)
        out_path.write_text(json.dumps(res, indent=2))
        results[run_dir.name] = res
        del model

    m = args.metric
    print(f"\nmetric={m}  (chance: " + ", ".join(
        f"{wf} {chance([e for e in data['text'] if e.workflow == wf]):.3f}" for wf in workflows) + ")\n")
    print("| run | variant | " + " | ".join(workflows) + " | overall |")
    print("|---" * (len(workflows) + 3) + "|")
    for name, res in results.items():
        for v in VARIANTS:
            cells = [f"{res[v].get(f'workflow={wf}', {}).get(m, float('nan')):.3f}" for wf in workflows]
            print(f"| {name} | {v} | " + " | ".join(cells) + f" | {res[v]['overall'][m]:.3f} |")


if __name__ == "__main__":
    main()
