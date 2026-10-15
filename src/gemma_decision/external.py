"""`gd-score-external runs/julia/typed_test.jsonl` - score another model's dumped probabilities with our metrics.

Input JSONL, one line per question: {"id", "question", "keys", "probs"} (see scripts/julia_bench.py).
Probabilities are re-ordered by key name to match our option order, then fed to evaluate.report, so the
numbers are directly comparable with gd-eval / gd-mm-eval output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_examples, load_mm_examples
from .evaluate import format_report, report


def records_for(preds: dict, examples) -> list[dict]:
    records = []
    for ex in examples:
        for q in ex.questions:
            keys, probs = preds[(ex.id, q.name)]
            by_key = dict(zip(keys, probs))
            records.append({
                "id": ex.id, "workflow": ex.workflow, "question": q.name, "type": q.type, "keys": q.keys,
                "probs": [by_key[k] for k in q.keys], "target": q.target, "label": q.label,
                "agree": q.agree, "gold_score": q.gold_score,
            })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--mm", type=Path, help="score against a gd-mm-build root (text variant) instead of typed-decisions")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    preds = {}
    for line in args.predictions.read_text().splitlines():
        p = json.loads(line)
        preds[(p["id"], p["question"])] = (p["keys"], p["probs"])
    examples = load_mm_examples(args.mm, "test", "text") if args.mm else load_examples(split="test")
    examples = [ex for ex in examples if all((ex.id, q.name) in preds for q in ex.questions)]
    res = report(records_for(preds, examples))
    print(format_report(res))
    out = args.out or args.predictions.with_name(args.predictions.stem + "_eval.json")
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
