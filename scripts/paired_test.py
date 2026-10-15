"""Paired comparison of a trained run vs an external model's dumped predictions on typed-decisions test.

  uv run python scripts/paired_test.py runs/full_e5 runs/julia/typed_test.jsonl
Prints both accuracies, McNemar's exact test, and a paired bootstrap 95% CI of the accuracy difference
(resampling rows, since a row's questions are not independent).
"""
import json
import random
import sys
from math import comb
from pathlib import Path

from gemma_decision.data import load_examples
from gemma_decision.evaluate import run
from gemma_decision.external import records_for
from gemma_decision.train import load_checkpoint

run_dir, ext = Path(sys.argv[1]), Path(sys.argv[2])
model, tok, cfg, device = load_checkpoint(run_dir)
examples = load_examples(cfg["dataset"], cfg["dataset_config"], "test")
ours, _ = run(model, tok, examples, device, 8, cfg["max_length"])
preds = {}
for line in ext.read_text().splitlines():
    p = json.loads(line)
    preds[(p["id"], p["question"])] = (p["keys"], p["probs"])
theirs = records_for(preds, examples)

def correct(r):
    return int(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["label"])

key = lambda r: (r["id"], r["question"])  # noqa: E731
a = {key(r): correct(r) for r in ours}
b = {key(r): correct(r) for r in theirs}
keys = sorted(a)
only_a = sum(a[k] and not b[k] for k in keys)
only_b = sum(b[k] and not a[k] for k in keys)
n = only_a + only_b
p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)

rows = sorted({k[0] for k in keys})
by_row = {r: [k for k in keys if k[0] == r] for r in rows}
rng = random.Random(0)
diffs = []
for _ in range(2000):
    sample = [k for r in (rng.choice(rows) for _ in rows) for k in by_row[r]]
    diffs.append(sum(a[k] - b[k] for k in sample) / len(sample))
diffs.sort()
print(json.dumps({
    "ours": run_dir.name, "acc_ours": sum(a.values()) / len(keys), "acc_external": sum(b.values()) / len(keys),
    "only_ours_correct": only_a, "only_external_correct": only_b, "mcnemar_p": round(p, 4),
    "diff_95ci": [round(diffs[50], 4), round(diffs[1949], 4)],
}, indent=2))
