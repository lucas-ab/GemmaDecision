"""Probe catch-all and flip behavior on typed-decisions TEST rows (never trained on) for several runs.

usage: uv run python scripts/synth_probe.py runs/a runs/b ...
"""
import copy
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset

from gemma_decision.data import row_to_example
from gemma_decision.evaluate import metrics, run
from gemma_decision.synth_build import NONE_KEY, catch_all, flip
from gemma_decision.train import load_checkpoint

raw = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="test"))


def variants():
    out = {"catch_removed": [], "catch_kept": [], "flip": []}
    for r in raw:
        r = {**r, "source": "typed_decisions"}
        for tag in ("removed", "kept"):
            # force the branch: rng.random() < 0.5 removes
            class R(random.Random):
                def random(self, _t=tag):
                    return 0.0 if _t == "removed" else 0.99
            x = catch_all(copy.deepcopy(r), R(r["id"]))
            if x:
                name = x["id"].split(f"catchall-{tag}-", 1)[1]
                out[f"catch_{tag}"].append((row_to_example({**x, "state": r["state"]}), name))
        x = flip(copy.deepcopy(r), random.Random(r["id"]))
        if x:
            out["flip"].append((row_to_example({**x, "state": r["state"]}), x["id"].split("~flip-", 1)[1]))
    return out


V = variants()
print({k: len(v) for k, v in V.items()}, flush=True)
for run_dir in map(Path, sys.argv[1:]):
    model, tok, cfg, device = load_checkpoint(run_dir)
    res = {}
    for k, pairs in V.items():
        exs = [e for e, _ in pairs]
        want = {(e.id, n) for e, n in pairs}
        recs, _ = run(model, tok, exs, device, 8, cfg["max_length"])
        recs = [r for r in recs if (r["id"], r["question"]) in want]
        m = metrics(recs)
        none_rate = None
        if k.startswith("catch"):
            picks = [r["keys"][max(range(len(r["probs"])), key=r["probs"].__getitem__)] for r in recs]
            none_rate = sum(p == NONE_KEY for p in picks) / len(picks)
        res[k] = {"n": len(recs), "acc": round(m["acc"], 4), "kl": round(m["kl"], 4),
                  "picks_none": none_rate and round(none_rate, 4)}
    print(run_dir, json.dumps(res), flush=True)
    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
