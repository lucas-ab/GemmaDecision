"""Inference, metrics, temperature calibration, and the `gd-eval` CLI.

Metrics are per question (a row has ~5), split by type and workflow:
  acc          argmax == gold label
  acc_agreed   same, only on questions where annotators' argmax agreed (cleaner signal)
  annot_agree  mean gold probability of the predicted option = expected agreement with a random annotator;
               `oracle` is the same for an ideal predictor (max gold probability), i.e. the ceiling
  row_acc      every question in the row correct
  ece          expected calibration error of top-1 confidence vs correctness (15 bins)
  brier, kl    distance of predicted distribution to gold distribution
  score_mae    |expected predicted level - gold expected level|, score questions only
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import TYPES, collate, encode
from .model import probabilities, to_device


@torch.no_grad()
def run(model, tokenizer, examples, device, batch_size: int = 8, max_length: int = 1024, calibrated: bool = True):
    """Returns (records, logits list). One record per question."""
    model.eval()
    records, all_logits = [], []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        enc = [encode(tokenizer, ex.state, ex.questions, max_length) for ex in chunk]
        batch = collate(enc, tokenizer.pad_token_id)
        meta = batch.pop("q_meta")
        logits = model(to_device(batch, device), calibrated=calibrated).float().cpu()
        probs = probabilities(logits)
        for j, (b, q) in enumerate(meta):
            k = len(q.keys)
            records.append({
                "id": chunk[b].id, "workflow": chunk[b].workflow, "question": q.name, "type": q.type,
                "keys": q.keys, "probs": probs[j, :k].tolist(), "target": q.target, "label": q.label,
                "agree": q.agree, "gold_score": q.gold_score,
            })
            all_logits.append((TYPES[q.type], logits[j, :k], torch.tensor(q.target)))
    return records, all_logits


def _ece(conf, correct, bins: int = 15) -> float:
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - avg)
    return err


def metrics(records: list[dict]) -> dict:
    if not records:
        return {}
    pred = [max(range(len(r["probs"])), key=r["probs"].__getitem__) for r in records]
    correct = [int(p == r["label"]) for p, r in zip(pred, records)]
    conf = [r["probs"][p] for p, r in zip(pred, records)]
    agreed = [c for c, r in zip(correct, records) if r["agree"]]
    out = {
        "n": len(records),
        "acc": sum(correct) / len(records),
        "acc_agreed": sum(agreed) / len(agreed) if agreed else float("nan"),
        "annot_agree": sum(r["target"][p] for p, r in zip(pred, records)) / len(records),
        "oracle": sum(max(r["target"]) for r in records) / len(records),
        "ece": _ece(conf, correct),
        "brier": sum(sum((a - b) ** 2 for a, b in zip(r["probs"], r["target"])) for r in records) / len(records),
        "kl": sum(
            sum(t * (math.log(max(t, 1e-12)) - math.log(max(p, 1e-12))) for p, t in zip(r["probs"], r["target"]))
            for r in records
        ) / len(records),
    }
    scores = [r for r in records if r["type"] == "score"]
    if scores:
        out["score_mae"] = sum(
            abs(sum(i * p for i, p in enumerate(r["probs"])) - r["gold_score"]) for r in scores
        ) / len(scores)
    rows = defaultdict(list)
    for c, r in zip(correct, records):
        rows[r["id"]].append(c)
    out["row_acc"] = sum(all(v) for v in rows.values()) / len(rows)
    return out


def report(records: list[dict]) -> dict:
    res = {"overall": metrics(records)}
    for key in ("type", "workflow"):
        for v in sorted({r[key] for r in records}):
            res[f"{key}={v}"] = metrics([r for r in records if r[key] == v])
    return res


def format_report(res: dict) -> str:
    cols = ["n", "acc", "acc_agreed", "annot_agree", "oracle", "row_acc", "ece", "brier", "kl", "score_mae"]
    lines = ["| slice | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in res.items():
        cells = [f"{m[c]:.3f}" if isinstance(m.get(c), float) else str(m.get(c, "")) for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fit_temperatures(model, tokenizer, examples, device, batch_size: int = 8, max_length: int = 1024) -> list[float]:
    """Fit one temperature per question type on held-out data by minimizing soft cross-entropy."""
    _, logits = run(model, tokenizer, examples, device, batch_size, max_length, calibrated=False)
    temps = []
    for t in range(len(TYPES)):
        items = [(lg, tg) for ty, lg, tg in logits if ty == t]
        if not items:
            temps.append(1.0)
            continue
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = sum(-(tg * F.log_softmax(lg / log_t.exp(), -1)).sum() for lg, tg in items) / len(items)
            loss.backward()
            return loss

        opt.step(closure)
        temps.append(log_t.detach().exp().clamp(0.05, 20.0).item())
    model.head.log_temp.copy_(torch.tensor(temps).log())
    return temps


def main():
    from .train import load_checkpoint

    ap = argparse.ArgumentParser(description="Evaluate a trained decision model on the typed-decisions test split.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from .data import load_examples

    model, tokenizer, cfg, device = load_checkpoint(args.run_dir)
    examples = load_examples(cfg["dataset"], cfg.get("dataset_config", "all"), args.split)
    records, _ = run(model, tokenizer, examples, device, args.batch_size, cfg["max_length"])
    res = report(records)
    print(format_report(res))
    (args.run_dir / f"eval_{args.split}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
