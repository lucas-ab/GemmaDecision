"""Score a trained run on S1MB English v1 (System One Mosaic Benchmark, github.com/hotchpotch/S1MB).

  uv run python scripts/s1mb_bench.py runs/continue_synth            # all 137 benchmarks
  uv run python scripts/s1mb_bench.py runs/continue_synth --limit 20 # quick look

Data: hotchpotch/S1MB-english-v1 parquet files in data/s1mb/english-v1/data/<subset>/test/, plus
data/s1mb/english-v1/benchmarks.json (the category and benchmark definitions from the S1MB evaluator).

Inputs follow the S1MB adapter rules: only the benchmark's task questions per case, authored instructions
and criteria in declared order, no case or question IDs in the text (the question is named "decision"),
anonymous choice keys (A, B, ...), Score levels ordered by their numeric value. Metrics reimplement
evaluator/src/s1mb/metrics.py and diagnostics.py (primary metric, baselines, baseline-adjusted score);
Task Avg is the mean of the three task means of the clipped adjusted scores, as on the leaderboard.
Writes runs/<run>/s1mb/<benchmark>.json (predictions + metrics) and runs/<run>/s1mb/summary.json.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq
import torch

from gemma_decision.data import _render
from gemma_decision.evaluate import run
from gemma_decision.s1mb_format import case_from_row, split_case, to_example
from gemma_decision.train import load_checkpoint

ROOT = Path("data/s1mb/english-v1")
EPS = 1e-12


def load_cases(subset: str, task: str) -> list[dict]:
    """Test cases with only `task` questions (see gemma_decision.s1mb_format.case_from_row)."""
    cases = []
    for path in sorted((ROOT / "data" / subset / "test").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if row["split"] == "test" and (case := case_from_row(row, task)):
                cases.append(case)
    return cases


def to_examples(case: dict):
    return [to_example(part) for part in split_case(case)]



def metrics(cases: list[dict], preds: dict, task: str) -> dict:
    """S1MB primary metric + baseline-adjusted score; `preds[(case, question)] = {option_id: p}`.

    Baselines depend only on the benchmark entries, never on predictions. With no answered decisions
    (every case failed) the prediction-based metrics and scores are None, so the run continues and the
    summary marks it incomplete."""
    entries = [(q, c["targets"][q["id"]]) for c in cases for q in c["questions"]]
    rows = [(q, c["targets"][q["id"]], preds[(c["id"], q["id"])])
            for c in cases for q in c["questions"] if (c["id"], q["id"]) in preds]
    n = len(rows)
    out = {"decisions": len(entries), "answered": n}
    if task == "noul":
        prevalence = sum(t["true"] for _, t in entries) / len(entries) if entries else None
        tp = sum(t["true"] for _, t, p in rows if p["true"] >= 0.5)
        fp = sum(t["false"] for _, t, p in rows if p["true"] >= 0.5)
        fn = sum(t["true"] for _, t, p in rows if p["true"] < 0.5)
        tn = sum(t["false"] for _, t, p in rows if p["true"] < 0.5)
        recall = tp / (tp + fn) if tp + fn > EPS else None
        spec = tn / (tn + fp) if tn + fp > EPS else None
        ba = (recall + spec) / 2 if recall is not None and spec is not None else None
        out.update(binary_brier=sum((p["true"] - t["true"]) ** 2 for _, t, p in rows) / n if n else None,
                   accuracy=(tp + tn) / n if n else None, balanced_accuracy=ba, positive_prevalence=prevalence)
        skill = 2 * ba - 1 if ba is not None else None
    elif task == "choice":
        base = None
        if entries:
            uniform = sum(1 / len(q["options"]) for q, _ in entries) / len(entries)
            ids = {o["id"] for q, _ in entries for o in q["options"]}
            fixed_id = max(sum(t.get(k, 1 / len(q["options"])) for q, t in entries) / len(entries) for k in ids)
            fixed_pos = max(sum(t[q["options"][i]["id"]] if i < len(q["options"]) else 1 / len(q["options"])
                                for q, t in entries) / len(entries)
                            for i in range(max(len(q["options"]) for q, _ in entries)))
            base = max(uniform, fixed_id, fixed_pos)
        # argmax with ties resolved in the original candidate order
        acc = sum(t[max(q["options"], key=lambda o: p[o["id"]])["id"]] for q, t, p in rows) / n if n else None
        out.update(target_mass_at_prediction=acc, fixed_answer_accuracy_baseline=base)
        skill = (acc - base) / (1 - base) if acc is not None and base is not None and base < 1 - EPS else None
    else:
        def norm_ev(q, probs):
            vals = {o["id"]: float(o["value"]) for o in q["options"]}
            lo, hi = min(vals.values()), max(vals.values())
            return sum((v - lo) / (hi - lo) * probs[k] for k, v in vals.items())

        targets = [norm_ev(q, t) for q, t in entries]
        base = None
        if targets:
            const = median(targets)
            base = sum(abs(t - const) for t in targets) / len(targets)
        errs = []
        for q, t, p in rows:
            vals = {o["id"]: o["value"] for o in q["options"]}
            errs.append(sum(v * (p[k] - t[k]) for k, v in vals.items()) / (max(vals.values()) - min(vals.values())))
        mae = sum(abs(e) for e in errs) / n if n else None
        out.update(normalized_expected_score_mae=mae, constant_mae_baseline=base)
        skill = 1 - mae / base if mae is not None and base is not None and base > EPS else None
    out["baseline_adjusted_skill"] = skill
    out["baseline_adjusted_score"] = max(0.0, min(1.0, skill)) if skill is not None else None
    return out


def run_capped(model, tok, examples, device, max_length):
    """run() on a group; on out-of-memory split the group in half (a single row that still fails is unanswered)."""
    try:
        recs, _ = run(model, tok, examples, device, len(examples), max_length)
    except RuntimeError as err:
        if "out of memory" not in str(err):
            raise
        if device.type == "mps":
            torch.mps.empty_cache()
        if len(examples) == 1:
            print(f"  OOM on {examples[0].id}; left unanswered", flush=True)
            return []
        half = len(examples) // 2
        return run_capped(model, tok, examples[:half], device, max_length) + \
            run_capped(model, tok, examples[half:], device, max_length)
    if device.type == "mps" and max(len(ex.state) for ex in examples) > 8000:
        torch.mps.empty_cache()  # long rows leave large cached blocks behind
    return recs


def predict(model, tok, device, cases, max_length, token_budget):
    exs = [x for c in cases for x in to_examples(c)]
    lengths = [len(tok(_render(ex.state, ex.questions)[0])["input_ids"]) for ex, _, _ in exs]
    order = sorted(range(len(exs)), key=lengths.__getitem__)
    preds, truncated, i = {}, 0, 0
    while i < len(order):  # length-sorted groups under a token budget
        j = i + 1
        while j < len(order) and (j - i + 1) * min(lengths[order[j]], max_length) <= token_budget:
            j += 1
        group = order[i:j]
        truncated += sum(lengths[k] > max_length for k in group)
        recs = run_capped(model, tok, [exs[k][0] for k in group], device, max_length)
        by_case = {}
        for r in recs:
            by_case.setdefault(r["id"], []).append(r)
        for k in group:
            ex, qs, id_order = exs[k]
            case_id = ex.id.rsplit("#", 1)[0]
            for qi, r in enumerate(by_case.get(ex.id, [])):  # records keep question order within a row
                preds[(case_id, qs[qi]["id"])] = dict(zip(id_order[qi], r["probs"]))
        i = j
    return preds, truncated


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--limit", type=int, help="first N benchmarks only (smoke test)")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--token-budget", type=int, default=16000)
    ap.add_argument("--mps-memory-fraction", type=float, default=0.6)
    args = ap.parse_args()

    spec = json.loads((ROOT / "benchmarks.json").read_text())
    benches = spec["benchmarks"][: args.limit]
    model, tok, cfg, device = load_checkpoint(args.run_dir)
    if device.type == "mps":  # fail fast with OOM (handled per group) instead of swapping the machine
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    out_dir = args.run_dir / "s1mb"
    out_dir.mkdir(exist_ok=True)
    results = {}
    for b in benches:
        dest = out_dir / f"{b['id']}.json"
        if dest.exists():  # resumable
            results[b["id"]] = json.loads(dest.read_text())["metrics"]
            continue
        t0 = time.time()
        cases = load_cases(b["dataset"].split("/", 1)[1], b["task"])
        n_dec = sum(len(c["questions"]) for c in cases)
        if len(cases) != b["case_count"] or n_dec != b["decision_count"]:
            raise ValueError(f"{b['id']}: {len(cases)} cases / {n_dec} decisions, expected "
                             f"{b['case_count']} / {b['decision_count']}")
        preds, truncated = predict(model, tok, device, cases, args.max_length, args.token_budget)
        m = metrics(cases, preds, b["task"])
        m.update(truncated_cases=truncated, seconds=round(time.time() - t0, 1))
        dest.write_text(json.dumps({"benchmark": b, "metrics": m,
                                    "predictions": [{"case_id": c, "question_id": q, "probabilities": p}
                                                    for (c, q), p in preds.items()]}))
        results[b["id"]] = m
        print(f"{b['id']:60s} {b['task']:6s} adj {m['baseline_adjusted_score']}  answered {m['answered']}/{m['decisions']}"
              f"  trunc {truncated}  {m['seconds']}s", flush=True)

    summary = summarize(benches, results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_benchmark"}, indent=2))


def summarize(benches, results):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return 100 * sum(xs) / len(xs) if xs else None

    task_of = {b["id"]: b["task"] for b in benches}
    tasks = {t: mean([results[b]["baseline_adjusted_score"] for b in results if task_of[b] == t])
             for t in ("noul", "choice", "score")}
    general = {t: mean([results[b]["baseline_adjusted_score"] for b in results
                        if task_of[b] == t and b.startswith("s1mb-generalization-")]) for t in tasks}
    # a benchmark with no answered decisions has a None score and fails answered == decisions
    complete = all(r["answered"] == r["decisions"] for r in results.values())
    return {
        "benchmarks": len(results), "complete": complete,
        "task_avg": sum(tasks.values()) / 3 if all(v is not None for v in tasks.values()) else None,
        "tasks": tasks, "general": general,
        "truncated_cases": sum(r.get("truncated_cases", 0) for r in results.values()),
        "per_benchmark": {b: results[b]["baseline_adjusted_score"] for b in results},
    }


if __name__ == "__main__":
    main()
