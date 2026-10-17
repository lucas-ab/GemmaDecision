"""Place our S1MB runs in the public S1MB English v1 leaderboard (Task Avg and Borda, as the viewer computes them).

  curl -o data/s1mb/leaderboard/display.json https://hotchpotch-s1mb-leaderboard.hf.space/api/display/
  uv run python scripts/s1mb_leaderboard.py runs/continue_synth runs/modal/pilot_full-r2-s3407

Reads each run's runs/<run>/s1mb/summary.json (from scripts/s1mb_bench.py). The cohort is every complete,
measured English v1 model plus our runs; Borda = mean over all 137 benchmarks of 100 * (N - rank) / (N - 1),
ties averaged. Adding our runs to the cohort shifts everyone's Borda slightly, so the published models'
Borda is also printed without them to check the reproduction.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DISPLAY = Path("data/s1mb/leaderboard/display.json")


def _succeeded(result):
    counts = result.get("counts") or {}
    return counts.get("succeeded") or 0


def collect_scores(payload, bench_ids):
    """Measured, complete scores per (model index, benchmark id) -> (scores, duplicates).

    Results carry no timestamp or dataset revision, only counts, so when the same (model, benchmark) has more
    than one measured complete result the one with the most succeeded decisions wins, and among equals the
    later one in the list (the previous overwrite behavior). `duplicates` is the number of surplus results."""
    best = {}
    duplicates = 0
    for r in payload["results"]:
        b = r["benchmark"]
        bid = payload["benchmarks"][b]["id"] if isinstance(b, int) else b["id"]
        if bid not in bench_ids or r["provenance"] != "measured" or r["status"] != "complete":
            continue
        key = (r["model"], bid)
        if key in best:
            duplicates += 1
            if _succeeded(r) < _succeeded(best[key]):
                continue
        best[key] = r
    scores = defaultdict(dict)
    for (model, bid), r in best.items():
        scores[model][bid] = r["metrics"].get("baseline_adjusted_score")
    return scores, duplicates


def build_models(payload, bench_ids):
    """Complete models (a score on every benchmark) keyed "name [run_id]"; warns on duplicate results or keys."""
    scores, duplicates = collect_scores(payload, bench_ids)
    if duplicates:
        print(f"warning: {duplicates} duplicate measured result(s) for the same model and benchmark in "
              f"display.json; kept the one with the most succeeded decisions (ties: the later)", file=sys.stderr)
    models = {}
    for i, m in enumerate(payload["models"]):
        s = scores.get(i, {})
        if len(s) == len(bench_ids) and all(v is not None for v in s.values()):
            info = m["model"]
            name = info.get("display_name") or info["id"]
            key = f"{name} [{m['run_id']}]"
            if key in models:
                print(f"warning: duplicate complete model entry {key!r} (index {i}); keeping the later one",
                      file=sys.stderr)
            models[key] = {"scores": s, "total_params": info.get("total_params")}
    return models


def published(category="english-v1"):
    p = json.loads(DISPLAY.read_text())["payload"]
    cat = next(c for c in p["categories"] if c["id"] == category)
    bench_ids = cat["benchmarks"]
    task_of = {b["id"]: b["task"] for b in p["benchmarks"]}
    return bench_ids, task_of, build_models(p, bench_ids)


def task_avg(scores, bench_ids, task_of):
    tasks = {}
    for t in ("noul", "choice", "score"):
        xs = [scores[b] for b in bench_ids if task_of[b] == t]
        tasks[t] = 100 * sum(xs) / len(xs)
    gen = {t: 100 * sum(v) / len(v) for t in tasks
           for v in [[scores[b] for b in bench_ids if task_of[b] == t and b.startswith("s1mb-generalization-")]]}
    return sum(tasks.values()) / 3, tasks, gen


def borda(models, bench_ids):
    names = list(models)
    n = len(names)
    pts = defaultdict(float)
    for b in bench_ids:
        vals = sorted(((models[m]["scores"][b], m) for m in names), key=lambda x: -x[0])
        i = 0
        while i < n:  # average the points of tied positions
            j = i
            while j + 1 < n and vals[j + 1][0] == vals[i][0]:
                j += 1
            avg = sum(100 * (n - (r + 1)) / (n - 1) for r in range(i, j + 1)) / (j - i + 1)
            for k in range(i, j + 1):
                pts[vals[k][1]] += avg
            i = j + 1
    return {m: pts[m] / len(bench_ids) for m in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=0, help="also print the top N rows")
    args = ap.parse_args()

    bench_ids, task_of, models = published()
    base_borda = borda(models, bench_ids)
    ours = {}
    for run in args.runs:
        s = json.loads((run / "s1mb" / "summary.json").read_text())
        if (not s["complete"] or s["benchmarks"] != len(bench_ids)
                or any(v is None for v in s["per_benchmark"].values())):
            missing = sorted(b for b, v in s["per_benchmark"].items() if v is None)
            raise SystemExit(f"{run}: incomplete S1MB run ({s['benchmarks']} of {len(bench_ids)} benchmarks, "
                             f"complete={s['complete']}, {len(missing)} without a score"
                             + (f": {', '.join(missing[:5])}" if missing else "") + "); rerun scripts/s1mb_bench.py")
        ours[f"OURS {run.name}"] = {"scores": s["per_benchmark"], "total_params": None}
    cohort = {**models, **ours}
    b_all = borda(cohort, bench_ids)
    rows = []
    for m, info in cohort.items():
        avg, tasks, gen = task_avg(info["scores"], bench_ids, task_of)
        rows.append((b_all[m], avg, tasks, gen, m, info["total_params"]))
    rows.sort(key=lambda r: -r[0])
    print(f"cohort: {len(models)} published + {len(ours)} ours, {len(bench_ids)} benchmarks")
    print(f"{'#':>4} {'model':58s} {'Borda':>6} {'TaskAvg':>7} {'Noul':>6} {'Choice':>6} {'Score':>6} "
          f"{'GNoul':>6} {'GChoi':>6} {'GScor':>6} {'params':>8}")
    for i, (b, avg, t, g, m, tp) in enumerate(rows, 1):
        if i <= args.top or m in ours or "bekko" in m or "Julia" in m or "julia" in m:
            size = f"{tp / 1e6:.0f}M" if tp else "740M" if m in ours else "-"
            print(f"{i:4d} {m[:58]:58s} {b:6.2f} {avg:7.2f} {t['noul']:6.2f} {t['choice']:6.2f} {t['score']:6.2f} "
                  f"{g['noul']:6.2f} {g['choice']:6.2f} {g['score']:6.2f} {size:>8}")
    # check: published models' Borda without our runs, for comparison with the live page
    top = max(base_borda, key=base_borda.get)
    print(f"\nreproduction check (cohort without ours): {top} Borda {base_borda[top]:.2f}, "
          f"Task Avg {task_avg(models[top]['scores'], bench_ids, task_of)[0]:.2f}")


if __name__ == "__main__":
    main()
