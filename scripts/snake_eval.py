"""Play Snake with decision models and reference policies on the same seeded games.

  uv run python scripts/snake_eval.py --policies teacher greedy random runs/full_e5-modal --modes grid hints image
  data/julia/.venv/bin/python -I scripts/snake_eval.py --julia data/julia/Julia-1 --modes grid hints

Each tick the model answers one `choice` question over the legal moves (see gemma_decision/snake.py); the
argmax is played. Results: runs/snake/<policy>__<mode>.json (per-game scores, teacher agreement, latency,
and the frames of the first 2 games for replays).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # also importable from Julia's env
from gemma_decision.snake import greedy, make_random, play, teacher  # noqa: E402


def model_policy(run_dir: Path, mode: str):
    import torch
    from transformers import AutoProcessor

    from gemma_decision.data import Example, collate, collate_mm, encode, media_reserve, parse_questions
    from gemma_decision.model import to_device
    from gemma_decision.train import load_checkpoint

    model, tok, cfg, device = load_checkpoint(run_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg["model_id"]) if mode == "image" else None
    latency = []

    @torch.inference_mode()
    def policy(game):
        t = time.perf_counter()
        row = game.decision(mode)
        ex = Example("snake", "snake", row["state"], parse_questions(row["questions"]), images=row["images"])
        enc = [encode(tok, ex.state, ex.questions, 2048, reserve=media_reserve(ex) if ex.images else 0)]
        batch = collate_mm(enc, [ex], processor) if ex.images else collate(enc, tok.pad_token_id)
        meta = batch.pop("q_meta")
        logits = model(to_device(batch, device))[0, : len(meta[0][1].keys)]
        move = meta[0][1].keys[int(logits.argmax())]
        latency.append(time.perf_counter() - t)
        return move

    return policy, latency


def julia_policy(checkpoint: Path, mode: str):
    sys.path.insert(0, str(checkpoint.resolve()))
    from julia.router.engine import FastEngine

    engine = FastEngine(checkpoint.resolve(), device="cpu", transformer_backend="torch", strict_encoding=True,
                        max_length=1024, head_length=512, batch_size=16, marker_only_head=False)
    latency = []

    def policy(game):
        t = time.perf_counter()
        row = game.decision(mode)
        ans = engine.predict(state=row["state"], questions=row["questions"])["answers"]["move"]
        latency.append(time.perf_counter() - t)
        return ans["choice"]

    return policy, latency


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="*", default=[], help="teacher | greedy | random | a run directory")
    ap.add_argument("--julia", type=Path, help="Julia-1 checkpoint dir (run inside Julia's env)")
    ap.add_argument("--modes", nargs="+", default=["grid", "hints", "image"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("runs/snake"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in args.policies:
        if p in ("teacher", "greedy", "random"):
            jobs.append((p, "-", None))
        else:
            jobs += [(Path(p).name, m, Path(p)) for m in args.modes]
    if args.julia:
        jobs += [("julia-1", m, args.julia) for m in args.modes if m != "image"]  # Julia is text-only

    for name, mode, src in jobs:
        latency = []
        if name in ("teacher", "greedy"):
            policy = {"teacher": teacher, "greedy": greedy}[name]
        elif name == "random":
            policy = None
        elif name == "julia-1":
            policy, latency = julia_policy(src, mode)
        else:
            policy, latency = model_policy(src, mode)
        games = []
        for i, seed in enumerate(range(args.seed0, args.seed0 + args.games)):
            pol = make_random(seed) if name == "random" else policy
            games.append(play(pol, seed, max_steps=args.max_steps, record=i < 2))
        summary = {
            "policy": name, "mode": mode, "games": len(games),
            "mean_score": statistics.mean(g["score"] for g in games),
            "median_score": statistics.median(g["score"] for g in games),
            "mean_steps": statistics.mean(g["steps"] for g in games),
            "teacher_agreement": statistics.mean(g["teacher_agreement"] for g in games),
            "deaths": {d: sum(g["death"] == d for g in games) for d in sorted({g["death"] for g in games})},
            "median_decision_ms": statistics.median(latency) * 1000 if latency else None,
        }
        (args.out / f"{name}__{mode}.json").write_text(json.dumps({"summary": summary, "games": games}))
        lat = f"{summary['median_decision_ms']:.0f} ms" if latency else "-"
        print(f"{name:20s} {mode:6s} score {summary['mean_score']:5.1f}  steps {summary['mean_steps']:6.1f}  "
              f"agree {summary['teacher_agreement']:.2f}  {lat:>7s}  deaths {summary['deaths']}", flush=True)


if __name__ == "__main__":
    main()
