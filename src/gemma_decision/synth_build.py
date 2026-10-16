"""`gd-synth-build` - synthetic training rows derived from existing data by code (no teacher model).

Recipes (each new row's gold follows from how it is built):
  catch_all  a choice question gets a "none of these" option; for half the rows the top gold option is removed
             and its probability moves to "none of these" (the other options keep theirs)
  flip       a yes/no (noul) question is asked in reverse: true and false swap, in the targets and in the
             option descriptions
  missing    the evidence is removed (empty case, no media); every target becomes the question's prior:
             the mean gold distribution of that question in the source data, renormalized over the row's options

Inputs: typed-decisions `train` rows outside the fixed validation holdout (split_seed 3407, the same rows
gd-train validates on), and gd-pilot-build train rows outside the mix holdout (stable_frac < 0.05). Test rows
are never read. Output: data/synth/train/<recipe>_<origin>.parquet in the gd-pilot-build schema, so a mix can
list data/synth next to data/pilot.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .mm_build import PREFIX_NOTE, stable_frac
from .pilot_build import SCHEMA

NONE_KEY = "none_of_these"
NONE_TEXT = "None of the other options applies."
FLIP_PREFIX = "Reverse the answer: answer true only if the following does NOT hold. "
EMPTY_STATE = "No case details were provided. There is no evidence to base the decision on."


def _typed_rows(holdout: float = 0.1, split_seed: int = 3407) -> list[dict]:
    """typed-decisions train rows as gd-pilot-build rows, minus gd-train's validation holdout."""
    from datasets import load_dataset

    from .data import row_to_example, split_holdout

    raw = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
    _, val = split_holdout([row_to_example(r) for r in raw], holdout, split_seed)
    val_ids = {ex.id for ex in val}
    out = []
    for r in raw:
        if r["id"] in val_ids:
            continue
        out.append({
            "id": r["id"], "workflow": r["workflow"], "source": "typed_decisions", "license": "apache-2.0",
            "state_text": r["state"], "state_media": r["state"], "state_both": r["state"], "variants": ["text"],
            "images": [], "audio": [], "questions": r["questions"], "gold": r["gold"],
            "label_agreement": r["label_agreement"], "n_questions": r["n_questions"],
        })
    return out


def _pilot_rows(root: Path, holdout: float = 0.05) -> list[dict]:
    rows = []
    for path in sorted((root / "train").glob("*.parquet")):
        rows += [r for r in pq.read_table(path).to_pylist() if stable_frac(r["id"]) >= holdout]
    return rows


def _rebuild(row: dict, rid: str, questions: dict, gold: dict, **overrides) -> dict:
    agreement = {}
    for name, g in gold.items():
        top = max(g["probabilities"], key=g["probabilities"].get)
        g["label"] = top
        g["confidence"] = g["probabilities"][top]
        agreement[name] = {"argmax_agree": g["probabilities"][top] >= 0.5, "argmax_majority": top,
                           "total_variation": 1 - g["probabilities"][top]}
    return {**row, "id": rid, "questions": json.dumps(questions), "gold": json.dumps(gold),
            "label_agreement": json.dumps(agreement), "n_questions": len(questions), **overrides}


def catch_all(row: dict, rng: random.Random) -> dict | None:
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    names = [n for n, q in questions.items() if q["type"] == "choice" and len(q.get("criteria") or {}) >= 3
             and NONE_KEY not in q["criteria"]]
    if not names:
        return None
    name = rng.choice(names)
    q, g = questions[name], gold[name]
    probs = {k: float(g["probabilities"].get(k, 0.0)) for k in q["criteria"]}
    if rng.random() < 0.5:  # remove the top gold option; its mass becomes "none of these"
        top = max(probs, key=probs.get)
        del q["criteria"][top]
        probs[NONE_KEY] = probs.pop(top)
        tag = "removed"
    else:
        probs[NONE_KEY] = 0.0
        tag = "kept"
    q["criteria"][NONE_KEY] = NONE_TEXT
    g["probabilities"] = probs
    return _rebuild(row, f"{row['id']}~catchall-{tag}-{name}", questions, gold)


def flip(row: dict, rng: random.Random) -> dict | None:
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    names = [n for n, q in questions.items() if q["type"] == "noul"]
    if not names:
        return None
    name = rng.choice(names)
    q, g = questions[name], gold[name]
    crit = q.get("criteria") or {"true": "Yes, the statement holds.", "false": "No, the statement does not hold."}
    q["instructions"] = FLIP_PREFIX + q["instructions"]
    q["criteria"] = {"true": crit.get("false"), "false": crit.get("true")}
    p = g["probabilities"]
    g["probabilities"] = {"true": float(p.get("false", 0.0)), "false": float(p.get("true", 0.0))}
    g["noul"] = g["probabilities"]["true"]
    return _rebuild(row, f"{row['id']}~flip-{name}", questions, gold)


def priors(rows: list[dict]) -> dict:
    """Mean gold probability per (source/workflow, question, option key) over the source rows."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for name, g in json.loads(r["gold"]).items():
            for k, p in g["probabilities"].items():
                acc[(r["workflow"], name)][k].append(p)
    return {key: {k: sum(v) / len(v) for k, v in opts.items()} for key, opts in acc.items()}


def missing(row: dict, prior: dict) -> dict:
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    for name, q in questions.items():
        keys = (["true", "false"] if q["type"] == "noul" else
                [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else list(q["criteria"]))
        p = {k: prior.get((row["workflow"], name), {}).get(k, 0.0) + 1e-3 for k in keys}
        z = sum(p.values())
        gold[name]["probabilities"] = {k: v / z for k, v in p.items()}
        if q["type"] == "noul":
            gold[name]["noul"] = gold[name]["probabilities"]["true"]
        if q["type"] == "score":
            gold[name]["score"] = sum(int(k) * v for k, v in gold[name]["probabilities"].items())
    head = ""
    if row["source"] != "typed_decisions":  # keep the one-line framing of the case, drop the evidence
        first = next(row[f"state_{v}"] for v in ("text", "both", "media") if row[f"state_{v}"])
        head = first.split("\n", 1)[0].replace(PREFIX_NOTE, "").strip()
    state = f"{head} {EMPTY_STATE}".strip()
    return _rebuild(row, f"{row['id']}~missing", questions, gold, state_text=state, state_media=state,
                    state_both=state, variants=["text"], images=[], audio=[])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=Path, default=Path("data/pilot"))
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--seed", type=int, default=3407)
    # share of each origin's rows that get each recipe (a row may get several)
    ap.add_argument("--catch-all", type=float, default=0.3)
    ap.add_argument("--flip", type=float, default=0.3)
    ap.add_argument("--missing", type=float, default=0.05)
    args = ap.parse_args()

    origins = {"typed": _typed_rows(), "pilot": _pilot_rows(args.pilot)}
    (args.out / "train").mkdir(parents=True, exist_ok=True)
    for origin, rows in origins.items():
        prior = priors(rows)
        for recipe, frac in (("catch_all", args.catch_all), ("flip", args.flip), ("missing", args.missing)):
            rng = random.Random(f"{args.seed}-{origin}-{recipe}")
            picked = [r for r in rows if rng.random() < frac]
            fn = {"catch_all": lambda r: catch_all(r, rng), "flip": lambda r: flip(r, rng),
                  "missing": lambda r: missing(r, prior)}[recipe]
            built = [x for x in (fn(copy.deepcopy(r)) for r in picked) if x is not None]
            for x in built:
                x["workflow"] = x["workflow"] if origin == "typed" else x["source"]
            pq.write_table(pa.Table.from_pylist(built, schema=SCHEMA), args.out / "train" / f"{recipe}_{origin}.parquet")
            print(f"{recipe:9s} {origin:5s} {len(built):6d} rows from {len(picked)} picked ({len(rows)} available)", flush=True)


if __name__ == "__main__":
    main()
