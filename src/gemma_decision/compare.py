"""`gd-compare runs/frozen runs/full runs/lora` - side-by-side test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ["acc", "acc_agreed", "annot_agree", "oracle", "row_acc", "ece", "brier", "kl", "score_mae"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", default="overall", help="e.g. overall, type=score, workflow=customer_service")
    args = ap.parse_args()

    print(f"split={args.split} slice={args.slice}\n")
    print("| run | " + " | ".join(COLS) + " | trainable enc | minutes |")
    print("|---" * (len(COLS) + 3) + "|")
    for r in args.runs:
        res = json.loads((r / f"eval_{args.split}.json").read_text())
        m, meta = res[args.slice], res.get("meta", {})
        cells = [f"{m[c]:.3f}" if c in m else "" for c in COLS]
        enc = f"{meta.get('trainable_encoder_params', 0) / 1e6:.1f}M"
        print(f"| {r.name} | " + " | ".join(cells) + f" | {enc} | {meta.get('train_minutes', '')} |")


if __name__ == "__main__":
    main()
