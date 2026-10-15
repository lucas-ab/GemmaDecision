"""`gd-mm-probe runs/frozen runs/lora runs/full` - head-free check of image/audio <-> text alignment.

Embeds each row's media alone and its text evidence alone with the run's encoder (mean pooling + L2 norm,
the model's normal embedding mode) and measures cross-modal retrieval within each workflow:
audio -> transcript (call_center), receipt image -> OCR text (receipts), post image -> post text (crisis).
`frozen` is the untouched base encoder, so a drop for lora/full means fine-tuning moved the media path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import load_mm_examples

TEXT_FIELD = {"call_center": "Call transcript: ", "receipts": "Receipt text (OCR):\n", "crisis_response": "Post text: "}


def evidence_text(ex) -> str:
    return ex.state.split(TEXT_FIELD[ex.workflow], 1)[1]


@torch.no_grad()
def embed(encoder, processor, device, text: str, images=None, audio=None) -> torch.Tensor:
    out = processor(text=text, images=images or None, audio=audio or None, return_tensors="pt")
    out = {k: v.to(device) for k, v in out.items()}
    h = encoder(**out).last_hidden_state[0]
    return F.normalize(h.mean(0), dim=-1).float().cpu()


def retrieval(q: torch.Tensor, d: torch.Tensor) -> dict:
    sims = q @ d.T
    ranks = (sims > sims.diag().unsqueeze(1)).sum(1)  # 0 = correct item ranked first
    return {"r@1": (ranks == 0).float().mean().item(), "r@5": (ranks < 5).float().mean().item(),
            "mrr": (1.0 / (ranks + 1).float()).mean().item(), "n": len(q)}


def main():
    from transformers import AutoProcessor

    from .train import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/mm-decisions"))
    args = ap.parse_args()

    text_ex = {e.id: e for e in load_mm_examples(args.data, "test", "text")}
    media_ex = load_mm_examples(args.data, "test", "media")
    workflows = sorted({e.workflow for e in media_ex})
    results = {}
    for run_dir in args.runs:
        model, _, cfg, device = load_checkpoint(run_dir)
        model.eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        res = {}
        for wf in workflows:
            rows = [e for e in media_ex if e.workflow == wf]
            m = torch.stack([embed(model.encoder, processor, device, "<|image|>" if e.images else "<|audio|>",
                                   e.images, e.audio) for e in rows])
            t = torch.stack([embed(model.encoder, processor, device, f"title: none | text: {evidence_text(text_ex[e.id])}")
                             for e in rows])
            res[wf] = retrieval(m, t)
            print(f"{run_dir.name:8s} {wf:16s} r@1 {res[wf]['r@1']:.3f}  r@5 {res[wf]['r@5']:.3f}  mrr {res[wf]['mrr']:.3f}", flush=True)
        (run_dir / "probe_mm.json").write_text(json.dumps(res, indent=2))
        results[run_dir.name] = res
        del model

    print("\nmedia -> text retrieval, MRR (R@1)\n")
    print("| run | " + " | ".join(workflows) + " |")
    print("|---" * (len(workflows) + 1) + "|")
    for name, res in results.items():
        print(f"| {name} | " + " | ".join(f"{res[w]['mrr']:.3f} ({res[w]['r@1']:.3f})" for w in workflows) + " |")


if __name__ == "__main__":
    main()
