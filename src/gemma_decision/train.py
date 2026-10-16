"""`gd-train configs/full.yaml` - train a decision model, calibrate on the holdout, evaluate on test.

Outputs in runs/<name>/: config.yaml, trainable.safetensors (only weights that differ from the base
checkpoint), log.jsonl, eval_val.json, eval_test.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from .data import (
    collate, collate_mm, encode, length_bucketed_batches, load_examples, load_pilot_examples, media_reserve,
    pick_rendering, split_holdout,
)
from .mm_build import stable_frac
from .evaluate import fit_temperatures, format_report, report, run
from .losses import decision_loss
from .model import MODEL_ID, ClefHead, DecisionModel, load_encoder, to_device

DEFAULTS = {
    "name": "run",
    "mode": "full",
    "model_id": MODEL_ID,
    "dataset": "LocalLLaMA/typed-decisions",
    "dataset_config": "all",
    "max_length": 1024,
    "batch_size": 4,
    "grad_accum": 2,
    "gradient_checkpointing": True,
    "mps_memory_fraction": 0.75,  # fail fast with OOM instead of swapping the machine
    "max_batch_tokens": None,  # split a length-bucketed batch when its estimated tokens exceed this (media runs)
    "save_every_steps": None,  # also write latest.safetensors every N optimizer steps (crash insurance)
    "amp": "auto",  # "auto" = bf16 autocast on CUDA, fp32 elsewhere; "bf16" | "fp32" to force (never fp16)
    "epochs": 3,
    "max_steps": None,
    "lr_encoder": 2e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "grad_clip": 1.0,
    "augment": True,
    "holdout": 0.1,
    "seed": 3407,
    "select_by": "kl",  # best epoch by val "kl" (lower), "acc" (higher), or "typed_kl" (KL on typed-decisions val rows only)
    # Extra training data from gd-pilot-build (None = typed-decisions only). Example:
    # mix: {pilot_dir: data/pilot (or a list of dirs), sources: null, typed_repeat: 3, holdout: 0.05,
    #       source_caps: {banking77: 300},  # optional: fixed random subset per workflow (replay)
    #       weights: {media: 0.5, both: 0.3, text: 0.2}}
    "mix": None,
    "init_from": None,  # a run dir (or .safetensors) whose trained weights start this run
    "loss": {"brier": 1.0, "ordinal": 0.5, "hard_weight": 0.0},
    "head": {"d": 512, "heads": 8, "n_cross": 2, "n_joint": 2, "dropout": 0.1},
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    for k, v in (yaml.safe_load(path.read_text()) or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """`key=value` or `section.key=value`; values parsed as YAML (numbers, bools, lists)."""
    for pair in pairs:
        key, _, raw = pair.partition("=")
        target = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            target = target.setdefault(p, {})
        target[leaf] = yaml.safe_load(raw)
    return cfg


def autocast(cfg: dict, device: torch.device):
    use_bf16 = cfg.get("amp") == "bf16" or (cfg.get("amp", "auto") == "auto" and device.type == "cuda")
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)


def build_model(cfg: dict) -> DecisionModel:
    encoder = load_encoder(cfg["mode"], cfg["model_id"], cfg["lora"])
    if cfg.get("gradient_checkpointing") and cfg["mode"] != "frozen":
        encoder.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = ClefHead(d_in=encoder.config.text_config.embedding_dim, **cfg["head"])
    return DecisionModel(encoder, head, cfg["mode"])


def load_checkpoint(run_dir: Path):
    from transformers import AutoTokenizer

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(load_file(run_dir / "trainable.safetensors"), strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    device = pick_device()
    return model.to(device), AutoTokenizer.from_pretrained(cfg["model_id"]), cfg, device


def param_groups(model: DecisionModel, cfg: dict):
    groups = []
    for module, lr in ((model.encoder, cfg["lr_encoder"]), (model.head, cfg["lr_head"])):
        decay, no_decay = [], []
        for n, p in module.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim < 2 or "norm" in n or "emb" in n else decay).append(p)
        groups += [
            {"params": decay, "lr": lr, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    return [g for g in groups if g["params"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--max-steps", type=int, help="override config (for smoke tests)")
    ap.add_argument("--name", help="override run name")
    ap.add_argument("--limit", type=int, help="use only N train / N//4 eval rows (smoke tests)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. batch_size=16 grad_accum=1 loss.hard_weight=0.2")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.name:
        cfg["name"] = args.name
    run_dir = Path("runs") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = pick_device()
    if device.type == "mps" and cfg.get("mps_memory_fraction"):
        torch.mps.set_per_process_memory_fraction(cfg["mps_memory_fraction"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    train_ex, val_ex = split_holdout(load_examples(cfg["dataset"], cfg["dataset_config"], "train"), cfg["holdout"],
                                     cfg.get("split_seed", 3407))  # fixed across training seeds
    test_ex = load_examples(cfg["dataset"], cfg["dataset_config"], "test")
    pilot_test, processor, weights = [], None, {}
    typed_val_ids = {ex.id for ex in val_ex}
    if cfg.get("mix"):
        from transformers import AutoProcessor

        mix = cfg["mix"]
        weights = mix.get("weights") or {"media": 0.5, "both": 0.3, "text": 0.2}
        dirs = mix["pilot_dir"] if isinstance(mix["pilot_dir"], list) else [mix["pilot_dir"]]
        pilot = [ex for d in dirs for ex in load_pilot_examples(d, "train", mix.get("sources"))]
        if mix.get("source_caps"):  # e.g. {banking77: 300}: keep a fixed random subset of those sources
            cap_rng = random.Random(f"caps-{cfg.get('split_seed', 3407)}")
            by_src: dict[str, list] = {}
            synthetic = [ex for ex in pilot if "~" in ex.id]  # gd-synth-build rows are never capped
            for ex in pilot:
                if "~" not in ex.id:
                    by_src.setdefault(ex.workflow, []).append(ex)
            pilot = synthetic
            for wf, exs in by_src.items():
                cap = mix["source_caps"].get(wf)
                if cap is not None and len(exs) > cap:
                    exs = cap_rng.sample(exs, cap)
                pilot += exs
        pilot_val = [pick_rendering(ex, None, weights) for ex in pilot if stable_frac(ex.id) < mix.get("holdout", 0.05)]
        pilot_train = [ex for ex in pilot if stable_frac(ex.id) >= mix.get("holdout", 0.05)]
        pilot_test = [pick_rendering(ex, None, weights) for d in dirs for ex in load_pilot_examples(d, "test", mix.get("sources"))]
        train_ex = train_ex * mix.get("typed_repeat", 1) + pilot_train
        val_ex = val_ex + pilot_val
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
    if args.limit:
        rng.shuffle(train_ex)
        train_ex, val_ex, test_ex = train_ex[: args.limit], val_ex[: args.limit // 4], test_ex[: args.limit // 4]
        pilot_test = pilot_test[: args.limit // 4]
    print(f"train {len(train_ex)}  val {len(val_ex)}  test {len(test_ex)}  pilot test {len(pilot_test)}  device {device}")

    model = build_model(cfg).to(device)
    if cfg.get("init_from"):
        src = Path(cfg["init_from"])
        src = src / "trainable.safetensors" if src.is_dir() else src
        missing, unexpected = model.load_state_dict(load_file(src), strict=False)
        print(f"init_from {src}: loaded, {len(unexpected)} unexpected keys")
        model.head.log_temp.zero_()  # temperatures are refitted at the end of this run
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"mode={cfg['mode']}  trainable encoder {n_enc / 1e6:.1f}M  head {n_head / 1e6:.1f}M")

    def est_len(ex):
        ex = pick_rendering(ex, None, weights)
        return len(tokenizer(ex.state).input_ids) + sum(len(q.instructions) // 3 + 12 * len(q.keys) for q in ex.questions) \
            + (media_reserve(ex) if "<|" in ex.state else 0)

    lengths = [est_len(ex) for ex in train_ex]
    steps_per_epoch = math.ceil(len(train_ex) / cfg["batch_size"] / cfg["grad_accum"])
    total = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(total * cfg["warmup_ratio"]))
    opt = torch.optim.AdamW(param_groups(model, cfg))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    log = (run_dir / "log.jsonl").open("w")
    step, micro, t0 = 0, 0, time.time()
    best_score, best_state = float("-inf"), None
    skipped: dict[str, int] = {}
    pbar = tqdm(total=total, desc=cfg["name"])
    epoch = 0
    def token_capped(batches):
        """Split batches whose estimated token total exceeds max_batch_tokens (long media rows go in smaller groups)."""
        cap = cfg.get("max_batch_tokens")
        for b in batches:
            if not cap:
                yield b
                continue
            cur, used = [], 0
            for i in b:
                if cur and used + lengths[i] > cap:
                    yield cur
                    cur, used = [], 0
                cur.append(i)
                used += lengths[i]
            if cur:
                yield cur

    oom_skips = 0
    while step < total:
        model.train()
        for idx in token_capped(length_bucketed_batches(lengths, cfg["batch_size"], rng)):
            exs, enc = [], []
            for i in idx:
                ex = pick_rendering(train_ex[i], rng, weights)
                media = "<|image|>" in ex.state or "<|audio|>" in ex.state
                try:
                    enc.append(encode(tokenizer, ex.state, ex.questions, cfg["max_length"],
                                      rng if cfg["augment"] else None, reserve=media_reserve(ex) if media else 0))
                    exs.append(ex)
                except ValueError as err:  # one oversized row must not end a long run
                    skipped[ex.workflow] = skipped.get(ex.workflow, 0) + 1
                    if sum(skipped.values()) <= 5:
                        print(f"skipping {ex.id}: {err}", flush=True)
            if not exs:
                continue
            has_media = any("<|image|>" in ex.state or "<|audio|>" in ex.state for ex in exs)
            batch = collate_mm(enc, exs, processor) if has_media else collate(enc, tokenizer.pad_token_id)
            batch.pop("q_meta")
            batch = to_device(batch, device)
            try:
                with autocast(cfg, device):
                    logits = model(batch, calibrated=False)
                loss, parts = decision_loss(logits.float(), batch, **cfg["loss"])
                (loss / cfg["grad_accum"]).backward()
            except RuntimeError as err:  # out of memory on one batch: drop it (and the partial accumulation), go on
                if "out of memory" not in str(err):
                    raise
                oom_skips += 1
                print(f"\nOOM on a batch of {len(exs)} rows (est. {sum(lengths[i] for i in idx)} tokens); skipped "
                      f"({oom_skips} so far)", flush=True)
                del batch
                opt.zero_grad(set_to_none=True)
                micro = 0
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            set_lr(step)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()  # long mixed-shape runs fragment the MPS heap
            if cfg.get("save_every_steps") and step % cfg["save_every_steps"] == 0:
                save_file({k: v.contiguous() for k, v in model.trainable_state_dict().items()}, run_dir / "latest.safetensors")
            if step % 10 == 0:
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item(), **parts, "oom_skips": oom_skips,
                                      "elapsed_s": round(time.time() - t0, 2)}) + "\n")
                log.flush()
            if step >= total:
                break
        epoch += 1
        # Validate at each epoch end (and at the end of a max_steps run); keep the best by KL to gold.
        val_records = run(model, tokenizer, val_ex, device, cfg["batch_size"], cfg["max_length"], calibrated=False,
                          processor=processor)[0]
        val = report(val_records)["overall"]
        # typed-decisions rows carry soft annotator distributions; pilot rows are mostly one-hot, so their KL
        # rewards under-confidence and must not pick the epoch
        val["typed_kl"] = report([r for r in val_records if r["id"] in typed_val_ids] or val_records)["overall"]["kl"]
        print(f"\nepoch {epoch} step {step}: val acc {val['acc']:.3f}  annot_agree {val['annot_agree']:.3f}  kl {val['kl']:.3f}  ece {val['ece']:.3f}", flush=True)
        log.write(json.dumps({"step": step, "epoch": epoch, "val": val, "skipped": skipped}) + "\n")
        log.flush()
        score = {"kl": -val["kl"], "typed_kl": -val["typed_kl"], "acc": val["acc"]}[cfg["select_by"]]
        if score > best_score:
            best_score, best_state = score, model.trainable_state_dict()
            # persisted now, not only at the end: a stopped run still leaves its best weights (uncalibrated)
            save_file({k: v.contiguous() for k, v in best_state.items()}, run_dir / "best_partial.safetensors")
            (run_dir / "best_partial.json").write_text(json.dumps({"epoch": epoch, "step": step, "select_by": cfg["select_by"],
                                                                   "score": score, "val": val}, indent=2))
    pbar.close()

    model.load_state_dict(best_state, strict=False)
    temps = fit_temperatures(model, tokenizer, val_ex, device, cfg["batch_size"], cfg["max_length"], processor)
    print(f"temperatures (choice, noul, score): {[round(t, 3) for t in temps]}")
    save_file({k: v.contiguous() for k, v in model.trainable_state_dict().items()}, run_dir / "trainable.safetensors")

    evals = [("val", val_ex), ("test", test_ex)] + ([("pilot_test", pilot_test)] if pilot_test else [])
    for split, exs in evals:
        res = report(run(model, tokenizer, exs, device, cfg["batch_size"], cfg["max_length"], processor=processor)[0])
        res["meta"] = {"train_minutes": round((time.time() - t0) / 60, 1), "steps": step, "temperatures": temps,
                       "trainable_encoder_params": n_enc, "head_params": n_head}
        (run_dir / f"eval_{split}.json").write_text(json.dumps(res, indent=2))
        print(f"\n== {split} ==\n" + format_report({k: v for k, v in res.items() if k != "meta"}))


if __name__ == "__main__":
    main()
