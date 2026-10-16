"""Train GemmaDecision runs on Modal GPUs, one container per config, all in parallel.

  uv run modal run modal_app.py::main --configs configs/full_e5.yaml,configs/lora.yaml
  uv run modal run modal_app.py::main --configs configs/full_e5.yaml --gpu H100 --set "batch_size=16 grad_accum=1"
  uv run modal run modal_app.py::main --configs configs/full.yaml --max-steps 4 --limit 32 --suffix smoke
  uv run modal run modal_app.py::bench                             # cost/benefit across GPUs
  uv run modal run modal_app.py::download --name full_e5          # checkpoint + evals -> runs/modal/full_e5/
  uv run modal run modal_app.py::upload_data                       # data/mm-decisions -> data volume

Same code path as local `gd-train`: the config YAML is read locally and sent with the call, so local
edits to configs never need an image rebuild. On CUDA, `amp: auto` trains in bf16 autocast (the model
card rules out fp16, not bf16). Each run's eval JSON comes back to runs/modal/<name>/; checkpoints stay
on the `gemma-decision-runs` volume until downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "gemma-decision"
GPU_DEFAULT = "L40S"  # 48 GB, bf16; override with --gpu (A10G, A100-80GB, H100, ...)
WORK = "/work"  # container cwd: runs/ and data/ resolve here, as they do locally

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .uv_pip_install(
        "torch==2.14.1", "torchvision==0.29.1", "transformers>=5.19", "peft>=0.21", "datasets>=5.1",
        "accelerate>=1.15", "safetensors>=0.8", "pyyaml", "tqdm", "pillow", "soundfile", "librosa", "pyarrow",
    )
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("gemma_decision")
)

hf_cache = modal.Volume.from_name("gemma-decision-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("gemma-decision-runs", create_if_missing=True)
data_vol = modal.Volume.from_name("gemma-decision-data", create_if_missing=True)
VOLUMES = {"/cache/hf": hf_cache, f"{WORK}/runs": runs_vol, f"{WORK}/data": data_vol}

app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU_DEFAULT, volumes=VOLUMES, timeout=8 * 3600, cpu=8, memory=32768)
def train(config_yaml: str, name: str, overrides: list[str], max_steps: int | None = None,
          limit: int | None = None) -> dict:
    import os
    import sys

    os.chdir(WORK)
    cfg_path = Path(f"/tmp/{name}.yaml")
    cfg_path.write_text(config_yaml)
    argv = ["gd-train", str(cfg_path), "--name", name]
    if max_steps:
        argv += ["--max-steps", str(max_steps)]
    if limit:
        argv += ["--limit", str(limit)]
    if overrides:
        argv += ["--set", *overrides]

    from gemma_decision import train as train_mod

    sys.argv = argv
    try:
        train_mod.main()
    finally:
        hf_cache.commit()
        runs_vol.commit()

    run_dir = Path(WORK) / "runs" / name
    return {f.name: f.read_text() for f in run_dir.iterdir() if f.suffix in (".json", ".jsonl", ".yaml")}


@app.local_entrypoint()
def main(configs: str, gpu: str = GPU_DEFAULT, set: str = "", max_steps: int = 0, limit: int = 0,
         suffix: str = ""):
    import yaml

    overrides = set.split() if set else []
    calls = []
    for path in [c.strip() for c in configs.split(",") if c.strip()]:
        text = Path(path).read_text()
        name = (yaml.safe_load(text) or {}).get("name") or Path(path).stem
        name = f"{name}-{suffix}" if suffix else name
        fn = train.with_options(gpu=gpu)
        calls.append((name, fn.spawn(text, name, overrides, max_steps or None, limit or None)))
        print(f"spawned {name} on {gpu}")

    out_root = Path("runs/modal")
    for name, call in calls:
        try:
            files = call.get()
        except Exception as e:  # keep collecting the other runs
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            continue
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (out / fname).write_text(content)
        test = json.loads(files.get("eval_test.json", "{}"))
        o = test.get("overall", {})
        meta = test.get("meta", {})
        print(f"[{name}] acc {o.get('acc', float('nan')):.3f}  annot_agree {o.get('annot_agree', float('nan')):.3f}"
              f"  kl {o.get('kl', float('nan')):.3f}  minutes {meta.get('train_minutes')}  -> {out}")


@app.function(gpu=GPU_DEFAULT, volumes=VOLUMES, timeout=3 * 3600)
def evaluate(name: str, mm: bool = True, probe: bool = True) -> dict:
    """gd-mm-eval (text / media / both on data/mm-decisions) and gd-mm-probe for a run on the volume."""
    import os
    import sys

    os.chdir(WORK)
    from gemma_decision import mm_eval, mm_probe

    run_dir = f"runs/{name}"
    try:
        if mm:
            sys.argv = ["gd-mm-eval", run_dir]
            mm_eval.main()
        if probe:
            sys.argv = ["gd-mm-probe", run_dir]
            mm_probe.main()
    finally:
        runs_vol.commit()
    d = Path(WORK) / run_dir
    return {f.name: f.read_text() for f in d.iterdir() if f.name in ("eval_mm.json", "probe_mm.json")}


@app.local_entrypoint()
def eval_runs(names: str, gpu: str = GPU_DEFAULT):
    """Evaluate several runs in parallel; results land in runs/modal/<name>/."""
    calls = [(n, evaluate.with_options(gpu=gpu).spawn(n)) for n in [x.strip() for x in names.split(",") if x.strip()]]
    for n, call in calls:
        try:
            files = call.get()
        except Exception as e:
            print(f"[{n}] FAILED: {type(e).__name__}: {e}")
            continue
        out = Path("runs/modal") / n
        out.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (out / fname).write_text(content)
        print(f"[{n}] -> {out}: {sorted(files)}")


@app.function(volumes=VOLUMES, cpu=4, memory=65536, timeout=4 * 3600)
def build_source(source: str, scale: float = 1.0, out: str = "data/pilot") -> str:
    """Build one gd-pilot-build source on a CPU container, straight onto the data volume."""
    import os
    import sys

    os.chdir(WORK)
    from gemma_decision import pilot_build

    sys.argv = ["gd-pilot-build", "--sources", source, "--scale", str(scale), "--out", out]
    try:
        pilot_build.main()
    finally:
        hf_cache.commit()
        data_vol.commit()
    return source


@app.function(volumes=VOLUMES, cpu=4, memory=65536, timeout=4 * 3600)
def build_synth(args: list[str]) -> str:
    """gd-synth-build on a CPU container, reading data/pilot and writing data/synth on the data volume."""
    import os
    import sys

    os.chdir(WORK)
    from gemma_decision import synth_build

    sys.argv = ["gd-synth-build", *args]
    try:
        synth_build.main()
    finally:
        hf_cache.commit()
        data_vol.commit()
    return "ok"


@app.local_entrypoint()
def synth(args: str = ""):
    """Build data/synth from data/pilot (+ typed-decisions train) on Modal."""
    print(build_synth.remote(args.split()))
    print(sorted(e.path for e in data_vol.listdir("synth/train")))


@app.function(volumes=VOLUMES, cpu=4, memory=32768, timeout=3600)
def check_pilot(root: str = "data/pilot", max_length: int = 2048) -> dict:
    """Encode every rendering of every pilot row (train + test) the way training will; report failures."""
    import os

    os.chdir(WORK)
    from transformers import AutoTokenizer

    from gemma_decision.data import check_encodable, load_pilot_examples

    tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
    return {split: check_encodable(load_pilot_examples(root, split), tok, max_length) for split in ("train", "test")}


@app.local_entrypoint()
def preflight(root: str = "data/pilot", max_length: int = 2048):
    res = check_pilot.remote(root, max_length)
    for split, stats in res.items():
        for wf, s in sorted(stats.items()):
            flag = "  <-- FAILURES" if s["failed"] else ""
            print(f"{split:5s} {wf:16s} ok {s['ok']:6d} failed {s['failed']:4d} max_tokens {s['max_tokens']:5d}{flag}")


@app.local_entrypoint()
def pilot(sources: str = "", scale: float = 1.0, out: str = "data/pilot"):
    """Build the pilot mix on Modal, one CPU container per source in parallel (no GPU)."""
    from gemma_decision.pilot_build import PILOT

    names = [s.strip() for s in sources.split(",") if s.strip()] or list(PILOT)
    calls = [(n, build_source.spawn(n, scale, out)) for n in names]
    for n, call in calls:
        try:
            call.get()
            print(f"built {n}")
        except Exception as e:
            print(f"[{n}] FAILED: {type(e).__name__}: {e}")
    for split in ("train", "test"):
        try:
            print(split, sorted(e.path for e in data_vol.listdir(f"{out.removeprefix('data/')}/{split}")))
        except Exception:
            pass


# $/hour = Modal's per-second list price x 3600 (modal.com/pricing, 2026-10-14). T4 is left out: no bf16.
GPU_PRICE_PER_HOUR = {
    "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
    "RTX-PRO-6000": 3.03, "H100": 3.95, "H200": 4.54, "B200": 6.25,
}


@app.local_entrypoint()
def bench(gpus: str = "L4,A10,L40S,A100-40GB,H100", config: str = "configs/full.yaml",
          set: str = "batch_size=16 grad_accum=1 gradient_checkpointing=true", steps: int = 30):
    """Same short run on several GPUs in parallel; reports throughput and $ per 1k training rows.

    Timing uses log.jsonl between step 10 and the last logged step, so container start, model download
    and evaluation are excluded (they are billed, but are a fixed cost per run, not per row).
    """
    import yaml

    overrides = set.split()
    batch = int(dict(o.split("=", 1) for o in overrides).get("batch_size", 8))
    text = Path(config).read_text()
    base = (yaml.safe_load(text) or {}).get("name", Path(config).stem)
    calls = []
    for gpu in [g.strip() for g in gpus.split(",") if g.strip()]:
        name = f"bench-{base}-{gpu.lower()}"
        calls.append((gpu, name, train.with_options(gpu=gpu).spawn(text, name, overrides, steps, 800)))
        print(f"spawned {name}")

    rows = []
    for gpu, name, call in calls:
        try:
            files = call.get()
        except Exception as e:
            print(f"[{gpu}] FAILED: {type(e).__name__}: {e}")
            continue
        logs = [json.loads(line) for line in files["log.jsonl"].splitlines()]
        steps_logged = [l for l in logs if "elapsed_s" in l]
        first, last = steps_logged[0], steps_logged[-1]
        sec_per_step = (last["elapsed_s"] - first["elapsed_s"]) / max(1, last["step"] - first["step"])
        rows_per_s = batch / sec_per_step
        price = GPU_PRICE_PER_HOUR.get(gpu, float("nan"))
        usd_per_1k = price / 3600 / rows_per_s * 1000
        rows.append((usd_per_1k, gpu, sec_per_step, rows_per_s, price))
    print("\n| gpu | $/h | s/step | rows/s | $ per 1k rows | $ per 15k-row epoch |")
    print("|---|---|---|---|---|---|")
    for usd, gpu, sps, rps, price in sorted(rows):
        print(f"| {gpu} | {price:.2f} | {sps:.2f} | {rps:.1f} | {usd:.4f} | {usd * 15:.2f} |")


@app.local_entrypoint()
def download(name: str):
    """Copy a run (checkpoint included) from the runs volume to runs/modal/<name>/."""
    out = Path("runs/modal") / name
    out.mkdir(parents=True, exist_ok=True)
    for entry in runs_vol.listdir(name):
        target = out / Path(entry.path).name
        with target.open("wb") as f:
            runs_vol.read_file_into_fileobj(entry.path, f)
        print(f"{entry.path} -> {target}")


@app.local_entrypoint()
def upload_data(local: str = "data/mm-decisions", remote: str = "mm-decisions"):
    """Upload a gd-mm-build output dir to the data volume (seen as data/<remote> inside containers)."""
    with data_vol.batch_upload(force=True) as batch:
        batch.put_directory(local, remote)
    print(f"uploaded {local} -> gemma-decision-data:/{remote}")
