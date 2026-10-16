"""Local training dashboard (MiMo-RL style): `uv run python scripts/dashboard.py` -> http://127.0.0.1:8770

Reads runs/<name>/ and runs/modal/<name>/ live on every request (log.jsonl, config.yaml, eval_*.json,
best_partial.json, the run's .out progress bar) plus runs/notices.jsonl, and serves scripts/dashboard.html.
Standard library only; the Modal spend comes from `modal billing summary`, cached for 10 minutes.
"""

import argparse
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HTML = Path(__file__).with_name("dashboard.html")
SKIP = {"julia", "snake", "modal"}
MODAL_RATE_PER_H = 1.95 + 8 * 0.0000131 * 3600 + 32 * 0.00000222 * 3600  # L40S + 8 cores + 32 GiB
PROGRESS_RE = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)")

_billing = {"t": 0.0, "metered": None, "error": None}


def _billing_refresh():
    try:
        out = subprocess.run(["uv", "run", "modal", "billing", "summary"], cwd=ROOT, capture_output=True,
                             text=True, timeout=60).stdout
        m = re.search(r"Metered Cost:\s+([\d.,]+)", out)
        _billing.update(metered=float(m.group(1).replace(",", "")) if m else None, error=None if m else "no data")
    except Exception as e:  # noqa: BLE001 - shown in the header, never fatal
        _billing["error"] = str(e)[:80]
    _billing["t"] = time.time()


def billing():
    if time.time() - _billing["t"] > 600:
        _billing["t"] = time.time()
        threading.Thread(target=_billing_refresh, daemon=True).start()
    return {"metered": _billing["metered"], "limit": 30.0, "error": _billing["error"]}


def _hms(s: str) -> float | None:
    if "?" in s:
        return None
    parts = [float(p) for p in s.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _out_file(d: Path) -> Path | None:
    for p in (d / "train.out", RUNS / f"{d.name}.out"):
        if p.exists():
            return p
    return None


def progress(out: Path | None):
    if out is None:
        return None, False
    with out.open("rb") as f:
        f.seek(max(0, out.stat().st_size - 20000))
        tail = f.read().decode("utf-8", "replace")
    hits = PROGRESS_RE.findall(tail)
    # tqdm prints one more bar after a traceback, so look for the traceback near the end, not after the bar
    crashed = "Traceback" in tail[-6000:] and time.time() - out.stat().st_mtime > 60
    if not hits:
        return None, crashed
    step, total, elapsed, eta, rate, unit = hits[-1]
    s_per_it = float(rate) if unit == "s/it" else 1 / float(rate)
    return {"step": int(step), "total": int(total), "elapsed_s": _hms(elapsed), "eta_s": _hms(eta),
            "s_per_it": s_per_it}, crashed


def load_run(d: Path) -> dict:
    lines = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]
    steps = [l for l in lines if "loss" in l]
    vals = [{"epoch": l["epoch"], "step": l["step"], **l["val"], "skipped": l.get("skipped")} for l in lines if "val" in l]
    cfg = {}
    try:
        import yaml  # optional; config summary only

        cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
    except Exception:  # noqa: BLE001
        pass
    out = _out_file(d)
    prog, crashed = progress(out)
    test, pilot = _read_json(d / "eval_test.json"), _read_json(d / "eval_pilot_test.json")
    last_write = max(p.stat().st_mtime for p in [d / "log.jsonl", *([out] if out else [])])
    if test:
        status = "finished"
    elif crashed:
        status = "crashed"
    elif out and time.time() - out.stat().st_mtime < 300:
        status = "running"
    else:
        status = "stopped"
    where = "modal" if d.parent.name == "modal" or (RUNS / "modal" / d.name).exists() else "local"
    elapsed = (prog or {}).get("elapsed_s") or (steps[-1]["elapsed_s"] if steps else 0) or 0
    if test and test.get("meta", {}).get("train_minutes"):
        elapsed = test["meta"]["train_minutes"] * 60
    mix = cfg.get("mix") or {}
    return {
        "name": d.name, "where": where, "status": status, "last_write": last_write,
        "config": {"mode": cfg.get("mode"), "epochs": cfg.get("epochs"), "seed": cfg.get("seed"),
                   "init_from": cfg.get("init_from"), "select_by": cfg.get("select_by"),
                   "batch": f"{cfg.get('batch_size')} x {cfg.get('grad_accum')}",
                   "sources": len(mix.get("sources") or []) if mix else 0, "max_length": cfg.get("max_length")},
        "progress": prog,
        "elapsed_s": elapsed,
        "cost_est": round(elapsed / 3600 * MODAL_RATE_PER_H, 2) if where == "modal" else 0.0,
        "oom_skips": steps[-1].get("oom_skips", 0) if steps else 0,
        "steps": [{k: s.get(k) for k in ("step", "epoch", "loss", "ce", "brier", "ordinal", "elapsed_s")} for s in steps],
        "val": vals,
        "best_partial": _read_json(d / "best_partial.json"),
        "test": {k: v for k, v in (test or {}).items() if k != "meta"} or None,
        "pilot_test": {k: v for k, v in (pilot or {}).items() if k != "meta"} or None,
    }


def state() -> dict:
    found = {}
    for d in sorted([*RUNS.glob("*/log.jsonl"), *RUNS.glob("modal/*/log.jsonl")]):
        d = d.parent
        if d.name in SKIP or d.name.startswith(("smoke", "bench-")):
            continue
        # a run copied locally and also under runs/modal/: keep the copy with more results
        prev = found.get(d.name)
        if prev is None or len(list(d.glob("eval_*.json"))) > len(list(prev.glob("eval_*.json"))):
            found[d.name] = d
    runs = []
    for d in found.values():
        try:
            runs.append(load_run(d))
        except Exception as e:  # noqa: BLE001 - one unreadable run must not blank the page
            runs.append({"name": d.name, "status": "unreadable", "error": str(e)[:120], "steps": [], "val": []})
    notices = []
    nf = RUNS / "notices.jsonl"
    if nf.exists():
        notices = [json.loads(l) for l in nf.read_text().splitlines() if l.strip()]
    return {"now": time.time(), "billing": billing(), "runs": runs, "notices": notices}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/state"):
            body, ctype = json.dumps(state()).encode(), "application/json"
        elif self.path in ("/", "/index.html") or self.path.startswith("/#"):
            body, ctype = HTML.read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other devices on the network")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    print(f"dashboard: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
