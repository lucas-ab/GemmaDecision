"""`gd-serve` - HTTP API (and a browser playground) for a trained decision model.

  uv run gd-serve --run runs/broad                     # http://127.0.0.1:8780  (playground at /, docs at /docs)
  uv run gd-serve --run runs/broad --host 0.0.0.0      # reachable from the local network
  GD_API_KEY=secret uv run gd-serve ...                # then send "Authorization: Bearer secret"

Endpoints
  GET  /health        model, device, calibration temperatures, limits
  POST /v1/decide     our format: {state, questions: {name: {type, instructions, criteria}}, images?, audio?}
  POST /v1/systemone  System One / S1MB format: {state, decisions: [{id, type, instructions, criteria: [{id,
                      description, value}]}]}; rendered exactly as in training (gemma_decision.s1mb_format)

Questions: type "choice" (criteria {key: description}), "noul" (yes/no; criteria {true, false} optional) or
"score" (criteria = level descriptions, lowest first). Images and audio are base64 strings (data: URLs work);
put <|image|> / <|audio|> in the state where each goes, or they are appended in order. Requests are not stored.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .data import MAX_AUDIO_S, Example, _audio_seconds, _render, chunk_long_audio, media_reserve, parse_questions
from .evaluate import run
from .s1mb_format import split_case, to_example

PLAYGROUND = Path(__file__).with_name("playground.html")


# --------------------------------------------------------------------------- request / response schemas

class QuestionIn(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = Field(
        default=None, description="choice: {key: description}; noul: {true, false} (optional); "
                                  "score: level descriptions, lowest first")


class DecideRequest(BaseModel):
    state: str | dict | list = Field(description="the case: text, or JSON (serialized with sorted keys)")
    questions: dict[str, QuestionIn] = Field(min_length=1, max_length=32)
    images: list[str] = Field(default=[], description="base64 images, in placeholder order")
    audio: list[str] = Field(default=[], description="base64 audio (wav/flac/ogg/mp3), in placeholder order")
    calibrated: bool = Field(default=True, description="apply the per-type temperatures fitted on validation")


class CriterionIn(BaseModel):
    id: str
    description: str | dict | list
    value: float | None = None


class DecisionIn(BaseModel):
    id: str
    type: Literal["choice", "noul", "score"]
    instructions: str | dict | list
    system_prompt: str = ""
    criteria: list[CriterionIn] = Field(min_length=2)


class SystemOneRequest(BaseModel):
    state: str | dict | list
    decisions: list[DecisionIn] = Field(min_length=1)
    calibrated: bool = True


# --------------------------------------------------------------------------- conversion (no model needed)

def _state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)


def _b64(s: str) -> bytes:
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(422, f"invalid base64 media: {err}") from err


def decide_example(req: DecideRequest) -> Example:
    state = _state_text(req.state)
    for tag, items in (("<|image|>", req.images), ("<|audio|>", req.audio)):
        have = state.count(tag)
        if have > len(items):
            raise HTTPException(422, f"state has {have} {tag} placeholders but {len(items)} items were sent")
        label = "Image" if "image" in tag else "Audio"
        state += "".join(f"\n{label}: {tag}" for _ in range(len(items) - have))
    questions = {}
    for name, q in req.questions.items():
        crit = q.criteria
        if q.type == "choice" and not isinstance(crit, dict):
            raise HTTPException(422, f"{name}: choice criteria must be {{key: description}}")
        if q.type == "score" and not isinstance(crit, list):
            raise HTTPException(422, f"{name}: score criteria must be a list of level descriptions")
        if q.type == "noul" and crit is not None and set(crit) != {"true", "false"}:
            raise HTTPException(422, f"{name}: noul criteria must have exactly the keys true and false")
        if q.type in ("choice", "score") and len(crit) < 2:
            raise HTTPException(422, f"{name}: at least two options are required")
        if q.type == "score" and len(crit) > 16:
            raise HTTPException(422, f"{name}: at most 16 score levels")
        questions[name] = {"type": q.type, "instructions": q.instructions, "criteria": crit}
    parsed = parse_questions(questions)
    for q in parsed:
        q.target = [1 / len(q.keys)] * len(q.keys)  # placeholder; run() stacks one per question
    images = [_b64(s) for s in req.images]
    audio = [_b64(s) for s in req.audio]
    if audio and any(_audio_seconds(a) > MAX_AUDIO_S for a in audio):
        state, audio = chunk_long_audio(state, audio)
    return Example(id="request", workflow="api", state=state, questions=parsed, images=images, audio=audio)


def decide_output(records: list[dict]) -> dict:
    out = {}
    for r in records:
        probs = dict(zip(r["keys"], r["probs"]))
        label = max(probs, key=probs.get)
        item = {"type": r["type"], "label": label, "confidence": probs[label], "probabilities": probs}
        if r["type"] == "noul":
            item["probability_true"] = probs["true"]
        if r["type"] == "score":
            item["score"] = sum(int(k) * p for k, p in probs.items())  # expected level, 0 = first criterion
        out[r["question"]] = item
    return out


def systemone_case(req: SystemOneRequest) -> dict:
    def text(v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    qs = []
    for d in req.decisions:
        ids = [c.id for c in d.criteria]
        if len(set(ids)) != len(ids):
            raise HTTPException(422, f"{d.id}: duplicate criterion ids")
        if d.type == "noul" and set(ids) != {"true", "false"}:
            raise HTTPException(422, f"{d.id}: noul criteria ids must be true and false")
        if d.type == "score" and (any(c.value is None for c in d.criteria) or len({c.value for c in d.criteria}) < 2):
            raise HTTPException(422, f"{d.id}: score criteria need distinct numeric values")
        instructions = text(d.instructions)
        if d.system_prompt:
            instructions = f"{d.system_prompt}\n{instructions}"
        qs.append({"id": d.id, "task": d.type, "instructions": instructions,
                   "options": [{"id": c.id, "description": text(c.description), "value": c.value} for c in d.criteria]})
    return {"id": "request", "state": _state_text(req.state), "questions": qs}


def systemone_output(parts: list[tuple[list[dict], list[list[str]]]], records_by_part: list[list[dict]]) -> dict:
    out = {}
    for (qs, id_order), recs in zip(parts, records_by_part):
        for q, ids, r in zip(qs, id_order, recs):
            probs = dict(zip(ids, r["probs"]))
            item = {"type": q["task"], "selected": max((o["id"] for o in q["options"]), key=probs.get),
                    "probabilities": probs}
            if q["task"] == "noul":
                item["probability_true"] = probs["true"]
            if q["task"] == "score":
                vals = {o["id"]: float(o["value"]) for o in q["options"]}
                lo, hi = min(vals.values()), max(vals.values())
                item["score"] = sum(vals[k] * p for k, p in probs.items())
                item["normalized_score"] = (item["score"] - lo) / (hi - lo)
            out[q["id"]] = item
    return out


# --------------------------------------------------------------------------- model + app

# Fixed padded sizes: the MPS / CUDA backends compile kernels per input shape, so an unseen length costs
# 60-350 ms on the M1 Max while a seen one runs in ~35 ms. Padding (masked everywhere) keeps shapes few.
T_BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


def _ceil(n: int, buckets=None, step: int = 1) -> int:
    if buckets:
        return next((b for b in buckets if b >= n), n)
    return -(-n // step) * step


def pad_batch(batch: dict, pad_id: int) -> dict:
    """Pad a single-row collate() batch to bucketed token (T), slot (S) and option (K) sizes."""
    import torch.nn.functional as F

    (_, T), (_, S, _), (_, K) = batch["input_ids"].shape, batch["slot_weights"].shape, batch["q_opt_mask"].shape
    t, s, k = _ceil(T, T_BUCKETS) - T, _ceil(S, step=8) - S, _ceil(K, step=4) - K
    out = dict(batch)
    out["input_ids"] = F.pad(batch["input_ids"], (0, t), value=pad_id)
    out["attention_mask"] = F.pad(batch["attention_mask"], (0, t))
    out["slot_weights"] = F.pad(batch["slot_weights"], (0, t, 0, s))
    for key in ("slot_kind", "slot_type", "slot_qid", "slot_level"):  # 0 = KIND_PAD, masked in the head
        out[key] = F.pad(batch[key], (0, s))
    out["q_opt_slots"] = F.pad(batch["q_opt_slots"], (0, k))
    out["q_opt_mask"] = F.pad(batch["q_opt_mask"], (0, k), value=False)
    out["q_target"] = F.pad(batch["q_target"], (0, k))
    return out


class Engine:
    """Inference for one run. backend "torch": PyTorch (MPS/CUDA/CPU). backend "mlx" (Apple silicon): mlx-vlm's
    EmbeddingGemma-2 with our fine-tuned text weights + our head in MLX, for text, images and audio; no PyTorch
    model is loaded (the Hugging Face processor still prepares pixels and audio features on the CPU)."""

    def __init__(self, run_dir: Path, device: str | None, max_length: int, backend: str = "torch",
                 mlx_dtype: str = "float32"):
        import yaml
        from transformers import AutoProcessor, AutoTokenizer

        self.run_dir, self.max_length, self.backend, self.device_arg = run_dir, max_length, backend, device
        self.cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.tok = AutoTokenizer.from_pretrained(self.cfg["model_id"])
        self.processor = AutoProcessor.from_pretrained(self.cfg["model_id"])
        self.lock = threading.Lock()  # one forward pass at a time on the GPU
        self.model = self.mlx = None
        if backend == "mlx":
            from concurrent.futures import ThreadPoolExecutor

            import mlx.core as mx

            from .mlx_backend import MLXDecisionModel

            # MLX streams belong to the thread that created them (mlx-vlm makes one at import), so loading and
            # every forward pass run on this one thread; FastAPI's request threads hand work to it.
            self.mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
            self.mlx = self.mlx_thread.submit(
                MLXDecisionModel.load, run_dir, getattr(mx, mlx_dtype), encoder="vlm").result()
            self.device = f"mlx-gpu ({mlx_dtype})"
        elif backend == "torch":
            self._torch()
        else:
            raise ValueError(f"unknown backend {backend!r}")
        meta = {}
        if (run_dir / "eval_test.json").exists():
            meta = json.loads((run_dir / "eval_test.json").read_text()).get("meta", {})
        self.info = {"run": str(run_dir), "model_id": self.cfg["model_id"], "mode": self.cfg["mode"],
                     "backend": backend, "device": str(self.device), "max_length": max_length,
                     "temperatures": dict(zip(("choice", "noul", "score"), meta.get("temperatures") or [None] * 3))}

    def _torch(self):
        """The PyTorch model (backend torch)."""
        if self.model is None:
            from safetensors.torch import load_file

            from .train import build_model, pick_device

            model = build_model(self.cfg)
            _, unexpected = model.load_state_dict(load_file(self.run_dir / "trainable.safetensors"), strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
            self.torch_device = torch.device(self.device_arg) if self.device_arg else pick_device()
            self.model = model.to(self.torch_device).eval()
            if self.backend == "torch":
                self.device = self.torch_device
        return self.model

    def warmup(self, up_to: int = 1024):
        """Compile the common padded shapes once at startup (a few seconds) instead of on user requests."""
        t0 = time.perf_counter()
        for T in [b for b in T_BUCKETS if b <= up_to]:
            for n_q in (1, 2, 3):
                questions = {f"q{i}": {"type": t, "instructions": "Decide.", "criteria": c} for i, (t, c) in enumerate(
                    [("choice", {"a": "A", "b": "B", "c": "C"}), ("noul", None), ("score", ["low", "mid", "high"])][:n_q])}
                qs = parse_questions(questions)
                for q in qs:
                    q.target = [1 / len(q.keys)] * len(q.keys)
                state = "word " * max(1, T - 40 - 25 * n_q)
                self._text_forward(Example(id="warmup", workflow="api", state=state[: 4 * T], questions=qs), True)
        return round(time.perf_counter() - t0, 1)

    @torch.inference_mode()
    def _text_forward(self, ex: Example, calibrated: bool) -> list[dict]:
        from .data import collate, encode
        from .model import probabilities, to_device

        enc = encode(self.tok, ex.state, ex.questions, self.max_length)
        batch = collate([enc], self.tok.pad_token_id)
        meta = batch.pop("q_meta")
        if self.mlx is not None:  # MLX compiles kernels per op, not per shape: no padding needed
            probs = torch.from_numpy(self.mlx_thread.submit(self.mlx.probabilities, batch, calibrated).result())
        else:
            logits = self.model(to_device(pad_batch(batch, self.tok.pad_token_id), self.device), calibrated=calibrated)
            probs = probabilities(logits.float()).cpu()
        return [{"question": q.name, "type": q.type, "keys": q.keys, "probs": probs[j, : len(q.keys)].tolist()}
                for j, (_, q) in enumerate(meta)]

    def _media_forward_mlx(self, ex: Example, calibrated: bool) -> list[dict]:
        from .data import collate_mm, encode

        enc = encode(self.tok, ex.state, ex.questions, self.max_length, reserve=media_reserve(ex))
        batch = collate_mm([enc], [ex], self.processor)
        meta = batch.pop("q_meta")
        probs = self.mlx_thread.submit(self.mlx.probabilities, batch, calibrated).result()
        return [{"question": q.name, "type": q.type, "keys": q.keys, "probs": probs[j, : len(q.keys)].tolist()}
                for j, (_, q) in enumerate(meta)]

    def infer(self, examples: list[Example], calibrated: bool) -> tuple[list[list[dict]], dict]:
        t0 = time.perf_counter()
        tokens, truncated = [], 0
        for ex in examples:
            n = len(self.tok(_render(ex.state, ex.questions)[0])["input_ids"])
            if ex.images or ex.audio:  # each placeholder expands to ~280 image / 25-per-second audio tokens
                n += media_reserve(ex)
            tokens.append(n)
            truncated += n > self.max_length
        with self.lock:
            out = []
            for ex in examples:
                try:
                    if (ex.images or ex.audio) and self.mlx is not None:
                        recs = self._media_forward_mlx(ex, calibrated)
                    elif ex.images or ex.audio:
                        recs, _ = run(self.model, self.tok, [ex], self.torch_device, 1, self.max_length,
                                      calibrated, self.processor)
                    else:
                        recs = self._text_forward(ex, calibrated)
                except ValueError as err:  # encode(): questions and options alone exceed max_length
                    raise HTTPException(422, str(err)) from err
                except RuntimeError as err:
                    if "out of memory" in str(err):
                        raise HTTPException(507, "out of memory for this input; shorten the state") from err
                    raise
                if not recs:
                    raise HTTPException(422, "the questions and options alone exceed max_length")
                out.append(recs)
        meta = {"input_tokens": tokens, "truncated": bool(truncated),
                "latency_ms": round(1000 * (time.perf_counter() - t0), 1), "run": self.run_dir.name}
        if truncated:
            meta["note"] = f"state text was cut at the end to fit {self.max_length} tokens"
        return out, meta


def create_app(engine: Engine | None, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="GemmaDecision API", version="1.0",
                  description="EmbeddingGemma-2 + Clef-style decision head. Choice / Noul / Score decisions.")

    def auth(authorization: str | None = Header(default=None)):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "missing or wrong API key")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def playground():
        return PLAYGROUND.read_text()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **(engine.info if engine else {})}

    @app.post("/v1/decide", dependencies=[Depends(auth)])
    def decide(req: DecideRequest) -> dict[str, Any]:
        ex = decide_example(req)
        (recs,), meta = engine.infer([ex], req.calibrated)
        return {"questions": decide_output(recs), "meta": meta}

    @app.post("/v1/systemone", dependencies=[Depends(auth)])
    def systemone(req: SystemOneRequest) -> dict[str, Any]:
        parts = [to_example(p) for p in split_case(systemone_case(req))]
        recs, meta = engine.infer([ex for ex, _, _ in parts], req.calibrated)
        return {"decisions": systemone_output([(qs, ids) for _, qs, ids in parts], recs), "meta": meta}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("runs/broad"), help="run dir with config.yaml + trainable.safetensors")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--device", help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--max-length", type=int, default=8192, help="model input limit (the encoder's context)")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch",
                    help="mlx: everything on MLX via mlx-vlm (Apple silicon); torch: PyTorch")
    ap.add_argument("--mlx-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--warmup-tokens", type=int, default=1024, help="pre-compile padded shapes up to this length")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    import uvicorn

    engine = Engine(args.run, args.device, args.max_length, args.backend, args.mlx_dtype)
    if not args.no_warmup and args.backend == "torch":  # MLX does not compile per input shape
        print(f"warming up input shapes up to {args.warmup_tokens} tokens...", flush=True)
        print(f"warm-up took {engine.warmup(args.warmup_tokens)} s", flush=True)
    print(f"loaded {args.run} on {engine.device}; http://{args.host}:{args.port}  (docs: /docs)", flush=True)
    uvicorn.run(create_app(engine, os.environ.get("GD_API_KEY")), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
