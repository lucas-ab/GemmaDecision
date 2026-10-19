# GemmaDecision

A Clef-style decision model on top of [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2):
one bidirectional pass over `state + questions + options`, then a small head scores every option of every
question together and returns calibrated probabilities. No text generation.

Same request format as Unsloth's decision models / Cloudflare Clef (`choice`, `noul`, `score`), trained on
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).

## Architecture

```
"task: classification | query: <state>
 [question] action (choice): ...   [option] stop: ...   [option] observe: ...
 [question] risk (score): ...      [option] 0: ...      [option] 1: ..."
        │
  EmbeddingGemma-2 (bidirectional, all 24 layers) → 768-d token states (normed + projected, unpooled)
        │
  ClefHead (src/gemma_decision/model.py)
    span mean-pool each question / option  + kind / type / question-id / level embeddings
    2× cross-attention: slots → all tokens
    2× joint self-attention across all questions' slots
    scorer MLP([option, question, option⊙question]) → softmax within each question
    per-type temperature, fitted on held-out data
```

Loss (`losses.py`): soft cross-entropy to the annotator distribution + Brier + squared-CDF distance
for `score` questions. Question order and choice/noul option order are shuffled every epoch.

## Encoder modes

| mode | trains | params |
|---|---|---|
| `frozen` | head only | 13.8M |
| `lora` | LoRA r=16 on q/k/v/o/gate/up/down of all 24 layers + head | 5.1M + head |
| `full` | text layers, PLE, final norm, 512→768 projection + head | 136.8M + head |

Always frozen: token embedding table (134M), vision tower, audio tower. The training data is text-only, so
nothing would train the towers, and freezing the token table helps keep image/audio soft tokens aligned
with the text backbone.

## Usage

```bash
uv sync
uv run gd-train configs/lora.yaml            # → runs/lora/
uv run gd-eval runs/lora --split test
uv run gd-compare runs/frozen runs/lora runs/full
./scripts/run_all.sh                          # all three, sequentially
```

Smoke test: `uv run gd-train configs/full.yaml --max-steps 4 --limit 32 --name smoke`.

## Metrics

Per question (400 test rows ≈ 2,000 questions). `acc` = argmax matches the gold label; `annot_agree` = gold
probability of the predicted option (expected agreement with one annotator), whose ceiling is `oracle`
(≈0.66 on test, because annotators often disagree). Also `acc_agreed` (only questions where annotators
agreed), `row_acc`, `ece`, `brier`, `kl`, `score_mae`.

Unsloth's published numbers use 12 extra source datasets and a different 3,000-row test set
(incl. BANKING77 / CLINC150), so they are not directly comparable.

## Results (M1 Max, 3 epochs, typed-decisions `all`)

Text, typed-decisions test (400 rows / 2,000 questions):

| run | acc | acc_agreed | annot_agree (oracle 0.659) | kl | score_mae | trainable enc | min |
|---|---|---|---|---|---|---|---|
| frozen | 0.502 | 0.557 | 0.451 | 0.317 | 0.573 | 0 | 8 |
| lora | 0.608 | 0.689 | 0.513 | 0.206 | 0.406 | 5.0M | 29 |
| full | **0.641** | **0.723** | **0.529** | **0.190** | **0.375** | 136.8M | 30 |

Longer training and target blending (full fine-tune):

| run | acc | annot_agree | kl | brier | ece | score_mae |
|---|---|---|---|---|---|---|
| full (3 epochs, soft) | 0.641 | 0.529 | 0.190 | 0.112 | 0.094 | 0.375 |
| **full_e5 (5 epochs, soft)** | **0.715** | **0.566** | **0.163** | **0.096** | 0.141 | **0.337** |
| full_e5_mix (5 epochs, 0.5 soft + 0.5 one-hot, best epoch by acc) | 0.669 | 0.543 | 0.184 | 0.107 | 0.108 | 0.364 |

### vs Julia-1 ([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), 144M, mmBERT-small)

Same test parquet (SHA-256 `4f294f21…`, the one Julia pins), same metric code (`gd-score-external`).
Julia run with its published CPU protocol (`scripts/julia_bench.py`: torch backend, strict encoding,
1,024 tokens, marker-only head off) in its own env; it reproduces Julia's published CPU counts exactly
(choice 426/600, score 542/800, noul 483/600).

| | Julia-1 | full_e5 (ours) |
|---|---|---|
| acc | 0.726 | 0.715 |
| choice / noul / score acc | 0.710 / 0.805 / 0.677 | 0.713 / 0.787 / 0.662 |
| row_acc (all questions right) | 0.260 | 0.265 |
| kl to annotator distribution | 2.949 | **0.163** |
| brier | 0.336 | **0.096** |
| ece | 0.231 | **0.141** |
| score_mae | 0.436 | **0.337** |
| CPU latency / row (4 threads) | 302 ms (1 pass per question) | 253 ms (1 pass per row)* |
| M1 Max GPU latency / row | n/a (runtime is CPU/CUDA) | 97 ms* |

\* measured on the 3-epoch `full` run; same architecture. Accuracy difference is not significant:
McNemar p = 0.34, paired row-bootstrap 95% CI of (ours - Julia) = [-3.6, +1.5] pp (`scripts/paired_test.py`).
Julia's training mix is much broader (86 validation sets incl. `typed`, the same four workflows,
Banking77, AG News, MASSIVE); on the multimodal set's text variant (zero-shot) it wins call_center
intents (0.815 vs 0.285, Banking77-like) and loses crisis_response (0.356 vs 0.508).

Image/audio <-> text alignment after text-only fine-tuning (`gd-mm-probe`, media -> its own text retrieval
on `data/mm-decisions/test`, MRR (R@1)):

| run | call_center (audio->transcript, 600) | crisis_response (image->tweet, 600) | receipts (image->OCR, 200) |
|---|---|---|---|
| frozen (base) | 0.694 (0.592) | 0.276 (0.203) | 0.977 (0.960) |
| lora | 0.680 (0.572) | 0.262 (0.182) | 0.981 (0.965) |
| full | 0.706 (0.602) | 0.274 (0.205) | 0.968 (0.945) |
| full_e5 | 0.720 (0.623) | 0.268 (0.197) | 0.975 (0.955) |

Neither fine-tune measurably moved the media path. Zero-shot decision accuracy on the multimodal set
(`gd-mm-eval`) is near chance for every run (heads never saw those questions), so it needs in-domain
multimodal training data before it says anything about decisions.

## Pilot training mix (commercial-safe)

`gd-pilot-build` builds 14 sources (text, audio, image; CC BY / CC BY-SA / Apache / MIT / ODbL only) into
`data/pilot/{train,test}`; see `docs/dataset-candidates.md`. `configs/pilot_full.yaml` trains on
typed-decisions x3 plus the pilot, picking a text / media / both rendering per row each epoch.

## Broad training (bekko) and S1MB

`gd-broad-build` samples text decision cases from
[hotchpotch/bekko-system-one-dataset-v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)
subsets whose upstream license is commercial-safe (124 used, 29 excluded; see `SAFE_LICENSES`), drops any case
that shares an input hash or group with the S1MB English test set, and renders them like S1MB inputs.
`--exclude <previous build>` skips cases a previous build trained on and reuses its test files.

```bash
uv run gd-broad-build --per-source 500 --out data/broad                              # broad
uv run gd-broad-build --per-source 1000 --exclude data/broad --out data/broad2       # broad2: 106k new cases
uv run gd-train configs/broad2.yaml                                                  # ~19 h on the M1 Max
uv run python scripts/s1mb_bench.py runs/broad2 && uv run python scripts/s1mb_leaderboard.py runs/broad2
```

`broad` continues `continue_synth` on 500 cases per subset; `broad2` continues `broad` on 1,000 new ones per
subset; both replay the pilot mix, synthetic rows and typed-decisions. S1MB English v1 (137 benchmarks, cohort of
168 published models; G = the generalization benchmarks):

| run | rank | Borda | Task Avg | Noul | Choice | Score | G Noul | G Choice | G Score |
|---|---|---|---|---|---|---|---|---|---|
| continue_synth | 148 | 19.9 | 8.4 | 6.4 | 15.2 | 3.5 | 18.0 | 44.4 | 3.9 |
| broad | 118 | 36.2 | 25.2 | 24.2 | 31.3 | 20.2 | 15.0 | 43.8 | 6.5 |
| **broad2** | **106** | **42.4** | **30.0** | **32.3** | **34.7** | **23.1** | **30.0** | 39.3 | 4.1 |
| bekko-system-one-v0-68m | 79 | 55.9 | 40.5 | 42.9 | 51.6 | 26.9 | 31.0 | 57.5 | 8.9 |
| bekko-system-one-v0-400m | 41 | 67.4 | 50.6 | 51.2 | 61.3 | 39.3 | 52.0 | 80.3 | 31.1 |

broad -> broad2 on the shared tests: typed-decisions acc 0.763 -> 0.768 (KL 0.100 -> 0.095); per-source test
(18,450 questions, identical files) acc 0.671 -> 0.689, bekko subsets 0.573 -> 0.616 (macro), image/audio
sources within ±2.5 pp. Gains are mostly in-domain: generalization Choice and Score went down.

## Training on Modal (`modal_app.py`)

```bash
uv run modal run modal_app.py::pilot                                   # build data/pilot on the data volume (CPU, parallel)
uv run modal run modal_app.py::main --configs configs/pilot_full.yaml,configs/full_e5.yaml
uv run modal run modal_app.py::eval_runs --names pilot_full,full_e5    # mm-decisions eval + alignment probe
uv run modal run modal_app.py::download --name pilot_full              # checkpoint -> runs/modal/pilot_full/
uv run modal run modal_app.py::bench                                   # GPU cost/benefit
```

GPU choice (`modal_app.py::bench`, full fine-tune, batch 16, bf16, 2026-10-14 prices):

| gpu | $/h | rows/s | $ per 1k rows |
|---|---|---|---|
| **L40S (default)** | 1.95 | 26.7 | **0.020** |
| A100-40GB | 2.10 | 26.7 | 0.022 |
| L4 | 0.80 | 8.2 | 0.027 |
| A10 | 1.10 | 10.7 | 0.029 |
| H100 | 3.95 | 35.6 | 0.031 |

## Multimodal decision set

`gd-mm-build --split test|train` builds `data/mm-decisions/` from CrisisMMD (image + tweet), MINDS-14
(call audio + transcript) and CORD-v2 (receipt image + OCR), each row rendered as text / media / both.
See `data/mm-decisions/README.md` for licenses (the CrisisMMD part is CC BY-NC-SA 4.0).

## API server (`gd-serve`)

```bash
uv run gd-serve --run runs/broad            # http://127.0.0.1:8780: playground at /, OpenAPI docs at /docs
uv run gd-serve --run runs/broad --host 0.0.0.0   # reachable from the network; set GD_API_KEY to require a key
```

- `POST /v1/decide`: `{state, questions: {name: {type, instructions, criteria}}, images?, audio?}`. Choice
  criteria are `{key: description}`, Noul `{true, false}` (optional), Score a list of level descriptions,
  lowest first. Images and audio are base64; put `<|image|>` / `<|audio|>` in the state, or they are appended.
  Returns per question `label`, `confidence`, `probabilities`, plus `probability_true` (Noul) or `score`
  (expected level, Score).
- `POST /v1/systemone`: the System One / S1MB format (`decisions` with criteria `{id, description, value}`),
  rendered exactly as in training; returns probabilities per option ID, `selected`, and the expected numeric
  `score` for Score decisions.
- `--backend mlx` (Apple silicon) runs everything on MLX: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)'s (MIT; it and its added dependencies are MIT, Apache-2.0 or BSD)
  EmbeddingGemma-2 (text + Gemma 4 vision and audio towers) with our fine-tuned text weights swapped in, plus our
  head ported to MLX ([mlx_backend.py](src/gemma_decision/mlx_backend.py); weights converted once to
  `runs/<run>/mlx_text.safetensors`). No PyTorch model is loaded; the Hugging Face processor still prepares
  pixels and audio features on the CPU. `--mlx-dtype float16` is ~15-20% faster (same top-1 on every check).
  Parity with PyTorch: `scripts/mlx_parity.py` (text) and `scripts/mlx_parity_media.py` (images, audio):
  fp32 probabilities within 1e-5 and the same top-1 on all 156 text and 96 image/audio questions.
  Speed (M1 Max, `broad`; text from `scripts/bench_backends.py`: median of 50 distinct requests per length, each
  after a warm-up of that length; media from `scripts/mlx_parity_media.py`, matched warm-up, excluding CPU
  preprocessing; PyTorch CPU uses 8 threads):

  | backend | 192 tok | 512 | 1,024 | 2,048 | 4,096 | unseen length med/p90 | rows/s | image | audio clip |
  |---|---|---|---|---|---|---|---|---|---|
  | PyTorch MPS fp32 (bucketed padding) | 46 ms | 95 | 207 | 519 | 1,600 | 83 / 149 | 10.5 | 425-525 ms | 125-570 ms |
  | PyTorch CPU fp32 (no padding) | 154 ms | 239 | 437 | 914 | 2,372 | 215 / 291 | 5.4 | - | - |
  | MLX fp32 | 30 ms | 60 | 116 | 265 | 718 | 84 / 225 (noisy) | 13.9 | 340-370 ms | 60-415 ms |
  | MLX fp16 | 30 ms | 55 | 103 | 225 | 593 | 48 / 71 | 15.0 | ~275 ms | 58-380 ms |
  | MLX bf16 | 32 ms | 60 | 122 | 305 | 836 | 56 / 81 | 12.2 | - | - |
  | MLX CPU fp32 | 285 ms | 694 | 1,424 | 3,090 | 7,483 | 521 / 851 | 1.4 | - | - |

  p90 is within ~5% of the median at fixed lengths. Bucketed padding helps on MPS (4,096 tok: 1.6 s vs 1.9 s
  without; it avoids per-shape kernel compiles) but not on CPU, where it only adds tokens (unseen p90 350 vs
  291 ms). MLX fp32's unseen-length time varied 51-139 ms median between identical passes; fp16 was stable.
  MLX on the CPU is ~2-3x slower than PyTorch CPU.

  Memory (text throughput run): weights 3.0 GB for PyTorch (1.9 GB of it the vision/audio towers, loaded but
  unused for text) vs 1.1 GB for MLX's text-only port (0.57 GB in fp16); peak live memory 7.6 GB (PyTorch MPS)
  vs 6.0-7.4 GB (MLX); both GPU backends' allocators then reserve 22-25 GB of cache. PyTorch CPU peaks at
  15.5 GB process RSS. So MLX saves about 1.2 GB of live memory, not the 19 vs 6 GB an earlier,
  non-comparable measurement suggested.

  Through the API (text, varied lengths): 36 ms median with MLX vs 58 ms with PyTorch.

- One request runs at a time on the GPU; inputs over 8,192 tokens have the state cut at the end (`meta.truncated`).
  Requests are not stored.

## Notes

- Precision: fp32. `float16` breaks EmbeddingGemma-2 (per the model card); bf16 gave no speedup on M1.
- `config.json` says the sliding window is 512 (the model card says 1024).
- Requires `transformers>=5.19` (model type `embedding_gemma2`).
