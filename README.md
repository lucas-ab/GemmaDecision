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

## Notes

- Precision: fp32. `float16` breaks EmbeddingGemma-2 (per the model card); bf16 gave no speedup on M1.
- `config.json` says the sliding window is 512 (the model card says 1024).
- Requires `transformers>=5.19` (model type `embedding_gemma2`).
