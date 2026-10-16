# Training data candidates (researched 2026-10-14)

> **Decision (2026-10-14): commercial-safe sources only, ~15k-row pilot first.** What was built is in
> `src/gemma_decision/pilot_build.py` (see "Pilot as built" at the end). Under that rule, Speech-MASSIVE,
> ScienceQA, CrisisMMD, Memotion, AG News, NLVR2 (research-only images) and A-OKVQA / VQAv2 (COCO photos
> carry per-image Flickr licenses, some non-commercial) are out; SLURP, TextVQA, GQA, EuroSAT and
> Oxford Pets came in.

Goal: widen the training mix beyond typed-decisions (1,080 rows), especially with image and audio
decisions, so the model (a) matches Julia-1's breadth on text and (b) learns multimodal decisions,
which are near chance zero-shot today.

All entries were checked on the Hub API for license, gating, format and size. "Mirror" = the repo is
a re-upload; the license column then gives the **original** dataset's terms.

## How sources become decision rows

Every source is rewritten into the typed-decisions schema (`state`, `questions`, `gold`), the same
way `gd-mm-build` already does for CrisisMMD / MINDS-14 / CORD:

- **Media in the state.** `<|image|>` / `<|audio|>` placeholders, plus a one-line framing of the
  situation. Where a text equivalent exists (transcript, OCR, tweet), also emit text / media / both
  renderings so the model learns to ground the same decision in either.
- **≤ 20 options per question** (Julia's limit too). Big label spaces (Banking77's 77, CLINC's 150,
  MASSIVE's 60 intents, ESC-50's 50) become a shortlist: the gold label + hard negatives from the
  same group (same MASSIVE scenario, same FLEURS language group) + random negatives, reshuffled per epoch.
- **Several questions per row** wherever a source has several labels (Memotion, Speech-MASSIVE,
  CrisisMMD, CORD), so the joint head is used the way it is on typed-decisions.
- **Soft gold** when the source keeps annotator votes: VQAv2 (10 answers per question),
  NLVR2 (`extra_validations`), Speech-MASSIVE (`judgments.intent_score`). Otherwise one-hot.
- **Every option gets a description**, not just a label id; the head scores option text.

## Tier 1 - build first

### Audio

| dataset | license | size | decisions it gives |
|---|---|---|---|
| [google/fleurs](https://huggingface.co/datasets/google/fleurs) (in your FineEnvs collection) | CC BY 4.0 | 379k train clips, 102 languages | `choice`: which transcript matches the recording (4-6 same-language candidates); `choice`: spoken language (8 candidates, same-group hard negatives); `noul`: "the speaker is saying: <English sentence>" via FLORES-aligned ids (cross-lingual grounding) |
| [FBK-MT/Speech-MASSIVE](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE) | CC BY-NC-SA 4.0 | 46k train + 49k dev, 12 languages | `choice`: scenario (18); `choice`: intent (shortlist); soft labels from `judgments`; paired with MASSIVE text, which Julia was trained on |
| [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) | CC BY 4.0 | 8.2k clips, 14 languages | `choice`: banking intent (14); use the non-English languages + the 60% English not in our test split |
| [WueNLP/belebele-fleurs](https://huggingface.co/datasets/WueNLP/belebele-fleurs) | CC BY-SA 4.0 | 29k questions, 92 languages (test-only source) | `choice`: 4-option reading comprehension over a **spoken passage** (multi-speaker, ~30-60 s, ~1k audio tokens). Split by passage; hold out English as eval |
| [confit/cremad-parquet](https://huggingface.co/datasets/confit/cremad-parquet) (mirror of CREMA-D) | ODbL | 7.4k clips | `choice`: emotion (6); `score`: intensity (low / medium / high, in the original filenames) |

### Images

| dataset | license | size | decisions it gives |
|---|---|---|---|
| [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) | Apache 2.0 (AI2) | 17k train | `choice`: native 4-option visual QA needing world knowledge |
| [lmms-lab-encoder/VQAv2](https://huggingface.co/datasets/lmms-lab-encoder/VQAv2) (mirror) | CC BY 4.0 annotations, COCO images | 214k val with answers | `noul` from yes/no questions with **soft gold from 10 annotators**; `score` from counting questions (0 / 1 / 2 / 3 / 4 / 5+) |
| [lmms-lab/NLVR2](https://huggingface.co/datasets/lmms-lab/NLVR2) (mirror) | CC BY 4.0 annotations | 17k (dev + test splits) | `noul`: statement about **two images**, soft gold from `extra_validations`; train on dev/test-public, keep test-unseen as eval |
| [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | CC BY-NC-SA 4.0 (original) | 12.7k train, ~half with images | `choice`: 2-5 option science questions, `hint` text as context |
| [mteb/MMSoc_Memotion](https://huggingface.co/datasets/mteb/MMSoc_Memotion) (mirror of Memotion 7k) | research-only (competition terms) - **check before publishing** | 5.6k train | meme image + OCR text; 5 questions per row: `score` sentiment (5), humor / sarcasm / offensive (4 levels each), `noul` motivational. Closest image analogue to typed-decisions' multi-question rows |
| QCRI/CrisisMMD train (builder exists) | CC BY-NC-SA 4.0 | 13.6k + 2.5k damage | `gd-mm-build --split train` |
| naver-clova-ix/cord-v2 train (builder exists) | CC BY 4.0 | 800 receipts | `gd-mm-build --split train` |

### Text (Julia-parity breadth)

| dataset | license | size | decisions |
|---|---|---|---|
| [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) | CC BY 3.0 | 15k train | intent shortlist incl. out-of-scope |
| [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | CC BY 4.0 (original) | 10k train | intent shortlist; this is the gap where Julia beat us on call intents |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) / `_scenario` | CC BY 4.0 (original) | 11.5k train x 51 locales | scenario + intent; text twin of Speech-MASSIVE |
| [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | non-commercial research use | 120k | 4-way topic |
| [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) | see original SST terms | 8.5k | `score`: 5-level sentiment |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | mostly CC BY-SA 3.0 / OANC | 393k | `choice`: entailment / neutral / contradiction |

## Tier 2 - useful, more work or a caveat

| dataset | why not tier 1 |
|---|---|
| [CLAPv2/ClothoAQA](https://huggingface.co/datasets/CLAPv2/ClothoAQA) (22k audio QA) | yes/no subset maps to `noul`, but license of the QA annotations is not stated on the mirror |
| [confit/esc50-parquet](https://huggingface.co/datasets/confit/esc50-parquet), [danavery/urbansound8K](https://huggingface.co/datasets/danavery/urbansound8K) | environmental sound classes; CC BY-NC; single-question rows |
| [Cognitive-Lab/NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025) (your collection) | 1M document pages, 22 languages, CC BY-NC 4.0; document-language `choice` is easy, its `vqa.json` needs answer-distractor generation |
| [pranavmr/MM-IMDb](https://huggingface.co/datasets/pranavmr/MM-IMDb) | poster + plot -> genre `noul`s; IMDb-derived content, terms unclear |
| [lmms-lab-encoder/VizWiz-VQA](https://huggingface.co/datasets/lmms-lab-encoder/VizWiz-VQA) | "is this answerable from the photo" `noul`; only val has answers (4.3k) |
| [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full), [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | good `score` / `choice` text sources; restrictive or unclear terms |

## Eval only - do not train on these (they are standard benchmarks)

[gamma-lab-umd/MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) (audio MCQ, 1k, CC BY-NC 4.0) ·
[lmms-lab/MMBench_EN](https://huggingface.co/datasets/lmms-lab/MMBench_EN) (dev 4.4k) ·
[lmms-lab-encoder/SEED-Bench](https://huggingface.co/datasets/lmms-lab-encoder/SEED-Bench) (18k) ·
[MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU) (val 900, Apache 2.0) ·
[BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) (val 1.9k, Apache 2.0, multi-image) ·
[BaiqiL/NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) (1.9k, Apache 2.0) ·
[lmms-lab-encoder/ai2d](https://huggingface.co/datasets/lmms-lab-encoder/ai2d) (test 3k).
All are already multiple-choice, so they convert to `choice` rows with no label engineering, and give
a public multimodal decision benchmark next to our own `data/mm-decisions` set.

## Rejected

| dataset | reason |
|---|---|
| Hateful Memes (any mirror) | Facebook dataset license restricts use and redistribution |
| HuggingFaceM4/SNLI-VE, aharley/rvl_cdip, google/speech_commands, Jeneral/fer-2013 | loading scripts only (unsupported by `datasets` 5); RVL-CDIP / FER also have restrictive or face-data terms |
| declare-lab/MELD | GPL-3.0 and raw tar.gz only |
| CMU-MOSEI mirrors | pickles / zips, no clean parquet |
| xBD mirrors | raw PNG/JSON dumps, CC BY-NC-SA, would need a satellite-tile pipeline |
| SROIE mirrors | license unknown |
| HuggingFaceM4/ChartQA | GPL-3.0, open-ended numeric answers |

## Budget on the M1 Max

Current full fine-tune speed is about 2 rows/s at ~500 tokens. An image row is ~350-450 tokens,
a 10 s audio row ~300 tokens, a Belebele passage ~1.2k. Rough mixtures:

| mix | rows / epoch | time / epoch (M1 Max) |
|---|---|---|
| pilot: typed x3 + 1-2k from each tier-1 multimodal source | ~15k | ~2 h |
| full tier 1 incl. text breadth | ~35k | ~5 h |

Training with media also needs the train loop to use `collate_mm` (the eval path already does) and a
memory check with the frozen vision/audio towers running forward in the batch.

Licenses split cleanly: CC BY / Apache / ODbL sources can go in a commercially usable mix and a
public dataset; the NC / NC-SA / research-only sources (Speech-MASSIVE, ScienceQA, CrisisMMD, Memotion,
AG News) should be a separate config if anything is published.

## Pilot as built (commercial-safe)

| source | modality | license | train / test rows | questions per row |
|---|---|---|---|---|
| banking77 | text | CC BY 4.0 | 800 / 200 | intent (10-option shortlist, hard negatives) |
| clinc | text | CC BY 3.0 | 800 / 200 | intent (9 + out-of-scope) |
| massive | text, 12 languages | CC BY 4.0 | 1,200 / 240 | scenario (18) + intent shortlist |
| mnli | text | CC BY-SA 3.0 / OANC | 600 / 200 | entailment / neutral / contradiction |
| fleurs | audio, 12 languages | CC BY 4.0 | 1,500 / 240 | transcript match (4), language (6), "speaker says X" (noul) |
| slurp | audio (+ transcript) | CC BY 4.0 | 1,000 / 200 | scenario + intent shortlist |
| minds14 | audio (+ transcript), 14 languages | CC BY 4.0 | 1,000 / 0 (test = `data/mm-decisions`) | banking intent (14) |
| belebele_fleurs | spoken passage (+ text) | CC BY-SA 4.0 | ~350 / ~40 (split by passage) | 4-option reading comprehension |
| cremad | audio | ODbL | 1,000 / 200 | emotion (6), intensity score when labeled |
| textvqa | image (+ OCR) | CC BY 4.0, images CC BY 2.0 | 1,500 / 200 | answer choice, soft gold from 10 annotators |
| gqa | image | CC BY 4.0 | 1,200 / 200 | up to 4 per photo: yes/no (noul) or same-type choice |
| eurosat | image | MIT | 500 / 150 | land use (10) |
| oxford_pets | image | CC BY-SA 4.0 | 500 / 150 | dog? (noul) + breed shortlist |
| cord | image (+ OCR) | CC BY 4.0 | 800 / 0 (test = `data/mm-decisions`) | payment, total, item count, discount |

## GitHub search (2026-10-15): decision-model data and synthetic recipes

The decision-model community publishes data and generators on GitHub; the curated index is
[awesome-decision-models](https://github.com/AnotiaWang/awesome-decision-models) (CC0).

| source | what | licence | used? |
|---|---|---|---|
| [Valen-Training-General-100k](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) ([GitHub](https://github.com/Liuziyu77/Valen)) | 100k image decisions (VQA, UI, games, documents), one question each, per-row upstream licence | Apache-2.0 card | **Yes**: 6,000 train / 600 test from the commercial-safe sources only (RICO ScreenQA CC BY 4.0, GameQA MIT, GQA, TextVQA, CLEVR CC BY 4.0, Valen's programmatic rows); dropped VQAv2 (COCO Flickr licences), ChartQA (GPL-3.0), DocVQA and ShowUI (upstream terms). `gd-pilot-build --sources valen` |
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | `decision-v7`: 10k public-dataset rows + 896 generated policy + 1,680 rule-structure rows; no teacher outputs | Apache-2.0 code | next: rule/policy generators |
| [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) | 246k-row mix, licence audit per source ([data/sources.md](https://github.com/strands-labs/strands-decider/blob/main/data/sources.md)); committed Qwen-written, Qwen-verified synthetic files; catch-all and instruction-flip recipes | Apache-2.0 | recipes adapted (below) |
| [Valen-Eval-Game](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) | 30k Sokoban decision levels | Apache-2.0 | candidate (planning from images) |
| Microsoft GUIDE (via [security-alert-triage](https://github.com/vishnukanchi9/security-alert-triage)) | 1.03M graded security incidents | to check | candidate for security workflow |
| [Inv3D generator](https://github.com/FelixHertlein/inv3d-generator) | invoice document image generator | MIT | candidate for invoice images |

### Synthetic rows built by code (`gd-synth-build`, no teacher model)

Sources: typed-decisions train rows outside gd-train's validation holdout, and gd-pilot-build train rows
outside the mix holdout; test rows are never read.

| recipe | what changes | gold | rows (typed / pilot) |
|---|---|---|---|
| catch_all | a choice question gets "None of the other options applies"; half the time the top option is removed | removed option's probability moves to "none" | 324 / 4,844 |
| flip | a yes/no question is asked in reverse ("Reverse the answer: ...") | true and false swap (targets and descriptions) | 345 / 1,405 |
| missing | evidence removed (no state, no media) | the question's prior from the source data | 54 / 810 |
