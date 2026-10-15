"""`gd-pilot-build` - commercial-safe pilot training mix (text, audio, image) in typed-decisions format.

Every source is licensed for commercial use (CC BY / CC BY-SA / Apache / MIT / ODbL / CC0); see
docs/dataset-candidates.md. Output: data/pilot/{train,test}/<source>.parquet with the gd-mm-build columns
plus `variants` (which of state_text / state_media / state_both are valid for training; e.g. a
transcript-matching row has no text-only variant, it would give the answer away).

Sources are read straight from parquet shards (hf_hub_download + pyarrow), so only the shards needed for
the pilot are downloaded.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .mm_build import MINDS_ALL, PREFIX_NOTE, _wav16k, build_calls, build_receipts, jpeg_bytes, stable_frac

LIC = {
    "banking77": "cc-by-4.0", "clinc": "cc-by-3.0", "massive": "cc-by-4.0", "mnli": "cc-by-sa-3.0 / oanc",
    "fleurs": "cc-by-4.0", "slurp": "cc-by-4.0", "minds14": "cc-by-4.0", "belebele_fleurs": "cc-by-sa-4.0",
    "cremad": "odbl-1.0", "textvqa": "cc-by-4.0 (images cc-by-2.0)", "gqa": "cc-by-4.0", "eurosat": "mit",
    "oxford_pets": "cc-by-sa-4.0", "cord": "cc-by-4.0", "snake": "apache-2.0 (generated)",
}

# --------------------------------------------------------------------------- helpers


def shards(repo: str, pattern: str) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern))


def read_rows(repo: str, path: str, max_rows: int | None = None, columns=None) -> list[dict]:
    """Whole shard via the HF cache, or with `max_rows` only the leading row groups via HTTP range reads."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    if max_rows:
        pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}/{path}", "rb", block_size=8 << 20))
    else:
        pf = pq.ParquetFile(hf_hub_download(repo, path, repo_type="dataset"))
    rows = []
    for batch in pf.iter_batches(batch_size=256, columns=columns):
        rows += batch.to_pylist()
        if max_rows and len(rows) >= max_rows:
            return rows[:max_rows]
    return rows


def human(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip().capitalize()


def shortlist(gold: str, pool: list[str], k: int, rng: random.Random, related: list[str] = ()) -> list[str]:
    """gold + up to k-1 distractors, preferring `related` (hard negatives), then random from `pool`."""
    picks = [gold]
    for cand in rng.sample(list(related), len(related)) + rng.sample(pool, len(pool)):
        if len(picks) >= k:
            break
        if cand not in picks:
            picks.append(cand)
    rng.shuffle(picks)
    return picks


def choice_q(instructions: str, keys: list[str], describe=human) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": {k: describe(k) for k in keys}}


def make_row(rid, source, questions: dict, labels: dict, texts: dict, variants: list[str], images=(), audio=()):
    """labels[name]: gold key (one-hot) or {key: probability} (soft)."""
    gold, agreement = {}, {}
    for name, q in questions.items():
        if q["type"] == "score":
            keys = [str(i) for i in range(len(q["criteria"]))]
        elif q["type"] == "noul":
            keys = ["true", "false"]
        else:
            keys = list(q["criteria"])
        lab = labels[name]
        probs = {k: float(lab.get(k, 0.0)) for k in keys} if isinstance(lab, dict) else {k: float(k == lab) for k in keys}
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        top = max(probs, key=probs.get)
        g = {"type": q["type"], "label": top, "confidence": probs[top], "probabilities": probs}
        if q["type"] == "noul":
            g["noul"] = probs["true"]
        if q["type"] == "score":
            g["score"] = sum(int(k) * v for k, v in probs.items())
        gold[name] = g
        agreement[name] = {"argmax_agree": probs[top] >= 0.5, "argmax_majority": top,
                           "total_variation": 1 - probs[top]}
    return {
        "id": rid, "workflow": source, "source": source, "license": LIC[source],
        "state_text": texts.get("text", ""), "state_media": texts.get("media", ""), "state_both": texts.get("both", ""),
        "variants": variants, "images": list(images), "audio": list(audio),
        "questions": json.dumps(questions), "gold": json.dumps(gold),
        "label_agreement": json.dumps(agreement), "n_questions": len(questions),
    }


def text_row(rid, source, head, body, questions, labels):
    t = f"{head} {PREFIX_NOTE}\n{body}"
    return make_row(rid, source, questions, labels, {"text": t, "media": t, "both": t}, ["text"])


def split_of(key: str, test_frac: float) -> str:
    return "test" if stable_frac(key) < test_frac else "train"


# --------------------------------------------------------------------------- text sources


def build_banking77(n, rng):
    out = {"train": [], "test": []}
    for split in ("train", "test"):
        rows = read_rows("mteb/banking77", shards("mteb/banking77", f"*{split}*.parquet")[0])
        labels = sorted({r["label_text"] for r in rows})
        rng.shuffle(rows)
        for i, r in enumerate(rows[: n[split]]):
            gold = r["label_text"]
            related = [l for l in labels if l != gold and set(l.split("_")) & set(gold.split("_"))]
            q = {"intent": choice_q("What does the bank customer want?", shortlist(gold, labels, 10, rng, related))}
            out[split].append(text_row(f"banking77_{split}_{i}", "banking77", "A message sent to a bank's support chat.",
                                       f"Message: {r['text']}", q, {"intent": gold}))
    return out


def build_clinc(n, rng):
    from huggingface_hub import hf_hub_download

    out = {"train": [], "test": []}
    describe = lambda k: "Out of scope: none of the other requests" if k == "oos" else human(k)  # noqa: E731
    for split in ("train", "test"):
        path = shards("clinc/clinc_oos", f"plus/{split}-*.parquet")[0]
        pf = pq.ParquetFile(hf_hub_download("clinc/clinc_oos", path, repo_type="dataset"))
        names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        in_scope = [x for x in names if x != "oos"]
        rows = pf.read().to_pylist()
        rng.shuffle(rows)
        for i, r in enumerate(rows[: n[split]]):
            gold = names[r["intent"]]
            # every question offers "out of scope" alongside 9 intents (the gold one when it is in scope)
            keys = shortlist(gold, in_scope, 10, rng) if gold == "oos" else shortlist(gold, in_scope, 9, rng) + ["oos"]
            rng.shuffle(keys)
            q = {"intent": choice_q("Which request is the user making to the assistant?", keys, describe)}
            out[split].append(text_row(f"clinc_{split}_{i}", "clinc", "A request typed to a virtual assistant.",
                                       f"Request: {r['text']}", q, {"intent": gold}))
    return out


MASSIVE_LANGS = ["en", "pt", "es", "fr", "de", "it", "ja", "zh-CN", "hi", "ar", "ru", "sw"]


def build_massive(n, rng):
    out = {"train": [], "test": []}
    for split in ("train", "test"):
        per_lang = max(1, n[split] // len(MASSIVE_LANGS))
        for lang in MASSIVE_LANGS:
            import gzip

            from huggingface_hub import hf_hub_download

            path = hf_hub_download("mteb/amazon_massive_intent", f"{split}/{lang}.json.gz", repo_type="dataset")
            rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8") if line.strip()]
            intents = sorted({r["label_text"] for r in rows})
            scenarios = sorted({i.split("_")[0] for i in intents})
            rng.shuffle(rows)
            for i, r in enumerate(rows[:per_lang]):
                gold = r["label_text"]
                scen = gold.split("_")[0]
                related = [x for x in intents if x.startswith(scen + "_") and x != gold]
                q = {
                    "scenario": choice_q("Which area does the request belong to?", scenarios),
                    "intent": choice_q("Which action is requested?", shortlist(gold, intents, 10, rng, related)),
                }
                out[split].append(text_row(f"massive_{lang}_{split}_{i}", "massive",
                                           "A request spoken to a home voice assistant (transcribed).",
                                           f"Request: {r['text']}", q, {"scenario": scen, "intent": gold}))
    return out


NLI = {"type": "choice", "instructions": "How does the second statement relate to the first?", "criteria": {
    "entailment": "The second statement must be true if the first is true.",
    "neutral": "The second statement may or may not be true.",
    "contradiction": "The second statement cannot be true if the first is true.",
}}


def build_mnli(n, rng):
    out = {"train": [], "test": []}
    names = ["entailment", "neutral", "contradiction"]
    for split, src in (("train", "train"), ("test", "validation_matched")):
        rows = read_rows("nyu-mll/multi_nli", shards("nyu-mll/multi_nli", f"*{src}*.parquet")[0], 20000,
                         columns=["premise", "hypothesis", "label", "pairID"])
        rows = [r for r in rows if r["label"] in (0, 1, 2)]
        rng.shuffle(rows)
        for r in rows[: n[split]]:
            out[split].append(text_row(f"mnli_{r['pairID']}", "mnli", "Two statements to compare.",
                                       f"First: {r['premise']}\nSecond: {r['hypothesis']}", {"relation": NLI},
                                       {"relation": names[r["label"]]}))
    return out


# --------------------------------------------------------------------------- audio sources

FLEURS_LANGS = {"en_us": "English", "pt_br": "Portuguese", "es_419": "Spanish", "fr_fr": "French",
                "de_de": "German", "hi_in": "Hindi", "ar_eg": "Arabic", "cmn_hans_cn": "Mandarin Chinese",
                "ja_jp": "Japanese", "sw_ke": "Swahili", "ru_ru": "Russian", "it_it": "Italian"}


def build_fleurs(n, rng):
    """Spoken sentences: which transcript matches (choice), which language (choice), does it say X (noul)."""
    out = {"train": [], "test": []}
    for split, src in (("train", "train"), ("test", "test")):
        per_lang = max(1, n[split] // len(FLEURS_LANGS))
        for code, lang in FLEURS_LANGS.items():
            path = f"parquet-data/{code}/{src}-00000-of-00001.parquet"
            rows = read_rows("google/fleurs", path, per_lang * 3)
            texts = [r["raw_transcription"] for r in rows]
            for i, r in enumerate(rows[:per_lang]):
                gold_t = r["raw_transcription"]
                others = [t for t in texts if t != gold_t]
                cands = [gold_t] + rng.sample(others, min(3, len(others)))
                rng.shuffle(cands)
                keys = [f"t{j + 1}" for j in range(len(cands))]
                says_true = rng.random() < 0.5
                claim = gold_t if says_true else rng.choice(others)
                langs = shortlist(lang, list(FLEURS_LANGS.values()), 6, rng)
                questions = {
                    "transcript": {"type": "choice", "instructions": "Which transcript matches the recording?",
                                   "criteria": dict(zip(keys, cands))},
                    "language": {"type": "choice", "instructions": "Which language is spoken?",
                                 "criteria": {l: l for l in langs}},
                    "says": {"type": "noul", "instructions": f"The speaker says: \"{claim}\"",
                             "criteria": {"true": "The recording says exactly this.", "false": "The recording says something else."}},
                }
                labels = {"transcript": keys[cands.index(gold_t)], "language": lang,
                          "says": "true" if says_true else "false"}
                head = f"A short recording of someone reading a sentence aloud. {PREFIX_NOTE}\n"
                audio = _wav16k(r["audio"]["bytes"])
                out[split].append(make_row(f"fleurs_{code}_{r['id']}_{i}", "fleurs", questions, labels,
                                           {"media": f"{head}Recording: <|audio|>"}, ["media"], audio=[audio]))
    return out


def _intent_question(gold, intents, rng):
    scen = gold.split("_")[0]
    related = [x for x in intents if x.startswith(scen + "_") and x != gold]
    return choice_q("Which action is requested?", shortlist(gold, intents, 10, rng, related))


def build_slurp(n, rng):
    out = {"train": [], "test": []}
    for split in ("train", "test"):
        path = shards("qmeeus/slurp", f"*/{split}-*.parquet")[0]
        from huggingface_hub import hf_hub_download

        pf = pq.ParquetFile(hf_hub_download("qmeeus/slurp", path, repo_type="dataset"))
        names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        rows = read_rows("qmeeus/slurp", path, n[split] * 4)
        seen, picked = set(), []
        for r in rows:  # several recordings per sentence: keep one
            if r["sentence"] not in seen and "_" in names[r["intent"]]:
                seen.add(r["sentence"])
                picked.append(r)
        intents = sorted({x for x in names if "_" in x})
        scenarios = sorted({x.split("_")[0] for x in intents})
        for i, r in enumerate(picked[: n[split]]):
            gold = names[r["intent"]]
            questions = {"scenario": choice_q("Which area does the request belong to?", scenarios),
                         "intent": _intent_question(gold, intents, rng)}
            head = f"A request spoken to a home voice assistant. {PREFIX_NOTE}\n"
            texts = {"text": f"{head}Transcript: {r['sentence']}", "media": f"{head}Recording: <|audio|>",
                     "both": f"{head}Transcript: {r['sentence']}\nRecording: <|audio|>"}
            out[split].append(make_row(f"slurp_{r['slurp_id']}_{i}", "slurp", questions,
                                       {"scenario": gold.split("_")[0], "intent": gold}, texts,
                                       ["text", "media", "both"], audio=[_wav16k(r["audio"]["bytes"])]))
    return out


def build_minds14(n, rng):
    """English rows outside our mm test split, plus the other 11 languages (train only)."""
    rows = build_calls("train", n["train"], rng, configs=MINDS_ALL)
    out = []
    for r in rows:
        r = {**r, "workflow": "minds14", "source": "minds14", "variants": ["text", "media", "both"]}
        out.append(r)
    return {"train": out, "test": []}


CREMA_EMOTION = choice_q("Which emotion does the speaker express?", ["anger", "disgust", "fear", "happy", "neutral", "sad"])
CREMA_INTENSITY = {"type": "score", "instructions": "How intense is the emotion?",
                   "criteria": ["Low intensity.", "Medium intensity.", "High intensity."]}
CREMA_SENTENCES = {"IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
                   "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
                   "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
                   "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
                   "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes."}


def build_cremad(n, rng):
    out = {"train": [], "test": []}
    for split in ("train", "test"):
        rows = read_rows("confit/cremad-parquet", f"data/{split}-00000-of-00001.parquet")
        rng.shuffle(rows)
        for i, r in enumerate(rows[: n[split]]):
            stem = r["file"].rsplit("/", 1)[-1].removesuffix(".wav")
            _, sent, _, level = stem.split("_")
            questions, labels = {"emotion": CREMA_EMOTION}, {"emotion": r["emotion"]}
            if level in ("LO", "MD", "HI"):
                questions["intensity"], labels["intensity"] = CREMA_INTENSITY, {"LO": "0", "MD": "1", "HI": "2"}[level]
            head = f"An actor says a short scripted sentence. {PREFIX_NOTE}\n"
            line = CREMA_SENTENCES.get(sent, "")
            texts = {"media": f"{head}Recording: <|audio|>", "both": f"{head}Script: {line}\nRecording: <|audio|>"}
            out[split].append(make_row(f"cremad_{stem}", "cremad", questions, labels, texts, ["media", "both"],
                                       audio=[_wav16k(r["audio"]["bytes"])]))
    return out


BELEBELE_LANGS = {"train": ["por_Latn", "spa_Latn", "fra_Latn", "deu_Latn"], "test": ["eng_Latn"]}
BELEBELE_TEST_FRAC = 0.2  # share of passages (by link) reserved for test in every language


def build_belebele(n, rng):
    """Spoken passage (sentences concatenated, first speaker each) + 4-option reading comprehension.

    Belebele has the same passages and questions in every language, so the train/test split is by passage:
    test passages are excluded from training in all languages (no translated leakage).
    """
    import numpy as np
    import soundfile as sf

    out = {"train": [], "test": []}
    for split, langs in BELEBELE_LANGS.items():
        for lang in langs:
            files = shards("WueNLP/belebele-fleurs", f"data/{lang}/test-*.parquet")
            rows = []
            for f in files[: 2 if split == "test" else 1]:
                rows += [r for r in read_rows("WueNLP/belebele-fleurs", f)
                         if split_of(r["link"], BELEBELE_TEST_FRAC) == split]
            for r in rows[: max(1, n[split] // len(langs))]:
                pieces = []
                for s in r["sentence_data"]:
                    wav, sr = sf.read(io.BytesIO(_wav16k(s["audio"][0]["bytes"])), dtype="float32")
                    pieces += [wav, np.zeros(4800, np.float32)]
                buf = io.BytesIO()
                sf.write(buf, np.concatenate(pieces), 16000, format="WAV", subtype="PCM_16")
                answers = [r[f"mc_answer{j}"] for j in range(1, 5)]
                keys = ["a", "b", "c", "d"]
                q = {"answer": {"type": "choice", "instructions": r["question"], "criteria": dict(zip(keys, answers))}}
                head = f"A short passage read aloud, followed by a question about it. {PREFIX_NOTE}\n"
                texts = {"text": f"{head}Passage: {r['flores_passage']}", "media": f"{head}Passage recording: <|audio|>",
                         "both": f"{head}Passage: {r['flores_passage']}\nPassage recording: <|audio|>"}
                out[split].append(make_row(f"belebele_{lang}_{r['link'][-40:]}_{r['question_number']}", "belebele_fleurs",
                                           q, {"answer": keys[int(r["correct_answer_num"]) - 1]}, texts,
                                           ["text", "media", "both"], audio=[buf.getvalue()]))
    return out


# --------------------------------------------------------------------------- image sources


def _img(b):
    from PIL import Image

    return jpeg_bytes(Image.open(io.BytesIO(b)), max_side=1024)


def build_textvqa(n, rng):
    """Text-in-image QA: options = annotator answers (soft gold from their counts) + OCR/other distractors."""
    out = {"train": [], "test": []}
    for split, src in (("train", "train"), ("test", "validation")):
        rows = read_rows("lmms-lab-encoder/textvqa", shards("lmms-lab-encoder/textvqa", f"*/{src}-*.parquet")[0], n[split] * 2)
        answer_pool = [a for r in rows for a in set(r["answers"])]
        for r in rows[: n[split]]:
            counts = Counter(a.strip().lower() for a in r["answers"] if a.strip())
            top = [a for a, _ in counts.most_common(3)]
            distract = [t.lower() for t in r.get("ocr_tokens") or [] if t.lower() not in counts]
            opts = list(dict.fromkeys(top + rng.sample(distract, min(2, len(distract)))
                                      + rng.sample(answer_pool, 4)))[:6]
            rng.shuffle(opts)
            keys = [f"o{j + 1}" for j in range(len(opts))]
            q = {"answer": {"type": "choice", "instructions": r["question"].capitalize(), "criteria": dict(zip(keys, opts))}}
            soft = {k: counts.get(o, 0) for k, o in zip(keys, opts)}
            head = f"A photo, and a question about text visible in it. {PREFIX_NOTE}\n"
            ocr = " ".join(r.get("ocr_tokens") or [])
            texts = {"media": f"{head}Photo: <|image|>", "both": f"{head}Text detected in the photo: {ocr}\nPhoto: <|image|>"}
            out[split].append(make_row(f"textvqa_{r['question_id']}", "textvqa", q, {"answer": soft}, texts,
                                       ["media", "both"], images=[_img(r["image"]["bytes"])]))
    return out


def build_gqa(n, rng):
    """Up to 4 questions about one image per row: yes/no -> noul, others -> choice among same-type answers."""
    out = {"train": [], "test": []}
    for split, src in (("train", "val"), ("test", "testdev")):
        img_rows = read_rows("lmms-lab-encoder/GQA", shards("lmms-lab-encoder/GQA", f"{src}_balanced_images/*.parquet")[0],
                             int(n[split] * 1.3))
        images = {r["id"]: r["image"]["bytes"] for r in img_rows}
        qs = []
        for path in shards("lmms-lab-encoder/GQA", f"{src}_balanced_instructions/*.parquet"):
            qs += [q for q in read_rows("lmms-lab-encoder/GQA", path, columns=["id", "imageId", "question", "answer", "types"])
                   if q["imageId"] in images]
        pool = defaultdict(set)
        for q in qs:
            pool[q["types"]["detailed"]].add(q["answer"])
        by_img = defaultdict(list)
        for q in qs:
            by_img[q["imageId"]].append(q)
        for img_id in list(images)[: n[split]]:
            items = by_img.get(img_id, [])
            rng.shuffle(items)
            questions, labels = {}, {}
            for j, q in enumerate(items[:4]):
                name = f"q{j + 1}"
                if q["answer"] in ("yes", "no"):
                    questions[name] = {"type": "noul", "instructions": q["question"],
                                       "criteria": {"true": "Yes.", "false": "No."}}
                    labels[name] = "true" if q["answer"] == "yes" else "false"
                else:
                    cands = shortlist(q["answer"], sorted(pool[q["types"]["detailed"]]), 6, rng)
                    if len(cands) < 2:
                        continue
                    questions[name] = {"type": "choice", "instructions": q["question"], "criteria": {c: c for c in cands}}
                    labels[name] = q["answer"]
            if not questions:
                continue
            head = f"A photo of an everyday scene, with questions about it. {PREFIX_NOTE}\n"
            out[split].append(make_row(f"gqa_{src}_{img_id}", "gqa", questions, labels,
                                       {"media": f"{head}Photo: <|image|>"}, ["media"], images=[_img(images[img_id])]))
    return out


EUROSAT = {"AnnualCrop": "Fields of annual crops", "Forest": "Forest", "HerbaceousVegetation": "Grassland or other herbaceous vegetation",
           "Highway": "A highway or major road", "Industrial": "Industrial buildings", "Pasture": "Pasture",
           "PermanentCrop": "Orchards, vineyards or other permanent crops", "Residential": "Residential buildings",
           "River": "A river", "SeaLake": "Sea or lake water"}


def build_eurosat(n, rng):
    out = {"train": [], "test": []}
    for split in ("train", "test"):
        rows = read_rows("timm/eurosat-rgb", shards("timm/eurosat-rgb", f"*{split}*.parquet")[0])
        rng.shuffle(rows)
        for r in rows[: n[split]]:
            label = r["image_id"].rsplit("_", 1)[0]
            q = {"land_use": {"type": "choice", "instructions": "What is the main land use in this satellite tile?",
                              "criteria": EUROSAT}}
            head = f"A small satellite image tile. {PREFIX_NOTE}\n"
            out[split].append(make_row(f"eurosat_{r['image_id']}", "eurosat", q, {"land_use": label},
                                       {"media": f"{head}Tile: <|image|>"}, ["media"], images=[_img(r["image"]["bytes"])]))
    return out


def build_pets(n, rng):
    rows = read_rows("enterprise-explorers/oxford-pets", shards("enterprise-explorers/oxford-pets", "*.parquet")[0])
    breeds = {"dog": sorted({r["label"] for r in rows if r["dog"]}), "cat": sorted({r["label"] for r in rows if not r["dog"]})}
    out = {"train": [], "test": []}
    rng.shuffle(rows)
    for r in rows:
        split = split_of(r["path"], 0.2)
        if len(out[split]) >= n[split]:
            continue
        species = "dog" if r["dog"] else "cat"
        q = {"dog": {"type": "noul", "instructions": "The animal in the photo is a dog.",
                     "criteria": {"true": "It is a dog.", "false": "It is a cat."}},
             "breed": choice_q("Which breed is the animal?", shortlist(r["label"], breeds[species], 8, rng))}
        head = f"A photo of a pet. {PREFIX_NOTE}\n"
        out[split].append(make_row(f"pets_{Path(r['path']).stem}", "oxford_pets", q,
                                   {"dog": "true" if r["dog"] else "false", "breed": r["label"]},
                                   {"media": f"{head}Photo: <|image|>"}, ["media"], images=[_img(r["image"]["bytes"])]))
    return out


def build_cord(n, rng):
    rows = build_receipts("train", n["train"], rng)
    return {"train": [{**r, "workflow": "cord", "source": "cord", "variants": ["text", "media", "both"]} for r in rows],
            "test": []}


def build_snake(n, rng):
    """Teacher-labelled Snake decisions (gemma_decision/snake.py). States come from games where the teacher
    plays with 25% random moves, so risky positions appear too; the label is always the teacher's move.
    Renderings: text = ASCII board, media = board image, both = both (never the `hints` mode)."""
    from .snake import SnakeGame, teacher

    out = {"train": [], "test": []}
    for split, seed0 in (("train", 0), ("test", 900_000)):  # disjoint game seeds
        seed = seed0
        while len(out[split]) < n[split]:
            game = SnakeGame(seed=seed)
            play_rng = random.Random(f"snake-play-{seed}")
            while game.alive and game.steps < 400 and len(out[split]) < n[split]:
                label = teacher(game)
                if rng.random() < 0.5:  # thin consecutive near-duplicate states
                    grid, img = game.decision("grid"), game.decision("image")
                    texts = {"text": grid["state"], "media": img["state"],
                             "both": grid["state"] + "\nBoard image: <|image|>"}
                    out[split].append(make_row(f"snake_{seed}_{game.steps}", "snake", grid["questions"],
                                               {"move": label}, texts, ["text", "media", "both"],
                                               images=img["images"]))
                game.step(label if play_rng.random() >= 0.25 else play_rng.choice(game.legal_moves()))
            seed += 1
    return out


# --------------------------------------------------------------------------- main

PILOT = {  # source: (builder, train rows, test rows)
    "banking77": (build_banking77, 800, 200), "clinc": (build_clinc, 800, 200), "massive": (build_massive, 1200, 240),
    "mnli": (build_mnli, 600, 200),
    "fleurs": (build_fleurs, 1500, 240), "slurp": (build_slurp, 1000, 200), "minds14": (build_minds14, 1000, 0),
    "belebele_fleurs": (build_belebele, 400, 100), "cremad": (build_cremad, 1000, 200),
    "textvqa": (build_textvqa, 1500, 200), "gqa": (build_gqa, 1200, 200), "eurosat": (build_eurosat, 500, 150),
    "oxford_pets": (build_pets, 500, 150), "cord": (build_cord, 800, 0),
}
EXTRA = {"snake": (build_snake, 6000, 600)}  # built on request (--sources snake), not part of the pilot mix
SCHEMA = pa.schema([
    ("id", pa.string()), ("workflow", pa.string()), ("source", pa.string()), ("license", pa.string()),
    ("state_text", pa.string()), ("state_media", pa.string()), ("state_both", pa.string()),
    ("variants", pa.list_(pa.string())), ("images", pa.list_(pa.binary())), ("audio", pa.list_(pa.binary())),
    ("questions", pa.string()), ("gold", pa.string()), ("label_agreement", pa.string()), ("n_questions", pa.int64()),
])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=list(PILOT), help=f"any of {list(PILOT) + list(EXTRA)}")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply row counts (e.g. 0.01 for a smoke build)")
    ap.add_argument("--out", type=Path, default=Path("data/pilot"))
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    for src in args.sources:
        builder, n_train, n_test = {**PILOT, **EXTRA}[src]
        n = {"train": max(1, int(n_train * args.scale)), "test": int(n_test * args.scale) if n_test else 0}
        built = builder(n, random.Random(f"{args.seed}-{src}"))
        for split, rows in built.items():
            if not rows:
                continue
            rows = [{**r, "variants": r.get("variants") or ["text", "media", "both"]} for r in rows]
            (args.out / split).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), args.out / split / f"{src}.parquet")
            print(f"{src:16s} {split:5s} {len(rows):5d} rows {sum(r['n_questions'] for r in rows):6d} questions", flush=True)


if __name__ == "__main__":
    main()
