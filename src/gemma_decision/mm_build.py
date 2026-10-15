"""`gd-mm-build --split test` - build the multimodal typed-decisions set from public datasets.

Each row has the same `questions` / `gold` / `label_agreement` JSON as LocalLLaMA/typed-decisions, plus the
same evidence in three renderings so a model's text-vs-media gap can be measured on identical decisions:

  state_text   text evidence only (tweet text / call transcript / receipt OCR words)
  state_media  media only (<|image|> / <|audio|> placeholders)
  state_both   text and media

Workflows (licenses follow the source; see data/mm-decisions/README.md):
  crisis_response  QCRI/CrisisMMD              tweet text + image        CC BY-NC-SA 4.0
  call_center      PolyAI/minds14 (English)    call transcript + audio   CC BY 4.0
  receipts         naver-clova-ix/cord-v2      OCR words + receipt image CC BY 4.0

Gold labels are single human labels (CrisisMMD, MINDS-14) or deterministic functions of CORD's parsed ground
truth, so `probabilities` are one-hot and `argmax_agree` is true.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PREFIX_NOTE = "Evidence for the decision follows."

# --------------------------------------------------------------------------- shared helpers


def gold_entry(qtype: str, keys: list[str], label: str) -> dict:
    probs = {k: float(k == label) for k in keys}
    g = {"type": qtype, "label": label, "confidence": 1.0, "probabilities": probs}
    if qtype == "noul":
        g["noul"] = probs["true"]
    if qtype == "score":
        g["score"] = float(int(label))
    return g


def make_row(rid, workflow, source, license_, questions, labels, texts, images=(), audio=()):
    gold, agreement = {}, {}
    for name, q in questions.items():
        keys = [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else (
            ["true", "false"] if q["type"] == "noul" else list(q["criteria"]))
        gold[name] = gold_entry(q["type"], keys, labels[name])
        agreement[name] = {"argmax_agree": True, "argmax_majority": labels[name], "total_variation": 0.0}
    return {
        "id": rid, "workflow": workflow, "source": source, "license": license_,
        "state_text": texts["text"], "state_media": texts["media"], "state_both": texts["both"],
        "images": list(images), "audio": list(audio),
        "questions": json.dumps(questions), "gold": json.dumps(gold),
        "label_agreement": json.dumps(agreement), "n_questions": len(questions),
    }


def jpeg_bytes(img, max_side: int = 1600) -> bytes:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def stable_frac(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------------------- crisis_response (CrisisMMD)

CRISIS_INFORMATIVE = {
    "type": "noul",
    "instructions": "This post contains information that is useful for humanitarian response.",
    "criteria": {
        "true": "Useful for responders: damage, casualties, needs, rescue or relief information.",
        "false": "Not useful for humanitarian response.",
    },
}
CRISIS_CATEGORY = {
    "type": "choice",
    "instructions": "Which humanitarian category best describes this post?",
    "criteria": {
        "affected_individuals": "People affected by the disaster: displaced, evacuated, sheltering or otherwise impacted.",
        "infrastructure_and_utility_damage": "Damage to buildings, roads, bridges, power lines or other infrastructure.",
        "injured_or_dead_people": "Injured or dead people.",
        "missing_or_found_people": "People reported missing, or missing people who were found.",
        "not_humanitarian": "Not relevant to humanitarian aid.",
        "other_relevant_information": "Relevant to the disaster but not covered by the other categories.",
        "rescue_volunteering_or_donation_effort": "Rescue operations, volunteering, donations or relief efforts.",
        "vehicle_damage": "Damaged vehicles.",
    },
}
CRISIS_DAMAGE = {
    "type": "score",
    "instructions": "How severe is the damage shown or described?",
    "criteria": [
        "Little or no damage.",
        "Mild damage: partial damage to buildings, roads or vehicles.",
        "Severe damage: destroyed buildings, collapsed infrastructure or widespread devastation.",
    ],
}
DAMAGE_LEVELS = {"little_or_no_damage": "0", "mild_damage": "1", "severe_damage": "2"}


def build_crisis(split: str, n: int, rng: random.Random) -> list[dict]:
    from huggingface_hub import hf_hub_download

    repo = "QCRI/CrisisMMD"
    src_split = {"test": "test", "train": "train", "dev": "dev"}[split]
    ann = {}
    for task in ("informative", "humanitarian", "damage"):
        path = hf_hub_download(repo, f"{task}/{src_split}.json", repo_type="dataset")
        for r in json.loads(Path(path).read_text()):
            item = ann.setdefault(r["image_id"], {"meta": r, "labels": {}})
            item["labels"][task] = r["label"]

    # The three tasks label mostly different images (only ~330 of 2,237 test images carry two labels).
    # Fill rows with multi-question images first, then damage-labeled ones (the most image-dependent task).
    ids = list(ann)
    rng.shuffle(ids)
    ids.sort(key=lambda k: (-len(ann[k]["labels"]), "damage" not in ann[k]["labels"]))
    picked = ids[:n]
    rng.shuffle(picked)

    def fetch(image_id):
        path = hf_hub_download(repo, ann[image_id]["meta"]["image_path"], repo_type="dataset")
        return Path(path).read_bytes()

    with ThreadPoolExecutor(8) as pool:
        images = list(pool.map(fetch, picked))

    rows = []
    for image_id, img in zip(picked, images):
        meta, lab = ann[image_id]["meta"], ann[image_id]["labels"]
        questions, labels = {}, {}
        if "informative" in lab:
            questions["informative"] = CRISIS_INFORMATIVE
            labels["informative"] = "true" if lab["informative"] == "informative" else "false"
        if "humanitarian" in lab:
            questions["category"] = CRISIS_CATEGORY
            labels["category"] = lab["humanitarian"]
        if "damage" in lab:
            questions["damage"] = CRISIS_DAMAGE
            labels["damage"] = DAMAGE_LEVELS[lab["damage"]]
        event = meta["event_name"].replace("_", " ")
        head = f"A social media post published during the {event}. {PREFIX_NOTE}\n"
        tweet = meta["tweet_text"].strip()
        texts = {
            "text": f"{head}Post text: {tweet}",
            "media": f"{head}Post image: <|image|>",
            "both": f"{head}Post text: {tweet}\nPost image: <|image|>",
        }
        rows.append(make_row(f"crisis_{image_id}", "crisis_response", repo, "cc-by-nc-sa-4.0",
                             questions, labels, texts, images=[img]))
    return rows


# --------------------------------------------------------------------------- call_center (MINDS-14)

CALL_INTENT = {
    "type": "choice",
    "instructions": "What does the caller want to do?",
    "criteria": {
        "abroad": "Use the card or account abroad, or tell the bank about travel.",
        "address": "Change or update the address on the account.",
        "app_error": "Report a problem or error with the banking app.",
        "atm_limit": "Ask about or change ATM withdrawal limits.",
        "balance": "Check the account balance.",
        "business_loan": "Apply for or ask about a business loan.",
        "card_issues": "Report a lost, stolen, damaged or non-working card.",
        "cash_deposit": "Deposit cash.",
        "direct_debit": "Set up or cancel a direct debit.",
        "freeze": "Freeze or block the account or card.",
        "high_value_payment": "Make a large payment or transfer.",
        "joint_account": "Open or manage a joint account.",
        "latest_transactions": "Review recent transactions.",
        "pay_bill": "Pay a bill.",
    },
}


def _wav16k(audio_bytes: bytes) -> bytes:
    import librosa
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1, 1), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


MINDS_ENGLISH = ("en-US", "en-GB", "en-AU")
MINDS_ALL = MINDS_ENGLISH + ("cs-CZ", "de-DE", "es-ES", "fr-FR", "it-IT", "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "zh-CN")


def build_calls(split: str, n: int, rng: random.Random, configs=MINDS_ENGLISH) -> list[dict]:
    from datasets import Audio, load_dataset

    names = None
    candidates = []
    for cfg in configs:
        ds = load_dataset("PolyAI/minds14", cfg, split="train").cast_column("audio", Audio(decode=False))
        names = ds.features["intent_class"].names
        for r in ds:
            # MINDS-14 ships one split; hold out 40% of the English files (stable hash of the path) as test.
            # Non-English files are never in the test split.
            in_test = cfg in MINDS_ENGLISH and stable_frac(r["path"]) < 0.4
            if (split == "test") == in_test:
                candidates.append((cfg, r))
    rng.shuffle(candidates)
    rows = []
    for cfg, r in candidates[:n]:
        label = names[r["intent_class"]]
        # English: the cleaned English transcript (as in the published test split); others: their own language
        transcript = ((r["english_transcription"] or r["transcription"]) if cfg in MINDS_ENGLISH
                      else (r["transcription"] or r["english_transcription"])).strip()
        head = f"A customer called the bank's phone line. {PREFIX_NOTE}\n"
        texts = {
            "text": f"{head}Call transcript: {transcript}",
            "media": f"{head}Call recording: <|audio|>",
            "both": f"{head}Call transcript: {transcript}\nCall recording: <|audio|>",
        }
        rid = "call_" + hashlib.sha1(r["path"].encode()).hexdigest()[:12]
        audio_bytes = r["audio"]["bytes"] or Path(r["audio"]["path"]).read_bytes()
        rows.append(make_row(rid, "call_center", "PolyAI/minds14", "cc-by-4.0",
                             {"intent": CALL_INTENT}, {"intent": label}, texts, audio=[_wav16k(audio_bytes)]))
    return rows


# --------------------------------------------------------------------------- receipts (CORD-v2)

RECEIPT_PAYMENT = {
    "type": "choice",
    "instructions": "How was this purchase paid?",
    "criteria": {
        "cash": "Paid in cash (cash amount or change shown).",
        "card": "Paid by credit or debit card.",
        "e_money": "Paid with e-money or a mobile wallet.",
        "not_shown": "The receipt does not show the payment method.",
    },
}
RECEIPT_TOTAL = {
    "type": "score",
    "instructions": "What is the receipt's total amount (Indonesian rupiah)?",
    "criteria": ["Under 25,000.", "25,000 to 49,999.", "50,000 to 99,999.", "100,000 to 199,999.", "200,000 or more."],
}
RECEIPT_ITEMS = {
    "type": "score",
    "instructions": "How many distinct menu items are on the receipt?",
    "criteria": ["One item.", "Two or three items.", "Four to six items.", "Seven or more items."],
}
RECEIPT_DISCOUNT = {
    "type": "noul",
    "instructions": "A discount was applied to this purchase.",
    "criteria": {"true": "The receipt shows a discount.", "false": "No discount is shown."},
}


def _amount(s) -> int | None:
    if isinstance(s, list):
        s = s[0] if s else None
    if not isinstance(s, str):
        return None
    digits = re.sub(r"[^\d]", "", s.split(",")[0] if s.count(",") == 1 and len(s.split(",")[1]) == 2 else s)
    return int(digits) if digits else None


def _bucket(x: int, edges: list[int]) -> str:
    return str(sum(x >= e for e in edges))


def build_receipts(split: str, n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    splits = ["test", "validation"] if split == "test" else ["train"]
    rows = []
    for s in splits:
        for i, r in enumerate(load_dataset("naver-clova-ix/cord-v2", split=s)):
            gt = json.loads(r["ground_truth"])
            parse = gt["gt_parse"]
            total = parse.get("total", {}) if isinstance(parse.get("total"), dict) else {}
            sub = parse.get("sub_total", {}) if isinstance(parse.get("sub_total"), dict) else {}
            menu = parse.get("menu", [])
            menu = [menu] if isinstance(menu, dict) else menu

            questions, labels = {}, {}
            if total.get("creditcardprice"):
                pay = "card"
            elif total.get("emoneyprice"):
                pay = "e_money"
            elif total.get("cashprice") or total.get("changeprice"):
                pay = "cash"
            else:
                pay = "not_shown"
            questions["payment"], labels["payment"] = RECEIPT_PAYMENT, pay
            amount = _amount(total.get("total_price"))
            if amount:
                questions["total"], labels["total"] = RECEIPT_TOTAL, _bucket(amount, [25_000, 50_000, 100_000, 200_000])
            if menu:
                questions["items"], labels["items"] = RECEIPT_ITEMS, _bucket(len(menu), [2, 4, 7])
            has_discount = bool(sub.get("discount_price")) or any(isinstance(m, dict) and m.get("discountprice") for m in menu)
            questions["discount"], labels["discount"] = RECEIPT_DISCOUNT, "true" if has_discount else "false"

            lines: dict[int, list[str]] = {}
            for vl in gt.get("valid_line", []):
                for w in vl["words"]:
                    lines.setdefault(w["row_id"], []).append(w["text"])
            ocr = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
            head = f"A scanned purchase receipt from a shop or restaurant. {PREFIX_NOTE}\n"
            texts = {
                "text": f"{head}Receipt text (OCR):\n{ocr}",
                "media": f"{head}Receipt image: <|image|>",
                "both": f"{head}Receipt text (OCR):\n{ocr}\nReceipt image: <|image|>",
            }
            rows.append(make_row(f"receipt_{s}_{i:04d}", "receipts", "naver-clova-ix/cord-v2", "cc-by-4.0",
                                 questions, labels, texts, images=[jpeg_bytes(r["image"])]))
    rng.shuffle(rows)
    return rows[:n]


# --------------------------------------------------------------------------- main

BUILDERS = {"crisis_response": build_crisis, "call_center": build_calls, "receipts": build_receipts}
SCHEMA = pa.schema([
    ("id", pa.string()), ("workflow", pa.string()), ("source", pa.string()), ("license", pa.string()),
    ("state_text", pa.string()), ("state_media", pa.string()), ("state_both", pa.string()),
    ("images", pa.list_(pa.binary())), ("audio", pa.list_(pa.binary())),
    ("questions", pa.string()), ("gold", pa.string()), ("label_agreement", pa.string()), ("n_questions", pa.int64()),
])


CARD = """---
license: other
license_name: mixed (per workflow, see below)
task_categories: [image-classification, audio-classification, text-classification]
language: [en]
pretty_name: Multimodal typed decisions
---

# Multimodal typed decisions

Decision rows in the [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
format (`questions` / `gold` / `label_agreement` JSON; `choice`, `noul`, `score` questions), built from public
image and audio datasets. Each row carries the same evidence in three renderings, so a decision model's
accuracy can be compared on identical decisions with text only, media only, or both:

| column | contents |
|---|---|
| `state_text` | text evidence only (tweet text / call transcript / receipt OCR words) |
| `state_media` | media only, as `<|image|>` / `<|audio|>` placeholders filled in order from `images` / `audio` |
| `state_both` | text and media |
| `images` | JPEG bytes |
| `audio` | 16 kHz mono PCM16 WAV bytes |

Gold labels are single human labels (CrisisMMD, MINDS-14) or deterministic functions of CORD's parsed
ground truth, so `probabilities` are one-hot.

## Workflows

| workflow | source | license | questions |
|---|---|---|---|
| `crisis_response` | [QCRI/CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD) | CC BY-NC-SA 4.0 | informative (noul), humanitarian category (choice, 8), damage severity (score, 3) |
| `call_center` | [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) en-US/GB/AU | CC BY 4.0 | banking intent (choice, 14) |
| `receipts` | [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | CC BY 4.0 | payment method (choice), total amount (score, 5), item count (score, 4), discount (noul) |

`crisis_response` inherits CrisisMMD's non-commercial share-alike license.
CrisisMMD's three tasks label mostly different images, so rows carry 1-3 questions.
MINDS-14 ships one split; 40% of files (by a stable hash of the file path) form the test split.
CORD's test split here is CORD's `test` + `validation`.

## Statistics

{stats}

Built with `gd-mm-build` from the GemmaDecision project.
"""


def write_card(root: Path):
    lines = ["| split | workflow | rows | questions |", "|---|---|---|---|"]
    for path in sorted(root.glob("*/*.parquet")):
        t = pq.read_table(path, columns=["n_questions"])
        lines.append(f"| {path.parent.name} | {path.stem} | {t.num_rows} | {sum(t.column('n_questions').to_pylist())} |")
    (root / "README.md").write_text(CARD.replace("{stats}", "\n".join(lines)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n", type=int, default=600, help="max rows per workflow")
    ap.add_argument("--workflows", nargs="+", default=list(BUILDERS))
    ap.add_argument("--out", type=Path, default=Path("data/mm-decisions"))
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    for wf in args.workflows:
        rows = BUILDERS[wf](args.split, args.n, random.Random(f"{args.seed}-{wf}-{args.split}"))
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out / f"{wf}.parquet")
        n_q = sum(r["n_questions"] for r in rows)
        print(f"{wf}: {len(rows)} rows, {n_q} questions -> {out / f'{wf}.parquet'}")
    write_card(args.out)


if __name__ == "__main__":
    main()
