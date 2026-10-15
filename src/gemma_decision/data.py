"""Typed-decision rows -> one serialized prompt per row, plus the token spans and targets the head needs.

A row has a `state` and a dict of `questions`, each of type:
  choice  criteria: {key: description}         gold.probabilities: {key: p}
  noul    criteria: {"true": .., "false": ..}  gold.probabilities: {"true": p, "false": p}  (criteria may be absent)
  score   criteria: [level0, level1, ...]      gold.probabilities: {"0": p, "1": p, ...}    (ordered, lowest first)

Every question is turned into a list of options, so the head scores all three types the same way.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import torch

PREFIX = "task: classification | query: "
TYPES = {"choice": 0, "noul": 1, "score": 2}
DEFAULT_NOUL = {"true": "Yes, the statement holds.", "false": "No, the statement does not hold."}

# Slot kinds in the head's slot sequence.
KIND_PAD, KIND_QUESTION, KIND_OPTION = 0, 1, 2


@dataclass
class Question:
    name: str
    type: str
    instructions: str
    keys: list[str]  # option keys; for score these are "0", "1", ...
    texts: list[str]  # option descriptions (may be empty)
    target: list[float] | None = None  # gold probabilities aligned with keys
    label: int | None = None  # index of gold label in keys
    gold_score: float | None = None  # expected level, score questions only
    agree: bool | None = None  # annotators' argmax agreed


@dataclass
class Example:
    id: str
    workflow: str
    state: str
    questions: list[Question] = field(default_factory=list)


def parse_questions(questions: dict, gold: dict | None = None, agreement: dict | None = None) -> list[Question]:
    out = []
    for name, q in questions.items():
        qtype = q["type"]
        crit = q.get("criteria")
        if qtype == "score":
            keys = [str(i) for i in range(len(crit))]
            texts = list(crit)
        elif qtype == "noul":
            crit = crit or DEFAULT_NOUL
            keys = ["true", "false"]
            texts = [crit.get(k) or DEFAULT_NOUL[k] for k in keys]
        else:
            keys = list(crit.keys())
            texts = [v or "" for v in crit.values()]
        question = Question(name, qtype, q.get("instructions", ""), keys, texts)
        if gold is not None:
            g = gold[name]
            probs = [float(g["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(probs)
            question.target = [p / s for p in probs] if s > 0 else [1.0 / len(keys)] * len(keys)
            question.label = keys.index(str(g["label"]))
            if qtype == "score":
                question.gold_score = float(g["score"])
        if agreement is not None and name in agreement:
            question.agree = bool(agreement[name]["argmax_agree"])
        out.append(question)
    return out


def row_to_example(row: dict) -> Example:
    agreement = json.loads(row["label_agreement"]) if row.get("label_agreement") else None
    return Example(
        id=row["id"],
        workflow=row.get("workflow", ""),
        state=row["state"],
        questions=parse_questions(json.loads(row["questions"]), json.loads(row["gold"]), agreement),
    )


def load_examples(name: str = "LocalLLaMA/typed-decisions", config: str = "all", split: str = "train") -> list[Example]:
    from datasets import load_dataset

    return [row_to_example(r) for r in load_dataset(name, config, split=split)]


def split_holdout(examples: list[Example], frac: float = 0.1, seed: int = 3407) -> tuple[list[Example], list[Example]]:
    """Hold out `frac` of each workflow for validation and temperature calibration."""
    rng = random.Random(seed)
    by_wf: dict[str, list[Example]] = {}
    for ex in examples:
        by_wf.setdefault(ex.workflow, []).append(ex)
    train, held = [], []
    for wf in sorted(by_wf):
        items = by_wf[wf][:]
        rng.shuffle(items)
        n = max(1, round(len(items) * frac))
        held += items[:n]
        train += items[n:]
    return train, held


# --------------------------------------------------------------------------- serialization


@dataclass
class Encoded:
    input_ids: list[int]
    # One entry per question, in serialized order:
    #   (question token span, [option token spans]) with spans as (start, end) token indices.
    spans: list[tuple[tuple[int, int], list[tuple[int, int]]]]
    questions: list[Question]  # possibly reordered copies (option order may be shuffled)


def _reorder(q: Question, perm: list[int]) -> Question:
    pick = lambda xs: None if xs is None else [xs[i] for i in perm]  # noqa: E731
    return Question(
        q.name, q.type, q.instructions, pick(q.keys), pick(q.texts), pick(q.target),
        None if q.label is None else perm.index(q.label), q.gold_score, q.agree,
    )


def _render(state: str, questions: list[Question]):
    """Build the prompt text and the character spans of every question and option."""
    parts = [PREFIX, state, "\n"]
    pos = sum(len(p) for p in parts)
    char_spans = []
    for q in questions:
        head = f"\n[question] {q.name} ({q.type}): {q.instructions}"
        q_span = (pos + len("\n[question] "), pos + len(head))
        parts.append(head)
        pos += len(head)
        opt_spans = []
        for key, text in zip(q.keys, q.texts):
            line = f"\n  [option] {key}: {text}" if text else f"\n  [option] {key}"
            opt_spans.append((pos + len("\n  [option] "), pos + len(line)))
            parts.append(line)
            pos += len(line)
        char_spans.append((q_span, opt_spans))
    return "".join(parts), char_spans


def _char_to_token_span(offsets: list[tuple[int, int]], span: tuple[int, int]) -> tuple[int, int]:
    s, e = span
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
    if not idx:
        raise ValueError(f"span {span} has no tokens (truncated?)")
    return idx[0], idx[-1] + 1


def encode(
    tokenizer, state: str, questions: list[Question], max_length: int = 1024,
    rng: random.Random | None = None,
) -> Encoded:
    """Serialize one decision request. With `rng`, question order and choice/noul option order are shuffled."""
    questions = list(questions)
    if rng is not None:
        rng.shuffle(questions)
        reordered = []
        for q in questions:
            perm = list(range(len(q.keys)))
            if q.type != "score":  # score levels keep their order
                rng.shuffle(perm)
            reordered.append(_reorder(q, perm))
        questions = reordered

    # Truncate the end of the state (never the questions) until the prompt fits.
    while True:
        text, char_spans = _render(state, questions)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        overflow = len(enc["input_ids"]) - max_length
        if overflow <= 0:
            break
        cut = max(overflow * 3, 16)  # ~3+ chars per token
        if cut >= len(state):
            raise ValueError("questions alone exceed max_length")
        state = state[:-cut]

    offsets = enc["offset_mapping"]
    spans = [
        (_char_to_token_span(offsets, qs), [_char_to_token_span(offsets, os) for os in opts])
        for qs, opts in char_spans
    ]
    return Encoded(enc["input_ids"], spans, questions)


# --------------------------------------------------------------------------- batching


def collate(encoded: list[Encoded], pad_id: int, max_levels: int = 16) -> dict:
    """Pad a list of Encoded into tensors for the encoder, head and loss.

    Slots (per row): every question vector followed by its option vectors. The head mean-pools each slot's
    token span via `slot_weights` [B, S, T].
    Questions are flattened across the batch: `q_opt_slots` [Q, K] indexes into the flattened B*S slots.
    """
    B = len(encoded)
    T = max(len(e.input_ids) for e in encoded)
    S = max(sum(1 + len(opts) for _, opts in e.spans) for e in encoded)
    K = max(len(q.keys) for e in encoded for q in e.questions)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    slot_weights = torch.zeros((B, S, T))
    slot_kind = torch.zeros((B, S), dtype=torch.long)
    slot_type = torch.zeros((B, S), dtype=torch.long)
    slot_qid = torch.zeros((B, S), dtype=torch.long)
    slot_level = torch.zeros((B, S), dtype=torch.long)  # 0 = not a score level, else level+1

    q_slot, q_opt_slots, q_opt_mask, q_type, q_target, q_label = [], [], [], [], [], []
    q_meta = []  # (row index, Question) for metrics / outputs

    for b, e in enumerate(encoded):
        n = len(e.input_ids)
        input_ids[b, :n] = torch.tensor(e.input_ids)
        attention_mask[b, :n] = 1
        s = 0
        for qi, ((qs, opts), q) in enumerate(zip(e.spans, e.questions)):
            t = TYPES[q.type]

            def put(span, kind, level=0):
                nonlocal s
                a, z = span
                slot_weights[b, s, a:z] = 1.0 / (z - a)
                slot_kind[b, s], slot_type[b, s], slot_qid[b, s], slot_level[b, s] = kind, t, qi, level
                s += 1
                return b * S + s - 1

            q_slot.append(put(qs, KIND_QUESTION))
            idx = [put(os, KIND_OPTION, (k + 1) if q.type == "score" else 0) for k, os in enumerate(opts)]
            q_opt_slots.append(idx + [0] * (K - len(idx)))
            q_opt_mask.append([True] * len(idx) + [False] * (K - len(idx)))
            q_type.append(t)
            tgt = q.target or [0.0] * len(idx)
            q_target.append(tgt + [0.0] * (K - len(idx)))
            q_label.append(-1 if q.label is None else q.label)
            q_meta.append((b, q))

    slot_level.clamp_(max=max_levels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "slot_weights": slot_weights,
        "slot_kind": slot_kind,
        "slot_type": slot_type,
        "slot_qid": slot_qid,
        "slot_level": slot_level,
        "q_slot": torch.tensor(q_slot),
        "q_opt_slots": torch.tensor(q_opt_slots),
        "q_opt_mask": torch.tensor(q_opt_mask),
        "q_type": torch.tensor(q_type),
        "q_target": torch.tensor(q_target),
        "q_label": torch.tensor(q_label),
        "q_meta": q_meta,
    }


def length_bucketed_batches(lengths: list[int], batch_size: int, rng: random.Random, pool: int = 50) -> list[list[int]]:
    """Shuffle, then sort within pools of `pool` batches so each batch has similar lengths (less padding)."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    span = batch_size * pool
    for i in range(0, len(idx), span):
        chunk = sorted(idx[i : i + span], key=lambda j: lengths[j])
        batches += [chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches

