"""System One judgment rows (S1MB, bekko-system-one-dataset) <-> our decision format.

One rendering is shared by training conversion (`gd-broad-build`) and S1MB evaluation (scripts/s1mb_bench.py),
following the S1MB adapter rules: authored instructions (system prompt first, if any) and criteria in declared
order, no case or question IDs in the text (questions are named "decision" / "decision_<i>"), anonymous choice
keys (A, B, ...), Noul as true/false with the authored descriptions, Score levels ordered by numeric value.
"""

from __future__ import annotations

import json
import string

from .data import Example, parse_questions

MAX_QUESTIONS = 32  # the head embeds question slots (64 max); larger cases are split into several parts
KEYS = list(string.ascii_uppercase) + [f"{a}{b}" for a in "ABCDE" for b in string.ascii_uppercase]


def _text(value_json: str) -> str:
    v = json.loads(value_json)
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def case_from_row(row: dict, task: str | None = None) -> dict | None:
    """{id, group, state, questions: [{id, task, instructions, options: [{id, description, value}]}], targets}.

    With `task`, only that task's questions are kept (None if the case has none)."""
    targets = {t["decision_id"]: dict(zip(t["ids"], t["probabilities"])) for t in row["targets"]}
    qs = []
    for d in row["input"]["decisions"]:
        if d["kind"] != "judgment" or (task and d["type"] != task):
            continue
        instructions = _text(d["instructions_json"])
        if d.get("system_prompt"):
            instructions = f"{d['system_prompt']}\n{instructions}"
        qs.append({"id": d["id"], "task": d["type"], "instructions": instructions,
                   "options": [{"id": c["id"], "description": _text(c["description_json"]), "value": c["value"]}
                               for c in d["criteria"]]})
    if not qs:
        return None
    state = json.loads(row["input"]["state_json"])
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    return {"id": row["case_id"], "group": row.get("group_id"), "state": state, "questions": qs,
            "targets": {q["id"]: targets[q["id"]] for q in qs}}


def split_case(case: dict) -> list[dict]:
    """Cases with more than MAX_QUESTIONS questions become several parts (ids suffixed #0, #1, ...)."""
    qs = case["questions"]
    return [{**case, "id": f"{case['id']}#{n}", "questions": qs[i : i + MAX_QUESTIONS]}
            for n, i in enumerate(range(0, len(qs), MAX_QUESTIONS))]


def render(case: dict) -> tuple[dict, dict, list[list[str]]]:
    """(questions in our schema, gold {name: {our key: probability}}, per question the option IDs in key order)."""
    questions, labels, id_order = {}, {}, []
    single = len(case["questions"]) == 1
    for i, q in enumerate(case["questions"]):
        opts = q["options"]
        if q["task"] == "noul":
            by_id = {o["id"]: o for o in opts}
            ids, keys = ["true", "false"], ["true", "false"]
            criteria = {"true": by_id["true"]["description"], "false": by_id["false"]["description"]}
        elif q["task"] == "score":
            opts = sorted(opts, key=lambda o: o["value"])
            ids, keys = [o["id"] for o in opts], [str(k) for k in range(len(opts))]
            criteria = [o["description"] for o in opts]
        else:
            if len(opts) > len(KEYS):
                raise ValueError(f"{case['id']}: {len(opts)} options > {len(KEYS)} keys")
            ids, keys = [o["id"] for o in opts], KEYS[: len(opts)]
            criteria = {k: o["description"] for k, o in zip(keys, opts)}
        name = "decision" if single else f"decision_{i + 1}"
        questions[name] = {"type": q["task"], "instructions": q["instructions"], "criteria": criteria}
        target = case["targets"].get(q["id"]) if case.get("targets") else None
        if target is not None:
            labels[name] = {k: float(target[oid]) for k, oid in zip(keys, ids)}
        id_order.append(ids)
    return questions, labels, id_order


def to_example(case: dict) -> tuple[Example, list[dict], list[list[str]]]:
    """Our model's input (no targets needed), its S1MB questions, and per question the option IDs in key order."""
    questions, _, id_order = render(case)
    parsed = parse_questions(questions)
    for q in parsed:  # unused placeholder target (evaluate.run stacks one per question)
        q.target = [1 / len(q.keys)] * len(q.keys)
    return Example(id=case["id"], workflow="s1mb", state=case["state"], questions=parsed), case["questions"], id_order
