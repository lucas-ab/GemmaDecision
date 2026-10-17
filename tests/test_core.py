import random

import numpy as np
import pytest
import torch
from PIL import Image

from gemma_decision.data import (
    Example, TYPES, collate, collate_mm, encode, media_reserve, parse_questions,
)
from gemma_decision.evaluate import metrics
from gemma_decision.losses import decision_loss

QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "access": "Login problems"}},
    "urgent": {"type": "noul", "instructions": "This needs a reply today."},
    "risk": {"type": "score", "instructions": "How risky?", "criteria": ["None", "Some", "High"]},
}
GOLD = {
    "route": {"type": "choice", "label": "billing", "probabilities": {"billing": 0.7, "shipping": 0.2, "access": 0.1}},
    "urgent": {"type": "noul", "label": "true", "probabilities": {"true": 0.6, "false": 0.4}},
    "risk": {"type": "score", "label": "2", "score": 1.5, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
}


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("google/embeddinggemma-2")


def span_text(tok, ids, span):
    return tok.decode(ids[span[0]:span[1]]).strip()


def test_spans_follow_shuffled_options_and_targets(processor):
    tok = processor.tokenizer
    qs = parse_questions(QUESTIONS, GOLD)
    for seed in range(5):
        e = encode(tok, "I was charged twice.", qs, rng=random.Random(seed))
        for (q_span, opt_spans), q in zip(e.spans, e.questions):
            assert span_text(tok, e.input_ids, q_span).startswith(q.name)
            for span, key, text in zip(opt_spans, q.keys, q.texts):
                assert span_text(tok, e.input_ids, span) == f"{key}: {text}"
            # target and label move with the options
            assert q.keys[q.label] == GOLD[q.name]["label"]
            assert q.target[q.label] == pytest.approx(GOLD[q.name]["probabilities"][GOLD[q.name]["label"]])
            if q.type == "score":
                assert q.keys == ["0", "1", "2"]  # ordered levels never shuffle


def test_state_truncation_keeps_questions(processor):
    tok = processor.tokenizer
    e = encode(tok, "word " * 3000, parse_questions(QUESTIONS, GOLD), max_length=300)
    assert len(e.input_ids) <= 300
    assert len(e.spans) == 3


def test_collate_flattens_questions(processor):
    tok = processor.tokenizer
    qs = parse_questions(QUESTIONS, GOLD)
    b = collate([encode(tok, "a", qs), encode(tok, "a much longer state " * 5, qs[:1])], tok.pad_token_id)
    assert b["q_opt_slots"].shape == (4, 3)
    assert b["q_opt_mask"].sum(1).tolist() == [3, 2, 3, 3]
    assert torch.allclose(b["slot_weights"].sum(-1)[b["slot_kind"] > 0], torch.ones(1))


def test_media_spans_remap_after_expansion(processor):
    tok = processor.tokenizer
    ex = Example("x", "w", "Photo: <|image|> call: <|audio|> done", parse_questions(QUESTIONS, GOLD),
                 images=[Image.new("RGB", (400, 300))], audio=[np.zeros(16000 * 2, np.float32)])
    e = encode(tok, ex.state, ex.questions, 1024, reserve=media_reserve(ex))
    b = collate_mm([e], [ex], processor)
    ids = b["input_ids"][0].tolist()
    assert len(ids) > len(e.input_ids) + 100  # placeholders expanded into soft tokens
    w = b["slot_weights"][0]
    texts = [tok.decode([ids[i] for i in (w[s] > 0).nonzero().flatten()]).strip() for s in range(w.shape[0])]
    assert texts[0].startswith("route") and texts[1] == "billing: Payments and refunds"
    assert {"pixel_values", "input_features", "input_features_mask"} <= set(b)


def _loss_batch():
    target = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.4, 0.0]])
    return {"q_opt_mask": torch.tensor([[True] * 3, [True, True, False]]), "q_target": target,
            "q_label": torch.tensor([0, 0]), "q_type": torch.tensor([TYPES["choice"], TYPES["noul"]])}


def test_hard_weight_blends_targets():
    b = _loss_batch()
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.0, float("-inf")]])
    soft, _ = decision_loss(logits, b)
    hard, _ = decision_loss(logits, b, hard=True)
    full_w, _ = decision_loss(logits, b, hard_weight=1.0)
    half, _ = decision_loss(logits, b, hard_weight=0.5)
    assert torch.isclose(hard, full_w)
    assert torch.isfinite(soft) and torch.isfinite(half)
    assert not torch.isclose(soft, hard)


def test_metrics_perfect_prediction():
    recs = [{"id": "a", "workflow": "w", "question": "q", "type": "choice", "keys": ["x", "y"],
             "probs": [0.9, 0.1], "target": [1.0, 0.0], "label": 0, "agree": True, "gold_score": None}]
    m = metrics(recs)
    assert m["acc"] == 1.0 and m["annot_agree"] == 1.0 and m["row_acc"] == 1.0


def test_long_audio_is_chunked_into_placeholders():
    from gemma_decision.data import MAX_AUDIO_S, chunk_long_audio

    wav = np.zeros(int(16000 * (2 * MAX_AUDIO_S + 5)), np.float32)
    state, audio = chunk_long_audio("Passage: <|audio|> end", [wav])
    assert state.count("<|audio|>") == 3 and len(audio) == 3
    assert sum(len(a) for a in audio) == len(wav)


def test_truncation_keeps_media_placeholders(processor):
    tok = processor.tokenizer
    state = "intro " * 400 + "<|image|>" + " tail" * 400 + "<|audio|>"
    e = encode(tok, state, parse_questions(QUESTIONS, GOLD), max_length=300)
    assert len(e.input_ids) <= 300
    assert "<|image|>" in e.text and "<|audio|>" in e.text


def test_label_descriptions_cover_canonical_inventories():
    from gemma_decision.label_descriptions import ASSISTANT_INTENTS, ASSISTANT_SCENARIOS, SLURP_LEGACY, slurp_canonical

    assert len(ASSISTANT_SCENARIOS) == 18 and len(ASSISTANT_INTENTS) == 60
    assert {i.split("_")[0] for i in ASSISTANT_INTENTS} == set(ASSISTANT_SCENARIOS)
    assert all(v in ASSISTANT_INTENTS for v in SLURP_LEGACY.values())
    assert slurp_canonical("hue_lightoff") == "iot_hue_lightoff"
    assert slurp_canonical("query") is None and slurp_canonical("music") is None


def _synth_row(questions, gold, source="banking77"):
    import json as _json

    return {"id": "r1", "workflow": source, "source": source, "license": "x", "state_text": "Head. Evidence for the decision follows.\nMessage: hi",
            "state_media": "", "state_both": "", "variants": ["text"], "images": [], "audio": [],
            "questions": _json.dumps(questions), "gold": _json.dumps(gold), "label_agreement": "{}", "n_questions": len(questions)}


def test_synth_catch_all_moves_removed_mass_to_none():
    import json as _json

    from gemma_decision.synth_build import NONE_KEY, catch_all

    q = {"route": QUESTIONS["route"]}
    g = {"route": {**GOLD["route"]}}
    seen = set()
    for seed in range(20):
        out = catch_all(_synth_row(q, g), random.Random(seed))
        qq, gg = _json.loads(out["questions"])["route"], _json.loads(out["gold"])["route"]
        probs = gg["probabilities"]
        assert set(probs) == set(qq["criteria"]) and NONE_KEY in probs
        assert abs(sum(probs.values()) - 1) < 1e-9
        if "billing" not in qq["criteria"]:  # removed: its 0.7 moved to none_of_these
            assert probs[NONE_KEY] == pytest.approx(0.7) and gg["label"] == NONE_KEY
            seen.add("removed")
        else:
            assert probs[NONE_KEY] == 0.0 and gg["label"] == "billing"
            seen.add("kept")
    assert seen == {"removed", "kept"}


def test_synth_flip_swaps_targets_and_descriptions():
    import json as _json

    from gemma_decision.synth_build import FLIP_PREFIX, flip

    q = {"urgent": {"type": "noul", "instructions": "Reply today.", "criteria": {"true": "Urgent", "false": "Can wait"}}}
    g = {"urgent": {**GOLD["urgent"]}}
    out = flip(_synth_row(q, g), random.Random(0))
    qq, gg = _json.loads(out["questions"])["urgent"], _json.loads(out["gold"])["urgent"]
    assert qq["instructions"].startswith(FLIP_PREFIX) and qq["criteria"] == {"true": "Can wait", "false": "Urgent"}
    assert gg["probabilities"] == {"true": 0.4, "false": 0.6} and gg["label"] == "false"


def test_synth_missing_uses_prior_and_drops_evidence():
    import json as _json

    from gemma_decision.synth_build import EMPTY_STATE, missing, priors

    q = {"route": QUESTIONS["route"]}
    rows = [_synth_row(q, {"route": {"probabilities": {"billing": 1.0, "shipping": 0.0, "access": 0.0}}}),
            _synth_row(q, {"route": {"probabilities": {"billing": 0.0, "shipping": 1.0, "access": 0.0}}})]
    out = missing(rows[0], priors(rows))
    probs = _json.loads(out["gold"])["route"]["probabilities"]
    assert probs["billing"] == pytest.approx(probs["shipping"]) and probs["access"] < 0.01
    assert EMPTY_STATE in out["state_text"] and "Evidence" not in out["state_text"] and "hi" not in out["state_text"]


def _s1_row(decisions, targets, state=None):
    import json as _json

    return {"case_id": "c1", "group_id": "g1", "input_hash": "h1", "split": "train", "language": "en",
            "input": {"state_json": _json.dumps(state or {"text": "hello"}), "decisions": decisions},
            "targets": targets}


def _s1_decision(did, typ, crit, instructions="Decide."):
    import json as _json

    return {"id": did, "kind": "judgment", "type": typ, "instructions_json": _json.dumps(instructions),
            "system_prompt": "", "documents": [], "scoring": None,
            "criteria": [{"id": i, "description_json": _json.dumps(d), "value": v} for i, d, v in crit]}


def test_s1mb_render_anonymizes_choice_and_orders_score():
    from gemma_decision.s1mb_format import case_from_row, render

    row = _s1_row(
        [_s1_decision("intent", "choice", [("refund", "Refund", None), ("card", "Card issue", None)]),
         _s1_decision("quality", "score", [("hi", "Good", 2.0), ("lo", "Bad", 0.0), ("mid", "Okay", 1.0)]),
         _s1_decision("urgent", "noul", [("false", "Can wait", None), ("true", "Urgent", None)])],
        [{"decision_id": "intent", "ids": ["refund", "card"], "probabilities": [0.0, 1.0]},
         {"decision_id": "quality", "ids": ["hi", "lo", "mid"], "probabilities": [0.0, 0.25, 0.75]},
         {"decision_id": "urgent", "ids": ["false", "true"], "probabilities": [0.3, 0.7]}])
    case = case_from_row(row)
    assert case["state"] == '{"text": "hello"}'
    questions, labels, ids = render(case)
    assert list(questions) == ["decision_1", "decision_2", "decision_3"]  # no source question IDs in the text
    assert questions["decision_1"]["criteria"] == {"A": "Refund", "B": "Card issue"}
    assert labels["decision_1"] == {"A": 0.0, "B": 1.0} and ids[0] == ["refund", "card"]
    assert questions["decision_2"]["criteria"] == ["Bad", "Okay", "Good"]  # sorted by value
    assert labels["decision_2"] == {"0": 0.25, "1": 0.75, "2": 0.0} and ids[1] == ["lo", "mid", "hi"]
    assert questions["decision_3"]["criteria"] == {"true": "Urgent", "false": "Can wait"}
    assert labels["decision_3"] == {"true": 0.7, "false": 0.3}
    assert case_from_row(row, "noul")["questions"][0]["id"] == "urgent"


def test_serve_decide_request_to_example_and_back():
    from gemma_decision.serve import DecideRequest, decide_example, decide_output

    req = DecideRequest(state={"b": 1, "a": "x"}, questions={
        "route": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Money", "tech": "Bugs"}},
        "urgent": {"type": "noul", "instructions": "Urgent."},
        "level": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}},
        images=["data:image/png;base64,iVBORw0KGgo="])
    ex = decide_example(req)
    assert ex.state == '{"a": "x", "b": 1}\nImage: <|image|>' and ex.images == [b"\x89PNG\r\n\x1a\n"]
    assert [q.keys for q in ex.questions] == [["billing", "tech"], ["true", "false"], ["0", "1", "2"]]
    recs = [{"question": "route", "type": "choice", "keys": ["billing", "tech"], "probs": [0.8, 0.2]},
            {"question": "urgent", "type": "noul", "keys": ["true", "false"], "probs": [0.3, 0.7]},
            {"question": "level", "type": "score", "keys": ["0", "1", "2"], "probs": [0.0, 0.5, 0.5]}]
    out = decide_output(recs)
    assert out["route"]["label"] == "billing" and out["urgent"]["probability_true"] == 0.3
    assert out["level"]["score"] == 1.5


def test_serve_systemone_maps_back_to_option_ids():
    from gemma_decision.s1mb_format import split_case, to_example
    from gemma_decision.serve import SystemOneRequest, systemone_case, systemone_output

    req = SystemOneRequest(state="hi", decisions=[
        {"id": "q", "type": "score", "instructions": "Rate.",
         "criteria": [{"id": "hi", "description": "good", "value": 10}, {"id": "lo", "description": "bad", "value": 0}]}])
    (part,) = [to_example(p) for p in split_case(systemone_case(req))]
    ex, qs, ids = part
    assert ex.questions[0].texts == ["bad", "good"] and ids == [["lo", "hi"]]
    out = systemone_output([(qs, ids)], [[{"probs": [0.25, 0.75]}]])
    assert out["q"]["probabilities"] == {"lo": 0.25, "hi": 0.75} and out["q"]["selected"] == "hi"
    assert out["q"]["score"] == 7.5 and out["q"]["normalized_score"] == 0.75


def test_serve_pad_batch_buckets_shapes():
    from transformers import AutoTokenizer

    from gemma_decision.data import collate, encode, parse_questions
    from gemma_decision.serve import pad_batch

    tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
    qs = parse_questions({"a": {"type": "choice", "instructions": "x", "criteria": {"p": "1", "q": "2", "r": "3"}}})
    b = collate([encode(tok, "hello", qs, 512)], tok.pad_token_id)
    b.pop("q_meta")
    p = pad_batch(b, tok.pad_token_id)
    assert p["input_ids"].shape[1] == 64 and p["attention_mask"][0, b["input_ids"].shape[1]:].sum() == 0
    assert p["slot_weights"].shape[1:] == (8, 64) and p["slot_kind"].shape[1] == 8
    assert p["q_opt_mask"].shape[1] == 4 and not p["q_opt_mask"][0, 3]



def test_mlx_head_matches_torch_head():
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from gemma_decision.data import collate, encode, parse_questions
    from gemma_decision.mlx_backend import ClefHeadMLX
    from gemma_decision.model import ClefHead
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
    qs = parse_questions({**QUESTIONS, "urgent": {"type": "noul", "instructions": "Urgent."}})
    b = collate([encode(tok, "hello there", qs, 256), encode(tok, "a longer case " * 20, qs, 256)], tok.pad_token_id)
    b.pop("q_meta")
    head = ClefHead().eval()
    head.log_temp.copy_(torch.tensor([0.1, -0.2, 0.3]))
    tokens = torch.randn(*b["input_ids"].shape, 768)
    with torch.no_grad():
        ref = head(tokens, b["attention_mask"], b).numpy()
    mh = ClefHeadMLX({k: mx.array(v.numpy()) for k, v in head.state_dict().items()})
    out = mh(mx.array(tokens.numpy()), mx.array(b["attention_mask"].numpy()),
             {k: mx.array(v.numpy()) for k, v in b.items()})
    out = np.array(out)
    finite = np.isfinite(ref)
    assert (finite == np.isfinite(out)).all()
    assert np.abs(out[finite] - ref[finite]).max() < 1e-4
