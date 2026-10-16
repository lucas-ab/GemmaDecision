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
