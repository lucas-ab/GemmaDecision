"""serve.py request conversion and auth; no model weights needed."""

import json

import pytest
from fastapi import HTTPException

from gemma_decision.serve import (
    DecideRequest,
    SystemOneRequest,
    _key_ok,
    _value_text,
    create_app,
    decide_example,
    systemone_case,
)


def _decide(qtype, criteria):
    q = {"type": qtype, "instructions": "Decide."}
    if criteria is not None:
        q["criteria"] = criteria
    return DecideRequest(state="hello", questions={"q": q})


@pytest.mark.parametrize("criteria", [["true", "false"], ["a"], {"true": "yes"}, {"yes": "y", "no": "n"},
                                      {"true": "y", "false": "n", "maybe": "m"}])
def test_noul_bad_criteria_is_422(criteria):
    with pytest.raises(HTTPException) as err:
        decide_example(_decide("noul", criteria))
    assert err.value.status_code == 422


def test_noul_valid_criteria():
    assert decide_example(_decide("noul", None)).questions[0].keys == ["true", "false"]
    ex = decide_example(_decide("noul", {"true": "it is", "false": "it is not"}))
    assert ex.questions[0].type == "noul"


def test_choice_list_and_score_dict_still_422():
    for qtype, crit in (("choice", ["a", "b"]), ("score", {"a": "x", "b": "y"})):
        with pytest.raises(HTTPException) as err:
            decide_example(_decide(qtype, crit))
        assert err.value.status_code == 422


def _old_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


CASES = ["plain text", "", "café ☃ \U0001f600", {"b": 1, "a": "café"}, [3, "ü", {"z": None, "y": [1.5, True]}],
         {"k": "日本語"}, 7, 1.5, None, True]


@pytest.mark.parametrize("value", CASES)
def test_shared_text_rendering_matches_old(value):
    assert _value_text(value) == _old_text(value)


def test_systemone_case_text_rendering():
    req = SystemOneRequest(state={"b": 1, "a": "x"}, decisions=[{
        "id": "d", "type": "choice", "instructions": {"k": "café"}, "system_prompt": "SYS",
        "criteria": [{"id": "a", "description": "just a string"}, {"id": "b", "description": ["x", {"y": "ü"}]}]}])
    case = systemone_case(req)
    q = case["questions"][0]
    assert case["state"] == '{"a": "x", "b": 1}'
    assert q["instructions"] == 'SYS\n{"k":"café"}'
    assert [o["description"] for o in q["options"]] == ["just a string", '["x",{"y":"ü"}]']


def test_key_ok():
    assert _key_ok(None, None) and _key_ok("anything", "")
    assert _key_ok("Bearer k", "k")
    assert not _key_ok(None, "k")
    assert not _key_ok("Bearer wrong", "k")
    assert not _key_ok("k", "k")
    assert not _key_ok("Bearer ké", "k")  # non-ASCII must not raise


def _auth_dependency(app):
    route = next(r for r in app.routes if getattr(r, "path", None) == "/v1/decide")
    return route.dependant.dependencies[0].call


def test_auth_dependency():
    auth = _auth_dependency(create_app(None, api_key="k"))
    for bad in (None, "Bearer nope", "k"):
        with pytest.raises(HTTPException) as err:
            auth(bad)
        assert err.value.status_code == 401
    assert auth("Bearer k") is None
    _auth_dependency(create_app(None))(None)  # no key configured: open


def test_http_auth():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(None, api_key="k"))
    body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    assert client.get("/health").status_code == 200
    assert client.post("/v1/decide", json=body).status_code == 401
    assert client.post("/v1/decide", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/systemone", json={}).status_code == 401


def test_truncation_detected_from_encoded_text():
    """Engine._forward flags truncation by comparing encode()'s text to the untruncated render (no 2nd tokenize)."""
    from gemma_decision.data import _render, encode

    def tok(text, return_offsets_mapping=False, add_special_tokens=True):  # 1 token per 3 characters
        offs = [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}

    def ex(state):
        e = decide_example(DecideRequest(state=state, questions={"q": {"type": "noul", "instructions": "Is it?"}}))
        return e

    short, long_ = ex("short state"), ex("word " * 400)
    fits = encode(tok, short.state, short.questions, 1000)
    cut = encode(tok, long_.state, long_.questions, 300)
    assert fits.text == _render(short.state, short.questions)[0]
    assert cut.text != _render(long_.state, long_.questions)[0]
    assert len(cut.input_ids) <= 300
