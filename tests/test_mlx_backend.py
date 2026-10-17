"""Staleness handling for the converted MLX weights (no real conversion or weights are loaded)."""

import json
import os

import pytest

pytest.importorskip("mlx.core")

from gemma_decision import mlx_backend as mb  # noqa: E402


def make_run(tmp_path, with_provenance=True):
    (tmp_path / "trainable.safetensors").write_bytes(b"weights-v1")
    (tmp_path / "config.yaml").write_text("a: 1\n")
    (tmp_path / mb.WEIGHTS).write_bytes(b"mlx")
    cfg = {"hidden_size": 8}
    if with_provenance:
        cfg["provenance"] = mb.source_provenance(tmp_path)
    (tmp_path / mb.CONFIG).write_text(json.dumps(cfg))
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    """Replace convert (needs torch + the base HF model) and the weight loading; record convert calls."""
    seen = []
    monkeypatch.setattr(mb, "convert", lambda run_dir: seen.append(run_dir))
    monkeypatch.setattr(mb.mx, "load", lambda path: {})
    monkeypatch.setattr(mb.MLXDecisionModel, "__init__", lambda self, *a, **k: None)
    return seen


def test_fresh_run_is_not_stale(tmp_path):
    assert mb.stale_reason(make_run(tmp_path)) is None


def test_load_converts_when_weights_missing(tmp_path, calls):
    run = make_run(tmp_path)
    (run / mb.WEIGHTS).unlink()
    mb.MLXDecisionModel.load(run)
    assert calls == [run]


def test_load_converts_when_config_missing(tmp_path, calls):
    run = make_run(tmp_path)
    (run / mb.CONFIG).unlink()
    assert mb.stale_reason(run) is not None


def test_load_skips_convert_when_provenance_matches(tmp_path, calls):
    run = make_run(tmp_path)
    mb.MLXDecisionModel.load(run)
    assert calls == []


def test_load_converts_when_trainable_changed(tmp_path, calls):
    run = make_run(tmp_path)
    (run / "trainable.safetensors").write_bytes(b"weights-v2-longer")
    mb.MLXDecisionModel.load(run)
    assert calls == [run]


def test_same_size_content_change_detected(tmp_path):
    run = make_run(tmp_path)
    f = run / "trainable.safetensors"
    f.write_bytes(b"weights-v2")  # same length as v1
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    assert "trainable.safetensors" in mb.stale_reason(run)


def test_touch_only_is_not_stale_and_refreshes_record(tmp_path):
    run = make_run(tmp_path)
    f = run / "config.yaml"
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    assert mb.stale_reason(run) is None
    rec = json.loads((run / mb.CONFIG).read_text())["provenance"]["sources"]["config.yaml"]
    assert rec["mtime_ns"] == f.stat().st_mtime_ns


def test_config_yaml_changed(tmp_path):
    run = make_run(tmp_path)
    (run / "config.yaml").write_text("a: 22222\n")
    assert "config.yaml" in mb.stale_reason(run)


def test_legacy_run_without_provenance_is_stale(tmp_path, calls):
    run = make_run(tmp_path, with_provenance=False)
    mb.MLXDecisionModel.load(run)
    assert calls == [run]
