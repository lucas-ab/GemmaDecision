"""Unit tests for scripts/s1mb_bench.py (metrics, summarize) and scripts/s1mb_leaderboard.py (dedupe)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import s1mb_bench  # noqa: E402
import s1mb_leaderboard  # noqa: E402


def _cases(task):
    if task == "noul":
        def q(i, truth):
            return {"id": f"q{i}", "options": [{"id": "true"}, {"id": "false"}]}, \
                {"true": float(truth), "false": float(not truth)}
    elif task == "choice":
        def q(i, truth):
            opts = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
            return {"id": f"q{i}", "options": opts}, {k: float(k == "ABC"[truth]) for k in "ABC"}
    else:
        def q(i, truth):
            opts = [{"id": "lo", "value": 0}, {"id": "mid", "value": 1}, {"id": "hi", "value": 2}]
            return {"id": f"q{i}", "options": opts}, {k: float(k == ("lo", "mid", "hi")[truth]) for k in
                                                       ("lo", "mid", "hi")}
    cases = []
    for c in range(4):
        question, target = q(0, c % 3 if task != "noul" else c % 2)
        cases.append({"id": f"c{c}", "questions": [question], "targets": {"q0": target}})
    return cases


def _perfect(cases):
    return {(c["id"], q["id"]): dict(c["targets"][q["id"]]) for c in cases for q in c["questions"]}


@pytest.mark.parametrize("task", ["noul", "choice", "score"])
def test_metrics_zero_predictions(task):
    out = s1mb_bench.metrics(_cases(task), {}, task)
    assert out["answered"] == 0 and out["decisions"] == 4
    assert out["baseline_adjusted_score"] is None
    assert out["baseline_adjusted_skill"] is None


@pytest.mark.parametrize("task", ["noul", "choice", "score"])
def test_metrics_no_entries_at_all(task):
    out = s1mb_bench.metrics([], {}, task)
    assert out["answered"] == 0 and out["baseline_adjusted_score"] is None


@pytest.mark.parametrize("task", ["noul", "choice", "score"])
def test_metrics_perfect_predictions(task):
    cases = _cases(task)
    out = s1mb_bench.metrics(cases, _perfect(cases), task)
    assert out["answered"] == out["decisions"] == 4
    assert out["baseline_adjusted_score"] == pytest.approx(1.0)


def test_summarize_with_none_score_is_incomplete():
    benches = [{"id": "s1mb-generalization-a", "task": "noul"}, {"id": "b", "task": "choice"},
               {"id": "c", "task": "score"}]
    ok = {"decisions": 4, "answered": 4, "baseline_adjusted_score": 0.5}
    dead = s1mb_bench.metrics(_cases("choice"), {}, "choice")
    s = s1mb_bench.summarize(benches, {"s1mb-generalization-a": ok, "b": dead, "c": ok})
    assert s["complete"] is False
    assert s["tasks"]["choice"] is None and s["task_avg"] is None
    assert s["per_benchmark"]["b"] is None
    assert s["tasks"]["noul"] == pytest.approx(50.0)


def _payload(results):
    return {
        "benchmarks": [{"id": "b0", "task": "noul"}, {"id": "b1", "task": "noul"}],
        "models": [{"run_id": "r0", "model": {"id": "m0"}}, {"run_id": "r1", "model": {"id": "m1"}}],
        "results": results,
    }


def _res(model, bench, score, succeeded=10, status="complete", provenance="measured"):
    return {"model": model, "benchmark": bench, "provenance": provenance, "status": status,
            "counts": {"succeeded": succeeded}, "metrics": {"baseline_adjusted_score": score}}


def test_collect_scores_no_duplicates():
    p = _payload([_res(0, 0, 0.1), _res(0, 1, 0.2), _res(1, 0, 0.3)])
    scores, dups = s1mb_leaderboard.collect_scores(p, ["b0", "b1"])
    assert dups == 0
    assert scores[0] == {"b0": 0.1, "b1": 0.2} and scores[1] == {"b0": 0.3}


def test_collect_scores_duplicates_prefer_most_succeeded_then_later(capsys):
    p = _payload([_res(0, 0, 0.1, succeeded=5), _res(0, 0, 0.9, succeeded=10), _res(0, 0, 0.2, succeeded=3),
                  _res(0, 1, 0.4, succeeded=7), _res(0, 1, 0.6, succeeded=7),
                  _res(1, 0, 0.5, status="partial"), _res(1, 0, 0.7, provenance="reported")])
    scores, dups = s1mb_leaderboard.collect_scores(p, ["b0", "b1"])
    assert dups == 3
    assert scores[0] == {"b0": 0.9, "b1": 0.6}
    assert 1 not in scores
    models = s1mb_leaderboard.build_models(p, ["b0", "b1"])
    assert list(models) == ["m0 [r0]"]
    assert "3 duplicate" in capsys.readouterr().err
