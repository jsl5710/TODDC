"""External coherence metrics (own-env) + simulator mode + pairwise scoring."""
import pytest

from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
from toddc.judge import HeuristicJudge
from toddc.operators import all_operators
from toddc.passes import run_dialogue
from toddc.simulator import (
    AlignScoreMetric,
    DiscoScoreMetric,
    MetricUnavailable,
    PDDMetric,
    available,
    load_coherence_metric,
    simulate_pairwise,
    simulate_record,
)

D = parse_dialogue(SGD_1_00000_RAW)


def _record(op_id):
    recs = run_dialogue(dialogue_id=D.dialogue_id, window=D.turns,
                        operators=all_operators(), services=D.services,
                        policy="all", seed=1, judge=HeuristicJudge())
    return next(r for r in recs if r.operator == op_id)


def test_registry_lists_offline_and_external():
    assert set(available()) >= {"heuristic", "entity_grid", "llm_coherence",
                                "alignscore", "discoscore", "pdd"}
    assert available(offline_only=True) == ["entity_grid", "heuristic"]


def test_external_metric_granularity():
    assert load_coherence_metric("alignscore").granularity == "per_turn"
    assert load_coherence_metric("discoscore").granularity == "dialogue"
    assert load_coherence_metric("pdd").granularity == "dialogue"


def test_external_unavailable_without_env(monkeypatch):
    monkeypatch.delenv("TODDC_ALIGNSCORE_PYTHON", raising=False)
    m = AlignScoreMetric()
    assert m.available() is False
    with pytest.raises(MetricUnavailable):
        m.score_all([{"speaker": "USER", "utterance": "hi"}])


def test_mode_intrinsic_metric_invariant():
    rec = _record("non_sequitur")
    metric = load_coherence_metric("heuristic")
    hist = simulate_record(rec, metric, mode="history")
    imm = simulate_record(rec, metric, mode="immediate")
    assert hist.mode == "history" and imm.mode == "immediate"
    assert [t.score for t in hist.turn_scores] == [t.score for t in imm.turn_scores]


def test_mode_context_aware_metric_differs():
    rec = _record("non_sequitur")
    metric = load_coherence_metric("entity_grid")
    hist = simulate_record(rec, metric, mode="history")
    imm = simulate_record(rec, metric, mode="immediate")
    # immediate has no context -> every turn scores 0; history uses prior turns
    assert all(t.score == 0.0 for t in imm.turn_scores)
    assert any(t.score > 0.0 for t in hist.turn_scores)


def test_simulate_pairwise_dialogue_level():
    # fake dialogue-level metric: incoherence = fraction of off-domain "weather"
    class _Fake:
        name = "fake_dialogue"
        granularity = "dialogue"
        def score_dialogue(self, window):
            return sum("weather" in t["utterance"].lower() for t in window) / max(1, len(window))
    rec = _record("non_sequitur")           # injects an off-domain "weather" turn
    res = simulate_pairwise(rec, _Fake())
    assert res.perturbed_score > res.coherent_score
    assert res.correct is True
