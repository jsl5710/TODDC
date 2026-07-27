"""TODDC Simulator: replay a sample and test coherence-metric localization (offline)."""
from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
from toddc.judge import HeuristicJudge
from toddc.operators import all_operators
from toddc.passes import run_dialogue
from toddc.simulator import HeuristicCoherenceMetric, simulate_record

D = parse_dialogue(SGD_1_00000_RAW)


def _record(op_id):
    recs = run_dialogue(dialogue_id=D.dialogue_id, window=D.turns,
                        operators=all_operators(), services=D.services,
                        policy="all", seed=1, judge=HeuristicJudge())
    return next(r for r in recs if r.operator == op_id)


def test_non_sequitur_localized_by_off_domain():
    res = simulate_record(_record("non_sequitur"), HeuristicCoherenceMetric())
    assert res.dimension == "relevance"
    assert res.predicted_idx == res.target_idx
    assert res.localized is True


def test_reference_break_localized_by_dangling_referent():
    res = simulate_record(_record("reference_break"), HeuristicCoherenceMetric())
    assert res.dimension == "cohesion"
    assert res.localized is True


def test_contradiction_localized_by_markers():
    res = simulate_record(_record("contradiction"), HeuristicCoherenceMetric())
    assert res.localized is True
    assert res.turn_scores[res.target_idx].score >= 0.5


def test_control_no_false_alarm():
    res = simulate_record(_record("coherent_paraphrase"), HeuristicCoherenceMetric())
    assert res.should_flag is False
    assert res.false_alarm is False
    assert res.localized is True   # no turn flagged == correct for a control


def test_slot_value_mismatch_is_missed_offline():
    # honest coverage limit: text metric can't catch a value not in the state
    res = simulate_record(_record("slot_value_mismatch"), HeuristicCoherenceMetric())
    assert res.localized is False


def test_scores_bounded_and_per_turn():
    res = simulate_record(_record("off_topic_insertion"), HeuristicCoherenceMetric())
    assert len(res.turn_scores) == len(D.turns)
    assert all(0.0 <= ts.score <= 1.0 for ts in res.turn_scores)
