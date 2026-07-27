"""End-to-end chain tests, runnable offline."""
from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
from toddc.judge import HeuristicJudge
from toddc.operators import all_operators, get_operator
from toddc.passes import run_chain, run_dialogue
from toddc.validate import check_invariants

DIALOGUE = parse_dialogue(SGD_1_00000_RAW)


def _rec(op_id):
    return run_chain(dialogue_id=DIALOGUE.dialogue_id, window=DIALOGUE.turns,
                     operator=get_operator(op_id), services=DIALOGUE.services,
                     judge=HeuristicJudge(), seed=42)


def test_reference_break_chain():
    rec = _rec("reference_break")
    d = rec.to_dict()
    assert set(d["passes"]) == {"analyse", "document", "apply", "confirm", "edit"}
    assert d["gold"]["label"] == "incoherent"
    assert d["gold"]["dimension"] == "cohesion"
    assert d["gold"]["should_flag"] is True
    assert d["passes"]["confirm"]["status"] == "accepted"
    assert check_invariants(d) == []


def test_control_is_coherent():
    rec = _rec("coherent_paraphrase")
    d = rec.to_dict()
    assert d["family"] == "control"
    assert d["gold"]["label"] == "coherent"
    assert d["gold"]["dimension"] is None
    assert d["gold"]["should_flag"] is False
    assert check_invariants(d) == []


def test_turn_reorder_is_structurally_confirmed():
    rec = _rec("turn_reorder")
    d = rec.to_dict()
    assert d["passes"]["confirm"]["structural_checks"]["turns_swapped"] is True
    assert d["passes"]["confirm"]["status"] == "accepted"   # confirmed structurally, not by text
    assert d["gold"]["dimension"] == "local"


def test_run_dialogue_spreads_across_turns_and_dimensions():
    recs = run_dialogue(dialogue_id=DIALOGUE.dialogue_id, window=DIALOGUE.turns,
                        operators=all_operators(), services=DIALOGUE.services,
                        policy="all", seed=1, judge=HeuristicJudge())
    dims = {r.dimension for r in recs}
    assert {"cohesion", "relevance", "global", "local", "state_consistency"} <= dims
    for r in recs:
        assert check_invariants(r.to_dict()) == []


def test_unknown_operator_raises():
    import pytest
    with pytest.raises(KeyError):
        get_operator("does_not_exist")
