"""Eval metrics + example-record schema validation."""
import json
from pathlib import Path

import pytest

from toddo.eval import (
    detection_accuracy,
    entity_grid_continuity,
    false_incoherence_rate,
    next_utterance_rank,
)
from toddo.validate import check_invariants

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "record_slot_value_mismatch.json"


def test_detection_metrics():
    assert detection_accuracy([True, False, True], [True, False, False]) == 2 / 3
    # one truly-coherent sample wrongly flagged -> 0.5 false-incoherence
    assert false_incoherence_rate([True, False], [False, False]) == 0.5


def test_entity_grid_continuity():
    coherent = [{"utterance": "I want food in San Jose"},
                {"utterance": "San Jose has great food options"}]
    broken = [{"utterance": "I want food in San Jose"},
              {"utterance": "The weather is nice today"}]
    assert entity_grid_continuity(coherent) == 1.0
    assert entity_grid_continuity(broken) == 0.0


def test_next_utterance_rank():
    assert next_utterance_rank(0.9, [0.2, 0.5]) == 1     # correct ranked top
    assert next_utterance_rank(0.3, [0.9, 0.5]) == 3     # correct ranked last


def test_example_passes_invariants():
    assert check_invariants(json.loads(EXAMPLE.read_text())) == []


def test_example_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from toddo.validate import load_schema
    jsonschema.validate(json.loads(EXAMPLE.read_text()), load_schema())
