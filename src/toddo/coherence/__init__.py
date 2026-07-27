"""Coherence taxonomy: operator -> dimension, severity, and gold label.

The gold label is derived by rule (never by an LLM) so labels stay consistent.
Every violation operator yields `incoherent`; the control yields `coherent`.
"""
from __future__ import annotations

from toddo.schema import CoherenceLabel, Dimension, Severity

# Operator -> coherence dimension it targets.
OPERATOR_DIMENSION: dict[str, Dimension | None] = {
    "reference_break": "cohesion",
    "contradiction": "state_consistency",
    "slot_value_mismatch": "state_consistency",
    "non_sequitur": "relevance",
    "off_topic_insertion": "global",
    "turn_reorder": "local",
    "coherent_paraphrase": None,   # control
}

# Operator -> label. Control preserves coherence; everything else breaks it.
OPERATOR_LABEL: dict[str, CoherenceLabel] = {
    op: ("coherent" if op == "coherent_paraphrase" else "incoherent")
    for op in OPERATOR_DIMENSION
}

# Operators whose severity is fixed; others default to "minor".
OPERATOR_SEVERITY: dict[str, Severity] = {
    "coherent_paraphrase": "none",
    "contradiction": "major",
    "slot_value_mismatch": "major",
}


def derive_dimension(operator: str) -> Dimension | None:
    if operator not in OPERATOR_DIMENSION:
        raise KeyError(f"Unknown operator: {operator!r}")
    return OPERATOR_DIMENSION[operator]


def derive_label(operator: str) -> CoherenceLabel:
    return OPERATOR_LABEL[operator]


def derive_severity(operator: str) -> Severity:
    return OPERATOR_SEVERITY.get(operator, "minor")


def should_flag(label: CoherenceLabel) -> bool:
    return label == "incoherent"


__all__ = [
    "OPERATOR_DIMENSION", "OPERATOR_LABEL", "OPERATOR_SEVERITY",
    "derive_dimension", "derive_label", "derive_severity", "should_flag",
]
