"""Operator registry (v1)."""
from toddc.operators.base import Operator
from toddc.operators.coherence_ops import (
    CoherentParaphrase,
    Contradiction,
    NonSequitur,
    OffTopicInsertion,
    ReferenceBreak,
    SlotValueMismatch,
    TurnReorder,
)

_OPERATORS: list[type[Operator]] = [
    ReferenceBreak, Contradiction, SlotValueMismatch,   # cohesion / state_consistency
    NonSequitur, OffTopicInsertion,                     # relevance / global
    TurnReorder,                                        # local
    CoherentParaphrase,                                 # control
]

REGISTRY: dict[str, type[Operator]] = {op.id: op for op in _OPERATORS}


def get_operator(op_id: str) -> Operator:
    if op_id not in REGISTRY:
        raise KeyError(f"Unknown operator {op_id!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[op_id]()


def all_operators() -> list[Operator]:
    return [cls() for cls in _OPERATORS]


__all__ = ["Operator", "REGISTRY", "get_operator", "all_operators"]
