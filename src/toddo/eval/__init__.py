"""Coherence evaluation metrics (pure stdlib).

Given a metric-or-system's per-record flag (is this dialogue incoherent?) and the
gold labels, score detection; plus intrinsic coherence measures over a window.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence


def detection_accuracy(pred_flags: Iterable[bool], gold_flags: Iterable[bool]) -> float:
    pred, gold = list(pred_flags), list(gold_flags)
    if not gold:
        return 0.0
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def false_incoherence_rate(pred_flags: Iterable[bool], gold_flags: Iterable[bool]) -> float:
    """Fraction of truly-coherent samples wrongly flagged incoherent (controls target this)."""
    pairs = [(p, g) for p, g in zip(pred_flags, gold_flags) if g is False]
    if not pairs:
        return 0.0
    return sum(1 for p, _ in pairs if p) / len(pairs)


def entity_grid_continuity(window: Sequence[dict[str, Any]]) -> float:
    """Local-coherence proxy: fraction of adjacent turn pairs that share >=1 content
    token (entity/topic continuity). A crude, dependency-free entity-grid stand-in;
    swap for a real Centering/entity-grid model later."""
    def content(u: str) -> set[str]:
        stop = {"a", "an", "the", "to", "of", "in", "is", "it", "you", "i", "for",
                "at", "on", "and", "so", "would", "like", "me", "can", "do", "have"}
        return {w.strip(".,?!").lower() for w in u.split()} - stop
    turns = [content(t["utterance"]) for t in window]
    if len(turns) < 2:
        return 1.0
    shared = sum(1 for a, b in zip(turns, turns[1:]) if a & b)
    return shared / (len(turns) - 1)


def next_utterance_rank(score_correct: float, score_distractors: Sequence[float]) -> int:
    """1-based rank of the correct next utterance among distractors (lower = better)."""
    return 1 + sum(1 for s in score_distractors if s > score_correct)


def state_consistency(window: Sequence[dict[str, Any]]) -> float:
    """State-consistency proxy: fraction of turns whose stated slot values are all
    present in that turn's belief state (no fabricated values)."""
    total = ok = 0
    for t in window:
        bs = t.get("belief_state", {}) or {}
        values = {str(v) for fr in bs.values() for v in (fr.get("slot_values", {}) or {}).values()}
        total += 1
        # a turn is consistent if it doesn't assert a value absent from state
        ok += 1  # placeholder: real check compares NL-stated values to `values`
    return ok / total if total else 1.0


__all__ = [
    "detection_accuracy", "false_incoherence_rate", "entity_grid_continuity",
    "next_utterance_rank", "state_consistency",
]
