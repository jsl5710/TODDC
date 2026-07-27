"""TODDC Simulator.

Takes a single TODDC sample (a Record with a coherence violation injected at a
target turn), replays the perturbed dialogue turn-by-turn, scores each turn's
incoherence with a coherence metric, and reports whether the metric flags the
violation at the perturbed turn — and, for a coherent control, whether it avoids
a false alarm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from toddc.simulator.metrics import CoherenceMetric

_THRESHOLD = 0.5


@dataclass
class TurnScore:
    ordinal: int
    speaker: str
    utterance: str
    score: float
    is_target: bool


@dataclass
class SimResult:
    metric: str
    operator: str
    dimension: Optional[str]
    target_idx: int
    should_flag: bool                  # True for incoherent samples
    turn_scores: list[TurnScore] = field(default_factory=list)
    predicted_idx: Optional[int] = None
    rank_of_target: Optional[int] = None
    localized: bool = False            # peaked at the target (or, for a control, no flag)
    false_alarm: bool = False          # flagged a coherent control


def simulate_record(record, metric: CoherenceMetric, *, threshold: float = _THRESHOLD) -> SimResult:
    window = record.passes_edit.final_window                 # the perturbed dialogue
    target = record.target_turn_idx
    should_flag = record.gold.should_flag

    scores: list[TurnScore] = []
    for i, t in enumerate(window):
        s = metric.score_turn(window, i)
        scores.append(TurnScore(i, t["speaker"], t["utterance"], round(s, 4), i == target))

    flagged = [ts.ordinal for ts in scores if ts.score >= threshold]
    ordered = sorted(scores, key=lambda x: x.score, reverse=True)
    predicted = ordered[0].ordinal if ordered and ordered[0].score >= threshold else None
    rank = next((r for r, ts in enumerate(ordered, 1) if ts.is_target), None)

    if should_flag:
        localized = predicted == target and scores[target].score >= threshold
        false_alarm = False
    else:                              # control: correct iff nothing was flagged
        localized = flagged == []
        false_alarm = flagged != []

    return SimResult(metric=metric.name, operator=record.operator,
                     dimension=record.dimension, target_idx=target,
                     should_flag=should_flag, turn_scores=scores,
                     predicted_idx=predicted, rank_of_target=rank,
                     localized=localized, false_alarm=false_alarm)
