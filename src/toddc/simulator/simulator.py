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
    mode: str = "history"             # "immediate" (turn alone) | "history" (turn + prior context)


@dataclass
class PairResult:
    """Dialogue-level (DiscoScore/PDD): compare coherent original vs perturbed."""
    metric: str
    operator: str
    dimension: Optional[str]
    coherent_score: float             # incoherence of the original window
    perturbed_score: float            # incoherence of the perturbed window
    correct: bool                     # perturbed judged MORE incoherent than original


def _turn_scores_for_mode(window, metric, mode: str) -> list[float]:
    if mode == "history":
        return metric.score_all(window)                 # each turn sees prior context
    # immediate: score each turn alone (no context)
    return [metric.score_all([window[i]])[0] for i in range(len(window))]


def simulate_record(record, metric: CoherenceMetric, *, mode: str = "history",
                    threshold: float = _THRESHOLD) -> SimResult:
    """Per-turn coherence localization.

    mode: "history" = each turn scored with its preceding context; "immediate" =
    each turn scored in isolation. (Intrinsic metrics score the same either way;
    context-aware ones — entity_grid, alignscore, llm — differ.)
    """
    if mode not in ("history", "immediate"):
        raise ValueError(f"mode must be 'history' or 'immediate', got {mode!r}")
    window = record.passes_edit.final_window                 # the perturbed dialogue
    target = record.target_turn_idx
    should_flag = record.gold.should_flag

    raw = _turn_scores_for_mode(window, metric, mode)
    scores = [TurnScore(i, window[i]["speaker"], window[i]["utterance"],
                        round(raw[i], 4), i == target) for i in range(len(window))]

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
                     localized=localized, false_alarm=false_alarm, mode=mode)


def simulate_pairwise(record, metric) -> PairResult:
    """Dialogue-level scoring (DiscoScore / PDD): score the original coherent
    window vs the perturbed window; a good metric rates the perturbed as more
    incoherent (unless it's a coherent control, where they should be close)."""
    coherent = [{"speaker": t.speaker, "utterance": t.utterance} for t in record.source_window]
    perturbed = record.passes_edit.final_window
    cs = metric.score_dialogue(coherent)
    ps = metric.score_dialogue(perturbed)
    correct = ps > cs if record.gold.should_flag else abs(ps - cs) < 1e-6
    return PairResult(metric=metric.name, operator=record.operator,
                      dimension=record.dimension, coherent_score=round(cs, 4),
                      perturbed_score=round(ps, 4), correct=correct)
