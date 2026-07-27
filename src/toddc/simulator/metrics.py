"""Coherence metrics for the simulator.

Each metric scores ONE turn's *incoherence* in [0, 1] given the dialogue window,
without seeing the violation label. The simulator scores every turn and checks
whether the metric peaks at the perturbed turn.

- HeuristicCoherenceMetric : offline. Combines off-domain detection (a turn with
  content but no task vocabulary), contradiction markers, topic-drift markers, and
  dangling referents. Catches relevance / global / cohesion / state-contradiction
  violations; misses slot-value mismatch (needs a state cross-check) and reorder
  (needs a positional check) — which is the point: metric coverage by dimension.
- LLMCoherenceMetric : response-based. Asks a model to rate the turn's coherence
  in context (needs a live model).
"""
from __future__ import annotations

import re
from typing import Any, Optional, Protocol, runtime_checkable

# Task/domain vocabulary for the restaurant/booking SGD dialogues. A turn with
# content tokens but none of these is off-domain (a relevance/topic break).
_DOMAIN = {
    "restaurant", "restaurants", "food", "eat", "eating", "place", "cuisine",
    "american", "mexican", "italian", "city", "san", "jose", "address", "street",
    "phone", "number", "contact", "hungry", "located", "area", "book", "table",
    "suggest", "reservation", "menu", "pedro", "peter",
}
_STOP = {
    "i", "you", "the", "a", "an", "to", "of", "in", "is", "it", "for", "at", "on",
    "and", "so", "would", "like", "me", "can", "do", "have", "there", "that", "this",
    "with", "which", "want", "some", "give", "my", "be", "if", "your", "specific",
    "such", "as", "or", "type", "enjoy", "usually", "feeling", "find", "them",
    "am", "would", "good", "see", "at", "will", "how",
}
_CONTRADICTION = ("actually", "forget", "instead", "never mind", "no wait", "scratch that")
_DRIFT = ("by the way", "anyway", "speaking of", "changing the subject", "off topic")
_DANGLING = ("that one", "over there", "this one")


@runtime_checkable
class CoherenceMetric(Protocol):
    name: str
    def score_turn(self, window: list[dict[str, Any]], i: int) -> float: ...


class HeuristicCoherenceMetric:
    name = "heuristic"

    def score_turn(self, window: list[dict[str, Any]], i: int) -> float:
        low = window[i]["utterance"].lower()
        score = 0.0
        if any(m in low for m in _CONTRADICTION):
            score = max(score, 0.8)          # state contradiction
        if any(m in low for m in _DRIFT):
            score = max(score, 0.7)          # global topic drift
        if any(p in low for p in _DANGLING):
            score = max(score, 0.7)          # cohesion: dangling referent
        content = set(re.findall(r"[a-z']+", low)) - _STOP
        if len(content) >= 2 and not (content & _DOMAIN):
            score = max(score, 0.8)          # relevance: off-domain turn
        return score


class LLMCoherenceMetric:
    name = "llm_coherence"
    _PROMPT = ("Dialogue so far:\n{ctx}\n\nRate how COHERENT this next turn is in "
               "context, 0.0 (incoherent — irrelevant, contradictory, or breaks "
               "reference) to 1.0 (fully coherent). Reply with only the number.\n"
               "{speaker}: {utt}")

    def __init__(self, llm):
        self.llm = llm

    def score_turn(self, window: list[dict[str, Any]], i: int) -> float:
        ctx = "\n".join(f"{t['speaker']}: {t['utterance']}" for t in window[:i])
        raw = self.llm.generate(self._PROMPT.format(
            ctx=ctx, speaker=window[i]["speaker"], utt=window[i]["utterance"]))
        m = re.search(r"[0-1](?:\.\d+)?", raw)
        coh = float(m.group()) if m else 0.5
        return max(0.0, min(1.0, 1.0 - coh))   # incoherence = 1 - coherence
