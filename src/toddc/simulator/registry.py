"""Load any coherence metric of choice by name.

Offline (no extra env): heuristic, entity_grid, llm_coherence (llm needs a model
client). External (own env): alignscore, discoscore, pdd — see external.py.
"""
from __future__ import annotations

from toddc.simulator.external import (
    AlignScoreMetric,
    DiscoScoreMetric,
    PDDMetric,
)
from toddc.simulator.metrics import (
    EntityGridMetric,
    HeuristicCoherenceMetric,
    LLMCoherenceMetric,
)

_BUILDERS = {
    "heuristic": HeuristicCoherenceMetric,      # offline, intrinsic
    "entity_grid": EntityGridMetric,            # offline, context-aware (local)
    "llm_coherence": LLMCoherenceMetric,        # needs a model client
    "alignscore": AlignScoreMetric,             # external env (per-turn)
    "discoscore": DiscoScoreMetric,             # external env (dialogue-level)
    "pdd": PDDMetric,                           # external env (dialogue-level)
}

_OFFLINE = {"heuristic", "entity_grid"}


def available(*, offline_only: bool = False) -> list[str]:
    names = _OFFLINE if offline_only else set(_BUILDERS)
    return sorted(names)


def load_coherence_metric(name: str, **kwargs):
    if name not in _BUILDERS:
        raise KeyError(f"Unknown coherence metric {name!r}. Available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](**kwargs)
