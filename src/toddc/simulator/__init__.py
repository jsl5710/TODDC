"""TODDC Simulator — replay a (coherent, perturbed) sample turn-by-turn and test
whether a coherence metric flags the violation at the perturbed turn.

Two modes (immediate | history) and pluggable metrics: offline (heuristic,
entity_grid, llm_coherence) and external own-env metrics (alignscore, discoscore,
pdd — see external.py)."""
from toddc.simulator.external import (
    AlignScoreMetric,
    DiscoScoreMetric,
    MetricUnavailable,
    PDDMetric,
)
from toddc.simulator.metrics import (
    CoherenceMetric,
    EntityGridMetric,
    HeuristicCoherenceMetric,
    LLMCoherenceMetric,
)
from toddc.simulator.registry import available, load_coherence_metric
from toddc.simulator.simulator import (
    PairResult,
    SimResult,
    TurnScore,
    simulate_pairwise,
    simulate_record,
)

__all__ = [
    "CoherenceMetric", "HeuristicCoherenceMetric", "EntityGridMetric", "LLMCoherenceMetric",
    "AlignScoreMetric", "DiscoScoreMetric", "PDDMetric", "MetricUnavailable",
    "load_coherence_metric", "available",
    "SimResult", "PairResult", "TurnScore", "simulate_record", "simulate_pairwise",
]
