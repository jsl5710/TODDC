"""TODDC Simulator — replay a (coherent, perturbed) sample turn-by-turn and test
whether a coherence metric flags the violation at the perturbed turn."""
from toddc.simulator.metrics import (
    CoherenceMetric,
    HeuristicCoherenceMetric,
    LLMCoherenceMetric,
)
from toddc.simulator.simulator import SimResult, TurnScore, simulate_record

__all__ = [
    "CoherenceMetric", "HeuristicCoherenceMetric", "LLMCoherenceMetric",
    "SimResult", "TurnScore", "simulate_record",
]
