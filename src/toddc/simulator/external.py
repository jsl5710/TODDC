"""External coherence metrics — AlignScore, DiscoScore, PDD.

These are model-heavy metrics with **conflicting dependencies**, so each runs in
its OWN environment (a dedicated venv/conda env — see `env/<name>/`). This module
is a thin integration layer: it serializes the dialogue to JSON, invokes a runner
script *inside that metric's environment* via subprocess, and reads scores back.
Nothing heavy is imported into the main TODDC env.

Papers:
- AlignScore — "AlignScore: Evaluating Factual Consistency with a Unified
  Alignment Function" (Zha et al., ACL 2023).
- DiscoScore — "DiscoScore: Evaluating Text Generation with BERT and Discourse
  Coherence" (Zhao et al., EACL 2023).
- PDD — "Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for
  Positional Discourse Coherence" (2024).

Backend contract (per metric, in its own env):
    stdin  : {"granularity": "per_turn"|"dialogue", "items": [...]}
    stdout : {"scores": [float, ...]}          # incoherence in [0, 1]
The interpreter for metric NAME is taken from env var  TODDC_<NAME>_PYTHON
(pointing at that env's `python`); the runner script is `env/<name>/runner.py`.
If the interpreter/runner is missing, `MetricUnavailable` is raised with setup
instructions — the metric is never silently faked.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

_ENV_DIR = Path(__file__).resolve().parents[3] / "env"


class MetricUnavailable(RuntimeError):
    """Raised when an external metric's environment is not configured."""


class ExternalCoherenceMetric:
    name: str = "external"
    granularity: str = "per_turn"     # "per_turn" or "dialogue"

    def __init__(self, *, python: str | None = None, timeout: float = 120.0):
        env_var = f"TODDC_{self.name.upper()}_PYTHON"
        self.python = python or os.environ.get(env_var)
        self.runner = _ENV_DIR / self.name / "runner.py"
        self.timeout = timeout
        self._env_var = env_var

    def available(self) -> bool:
        return bool(self.python) and Path(self.python).exists() and self.runner.exists()

    def _require(self) -> None:
        if not self.python or not Path(self.python).exists():
            raise MetricUnavailable(
                f"{self.name}: set {self._env_var} to the python of its env. "
                f"Setup: env/{self.name}/README.md")
        if not self.runner.exists():
            raise MetricUnavailable(f"{self.name}: missing runner {self.runner}")

    def _run(self, items: list[dict[str, Any]]) -> list[float]:
        self._require()
        payload = json.dumps({"granularity": self.granularity, "items": items})
        try:
            out = subprocess.run([self.python, str(self.runner)], input=payload,
                                 capture_output=True, text=True, timeout=self.timeout, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise MetricUnavailable(f"{self.name} runner failed: {e}") from e
        return json.loads(out.stdout)["scores"]

    # ---- per-turn interface (AlignScore) ---------------------------------
    def score_all(self, window: list[dict[str, Any]]) -> list[float]:
        # each turn scored against its preceding context (the "claim vs context")
        items = [{"context": " ".join(t["utterance"] for t in window[:i]),
                  "claim": window[i]["utterance"]} for i in range(len(window))]
        return self._run(items)

    # ---- dialogue-level interface (DiscoScore, PDD) ----------------------
    def score_dialogue(self, window: list[dict[str, Any]]) -> float:
        text = "\n".join(f"{t['speaker']}: {t['utterance']}" for t in window)
        return self._run([{"text": text}])[0]


class AlignScoreMetric(ExternalCoherenceMetric):
    """Per-turn factual-consistency: incoherence = 1 - alignment(context, claim).
    Fits relevance / state-consistency violations."""
    name = "alignscore"
    granularity = "per_turn"


class DiscoScoreMetric(ExternalCoherenceMetric):
    """Dialogue-level discourse coherence (BERT + discourse). Used pairwise:
    coherent original vs perturbed should differ."""
    name = "discoscore"
    granularity = "dialogue"


class PDDMetric(ExternalCoherenceMetric):
    """Dialogue-level positional discourse coherence. Used pairwise."""
    name = "pdd"
    granularity = "dialogue"
