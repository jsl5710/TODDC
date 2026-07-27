"""LLM-as-judge (+ deterministic heuristic + offline null) for Pass 4.

Validates whether the intended coherence violation actually landed. NullJudge
never approves (offline default -> needs_review); HeuristicJudge is a rule-based
gate for bootstrapping a seed set without an LLM; Judge wraps an LLMClient.
"""
from __future__ import annotations

from typing import Any

from toddc.prompts import render_judge_prompt
from toddc.runners.base import LLMClient


class NullJudge:
    judge_model = "null-judge"

    def validate(self, change_from: str, change_to: str, dimension, gold_label: str) -> dict[str, Any]:
        return {"fidelity": "uncertain", "violation_present": False, "naturalness": 0.0,
                "_offline": True}


class HeuristicJudge:
    """Deterministic gate. Confirms what's checkable by rule: control edits keep
    meaning (pass); violation edits must actually change the surface. Naturalness
    is a fixed, clearly-labeled heuristic — not a model score."""
    judge_model = "heuristic-judge"

    def validate(self, change_from: str, change_to: str, dimension, gold_label: str) -> dict[str, Any]:
        is_control = gold_label == "coherent"
        changed = change_to.strip() != change_from.strip()
        fidelity = "pass" if (is_control or changed) else "fail"
        violation_present = is_control or changed
        return {"fidelity": fidelity, "violation_present": violation_present,
                "naturalness": 0.7 if change_to.strip() else 0.0, "_heuristic": True}


class Judge:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.judge_model = llm.model_id

    def validate(self, change_from: str, change_to: str, dimension, gold_label: str) -> dict[str, Any]:
        import json
        raw = self.llm.generate(render_judge_prompt(change_from, change_to, dimension, gold_label))
        try:
            d = json.loads(raw)
            return {"fidelity": d.get("fidelity", "uncertain"),
                    "violation_present": bool(d.get("violation_present", False)),
                    "naturalness": float(d.get("naturalness", 0.0))}
        except (ValueError, TypeError):
            return {"fidelity": "uncertain", "violation_present": False,
                    "naturalness": 0.0, "_parse_error": True}
