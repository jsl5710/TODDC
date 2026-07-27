"""The chain-of-passes orchestrator: analyse → document → apply → confirm → edit.

Produces one Record per (window, operator, target-turn). Deterministic passes own
the gold label; apply/confirm may use an LLM + judge.
"""
from __future__ import annotations

from typing import Optional

from toddc.coherence import derive_dimension, should_flag
from toddc.judge import HeuristicJudge, Judge, NullJudge
from toddc.operators.base import Operator
from toddc.positioning import Site, enumerate_sites, position_of, select_sites
from toddc.runners.base import LLMClient
from toddc.schema import (
    ConfirmPass,
    DialogueTurn,
    EditPass,
    Gold,
    Position,
    Provenance,
    Record,
)


def run_chain(
    *,
    dialogue_id: str,
    window: list[DialogueTurn],
    operator: Operator,
    services: list[str],
    position: Optional[Position] = None,
    llm: Optional[LLMClient] = None,
    judge: Optional[Judge | NullJudge | HeuristicJudge] = None,
    seed: int = 0,
    sgd_version: str = "GEM/schema_guided_dialog",
) -> Optional[Record]:
    judge = judge or NullJudge()

    analysis = operator.analyse(window)          # Pass 1
    if not analysis.modifiable:
        return None
    idx = analysis.target_turn_idx
    if position is None:
        position = position_of(idx or 0, len(window))

    document = operator.document(window, analysis)   # Pass 2 (owns the label)
    apply = operator.apply(window, document, llm)    # Pass 3

    # Pass 4 — confirm
    structural = _structural_checks(window, document, apply)
    verdict = judge.validate(document.change_from, apply.modified_utterance,
                             document.dimension, document.gold_label)
    status = _decide_status(operator, structural, verdict)
    confirm = ConfirmPass(change_applied=all(structural.values()), status=status,
                          structural_checks=structural, judge_verdict=verdict)

    # Pass 5 — edit (finalize / repair)
    edit = _edit(document, apply, confirm)

    gold = Gold(label=document.gold_label, dimension=derive_dimension(operator.id),
                violation_type=document.violation_type,
                severity=document.expected_severity, should_flag=should_flag(document.gold_label))

    return Record(
        record_id=f"{dialogue_id}:{idx}:{operator.id}:{seed}",
        dialogue_id=dialogue_id, services=services, target_turn_idx=idx or 0,
        family=operator.family, operator=operator.id,
        dimension=derive_dimension(operator.id), position=position,
        source_window=window,
        passes_analyse=analysis, passes_document=document, passes_apply=apply,
        passes_confirm=confirm, passes_edit=edit, gold=gold,
        provenance=Provenance(sgd_version=sgd_version, seed=seed,
                              generator_model=getattr(llm, "model_id", None),
                              judge_model=getattr(judge, "judge_model", None)),
    )


def run_dialogue(*, dialogue_id: str, window: list[DialogueTurn], operators: list[Operator],
                 services: list[str], policy: str = "all", k: Optional[int] = None,
                 llm: Optional[LLMClient] = None, judge=None, seed: int = 0) -> list[Record]:
    """Generate multiple samples from one dialogue, each violating a different turn."""
    sites: list[Site] = select_sites(enumerate_sites(window, operators),
                                      policy=policy, k=k, seed=seed)
    out = []
    for s in sites:
        rec = run_chain(dialogue_id=dialogue_id, window=window, operator=s.operator,
                        services=services, position=s.position, llm=llm, judge=judge, seed=seed)
        if rec is not None:
            out.append(rec)
    return out


def _structural_checks(window, document, apply) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if document.operator == "coherent_paraphrase":
        checks["is_control"] = True
    elif "swap_with" in document.edit:
        i, j = document.edit["turn_idx"], document.edit["swap_with"]
        checks["turns_swapped"] = (apply.new_window[i]["utterance"] == window[j].utterance)
    else:
        checks["utterance_changed"] = apply.modified_utterance != document.change_from
    return checks or {"noop": True}


def _decide_status(operator, structural, verdict) -> str:
    if not all(structural.values()):
        return "rejected"
    if verdict.get("_offline") or verdict.get("_parse_error"):
        return "needs_review"
    if operator.family == "control":
        return "accepted" if verdict.get("fidelity") == "pass" else "needs_review"
    # Structural-only violations (e.g. turn_reorder) leave the target text
    # unchanged; they're confirmed by the structural check, not a text-diff judge.
    if "turns_swapped" in structural:
        return "accepted" if structural["turns_swapped"] else "rejected"
    if verdict.get("fidelity") == "pass" and verdict.get("violation_present"):
        return "accepted"
    if verdict.get("fidelity") == "fail":
        return "rejected"
    return "needs_review"


def _edit(document, apply, confirm) -> EditPass:
    if confirm.change_applied:
        return EditPass(mode="copy", final_utterance=apply.modified_utterance,
                        final_window=apply.new_window, final_status="finalized")
    # deterministic repair: re-apply the documented surface change
    changes = ["repaired: re-applied documented edit"]
    return EditPass(mode="repair", final_utterance=document.change_to,
                    final_window=apply.new_window, final_status="finalized", changes=changes)
