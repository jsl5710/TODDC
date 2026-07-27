"""Coherence-violation operators (v1). Each owns its gold label via the
coherence taxonomy; the LLM only rewords its output.
"""
from __future__ import annotations

from typing import Optional

from toddc.coherence import derive_dimension, derive_label, derive_severity
from toddc.operators.base import Operator
from toddc.runners.base import LLMClient
from toddc.schema import AnalysePass, ApplyPass, DialogueTurn, DocumentPass

_ALT_ENTITY = {"San Jose": "Oakland", "American": "Ethiopian", "Saint Peter": "Union Square"}


class _WindowOp(Operator):
    """Shared document/apply for single-turn edits."""

    def _doc(self, window, analysis, change_to, edit) -> DocumentPass:
        idx = analysis.target_turn_idx
        return DocumentPass(
            operator=self.id, change_from=window[idx].utterance, change_to=change_to,
            edit=edit, dimension=derive_dimension(self.id),
            expected_severity=derive_severity(self.id), gold_label=derive_label(self.id),
            violation_type=self.id if derive_label(self.id) == "incoherent" else None,
        )

    def apply(self, window, spec, llm: Optional[LLMClient]) -> ApplyPass:
        new = self._clone(window)
        idx = spec.edit.get("turn_idx", 0)
        if "swap_with" in spec.edit:  # reorder
            j = spec.edit["swap_with"]
            new[idx], new[j] = new[j], new[idx]
        else:
            new[idx].utterance = spec.change_to
        method, variants = self._maybe_paraphrase(spec.change_to, llm)
        return ApplyPass(modified_utterance=spec.change_to, method=method,
                         paraphrase_variants=variants, new_window=self._window_dicts(new))


class CoherentParaphrase(_WindowOp):
    id = "coherent_paraphrase"
    family = "control"

    def is_applicable(self, window):
        return any(t.speaker == "USER" for t in window)

    def analyse(self, window):
        idx = next(i for i, t in enumerate(window) if t.speaker == "USER")
        return AnalysePass(modifiable=True, target_turn_idx=idx, target_speaker="USER",
                           candidate_operators=[self.id],
                           rationale="Control: reword a turn while preserving coherence.")

    def document(self, window, analysis):
        idx = analysis.target_turn_idx
        return self._doc(window, analysis, window[idx].utterance, {"turn_idx": idx})


class ReferenceBreak(_WindowOp):
    id = "reference_break"
    family = "perturbation"

    def is_applicable(self, window):
        return self._first_slot_entity(window) is not None

    def analyse(self, window):
        hit = self._first_slot_entity(window)
        if hit is None:
            return AnalysePass(modifiable=False, rationale="No entity to break reference on.")
        idx, entity = hit
        return AnalysePass(modifiable=True, target_turn_idx=idx,
                           target_speaker=window[idx].speaker, target_entity=entity,
                           candidate_operators=[self.id],
                           rationale=f"Replace entity {entity!r} with an unresolved referent.")

    def document(self, window, analysis):
        idx, entity = analysis.target_turn_idx, analysis.target_entity
        change_to = window[idx].utterance.replace(entity, "that one")
        return self._doc(window, analysis, change_to,
                         {"turn_idx": idx, "entity": entity, "to": "that one"})


class Contradiction(_WindowOp):
    id = "contradiction"
    family = "perturbation"

    def is_applicable(self, window):
        return self._carried_slot(window) is not None

    def analyse(self, window):
        hit = self._carried_slot(window)
        if hit is None:
            return AnalysePass(modifiable=False, rationale="No carried slot to contradict.")
        idx, slot, value = hit
        return AnalysePass(modifiable=True, target_turn_idx=idx, target_speaker="USER",
                           target_entity=str(value), candidate_operators=[self.id],
                           rationale=f"Contradict carried {slot}={value!r} in a later turn.")

    def document(self, window, analysis):
        idx, value = analysis.target_turn_idx, analysis.target_entity
        change_to = window[idx].utterance.rstrip(".?! ") + f". Actually, forget {value} — somewhere completely different."
        return self._doc(window, analysis, change_to, {"turn_idx": idx, "contradicts": value})


class SlotValueMismatch(_WindowOp):
    id = "slot_value_mismatch"
    family = "perturbation"

    def is_applicable(self, window):
        return self._system_entity_turn(window) is not None

    def analyse(self, window):
        hit = self._system_entity_turn(window)
        if hit is None:
            return AnalysePass(modifiable=False, rationale="No system entity to mismatch.")
        idx, entity = hit
        return AnalysePass(modifiable=True, target_turn_idx=idx, target_speaker="SYSTEM",
                           target_entity=entity, candidate_operators=[self.id],
                           rationale=f"System states {entity!r}; replace with a value not in state.")

    def document(self, window, analysis):
        idx, entity = analysis.target_turn_idx, analysis.target_entity
        alt = _ALT_ENTITY.get(entity, "some other place")
        change_to = window[idx].utterance.replace(entity, alt)
        return self._doc(window, analysis, change_to,
                         {"turn_idx": idx, "entity": entity, "to": alt})


class NonSequitur(_WindowOp):
    id = "non_sequitur"
    family = "injection"

    def is_applicable(self, window):
        return any(t.speaker == "SYSTEM" for t in window)

    def analyse(self, window):
        idx = next(i for i, t in enumerate(window) if t.speaker == "SYSTEM")
        return AnalysePass(modifiable=True, target_turn_idx=idx, target_speaker="SYSTEM",
                           candidate_operators=[self.id],
                           rationale="Replace a system turn with an off-intent response (relevance break).")

    def document(self, window, analysis):
        idx = analysis.target_turn_idx
        change_to = "The weather has been lovely this week, hasn't it?"
        return self._doc(window, analysis, change_to, {"turn_idx": idx, "replaced": True})


class OffTopicInsertion(_WindowOp):
    id = "off_topic_insertion"
    family = "injection"

    def is_applicable(self, window):
        return any(t.speaker == "USER" for t in window)

    def analyse(self, window):
        # pick a middle user turn so the drift is mid-dialogue
        users = [i for i, t in enumerate(window) if t.speaker == "USER"]
        idx = users[len(users) // 2]
        return AnalysePass(modifiable=True, target_turn_idx=idx, target_speaker="USER",
                           candidate_operators=[self.id],
                           rationale="Insert an unrelated topic mid-dialogue (global-coherence break).")

    def document(self, window, analysis):
        idx = analysis.target_turn_idx
        change_to = window[idx].utterance.rstrip(".?! ") + ". By the way, did you catch the football game last night?"
        return self._doc(window, analysis, change_to, {"turn_idx": idx, "inserted_topic": "football"})


class TurnReorder(_WindowOp):
    id = "turn_reorder"
    family = "perturbation"

    def is_applicable(self, window):
        return len(window) >= 3

    def analyse(self, window):
        # swap an adjacent pair in the middle of the window
        idx = len(window) // 2
        return AnalysePass(modifiable=True, target_turn_idx=idx,
                           target_speaker=window[idx].speaker, candidate_operators=[self.id],
                           rationale=f"Swap turns {idx} and {idx + 1} to break sequence (local coherence).")

    def document(self, window, analysis):
        idx = analysis.target_turn_idx
        j = min(idx + 1, len(window) - 1)
        return self._doc(window, analysis, window[idx].utterance,
                         {"turn_idx": idx, "swap_with": j})
