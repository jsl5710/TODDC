"""Operator contract: a deterministic coherence-violation edit that owns the label.

Operators act on a dialogue *window* (ordered turns) and a target turn. analyse +
document are pure (they set the gold label); apply may call an LLM for fluent
paraphrase variants. The window lets relevance/cohesion operators see context.
"""
from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from typing import Optional

from toddo.runners.base import LLMClient
from toddo.schema import AnalysePass, ApplyPass, DialogueTurn, DocumentPass, Family

_ENTITY = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")  # multiword proper nouns


class Operator(ABC):
    id: str
    family: Family

    @abstractmethod
    def is_applicable(self, window: list[DialogueTurn]) -> bool: ...

    @abstractmethod
    def analyse(self, window: list[DialogueTurn]) -> AnalysePass: ...

    @abstractmethod
    def document(self, window: list[DialogueTurn], analysis: AnalysePass) -> DocumentPass: ...

    @abstractmethod
    def apply(self, window: list[DialogueTurn], spec: DocumentPass,
              llm: Optional[LLMClient]) -> ApplyPass: ...

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _clone(window: list[DialogueTurn]) -> list[DialogueTurn]:
        return copy.deepcopy(window)

    @staticmethod
    def _window_dicts(window: list[DialogueTurn]) -> list[dict]:
        return [{"speaker": t.speaker, "utterance": t.utterance} for t in window]

    @staticmethod
    def _first_slot_entity(window: list[DialogueTurn]) -> Optional[tuple[int, str]]:
        """First (turn_idx, value) where a slot value appears verbatim in a turn."""
        for i, t in enumerate(window):
            for fr in t.belief_state.values():
                for value in fr.slot_values.values():
                    if value and str(value) in t.utterance:
                        return i, str(value)
        return None

    @staticmethod
    def _carried_slot(window: list[DialogueTurn]) -> Optional[tuple[int, str, object]]:
        """A user turn that carries a slot introduced earlier (target for contradiction)."""
        seen: dict[str, object] = {}
        for i, t in enumerate(window):
            if t.speaker != "USER":
                continue
            for fr in t.belief_state.values():
                for slot, value in fr.slot_values.items():
                    if slot in seen and value is not None:
                        return i, slot, value
                    seen[slot] = value
        return None

    @staticmethod
    def _system_entity_turn(window: list[DialogueTurn]) -> Optional[tuple[int, str]]:
        """A SYSTEM turn stating a proper-noun entity (target for slot_value_mismatch)."""
        for i, t in enumerate(window):
            if t.speaker == "SYSTEM":
                m = _ENTITY.search(t.utterance)
                if m:
                    return i, m.group(1)
        return None

    def _maybe_paraphrase(self, text: str, llm: Optional[LLMClient], n: int = 2):
        if llm is None:
            return "template", []
        from toddo.prompts import render_paraphrase_prompt
        out = llm.generate(render_paraphrase_prompt(text, n=n))
        variants = [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()][:n]
        return "template+llm_paraphrase", variants
