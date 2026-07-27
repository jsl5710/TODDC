"""Dataclass mirror of data/schema/annotation.schema.json (stdlib, dep-free).

A TODDC record is a (coherent, perturbed) sample: a dialogue window with one
coherence violation injected at a target turn, plus the 5-pass provenance and a
gold coherence label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Dimension = Literal["local", "cohesion", "global", "relevance", "state_consistency"]
Family = Literal["control", "perturbation", "injection"]
Severity = Literal["none", "minor", "major"]
CoherenceLabel = Literal["coherent", "incoherent"]
Status = Literal["accepted", "needs_review", "rejected"]
Speaker = Literal["USER", "SYSTEM"]


@dataclass
class Frame:
    active_intent: Optional[str] = None
    requested_slots: list[str] = field(default_factory=list)
    slot_values: dict[str, Any] = field(default_factory=dict)


BeliefState = dict[str, Frame]


@dataclass
class DialogueTurn:
    speaker: Speaker
    utterance: str
    belief_state: BeliefState = field(default_factory=dict)


@dataclass
class AnalysePass:
    modifiable: bool
    target_turn_idx: Optional[int] = None
    target_speaker: Optional[Speaker] = None
    target_entity: Optional[str] = None
    candidate_operators: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DocumentPass:
    operator: str
    change_from: str
    change_to: str
    edit: dict[str, Any]                     # structured description of the change
    dimension: Dimension
    expected_severity: Severity
    gold_label: CoherenceLabel
    violation_type: Optional[str] = None


@dataclass
class ApplyPass:
    modified_utterance: str
    method: str
    paraphrase_variants: list[str] = field(default_factory=list)
    new_window: list[dict[str, Any]] = field(default_factory=list)  # perturbed turns


@dataclass
class ConfirmPass:
    change_applied: bool
    status: Status
    structural_checks: dict[str, Any] = field(default_factory=dict)
    judge_verdict: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class EditPass:
    mode: Literal["copy", "repair"]
    final_utterance: str
    final_window: list[dict[str, Any]] = field(default_factory=list)
    final_status: Literal["finalized", "unresolved"] = "finalized"
    changes: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Gold:
    label: CoherenceLabel
    dimension: Optional[Dimension]
    violation_type: Optional[str]
    severity: Severity
    should_flag: bool                         # incoherent -> a metric/system should flag it


@dataclass
class Provenance:
    sgd_version: str
    seed: int
    generator_model: Optional[str] = None
    judge_model: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Position:
    turn_ordinal: int
    num_turns: int
    relative_position: float
    band: Literal["early", "middle", "late"] = "early"


@dataclass
class Record:
    record_id: str
    dialogue_id: str
    services: list[str]
    target_turn_idx: int
    family: Family
    operator: str
    dimension: Optional[Dimension]
    position: Position
    source_window: list[DialogueTurn]
    passes_analyse: AnalysePass
    passes_document: DocumentPass
    passes_apply: ApplyPass
    passes_confirm: ConfirmPass
    passes_edit: EditPass
    gold: Gold
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        def turn(t: DialogueTurn) -> dict[str, Any]:
            return {"speaker": t.speaker, "utterance": t.utterance,
                    "belief_state": {s: asdict(f) for s, f in t.belief_state.items()}}
        return {
            "record_id": self.record_id,
            "dialogue_id": self.dialogue_id,
            "services": self.services,
            "target_turn_idx": self.target_turn_idx,
            "family": self.family,
            "operator": self.operator,
            "dimension": self.dimension,
            "position": asdict(self.position),
            "source_window": [turn(t) for t in self.source_window],
            "passes": {
                "analyse": asdict(self.passes_analyse),
                "document": asdict(self.passes_document),
                "apply": asdict(self.passes_apply),
                "confirm": asdict(self.passes_confirm),
                "edit": asdict(self.passes_edit),
            },
            "gold": asdict(self.gold),
            "provenance": asdict(self.provenance),
        }
