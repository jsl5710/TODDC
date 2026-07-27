"""Ingest SGD into full dialogue-turn windows (USER + SYSTEM).

Coherence is inter-turn, so — unlike TODUQ, which keeps only user turns — TODDO
keeps the whole ordered sequence (system responses included), plus the per-user
belief state. `parse_dialogue` is pure; `load_sgd` is the HF seam.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from toddo.schema import BeliefState, DialogueTurn, Frame


@dataclass
class Dialogue:
    dialogue_id: str
    services: list[str]
    turns: list[DialogueTurn]


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: (v[0] if isinstance(v, list) and v else v) for k, v in raw.items()}


def _belief(frames: list[dict[str, Any]]) -> BeliefState:
    bs: BeliefState = {}
    for fr in frames:
        st = fr.get("state", {}) or {}
        bs[fr["service"]] = Frame(
            active_intent=st.get("active_intent"),
            requested_slots=list(st.get("requested_slots", []) or []),
            slot_values=_flatten(st.get("slot_values", {}) or {}),
        )
    return bs


def parse_dialogue(raw: dict[str, Any]) -> Dialogue:
    turns = [
        DialogueTurn(speaker=t["speaker"], utterance=t.get("utterance", ""),
                     belief_state=_belief(t.get("frames", [])))
        for t in raw["turns"]
    ]
    return Dialogue(raw["dialogue_id"], list(raw.get("services", [])), turns)


def load_sgd(split: str = "validation") -> Iterator[dict[str, Any]]:  # pragma: no cover
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:
        raise ImportError("Install `datasets` to load SGD.") from e
    raise NotImplementedError(
        "Map the HF export to {dialogue_id, services, turns[{speaker, utterance, frames}]} "
        "and call parse_dialogue(). Wired against fixtures in tests/."
    )


# --- Fixture: SGD 1_00000 (Restaurants_1), full turn sequence ----------------
SGD_1_00000_RAW: dict[str, Any] = {
    "dialogue_id": "1_00000",
    "services": ["Restaurants_1"],
    "turns": [
        {"speaker": "USER", "utterance": "I am feeling hungry so I would like to find a place to eat.",
         "frames": [{"service": "Restaurants_1", "state": {"active_intent": "FindRestaurants", "requested_slots": [], "slot_values": {}}}]},
        {"speaker": "SYSTEM", "utterance": "Do you have a specific area where you want the eating place to be located?",
         "frames": [{"service": "Restaurants_1"}]},
        {"speaker": "USER", "utterance": "I would like for it to be in San Jose.",
         "frames": [{"service": "Restaurants_1", "state": {"active_intent": "FindRestaurants", "requested_slots": [], "slot_values": {"city": ["San Jose"]}}}]},
        {"speaker": "SYSTEM", "utterance": "Is there a specific cuisine type you enjoy, such as Mexican, Italian or something else?",
         "frames": [{"service": "Restaurants_1"}]},
        {"speaker": "USER", "utterance": "I usually like eating the American type of food.",
         "frames": [{"service": "Restaurants_1", "state": {"active_intent": "FindRestaurants", "requested_slots": [], "slot_values": {"city": ["San Jose"], "cuisine": ["American"]}}}]},
        {"speaker": "SYSTEM", "utterance": "I see that at 71 Saint Peter there is a good restaurant which is in San Jose.",
         "frames": [{"service": "Restaurants_1"}]},
        {"speaker": "USER", "utterance": "Can you give me the address of this restaurant?",
         "frames": [{"service": "Restaurants_1", "state": {"active_intent": "FindRestaurants", "requested_slots": ["street_address"], "slot_values": {"city": ["San Jose"], "cuisine": ["American"]}}}]},
        {"speaker": "SYSTEM", "utterance": "If you want to go to this restaurant you can find it at 71 North San Pedro Street.",
         "frames": [{"service": "Restaurants_1"}]},
    ],
}
