"""Validate records against the JSON Schema + design invariants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA = Path(__file__).resolve().parents[2] / "data" / "schema" / "annotation.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))


def validate_schema(record: dict[str, Any]) -> None:
    import jsonschema
    jsonschema.validate(record, load_schema())


def check_invariants(record: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    gold = record["gold"]
    doc = record["passes"]["document"]
    confirm = record["passes"]["confirm"]

    # 1. should_flag == (label == incoherent)
    if gold["should_flag"] != (gold["label"] == "incoherent"):
        errs.append("should_flag must equal (label == 'incoherent')")
    # 2. control family -> coherent + no dimension
    if record["family"] == "control":
        if gold["label"] != "coherent":
            errs.append("control family must be labeled 'coherent'")
        if gold["dimension"] is not None:
            errs.append("control family must have no dimension")
    # 3. incoherent -> a dimension + violation_type
    if gold["label"] == "incoherent":
        if gold["dimension"] is None or gold["violation_type"] is None:
            errs.append("incoherent records need a dimension and violation_type")
    # 4. accepted -> change_applied
    if confirm["status"] == "accepted" and not confirm["change_applied"]:
        errs.append("accepted status requires change_applied == true")
    # 5. edit.mode copy -> change_applied
    if record["passes"]["edit"]["mode"] == "copy" and not confirm["change_applied"]:
        errs.append("edit.mode 'copy' requires confirm.change_applied == true")
    return errs
