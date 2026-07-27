"""Seed-set generation: run the chain over dialogues, partition accepted vs review."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
from toddc.judge import HeuristicJudge
from toddc.operators import all_operators
from toddc.passes import run_dialogue
from toddc.runners.base import LLMClient
from toddc.validate import check_invariants

_OUT = Path(__file__).resolve().parents[2] / "data" / "seed_v1"


@dataclass
class SeedStats:
    total: int
    accepted: int
    needs_review: int
    rejected: int
    invariant_failures: int
    by_dimension: dict[str, int]
    coherent_vs_incoherent: dict[str, int]


def generate_seed(raw_dialogues: Optional[list[dict[str, Any]]] = None, *,
                  llm: Optional[LLMClient] = None, judge=None, policy: str = "all",
                  seed: int = 42, out_dir: Path = _OUT) -> SeedStats:
    raw_dialogues = raw_dialogues or [SGD_1_00000_RAW]
    judge = judge or HeuristicJudge()
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted, review = [], []
    by_dim: dict[str, int] = {}
    by_label: dict[str, int] = {}
    inv = 0
    for raw in raw_dialogues:
        d = parse_dialogue(raw)
        for rec in run_dialogue(dialogue_id=d.dialogue_id, window=d.turns,
                                operators=all_operators(), services=d.services,
                                policy=policy, seed=seed, llm=llm, judge=judge):
            payload = rec.to_dict()
            errs = check_invariants(payload)
            if errs:
                inv += 1
                payload["_invariant_errors"] = errs
            by_dim[str(rec.dimension)] = by_dim.get(str(rec.dimension), 0) + 1
            by_label[rec.gold.label] = by_label.get(rec.gold.label, 0) + 1
            (accepted if rec.passes_confirm.status == "accepted" and not errs
             else review).append(payload)

    _write(out_dir / "records.jsonl", accepted)
    _write(out_dir / "review_queue.jsonl", review)
    stats = SeedStats(
        total=len(accepted) + len(review), accepted=len(accepted),
        needs_review=sum(1 for r in review if r["passes"]["confirm"]["status"] == "needs_review"),
        rejected=sum(1 for r in review if r["passes"]["confirm"]["status"] == "rejected"),
        invariant_failures=inv, by_dimension=dict(sorted(by_dim.items())),
        coherent_vs_incoherent=dict(sorted(by_label.items())),
    )
    manifest = {
        "sgd_version": "GEM/schema_guided_dialog", "num_dialogues": len(raw_dialogues),
        "policy": policy, "seed": seed, "generator_model": getattr(llm, "model_id", "echo-stub"),
        "judge_model": getattr(judge, "judge_model", "heuristic-judge"), "stats": stats.__dict__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
