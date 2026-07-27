"""Where in the dialogue to inject the violation — spread across positions.

Like TODUQ: enumerate applicable (window, operator) sites and stratify by
position so violations land at early/middle/late points, not a fixed turn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from toddo.operators.base import Operator
from toddo.schema import DialogueTurn, Position

Band = Literal["early", "middle", "late"]


def position_of(ordinal: int, num_turns: int) -> Position:
    n = max(1, num_turns)
    rel = 0.0 if n <= 1 else ordinal / (n - 1)
    band: Band = "early" if rel < 1 / 3 else "middle" if rel < 2 / 3 else "late"
    return Position(turn_ordinal=ordinal, num_turns=num_turns,
                    relative_position=round(rel, 4), band=band)


@dataclass
class Site:
    operator: Operator
    target_turn_idx: int
    position: Position


def enumerate_sites(window: Sequence[DialogueTurn], operators: Sequence[Operator]) -> list[Site]:
    sites: list[Site] = []
    n = len(window)
    for op in operators:
        if op.is_applicable(list(window)):
            analysis = op.analyse(list(window))
            if analysis.modifiable and analysis.target_turn_idx is not None:
                sites.append(Site(op, analysis.target_turn_idx,
                                  position_of(analysis.target_turn_idx, n)))
    return sites


def select_sites(sites: Sequence[Site], *, policy: str = "all", k: Optional[int] = None,
                 seed: int = 0) -> list[Site]:
    rng = random.Random(seed)
    pool = list(sites)
    rng.shuffle(pool)
    if policy == "all":
        return sorted(pool, key=lambda s: (s.target_turn_idx, s.operator.id))
    buckets: dict[Band, list[Site]] = {"early": [], "middle": [], "late": []}
    for s in pool:
        buckets[s.position.band].append(s)  # type: ignore[index]
    target = k or len(pool)
    chosen: list[Site] = []
    order: list[Band] = ["early", "middle", "late"]
    while len(chosen) < target and any(buckets[b] for b in order):
        for b in order:
            if buckets[b] and len(chosen) < target:
                chosen.append(buckets[b].pop())
    return sorted(chosen, key=lambda s: (s.target_turn_idx, s.operator.id))
