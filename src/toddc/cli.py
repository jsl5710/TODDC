"""Minimal CLI for TODDC."""
from __future__ import annotations

import argparse
import json
import sys


def _demo() -> int:
    from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
    from toddc.operators import get_operator
    from toddc.passes import run_chain
    d = parse_dialogue(SGD_1_00000_RAW)
    rec = run_chain(dialogue_id=d.dialogue_id, window=d.turns,
                    operator=get_operator("reference_break"), services=d.services, seed=42)
    if rec is None:
        print("not a viable site", file=sys.stderr)
        return 1
    print(json.dumps(rec.to_dict(), indent=2))
    return 0


def _dialogue() -> int:
    from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
    from toddc.operators import all_operators
    from toddc.passes import run_dialogue
    d = parse_dialogue(SGD_1_00000_RAW)
    recs = run_dialogue(dialogue_id=d.dialogue_id, window=d.turns,
                        operators=all_operators(), services=d.services, policy="all", seed=1)
    print(f"{len(recs)} samples from one dialogue:")
    for r in sorted(recs, key=lambda x: (x.target_turn_idx, x.operator)):
        p = r.position
        print(f"  turn {r.target_turn_idx} [{p.band:6}] {r.operator:20} "
              f"{str(r.dimension):17} label={r.gold.label}")
    return 0


def _generate(live: bool) -> int:
    from toddc.generate import generate_seed
    llm = judge = None
    if live:
        from toddc.judge import Judge
        from toddc.runners.factory import client_for_role
        llm = client_for_role("generator")
        judge = Judge(client_for_role("judge"))
    stats = generate_seed(llm=llm, judge=judge)
    print("Seed set written to data/seed_v1/")
    print(json.dumps(stats.__dict__, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="toddc")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the chain on a fixture and print the record")
    sub.add_parser("dialogue", help="spread violations across a fixture dialogue")
    g = sub.add_parser("generate", help="build the seed set into data/seed_v1/")
    g.add_argument("--live", action="store_true", help="use live generator+judge (needs keys)")
    args = p.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "dialogue":
        return _dialogue()
    if args.cmd == "generate":
        return _generate(args.live)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
