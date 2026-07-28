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


def _generate(live: bool, workers: int = 0, balance: bool = True,
              balance_ratio: float = 1.0, control_multiplier: str = "auto") -> int:
    from toddc.generate import generate_seed
    llm = pool = judge = judge_pool = None
    if live:
        from toddc.judge import Judge
        from toddc.runners.factory import (
            build_generator_pool, build_judge_pool, client_for_role,
        )
        pool = build_generator_pool()          # multi-model even split, if configured
        if pool is not None:
            print(f"Generators: {len(pool.clients)}-model pool "
                  f"(split={pool.strategy}): {', '.join(pool.labels)}")
        else:
            llm = client_for_role("generator")
        judge_pool = build_judge_pool()        # split the judge across models too
        if judge_pool is not None:
            print(f"Judges: {len(judge_pool.clients)}-model pool: {', '.join(judge_pool.labels)}")
        else:
            judge = Judge(client_for_role("judge"))
    cm: object = "auto" if control_multiplier == "auto" else int(control_multiplier)
    stats = generate_seed(llm=llm, pool=pool, workers=workers, judge=judge,
                          judge_pool=judge_pool, balance=balance,
                          balance_ratio=balance_ratio, control_multiplier=cm)
    print("Seed set written to data/seed_v1/ (records.jsonl = balanced, "
          "records_all.jsonl = full)")
    print(json.dumps(stats.__dict__, indent=2))
    print(f"Class balance: {stats.positive} positive (incoherent) vs "
          f"{stats.negative} negative (coherent) -> {stats.balanced_total} balanced")
    if pool is not None:
        print("Generator split:", json.dumps(pool.summary()))
    if judge_pool is not None:
        print("Judge split:", json.dumps(judge_pool.summary()))
    return 0


def _dry_run(control_multiplier: str = "auto") -> int:
    """Validate the model config and print the planned split — no generation."""
    from toddc.generate import plan_run
    from toddc.runners.factory import check_config, planned_split

    report = check_config()
    print(f"Config: {report['config']}")
    if not report.get("exists"):
        print("  (no models.yaml — a live run would fall back to the offline EchoClient)")
        return 0

    all_ok = True
    for role in ("generators", "judges"):
        checks = report[role]
        print(f"\n{role} ({len(checks)}):")
        if not checks:
            print("  (none configured)")
        for c in checks:
            mark = "OK " if c["ok"] else "XX "
            all_ok = all_ok and c["ok"]
            print(f"  [{mark}] {c['model_id']:<40} {c['kind']:<6} {c['detail']}")

    cm: object = "auto" if control_multiplier == "auto" else int(control_multiplier)
    plan = plan_run(control_multiplier=cm)
    units = plan["total_units"]
    gsplit = planned_split(units, "generators", "generation")
    jsplit = planned_split(units, "judges", "judging")
    print(f"\nPlanned run: {units} generation units "
          f"({plan['violation_units']} violation + {plan['control_units']} control, "
          f"control_multiplier={plan['control_multiplier']}) over "
          f"{plan['num_dialogues']} dialogue(s).")
    if gsplit is not None:
        print("  generator split:", json.dumps(gsplit))
    if jsplit is not None:
        print("  judge split:    ", json.dumps(jsplit))
    print(f"\n{'All endpoints/keys OK — ready to run.' if all_ok else 'Some checks FAILED — fix the marked entries before --live.'}")
    return 0 if all_ok else 1


def _simulate(metric_name: str = "heuristic", mode: str = "history") -> int:
    from toddc.ingest import SGD_1_00000_RAW, parse_dialogue
    from toddc.judge import HeuristicJudge
    from toddc.operators import all_operators
    from toddc.passes import run_dialogue
    from toddc.simulator import load_coherence_metric, simulate_record

    d = parse_dialogue(SGD_1_00000_RAW)
    metric = load_coherence_metric(metric_name)
    records = run_dialogue(dialogue_id=d.dialogue_id, window=d.turns,
                           operators=all_operators(), services=d.services,
                           policy="all", seed=1, judge=HeuristicJudge())
    first: dict[str, object] = {}
    for r in records:
        first.setdefault(r.operator, r)

    print(f"TODDC Simulator — coherence metric = {metric.name}, mode = {mode}\n")
    hits = total = fa = 0
    order = ["non_sequitur", "off_topic_insertion", "reference_break", "contradiction",
             "slot_value_mismatch", "turn_reorder", "coherent_paraphrase"]
    for op_id in order:
        rec = first.get(op_id)
        if rec is None:
            continue
        res = simulate_record(rec, metric, mode=mode)
        total += 1
        hits += res.localized
        fa += res.false_alarm
        peak = ", ".join(f"t{ts.ordinal}={ts.score}" for ts in res.turn_scores if ts.score > 0) or "(no flag)"
        print(f"  {op_id:20} {str(res.dimension):17} target@t{res.target_idx:<2} "
              f"peak@t{res.predicted_idx} localized={str(res.localized):5} "
              f"false_alarm={str(res.false_alarm):5}  scores>0: {peak}")
    print(f"\nlocalization: {hits}/{total}, false alarms: {fa}. The heuristic metric "
          "catches relevance/global/cohesion/contradiction violations; slot-value "
          "mismatch (needs a state cross-check) and reorder (needs a positional "
          "check) need dedicated metrics — swap in LLMCoherenceMetric with a live model.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="toddc")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the chain on a fixture and print the record")
    sub.add_parser("dialogue", help="spread violations across a fixture dialogue")
    sim = sub.add_parser("simulate", help="replay a sample; test if a coherence metric localizes the violation")
    sim.add_argument("--metric", default="heuristic",
                     help="coherence metric (heuristic | entity_grid | llm_coherence | alignscore | discoscore | pdd)")
    sim.add_argument("--mode", default="history", choices=["history", "immediate"],
                     help="history = turn + prior context; immediate = current turn alone")
    g = sub.add_parser("generate", help="build the seed set into data/seed_v1/")
    g.add_argument("--live", action="store_true", help="use live generator+judge (needs keys)")
    g.add_argument("--dry-run", action="store_true",
                   help="validate configured endpoints/keys and print the planned "
                        "split — no generation")
    g.add_argument("--workers", type=int, default=0,
                   help="parallel generation across sites/models (0/1 = sequential)")
    g.add_argument("--no-balance", dest="balance", action="store_false",
                   help="skip class balancing (keep the raw positive-skewed set)")
    g.add_argument("--balance-ratio", type=float, default=1.0,
                   help="majority:minority cap after balancing (1.0 = exact 1:1)")
    g.add_argument("--control-multiplier", default="auto",
                   help="coherent-paraphrase variants per turn to grow the negative "
                        "class ('auto' sizes it from the operator mix)")
    args = p.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "dialogue":
        return _dialogue()
    if args.cmd == "simulate":
        return _simulate(args.metric, args.mode)
    if args.cmd == "generate":
        if args.dry_run:
            return _dry_run(args.control_multiplier)
        return _generate(args.live, args.workers, args.balance,
                         args.balance_ratio, args.control_multiplier)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
