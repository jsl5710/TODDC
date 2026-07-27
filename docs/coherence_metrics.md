# Coherence metrics

TODDC scores dialogue coherence with pluggable metrics, loaded by name from the
registry (`toddc.simulator.load_coherence_metric`). Two live in the main env; the
three literature metrics run in **their own environments** (heavy, conflicting
deps) and are invoked over a subprocess JSON protocol.

## Registry

| name | granularity | env | signal |
| --- | --- | --- | --- |
| `heuristic` | per-turn | main (offline) | off-domain / drift / dangling-referent / contradiction markers |
| `entity_grid` | per-turn | main (offline) | local continuity: 1 − token overlap with the previous turn (context-aware) |
| `llm_coherence` | per-turn | main (needs a model client) | asks a model to rate coherence; 1 − coherence |
| `alignscore` | per-turn | **own env** | 1 − alignment(context, turn) |
| `discoscore` | dialogue | **own env** | discourse coherence of the whole dialogue |
| `pdd` | dialogue | **own env** | positional discourse coherence of the dialogue |

```python
from toddc.simulator import load_coherence_metric, available
available()                       # all names
available(offline_only=True)      # ['entity_grid', 'heuristic']
m = load_coherence_metric("alignscore")
```

## Two scoring modes

Both the CLI (`--mode`) and `simulate_record(..., mode=...)` support:

- **`history`** — each turn is scored with its **preceding conversation history**
  (if any). Context-aware metrics (entity_grid, alignscore, llm) use it.
- **`immediate`** — each turn is scored **in isolation** (no context).

Intrinsic metrics (heuristic) score the same in both modes; context-aware metrics
differ — `entity_grid` scores 0 everywhere in `immediate` mode (no previous turn)
but produces signal in `history` mode.

## Per-turn vs dialogue-level

- **Per-turn** metrics (heuristic, entity_grid, llm, alignscore) → `simulate_record`
  localizes the violation to a turn.
- **Dialogue-level** metrics (discoscore, pdd) → `simulate_pairwise` scores the
  coherent original vs the perturbed dialogue; a good metric rates the perturbed
  as more incoherent.

## The three literature metrics (own environments)

Each has heavy, conflicting dependencies, so it runs in a dedicated venv/conda env
under [`env/<name>/`](../env). TODDC serializes the dialogue to JSON and invokes
`env/<name>/runner.py` **inside that env** via subprocess; nothing heavy is
installed in the main TODDC env. Point TODDC at each env with
`TODDC_<NAME>_PYTHON=/path/to/env/bin/python`; if unset, the metric raises
`MetricUnavailable` with setup instructions (never silently faked).

| Metric | Paper | Env |
| --- | --- | --- |
| **AlignScore** | AlignScore: Evaluating Factual Consistency with a Unified Alignment Function (Zha et al., ACL 2023) | [`env/alignscore/`](../env/alignscore) |
| **DiscoScore** | DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence (Zhao et al., EACL 2023) | [`env/discoscore/`](../env/discoscore) |
| **PDD** | Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for Positional Discourse Coherence (2024) | [`env/pdd/`](../env/pdd) |

### Backend contract
```
stdin  : {"granularity": "per_turn"|"dialogue", "items": [...]}
stdout : {"scores": [float, ...]}     # incoherence in [0, 1]
```
`env/<name>/runner.py` is a thin script that loads the real library and emits
scores; `env/<name>/requirements.txt` + `README.md` cover install. AlignScore is
per-turn (`{context, claim}` items); DiscoScore/PDD are dialogue-level
(`{text}` item).

> Status: the integration layer, runner contracts, and env scaffolding are in
> place and unit-tested (including the `MetricUnavailable` path). The metrics
> themselves are **not run in this repo's CI** — they require their own
> environments and model downloads. PDD ships a runner stub (no PyPI release yet)
> to be wired against the authors' repo.
