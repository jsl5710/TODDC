# TODDC Simulator

The simulator replays a TODDC sample — a dialogue with one **coherence violation**
injected at a target turn — **turn-by-turn**, scores each turn's *incoherence*
with a coherence metric, and reports whether the metric flags the violation at the
perturbed turn, and — for a coherent control — whether it avoids a **false alarm**.

```
TODDC sample (perturbed dialogue window + gold label)
        │ score every turn's incoherence
        ▼
   metric.score_turn(window, i)   (never sees the label)
        │
        ▼
   peak at the perturbed turn?  → localized
   coherent control: any turn flagged?  → false_alarm
```

## Components (`src/toddc/simulator/`)

- **Coherence metrics** (`metrics.py`) — each scores one turn's incoherence in
  `[0, 1]`:
  - `HeuristicCoherenceMetric` — offline. Off-domain detection (relevance),
    topic-drift markers (global), dangling referents (cohesion), and contradiction
    markers (state consistency).
  - `LLMCoherenceMetric` — asks a model to rate the turn's coherence in context;
    incoherence = 1 − coherence (needs a live model).
- **`simulate_record`** (`simulator.py`) — replays `passes.edit.final_window`,
  scores each turn, and returns a `SimResult` (`turn_scores`, `predicted_idx`,
  `rank_of_target`, `localized`, `false_alarm`).

## What "localized" means

- For an **incoherent** sample (`gold.should_flag == True`): the metric peaks at
  the target turn (≥ threshold).
- For a **coherent control** (`coherent_paraphrase`): **no** turn is flagged —
  a good metric must not call a meaning-preserving paraphrase incoherent
  (`false_alarm == False`).

## Two modes & metric choice

- `--mode history` (default) — each turn is scored with its **preceding context**.
- `--mode immediate` — each turn is scored **in isolation**.
- `--metric` — pick any coherence metric: `heuristic` | `entity_grid` |
  `llm_coherence` | `alignscore` | `discoscore` | `pdd`. The last three run in
  their own environments — see [`docs/coherence_metrics.md`](coherence_metrics.md).

Intrinsic metrics (heuristic) score the same in both modes; context-aware ones
(entity_grid, alignscore, llm) differ. Dialogue-level metrics (discoscore, pdd)
use `simulate_pairwise` (coherent vs perturbed) instead of per-turn localization.

## Run it

```bash
PYTHONPATH=src python -m toddc.cli simulate --metric heuristic --mode history
PYTHONPATH=src python -m toddc.cli simulate --metric entity_grid --mode immediate
```

Offline output (heuristic metric):

```
non_sequitur         relevance         target@t1 peak@t1   localized=True  false_alarm=False  t1=0.8
off_topic_insertion  global            target@t4 peak@t4   localized=True  false_alarm=False  t4=0.7
reference_break      cohesion          target@t2 peak@t2   localized=True  false_alarm=False  t2=0.7
contradiction        state_consistency target@t4 peak@t4   localized=True  false_alarm=False  t4=0.8
slot_value_mismatch  state_consistency target@t5 peak@—    localized=False false_alarm=False  (no flag)
turn_reorder         local             target@t4 peak@—    localized=False false_alarm=False  (no flag)
coherent_paraphrase  —                 target@t0 peak@—    localized=True  false_alarm=False  (no flag)
```

## Reading the result — metric coverage by dimension

The heuristic metric localizes **relevance** (off-domain non-sequitur), **global**
(topic drift), **cohesion** (dangling referent), and **state-contradiction**
violations, with **zero false alarms** on the control. It misses **slot-value
mismatch** (a plausible-sounding value that isn't in the belief state — needs a
**state cross-check**) and **turn reorder** (the target text is unchanged — needs
a **positional/entity-grid** check). That is the simulator's purpose: measure
*which coherence metric catches which dimension, at the right turn, without false
alarms*. Swap in `LLMCoherenceMetric` (or a state-aware / positional check) with a
live model to cover the remaining dimensions.

## Aggregate evaluation

Across many samples, `SimResult.localized` and `false_alarm` give a metric's
**localization accuracy** and **false-incoherence rate** per dimension — the
headline numbers for comparing coherence metrics (or dialogue systems) on TODDC
data.
