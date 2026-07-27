# TODDC — Task-Oriented Dialogue Discourse Coherence

A dataset-development framework for injecting **controlled discourse-coherence
violations** into task-oriented dialogue (TOD) and labeling them, so coherence
metrics and conversational / agentic AI systems can be trained and evaluated on
*whether a dialogue actually hangs together*.

TODDC is the sibling of [**TODUQ**](https://github.com/jsl5710/TODUQ): same base
dataset ([Schema-Guided Dialogue](https://huggingface.co/datasets/GEM/schema_guided_dialog)),
same **chain-of-passes** generation method, same provider-agnostic + LLM-as-judge
infrastructure — but where TODUQ injects *uncertainty*, TODDC injects
*incoherence*.

```
coherent SGD dialogue ──► inject a coherence violation ──► (coherent, perturbed) pair
                                                               │  labeled with:
                                                               ▼
                                    dimension · violation_type · severity · gold coherence label
```

## Why it pairs with TODUQ + TODUQ-MoA

Uncertainty-driven interventions — a clarification, a hand-off, a RAG detour — are
exactly the moments a task-oriented dialogue is most likely to lose its thread.
TODDC is the natural evaluator for those flows: **TODUQ** triggers the
intervention, **TODUQ-MoA** acts on it, **TODDC** scores whether the resulting
dialogue stayed coherent.

## Coherence dimensions (v1)

| Dimension | What it captures |
| --- | --- |
| **local** | adjacent-turn cohesion, entity/topic continuity (Centering / entity-grid) |
| **cohesion** | reference resolution — pronouns/definites resolve to the right antecedent |
| **global** | topic/goal consistency across the whole dialogue |
| **relevance** | does the system turn address the user's intent / `requested_slots` |
| **state_consistency** | does a turn respect the belief state (no slot contradiction) |

See [`docs/taxonomy.md`](docs/taxonomy.md).

## Violation operators (v1)

| id | family | dimension | what it does |
| --- | --- | --- | --- |
| `reference_break` | perturbation | cohesion | replace an entity with an unresolvable referent |
| `contradiction` | perturbation | state_consistency | make a turn contradict an accumulated slot |
| `slot_value_mismatch` | perturbation | state_consistency | system answers with a value that isn't in the state |
| `non_sequitur` | injection | relevance | replace a system turn with an off-intent response |
| `off_topic_insertion` | injection | global | insert an unrelated topic mid-dialogue |
| `turn_reorder` | perturbation | local | swap adjacent turns to break sequence |
| `coherent_paraphrase` | control | — | reword, preserving coherence (gold stays `coherent`) |

See [`docs/coherence_operators.md`](docs/coherence_operators.md).

## Chain-of-passes generation

Every sample is produced by the same deterministic **5-pass chain** as TODUQ, and
each pass is a retrievable key in one JSON record:

| Pass | Key | Question |
| --- | --- | --- |
| 1 · Analyse | `passes.analyse` | *Where* can violation type X be introduced? |
| 2 · Document | `passes.document` | *What* changes — `from` → `to`, gold coherence label |
| 3 · Apply | `passes.apply` | *Make* the edit (template + LLM paraphrase) |
| 4 · Confirm | `passes.confirm` | *Did* it break coherence as intended? judge-gate |
| 5 · Edit | `passes.edit` | *Fix* anything missed, else promote the final version |

The template operator owns the **label**; the LLM owns the **wording**; the judge
guards the join. See [`docs/pass_chain.md`](docs/pass_chain.md).

## Deliverable

v1 = a curated seed set of **(coherent, perturbed) pairs** — each carrying the
violation type, dimension, severity, and gold coherence label — plus the
reproducible pipeline that scales to full SGD.

## Evaluation

`toddc.eval`: entity-grid continuity, reference-resolution checks, belief-state
consistency, next-utterance ranking, and an LLM-judge coherence rubric — so both
a metric *and* a dialogue system can be scored on the same labeled violations.

## Simulator

The **TODDC Simulator** replays a (coherent, perturbed) sample turn-by-turn and
tests whether a coherence metric flags the violation at the perturbed turn while
avoiding a false alarm on the coherent control:

```bash
PYTHONPATH=src python -m toddc.cli simulate
```

See [`docs/simulator.md`](docs/simulator.md).

## License

Derived from SGD (CC BY-SA 4.0) — data artifacts inherit CC BY-SA 4.0; code under
MIT. See [`LICENSE`](LICENSE).
