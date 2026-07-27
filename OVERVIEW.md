# Project Overview — the TODUQ / TODUQ-MoA / TODDC program

This document explains the **three-repository research program** these projects
form, how they connect, and where **TODDC** sits in it.

## The problem

Conversational and agentic AI systems operating in **task-oriented dialogue**
(booking, search, support) fail in two coupled ways:

1. They **answer when they shouldn't** — confidently hallucinating instead of
   recognizing they are uncertain and deferring (asking, retrieving, escalating).
2. Even when they act correctly, the intervention (a clarification, a hand-off, a
   retrieval detour) can **break the discourse** — the dialogue stops hanging
   together.

Measuring either failure requires **labeled data**: dialogues where we know
*where* uncertainty was introduced and *what the right response is*, and dialogues
where we know *where* coherence was broken and *how*. Both are what this program
builds, from a common base: the **Schema-Guided Dialogue (SGD)** dataset.

## The three repositories

```
        ┌──────────────────────────────────────────────────────────────────┐
        │                        Schema-Guided Dialogue                     │
        │            (belief state: active_intent, requested_slots,         │
        │                          slot_values per turn)                    │
        └───────────────┬───────────────────────────────┬──────────────────┘
                        │                                │
             ┌──────────▼──────────┐          ┌──────────▼──────────┐
             │       TODUQ         │          │       TODDC         │
             │ inject + label      │          │ inject + label      │
             │ UNCERTAINTY         │          │ COHERENCE VIOLATIONS│
             │ → abstain/route     │          │ → coherent/incoherent│
             └──────────┬──────────┘          └──────────┬──────────┘
                        │ gold route                     │ gold coherence label
                        ▼                                │
             ┌─────────────────────┐                     │
             │     TODUQ-MoA       │                     │
             │ act on the route:   │                     │
             │ experts → aggregator│─────────────────────┘
             └─────────────────────┘   evaluate coherence of MoA output
```

| Repo | Builds | Question it answers |
| --- | --- | --- |
| **[TODUQ](https://github.com/jsl5710/TODUQ)** | uncertainty-injection dataset | *Where is the model uncertain, and what should it do — clarify, RAG, hand off, or escalate to a human?* |
| **[TODUQ-MoA](https://github.com/jsl5710/TODUQ-MoA)** | mixture-of-agents inference system | *Given an uncertainty flag, route to the right experts and let a reasoning model aggregate their outputs.* |
| **TODDC** (this repo) | coherence-violation dataset | *Did the dialogue stay coherent — locally, referentially, globally, in relevance, and in belief-state consistency?* |

## The shared method: chain-of-passes generation

All three datasets are generated the same way — a deterministic **5-pass chain**
per sample, serialized to one JSON record where each pass is a retrievable key:

`analyse` (where) → `document` (what change + **gold label**) → `apply` (make the
edit + LLM paraphrase) → `confirm` (judge-gate) → `edit` (finalize/repair).

The rule is invariant across repos: **the template operator owns the label, the
LLM owns the wording, the judge guards the join.** This keeps gold labels
reproducible and independent of any model's free generation.

## TODDC's role

TODDC injects **discourse-coherence violations** into SGD dialogues and labels
each as `coherent` or `incoherent` with the dimension broken:

| Dimension | Operator(s) |
| --- | --- |
| local | `turn_reorder` |
| cohesion | `reference_break` |
| global | `off_topic_insertion` |
| relevance | `non_sequitur` |
| state_consistency | `contradiction`, `slot_value_mismatch` |
| — (control) | `coherent_paraphrase` |

It produces **(coherent, perturbed) pairs** for training and evaluating coherence
metrics, and — critically — for **scoring the dialogues TODUQ-MoA produces**:
uncertainty-driven interventions are exactly the moments most likely to disrupt
flow, so TODDC is the natural evaluator of whether MoA's clarify / RAG / hand-off
behavior kept the conversation coherent.

## How the loop closes

1. **TODUQ** perturbs a user turn and labels the correct route.
2. **TODUQ-MoA** flags the uncertainty and routes it to experts (RAG / HITL /
   clarify), then a reasoning model aggregates.
3. **TODDC** checks whether the resulting multi-turn dialogue stayed coherent.

Each repo runs standalone and offline; together they form an end-to-end pipeline
for building, acting on, and evaluating uncertainty-aware task-oriented dialogue.

## Consolidation note

The three repos currently each carry their own copy of the shared infrastructure
(SGD ingest, provider-agnostic `LLMClient` runners, the judge, the 5-pass
pipeline). A planned `tod-core` package would factor these out so all three import
one implementation.
