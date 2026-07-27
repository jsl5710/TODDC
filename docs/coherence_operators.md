# Coherence-violation operators (v1)

An **operator** is a deterministic transformation of a dialogue window that
injects one coherence violation. It owns the gold label; the LLM only rewords its
output. Operators are grouped by manipulation **family** and tagged with the
**dimension** they break.

| id | family | dimension | what it does | applicability |
| --- | --- | --- | --- | --- |
| `reference_break` | perturbation | cohesion | replace a slot-value entity with an unresolved referent ("that one") | a turn verbalizes a slot value |
| `contradiction` | perturbation | state_consistency | append a clause contradicting a carried slot | a slot carried from an earlier turn |
| `slot_value_mismatch` | perturbation | state_consistency | system asserts an entity/value not in the state | a SYSTEM turn names an entity |
| `non_sequitur` | injection | relevance | replace a system turn with an off-intent response | any SYSTEM turn |
| `off_topic_insertion` | injection | global | insert an unrelated topic mid-dialogue | any USER turn |
| `turn_reorder` | perturbation | local | swap adjacent turns to break sequence | window ≥ 3 turns |
| `coherent_paraphrase` | control | — | reword preserving coherence (label stays `coherent`) | any USER turn |

## Operator contract

```python
class Operator:
    id: str
    family: Literal["control", "perturbation", "injection"]
    def is_applicable(self, window) -> bool: ...
    def analyse(self, window) -> AnalysePass: ...          # locate the site
    def document(self, window, analysis) -> DocumentPass:  # sets the gold label
    def apply(self, window, spec, llm) -> ApplyPass:       # realizes the edit
```

`is_applicable` + `analyse` + `document` are pure/deterministic — the gold label
is never produced by an LLM. `apply` may call the LLM for fluent paraphrase
variants of the template edit.

## Structural vs. textual operators

Most operators edit one turn's **text**; `turn_reorder` edits the **sequence**
(the target turn's text is unchanged). The confirm pass validates each
accordingly — a text-diff check for textual edits, a swap check for structural
ones — so a reorder is confirmed by the swap, not by a (nonexistent) text change.
