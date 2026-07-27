# Coherence taxonomy

TODDO organizes injection around five **dimensions** of discourse coherence,
grounded in discourse/dialogue theory. This page is the reference operators and
labelers cite by name.

## Dimensions

### local — sequential/adjacent coherence
Do adjacent turns follow one another sensibly? Rooted in Centering Theory and the
**entity grid**: entities/topics should transition smoothly turn to turn.
> Breaking it: reorder adjacent turns so the reply precedes its prompt.

### cohesion — reference resolution
Do referring expressions (pronouns, definite descriptions) resolve to the right
antecedent?
> Breaking it: replace a named entity with a dangling "that one" / "it".

### global — topic/goal consistency
Does the whole dialogue stay on one task/topic?
> Breaking it: insert an unrelated topic mid-dialogue (topic drift).

### relevance — response appropriateness
Does a system turn actually address the user's intent / `requested_slots`?
> Breaking it: replace a system turn with an off-intent non-sequitur.

### state_consistency — belief-state fidelity
Does a turn respect the accumulated belief state (no contradiction, no fabricated
slot value)?
> Breaking it: contradict a carried slot, or have the system assert a value that
> isn't in the state.

## Labels & severity

Each sample carries a **gold coherence label** (`coherent` | `incoherent`),
derived by rule from the operator:

| Family | Label | Severity |
| --- | --- | --- |
| `control` (paraphrase) | `coherent` | `none` |
| `perturbation` / `injection` | `incoherent` | `minor`, or `major` for contradiction / slot_value_mismatch |

`gold.should_flag == (label == "incoherent")` — a coherence metric or a dialogue
system should flag exactly the incoherent samples. Controls exist to measure the
**false-incoherence rate** (flagging a coherent dialogue as broken).
