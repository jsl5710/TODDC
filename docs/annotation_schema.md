# Annotation record schema

One (coherent, perturbed) sample = one JSON record. Canonical machine schema:
`data/schema/annotation.schema.json`. Human companion below.

| Field | Meaning |
| --- | --- |
| `record_id` | `{dialogue_id}:{turn}:{operator}:{seed}` |
| `dialogue_id`, `services` | SGD provenance |
| `target_turn_idx` | index of the perturbed turn in the window |
| `family` | `control` \| `perturbation` \| `injection` |
| `operator` | operator id (see coherence_operators.md) |
| `dimension` | coherence dimension broken (null for controls) |
| `position` | where the turn sits (ordinal, band, relative_position) |
| `source_window` | the original ordered turns (speaker + utterance + belief_state) |
| `passes` | the 5-pass chain output, each retrievable by key |
| `gold` | `{label, dimension, violation_type, severity, should_flag}` |
| `provenance` | sgd_version, seed, generator/judge models |

## Design invariants (enforced by tests)

1. `gold.should_flag == (gold.label == "incoherent")`.
2. `family == "control"` ⇒ `label == "coherent"` and `dimension == null`.
3. `label == "incoherent"` ⇒ `dimension` and `violation_type` are non-null.
4. `passes.confirm.status == "accepted"` ⇒ `change_applied == true`.
5. `passes.edit.mode == "copy"` ⇒ `confirm.change_applied == true`.

## Using the pairs

Each record is one half of a pair: `source_window` (coherent) vs. the perturbed
window in `passes.edit.final_window` (incoherent, unless a control). Feed both to
a coherence metric or a dialogue system and check it flags the perturbed one and
not the original — scored with `toddo.eval.detection_accuracy` /
`false_incoherence_rate`.
