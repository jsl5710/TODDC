# The chain-of-passes pipeline

Identical in shape to TODUQ: every sample is produced by a **5-pass chain**, and
the whole thing serializes to one JSON record where each pass is retrievable by
key (`passes.analyse` … `passes.edit`).

```
window ─►│1 ANALYSE│─►│2 DOCUMENT│─►│3 APPLY │─►│4 CONFIRM │─►│5 EDIT │─► record
          where can    from→to,      make the   did it       repair
          violation    gold label    edit +     break        or copy
          X go?        (rule-owned)  paraphrase  coherence?   final
```

- **Analyse** — locate the target turn/entity for the violation; drop if not viable.
- **Document** — specify `from → to`, the structured `edit`, the `dimension`, and
  the **derived** gold label + severity. Owns the label (template/rule-driven).
- **Apply** — realize the edit (template canonical form + optional LLM paraphrase
  variants); record the perturbed `new_window`.
- **Confirm** — structural checks (text changed / turns swapped) + an LLM-judge
  gate (fidelity, violation_present, naturalness) → `accepted` / `needs_review` /
  `rejected`.
- **Edit** — promote the confirmed output as the canonical final version, or
  deterministically repair a missed edit.

Why five passes: auditability, label integrity (the label is a rule, not an LLM
generation), cheap controls, reproducibility, and one canonical output per record.
