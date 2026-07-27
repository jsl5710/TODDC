"""PDD runner — executes INSIDE the pdd env.

**Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for
Positional Discourse Coherence** (2024).

Protocol:
  stdin : {"granularity":"dialogue","items":[{"text": "<full dialogue>"}]}
  stdout: {"scores":[incoherence in [0,1]]}

PDD has no PyPI release as of writing: clone the authors' repository, install it
into this env, and replace `_pdd_coherence` below with the real call. The adapter
maps a PDD coherence in [0,1] to incoherence = 1 - coherence.
"""
import json, sys

def _pdd_coherence(text: str) -> float:
    # TODO: from pdd import score_positional_coherence; return score(text)
    raise NotImplementedError(
        "Wire the PDD repo's scoring function here (see env/pdd/README.md).")

def main():
    req = json.load(sys.stdin)
    text = req["items"][0]["text"]
    coh = _pdd_coherence(text)                 # [0,1] coherence
    print(json.dumps({"scores": [max(0.0, min(1.0, 1.0 - float(coh)))]}))

if __name__ == "__main__":
    main()
