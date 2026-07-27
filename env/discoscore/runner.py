"""DiscoScore runner — executes INSIDE the discoscore env.

Protocol:
  stdin : {"granularity":"dialogue","items":[{"text": "<full dialogue>"}]}
  stdout: {"scores":[incoherence in [0,1]]}

DiscoScore is discourse-coherence for text generation (BERT + discourse). We use
the DS-Focus (NN) variant as a coherence proxy for the dialogue text and map it
to an incoherence in [0,1] via a saturating transform (higher raw = more
coherent). Tune the normalization for your corpus.
"""
import json, math, sys

def _to_incoherence(raw: float) -> float:
    # DiscoScore raw is unbounded/relative; squash so higher coherence -> lower
    # incoherence. 1 / (1 + exp(raw)) is monotone decreasing in raw.
    return 1.0 / (1.0 + math.exp(raw))

def main():
    req = json.load(sys.stdin)
    text = req["items"][0]["text"]
    from disco_score import DiscoScorer
    scorer = DiscoScorer(device="cpu", model_name="bert-base-uncased")
    # DS_Focus_NN(system, [reference]); reference-free coherence proxy uses self
    raw = scorer.DS_Focus_NN(text, [text])
    print(json.dumps({"scores": [_to_incoherence(float(raw))]}))

if __name__ == "__main__":
    main()
