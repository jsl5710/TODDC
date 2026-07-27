"""AlignScore runner — executes INSIDE the alignscore env (not the TODDC env).

Protocol (see toddc/simulator/external.py):
  stdin : {"granularity":"per_turn","items":[{"context":..,"claim":..}, ...]}
  stdout: {"scores":[incoherence in [0,1], ...]}   # incoherence = 1 - alignment

Set the checkpoint path via ALIGNSCORE_CKPT. AlignScore returns alignment in
[0,1]; coherence/consistency incoherence is 1 - alignment.
"""
import json, os, sys

def main():
    req = json.load(sys.stdin)
    items = req["items"]
    from alignscore import AlignScore  # provided by the alignscore package
    scorer = AlignScore(
        model="roberta-large",
        batch_size=16,
        device=os.environ.get("ALIGNSCORE_DEVICE", "cpu"),
        ckpt_path=os.environ["ALIGNSCORE_CKPT"],
        evaluation_mode="nli_sp",
    )
    contexts = [it["context"] or it["claim"] for it in items]   # empty context -> self
    claims = [it["claim"] for it in items]
    align = scorer.score(contexts=contexts, claims=claims)      # list[float] in [0,1]
    print(json.dumps({"scores": [max(0.0, min(1.0, 1.0 - a)) for a in align]}))

if __name__ == "__main__":
    main()
