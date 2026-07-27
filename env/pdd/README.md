# PDD environment

**Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for
Positional Discourse Coherence** (2024).

Dialogue-level metric (use `simulate_pairwise`). PDD measures positional
discourse coherence; incoherence = 1 − coherence.

## Setup
```bash
python -m venv ~/envs/pdd && source ~/envs/pdd/bin/activate
# clone the authors' repository and install it + its requirements
pip install -r requirements.txt
# implement _pdd_coherence() in runner.py to call the repo's scorer
export TODDC_PDD_PYTHON=~/envs/pdd/bin/python
```

PDD has no PyPI package yet, so `runner.py` ships a stub that raises until you
wire the repo's scoring call — TODDC surfaces this as `MetricUnavailable`.
