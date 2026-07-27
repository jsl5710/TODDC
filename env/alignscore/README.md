# AlignScore environment

**AlignScore: Evaluating Factual Consistency with a Unified Alignment Function**
(Zha, Yang, Li, Hu — ACL 2023). Repo: https://github.com/yuh-zha/AlignScore

Per-turn metric: incoherence(turn) = 1 − alignment(context = prior turns, claim = turn).

## Setup (its own env)
```bash
python -m venv ~/envs/alignscore && source ~/envs/alignscore/bin/activate
pip install -r requirements.txt
# download a checkpoint (see the repo), e.g. AlignScore-large.ckpt
export ALIGNSCORE_CKPT=/path/to/AlignScore-large.ckpt
```

## Point TODDC at it
```bash
export TODDC_ALIGNSCORE_PYTHON=~/envs/alignscore/bin/python
PYTHONPATH=src python -m toddc.cli simulate --metric alignscore
```
TODDC invokes `runner.py` in this env via subprocess; nothing AlignScore-related
is installed in the main TODDC env.
