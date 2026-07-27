# DiscoScore environment

**DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence**
(Zhao, Strube, Eger — EACL 2023). Repo: https://github.com/AIPHES/DiscoScore

Dialogue-level metric (use `simulate_pairwise`): score the coherent original vs
the perturbed dialogue; the perturbed should be judged more incoherent.

## Setup
```bash
python -m venv ~/envs/discoscore && source ~/envs/discoscore/bin/activate
pip install -r requirements.txt
export TODDC_DISCOSCORE_PYTHON=~/envs/discoscore/bin/python
```

Note: DiscoScore has several variants (DS_Focus_NN, DS_SENT_NN, ...). The runner
uses DS_Focus_NN as a coherence proxy and maps it to incoherence in [0,1]; adjust
the variant/normalization for your data.
