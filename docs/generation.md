# Generation on a server — multi-model, even split

For a large generation run you can split the Pass-3 paraphrasing **evenly across
several models** (Qwen, Llama 3.1, Mistral, …), each served on its own
OpenAI-compatible endpoint (vLLM / Ollama / TGI). Work is divided by a
`ModelPool`, each sample records the model that produced it, and sites can run in
parallel across endpoints. The judge (Pass 4) can be split the same way.

## 1. Serve the models

Each model on its own endpoint, e.g. with vLLM:

```bash
# gpu0
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --port 8000
# gpu1
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --port 8000
# gpu2
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-Small-Instruct-2409 --port 8000
```

## 2. Configure the pool

Copy [`configs/models/servers.example.yaml`](../configs/models/servers.example.yaml)
to `configs/models/models.yaml` and list the `generators` + split strategy:

```yaml
generators:
  - {adapter: toddc.runners.open_source:VLLMClient, model_id: Qwen/Qwen2.5-32B-Instruct, endpoint: http://gpu0:8000/v1}
  - {adapter: toddc.runners.open_source:VLLMClient, model_id: meta-llama/Llama-3.1-70B-Instruct, endpoint: http://gpu1:8000/v1}
  - {adapter: toddc.runners.open_source:VLLMClient, model_id: mistralai/Mistral-Small-Instruct-2409, endpoint: http://gpu2:8000/v1}
generation:
  split: round_robin     # round_robin (exactly even) | random | weighted
  # weights: [2, 1, 1]   # weighted: faster GPUs get a bigger share
  seed: 0
  workers: 4             # parallel generation across sites
```

Split the judge across models too by adding a `judges:` list (omit it to keep a
single `roles.judge`):

```yaml
judges:
  - {adapter: toddc.runners.open_source:VLLMClient, model_id: Qwen/Qwen2.5-72B-Instruct, endpoint: http://gpu4:8000/v1}
  - {adapter: toddc.runners.claude:ClaudeClient,   model_id: claude-sonnet-5,           api_key_env: ANTHROPIC_API_KEY}
judging:
  split: round_robin
  seed: 0
```

## 3. Validate the config first (`--dry-run`)

Before a big run, check every endpoint/key and preview the split — **no
generation, no SDK or GPU needed** (uses only the stdlib):

```bash
PYTHONPATH=src python -m toddc.cli generate --dry-run
```

It `GET`s `/v1/models` on each `generators:`/`judges:` endpoint (marking each
`OK`/`XX` with the reason), checks the env var for any closed API, and prints the
planned per-model split over the real unit count. Exit code is `0` only when every
check passes, so it drops straight into a CI/pre-flight gate. Example:

```
generators (3):
  [OK ] Qwen/Qwen2.5-32B-Instruct       open   reachable, serving Qwen/Qwen2.5-32B-Instruct
  [OK ] meta-llama/Llama-3.1-70B-Instr… open   reachable, serving meta-llama/Llama-3.1-70B-Instruct
  [XX ] mistralai/Mistral-Small-…       open   UNREACHABLE at http://gpu2:8000/v1/models (...)
Planned run: 12 units (6 violation + 6 control, control_multiplier=6) over 1 dialogue(s).
  generator split: {"Qwen/...": 3, "meta-llama/...": 3, "mistralai/...": 3, "gemma2:27b": 3}
```

## 4. Run

```bash
PYTHONPATH=src python -m toddc.cli generate --live --workers 8
```

- `--workers N` runs sites concurrently across the model endpoints (0/1 =
  sequential). Model **assignment** is done sequentially first, so the split
  stays deterministic and even regardless of `--workers`.
- With a `judges:` list the judge (Pass 4) is split too; each site is validated
  by the next judge model in the rotation and the record records which one.

## Split strategies

| strategy | behavior |
| --- | --- |
| `round_robin` | exactly even — model *i* gets sites `i, i+len, i+2·len, …` |
| `weighted` | shares proportional to `weights` (e.g. bigger share to faster GPUs) |
| `random` | uniform random assignment |

One `ModelPool` is shared across the whole run, so the split is even over **all**
dialogues, not just within one.

## Provenance & the manifest

Every record's `provenance.generator_model` names the model that generated it
and `provenance.judge_model` names the model that validated it.
`data/seed_v1/manifest.json` reports each split two ways:

```json
"generators":    {"Qwen/...": 11, "meta-llama/...": 11, "mistralai/...": 11},  // pool counters
"by_generator":  {"Qwen/...": 11, "meta-llama/...": 11, "mistralai/...": 11},  // from record provenance
"judges":        {"Qwen/...": 17, "claude-sonnet-5": 16},                      // judge pool counters
"by_judge":      {"Qwen/...": 17, "claude-sonnet-5": 16}                       // from record provenance
```

so you can confirm the work was divided as intended and trace any sample back to
its generator and judge.

## Class balance (positive vs negative)

Each turn gets many *violation* samples but only one *control* (coherent) sample,
so the raw output is ~85 % positive (incoherent). Generation balances it in two
steps so `records.jsonl` ships **1:1** by default:

1. **Grow the negative class** — `--control-multiplier` (default `auto`) emits N
   distinct coherent-paraphrase variants per turn. `auto` sizes N from the
   operator mix (6 violation ops + 1 control → 6), so a live sampling model
   produces ~6 diverse coherent samples per turn to match its violations. No
   positives are duplicated or discarded to grow negatives.
2. **Exact trim** — `balance()` then undersamples whichever class is still the
   majority down to `--balance-ratio × minority` (default `1.0` = exact 1:1),
   deterministically (seeded). The dropped rows are cheap paraphrases, never
   positives when positives are the minority.

```bash
PYTHONPATH=src python -m toddc.cli generate --live --workers 8            # 1:1 (default)
PYTHONPATH=src python -m toddc.cli generate --balance-ratio 2            # 2:1 positive:negative
PYTHONPATH=src python -m toddc.cli generate --no-balance                 # raw, skewed
```

Two files are written: **`records.jsonl`** (balanced — the shipped set) and
**`records_all.jsonl`** (the full accepted set, nothing lost). The manifest's
`class_balance` block reports `positive` / `negative` / `majority_class` /
`dropped` / `balanced_total`, and `control_multiplier` records the N used.

## Library use

```python
from toddc.runners import ModelPool
from toddc.runners.factory import build_client
from toddc.generate import generate_seed

gen = ModelPool([build_client(s) for s in generator_specs], strategy="round_robin")
jud = ModelPool([build_client(s) for s in judge_specs], strategy="round_robin")
generate_seed(pool=gen, judge_pool=jud, workers=8, raw_dialogues=dialogues,
              balance=True, balance_ratio=1.0, control_multiplier="auto")
```
