# Finance Agent

An open-source, small-footprint financial research agent built on Cohere Labs'
`Tiny Aya Global` (3.35B). It retrieves live market data, normalises it into
strict schemas, records evidence, and refuses to state a figure it cannot
source.

## The rule the project is built around

> The model never writes numbers. It chooses tools and writes sentences with
> evidence placeholders. Code fills in the numbers and checks them.

A turn looks like this:

1. The model emits one JSON action: a tool call (`{"tool": ..., "args": {...}}`)
   or `finish`.
2. The harness runs the tool. Tools return a `ToolResult` envelope and never
   raise; failures come back as data.
3. Results enter an evidence ledger (`E1..En`, monotonic, nulls recorded).
4. The model writes the answer using `{{E1}}`-style placeholders, each naming
   one ledger entry and nothing else — no digits, no hedging about values.
5. The renderer substitutes each placeholder with a self-contained quantity
   (value, symbol, timestamp). The validator rejects any number, claim or
   citation that the ledger cannot support.

Because the numbers never pass through the model, a fluent-sounding wrong figure
is structurally impossible: the worst case is a refusal or a missing value.

## Results

QLoRA adapter (r 16, 2 epochs, best epoch 1) trained on 620 gold trajectories.
Base and tuned share one system prompt, one parser and one decode rule (greedy,
stopping at the first complete JSON object).

| Set | System | Tool selection | Arguments | Schema valid | Abstain recall |
|---|---|---|---|---|---|
| val, unseen tickers (51) | base | 0.5294 | 0.0588 | 1.0000 | 0.00 |
| val, unseen tickers (51) | tuned | **0.8235** | **0.8824** | 1.0000 | 0.40 |
| test, frozen (150) | base | 0.5400 | 0.0400 | 1.0000 | 0.00 |
| test, frozen (150) | tuned | **0.8333** | **0.8733** | 0.9933 | 0.46 |

Paired on the frozen set: tool selection gained 48 examples, lost 4. The base
arm reproduces the earlier hosted-provider baseline within 0.013 tool-selection,
which is why the before/after pair is trusted. Abstention precision is 1.00 and
specificity 1.00 — the model does not over-refuse, it under-refuses, and that is
the next workstream.

Full tables, per-intent breakdown and Wilson intervals: `artifacts/sft/comparison.md`.

## Project structure

- `app/schemas/` — strict tool, evidence and report schemas (pydantic v2)
- `app/tools/` — hardened yfinance tools: timeout, retries, TTL cache, per-tool
  circuit breaker, staleness handling, plus offline fixture replay
- `app/evidence/` — ledger, registered calculations, placeholder renderer,
  strict grounding validator
- `app/chat_format.py` — the one place that maps messages onto Aya's chat
  template, so training and inference cannot disagree
- `pipelines/` — numbered stages: record fixtures, inject failures, seed
  questions, gold trajectories, enrichment, curation, split
- `data/fixtures/`, `data/gold/`, `data/splits/` — committed snapshots, 800 gold
  trajectories, and the 620 / 51 split
- `eval/` — frozen 150-example test set (hashed), metrics, baseline harness,
  abstention rescoring, adapter comparison
- `training/` — `data.py` (shared tokenisation, masking, gold extraction),
  `train_qlora.py` (smoke run), `train_full.py` (full run and eval entrypoint)
- `tests/` — unit and determinism tests
- `artifacts/sft/` — recorded run payloads and the comparison table
- `project-plan.md` — the hackathon execution plan

## Setup

Requires Python 3.11.

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in credentials locally. Never commit
`.env` and never paste tokens into a shell you do not control.

## Verification

```powershell
python -m pytest -q
python -m ruff check .
python pipelines/01_record_fixtures.py --verify
```

Current state: `pytest -q` → 148 passed, `ruff check .` clean, fixture replay
deterministic and complete.

## Running the agent and the evaluation

Live tool baseline in the terminal:

```powershell
python -m scripts.run_baseline --ticker AAPL --period 1mo
```

A frozen-test evaluation of the base model (cached per example, so reruns are
cheap):

```powershell
python -m eval.run_baseline --models tiny-aya-global --limit 10
```

Training and adapter evaluation run on a GPU through Modal, because the model
does not fit locally in 4 bits:

```powershell
modal run training/train_full.py            # 620 rows, 2 epochs, saves the adapter
modal run training/train_full.py::eval_main  # base vs tuned under one decode rule
python -m eval.compare_adapter              # regenerate the table above
```

Results are written to the Modal volume and to `artifacts/sft/`; adapter weights
are published to a private Hugging Face model repo and are gitignored here.

## Evaluation design

- **Two held-out sets, two questions.** The 51 validation rows hold out whole
  tickers, so they measure generalisation to symbols the model never saw. The
  150 frozen examples provide the official before/after comparison and are
  hashed before and after every run.
- **Abstention is a binary decision,** scored with precision, recall,
  specificity and balanced accuracy plus Wilson intervals. Only a structured
  `finish` counts as declining, so a malformed answer is never credited as a
  refusal. Turn-one abstention has a low ceiling by construction (many refusals
  only make sense after a tool result), which is recorded with the metric.
- **Paired comparisons, not just averages,** so a gain cannot hide a regression
  on the same examples.

## Safety and grounding

Prices, ratios, history and news come from provider responses, not from the
model. Missing or stale values stay unavailable rather than being filled in.
Every report carries source timestamps, stated limitations and a non-advisory
disclaimer.

This project provides informational research only. It is not personalised
investment advice.

## Roadmap

- **WS6 — refusal behaviour.** A safety set of adversarial prompts with an
  explicit refusal affordance and a replayed tool trajectory, since turn-one
  abstention recall is currently 0.46.
- **WS7 — deployment.** A served endpoint, a demo Space with cached examples,
  and the published model card.
- Optional dataset work: merge the enrichment paraphrases and record the four
  missing price periods; `comparison` and `period_ambiguity` are the intents the
  adapter did not improve.
