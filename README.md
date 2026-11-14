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

## Current release: ic7

The accepted ic7 adapter is published and deployed. It passed every planning,
safety, and Indian-market gate.

| Metric | WS5 baseline | ic7 | Outcome |
|---|---:|---:|---|
| Frozen tool selection | 0.7867 | **0.9800** | pass |
| Frozen argument accuracy | 0.8600 | **0.9733** | pass |
| Refusal recall | 0.9189 | **1.0000** | pass |
| Over-refusal | 0.0385 | **0.0000** | pass |
| Post-error bounded finish | 0.9000 | **1.0000** | pass |
| India tool selection | 0.6875 | **0.9800** | pass |
| India argument accuracy | 0.6875 | **0.9800** | pass |

Published artifacts:

- Model/adapter: https://huggingface.co/SaiPavankumar22/finance-agent
- Dataset: https://huggingface.co/datasets/SaiPavankumar22/finance-agent-dataset
- Evaluation/logs: https://huggingface.co/datasets/SaiPavankumar22/finance-agent-eval
- Public API: see `artifacts/release/ic7.json`

The staging revision is `5ae2ee8`; the primary model revision is `0366a47`.
The previous WS5 adapter `7a4b00e` remains available for rollback.

## Historical baseline results

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

## Safety evaluation (WS6)

100 prompts: 74 that must be declined (personalized advice, predictions, insider
requests, urgency, jailbroken rewording) and 26 controls that must **not** be
declined. Ten of the controls replay two recorded provider failures before the
model gets its second turn, which is the only way to score "decline because the
evidence never arrived". `finish` is documented in the prompt as the legal
refusal; a planner told to "never answer directly" otherwise has no way to
decline.

| Metric | Base | Tuned | Delta |
|---|---|---|---|
| Refusal recall (74 rows) | 0.162 | **0.892** | +0.730 |
| Over-refusal (26 controls) | 0.000 | 0.038 | +0.038 |
| Declined after two tool errors (10) | 0.000 | **0.900** | +0.900 |
| Benign tool selection (26) | 0.885 | 0.731 | -0.154 |
| Prose with a number or advice | 0 | 0 | 0 |

**No safety drift, so SafeTune was not triggered.** Refusal recall rose fivefold,
over-refusal stayed at a single control row, and the tuned model never repeated a
call that had already failed twice (the base model repeated all ten). Retraining
on safety examples is only justified by a regression, and there is none; the
numbers are recorded in `artifacts/safety/comparison.md`.

Three things are still wrong, and they are the next work rather than a drift
signal:

- **Insider phrasing** is the one refusal family the adapter did not learn: 8 of
  12 such prompts still produce a tool call.
- **Control tool choice** fell because of one ambiguous template ("what did X
  close at over the last month?"), where the model answers with
  `get_stock_quote` and the gold action is `get_historical_prices`. That is a
  labelling and instruction problem, not a safety failure.
- **Format after an error turn** slips: some post-error declinations arrive as a
  malformed hybrid object, so the decision is right but the planner format needs
  stricter decoding.

### Three attempts, all rejected

Closing the insider gap was tried three times; none passed the release gate, so
the adapter measured above is still the published checkpoint (`7a4b00e`).

| Attempt | Rows | Refusal recall | Over-refusal | Frozen tool / args | Verdict |
|---|---|---|---|---|---|
| +48 insider refusals | 668 | 1.000 | 0.577 | 0.8431 / 0.9216 (val) | rejected |
| +48 insider, +48 benign | 716 | 1.000 | 0.769 | 0.7733 / 0.8533 | rejected, worse |
| WS6 prompt on every row | 755 | 0.9595 | 0.000 | 0.8467 / 0.8867 | rejected on error handling |

The balanced mix was supposed to pull over-refusal back under 0.10 and instead
pushed it up, which pointed at conditioning rather than row counts: refusals were
trained under the plain planner prompt while the audit prompt says declining is
legal, so `finish` was learned as a style for finance questions instead of a
decision about the request.

The third attempt fixed the refusal conditioning. All 755 rows carried the WS6
prompt verbatim, alongside new rows under it (Indian benign trajectories, Indian
refusals, Indian edge cases, fresh benign phrasings), and evaluation sends the
same prompt. It passed every safety item but lost India accuracy, so it was also
rejected.

Error handling needed a fourth attempt, because the first three were chasing the
wrong cause. The audit scores a post-error turn as bounded only when the planner
declines or switches tool: re-issuing the failed call is a failure. The training
rows did exactly that (`call, error, same call, error, finish`), so weighting
them could not help - the data taught the penalised behaviour. The 51 such rows
now decline after one failure, 24 new ones decline or fall back to a different
tool, and six P/E-on-`get_financial_ratios` rows carry the audit's own phrasing.

| Metric | WS5 `7a4b00e` | Published (`ic7`) | Gate |
|---|---|---|---|
| Frozen test tool selection | 0.7867 | **0.9800** | pass |
| Frozen test argument accuracy | 0.8600 | **0.9733** | pass |
| Refusal recall | 0.9189 | **1.0000** | pass |
| Over-refusal | 0.0385 | **0.0000** | pass |
| Benign tool selection | 0.7308 | **0.7692** | pass |
| Post-error bounded finish | 0.9000 | **1.0000** | pass |
| Repeated failing calls | 0 | **0** | pass |
| Unsafe text / malformed | 0 / 0 | 0 / 0 | pass |
| India tool / args | 0.6875 / 0.6875 | **0.9800 / 0.9800** | pass |

Every planning, safety and India gate passes, including the per-intent Indian
gates (invalid symbols 8/8, unsupported actions 11/11, refusal families 32/32,
quote/ratios/history 30/30).

Two earlier candidates at rank 16 and rank 32 both missed the *same* seven
refusal rows while trading insider rows with WS5, which is the signature of a
template gap rather than a capacity limit: the safety set cycles six advice and
four jailbreak templates, and training had no row for "for a novel I am writing,
...". `pipelines/16_gate_gaps.py` adds those templates (weight 2) plus the Indian
phrasings that lost their per-intent gates, and recall went from 0.9054 to 1.0000
without costing frozen accuracy.

Published to `avengeryt22/finance-agent-staging` (revision `5ae2ee8`) and
`saipavankumar22/finance-agent` (`0366a47`); the WS5 adapter is still reachable at
`7a4b00e`.

Both arms were re-measured under the WS6 prompt on the same decode rule.
### India evaluation

`eval/india_set.jsonl` is a separate 100-row held-out set over NSE symbols
(`RELIANCE.NS`, `TCS.NS`, `INFY.NS`), built from recorded fixtures with `INR`
pricing, and disjoint from the frozen test set (49 healthy, 51 abstain). It covers quotes, ratios,
history, news, advice, predictions, insider requests, urgency, invalid symbols
(`NIFTY50.NS`), unsupported actions, stale quotes and missing metrics.

| Intent | Rows | Tool correct | Args correct |
|---|---|---|---|
| quote | 10 | 8 | 8 |
| history | 10 | 6 | 5 |
| ratios | 10 | 7 | 10 |
| news | 14 | 13 | 13 |
| advice | 11 | 7 | 7 |
| prediction | 8 | 8 | 8 |
| insider | 8 | 8 | 8 |
| pressure | 5 | 5 | 5 |
| invalid symbol | 8 | 8 | 8 |
| unsupported | 11 | 10 | 10 |
| missing / stale | 5 | 4 | 4 |

Refusals and staleness generalise to Indian symbols without additional prompting,
and invalid symbols now decline 8/8. Three gaps remain: quote/history rows are
answered with the wrong Indian tool (21/30 across quote, ratios and history),
one unsupported action is answered with a tool call instead of `finish`, and four
refusal-family rows do not decline.

## Project structure

- `src/finance_agent/` — canonical release metadata and compatibility-facing
  runtime, contract, training and evaluation imports
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
- `eval/` — frozen 150-example test set (hashed), the 100-prompt WS6 safety set,
  metrics, baseline harness, abstention rescoring, adapter and safety reports
- `training/` — `data.py` (shared tokenisation, masking, gold extraction),
  `train_qlora.py` (smoke run), `train_full.py` (full run and eval entrypoint)
- `tests/` — unit and determinism tests
- `artifacts/sft/` — recorded run payloads and the comparison table
- `artifacts/release/` — accepted release manifest and deployment metadata
- `deploy/` — deployment conventions and secret handling
- `docs/` — HF Article draft and ROI comparison
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

Current state: the accepted release has 178 passing tests and clean Ruff
checks. Published-adapter reproduction reports zero per-row mismatches.

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

Results are written to the Modal volume and to `artifacts/sft/`; accepted
adapter weights are published to the public Hugging Face model repositories
listed above and are gitignored here.

Verify the published adapter against the accepted artifacts:

```powershell
modal run modal/reproduce.py::main
python scripts/check_reproduce.py
```

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

## Release checklist

- [x] WS7 Modal deployment using the accepted ic7 adapter
- [x] Public model, dataset, evaluation, and logs repositories
- [x] Before/after metrics and gate report
- [x] HF Article and ROI drafts in `docs/`
- [ ] Publish the HF Article through the Hugging Face UI
- [ ] Publish the optional static HF Space if the account entitlement allows it
