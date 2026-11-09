# Finance Agent

An open-source, small-footprint financial research agent built around
Cohere Labs' `Tiny Aya Global`. The project retrieves live market data,
normalizes it into strict schemas, records evidence, and rejects unsupported
financial claims instead of inventing figures.

## Current status

Completed:

- WS1: hardened tool envelope, evidence ledger, deterministic calculations,
  renderer, validator, retries, caching, staleness handling, circuit breaker,
  and tests.
- WS2: recorded fixtures, offline replay, deterministic failure injection, and
  a frozen 150-example English evaluation set.
- Tiny Aya Global inference smoke test through Hugging Face.

Validation currently passes:

```text
126 tests passed
ruff check: clean
fixture replay: deterministic and complete
```

The next milestone is WS3: run Tiny Aya Base, Tiny Aya Global, and the
Anthropic oracle on the frozen test set and produce baseline metrics.

## Project structure

- `app/schemas/` — strict tool and report schemas
- `app/tools/` — validated yfinance tools and offline replay
- `app/evidence/` — ledger, calculations, rendering, and grounding checks
- `pipelines/` — fixture recording, failure injection, and test-set generation
- `data/fixtures/` — committed provider snapshots
- `eval/` — frozen test set and metadata
- `tests/` — unit and determinism tests
- `scripts/` — baseline and model smoke-test scripts
- `project-plan.md` — complete hackathon execution plan
- `status.md` — current handover status

## Setup

Requires Python 3.11.

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add credentials locally. Never commit `.env`
or print tokens.

## Verification

```powershell
python -m pytest -q
python -m ruff check .
python pipelines/01_record_fixtures.py --verify
```

Regenerate deterministic failure cases with:

```powershell
python pipelines/02_inject_failures.py
```

Run a live yfinance baseline:

```powershell
python -m scripts.run_baseline --ticker AAPL --period 1mo
```

Run a small Tiny Aya Global inference request using the secondary Hugging Face
account:

```powershell
python scripts/test_tiny_aya.py
```

## Safety and grounding

The model is not treated as a source of current financial figures. Prices,
ratios, history, and news come from provider responses. Missing or stale values
remain unavailable, and every report must include source timestamps,
limitations, and a non-advisory disclaimer.

This project provides informational research only. It is not personalized
investment advice.

## Accounts and artifacts

- `avengeryt22/finance-agent-staging` — secondary-account training staging
- `SaiPavankumar22/finance-agent` — primary final artifacts
- `SaiPavankumar22/finance-agent-demo` — public static demo Space

Use the secondary Hugging Face credits for training and evaluation first.
Anthropic is reserved for development and synthetic-data generation. Modal is
the fallback for GPU training or serving when Hugging Face capacity or credits
are insufficient.
