# Finance Agent — Status

_Last updated: 2026-11-10._

## Where the project stands

WS1 (hardened foundation), WS2 (fixtures, failure injection, frozen test set),
WS3 (first-turn baseline evaluation), and WS4 data generation/curation are
complete locally. `ruff check .` is clean and `pytest -q` reports 129 passing
tests. The frozen evaluation set has 150-example results for Tiny Aya Global
and the Anthropic oracle; WS4 has 800 gold trajectories with no frozen-test
question overlap and 170 CuratorKIT-approved messages.

## Commit history

| Commit | Contents |
|---|---|
| `9b37baa` | Initial repository commit (schemas, validator, yfinance tools, helper scripts) |
| `86b1f21` | WS1: `ToolResult` envelope, evidence ledger, registered calcs, renderer, strict validator, hardened tools, 96 tests |
| `15319b7` | WS2: fixture recorder, offline replay, ten failure mutations, frozen 150-example test set |
| `3f0db0e` | English-only test set and templates |
| `f6a5292` | `project-plan.md` updated for the English-only decision |
| local | WS3 baseline evaluator, metrics, and results |

## Done — WS1: hardened foundation

| Path | What it does |
|---|---|
| `app/schemas/tools.py` | `ToolResult` envelope (`ok/unavailable/error/stale`), `ToolCall`; timezone-aware timestamps required; `ok` needs data, `error` needs a code |
| `app/schemas/report.py` | `Claim`, `MissingItem`, `ReportSkeleton` (model output), `RenderedClaim`, `ComposedReport`, `DISCLAIMER` |
| `app/evidence/ledger.py` | Stable `E1..En` IDs, entries carry value/unit/currency/as_of/source/derived_from/status; nulls recorded, never zeroed |
| `app/evidence/calcs.py` | Registry: `pct_change`, `diff`, `ratio`, `sma`; refuses null inputs, cross-currency maths and unit mismatches; derived `as_of` = oldest input; stale inputs propagate |
| `app/evidence/render.py` | `{{E7}}` → `189.23 USD` / `2.90T USD` / `12.50%` / quoted headline; ledger renders as a compact ID-first block |
| `app/evidence/validator.py` | Rejects digits outside placeholders, malformed/unknown/undeclared placeholders, unknown evidence, null evidence, stale evidence (policy), mixed currencies, mixed units |
| `app/tools/registry.py` | Four data tools + terminal `finish`; owns which fields become evidence and how they are styled |
| `app/tools/yfinance_tools.py` | Never raises; 8s timeout, 2 retries with backoff, TTL cache, per-tool circuit breaker, staleness downgrade, `invalid_ticker` / `malformed_response` classification |
| `app/tools/replay.py` | Offline fixture replay; missing recording → explicit `fixture_missing`, never a live call |

Rules worth remembering:

- The model never writes digits. Validation runs on the template, substitution after, so rendered digits can only come from placeholders.
- Percents are percentage points (`12.5` = 12.5%). Diffs keep unit and currency; ratios are dimensionless.
- Period enum is `1mo, 3mo, 6mo, 1y, 2y, 5y`; ambiguous questions default to `1y`.
- Staleness thresholds: quote 5 days, ratios 120 days, history 10 days, news 90 days. TTL cache: quote 60s, ratios 10m, history 24h, news 15m.
- Circuit breaker is **per tool** (5 consecutive outages, 300s cooldown) so one dead endpoint cannot blind the others.
- One `.info` payload per symbol is shared by the quote and ratio tools for 60s.

## Done — WS2: fixtures, injections, frozen test set

**Fixtures** (`data/fixtures/`, committed): 365 recorded responses + manifest,
73 tickers (48 US, 18 non-US, 7 ETF/ADR already in `data/seeds/tickers.json`),
history for `1mo` and `1y`, snapshot `2026-11-09`. Recorded status:
**276 ok / 69 unavailable / 20 error**; 69 tickers have a usable
quote + ratios + history. Content hash lives in `data/fixtures/manifest.json`.
The recorder is resumable (skips usable fixtures, retries failures, `--force`
re-records everything).

**Offline replay**: `python pipelines/01_record_fixtures.py --verify` replays
every fixture through the ledger with socket access disabled.
Result: `deterministic=PASS`, `manifest_complete=PASS`, digest `33397eb7…`.

**Failure injections** (`pipelines/02_inject_failures.py`): `null_ratio`,
`news_unavailable`, `stale_quote`, `provider_error`, `malformed`,
`currency_mismatch`, `unit_mismatch`, `conflict`, `invalid_ticker`,
`loop_trap`. Each is deterministic given a seed, leaves its input untouched and
carries an `expected_behavior` label. 300 sets generated across 30 source
tickers. `data/injected/` is gitignored because `make inject` regenerates it
byte-identically.

**Frozen test set** (`eval/testset.jsonl`): **150 examples**, 37 of them
expecting abstention, English only, SHA-256
`025c345dc028d457f99a1d9cf2f6c4c1b7069f72db166a3fe1ca41c5d2fd2e94`
(also in `eval/testset.sha256` and `eval/testset.meta.json`). Intent mix:
overview 30, metric_pe 15, comparison 15, missing_ratios 14, advice_request 14,
invalid_ticker 11, non_us_currency 11, malformed_error 8, news_unavailable 8,
stale_data 8, period_ambiguity 6, unsupported_metric 6, loop_trap 4.

Each row carries: `id`, `intent`, `language`, `tickers`, `period`, `mutation`,
`expected_behavior`, `question`, `gold_tools`, `expect_abstain`,
`fixture_status`. The generator refuses to overwrite a differing set without
`--force`; `--force` changes the hash and invalidates any recorded baseline.

## Decisions taken

1. **English only.** Questions, training data, the frozen test set, prompts and
   reports are all English. The plan's multilingual P1 item is cut (see
   `project-plan.md` §2 and §11.2).
2. **No SEC EDGAR yet** (WS8, P1), so the `conflict` intent is excluded from the
   frozen test set even though the mutation exists and is tested.
3. **Fixtures are committed, injected sets are not** — the latter are
   deterministic outputs of committed fixtures.
4. **`make` is not installed on this Windows host.** The Makefile is committed,
   but the canonical commands are the `python pipelines/…` forms below.
5. **News was down during recording.** 69 news fixtures are `unavailable`; that
   is recorded provider truth, not a bug. Overview examples are allowed to
   report news as missing.

## How to run

```bash
python -m pytest -q                                   # 126 tests
python -m ruff check .                                # lint
python pipelines/01_record_fixtures.py                # record (network, resumable)
python pipelines/01_record_fixtures.py --verify       # offline determinism check
python pipelines/02_inject_failures.py                # regenerate data/injected
python pipelines/03_make_seed_questions.py            # rebuild; needs --force to change
python -m scripts.run_baseline --ticker AAPL --period 1mo   # live smoke test
python -m eval.run_baseline --models tiny-aya-global,anthropic
```

## Done — WS3: first-turn baseline evaluation

`eval/run_baseline.py` evaluates the first tool decision on the frozen
150-example set and caches raw outputs. `eval/metrics.py` reports schema
validity, tool selection, argument accuracy, abstention accuracy, per-intent
scores, and tool distribution.

Results are saved in `eval/results/baseline.json`.

- Tiny Aya Global: schema validity `98.67%`, tool selection `55.33%`,
  argument accuracy `5.33%`, abstention accuracy `75.33%`.
- Anthropic oracle: schema validity `90.00%`, tool selection `54.00%`,
  argument accuracy `65.33%`, abstention accuracy `75.33%`.
- Tiny Aya Base: unavailable through the Hugging Face Cohere inference
  provider; a local or Modal serving path is required for a fair comparison.

These are first-turn planning metrics, not full multi-step agent performance.
The low Tiny Aya Global argument score is a useful training target: the model
often adds irrelevant arguments such as `period` to tools that do not accept it.

## Done locally — WS4: deterministic dataset curation

`pipelines/04_gold_trajectories.py` creates `data/gold/trajectories.jsonl`
from the real replay fixtures and rule-based intent plans. The generated set
contains 800 examples across all 13 intents, includes failure mutations, and
excludes every question in the frozen 150-example test set. Its manifest and
SHA-256 are in `data/gold/manifest.json`.

`pipelines/06_curate.py` runs CuratorKIT `0.1.23` with schema gating, MinHash
deduplication, cleaning, and messages export, followed by finance-specific
trajectory checks. CuratorKIT accepted 170 rows after its generic gates, and
all 170 passed the project-specific checks in `data/curated/gate_counts.json`.

`pipelines/05_llm_enrich.py` makes a cached Anthropic enrichment pass over 100
rows. It accepted 38 and rejected 62 under strict placeholder, grounding, and
safety checks; rejected rows are retained only as diagnostics, never training
data. The cache is ignored and the manifest is in `data/enriched/manifest.json`.

Tiny Aya Global's tokenizer was also checked locally: it has a native chat
template and ends the assistant prompt with `<|START_RESPONSE|>`. The Modal
GPU verification script is ready, but both T4 and L4 attempts require a payment
method before the function starts, so constrained decoding is not yet verified.

## Known gaps

- **Constrained decoding is unverified** (plan §10.3). Modal's GPU invocations
  are blocked until a payment method is added; the chat template itself is
  verified locally. The verifier accepts `MODAL_GPU` for trying another GPU.
- 4 of 73 tickers produced no usable data; 20 fixtures are hard errors.
- Missing entirely: `app/agent/`, `app/api/`, `app/config.py`, `deploy/`,
  `space/`, `docs/`, `configs/`, and the training pipeline.

## Next steps

1. Split the curated set by ticker/template and run contamination checks.
2. Verify vLLM JSON-schema decoding on Modal after billing is enabled.
3. WS5 fine-tuning, WS6 AuditKIT + SafeTune, WS7 deploy and publish.
