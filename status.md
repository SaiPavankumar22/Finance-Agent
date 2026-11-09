# Finance Agent — Status

_Last updated: 2026-11-09._

## Where the project stands

WS1 (hardened foundation) and WS2 (fixtures, failure injection, frozen test set)
are complete and committed. `ruff check .` is clean and `pytest -q` reports 126
passing tests. The frozen evaluation set exists, so training data can now be
built against a stable target. **No model has been run yet**: there are no
baselines, no training data, no adapter, no eval metrics and no deployment.

## Commit history

| Commit | Contents |
|---|---|
| `9b37baa` | Initial repository commit (schemas, validator, yfinance tools, helper scripts) |
| `86b1f21` | WS1: `ToolResult` envelope, evidence ledger, registered calcs, renderer, strict validator, hardened tools, 96 tests |
| `15319b7` | WS2: fixture recorder, offline replay, ten failure mutations, frozen 150-example test set |
| `3f0db0e` | English-only test set and templates |
| `f6a5292` | `project-plan.md` updated for the English-only decision |

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
```

## Known gaps

- **No model inference yet**: no baselines, no `eval/metrics.py`, no
  `eval/results/baseline.json`, so the WS3 decision gate is still open.
- **Tiny Aya chat template and constrained decoding are unverified** (plan
  §10.1 and §10.3). Both decide the training format and can invalidate a day of
  work, so they should be checked before WS4.
- 4 of 73 tickers produced no usable data; 20 fixtures are hard errors.
- No git remote yet — the submission requires a public repository.
- Missing entirely: `app/agent/`, `app/api/`, `app/config.py`, `deploy/`,
  `space/`, `docs/`, `configs/`, `eval/metrics.py`, training pipeline.

## Next steps

1. **WS3** — score Tiny Aya Base, Tiny Aya Global (prompted) and the Claude
   oracle on the frozen set over fixtures; save `eval/results/baseline.json`;
   apply the decision gate (if prompted Global already scores high on tool calls,
   shift fine-tuning emphasis to efficiency, abstention and grounding).
2. Verify the chat template + constrained decoding path on Modal.
3. **WS4** — deterministic gold planner, Claude enrichment, CuratorKIT gates,
   splits by ticker group and template.
4. WS5 fine-tuning, WS6 AuditKIT + SafeTune, WS7 deploy and publish.
