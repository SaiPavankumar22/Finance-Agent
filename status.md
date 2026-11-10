# Finance Agent — Status

_Last updated: 2026-11-10, after an audit of the WS3 baseline and the WS4
CuratorKIT dataset._

## Verdict

WS1 and WS2 are complete and committed. WS3's harness exists and two of three
models produced numbers, but the base model never ran and the abstention metric
is degenerate. WS4's gold set (800 deterministic trajectories) is good; the
**CuratorKIT curation is broken and the 170-example export must not be trained
on**. `ruff check .` is clean and `pytest -q` reports 129 passing tests.

## Commit history

| Commit | Contents |
|---|---|
| `9b37baa` | Initial repository commit |
| `86b1f21` | WS1: envelope, ledger, calcs, renderer, validator, hardened tools |
| `15319b7` | WS2: fixtures, offline replay, injections, frozen test set |
| `3f0db0e` | English-only test set and templates |
| `f6a5292` | Plan updated for English-only |
| `93b8e96` | Status document (WS1/WS2) |
| `a8980b4` | Project README |
| `1ca95f9` | WS3 baseline evaluation harness |
| `b4092d6` | WS4 gold trajectories |
| `42f6d72` | Modal verifier aligned with docs |
| `7be6979` | CuratorKIT integration + Anthropic enrichment |
| `5232f73` | Configurable Modal GPU |
| `2e5f300` | Tiny Aya vLLM decoding verified on Modal |
| `1ca432c` | WS4 docs update |
| `HEAD` | This corrected status |

## WS1 — hardened foundation (done)

`ToolResult` envelope, evidence ledger (`E1..En`, monotonic, nulls recorded),
registered calcs (`pct_change`, `diff`, `ratio`, `sma`), placeholder renderer,
strict validator, five-tool registry, hardened yfinance layer (8s timeout, 2
retries, TTL cache, per-tool breaker, staleness downgrade, never raises), and
offline replay. The model never writes digits: validation runs on the template,
substitution afterwards.

Rules: percents are percentage points; periods `1mo,3mo,6mo,1y,2y,5y` with `1y`
default; staleness quote 5d / ratios 120d / history 10d / news 90d.

## WS2 — fixtures, injections, frozen test set (done)

365 fixtures, 73 tickers, `1mo` + `1y`, snapshot 2026-11-09; 276 ok / 69
unavailable / 20 error; 69 tickers fully usable. Offline replay verified
(`deterministic=PASS`, digest `33397eb7…`). Ten deterministic mutations, 300
injected sets (gitignored, regenerable). Frozen test set: 150 examples, 37
expecting abstention, English only, SHA-256
`025c345dc028d457f99a1d9cf2f6c4c1b7069f72db166a3fe1ca41c5d2fd2e94`.

## WS3 — baselines (partial, two problems)

`eval/run_baseline.py` + `eval/metrics.py` exist; `eval/results/baseline.json`
holds 150 predictions per model:

| Model | Examples | Schema valid | Tool selection | Argument acc. | Abstention |
|---|---|---|---|---|---|
| `CohereLabs/tiny-aya-base` | 0 | — | — | — | — |
| `CohereLabs/tiny-aya-global` | 150 | 0.9867 | 0.5533 | 0.0533 | 0.7533 |
| `anthropic` (oracle) | 150 | 0.9000 | 0.5400 | 0.6533 | 0.7533 |

**Problem 1: the base model never ran.**
`Model CohereLabs/tiny-aya-base is not supported by provider cohere` — the run
used a Cohere provider path that does not host the Base checkpoint. There is
therefore no pre-training reference point for the fine-tuning delta.

**Problem 2: `abstention_accuracy` is degenerate.** Both models score exactly
`0.7533`, which equals `1 - 37/150`: the rate you get by never abstaining. The
metric is plain accuracy on an imbalanced label, so it hides total abstention
failure. It needs to be reported as recall/precision on the abstain class (plus
a "did it abstain when it should not have" count).

Context for the decision gate: Global's tool selection at 55.3% with argument
accuracy 5.3% is *not* high, so SFT is justified — but the gate cannot be closed
until Base runs.

## WS4 — gold trajectories (done) and curation (broken)

**Gold set** (`data/gold/trajectories.jsonl`): 800 deterministic trajectories
over 68 tickers, all 13 intents, ending in `finish`, ≤6 calls, no duplicate
calls, reports built by code with `{{E<n>}}` placeholders.

| Intent | Gold | Kept by CuratorKIT |
|---|---|---|
| comparison | 110 | 74 |
| overview | 94 | 60 |
| metric_pe | 93 | **0** |
| advice_request | 87 | 25 |
| period_ambiguity | 81 | **0** |
| unsupported_metric | 81 | **0** |
| non_us_currency | 69 | 11 |
| missing_ratios | 44 | **0** |
| news_unavailable | 40 | **0** |
| stale_data | 38 | **0** |
| malformed_error | 33 | **0** |
| loop_trap | 26 | **0** |
| invalid_ticker | 4 | **0** |

**What CuratorKIT actually did** (`06_curate.py`, `format="auto"`): it read each
trajectory as a single-turn pair — `instruction` = the user question,
`output` = one tool call — then applied a minimum-length gate to that pair.
All 538 rejections are `SchemaGate / below_min_tokens:N`, and `N` equals
`tokens(instruction) + tokens(output)` in **538 of 538** cases, with `output`
always exactly 1 token. So the gate measured question phrasing length, not
trajectory quality.

Consequences:

1. The 170 survivors are an artifact of long question templates (overview,
   comparison, advice, non-US). **Nine of thirteen intents are at zero**,
   including every abstention, missing-data, stale, malformed and efficiency
   category — the exact behaviours the project exists to test.
2. The export keeps **only `messages`**. `intent`, `tickers`, `period`,
   `mutation` are gone, so the plan's split-by-ticker-group and
   per-(intent, ticker) caps cannot be computed from the file. Labels are
   recoverable: all 170 question texts match gold exactly.
3. The 38 accepted enrichment examples (of 100 attempted) are a separate
   artifact and are **not** in the curated set; the 170 contain no paraphrases.
4. Provenance is inconsistent: `README.md`/`dataset_card.md` advertise
   `sha256 29e1614…` while the file hashes `e0370159…`;
   `validated_messages.jsonl` differs from `sft_messages.jsonl` only by key
   ordering; `checksums.txt` omits both validated and the dataset card. Nothing
   pins `-text` for these `.jsonl` files, so Windows CRLF conversion will drift
   every hash on clone.

Good news: question overlap with the frozen test set is **0**, and gold
trajectories are internally consistent (project gates accepted 170/170 of what
CuratorKIT passed, and rejected nothing else).

## Decisions taken

- **English only** (plan §2, §11.2).
- Fixtures committed; `data/injected/` gitignored as a deterministic derivative.
- `make` is unavailable on this Windows host; `python pipelines/…` is canonical.
- Provider news was down during recording → 69 news fixtures `unavailable`.

## Known gaps

- Base baseline missing; abstention metric must be redefined.
- Curated dataset unusable as-is: 9/13 intents empty, no metadata.
- No train/val/test split, no ticker-group contamination check, no
  per-(intent, ticker) cap.
- Enrichment got 38/100 through validation and is unmerged.
- No training run, no AuditKIT/SafeTune results, no deployment, no git remote.

## Next steps

1. **Fix `06_curate.py`**: feed CuratorKIT conversation-shaped records (or
   disable its length gate, min_tokens 0), keep `id` + `metadata` through to the
   export, and let the project gates do schema/behaviour checks. Target 700–800
   kept; a 67% drop is the plan's stated signal of an upstream bug.
2. **Merge enrichment**: 38 accepted paraphrase/interpretation records into the
   pool, keyed by gold `id`.
3. **`07_split_and_format.py`**: hold out ~15 tickers and ~2 templates per
   intent; assert zero train/test overlap on (question, ticker-group-intent).
4. **Re-run Base**: use the HF/Modal path, not the Cohere provider.
5. Manual review: 50 random examples plus 5 per hard category, recorded in the
   dataset card.
6. Only then WS5 (LoRA dry run).
