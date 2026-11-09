# Finance Agent: Project Plan

An open-source, small-footprint, evidence-grounded financial research agent built on **CohereLabs/tiny-aya-global**.

**Hard time limit: about 3 working days.** This plan is organized by **priority tier and workstream**, not by calendar day. Work top to bottom. If time runs short, stop at the tier boundary and ship what you have. A working public demo with honest before/after numbers beats an ambitious pipeline that is not deployed.

---

## 0. How to read this plan

| Tier | Meaning | Rule |
|---|---|---|
| **P0** | Required for a valid submission | Must be finished |
| **P1** | Makes the submission strong | Do if P0 is done and healthy |
| **P2** | Nice to have | Only with spare time and credits |

Rough time budgets (in hours) are guidance for planning, not a schedule.

**The one design rule that makes this project tractable:**
> The model never writes numbers. It chooses tools and writes sentences with evidence placeholders. Code fills in the numbers and checks them.

This is what makes the "every numerical claim is mechanically traceable or explicitly rejected/marked unavailable" claim defensible, and it also makes dataset preparation far easier (see section 11).

---

## 1. Goal and claims

**Goal:** accept an investor question, call validated finance tools, and return a structured, source-timestamped equity research report that never invents figures and abstains on missing, stale, invalid, or conflicting data.

**Claims we will make (and can defend):**
1. Every numerical value in a report is traceable to a timestamped tool response or a registered deterministic calculation. Anything else is removed or marked unavailable.
2. Fine-tuning Tiny Aya Global improves tool selection, argument accuracy, efficiency, and abstention on a frozen test set, with confidence intervals.
3. Safety/non-advisory behavior does not degrade after fine-tuning.
4. The self-hosted model costs X per 1,000 requests versus Y for a closed-model API (measured, with assumptions stated).

**Claims we will NOT make:** "zero hallucinations", "financial advice", or anything about live-market accuracy beyond the provider's own data.

---

## 2. Decisions already made (do not reopen)

- **Submitted model:** `CohereLabs/tiny-aya-global` (3.35B, 8K context), fine-tuned with LoRA/QLoRA.
- **Control:** `CohereLabs/tiny-aya-base` (prompted only; optionally also SFT'd as a control if time allows, P2).
- **Comparison:** `CohereLabs/aya-expanse-8b` (P2; not recommended for tool use by its own card).
- **Regional variants (Earth/Water/Fire):** skipped.
- **English only.** Seed questions, training data, the frozen test set, prompts and reports are all English. The old P1 multilingual item is cut, so prompt format, tool calls and report prose share exactly one language.
- **Anthropic:** development only (synthetic data, oracle baseline). Never a runtime dependency.
- **No training from scratch.**
- **Data sources:** yfinance first, SEC EDGAR as the second provider (P1).

---

## 3. Working setup: Cursor + Freebuff

You have two AI coding tools. Use them for different jobs so you do not waste the stronger one on boilerplate.

### 3.1 Model routing

| Task type | Use | Why |
|---|---|---|
| Boilerplate, CRUD, pydantic models, FastAPI endpoints, tests, README text, docstrings, repetitive scripts | **Freebuff (fast/cheap model)** | Cheap and quick, correctness is easy to verify |
| Validator logic, evidence ledger, training configs, data-generation logic, eval metrics, anything where a subtle bug silently corrupts results | **Cursor (stronger model)** | Bugs here ruin the whole project |
| Reviewing the other tool's output on critical files | **Cursor** | Second pair of eyes |

I do not know these specific models' limits, so apply this rule: **anything in `app/evidence/`, `pipelines/`, `eval/` gets written or reviewed by the stronger model and covered by unit tests.**

### 3.2 Repo-level instructions (put in both tools)

Create `AGENTS.md` at the repo root (Cursor reads it; point Freebuff at it too), and mirror it in `.cursor/rules/project.mdc`:

```markdown
# Project rules
- Python 3.11. Type hints everywhere. pydantic v2 for all schemas.
- Never print, log, or commit secrets. Read secrets only from environment variables.
- Never open or read `.env` contents into chat. Use `os.environ` / `python-dotenv`.
- The model never writes numbers in report text; numbers come from evidence placeholders `{{E<n>}}`.
- Tools return a `ToolResult` envelope; they never raise into the agent loop.
- Every pipeline stage is one script in `pipelines/`, reads/writes files in `data/`, and is deterministic given a seed.
- Add a unit test for every function in `app/evidence/` and `eval/metrics.py`.
- Do not add dependencies without adding them to `requirements.txt` with a pinned version.
- Small commits. Run `ruff check` and `pytest -q` before finishing a task.
```

### 3.3 Protect secrets from the editors

Add a `.cursorignore` (and equivalent ignore for Freebuff) containing:

```
.env
.env.*
*.key
artifacts/private/
```

Also keep `.env` in `.gitignore`. Do not paste tokens into chat. If a key was ever pasted into a chat or committed, rotate it.

### 3.4 Working loop per task

1. Write the task as a short spec: inputs, outputs, acceptance test.
2. Give the spec to the appropriate tool (section 22 has copy-paste prompts).
3. Run tests yourself. Do not accept "it should work."
4. Commit. Move on.

---

## 4. Credentials and accounts

| Credential | Where it lives | How code gets it |
|---|---|---|
| Secondary HF token (`avengeryt22`) | `.env` as `HF_SECONDARY_TOKEN` | `load_dotenv()` then `os.environ["HF_SECONDARY_TOKEN"]` |
| Primary HF (`SaiPavankumar22`) | Browser-authenticated login, stored by the Hugging Face CLI | Default `huggingface_hub` behavior (no token passed), or `HF_PRIMARY_TOKEN` if you export one |
| Anthropic key | Shell environment | `os.environ["ANTHROPIC_API_KEY"]` |
| Modal | `modal token new` (browser) | Modal CLI stores it |

**Important gotchas:**
- Two Hugging Face identities in one process is a classic source of "wrote to the wrong account." Always pass `token=` **explicitly** to `HfApi(token=...)` and `snapshot_download(..., token=...)` rather than relying on the default.
- Create a tiny helper `app/config.py` with `hf_secondary()` and `hf_primary()` that return configured `HfApi` objects and **print only the username** (via `whoami`) on first use. Use it everywhere.
- Gated model access must be accepted on **both** accounts (you already verified this).
- HF Jobs run in a remote container, so the token must be passed as a Job secret, not read from your local environment. Check `hf jobs --help` / current HF Jobs docs for the exact secrets flag (I am not certain of the current CLI syntax).

---

## 5. Architecture

```
User
 -> Static HF Space (UI)
 -> Modal FastAPI service  (auth, rate limit, request IDs, structured logs)
 -> Agent runner
      Stage 1: Tiny Aya Global emits tool calls (JSON, schema-constrained)
         -> tools: yfinance quote / ratios / history / news, SEC facts
         -> every field enters the Evidence Ledger with an ID (E1, E2, ...)
      Stage 2: Tiny Aya Global emits a report as claims with {{E<n>}} placeholders
 -> Composer: substitutes ledger values into placeholders
 -> Deterministic validator: rejects digits not coming from ledger/calcs
 -> Structured report
      (retrieved facts | calculations | news | interpretation | missing/stale | limitations | disclaimer)
```

Why two stages: a 3B model is much more reliable doing "pick tools" and "write prose about listed evidence" as separate narrow tasks than doing both in one long generation.

---

## 6. Repository structure

```
finance-agent/
  AGENTS.md
  .cursor/rules/project.mdc
  .env.example
  .gitignore  .cursorignore
  requirements.txt
  Makefile
  app/
    config.py
    schemas/        finance.py  tools.py  report.py
    tools/          yfinance_tools.py  sec_tools.py  registry.py
    evidence/       ledger.py  validator.py  calcs.py  render.py
    agent/          runner.py  prompts.py  compose.py  decoding.py
    api/            main.py  auth.py  ratelimit.py
  data/
    seeds/          tickers.json  question_templates.yaml  intents.yaml
    fixtures/       <ticker>_<tool>.json  (recorded real responses)
    injected/       fixtures with deliberate failures
    raw/  curated/  splits/
  pipelines/
    01_record_fixtures.py
    02_inject_failures.py
    03_make_seed_questions.py
    04_gold_trajectories.py
    05_llm_enrich.py
    06_curate.py
    07_split_and_format.py
    08_train.py
    09_eval.py
    10_publish.py
  configs/          curator.yaml  agenttune.yaml  aligntune.yaml  safetune.yaml  audit.yaml
  eval/             testset.jsonl  safety_set.jsonl  metrics.py  run_eval.py
  deploy/modal_app.py
  space/            index.html  app.js  examples.json
  docs/             model_card.md  dataset_card.md  roi.md  article.md
  tests/
  scripts/          (your existing baseline + smoke-test scripts)
```

Your existing files (`app/schemas/finance.py`, `app/tools/yfinance_tools.py`, `app/evidence/validator.py`, the scripts) slot into this layout. Refactor, do not restart.

---

## 7. Workstreams in priority order

### WS1 (P0, ~4h): Harden the foundation
- Extend schemas with the `ToolResult` envelope (section 8).
- Evidence ledger with stable IDs; calculation registry; placeholder renderer.
- Validator that rejects any number not from the ledger or a registered calc.
- Timeouts, retries, TTL cache, `status` values (`ok | unavailable | error | stale`).
- **Acceptance:** unit tests for 20+ cases (null ratio, unit mismatch, stale quote, malformed tool output, digit smuggled into prose). All pass.

### WS2 (P0, ~3h): Fixtures and the frozen test set
- Record real tool responses for a ticker pool into `data/fixtures/`.
- Generate failure variants into `data/injected/` (section 11.4).
- Build `eval/testset.jsonl` (~120-150 examples) **before** any training data. Freeze it with a hash.
- **Acceptance:** `make eval-fixtures` replays fixtures deterministically (no network).

### WS3 (P0, ~3h): Baselines
- Run Tiny Aya Base, Tiny Aya Global (prompted, with constrained decoding if available), and the Anthropic oracle on the test set.
- **Early decision gate:** if Global prompted already scores very high on tool calls, shift the fine-tuning emphasis to efficiency, abstention, and report-composition quality. Do not hide this; report it.
- **Acceptance:** a baseline metrics table saved to `eval/results/baseline.json`.

### WS4 (P0, ~8h): Dataset (the part you are worried about; see section 11)
- Deterministic gold trajectories + LLM enrichment + CuratorKIT filtering + splits.
- **Acceptance:** `data/splits/{train,val}.jsonl` plus a dataset card; 100% pass the validator; manual review of a sample done.

### WS5 (P0, ~5h): Fine-tuning
- AgentTune/AlignTune SFT with QLoRA on Tiny Aya Global (section 12).
- **Acceptance:** adapter pushed to the staging repo; training loss curve logged; the eval set improves over baseline on at least tool accuracy or efficiency.

### WS6 (P0, ~3h): Evaluation and safety
- AuditKIT final eval; SafeTune before/after audit (section 13-14).
- **Acceptance:** `eval/results/final.json`, safety delta table.

### WS7 (P0, ~5h): Deploy and publish
- Modal inference endpoint, static Space with cached examples, HF repos populated, Article written (sections 15-16).
- **Acceptance:** a stranger can open the Space link and get a report; the Article links every artifact.

### WS8 (P1, ~3h): SEC EDGAR second provider and conflict handling
### WS9 (P1, ~2h): ROI measurement (section 17)
### WS10 (P1, ~2h): Preference optimization (DPO) on residual errors
### WS11 (P2): Aya Expanse 8B comparison, Base+SFT control, CircuitKIT, regional variants

**Cut lines:** if behind, drop WS10, then WS8, then WS9 narrative depth. Never drop WS2, WS3, WS6, WS7.

---

## 8. API and schema design

### 8.1 Tool envelope

```python
class ToolResult(BaseModel):
    status: Literal["ok", "unavailable", "error", "stale"]
    tool: str
    args: dict
    data: dict | None
    source: str                  # "yfinance" | "sec_edgar"
    retrieved_at: datetime
    as_of: datetime | None       # timestamp of the underlying market/filing data
    error: str | None
```

Rules:
- Numeric fields are `float | None` (never strings). Missing means `None`, never `0`.
- Every monetary field carries `currency`; every ratio/volume field carries `unit`.
- Tools never raise into the agent. Failures become `status="error" | "unavailable"`.

### 8.2 The four required tools

1. `get_stock_quote(ticker)`: price, currency, market cap, volume, 52w low/high, retrieved_at
2. `get_financial_ratios(ticker)`: P/E, P/B, EV/EBITDA, debt-to-equity, ROE (null if missing), source timestamp
3. `get_historical_prices(ticker, period)`: OHLC, SMA-20, SMA-50, EMA-20, annualized volatility, adjustment method, retrieved_at
4. `get_company_news(ticker, limit)`: headline, publisher, URL, published timestamp, or explicit `unavailable`

Add (P1): `get_sec_facts(ticker_or_cik, concept)`.
Add a terminal pseudo-tool `finish` so "I'm done calling tools" is an explicit, learnable action.

`period` is an enum: `1mo, 3mo, 6mo, 1y, 2y, 5y`. Ambiguous periods in the question ("recently") must map to a default or trigger a clarification/abstention behavior; decide and document the rule.

### 8.3 Evidence ledger

```json
{"id": "E7", "tool_call_id": "c2", "path": "quote.price", "value": 189.23,
 "unit": null, "currency": "USD", "as_of": "2026-11-09T09:30:00Z", "source": "yfinance"}
```

- Calculations are registered functions (`pct_change`, `ratio`, `diff`, `sma`, ...) taking ledger IDs and producing new entries with `derived_from: ["E7","E9"]`.
- The ledger is rendered to the model as a compact list, for example: `E7 price=189.23 USD as_of=... [yfinance]`. Short and ID-first, to save tokens.

### 8.4 Report format (what the model emits in stage 2)

```json
{
  "claims": [
    {"type": "fact", "text": "Price is {{E7}} {{E8}} with a market cap of {{E9}}.", "evidence": ["E7","E8","E9"]},
    {"type": "calculation", "text": "Price is {{E21}} above its 52-week low.", "evidence": ["E21"]},
    {"type": "interpretation", "text": "The ratios suggest a premium valuation relative to the leverage shown.", "evidence": ["E11","E14"]}
  ],
  "missing": [{"item": "news", "reason": "provider_unavailable"}],
  "limitations": ["Data from a single provider.", "No forward estimates retrieved."]
}
```

**Rule:** claim text contains no digits except inside `{{E<n>}}` placeholders. The validator enforces this. This is what turns "the model must not hallucinate numbers" from a hope into a check.

### 8.5 HTTP API
- `POST /v1/report` body `{question, request_id?}` returns the report, plus `model_version`, `dataset_version`, `code_sha`, and `source_timestamps`
- `GET /health`: model loaded and provider reachability
- `GET /version`
- Header `X-API-Key`; per-key rate limit; global daily budget kill switch; CORS restricted to the Space origin.

---

## 9. yfinance integration

- Wrap calls with an 8s timeout, 2 retries with backoff, TTL cache (quotes 60s, history 24h, news 15m).
- Normalize into the schemas; return `None` for missing fields.
- Staleness rule: if `as_of` is older than a threshold (documented), return `status="stale"`.
- Invalid ticker detection: empty or minimal-info responses become `status="error", error="invalid_ticker"`.
- yfinance is unofficial and can break or rate-limit. Therefore: **evaluation and demo-fallbacks use recorded fixtures**, and only the live demo path hits Yahoo.
- Circuit breaker: after N consecutive provider failures in a window, return an explicit "provider unavailable" report instead of partial data. This is your "kill switch."
- Attribute data sources in every report and note the provider terms in the model card.

---

## 10. Tool-calling strategy for Tiny Aya Global

1. **Verify before building on it.** Test how Tiny Aya Global's chat template handles tools and a tool-result role. Run `tokenizer.apply_chat_template` on a short multi-turn example and read the output. If there is no tool role, wrap tool results in a user-turn block (for example `[TOOL_RESULT call_id=c1] ... [/TOOL_RESULT]`) and train with the same format. Decide this once and use the same format for training and inference.
2. **Use a simple, fixed JSON format:** `{"tool": "get_stock_quote", "args": {"ticker": "AAPL"}}`, one call per assistant turn.
3. **Constrained decoding at inference** (JSON-schema guided generation). Try vLLM guided decoding on Modal; if Tiny Aya's architecture is not supported by your vLLM version, fall back to transformers plus a JSON-schema grammar library, and as a last resort, validate-and-retry with a retry cap. I am not sure which of these works with this model; check early.
4. **Loop limits:** max 6 tool calls, duplicate-call rejection (identical tool+args), invalid-call counter that aborts after 2.
5. **Small tool surface** (5-6 tools) and short descriptions. Small models degrade quickly with long tool lists.
6. Training teaches **judgment** (which tools, in what order, when to abstain or stop). Grammar handles **format**.

---

## 11. Dataset preparation for SFT (detailed)

This is the section to read slowly. The fear is reasonable, but the task is more mechanical than it looks if you follow one principle:

> **Make the correct answer computable.** Where possible, generate the "gold" trajectory with code, not with an LLM. Use the LLM only where code cannot produce variety (question phrasing and interpretive sentences), and then check its output with code.

### 11.1 What you are teaching the model

Two narrow skills, each trained as chat-format examples:

1. **Planning/tool use:** given a question (plus prior tool results), emit the next tool call, or `finish`.
2. **Report composition:** given the ledger, emit the claims JSON with placeholders, `missing`, and `limitations`, or an abstention/refusal.

You are *not* teaching finance knowledge, and you are *not* teaching the model to know prices.

### 11.2 Target size and mix

For a 3-day project, aim for **~800-1,000 curated examples** (about 1,500 raw before filtering). Small, clean, and diverse beats large and noisy.

| Category | Approx. share | Gold behavior |
|---|---|---|
| Single-company overview (valid US ticker) | 20% | quote + ratios + history + news, then finish |
| Targeted metric question (e.g. "what's the P/E of X?") | 10% | only the needed tool(s), then finish (teaches efficiency) |
| Multi-company comparison (2-3 tickers) | 10% | right calls per ticker, currency/unit alignment, comparison claims via calcs |
| Non-US tickers / currency differences | 7% | correct ticker format, currency attached, no cross-currency arithmetic without flag |
| Invalid / ambiguous ticker | 7% | try once, then abstain with clear reason (no guessing) |
| Missing / null fields (e.g. ratios null) | 8% | report `missing`, no fabrication |
| News unavailable | 5% | report `news unavailable`, still deliver the rest |
| Stale data | 5% | mark stale, caveat or abstain per rule |
| Conflicting providers (yfinance vs EDGAR) | 4% | surface conflict, no silent pick (P1) |
| Malformed / error tool responses | 5% | retry once or abstain; never invent |
| Historical period ambiguity | 4% | apply documented default, state it |
| Unsupported metric requests | 4% | say unsupported, offer closest available |
| Personalized advice / guaranteed-return requests | 8% | refuse advice/predictions, offer factual analysis, include disclaimer |
| Loop / redundancy traps (tool returns same thing) | 3% | stop after bounded attempts |

**English only.** Questions, tool calls, reports and refusals are English; the non-English category (previously P1) is dropped. Removing the language dimension eliminates a whole class of prompt/format mismatch between training and inference and shrinks the labelling surface. The frozen test set in `eval/testset.jsonl` is English exclusively and the training data must match it.

### 11.3 The pipeline, step by step

**Step A: Record real fixtures** (`01_record_fixtures.py`)
- Pick ~80 tickers: ~50 US (mix of large/mid/small cap, several sectors), ~20 non-US (e.g. `.NS`, `.L`, `.T`, `.DE` suffixes), a few ETFs/ADRs as edge cases.
- For each, call every tool once and save the **normalized `ToolResult` JSON** to `data/fixtures/`. This is the real-world-looking input your examples are built on.
- Record the retrieval date in a manifest. Fixtures are a frozen snapshot; that is the point.

**Step B: Inject failures with code** (`02_inject_failures.py`)
Deterministically mutate fixtures to produce adversarial variants. Each mutation is a function with a name, so the gold behavior is known:
- `null_ratio`: set one or more ratio fields to `None`
- `news_unavailable`: replace news with `status="unavailable"`
- `stale_quote`: push `as_of` back past the staleness threshold
- `provider_error`: `status="error"`, with an error string
- `malformed`: truncated/garbled JSON string as the tool output
- `currency_mismatch`: two tickers with different currencies in a comparison
- `unit_mismatch`: mismatched volume/market-cap scale flagged in `unit`
- `conflict`: EDGAR value differing from yfinance by more than a tolerance
- `invalid_ticker`: ticker with no data
- `loop_trap`: provider returns the same unhelpful error repeatedly

You never need an LLM to invent a failure. This is the biggest source of risk reduction.

**Step C: Seed questions** (`03_make_seed_questions.py`)
- Templates per intent in `data/seeds/question_templates.yaml` (for example 8-15 phrasings per intent), crossed with tickers and periods.
- Hand-write ~40 questions yourself for the hard categories (ambiguity, advice requests, multi-company). This is the cheapest high-value human work in the project.
- Store each seed with `{id, intent, tickers, period, mutation, expected_behavior, language}`.

**Step D: Deterministic gold trajectories** (`04_gold_trajectories.py`)
A rule-based planner maps `intent + fixture state` to the exact correct trajectory:

```
overview            -> quote, ratios, history(period), news, finish
metric_pe           -> ratios, finish
comparison(A,B)     -> quote(A), quote(B), ratios(A), ratios(B), finish
invalid_ticker      -> quote(T) -> error invalid_ticker -> finish(abstain)
advice_request      -> (no tools or optional quote) -> finish(refuse_advice)
```

The planner then runs the real tool layer against fixtures, builds the ledger, and fills the report skeleton: `fact` and `calculation` claims written from templates with `{{E<n>}}` placeholders; `missing` and `limitations` filled from tool statuses; disclaimer added by the composer. These parts are **100% correct by construction**.

**Step E: LLM enrichment** (`05_llm_enrich.py`, Anthropic)
Use Claude only for what code cannot do well, with the ledger as input and strict output format:
1. **Paraphrase user questions** (diversity of tone, typos, informal/formal). Input: seed question and intent; output: N variants. English only. Verify by code that tickers/periods are preserved.
2. **Write 1-2 `interpretation` sentences per report** using only placeholders. Prompt: "Write cautious, non-advisory interpretive sentences referencing evidence only via `{{E<n>}}`. Do not write any digits. Do not recommend buying or selling."
3. **(Optional, P1)** Write varied English refusals for advice requests (shorter, firmer, more explanatory). No translations: the project is English only.

Then **verify by code**: no digits outside placeholders, every placeholder exists in the ledger, no advice phrases (keyword plus classifier check), length bounds. Failures are dropped, not "fixed."

Model choice: use your cheapest suitable Claude model for bulk paraphrase (it is simple), and a stronger one only for the interpretation sentences if quality is poor. Configure via `CLAUDE_MODEL` env var. Estimated spend for ~1,500 examples is a small fraction of $100, but log token usage per run so you can see it.

Check Anthropic's current terms on using outputs to train other models before you publish the dataset. I have not verified the current wording.

**Step F: CuratorKIT filtering** (`06_curate.py`)
Wire your gates through CuratorKIT. Read its README first (I have not seen its API) and map these gates onto whatever it exposes; write thin adapter functions for anything it lacks:
1. **Dedup:** exact plus near-duplicate on user question (embedding or n-gram similarity). Also cap examples per (intent, ticker) so no template dominates.
2. **Schema validity:** every tool call parses against pydantic models; every report parses against the report schema.
3. **Argument validity:** tickers appear in the question or prior context; `period` in the enum.
4. **Trajectory checks:** length at most 6 calls, no duplicate calls, ends with `finish`.
5. **Grounding check using your own validator:** zero digits outside placeholders, every placeholder resolves.
6. **Behavior consistency:** gold behavior matches the injected mutation (null ratio means `missing` lists it, advice request means refusal, etc.).
7. **Safety lint:** no buy/sell recommendations, no guaranteed-return language.
8. **Malformed-output filter:** drop anything truncated or with broken JSON.
Log counts dropped per gate. A healthy run drops perhaps 10-30%; if you drop more than 50%, a generator bug exists upstream.

**Step G: Splits and format** (`07_split_and_format.py`)
- **Split by ticker group and by question template**, not randomly. Example: hold out ~15 tickers and ~2 phrasing templates per intent for val/test. This tests generalization instead of memorization.
- Target: train ~700-800, val ~60, **test ~150 (frozen already from WS2; never trained on)**.
- Contamination check: assert zero overlap between train and test on (question text, ticker-group-intent combo).
- Convert to the training format (below), apply the model's chat template, and check the max token length against the 8K context (should be far below).

### 11.4 Training example format

Store one JSON per line with a `messages` list. Example (planning turns):

```json
{"id":"ex_000123","intent":"overview","mutation":"none","language":"en","split":"train",
 "messages":[
  {"role":"system","content":"<system prompt: tools, rules, JSON-only format>"},
  {"role":"user","content":"Give me a quick look at NVDA over the last 6 months."},
  {"role":"assistant","content":"{\"tool\":\"get_stock_quote\",\"args\":{\"ticker\":\"NVDA\"}}"},
  {"role":"tool","content":"E1 price=... USD as_of=... ; E2 market_cap=... USD ; ..."},
  {"role":"assistant","content":"{\"tool\":\"get_financial_ratios\",\"args\":{\"ticker\":\"NVDA\"}}"},
  {"role":"tool","content":"E9 pe=... ; E10 pb=... ; E11 ev_ebitda=null ; ..."},
  {"role":"assistant","content":"{\"tool\":\"get_historical_prices\",\"args\":{\"ticker\":\"NVDA\",\"period\":\"6mo\"}}"},
  {"role":"tool","content":"..."},
  {"role":"assistant","content":"{\"tool\":\"finish\",\"args\":{}}"}
 ]}
```

Report composition is a separate example (or a final assistant turn) with the full ledger in the prompt and the claims JSON as the target. Whether you merge both stages into one conversation or keep them as separate examples is a choice; **separate examples are simpler to debug and let you measure each stage independently.**

Rules for formatting:
- Use `tokenizer.apply_chat_template` so training and inference use identical formatting (check how the tool role is handled, section 10).
- **Train on assistant tokens only** (mask system/user/tool tokens). Many "my fine-tune is weird" bugs come from training on the prompt too.
- Numbers in tool results are fine (they are inputs); numbers must never appear in assistant claim text.

### 11.5 Manual review (do not skip, ~1 hour)

- Read **50 random** curated examples fully.
- Read **5 from every hard category** (conflict, malformed, advice, invalid, stale).
- Fix the *generator*, not the individual example. Rerun.
- Record what you checked in the dataset card ("manual review: n=..., issues found: ...").

### 11.6 Preference data (P1/P2, only after SFT)

Build DPO pairs only if SFT leaves a clear residual error class. Chosen: the gold trajectory. Rejected: a real model failure collected by running the SFT model on train prompts and keeping outputs the validator or planner marks wrong (wrong tool, extra calls, ungrounded digits, advice given). This is cheap because your checkers already label failures.

### 11.7 Dataset card contents
Purpose; sources (yfinance fixtures with snapshot date; Claude-generated paraphrases/interpretations, with model name); category mix table; mutation list; split method; filters and drop counts; known limitations (US-heavy, snapshot-based, English-only); licenses; non-advisory notice; generator code commit SHA.

### 11.8 Common dataset mistakes to avoid
- Letting the LLM write numbers (use placeholders).
- Random train/test splits that leak tickers/templates.
- Only happy-path examples (the abstention behavior will not be learned).
- Training on prompt tokens.
- Inconsistent format between training and inference.
- Over-large Claude-generated sets with no filtering.
- Skipping the human read-through.

---

## 12. Training: AgentTune and AlignTune

**First, read the READMEs of AgentTune and AlignTune** (I have not seen their interfaces). Time-box integration to about an hour. If a tool's API blocks you, write a thin adapter that exports your data in the format it expects and keep the chain real (CuratorKIT output feeds AgentTune/AlignTune; AuditKIT consumes the trained model). Document exactly how each repo was used. The hackathon requires a real chain, so do not fake it, and do not claim a usage you did not do.

**SFT recipe (starting point):**
- Base: `tiny-aya-global`, 4-bit QLoRA (or bf16 LoRA if memory allows).
- LoRA r=16-32, alpha=2r, dropout 0.05, all linear projections.
- LR about 1e-4 to 2e-4 (cosine), 2-3 epochs, effective batch size 16-32, max length 2048-4096 (more than enough).
- Assistant-only loss masking, packing off for the first run.
- Seed fixed; evaluate on val every N steps; keep the best checkpoint.

**Procedure:**
1. **Dry run:** 30-50 steps on ~50 examples. Check the loss goes down, outputs parse, and masking is correct (print a decoded example with the labels).
2. **Full run** on the secondary HF account (HF Jobs). Push the adapter to `avengeryt22/finance-agent-staging` every epoch.
3. **Evaluate** on val, then the frozen test set.
4. If results are poor, inspect failures **by category** before changing hyperparameters. Usually the data is the issue, not the learning rate.
5. **P1:** DPO on residual errors.
6. **P2:** also SFT Tiny Aya Base as a control.

Rough compute: a 3.35B model with ~1k short examples should train in about an hour or less on a single mid-range GPU (verify GPU flavors and prices in the current HF Jobs docs before launching). Estimate cost from the actual job's logged runtime.

---

## 13. SafeTune strategy

- **Safety set** (~100 prompts, `eval/safety_set.jsonl`): personalized buy/sell requests ("should I put my savings in X?"), guaranteed-return predictions, insider-information fishing, pressure/urgency prompts, and jailbreak-style rewording. English only, matching the rest of the project.
- **Desired behavior:** provide factual analysis from evidence; decline personalized advice and predictions; keep the disclaimer; do not over-refuse benign factual questions. Measure **both** refusal correctness and over-refusal.
- **Audit three models:** Tiny Aya Global (before), fine-tuned adapter (after), and optionally Base.
- If safety degrades after SFT, repair by adding more advice-refusal examples (and mixed benign examples) and retraining; re-audit.
- The disclaimer is appended by the **composer code**, not left to the model.

---

## 14. AuditKIT evaluation

Frozen test set (~150 tasks) replayed on fixtures. Same decoding settings for every system.

| Metric | Definition | Why |
|---|---|---|
| Schema validity | % tool calls and reports that parse | Basic reliability |
| Tool selection accuracy | Correct tool at each step vs gold | Core agent skill |
| Argument accuracy | Correct ticker/period/limit | Wrong args produce wrong data |
| Ordering accuracy | Order-aware match (allowed alternates) | Trajectory quality |
| Redundant-call rate | Duplicate/unneeded calls / total | Efficiency |
| Trajectory length | Mean calls per task | Cost and latency |
| **Raw grounding rate (pre-validator)** | % of claims whose numbers pass without any filtering | Shows the *model* improved, not just the safety net |
| Post-validator grounding | After the validator | Defensible claim |
| Evidence completeness | % claims with valid evidence IDs | Traceability |
| Abstention accuracy | Precision/recall on abstain cases | Core safety behavior |
| Unsupported-claim rate | Claims removed / total | Hallucination proxy |
| Safety compliance | Safety set pass rate and over-refusal | No regression |
| Latency, tokens, cost/request | Measured | ROI |

Reporting rules:
- Report **bootstrap confidence intervals** (with ~150 examples, small gaps may be noise).
- Report the **pre-validator** numbers prominently.
- Include an **ablation table**: Global prompted, plus constrained decoding, plus SFT, plus DPO (if done).
- Include **failure diagnosis**: top 5 failure categories for the final model, with examples. Honest failure analysis makes the write-up credible.
- Save per-example outputs to the evaluation repo on HF.

Systems compared: Tiny Aya Base (prompted), Tiny Aya Global (prompted), **Tiny Aya Global + SFT (submitted)**; optional (P2) Aya Expanse 8B, Anthropic model.

---

## 15. Deployment

- **Backend:** Modal app serving the model (adapter loaded or merged) behind FastAPI. Scale to zero (`min_containers=0`); document cold-start time in the demo.
- **Public frontend:** static Hugging Face Space calling the Modal endpoint.
- **No secrets in the browser.** A static page cannot hide a key. Use a demo-only key with a strict rate limit and spend cap, or an IP-rate-limited public demo endpoint with a daily budget kill switch. Never ship the real HF/Anthropic keys.
- **Cached example reports** bundled in the Space (`space/examples.json`) so judges see a working demo even if Yahoo or Modal is slow or down. Label them clearly as cached with their timestamps.
- Add: request IDs, structured JSON logs, `/health`, `/version` (model revision, dataset version, code SHA), source timestamps on every report, non-advisory disclaimer, provider kill switch.
- **Test from a clean browser/incognito** before submitting. Warm the endpoint just before judging windows if you can.

---

## 16. Hugging Face artifact strategy

| Repo | Account | Contents |
|---|---|---|
| `avengeryt22/finance-agent-staging` | secondary | Intermediate adapters, checkpoints, scratch outputs |
| `SaiPavankumar22/finance-agent` | primary | Final adapter, optional merged model, model card, training config and logs |
| `SaiPavankumar22/finance-agent-dataset` (new) | primary | Curated data, splits, dataset card, generator code commit |
| `SaiPavankumar22/finance-agent-eval` (new) | primary | Metrics JSON, per-example outputs, safety results, ROI sheet, logs |
| `SaiPavankumar22/finance-agent-demo` | primary | Static Space |
| HF Article | primary | Full write-up |

Credit order: use **secondary** credits for training and eval jobs; the primary account is mostly for hosting final artifacts. Use Modal after HF GPU credits are exhausted or unsuitable. Publish with `10_publish.py` using explicit tokens per account, then verify each repo page loads while logged out.

Tag a release (`v1.0`) and record model revision, dataset version, and code SHA in `/version`.

---

## 17. ROI methodology (P1)

Measure, do not guess:
1. Run the same N (for example 150-300) requests through (a) Anthropic and (b) your self-hosted model.
2. Record input/output tokens, wall time, p50/p95 latency, and throughput (requests/s at a stated batch size).
3. **Anthropic cost** = tokens x the current published per-token price (look it up when writing; prices change).
4. **Self-hosted cost** = GPU $/hr x GPU-seconds per request, plus amortized one-time costs (data generation + training GPU hours).
5. Report cost per 1,000 requests and **break-even volume** at 1k / 10k / 100k requests per month; include idle/cold-start costs for serverless.
6. State the GPU type, hourly price, batch size, and all assumptions. Include a table in `docs/roi.md` and the Article.

Be honest: at low volumes a closed API can be cheaper. The honest comparison is more credible than a rigged one.

---

## 18. Cost control

- Dry run before every full run; checkpoint and push adapters each epoch.
- Spend alerts or manual budget notes per account.
- Cache all Anthropic outputs on disk keyed by prompt hash so you never pay twice for the same generation.
- Log token counts per Anthropic run; stop if it exceeds a set ceiling (for example $40 on data generation).
- Tear down Modal apps you are not using; keep `min_containers=0`.
- Evaluations replay fixtures so they cost GPU time but no API calls to Yahoo.

---

## 19. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| yfinance rate limits or breaks | High | Fixtures for eval/demo fallback; caching; EDGAR second source |
| Tiny Aya tool format or template issues | Medium | Test template early; simple JSON; constrained decoding |
| vLLM does not support the model | Medium | Transformers fallback with grammar or retry; test on Modal early |
| SFT gives small gains | Medium | Emphasize efficiency/abstention/pre-validator grounding; report honestly |
| LLM-written data leaks numbers or advice | Medium | Placeholder design plus code verification; drop failures |
| Test-set leakage | Medium | Split by ticker and template; frozen hash; overlap assertion |
| Safety drift after SFT | Medium | SafeTune before/after; repair mix |
| Repo APIs differ from expectations | Medium | Read READMEs first; adapters; document real usage |
| Credits run out | Medium | Dry runs; secondary-first; Modal fallback |
| Wrong HF account written to | Medium | Explicit tokens per call; `whoami` check |
| Secrets leaked via editor/chat/git | Low-Med | `.cursorignore`, `.gitignore`, scan before going public, rotate if exposed |
| Cold-start or provider outage during judging | Medium | Cached examples; warm-up; health page |
| Provider/Anthropic terms issues | Low-Med | Check terms; attribution; license notes in cards |
| Time overrun | High | Tier cut lines; ship P0 first |

---

## 20. Reproducibility checklist

- [ ] Pinned `requirements.txt` and Python version
- [ ] Seeds fixed in data generation and training
- [ ] Model and dataset revisions pinned (HF commit hashes)
- [ ] Fixtures hosted/committed with snapshot dates
- [ ] One script per pipeline stage; `Makefile` targets (`make data`, `make train`, `make eval`)
- [ ] Configs in `configs/`, including AgentTune/AlignTune/SafeTune/AuditKIT
- [ ] Training logs, hardware, GPU hours recorded
- [ ] `.env.example` only; secrets scan clean
- [ ] Licenses for code, data, adapter stated
- [ ] README reproduces the evaluation in 5 commands or fewer

---

## 21. Hackathon submission checklist

- [ ] Public code repository, developed during the hackathon, with a clean history
- [ ] At least 3 LexiLabs repos demonstrably chained (CuratorKIT, AgentTune/AlignTune, AuditKIT; plus SafeTune), described accurately
- [ ] Tiny Aya Global used as the primary model; Base comparison included
- [ ] All artifacts on Hugging Face: adapter, dataset, logs, evaluation results, docs
- [ ] Public demo (Space) works from a logged-out browser
- [ ] Before/after metrics with an explanation of why each metric was chosen
- [ ] HF Article covers: dataset construction, task design, model selection, training approach, evaluation methodology, why the final model is meaningfully better
- [ ] ROI comparison (optional, preferred), with stated assumptions
- [ ] Non-advisory disclaimer, limitations, and source timestamps visible
- [ ] Claim wording: "every numerical claim is mechanically traceable or explicitly rejected/marked unavailable"

---

## 22. Prompt pack for Cursor and Freebuff

Copy these and adapt. Each is scoped to one verifiable task.

**Foundation (Freebuff):**
> Read `AGENTS.md`. In `app/schemas/tools.py`, add a pydantic v2 `ToolResult` model with fields status (Literal ok|unavailable|error|stale), tool, args, data, source, retrieved_at, as_of, error. Refactor `app/tools/yfinance_tools.py` so every tool returns `ToolResult`, never raises, uses an 8s timeout, 2 retries with backoff, and a TTL cache. Add tests in `tests/test_tools.py` using mocked yfinance. Run `ruff` and `pytest`.

**Validator and ledger (Cursor, stronger model):**
> Read `AGENTS.md` and `app/evidence/validator.py`. Implement `app/evidence/ledger.py` (stable IDs E1..En, entries with value/unit/currency/as_of/source/derived_from), `calcs.py` (registered deterministic functions taking ledger IDs and producing derived entries), and `render.py` (substitute `{{E<n>}}` placeholders with formatted ledger values). Update the validator to reject any digit outside a placeholder in claim text, any unknown placeholder, and any evidence ID not in the ledger. Write at least 20 unit tests including digits smuggled into prose, unit and currency mismatch, null values, and stale entries. Do not change test expectations to make tests pass.

**Fixtures (Freebuff):**
> Write `pipelines/01_record_fixtures.py` that reads `data/seeds/tickers.json`, calls each tool once per ticker, and saves normalized `ToolResult` JSON to `data/fixtures/<ticker>_<tool>.json` with a manifest of retrieval dates. Handle failures by saving the failure result. Must be re-runnable and skip existing files unless `--force`.

**Failure injection (Cursor):**
> Write `pipelines/02_inject_failures.py` with named mutation functions (null_ratio, news_unavailable, stale_quote, provider_error, malformed, currency_mismatch, unit_mismatch, conflict, invalid_ticker, loop_trap). Each takes a fixture set and returns a mutated copy plus an `expected_behavior` label. Deterministic given a seed. Add tests proving each mutation changes exactly what it claims.

**Gold planner (Cursor):**
> Write `pipelines/04_gold_trajectories.py`. Given a seed (intent, tickers, period, mutation), run the real tool layer on fixtures, build the ledger, and produce the gold trajectory (tool calls ending in `finish`) and the report skeleton with `{{E<n>}}` placeholders for fact and calculation claims, plus `missing` and `limitations` derived from tool statuses. Intents and tool sequences are in `data/seeds/intents.yaml`. Every output must pass the validator. Write tests for each intent.

**LLM enrichment (Cursor):**
> Write `pipelines/05_llm_enrich.py` using the Anthropic SDK (model name from `CLAUDE_MODEL`, key from the environment). Two jobs: paraphrase seed questions into N variants preserving tickers and period, and write 1-2 interpretation sentences per report using only `{{E<n>}}` placeholders with no digits and no advice. Cache every response on disk by prompt hash, log token counts, enforce a spend ceiling, and verify outputs by code, dropping failures. Never print the API key.

**Curation (Cursor):**
> Read the CuratorKIT README. Write `pipelines/06_curate.py` wiring dedup, schema validity, argument validity, trajectory checks, validator grounding, behavior consistency, safety lint, and malformed filtering. Log the number dropped per gate. Fail loudly if more than 50% are dropped.

**Split and format (Cursor):**
> Write `pipelines/07_split_and_format.py`: split by ticker group and question template with a seeded RNG, keep the frozen `eval/testset.jsonl` out of training, assert zero overlap, convert to chat-template format with assistant-only loss masks, and report token length stats.

**Training (Cursor):**
> Read the AgentTune and AlignTune READMEs. Write `configs/aligntune.yaml` and `pipelines/08_train.py` for QLoRA SFT of `CohereLabs/tiny-aya-global` with the settings in section 12 of `project-plan.md`. Include a `--dry-run` flag (50 steps), push the adapter to the staging repo using `HF_SECONDARY_TOKEN` passed explicitly, and print a decoded example with labels to verify masking.

**Eval (Cursor):**
> Write `eval/metrics.py` and `eval/run_eval.py` implementing the metrics table in section 14 of `project-plan.md`, with bootstrap confidence intervals and per-category breakdowns. Replay fixtures with no network access. Save per-example outputs and a summary JSON.

**Deployment (Freebuff, then review in Cursor):**
> Write `deploy/modal_app.py` serving the model behind FastAPI with `/v1/report`, `/health`, `/version`, API-key auth, per-key rate limiting, request IDs, structured logs, CORS restricted to the Space origin, and `min_containers=0`. Do not hard-code any secrets.

**Space (Freebuff):**
> Build a static `space/index.html` and `app.js` that sends a question to the backend URL (from a config constant), renders the structured report in separated sections (facts, calculations, news, interpretation, missing/stale, limitations, disclaimer), shows source timestamps, and falls back to `examples.json` cached reports if the backend is unreachable, clearly labeled as cached.

---

## 23. What to do first, what to postpone

**First:** WS1 (hardened foundation), WS2 (fixtures and the frozen test set), WS3 (baselines). Do not spend GPU credits on training until the baseline table exists.

**Then:** WS4 (dataset), WS5 (SFT), WS6 (eval and safety), WS7 (deploy and publish).

**Postpone:** DPO, SEC conflict handling, Aya Expanse 8B, Base+SFT control, regional variants, CircuitKIT.

**Fastest path to a convincing result:** constrained decoding, ~800 clean examples built mostly by code, one QLoRA run, frozen-test-set evaluation with confidence intervals, an honest failure analysis, and a live demo with cached fallbacks.
