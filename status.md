# Finance Agent — Status

_Last updated: 2026-11-10, after the full WS5 training run and its before/after
evaluation on the held-out-ticker validation split and the frozen test set._

## Verdict

WS1 and WS2 are complete. WS4 produces a usable dataset: 800 gold trajectories,
all passing curation with 13 of 13 behaviours intact, split into 620 train / 51
ticker-held-out validation examples. **WS5 now has a trained adapter with a
measured before/after**: tool selection 0.5400 → 0.8333 on the frozen test set
and 0.5294 → 0.8235 on unseen tickers, with argument accuracy 0.0400 → 0.8733.
Abstention recall rises 0.00 → 0.46 at precision 1.00, so refusal is still the
main gap and belongs to WS6. `ruff check .` is clean and `pytest -q` reports 148
passing tests.

## Commit history

| Commit | Contents |
|---|---|
| `9b37baa` | Initial repository commit |
| `86b1f21` | WS1: envelope, ledger, calcs, renderer, validator, hardened tools |
| `15319b7` | WS2: fixtures, offline replay, injections, frozen test set |
| `3f0db0e` | English-only test set and templates |
| `f6a5292` | Plan updated for English-only |
| `a8980b4` | Project README |
| `1ca95f9` | WS3 baseline evaluation harness |
| `b4092d6` | WS4 gold trajectories |
| `7be6979` | CuratorKIT integration + Anthropic enrichment |
| `2e5f300` | Tiny Aya vLLM decoding verified on Modal |
| `a3e2d46` | Corrected status after auditing WS3 and WS4 |
| `1da5791` | WS4: varied invalid-ticker symbols (4 → 60 rows) |
| `41f3bb4` | WS4: fixed the CuratorKIT gate, restored metadata |
| `7ffe976` | WS4: ticker-holdout split and dataset card |
| `d182053` | Status document updated for the repaired WS4 dataset |
| `dcb13de` | WS3: abstention scored as precision/recall, cached baselines rescored |
| `983b94f` | WS5: Aya chat formatter and QLoRA smoke run |
| `49bff4c` | WS5: full 620-row run, evaluation entrypoint, before/after comparison |
| `785b2a6` | WS5: fair decode rule, re-measured before/after, status updated |

## WS1 — hardened foundation (done)

`ToolResult` envelope, evidence ledger (`E1..En`, monotonic, nulls recorded),
registered calcs, placeholder renderer, strict validator, five-tool registry,
hardened yfinance layer (timeout, retries, TTL cache, per-tool breaker,
staleness downgrade, never raises), offline replay. The model never writes
digits: validation runs on the template, substitution afterwards.

Rules: percents are percentage points; periods `1mo,3mo,6mo,1y,2y,5y` with `1y`
default; staleness quote 5d / ratios 120d / history 10d / news 90d.

## WS2 — fixtures, injections, frozen test set (done)

365 fixtures, 73 tickers, `1mo` + `1y`, snapshot 2026-11-09; 276 ok / 69
unavailable / 20 error; 69 tickers fully usable. Offline replay verified
(`deterministic=PASS`, digest `33397eb7…`). Ten deterministic mutations, 300
injected sets (gitignored, regenerable). Frozen test set: 150 examples, 37
expecting abstention, English only, SHA-256
`025c345dc028d457f99a1d9cf2f6c4c1b7069f72db166a3fe1ca41c5d2fd2e94`.
`*.jsonl` is pinned `-text` in `.gitattributes`, so line endings cannot drift
the hashes.

## WS3 — baselines (partial, abstention now measurable)

`eval/run_baseline.py` + `eval/metrics.py` exist; `eval/results/baseline.json`
holds 150 predictions per model:

| Model | Examples | Schema valid | Tool selection | Argument acc. | Abstain recall |
|---|---|---|---|---|---|
| `CohereLabs/tiny-aya-base` | 0 | — | — | — | — |
| `CohereLabs/tiny-aya-global` | 150 | 0.9867 | 0.5533 | 0.0533 | 0.00 |
| `anthropic` (oracle) | 150 | 0.9000 | 0.5400 | 0.6533 | 0.00 |

The base model never ran (`not supported by provider cohere`). It is a **P2
control** (WS11), not a gate: the plan's decision gate is about Global prompted,
which does not score high, so SFT is justified.

**Abstention rescored** (`eval/rescore_abstention.py`, offline, no API calls).
The old `abstention_accuracy` was overall accuracy, so never abstaining scored
`1 - 37/150 = 0.7533`. Abstention is now a binary decision with precision,
recall, specificity, balanced accuracy and over/under-abstention rates, and only
a structured `finish` call counts as declining:

| Model | Recall | 95% CI | Precision | Specificity | Over-abstain | Under-abstain |
|---|---|---|---|---|---|---|
| `tiny-aya-global` | 0.00 | 0.000–0.094 | n/a (no positive) | 1.000 | 0.000 | 1.000 |
| `anthropic` | 0.00 | 0.000–0.094 | n/a (no positive) | 1.000 | 0.000 | 1.000 |

Neither model ever declines, and neither ever refuses a healthy question. Two
measurement caveats are recorded in `eval/results/abstention.json` rather than
hidden in the number: 23 of the 37 abstain rows expect a tool call *before* the
abstention (gold declines at turn one on only 14, a turn-one recall ceiling of
0.378), and the baseline prompt says "Never answer the user directly", which
makes a refusal off-contract. Turn-one scoring therefore cannot measure
abstention; WS6 has to judge it after a tool error on a replayed trajectory with
an explicit refusal affordance.

## WS4 — dataset (repaired)

**Gold** (`data/gold/trajectories.jsonl`, 800 rows, SHA-256 `c7e7acd3…`): a
correct tool-calling trajectory per seed question, generated by code from
recorded fixtures; reports carry `{{E<n>}}` placeholders validated against the
ledger. 68 recorded symbols plus 12 bogus ones, 157 distinct question
phrasings, `1mo`/`1y` only.

**Curated** (`data/curated/curated.jsonl`, 800 rows, SHA-256 `b2112851…`): the
CuratorKIT run plus the project gates, with `id`/`intent`/`tickers`/`period`
re-attached.

| Intent | Gold | Curated | Train |
|---|---|---|---|
| comparison | 99 | 99 | 64 |
| overview | 86 | 86 | 68 |
| metric_pe | 82 | 82 | 64 |
| advice_request | 81 | 81 | 63 |
| period_ambiguity | 77 | 77 | 65 |
| unsupported_metric | 75 | 75 | 59 |
| non_us_currency | 63 | 63 | 43 |
| invalid_ticker | 60 | 60 | 40 |
| missing_ratios | 42 | 42 | 34 |
| news_unavailable | 40 | 40 | 36 |
| stale_data | 37 | 37 | 33 |
| malformed_error | 32 | 32 | 28 |
| loop_trap | 26 | 26 | 23 |

Four defects were fixed, in order:

1. **The length gate.** CuratorKIT's schema gate counts the user question plus
   the *first assistant call*, which is always one JSON token. Its 10-token
   floor therefore measured question phrasing: 538 of 538 rejections were
   `below_min_tokens`, equal to `tokens(question) + 1`. Nine intents were
   emptied. Now run with `min_tokens=0`; trajectory length is checked by the
   project gate instead.
2. **Lost metadata.** The messages exporter writes `{"messages": [...]}` only.
   Ids and labels are joined back by exact question text (0 unmatched).
3. **Retry misread as looping.** The gate rejected any repeated call, which
   rejected every `loop_trap` and `malformed_error` row. One repeat is a
   legitimate bounded retry when the previous result was unusable; a repeat
   after a usable result, or a second retry, is still rejected.
4. **Untrainable abstention.** The bogus pool was a constant `ZZZZ`, so
   question de-duplication capped invalid tickers at 4 rows. Symbols now vary
   over 12 values and the category has 60 rows.

MinHash de-duplication at 0.85 was replaced with exact de-duplication: it
removed 327 of 800 rows, almost all questions that differed only by ticker,
which is precisely the coverage the split needs.

**Splits** (`data/splits/`, SHA-256 `abc18cb1…` / `0a65affd…`): 620 train, 51
val, seed 13. Validation is composed only of 15 held-out real tickers (ARKK,
AVGO, CAT, CROX, JPM, NESN.SW, NFLX, NKE, ROKU, SHOP.TO, TCS.NS, TTE.PA, TXN,
ULVR.L, WMT) plus two bogus symbols; 31 mixed-group rows and 21 rows over the
per-cell cap were dropped. Asserted: no train/val question overlap, no frozen
test question, no ticker leakage, all 13 intents in train.

## WS4 validation review and enrichment decision

The 51 validation examples were reviewed across all 13 intents. The review
confirmed:

- every row has the expected intent and behaviour;
- valid rows use registered tools and valid arguments;
- advice and invalid-ticker cases terminate without unsupported analysis;
- malformed and loop-trap cases perform at most one bounded retry and then
  finish;
- observations are recorded fixture payloads, with null, stale, unavailable,
  and error states preserved;
- train/validation questions do not overlap, held-out tickers do not leak into
  train, and no validation question is in the frozen test set.

The 38 accepted Anthropic enrichments remain separate. They contain useful
language variants and optional interpretation text, but they are not complete
training trajectories and their interpretations are not part of the runtime
message schema. The 62 rejected records also show that the enrichment gate
still needs refinement for ticker digits, evidence references, and safety
phrasing. **Decision: do not merge the 38 records into the SFT dataset yet.**
Keep them as an auditable enrichment artifact; later, independently validate
their question variants and reconstruct full trajectories from the deterministic
gold rows before using any as augmentation. This preserves the strict
grounding guarantee and avoids allowing Anthropic prose to become authoritative
financial data.

## Known gaps

- Enrichment (`data/enriched/`, 38 accepted of 100 attempted) is not merged;
  question phrasings remain template-derived (157).
- Fixtures cover 2 of 6 periods, so the period argument is trained on `1mo`
  and `1y` only.
- 54 of the frozen test set's 57 tickers also appear in train. The test set
  cannot change and is not question-contaminated, but it does not measure
  unseen-ticker generalisation; the split's held-out val does.
- The local base baseline (adapter off, 4-bit) is measured; the fp16 provider
  baseline from WS3 remains the published "before" number.
- The adapter lives on the Modal volume `finance-agent-sft-adapters` only; it
  is not yet pushed to `avengeryt22/finance-agent-staging`.
- Abstention recall is 0.46 on the frozen test set: the tuned model still
  answers 20 of 37 rows that should be declined.
- Anthropic enrichment is intentionally not merged; question diversity remains
  template-derived (157 phrasings).
- No AuditKIT/SafeTune results, no deployment, no git remote.

## WS5 — fine-tuning (full run complete)

`training/train_qlora.py` is a Modal job; `artifacts/sft/smoke.json` holds the
run. 40 optimizer steps on a 64-row slice, 4-bit QLoRA, r=16, alpha=32, LR
2e-4, effective batch 4, `max_len` 6144:

| Check | Result |
|---|---|
| Loss, first 5 steps → last 5 | 2.036 → 0.036 |
| Trainable parameters | 30,228,480 (1.53% of 1.97B) |
| Supervised tokens in slice | 2,974 (mean 46.5 per row) |
| Masking | 727 masked / 33 supervised; every label before the first assistant turn is −100 |
| Decoded supervision | `{"tool":"get_financial_ratios","args":{"ticker":"DIS"}}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` |
| Greedy next token (first turn) | gold `{"`, p = 0.9996 |
| Runtime / peak memory | 44 s training, 29.6 GB on A100-40GB |

Three format facts the run established, all now encoded in `app/chat_format.py`
so training and inference cannot diverge:

1. The template rejects `role="tool"` (`Conversation roles must alternate
   user/assistant/...`), so tool results go inside a user turn as
   `[TOOL_RESULT call_id=c1] {...} [/TOOL_RESULT]`.
2. It injects a fixed ~200-token system preamble, so sequences are longer than
   the message text suggests; 96 of 620 rows reach 4.1k–6.4k tokens because
   four tool results carry 250-point histories. The plan's 2048–4096 window is
   too small; 6144 with gradient checkpointing keeps every trajectory.
3. `apply_chat_template(tokenize=True)` returns a `BatchEncoding` in
   transformers 5 and already holds BOS, so re-tokenizing the rendered string
   would double it.

Two GPU facts: 22 GB A10G OOMs at 6144 tokens, because a 262k-vocabulary
`lm_head` costs ~6 GB of logits on its own. The first two runs also produced
degenerate repeated-token output; it disappeared when generation moved to eval
mode with the KV cache and the head was kept in bf16. Those three changes were
made together, so which one was decisive is not isolated.

Behaviour after 40 steps, base vs tuned on the same prompts:

| Probe | Base | Tuned (gold) |
|---|---|---|
| `Tell me DIS's P/E` (first turn) | fenced JSON, 3 lines, no turn end | exact gold call, then repeats it |
| same row, final turn | answers in prose with a number | `finish`, turn ended |
| `I need BRK-B to double. When will it?` | invents `include_volatility`, prose advice | valid calls, but calls tools instead of declining |

So the output *format* and in-distribution tool choice are learned; turn
termination and advice refusal are not, at 40 steps on 64 rows. That the tuned
model calls tools on the advice prompt matches the rescored abstention result
(recall 0.00) and is the failure WS6 must measure.

### Full run: 620 rows, 2 epochs

`training/train_full.py` trains on every training row and reuses
`eval/run_baseline.py`'s system prompt and parser through `training/data.py`,
so the before/after pair shares one prompt, one parser and one decode rule.
The 51 held-out-ticker rows and the 150 frozen examples are never trained on:
`assert_disjoint` refuses the run if a question or validation ticker appears in
train, and the frozen test set is hashed before and after (unchanged).

| Setting | Value |
|---|---|
| Rows / epochs / optimizer steps | 620 / 2 / 78 |
| Effective batch | grad_accum 16, one example per micro-batch |
| LR / schedule | 1e-4 OneCycle, 3% warmup, grad-norm 1.0 |
| LoRA | r 16, alpha 32, dropout 0.05, all linear |
| Window | 6464 tokens, 0 rows truncated, 2,066,586 tokens seen |
| Supervision | 30,542 supervised tokens; 30,228,480 trainable (1.53% of 1.97B) |
| Runtime / peak | 27.2 min training on A100-40GB, 30.7 GB peak |

Validation on the unseen tickers, both epochs: 0.8235 / 0.8824, then 0.7843 /
0.8627 (tool selection / argument accuracy). Epoch 2 is worse, so best-epoch
selection kept epoch 1 — 620 rows at 1e-4 fit within one pass and the second
is memorisation.

The base row is the same weights with the adapter disabled, run locally in
4-bit. It lands within 0.013 of the WS3 provider baseline on the frozen set
(tool 0.5400 vs 0.5533, args 0.0400 vs 0.0533), which is the evidence that the
two harnesses agree before any adapter number is trusted.

| Set | System | Tool selection | Arguments | Schema | Abstain recall |
|---|---|---|---|---|---|
| val, unseen tickers | base | 0.5294 | 0.0588 | 1.0000 | 0.00 |
| val, unseen tickers | tuned | **0.8235** | **0.8824** | 1.0000 | 0.40 |
| test, frozen | base | 0.5400 | 0.0400 | 1.0000 | 0.00 |
| test, frozen | tuned | **0.8333** | **0.8733** | 0.9933 | 0.46 |
| test, frozen | provider baseline (WS3) | 0.5533 | 0.0533 | 0.9867 | 0.00 |

Paired on the 150 frozen examples, tool selection gained 48, lost 4, correct in
both 77, wrong in both 21. Arguments gained 128, lost 3, correct in both 3.
Wilson 95% intervals are stored per metric in `artifacts/sft/eval-json-stop.json`.

Abstention precision 1.00 and specificity 1.00 in both arms: the tuned model
never declines something it should answer, but still answers 20 of 37 rows that
should be declined. Recall 0.46 is measured on the first turn, where abstention
is structurally hard — 23 of those 37 rows need a tool result before declining
means anything, and the prompt tells the model never to answer the user
directly (ceiling 0.378, see `eval/results/abstention.json`). WS6 has to move
this with a replayed trajectory and an explicit refusal affordance.

Where it still fails, by intent on the frozen set: `comparison` 0.53 (no change),
`period_ambiguity` 0.50, `invalid_ticker` 0.73 (no change), `non_us_currency`
0.64 (0.75 → 0.50 on val, n=4, noise). Biggest gains: `unsupported_metric`
0.17 → 1.00, `overview` 0.30 → 0.80, `news_unavailable` 0.00 → 1.00,
`loop_trap` 0.75 → 1.00, `advice_request` 0.00 → 1.00 — the last one now
declines where the base model gave advice.

**Decoding rule matters.** The training loop scored validation with a 24-token
budget, which truncated the base model mid-`args` and reported 0.00 for every
base metric. Generation now stops at the first complete JSON object (96-token
cap, `first_json_object_complete` in `training/data.py`, unit-tested) and both
models are re-measured under it by
`modal run training/train_full.py::eval_main`. `eval/compare_adapter.py`
regenerates the table above from `artifacts/sft/eval-json-stop.json`;
`artifacts/sft/full.json` keeps the training payload and per-epoch validation.

Two failures worth recording, both now fixed: PEFT's input-grad hook has to be
restored when gradient checkpointing is re-enabled after generation, otherwise
epoch 2 kept every activation and OOMed at 73 GiB; and the result payload is
written to the volume, because it otherwise only exists on the client that can
die mid-run.

## Next steps

1. **Push the adapter** from the Modal volume to
   `avengeryt22/finance-agent-staging` and record its revision.
2. **WS6**: safety set (~100 prompts) with an explicit refusal affordance and a
   replayed trajectory so abstention is judged after tool errors.
3. **WS7**: Modal endpoint using `app/chat_format.py`, Space with cached
   examples, HF repos, Article.
4. Parallel/optional: merge the enrichment paraphrases and record the four
   missing periods; `comparison` and `period_ambiguity` are the intents an
   extra pass would target, since the adapter did not move them.
5. Inference detail for WS7: stop on `<|END_OF_TURN_TOKEN|>` and require one
   JSON object per turn; the dry run shows the model will otherwise continue
   emitting calls.
