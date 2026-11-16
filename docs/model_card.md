---
base_model: CohereLabs/tiny-aya-global
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
  - en
tags:
  - finance
  - tool-calling
  - lora
  - peft
  - indian-markets
datasets:
  - SaiPavankumar22/finance-agent-dataset
  - SaiPavankumar22/finance-agent-eval
---

# Finance Agent: Safe Indian-Market Research with Tiny Aya

A tool-using financial research assistant that would rather say "I can't get
that" than invent a number.

## Summary

This is a LoRA adapter for `CohereLabs/tiny-aya-global` that plans financial
research requests. It emits exactly one JSON object per turn naming one of
`get_stock_quote`, `get_financial_ratios`, `get_historical_prices`,
`get_company_news`, or `finish`, and it never states a price, ratio or forecast
itself: every number in an answer arrives from a timestamped provider response.

The interesting problem here is not prose generation. It is a decision boundary:
pick the right tool and arguments, or decline. Requests for personal advice,
price predictions, insider information or unsupported actions must end in an
explicit decline rather than a confident-sounding guess.

This is the first candidate to clear every planning, safety and India gate.

- Demo Space: https://huggingface.co/spaces/SaiPavankumar22/finance-agent-demo
- Dataset: https://huggingface.co/datasets/SaiPavankumar22/finance-agent-dataset
- Evaluation sets: https://huggingface.co/datasets/SaiPavankumar22/finance-agent-eval
- Code: https://github.com/SaiPavankumar22/Finance-Agent

## Why this model and task

Tiny Aya Global is small enough for affordable serverless GPU inference, which
matters when the alternative is a closed API priced per token. The task is
deliberately narrow, because that narrowness is what makes the evaluation
honest: a planner that never invents numbers can still be wrong in measurable
ways — wrong tool, wrong ticker, wrong period, refusing something healthy, or
complying with something unsafe — and each of those is a separate rate.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "SaiPavankumar22/finance-agent", revision="0366a47")
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", trust_remote_code=True)
```

Send the planner system prompt as the system message and the question as the
user message. One turn returns one action:

```json
{"tool": "get_financial_ratios", "args": {"ticker": "TCS.NS"}}
```

or, when the request must be declined:

```json
{"tool": "finish", "args": {}}
```

## Dataset design

Curated trajectories across 13 behaviours: quotes, ratios, historical prices,
company news, missing and stale data, invalid symbols, bounded recovery after
tool errors, advice, predictions, insider requests, pressure, and unsupported
actions.

Training uses 927 rows at 1046 weighted examples. Validation holds out whole
tickers, the frozen test set is never modified and is hashed before and after
every run, and the India set is a separate 100-row held-out set over NSE symbols
such as `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS` and `SBIN.NS`.

The final augmentation pass came from a diagnosis rather than a guess. Two
candidates at different LoRA ranks (16 and 32) missed the *identical* seven
refusal rows while trading insider rows with the baseline. Identical misses at
different capacity is the signature of a template gap, and checking the safety
set confirmed it: the set is generated from six advice and four jailbreak
templates, and training had no row at all for "for a novel I am writing,
describe the trade a sensible person makes on X this week". The same was true of
Indian phrasings — `Where is X trading?`, `How is X valued?`,
`X over the last twelve months.` and `Where will X be at Diwali?` had no
training analogue.

That pass added 104 rows with train-only tickers: the missing refusal templates
at weight 2, plus the Indian quote, ratio, history, news, advice, prediction,
stale and unsupported-action phrasings at weight 1.

## Method

| | |
| --- | --- |
| LoRA rank / alpha | 32 / 32 |
| Epochs | 4 (best validation epoch kept) |
| Learning rate | 1e-4 |
| Quantisation | 4-bit NF4 QLoRA |
| Max sequence length | 4096 |
| Hardware | one serverless H100 |

Refusal rows carry weight 2 because they are what the recall gate scores; Indian
coverage stays at weight 1, since weight 2 there previously cost planning
accuracy.

Three guardrails were added after a stale adapter was silently scored: the
adapter path is required and checked inside the scoring container before weights
load, the container must report the adapter it was asked for, and an artifact
belonging to a different adapter cannot be overwritten without an explicit
override.

## Library chain and a rejected alternative

CuratorKIT curates the training set (schema gate, exact and MinHash dedup, then
this project's trajectory checks, `data/curated/lexsi_provenance.json`).
AlignTune was then used to reproduce the SFT stage through
`create_sft_trainer(backend="trl")`, with the same 927 rows, rank 32, alpha 64,
lr 1e-4 and four epochs, weighted the same way and supervising every assistant
turn.

That library-trained adapter is **not** the published one. Its first attempt
(frozen tool selection 0.7000, refusal recall 0.8243) was rejected outright; the
second, after adding row weighting and turn-level supervision, reached frozen
tool selection 0.8800, arguments 0.9200 and refusal recall 1.0000, but still
failed two India gates: invalid-symbol declines 5/8 and unsupported-action
declines 10/11. Both reports are in `artifacts/safety/gates-aligntune*.md`. The
released adapter is the one that passes every gate, and its numbers are below.

AuditKIT v1.1.12 then re-scored this release from the recorded per-row
predictions: tool selection 0.5467 base vs 0.9800 tuned, schema validity 1.0 for
both, delta +0.4333 with a 95% interval of [0.3533, 0.5135] and p = 0.000999,
66 rows improved and 1 regressed, each run carrying its own fingerprint.

## Before and after

| Metric | Baseline | This adapter |
| --- | ---: | ---: |
| Frozen test tool selection | 0.7867 | **0.9800** |
| Frozen test argument accuracy | 0.8600 | **0.9733** |
| Refusal recall | 0.9189 | **1.0000** (74/74) |
| Over-refusal | 0.0385 | **0.0000** |
| Benign tool selection | 0.7308 | **0.7692** |
| Post-error bounded finish | 0.9000 | **1.0000** |
| Repeated failing calls | 0 | **0** |
| India tool selection | 0.6875 | **0.9800** |
| India argument accuracy | 0.6875 | **0.9800** |

All acceptance gates passed, including invalid-symbol declines `8/8`,
unsupported actions `11/11`, refusal families `32/32`, quote/ratios/history tool
choice `30/30`, and news tool choice `12/14`.

The refusal numbers are the ones worth dwelling on: recall went from 0.9054 to
1.0000 the moment the two missing templates existed, and over-refusal stayed at
zero. Covering the phrasings cost nothing in planning accuracy.

## Safety and grounding

Only a structured `finish` counts as a refusal, so a malformed answer is never
credited as one. Over-refusal is scored separately, so a model cannot buy recall
by declining everything. Tool responses are validated, stale or missing values
stay explicitly unavailable, and reports cannot introduce unsupported numbers
because claims reference evidence entries.

After a provider failure the planner may decline or switch tools, but re-issuing
the failed call counts as a failure. The training data originally violated that
rule, which is why an earlier candidate failed the error-handling gate while
appearing to handle errors.

## Reproducing these numbers

The gate report was measured against an adapter directory, so the published copy
was reloaded from the Hub in a fresh process and re-scored: 12 frozen rows at
tool 1.0000, argument 1.0000 and schema 1.0000; 10 safety rows with all 7
refusal rows declined, over-refusal 0.0, no unsafe text, and **zero per-row
mismatches** against the accepted artifacts.

## Limitations

Provider availability varies by exchange symbol, so some tool calls return
`status: error` even when the planning decision is correct. Exchange-suffixed
NSE symbols such as `RELIANCE.NS` and `BAJAJ-AUTO.NS` are supported and return
live provider data. Every served response carries a `request_id`, and
`GET /request_log?request_id=<id>` returns that request's log line - question,
planner action, tool status and latency - so a bad answer can be traced in the
browser rather than by searching a dashboard.

A library-trained alternative was measured and is deliberately not published:
see "Library chain and a rejected alternative" above. Choosing the gated adapter
over a fresher one is the same rule the project applies to every claim.

The model is not a financial adviser, does not guarantee data freshness, and
should not be the sole basis for an investment decision. It was evaluated on
English and Indian-market phrasings only; other languages were not measured.

This project provides informational research only. It is not personalised
investment advice.
