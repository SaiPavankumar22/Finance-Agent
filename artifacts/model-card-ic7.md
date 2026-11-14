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
---

# finance-agent — planner adapter (`ic7`)

A LoRA adapter for `CohereLabs/tiny-aya-global` that turns a finance research
request into exactly one planning action per turn: `get_stock_quote`,
`get_financial_ratios`, `get_historical_prices`, `get_company_news`, or `finish`.
It never states a price, a number or a forecast — numbers enter the answer only
through tool responses.

This is the first candidate to clear every planning, safety and India gate.

## Training

| | |
| --- | --- |
| Data | 927 rows, 1046 weighted examples |
| Rank / alpha | 32 / 32 |
| Epochs | 4 (best epoch kept) |
| Learning rate | 1e-4 |
| Quantisation | 4-bit NF4 QLoRA |
| Max sequence length | 4096 |

The mixture is gold trajectories plus three coverage passes: error-replay rows
that decline or fall back after a provider failure, Indian coverage (news,
invalid NSE symbols, unsupported actions), and template coverage for the refusal
families and Indian phrasings that earlier candidates missed.

## Measured behaviour

Frozen held-out test set (150 rows), same prompt and decode rule for both arms:

| Metric | base | this adapter |
| --- | --- | --- |
| Tool selection | — | **0.9800** |
| Argument accuracy | — | **0.9733** |
| Schema validity | — | 1.0000 |

Safety set (100 rows, 74 refusal / 26 benign):

| Metric | value |
| --- | --- |
| Refusal recall | **1.0000** (74/74) |
| Over-refusal | 0.0000 |
| Benign tool selection | 0.7692 |
| Bounded finish after tool errors | 1.0000 |
| Repeated failing calls | 0 |
| Unsafe text rows | 0 |

India held-out set (100 rows over NSE symbols, INR pricing):

| Intent | Rows | Tool correct |
| --- | --- | --- |
| quote | 10 | 10 |
| ratios | 10 | 10 |
| history | 10 | 10 |
| news | 14 | 12 |
| advice | 11 | 11 |
| prediction | 8 | 8 |
| insider | 8 | 8 |
| pressure | 5 | 5 |
| invalid symbol | 8 | 8 |
| unsupported | 11 | 11 |
| missing / stale | 5 | 5 |

India aggregate: tool 0.9800, args 0.9800. Indian refusals, invalid symbols and
unsupported actions all decline.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "avengeryt22/finance-agent-staging")
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", trust_remote_code=True)
```

Emit the planner prompt as the system message and the user question as the user
message; the model replies with one JSON object, e.g.
`{"tool": "get_financial_ratios", "args": {"ticker": "TCS.NS"}}` or
`{"tool": "finish", "args": {}}`.

## Limitations

- Informational research only. It is not personalised investment advice, and the
  adapter is trained to decline requests for advice, price predictions and
  non-public information.
- It plans tool calls; it does not answer from its own knowledge. Every number
  must come from a tool response, and missing or stale data stays unavailable.
- 12 of 14 Indian news rows and 5 of 5 missing/stale rows are the only non-perfect
  families; both are recency-dependent rather than reasoning failures.
- English and Indian-market phrasings only; other languages were not evaluated.
