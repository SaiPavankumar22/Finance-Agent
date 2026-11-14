# Finance Agent: Safe Indian-Market Research with Tiny Aya

## Summary

Finance Agent is a tool-using financial research assistant built on
`CohereLabs/tiny-aya-global`. It plans a structured tool call, retrieves
current market data with yfinance, and returns evidence-backed results rather
than inventing prices or ratios.

The accepted adapter is published at
[`SaiPavankumar22/finance-agent`](https://huggingface.co/SaiPavankumar22/finance-agent)
at revision `0366a47`.

## Why this model and task

Tiny Aya Global was selected because it is small enough for affordable
serverless GPU inference while supporting the project's multilingual-oriented
base model family. The task is deliberately narrow: select one safe financial
tool and its arguments, or decline requests for advice, predictions, insider
information, and unsupported actions.

The system does not ask the model to memorize market prices. Quotes, ratios,
history, and news come from provider tools and carry timestamps, source labels,
and explicit unavailable states.

## Dataset design

The dataset contains 800 curated trajectories covering 13 behaviors, including
quotes, ratios, historical prices, company news, missing or stale data,
invalid symbols, bounded tool-error recovery, advice, predictions, insider
requests, pressure, and unsupported actions.

The training split contains 823 rows after targeted safety and Indian-market
augmentation. Validation holds out ticker groups, and the frozen test set is
never modified. The India evaluation contains 100 held-out examples using NSE
symbols such as `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, and `SBIN.NS`.

The final augmentation addressed measured template gaps rather than simply
increasing model capacity:

- advice and jailbreak phrasings that had no training analogue;
- Indian quote, ratio, history, news, advice, prediction, stale, and
  unsupported-action phrasings;
- explicit bounded behavior after failed tool calls.

## Before and after

| Metric | Base/WS5 | Accepted ic7 | Why it matters |
| --- | ---: | ---: | --- |
| Frozen test tool selection | 0.7867 | **0.9800** | Chooses the correct data tool |
| Frozen test argument accuracy | 0.8600 | **0.9733** | Preserves ticker and period arguments |
| Refusal recall | 0.9189 | **1.0000** | Declines unsafe or unsupported requests |
| Over-refusal | 0.0385 | **0.0000** | Does not reject healthy requests |
| Benign tool selection | 0.7308 | **0.7692** | Maintains useful behavior |
| Post-error bounded finish | 0.9000 | **1.0000** | Stops after unavailable evidence |
| India tool selection | 0.6875 | **0.9800** | Handles Indian-market requests |
| India argument accuracy | 0.6875 | **0.9800** | Uses correct NSE symbols and periods |

All acceptance gates passed, including invalid-symbol declines `8/8`,
unsupported actions `11/11`, refusal families `32/32`, quote/ratios/history
tool choice `30/30`, and news tool choice `12/14`.

## Safety and grounding

The model emits one structured action per turn. `finish` is the only refusal
action. Tool responses are validated, stale or missing values remain explicit,
and reports cannot introduce unsupported numbers because claims reference
evidence entries.

The public demo is informational research only and includes a non-advisory
disclaimer. It does not provide personalized investment advice.

## Deployment

The public Modal demo uses the accepted staging revision
`avengeryt22/finance-agent-staging@5ae2ee8` and scales to zero between
requests. Cold starts are expected because the model is loaded on demand.

- Public API: `https://devisettisaipavankumar--finance-agent-ws5-demo-financeag-dc3c3e.us-east.modal.run`
- Model repository: https://huggingface.co/SaiPavankumar22/finance-agent
- Dataset repository: https://huggingface.co/datasets/SaiPavankumar22/finance-agent-dataset
- Evaluation repository: https://huggingface.co/datasets/SaiPavankumar22/finance-agent-eval

## Limitations

Provider availability can vary for individual exchange symbols. The model is
not a financial adviser, does not guarantee market data freshness, and should
not be used as the sole basis for an investment decision.

