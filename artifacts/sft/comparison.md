# Adapter vs baseline

Adapter: `/out/adapter-r16-e2-lr0.0001-best`.
Decoding: greedy, stop at first complete JSON object (max 96 new tokens).

| Set | System | Tool selection | Argument acc. | Schema valid | Abstention |
|---|---|---|---|---|---|
| val | base (same prompt) | 0.5294 | 0.0588 | 1.0000 | R 0.00 / P n/a / spec 1.00 |
| val | tuned (same prompt) | 0.8235 | 0.8824 | 1.0000 | R 0.40 / P 1.00 / spec 1.00 |
| test | base (same prompt) | 0.5400 | 0.0400 | 1.0000 | R 0.00 / P n/a / spec 1.00 |
| test | tuned (same prompt) | 0.8333 | 0.8733 | 0.9933 | R 0.46 / P 1.00 / spec 1.00 |
| test | provider baseline (fp16, WS3) | 0.5533 | 0.0533 | 0.9867 | R 0.00 / P n/a / spec 1.00 |

## Paired per-example changes (test)

- tool_selection: gained 48, lost 4, correct in both 77, wrong in both 21
- argument_accuracy: gained 128, lost 3, correct in both 3, wrong in both 16

## Per-intent tool selection

| Intent | val base | val tuned | test base | test tuned |
|---|---|---|---|---|
| advice_request | 0.00 | 1.00 | 0.00 | 1.00 |
| comparison | 0.50 | 0.50 | 0.53 | 0.53 |
| invalid_ticker | 0.50 | 0.50 | 0.73 | 0.73 |
| loop_trap | 0.33 | 1.00 | 0.75 | 1.00 |
| malformed_error | 1.00 | 1.00 | 1.00 | 1.00 |
| metric_pe | 0.75 | 1.00 | 0.80 | 0.93 |
| missing_ratios | 1.00 | 1.00 | 1.00 | 0.93 |
| news_unavailable | 0.00 | 1.00 | 0.00 | 1.00 |
| non_us_currency | 0.75 | 0.50 | 0.64 | 0.64 |
| overview | 0.25 | 0.75 | 0.30 | 0.80 |
| period_ambiguity | 0.50 | 0.50 | 0.50 | 0.50 |
| stale_data | 1.00 | 1.00 | 1.00 | 1.00 |
| unsupported_metric | 0.25 | 1.00 | 0.17 | 1.00 |

## Data integrity

- val / test rows evaluated: 51 / 150
- frozen test set unchanged: True
