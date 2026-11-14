# ROI comparison

This is an illustrative operating comparison for the hackathon submission.
Replace the assumptions with the provider's current published prices before
final submission if pricing changes.

## Assumptions

| Item | Assumption |
| --- | ---: |
| Monthly requests | 10,000 |
| Input tokens/request | 600 |
| Output tokens/request | 250 |
| Closed-model input price | $3.00 / million tokens |
| Closed-model output price | $15.00 / million tokens |
| Self-hosted GPU | Modal A10G |
| GPU price assumption | $1.10 / GPU-hour |
| Warm throughput | 10 requests/minute |
| Warm GPU time | 1,000 minutes/month |

## Estimated monthly cost

Closed-source API:

```text
input  = 10,000 × 600 / 1,000,000 × $3.00  = $18.00
output = 10,000 × 250 / 1,000,000 × $15.00 = $37.50
total  = $55.50/month
```

Self-hosted Modal inference:

```text
GPU time = 1,000 minutes / 60 = 16.67 GPU-hours
runtime  = 16.67 × $1.10 = $18.34/month
```

The self-hosted estimate excludes one-time training, storage, and cold-start
overhead. At this traffic level it is approximately `$37.16/month` cheaper
under these assumptions. At low traffic, scale-to-zero makes self-hosting
competitive; at very low traffic, a closed API may still be simpler and
cheaper.

## Training cost

Training and evaluation used Modal GPU jobs and are one-time project costs,
separate from inference. The repository records the exact training artifacts,
adapter revision, evaluation reports, and reproduction result so the estimate
can be recomputed from actual runtime logs.

## Throughput caveat

The 10 requests/minute figure is a planning estimate, not a benchmark. The
demo uses greedy structured generation, yfinance calls, and scale-to-zero.
Measure p50/p95 latency and sustained throughput under the final traffic
profile before making a production cost claim.

