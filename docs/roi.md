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

Measured for the library-trained SFT stage, at Modal list prices checked
2026-11-16 (H100 SXM $3.95/GPU-hour, A100 40 GB $2.10/GPU-hour):

| Run | Hardware | Time | Approx. cost |
| --- | --- | ---: | ---: |
| AlignTune SFT, 4 epochs (584 steps) | H100 | 59 min | $3.90 |
| Frozen + safety + India evaluation of one adapter | A100 40 GB | ~45 min total | $1.60 |
| Aborted earlier attempts (A100, restarted) | A100 40 GB | ~90 min | $3.20 |

One full train-and-gate cycle is therefore about $6, including the eval passes
that decide whether the adapter ships. CPU and memory are billed on top of the
GPU rate and are a small fraction at these runtimes; storage on the adapter
volume is $0.09 per GiB-month. Training on an H100 at 6.0 s/step was roughly
twice as fast as the same job on an A100 40 GB at 11.5 s/step, which is why the
larger card is cheaper per completed run.

## Throughput caveat

The 10 requests/minute figure is a planning estimate, not a benchmark. The
demo uses greedy structured generation, yfinance calls, and scale-to-zero.
Measure p50/p95 latency and sustained throughput under the final traffic
profile before making a production cost claim.

