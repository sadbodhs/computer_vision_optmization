# Triton tuning — the knobs the study fixed without justifying

[← index](../README.md) · related: [CUDA graphs](cuda-graphs.md) · [Batching](batching.md)

Two settings were hardcoded across every measurement in this study and never
swept: `instance_group count` (always 2) and CUDA graphs (never enabled). This
page tests both.

Measured on flow **B2** (`trt_grpc_cuda`, CUDA shared memory) against yolov8s at
4 concurrent streams, 3 repeats per cell, server restarted for each config.
Script: [`scripts/triton_knobs.sh`](../scripts/triton_knobs.sh) ·
raw data: [`results/v3/triton_knobs.tsv`](../results/v3/triton_knobs.tsv).

---

## Result

| `instance_group count` | CUDA graphs | fps | p50 | p95 |
|---|---|---|---|---|
| 1 | off | 810.4 | 4.29 ms | 6.36 ms |
| 1 | **on** | **920.2** (+13.6%) | 4.25 ms | **4.35 ms** (−31.6%) |
| 2 | off | 1056.4 | 4.04 ms | 4.54 ms |
| 2 | **on** | 1089.1 (+3.1%) | 3.99 ms | 4.35 ms |
| 4 | off | 1080.9 | 3.71 ms | 4.59 ms |
| 4 | on | **server fails to start** — see below |

## 1. `count: 2` was the right call — and 4 is not better

Going 1 → 2 instances is worth **+30.4%** (810 → 1056 fps). Going 2 → 4 buys
**+2.3%** (1056 → 1081), for double the GPU memory and a longer startup.

The study's hardcoded `count: 2` is validated, not just inherited. Worth stating
explicitly, since it was previously an unexamined constant sitting underneath
every Triton number in the report.

## 2. CUDA graphs and multiple instances are substitutes, not additives

This is the interesting one. [At the engine level](cuda-graphs.md), CUDA graphs
were worth **+14.9%** on yolov8s. In the server:

- with **1 instance**: **+13.6%** — the engine-level gain arrives essentially intact
- with **2 instances**: **+3.1%** — most of it is already gone
- with **4 instances**: nothing left to win (and it fails to start)

Both mechanisms attack the same cost. CUDA graphs *remove* per-launch overhead;
multiple instances *hide* it by overlapping one instance's launch gap with
another's compute. Once two instances overlap, the launch bubbles are already
filled, so graphs have little left to recover.

> **The practical reading:** you do not get to add these two optimizations
> together. If you already run ≥2 instances, CUDA graphs buy you ~3% of
> throughput, not ~15%. Budget accordingly.

## 3. Where graphs still earn their place: tail latency

At `count: 1`, graphs cut **p95 from 6.36 ms to 4.35 ms (−31.6%)** while p50
barely moves (4.29 → 4.25). Launch overhead is not a constant tax — it is a
source of *jitter*, and removing it flattens the tail far more than it shifts the
median.

That matters more than the throughput number for a latency-SLA deployment: the
p95 improvement survives even at `count: 2` (4.54 → 4.35 ms) where the throughput
gain has essentially vanished.

## 4. `count: 4` + graphs takes the whole server down

Not a graceful degradation — Triton refuses to start:

```
IExecutionContext::enqueueV3: Error Code 1: Cask (Cask convolution execution)
unable to record CUDA graph for yolov8s_0_2
unable to finish CUDA graph for yolov8s_0_2: operation failed due to a previous
  error during capture
error: creating server: Internal - failed to load all models
```

Graph capture fails on the third instance, and because Triton treats a failed
model load as fatal, **the server exits rather than falling back to non-graph
execution.** If you enable graphs in production, treat instance count as part of
the same change and test them together — a config that works at `count: 2` can
hard-fail at `count: 4`.

## Not covered

- Only yolov8s and only flow B2. Flow D's dynamic-batch engines need graph capture
  per batch size (`graph_spec`) and were not tested.
- `preferred_batch_size` and `max_queue_delay_microseconds` are still fixed at
  `[4,8]` / 5000 µs — see [batching](batching.md).
- Model warmup, response cache, rate limiter, and NVIDIA's Model Analyzer remain
  untouched; see [roadmap](roadmap.md).

---

[← index](../README.md) · related: [CUDA graphs](cuda-graphs.md) · [Batching](batching.md)
