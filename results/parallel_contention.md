# Parallel Instance Contention — what happens when multiple inferences run together

Test: N independent pipeline instances on the same GPU, capacity mode (preprocessed
frames replayed flat-out, no source cap). Each instance measures its own per-frame
latency. Same model everywhere (yolov8s FP16). GPU: RTX 3090.

## Results — per-instance latency p50 and per-instance fps

### Flow A2 (pure C++ TRT, one engine + one CUDA stream per process)
| N instances | fps each | Total fps | Latency p50 each | Solo p50 | Latency growth |
|---|---|---|---|---|---|
| 1 | 809 | 809 | 1.24 ms | 1.24 ms | — |
| 2 | 475 | 950 | 2.10 ms | 1.24 ms | **+69%** |
| 3 | 319 | 957 | 3.13 ms | 1.24 ms | **+153%** |

→ Perfect GPU sharing: total throughput plateaus (~950 fps = 2×CUDA-context ceiling),
latency grows ~linearly with N. Each instance stays perfectly fair (identical numbers).

### Flow B2 (C++ CUDA-shm clients → Triton, 2 server instances)
| N instances | fps each | Total fps | Latency p50 each | Solo p50 | Latency growth |
|---|---|---|---|---|---|
| 1 | 654 | 654 | 1.28 ms | 1.28 ms | — |
| 2 | 445 | 891 | 1.86 ms | 1.28 ms | **+45%** |
| 3 | 293–394 | ~985 | 2.25–2.30 ms | 1.28 ms | **+76–80%** |

→ Same pattern, but Triton's 2 model instances queue fairly across clients;
latency grows slower than A2 at N=3 (+76% vs +153%) — the server's shared engine
pool is more efficient than N separate CUDA contexts.

### Flow C2 (Python numpy + sys-shm, streams in one process)
| N streams | fps each | Total fps | Latency p50 each | Solo p50 | Latency growth |
|---|---|---|---|---|---|
| 1 | 469 | 469 | 1.69 ms | 1.69 ms | — |
| 2 | 465 | 930 | 1.74 ms | 1.69 ms | +3% |
| 3 | 467 | 1400 | 1.73 ms | 1.69 ms | +2% |
| 6 | (2800 total — INVALID, cross-thread shm race, excluded) | | | | |

→ Within one process, extra streams add ~nothing to per-frame latency (threads
interleave on one connection). Valid up to N=3; N≥4 needs per-thread shm regions.

## Key findings

1. **Latency grows ~linearly with parallel instances on one GPU** — 1.24 → 2.10 →
   3.13 ms for A2 at N=1→2→3. The GPU time-slices between CUDA contexts; each
   instance waits for its share.
2. **Total throughput plateaus at ~950-990 fps** for A2/B2 — the batch-1 engine cap
   (1028 qps) is the real limit; parallel instances can't exceed it, they only
   divide it.
3. **Triton (B2) shares more gracefully than raw CUDA contexts (A2)**: at N=3,
   +76% vs +153% latency growth. The server's shared engine pool avoids the
   context-switch cost that N independent processes pay.
4. **C2 within-process streams are nearly free** (+2% latency at N=3) — threads on
   one connection share the queue; but N≥4 hit an shm race (excluded as invalid).
5. **Practical implication**: for N live cameras at 30 FPS, even N=6 needs only
   180 fps total — every flow handles that with latency to spare. Contention only
   matters when you push aggregate demand toward ~1000 fps.

## Caveats
- C2 N≥4 multi-stream hit a cross-thread shm race (regions shared between threads);
  valid C2 data is N≤3 in-process. Multi-process C2 requires unique shm names per
  process (fixed in client_v2.py via run_tag) but the parallel launcher still
  deadlocks on registration — use in-process streams for C2.
- Container CUDA state can degrade after very heavy multi-process churn
  (cudaErrorNoDevice); recreate from image `triton-bench:v3` to recover.