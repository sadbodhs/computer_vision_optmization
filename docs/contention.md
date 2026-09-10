# Contention — N pipelines, one GPU

[← index](../README.md) · prev: [Batching](batching.md) · next: [Stage decomposition](stage-decomposition.md)

What happens to *your* latency when someone else is using the GPU too.

Full study: [`results/parallel_contention.md`](../results/parallel_contention.md).

---

## Results

N independent pipeline instances on the same GPU, capacity mode, same model
(yolov8s FP16). Each instance measures its own per-frame latency.

| N instances | A2 latency each | B2 latency each | Total fps (A2 / B2) |
|---|---|---|---|
| 1 | 1.24 ms | 1.28 ms | 809 / 654 |
| 2 | 2.10 ms (+69%) | 1.86 ms (+45%) | 950 / 891 |
| 3 | 3.13 ms (+153%) | 2.25-2.30 ms (+76%) | 957 / 985 |

## Three findings

**1. Latency grows ~linearly with N.** 1.24 → 2.10 → 3.13 ms for A2 is almost
exactly 1×, 1.7×, 2.5×. The GPU time-slices between CUDA contexts; each instance
waits its turn.

**2. Total throughput plateaus at the engine cap** (~950–990 fps). Parallel
instances divide the pie; they don't grow it. The GPU is not "more GPU" because
there are more processes.

**3. Triton shares the GPU more gracefully than raw CUDA contexts.** At N=3, B2's
latency grew +76% while A2's grew +153%. The server's shared engine pool avoids
the context-switch cost that N independent processes pay each other.

Finding 3 is the one with a caveat attached — see below.

## In-process streams are nearly free

Flow C2 with threads inside one process: **+2–3% latency at N=3** (threads
interleave on one connection), versus +153% for N separate processes. If you
control the client, in-process concurrency beats process-per-stream.

## Practical implication

Six live 30 fps cameras need 180 fps of aggregate capacity. Every configuration
in this study covers that with room to spare.

**Contention only becomes real when aggregate demand approaches ~1000 fps.** Below
that, pick your flow on latency and operational fit, not on contention behaviour.

## Caveats

- **MPS was not tested.** Finding 3 (Triton shares better than raw contexts)
  compares Triton's shared engine pool against N processes *without* CUDA MPS.
  MPS exists specifically to remove inter-process context-switch serialization,
  and the box supports it. This is the single most likely finding in the study to
  change under retest — see [roadmap](roadmap.md).
- C2 at N≥4 in-process hit a cross-thread shm race (regions shared between
  threads); those numbers are excluded as invalid. Valid C2 in-process data is
  N≤3. Multi-process C2 needs unique shm names per process (fixed in
  `client_v2.py` via `run_tag`), but the parallel launcher still deadlocks on
  registration — use in-process streams for C2.
- Container CUDA state can degrade after very heavy multi-process churn
  (`cudaErrorNoDevice`); recreate the container from the image to recover.

---

[← index](../README.md) · prev: [Batching](batching.md) · next: [Stage decomposition](stage-decomposition.md)
