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
instances divide the pie; they don't grow it. **← this turned out to be false;
see MPS below.**

**3. Triton shares the GPU more gracefully than raw CUDA contexts.** At N=3, B2's
latency grew +76% while A2's grew +153%. **← substantially an artifact of MPS
being off; see below.**

---

## MPS changes two of those conclusions - and inverts the third

Findings 2 and 3 were both measured with the **CUDA Multi-Process Service daemon
off**. Re-measured as a controlled A/B for **both** flows - same engine, same
`frames.bin`, same uid, only the daemon toggled
([`scripts/mps_contention.sh`](../scripts/mps_contention.sh), raw data in
[`results/v3/mps_contention_N3.tsv`](../results/v3/mps_contention_N3.tsv), 3
repeats each). Testing B2 required running Triton itself as non-root, because an
MPS client must share the daemon's uid.

| N=3 | MPS off | MPS on | Effect |
|---|---|---|---|
| **A2** total fps | 951.8 | **1258.9** | **+32.3%** |
| **A2** latency p50 | 3.145 ms | **2.487 ms** | **-20.9%** |
| **B2** total fps | 977.2 | 993.6 | +1.7% |
| **B2** latency p50 | **2.295 ms** | 2.760 ms | **+20.3% (worse)** |

Solo (N=1) A2 is unchanged: 1.254 ms off vs 1.259 ms on - **MPS costs nothing when
there is no contention.**

### Why the two flows react so differently

MPS exists to let **separate processes** share a GPU without serializing on
context switches.

- **A2 is three processes with three CUDA contexts.** Exactly MPS's problem, so it
  wins big.
- **B2 is three *client* processes, but all GPU work happens inside one Triton
  server process** with two model instances. There are no competing contexts for
  MPS to multiplex - so it adds nothing (+1.7%) while its indirection layer
  *costs* ~20% latency.

**Finding 3 is therefore mechanistically vindicated**: Triton's single-process,
multi-instance design achieves what MPS achieves for multi-process. It was never
clever scheduling - it is simply not having separate contexts to fight.

### But the practical conclusion inverts

- **Without MPS:** B2 (977 fps, 2.295 ms) beats A2 (952 fps, 3.145 ms). Triton
  wins, as originally reported.
- **With MPS:** A2 (1259 fps) beats B2 (994 fps) by **+26.7%**, and the latency gap
  narrows to 2.49 vs 2.76 ms - A2 now wins *both*.

> **Triton's contention advantage exists only against MPS-less processes.** Turn
> MPS on and the hand-rolled multi-process pipeline wins. Equally: **do not enable
> MPS for a Triton deployment** - you pay ~20% latency for ~2% throughput.

### And finding 2 is simply wrong with MPS on

A2 reaches **1259 fps against a `trtexec` batch-1 engine cap of 1022.98 qps** -
23% *above* the supposed ceiling. Without MPS, kernels from separate processes
serialize, so the "cap" was a time-slicing artifact rather than a hardware limit.
With MPS, parallel instances **do** grow the pie.

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

- **MPS: measured for both flows** (see above). Testing B2 needed a non-root
  Triton container, since an MPS client must share the daemon's uid and a root
  server cannot reach a user-owned daemon (`error 805`).
- C2 at N≥4 in-process hit a cross-thread shm race (regions shared between
  threads); those numbers are excluded as invalid. Valid C2 in-process data is
  N≤3. Multi-process C2 needs unique shm names per process (fixed in
  `client_v2.py` via `run_tag`), but the parallel launcher still deadlocks on
  registration — use in-process streams for C2.
- Container CUDA state can degrade after very heavy multi-process churn
  (`cudaErrorNoDevice`); recreate the container from the image to recover.

---

[← index](../README.md) · prev: [Batching](batching.md) · next: [Stage decomposition](stage-decomposition.md)
