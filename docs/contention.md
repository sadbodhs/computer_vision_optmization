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

## MPS changes two of those conclusions

Findings 2 and 3 were both measured with the **CUDA Multi-Process Service daemon
off**, and inter-process context-switch serialization is precisely what MPS
removes. Re-measured as a controlled A/B — same binary, same engine, same
`frames.bin`, same uid, only the daemon toggled
([`scripts/mps_contention.sh`](../scripts/mps_contention.sh), raw data in
[`results/v3/mps_contention_N3.tsv`](../results/v3/mps_contention_N3.tsv), 3
repeats each):

| A2, N=3 | Latency p50 each | Total fps (3 repeats) |
|---|---|---|
| MPS **off** | 3.115 ms | 953.6 · 960.6 · 960.4 |
| MPS **on** | **2.476 ms** | **1266.1 · 1262.0 · 1270.2** |

Solo (N=1) is unchanged: 1.254 ms off vs 1.259 ms on — **MPS costs nothing when
there is no contention.**

**What this changes:**

- **Latency growth at N=3 falls from +148% to +97%.** Turning on a daemon
  recovers most of the gap that finding 3 attributed to Triton's scheduler.
- **Finding 2 is simply wrong with MPS on.** Total throughput reaches **1266 fps
  against a `trtexec` batch-1 engine cap of 1022.98 qps** — 24% *above* the
  supposed ceiling. Without MPS, kernels from separate processes are serialized,
  so the "cap" was really a time-slicing artifact. With MPS they run
  concurrently, and parallel instances **do** grow the pie.
- A2 with MPS (1266 fps) now beats B2 without it (~985 fps) on total throughput,
  while B2 still holds a small latency edge (2.25–2.30 ms vs 2.476 ms).

**The honest limit of this test:** B2 was **not** re-measured under MPS. An MPS
client must run as the same uid as the daemon, and while a daemon is active any
CUDA process that cannot reach it dies with `error 805` — which includes the
root-owned `triton-server` container. A2 needs no server, so it could be tested;
B2 could not, without either a root MPS daemon or a non-root Triton container.
So finding 3 is **weakened, not cleanly overturned**: the fair comparison
(A2+MPS vs B2+MPS) is still missing.

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

- **MPS: now measured for A2, still open for B2.** See the MPS section above.
  A2's contention penalty nearly halves and total throughput exceeds the engine
  cap; B2 under MPS remains untested for the uid/`error 805` reason given there.
- C2 at N≥4 in-process hit a cross-thread shm race (regions shared between
  threads); those numbers are excluded as invalid. Valid C2 in-process data is
  N≤3. Multi-process C2 needs unique shm names per process (fixed in
  `client_v2.py` via `run_tag`), but the parallel launcher still deadlocks on
  registration — use in-process streams for C2.
- Container CUDA state can degrade after very heavy multi-process churn
  (`cudaErrorNoDevice`); recreate the container from the image to recover.

---

[← index](../README.md) · prev: [Batching](batching.md) · next: [Stage decomposition](stage-decomposition.md)
