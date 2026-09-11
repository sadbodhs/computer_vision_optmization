# Roadmap — what this study does *not* cover

[← index](../README.md) · prev: [Reproduce](reproduce.md)

This study measures one axis thoroughly: **the serving and transport layer**, at
FP16, at 640×640, for YOLO detection on one GPU. Below is what it deliberately
does not answer yet.

**Nothing on this page has been measured.** No numbers are claimed. Each entry
states the lever, why it matters *for the numbers already published here*, and how
it would be tested. Hardware support is verified against the actual rig
(RTX 3090, compute 8.6, TensorRT 10.7).

---

## Tier 1 — the missing axis

### Accuracy (mAP) - measured

Done: [accuracy](accuracy.md). COCO val2017 (500 images), study chain vs
ultralytics end-to-end on the same images and thresholds.

- With linear resize the study's chain is **within 0.06% of the PyTorch
  reference** - the letterbox geometry, FP16 engine, custom class-aware NMS and
  un-letterboxing math are all validated rather than assumed.
- **As shipped it used nearest-neighbour resize**, in the CUDA kernel *and* the
  numpy client, costing **-1.25% mAP50-95** across A2/B2/C2/D alike. Pure speed
  benchmarking could never surface this: every flow was equally wrong, so they
  agreed with each other. **Now fixed** - all implementations interpolate
  bilinearly, verified to match `cv2.INTER_LINEAR` exactly, at a cost of
  +0.0085 ms/frame (~0.6% of frame time).

Extended since to **all three models and both engine shapes** - every one lands
within +/-0.16% of its PyTorch reference, and the batch-8 engines match batch-1
within +/-0.26%, so dynamic batching costs no accuracy either.

**Still open:** 500 of 5000 images, and the comparison is chain-vs-chain so it
does not decompose preprocessing / precision / NMS individually.

### INT8 quantization - measured, question closed

Calibrated engine built and scored: **+32.8% throughput (1023 -> 1359 qps) for
-1.55 mAP50-95**, and it *dominates* the alternative of downgrading to a smaller
model (+22.5% for -5.79 points). See [precision](precision.md).

Building it needed a two-stage route, because ultralytics pip-installs its own
TensorRT and engines are locked to the exact build that made them: let ultralytics
calibrate, keep its portable calibration cache, then rebuild with the container's
own trtexec. `scripts/build_int8_engine.py` does both.

### Structured sparsity (2:4) — ✅ measured, and closed

Measured, and the answer is no: `--sparsity=force` yields **+1.1% (YOLOv8s) /
+1.7% (YOLOv8n)** — see [precision](precision.md). That is the *ceiling*, so the
sparse-aware retraining needed to make it accuracy-neutral cannot pay for itself
on this workload. Cheap negative result; question closed.

> **Not applicable on this rig:** FP8. `trtexec` advertises `--fp8`, but compute
> capability 8.6 has no FP8 tensor cores — that is Ada (8.9) and Hopper (9.0+).

## Tier 2 — levers that would move numbers already published

### CUDA Graphs — measured at engine level and inside A2

Done: [CUDA graphs](cuda-graphs.md). `trtexec --useCudaGraph` gives +14.9%
(YOLOv8s), and the saving is a roughly constant ~0.13 ms of launch overhead
rather than a percentage.

Now also measured **inside the A2 pipeline**, which was the open item: a
`--cuda-graph` flag on `trt_pipeline_cuda` captures the inference region and
replays it. Result **+11.8% fps at concurrency 1** (799 -> 894), p95 -10.6%, and
the plateau rises from ~1167 to ~1249 fps. The absolute saving, 0.1315 ms, lands
on the projection; the percentage is lower than the engine's because the fixed
cost is divided by a bigger per-frame budget.

Still open: **flow D**. Its dynamic-batch engines need one captured graph per
batch size (`graph_spec`), which is why [Triton tuning](triton-tuning.md) could
not cover it either. A1 is unmeasured but uninteresting - same fixed saving, a
slightly larger budget.

### CUDA MPS - measured for both flows, question closed

Done: [contention](contention.md). At N=3, MPS gives **A2 +32.3% fps / -20.9%
latency** but **B2 only +1.7% fps and +20.3% *worse* latency** - because B2's GPU
work all happens inside one server process, so there are no competing CUDA
contexts for MPS to multiplex.

Consequences: "throughput plateaus at the engine cap" is false (A2 reaches 1259
fps against a 1023 qps cap); Triton's contention advantage is real *only* against
MPS-less processes and inverts once MPS is on; and MPS should **not** be enabled
for a Triton deployment.

B2 was measured by running Triton as non-root - an MPS client must share the
daemon's uid, and a root server cannot reach a user-owned daemon (`error 805`).

### In-graph NMS - measured

Done: [in-graph NMS](in-graph-nms.md). Folding NMS into the engine shrinks the
output from `[1,84,8400]` (2.82 MB) to `[1,300,6]` (7.2 KB), 392x. The engine gets
**18.6% slower** (1023 -> 833 qps) but the raw-gRPC round trip gets **33% faster**
(258.9 -> 344.7 fps, 3.79 -> 2.86 ms): shipping that output costs ~1.15 ms/frame,
against a ~0.22 ms engine penalty.

It helps the naive paths (B1, C1/C2) and probably hurts the zero-copy ones (B2/D),
which already avoid the transfer with a GPU compact kernel - untested there,
because the C++ clients expect the `[1,84,8400]` shape.

### Triton knobs — ✅ partly swept

Done for `instance_group count` and CUDA graphs: [Triton tuning](triton-tuning.md).
`count: 2` is validated (+30.4% over 1; 4 adds only +2.3%), and server-side CUDA
graphs turn out to be a *substitute* for multiple instances rather than an
addition — +13.6% at `count: 1`, +3.1% at `count: 2`. `count: 4` + graphs fails
graph capture and takes the server down.

`preferred_batch_size` and `max_queue_delay_microseconds` swept too - see
[batching](batching.md). `[4,8] / 5000us` is confirmed optimal, the knobs span 28%
at concurrency 1 but only 2.7% at concurrency 8, and shortening the window makes
latency *worse*, which corrected an earlier claim that 5ms was the floor of D's
6.2ms latency.

**Still untouched:** model warmup, response cache, rate limiter, priority levels,
and NVIDIA's Model Analyzer. Flow D's dynamic-batch engines also need
per-batch-size graph capture (`graph_spec`), untested.

### DeepStream E2 in capacity mode - attempted, harness-bound

Run, and the result is that the measurement does not measure what it needs to.
`ds_bench --mode file` feeds NV12 frames from an appsrc per stream; at 8 streams
it reaches ~501 fps against D's 1640, which looks decisive until you check the
GPU: **median 0% utilisation**. The fixed batch-8 engine also performs identically
to the dynamic one despite being 3x faster at batch 1, and the frame count is
byte-identical across repeats of a time-limited run.

The harness is the bottleneck - a 345 KB CPU memcpy per frame per stream, plus a
sysmem->NVMM conversion. Same trap as the study's first pass, in new clothes.

**Still open.** Closing it means pushing pre-allocated NVMM buffers or using a
buffer pool so the feed leaves the critical path - a change to `ds_bench`, not a
parameter. What the attempt did establish: at 1 stream with a fixed batch-1
engine, E2 sustains **653 fps** in capacity mode against 45 fps source-bound, so
the source really was the limit in the E1/E2 tables. See
[deepstream](deepstream.md).

## Tier 3 — completeness for a reference

**Model-level** — input resolution (640 vs 416/512/960) is arguably the highest-leverage
knob in CV inference and is entirely absent; cost scales ~quadratically. Also:
model scaling beyond n/s, pruning, distillation.

**Application-level** — the biggest real-world wins, all unmeasured: detect every
N frames + track between, ROI/tiling for small objects, adaptive frame skipping,
cascades (cheap detector → expensive classifier). These routinely beat every
serving optimization in this repo by an order of magnitude.

**Build-time** — timing cache, `--builderOptimizationLevel`, workspace sizing,
layer-fusion inspection (`--dumpLayerInfo`, `--profilingVerbosity=detailed`).

**Profiling method** — this repo teaches you to *measure* known stages; it does not
teach you to *find* an unknown bottleneck. Nsight Systems + NVTX ranges would.

**System** — NVDEC session limits, multi-GPU scaling, GPU clocks/power limits and
thermal throttling, CPU affinity/NUMA.

**Deployment** — cold-start and engine load time, memory footprint per instance,
TRT engine portability across versions (already brushed up against: 10.7 vs 10.3
between the Triton and DeepStream containers).

---

## Contributing a measurement

The bar for adding a number here is the same one the rest of the repo is held to:
capacity mode where applicable, identical preprocessing, ≥3 runs, raw JSON
committed under `results/` with a provenance note. See
[methodology](methodology.md).

---

[← index](../README.md) · prev: [Reproduce](reproduce.md)
