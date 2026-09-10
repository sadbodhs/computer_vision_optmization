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

### CUDA Graphs — ✅ measured at engine level

Done: [CUDA graphs](cuda-graphs.md). `--useCudaGraph` gives **+23.2% (YOLOv8n),
+14.9% (YOLOv8s), +27.0% (YOLO11n)** — a roughly constant **~0.13 ms** of
per-iteration launch overhead removed, which is why the percentage is larger for
cheaper models. The published 0.97 ms ceiling is therefore launch-bound, not a
hardware floor.

**Still open:** no pipeline binary uses graphs. Capturing the infer + compact
kernel into a graph inside `main_cuda.cu` is an unmade code change; ~0.13 ms is
the projected headroom for A2, not a measurement. Triton's own
`optimization { cuda { graphs: true } }` is also untested for B2/D, and
fixed-shape capture complicates the dynamic-batch engines used by D.

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

### In-graph NMS (`EfficientNMS_TRT`)

Every flow ships a `[1,84,8400]` FP32 output = **2.8 MB/frame**, then post-processes
it. Folding NMS into the engine collapses that to a few KB at the source.

*Why it matters here*: this is likely a large share of B2/C2's residual cost and
would change the [transport](transport.md) conclusions — the copy you cannot
eliminate is the one you never make.

### Triton knobs — ✅ partly swept

Done for `instance_group count` and CUDA graphs: [Triton tuning](triton-tuning.md).
`count: 2` is validated (+30.4% over 1; 4 adds only +2.3%), and server-side CUDA
graphs turn out to be a *substitute* for multiple instances rather than an
addition — +13.6% at `count: 1`, +3.1% at `count: 2`. `count: 4` + graphs fails
graph capture and takes the server down.

**Still untouched:** `preferred_batch_size`, `max_queue_delay_microseconds`, model
warmup, response cache, rate limiter, priority levels, and NVIDIA's Model
Analyzer. Flow D's dynamic-batch engines also need per-batch-size graph capture
(`graph_spec`), untested.

### DeepStream E2 in capacity mode

E2 has only been run source-bound (5.4% GPU). Its true batched ceiling — the
direct comparison against D's 1665 fps — needs the `frames.bin` file-replay
treatment. See [DeepStream](deepstream.md#the-open-comparison).

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
