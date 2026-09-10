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

### Accuracy (mAP)

**The structural gap.** Every number in this repo is speed. There is no accuracy
axis at all, which is defensible only because every flow runs *identical FP16
weights* — so they are comparable to each other. It stops being defensible the
moment precision changes: "1.9× faster" is meaningless without "and −x mAP."

*Method*: COCO val2017 subset through each flow, mAP@0.5:0.95 vs the PyTorch
baseline. This is a prerequisite for everything in the rest of Tier 1, not an
optional extra.

### INT8 quantization

Everything here is FP16. INT8 is the largest remaining precision lever on Ampere
and the most common real-world question. Supported: `trtexec --int8 --calib=`.

*Method*: PTQ with an entropy calibrator over a few hundred representative frames;
report throughput **and** mAP delta together. Expect the interesting result to be
in the accuracy column, not the speed one.

### Structured sparsity (2:4)

Ampere (8.6) supports 2:4 structured sparsity; `trtexec --sparsity=` is available.

*Method*: `--sparsity=force` on the existing weights gives the **speed ceiling**
only — dense weights forced into a sparse kernel path lose accuracy. An honest
result needs sparse-aware retraining. Worth measuring the ceiling first to see if
the retraining is even worth it.

> **Not applicable on this rig:** FP8. `trtexec` advertises `--fp8`, but compute
> capability 8.6 has no FP8 tensor cores — that is Ada (8.9) and Hopper (9.0+).

## Tier 2 — levers that would move numbers already published

### CUDA Graphs

A2's overhead is **0.26 ms** on top of a 0.97 ms engine
([stage decomposition](stage-decomposition.md)); a meaningful share of that is
per-launch CPU cost, which CUDA graphs exist to amortize. Supported:
`trtexec --useCudaGraph`; Triton exposes it per model config.

*Why it matters here*: this attacks the **headline** A2 latency directly, and
Triton's equivalent would narrow B2's remaining gap.

### CUDA MPS — ✅ measured (A2), partially open (B2)

Done, and it changed the study. See
[contention → MPS](contention.md#mps-changes-two-of-those-conclusions).
A2 at N=3: latency growth **+148% → +97%**, total throughput **960 → 1266 fps**,
which is **24% above the batch-1 engine cap** — so "throughput plateaus at the
engine cap" was a time-slicing artifact, not a hardware ceiling. Solo performance
is unaffected.

**Still open:** B2 under MPS. An MPS client must share the daemon's uid, and while
a daemon runs, any CUDA process that cannot reach it fails with `error 805` —
including the root-owned `triton-server`. Closing this needs either a root MPS
daemon (requires sudo on the host) or a non-root Triton container. Until then the
symmetric A2+MPS vs B2+MPS comparison is missing.

### In-graph NMS (`EfficientNMS_TRT`)

Every flow ships a `[1,84,8400]` FP32 output = **2.8 MB/frame**, then post-processes
it. Folding NMS into the engine collapses that to a few KB at the source.

*Why it matters here*: this is likely a large share of B2/C2's residual cost and
would change the [transport](transport.md) conclusions — the copy you cannot
eliminate is the one you never make.

### Triton knobs never swept

`instance_group count` was fixed at 2 throughout and never justified.
`preferred_batch_size` and `max_queue_delay_microseconds` were fixed at
`[4,8]`/5000. Also untouched: model warmup, response cache, rate limiter, priority
levels, and NVIDIA's own **Model Analyzer**, which automates exactly this sweep.

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
