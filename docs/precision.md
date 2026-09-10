# Precision and sparsity — speed ceilings

[← index](../README.md) · related: [CUDA graphs](cuda-graphs.md) · [Roadmap](roadmap.md)

> ## ⚠️ These are throughput numbers only
> **No accuracy was measured**, and the INT8 engines here were built **without a
> calibration set** — TensorRT falls back to generated scales, so their detections
> are meaningless. The purpose is to *bound* what INT8 and 2:4 sparsity could buy
> on this GPU, so you can judge whether the accuracy work (a real calibration set
> plus an mAP harness) is worth doing. **Do not quote these as "INT8 results".**

Script: [`scripts/precision_ceilings.sh`](../scripts/precision_ceilings.sh) ·
raw data: [`results/v3/precision_ceilings.tsv`](../results/v3/precision_ceilings.tsv).
Engines built fresh from the same ONNX into `/tmp`; the model repo is untouched.

---

## Result

| Model | FP16 | FP16 + `--sparsity=force` | INT8 (no calib) | `--best` (no calib) |
|---|---|---|---|---|
| YOLOv8s | 1020.7 qps | 1031.7 (**+1.1%**) | **1363.6 (+33.6%)** | 1357.2 (+33.0%) |
| YOLOv8n | 1476.5 qps | 1501.5 (**+1.7%**) | 1762.3 (+19.4%) | **1798.1 (+21.8%)** |

## Three readings

**1. INT8's ceiling is ~+34%, not the 2× folklore.** On YOLOv8s the best case is
1363 vs 1020 qps. That is a real gain, but it is the *upper bound* — a properly
calibrated engine will not exceed it, and whatever mAP it costs is subtracted
from this. The question "is INT8 worth it?" is therefore "is ≤34% worth an
accuracy budget and a calibration pipeline?", which is a much less obvious yes.

**2. Forced 2:4 sparsity buys nothing: +1.1% / +1.7%.** This is the *ceiling*, not
a fair accuracy-preserving result — dense weights forced down the sparse kernel
path. Since even the ceiling is ~1%, the sparse-aware retraining that would be
required to make it accuracy-neutral cannot pay for itself on this workload. **A
negative result worth having: it closes the question cheaply.**

**3. INT8 helps the *bigger* model more** (+33.6% on 8s vs +19.4% on 8n), the
mirror image of [CUDA graphs](cuda-graphs.md), which helps the *smaller* model
more. Quantization attacks compute; graphs attack fixed per-launch overhead. The
more compute-bound the model, the more INT8 wins; the more overhead-bound, the
more graphs win.

## The comparison that actually matters

Putting the two levers side by side on the same engines:

| Model | FP16 baseline | **CUDA graphs** (no accuracy cost) | **INT8** (accuracy cost, uncalibrated) |
|---|---|---|---|
| YOLOv8s | 1020.7 | 1165.5 (+14.9%) | 1363.6 (+33.6%) |
| YOLOv8n | 1476.5 | **1840.8 (+23.2%)** | 1762.3 (+19.4%) |

> **On YOLOv8n, CUDA Graphs beats INT8 outright — 1840.8 vs 1762.3 qps — and costs
> no accuracy at all.** On YOLOv8s graphs deliver about 44% of INT8's gain for
> none of the risk.

The practical order of operations follows: **remove launch overhead first, then
consider quantization.** Reaching for INT8 before graphs means paying an accuracy
bill for throughput that was available for free.

## Calibrated INT8: built, but blocked on a TensorRT version lock

With the [accuracy harness](accuracy.md) in place the obvious next step was to
score a *calibrated* engine and convert the ceiling above into a real trade.

The engine builds correctly — [`scripts/build_int8_engine.py`](../scripts/build_int8_engine.py)
runs MinMax calibration over 250 batches of COCO val images (the same
preprocessing the flows use) and produces a **14.6 MB** engine versus FP16's
25.6 MB, which is the expected shrink for genuine INT8 weights.

**It cannot be served.** Triton rejects it:

```
IRuntime::deserializeCudaEngine: Error Code 1: Serialization
(Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.
 Trying to load an engine created with incompatible serialization version.)
```

TensorRT engines are locked to the exact build that produced them. Ultralytics
pip-installs its own TensorRT to export (`tensorrt_cu13` **11.3** by default), and
the container serves with the **native 10.7.0.23**. Pinning the pip package to
`tensorrt==10.7.0.post1` does not fix it either: `10.7.0.post1` and `10.7.0.23`
are different builds of the same version, and the serialization check is exact.

**The fix is to calibrate with the container's own TensorRT** rather than a
pip-installed one — an `IInt8Calibrator` against the native library (C++, or
Python bindings built from the same 10.7.0.23 build). That is the next piece of
work, not a research question.

So the INT8 row remains a **ceiling, not a result**: +33.6% throughput is what it
could buy, and the mAP it costs is still unmeasured.

## Still not answered
- **Sparsity with retraining.** Not attempted, and finding 2 argues it is not
  worth attempting on this workload.
- **INT8 through the pipelines.** These are `trtexec` engine measurements; no A2/B2/D
  flow was run on an INT8 engine.

---

[← index](../README.md) · related: [CUDA graphs](cuda-graphs.md) · [Roadmap](roadmap.md)
