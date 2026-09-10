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

## Calibrated INT8: the actual result

The ceiling above is uncalibrated, so it is a bound, not a result. Here is the
real one — a properly calibrated engine, scored on the same 500 COCO images
against the same reference as everything in [accuracy](accuracy.md):

| | FP16 | **INT8 (calibrated)** | Δ |
|---|---|---|---|
| Throughput (`trtexec`) | 1023.26 qps | **1358.72 qps** | **+32.8%** |
| mAP50-95 | 0.47348 | **0.45794** | **−1.55 points (−3.3%)** |
| mAP50 | 0.64524 | 0.63192 | −1.33 points |
| Engine size | 25.6 MB | 14.25 MB | −44% |

**+32.8% throughput for 1.55 mAP points.** Whether that is a good trade is a
deployment question, but it is now a question with numbers on both sides.

Note the calibrated engine lands within 0.4% of the *uncalibrated* ceiling
(1358.7 vs 1363.6 qps) — calibration costs essentially no speed. Everything INT8
gives you was available in the bound; what calibration buys is the accuracy being
meaningful rather than garbage.

### It beats the other way of buying speed

The obvious alternative to quantising is downgrading the model. Both bought from
the same FP16 YOLOv8s baseline:

| Route | Throughput | mAP cost |
|---|---|---|
| **INT8 on YOLOv8s** | 1023 → 1359 (**+32.8%**) | **−1.55 points** |
| Switch to YOLO11n (FP16) | 1023 → 1253 (+22.5%) | −5.79 points |

**INT8 strictly dominates**: more speed *and* less accuracy lost. If you were
about to drop to a smaller model for throughput, quantise the bigger one instead.

And stacked against the free lever: [CUDA graphs](cuda-graphs.md) give +14.9% at
**zero** accuracy cost. The order of operations is therefore graphs first, then
INT8, then model choice last.

### How it was built

Not the obvious way. `ultralytics export(int8=True)` produces a correctly
calibrated engine that Triton cannot load — TensorRT engines are locked to the
exact build that produced them, ultralytics pip-installs its own TensorRT
(`tensorrt_cu13` 11.3), and the container serves native 10.7.0.23. Pinning the pip
package to `10.7.0.post1` does not help either: `post1` and `10.7.0.23` are
different builds and the check is exact.

The working route is two-stage, and
[`scripts/build_int8_engine.py`](../scripts/build_int8_engine.py) runs both:

1. **ultralytics calibrates.** Keep its calibration *cache* — a text file of
   per-tensor scales, portable across builds of the same TensorRT version — and
   its intermediate ONNX. Discard its engine.
2. **the container's own `trtexec` rebuilds** from that ONNX and cache:
   `--onnx=<int8 onnx> --int8 --fp16 --calib=<cache>`. The `--fp16` is required;
   the ultralytics ONNX carries fp16 layer precisions and the builder rejects the
   network without it.

The engine that comes out loads in Triton and is what the table above measures.

## Still not answered## Still not answered
- **Sparsity with retraining.** Not attempted, and finding 2 argues it is not
  worth attempting on this workload.
- **INT8 through the pipelines.** These are `trtexec` engine measurements; no A2/B2/D
  flow was run on an INT8 engine.

---

[← index](../README.md) · related: [CUDA graphs](cuda-graphs.md) · [Roadmap](roadmap.md)
