# Accuracy — does the pipeline preserve the model?

[← index](../README.md) · prev: [Contention](contention.md) · next: [Precision](precision.md)

Every other page in this study measures speed. This one asks the question speed
cannot: **does the custom preprocessing → TensorRT → custom NMS chain actually
produce the model's detections?**

COCO val2017, 500 images, scored with `pycocotools`. The study's chain
([`scripts/accuracy_eval.py`](../scripts/accuracy_eval.py)) is compared against
ultralytics running end-to-end on the *same images with the same conf/IoU*
([`scripts/accuracy_reference.py`](../scripts/accuracy_reference.py)), so the
comparison isolates our pipeline rather than thresholding policy.
Raw data: [`results/v3/accuracy.tsv`](../results/v3/accuracy.tsv).

---

## Result

| Pipeline | Engine | Resize | mAP50-95 | mAP50 | vs reference |
|---|---|---|---|---|---|
| ultralytics reference | PyTorch FP32 | linear | **0.47376** | 0.64535 | — |
| study chain | TensorRT FP16 | linear | **0.47348** | 0.64524 | **−0.00028 (−0.06%)** |
| **study chain as shipped** | TensorRT FP16 | **nearest** | 0.46782 | 0.63582 | **−0.00594 (−1.25%)** |

## 1. The chain is sound — the custom code costs essentially nothing

With linear resize, the study's pipeline lands **0.06% from the PyTorch
reference**. That single number validates a lot at once:

- the centered-letterbox geometry (pad 114, /255, RGB) is correct
- **FP16 costs no measurable accuracy** versus FP32 on this model
- the hand-rolled class-aware NMS matches ultralytics' behaviour
- the box un-letterboxing math (remove pad, rescale, clip) is right

That is the reassurance the whole benchmark rested on but never demonstrated. It
is now demonstrated rather than assumed.

## 2. …but as shipped, every flow leaves ~0.6 mAP on the table

The pipeline does **not** use linear resize. It uses nearest-neighbour, and not
only in the Python client — the fused CUDA kernel does too:

```c
// cpp/src/main_cuda.cu — nv12_letterbox_kernel
// nearest-neighbor mapping into source
int u = sx * src_w / nw, v = sy * src_h / nh;
```

and [`triton/client_v2.py`](../triton/client_v2.py) mirrors it in numpy. So this
applies to **A2, B2, C2 and D alike** — the entire study.

The cost is **−0.0059 mAP50-95 (−1.25%)** and **−0.0095 mAP50** against the
reference, for a preprocessing shortcut that buys a handful of memory loads in a
kernel that is not the bottleneck ([stage decomposition](stage-decomposition.md)
puts preprocessing at 0.15 ms against a 0.97 ms engine).

> **This is the finding that justifies having an accuracy axis at all.** No amount
> of throughput measurement could surface it: every flow was *equally* wrong, so
> they all agreed with each other, and detection counts looked plausible. It took
> an external ground truth to see it. It is now fixed — see below.

## 3. Fixed — bilinear costs 0.6% of frame time, buys 1.25% mAP

All four implementations now interpolate bilinearly with center-aligned sampling
(`src = (dst + 0.5) * scale - 0.5`, matching `cv2.INTER_LINEAR`):

| File | What changed |
|---|---|
| `cpp/src/main_cuda.cu` | `nv12_letterbox_kernel` — bilinear luma, nearest chroma |
| `cpp/src/grpc_client_cuda.cu` | same kernel |
| `cpp/src/main_cuda.cpp` | same kernel (not built, kept consistent) |
| `triton/client_v2.py` | `letterbox_nv12_np` — vectorised bilinear |

Chroma stays nearest because NV12 already subsamples it 2×; the luma plane
carries the detail the detector uses.

**Verified numerically:** the fixed numpy path reproduces `cv2.INTER_LINEAR`
to **0.00000 max absolute difference** on a structured test frame — so it is the
same interpolation the +1.25% mAP above was measured with, not an approximation
of it.

**Cost**, measured on A2 in RTSP mode (the only mode that runs the letterbox —
capacity mode replays already-preprocessed tensors, so it is unaffected), 3 runs
of 20 s each:

| Kernel | preprocess stage | p50 latency |
|---|---|---|
| nearest | 0.1860 ms | 1.305–1.326 ms |
| **bilinear** | **0.1945 ms** | 1.325–1.327 ms |

**+0.0085 ms per frame** — 4.6% of a preprocessing stage that is itself ~0.19 ms
of a ~1.35 ms pipeline, so roughly **0.6% of total frame time for a 1.25% mAP
recovery**. At 30 fps it is 0.03% of the 33.3 ms budget.

> RTSP *fps* is not a valid comparator here (19.5–30.9 across runs on both
> kernels) because that mode is source-bound and the stream drops frames. The
> per-stage timer is the measurement; the throughput column is noise.

## 4. Every scenario, not just YOLOv8s

The result above was one model. Repeated across all three architectures and both
engine shapes (batch-1 and the batch-8 `_dyn` engines flow D uses), same 500
images, same thresholds:

| Model | reference (PyTorch) | study chain (TRT FP16 b1) | Δ | `_dyn` (TRT FP16 b8) | Δ vs b1 |
|---|---|---|---|---|---|
| YOLOv8s | 0.47376 | **0.47348** | −0.06% | 0.47367 | +0.04% |
| YOLOv8n | 0.39274 | **0.39212** | −0.16% | 0.39315 | +0.26% |
| YOLO11n | 0.41510 | **0.41554** | +0.11% | 0.41465 | −0.21% |

*(mAP50-95; full table including mAP50/mAP75 in
[`results/v3/accuracy.tsv`](../results/v3/accuracy.tsv))*

**Calibrated INT8**, same images and harness, is the one precision change that
does cost something measurable:

| YOLOv8s | Throughput | mAP50-95 |
|---|---|---|
| FP16 | 1023.26 qps | 0.47348 |
| **INT8 (calibrated)** | **1358.72 qps (+32.8%)** | **0.45794 (−1.55 points)** |

That trade, and why it beats downgrading the model, is in
[precision](precision.md#calibrated-int8-the-actual-result).

**Two things fall out.**

**The chain is validated for every architecture, not just the one.** All three land
within ±0.16% of their PyTorch reference, and the deltas scatter in both
directions — which is what noise looks like, rather than a systematic loss.

**Dynamic batching is free on the accuracy axis.** The batch-8 engines match their
batch-1 counterparts within ±0.26%, again scattering both ways. So
[flow D](batching.md)'s throughput win costs queue latency and *nothing else* —
worth knowing, because "does batching change my detections?" is a reasonable thing
to suspect and the answer here is no.

## 5. Speed and accuracy together, per model

The study compares the three models on throughput alone, which is only half the
picture. With the accuracy axis, the actual trade:

| Model | `trtexec` batch-1 | mAP50-95 | vs YOLOv8n |
|---|---|---|---|
| YOLOv8n | **1490 qps** | 0.39212 | — |
| YOLO11n | 1253 qps (−16%) | 0.41554 | **+2.3 mAP points** |
| YOLOv8s | 1023 qps (−31%) | **0.47348** | **+8.1 mAP points** |

**YOLO11n is the interesting one**: it buys +2.3 mAP points for 16% throughput,
a better exchange rate than YOLOv8s offers (+8.1 points for 31%). If you are
picking a model on the strength of the fps tables elsewhere in this study, this is
the column those tables were missing.

## Caveats

- **500 images, not the full 5000.** Enough to resolve a 1.25% gap and to tell
  models apart by 2+ points, but not enough for a headline "YOLOv8s scores X"
  claim, and not enough to call sub-0.3% deltas anything but noise. The absolute numbers here run above
  ultralytics' published 44.9 partly for that reason and partly because NMS IoU is
  0.45 (the flows' value) rather than ultralytics' 0.7 default.
- **The comparison is chain-vs-chain**, so it bundles preprocessing, engine
  precision and NMS. The linear-resize row being ~0 means none of those three
  individually hurts; it does not decompose them.
- Measured at `conf 0.001` (the mAP convention). The flows *deploy* at `conf 0.25`,
  which is an operating-point choice and not what mAP measures.

## What this unblocks

INT8 now has somewhere to report to. [Precision](precision.md) established a
**+33.6% throughput ceiling** for INT8 but deliberately refused to call it a
result, because an uncalibrated engine's detections are meaningless. With this
harness in place, a calibrated INT8 engine can be scored on the same 500 images
against the same reference, turning that ceiling into an actual
speed-versus-accuracy trade.

---

[← index](../README.md) · prev: [Contention](contention.md) · next: [Precision](precision.md)
